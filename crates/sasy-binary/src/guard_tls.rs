//! Per-launch loopback credentials. The issuing key never leaves memory.
//!
//! This authenticates the guard-to-engine connection. It is not isolation from
//! other processes running as the same OS user, who can read that user's keys.

use std::path::Path;

use anyhow::{bail, Context, Result};
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair, KeyUsagePurpose,
};
use time::{Duration, OffsetDateTime};

pub fn generate(output_dir: &Path, entity: &str) -> Result<()> {
    validate_entity(entity)?;
    let now = OffsetDateTime::now_utc();
    let mut ca = params("SASY guard launch CA", now)?;
    ca.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    ca.key_usages = vec![KeyUsagePurpose::KeyCertSign];
    let ca_key = KeyPair::generate()?;
    let ca_cert = ca.self_signed(&ca_key)?;
    let issuer = Issuer::new(ca, ca_key);

    let mut server = params("localhost", now)?;
    server.subject_alt_names =
        CertificateParams::new(vec!["localhost".into(), "127.0.0.1".into(), "::1".into()])?
            .subject_alt_names;
    server.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let server_key = KeyPair::generate()?;
    let server_cert = server.signed_by(&server_key, &issuer)?;

    let mut client = params(entity, now)?;
    client.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
    let client_key = KeyPair::generate()?;
    let client_cert = client.signed_by(&client_key, &issuer)?;

    persist(
        output_dir,
        &[
            ("ca.pem", ca_cert.pem()),
            ("server.pem", server_cert.pem()),
            ("server-key.pem", server_key.serialize_pem()),
            ("client.pem", client_cert.pem()),
            ("client-key.pem", client_key.serialize_pem()),
            ("auth-provider.json", "{\"type\":\"mtls\"}\n".into()),
        ],
    )
}

fn validate_entity(entity: &str) -> Result<()> {
    let bytes = entity.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 64
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || b"._@-".contains(c))
    {
        bail!("entity must be 1–64 ASCII letters/digits or . _ @ -, starting with a letter/digit");
    }
    Ok(())
}

fn params(name: &str, now: OffsetDateTime) -> Result<CertificateParams> {
    let mut params = CertificateParams::new(Vec::<String>::new())?;
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(DnType::CommonName, name);
    params.not_before = now - Duration::minutes(5);
    params.not_after = now + Duration::days(30);
    params.is_ca = IsCa::ExplicitNoCa;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.use_authority_key_identifier_extension = true;
    Ok(params)
}

#[cfg(unix)]
fn persist(output_dir: &Path, materials: &[(&str, String)]) -> Result<()> {
    use std::ffi::CString;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    // Anchor all mutations to directory descriptors: renaming an ancestor
    // cannot redirect a key write into an attacker-supplied path.
    let parent_path = output_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(parent_path)
        .context("open existing output parent directory")?;
    let metadata = parent.metadata()?;
    // SAFETY: geteuid has no arguments or preconditions.
    if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o022 != 0 {
        bail!("output parent must be owned by this user and not writable by group or others");
    }
    let name = output_dir
        .file_name()
        .context("output directory needs a new final path component")?;
    let name = CString::new(name.as_bytes()).context("invalid output directory name")?;
    // SAFETY: live directory fd, NUL-terminated child name, fixed permission mode.
    if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
        return Err(std::io::Error::last_os_error())
            .context("create new output directory (no overwrite)");
    }
    // SAFETY: the same live parent fd and child name; do not follow a symlink.
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        // SAFETY: remove only the newly created empty directory; no recursion.
        if unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR) } != 0 {
            return Err(error).context(format!(
                "open new private output directory; empty directory cleanup also failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        return Err(error).context("open new private output directory");
    }
    // SAFETY: openat returned a fresh owned fd, transferred exactly once.
    let directory = unsafe { File::from_raw_fd(fd) };
    let mut created = Vec::new();
    let result = (|| -> Result<()> {
        for (filename, contents) in materials {
            let filename = CString::new(*filename)?;
            // SAFETY: live private directory fd, fixed child filenames,
            // exclusive creation without symlink following. No overwrite.
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    filename.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o600,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error()).context("create credential file");
            }
            created.push(filename);
            // SAFETY: openat returned a fresh owned fd, transferred exactly once.
            let mut file = unsafe { File::from_raw_fd(fd) };
            file.write_all(contents.as_bytes())
                .context("write credential file")?;
            file.sync_all().context("sync credential file")?;
        }
        Ok(())
    })();
    if result.is_err() {
        let mut cleanup_errors = Vec::new();
        for filename in created {
            // SAFETY: unlink only files this call created in the anchored directory.
            if unsafe { libc::unlinkat(directory.as_raw_fd(), filename.as_ptr(), 0) } != 0 {
                cleanup_errors.push(std::io::Error::last_os_error().to_string());
            }
        }
        // SAFETY: remove the now-empty child, never recursively remove other content.
        if unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR) } != 0 {
            cleanup_errors.push(std::io::Error::last_os_error().to_string());
        }
        if !cleanup_errors.is_empty() {
            return result.context(format!(
                "private credential cleanup incomplete at {}: {}",
                output_dir.display(),
                cleanup_errors.join("; ")
            ));
        }
    }
    result
}

#[cfg(not(unix))]
fn persist(_output_dir: &Path, _materials: &[(&str, String)]) -> Result<()> {
    bail!("guard-tls currently supports Linux and macOS private file permissions only")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{BufReader, Cursor};
    use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
    use std::sync::Arc;

    use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};

    fn cert(path: &Path) -> CertificateDer<'static> {
        rustls_pemfile::certs(&mut BufReader::new(fs::File::open(path).unwrap()))
            .next()
            .unwrap()
            .unwrap()
    }

    fn key(path: &Path) -> PrivateKeyDer<'static> {
        rustls_pemfile::private_key(&mut BufReader::new(fs::File::open(path).unwrap()))
            .unwrap()
            .unwrap()
    }

    fn handshake(
        server_dir: &Path,
        client_dir: &Path,
        host: &str,
        client_identity: Option<&str>,
        trusted_server_dir: &Path,
    ) -> Result<()> {
        use rustls::{
            ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection,
        };
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let mut server_roots = RootCertStore::empty();
        server_roots.add(cert(&server_dir.join("ca.pem")))?;
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(server_roots),
            provider.clone(),
        )
        .build()?;
        let server_config = ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(verifier)
            .with_single_cert(
                vec![cert(&server_dir.join("server.pem"))],
                key(&server_dir.join("server-key.pem")),
            )?;
        let mut roots = RootCertStore::empty();
        roots.add(cert(&trusted_server_dir.join("ca.pem")))?;
        let client_builder = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots);
        let client_config = match client_identity {
            Some(stem) => client_builder.with_client_auth_cert(
                vec![cert(&client_dir.join(format!("{stem}.pem")))],
                key(&client_dir.join(format!("{stem}-key.pem"))),
            )?,
            None => client_builder.with_no_client_auth(),
        };
        let mut client = ClientConnection::new(
            Arc::new(client_config),
            ServerName::try_from(host.to_owned())?,
        )?;
        let mut server = ServerConnection::new(Arc::new(server_config))?;
        for _ in 0..10 {
            let mut wire = Vec::new();
            client.write_tls(&mut wire)?;
            server.read_tls(&mut Cursor::new(wire))?;
            server.process_new_packets()?;
            let mut wire = Vec::new();
            server.write_tls(&mut wire)?;
            client.read_tls(&mut Cursor::new(wire))?;
            client.process_new_packets()?;
            if !client.is_handshaking() && !server.is_handshaking() {
                return Ok(());
            }
        }
        bail!("handshake did not complete")
    }

    #[test]
    fn credentials_authenticate_only_the_correct_client_and_loopback_names() {
        let temp = tempfile::tempdir().unwrap();
        let own = temp.path().join("own");
        let other = temp.path().join("other");
        generate(&own, "local").unwrap();
        generate(&other, "local").unwrap();
        for host in ["localhost", "127.0.0.1", "::1"] {
            handshake(&own, &own, host, Some("client"), &own).unwrap();
        }
        assert!(handshake(&own, &own, "other.example", Some("client"), &own).is_err());
        assert!(handshake(&own, &own, "localhost", None, &own).is_err());
        assert!(handshake(&own, &other, "localhost", Some("client"), &own).is_err());
        assert!(handshake(&own, &own, "localhost", Some("server"), &own).is_err());
        assert!(handshake(&other, &own, "localhost", Some("client"), &own).is_err());
    }

    #[test]
    fn output_has_private_modes_exact_files_entity_and_bounded_validity() {
        use x509_parser::prelude::*;
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("tls");
        generate(&output, "user@example.com").unwrap();
        assert_eq!(fs::metadata(&output).unwrap().mode() & 0o777, 0o700);
        let mut files: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .map(|p| p.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        files.sort();
        assert_eq!(
            files,
            [
                "auth-provider.json",
                "ca.pem",
                "client-key.pem",
                "client.pem",
                "server-key.pem",
                "server.pem"
            ]
        );
        for filename in &files {
            assert_eq!(
                fs::metadata(output.join(filename)).unwrap().mode() & 0o777,
                0o600
            );
        }
        assert_eq!(
            fs::read_to_string(output.join("auth-provider.json")).unwrap(),
            "{\"type\":\"mtls\"}\n"
        );
        let der = cert(&output.join("client.pem"));
        let (_, parsed) = X509Certificate::from_der(&der).unwrap();
        assert_eq!(
            parsed
                .subject()
                .iter_common_name()
                .next()
                .unwrap()
                .as_str()
                .unwrap(),
            "user@example.com"
        );
        assert!(!parsed.is_ca());
        assert!(parsed.validity().is_valid());
        let duration =
            parsed.validity().not_after.timestamp() - parsed.validity().not_before.timestamp();
        assert_eq!(duration, 30 * 86400 + 300);
    }

    #[test]
    fn invalid_entities_leave_no_directory() {
        let temp = tempfile::tempdir().unwrap();
        for entity in [
            "",
            " leading",
            "x\r\ninjected",
            "x/y",
            "x:y",
            "é",
            ".x",
            &"x".repeat(65),
        ] {
            let output = temp.path().join("tls");
            assert!(generate(&output, entity).is_err(), "accepted {entity:?}");
            assert!(!output.exists());
        }
    }

    #[test]
    fn existing_directory_file_and_symlink_are_never_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("directory");
        fs::create_dir(&directory).unwrap();
        assert!(generate(&directory, "local").is_err());
        let file = temp.path().join("file");
        fs::write(&file, "preserve").unwrap();
        assert!(generate(&file, "local").is_err());
        assert_eq!(fs::read_to_string(&file).unwrap(), "preserve");
        let link = temp.path().join("link");
        symlink(&directory, &link).unwrap();
        assert!(generate(&link, "local").is_err());
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
    }

    #[test]
    fn rejects_shared_writable_parent_and_cleans_up_partial_writes() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("shared");
        fs::create_dir(&parent).unwrap();
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(generate(&parent.join("tls"), "local").is_err());
        assert!(!parent.join("tls").exists());
        let output = temp.path().join("partial");
        // A second exclusive creation of the same filename fails after the
        // first successful write, exercising cleanup without filesystem mocks.
        assert!(persist(
            &output,
            &[
                ("key.pem", "secret".into()),
                ("key.pem", "collision".into())
            ]
        )
        .is_err());
        assert!(!output.exists());
    }
}
