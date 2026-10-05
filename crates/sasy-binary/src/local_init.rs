//! First-start setup for a self-contained local engine (the Docker image).
//!
//! `sasy local-init DIR` writes everything `sasy serve` needs into DIR: random
//! API keys, the role and transform config, and a private CA with a localhost
//! server certificate. A later run reuses what is there, so keys and the CA stay
//! stable across restarts. Either way it prints the client settings (API key and
//! CA certificate) as JSON on stdout, for a launcher to hand to the SDK.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair, KeyUsagePurpose,
};
use time::{Duration, OffsetDateTime};

/// Identity settings requested by the launcher. Omitted fields retain existing
/// settings, or use these local defaults when creating a fresh volume.
#[derive(Debug, Default)]
pub struct IdentityOptions {
    pub client_entity: Option<String>,
    pub admin_entity: Option<String>,
    pub tenant: Option<String>,
    pub trust_domain: Option<String>,
}

#[derive(Debug)]
struct Identity {
    client_entity: String,
    admin_entity: String,
    tenant: String,
    trust_domain: String,
}

impl IdentityOptions {
    fn first_start(&self) -> Result<Identity> {
        let identity = Identity {
            client_entity: self.client_entity.as_deref().unwrap_or("client").into(),
            admin_entity: self.admin_entity.as_deref().unwrap_or("admin").into(),
            tenant: self.tenant.as_deref().unwrap_or("default").into(),
            trust_domain: self.trust_domain.as_deref().unwrap_or("sasy.local").into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    fn check_existing(&self, identity: &Identity) -> Result<()> {
        for (name, requested, actual) in [
            (
                "client-entity",
                &self.client_entity,
                &identity.client_entity,
            ),
            ("admin-entity", &self.admin_entity, &identity.admin_entity),
            ("tenant", &self.tenant, &identity.tenant),
            ("trust-domain", &self.trust_domain, &identity.trust_domain),
        ] {
            if let Some(requested) = requested {
                validate_label(name, requested)?;
                if requested != actual {
                    bail!("--{name} differs from the existing engine configuration; use a separate data volume for different identities");
                }
            }
        }
        Ok(())
    }
}

impl Identity {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("client-entity", &self.client_entity),
            ("admin-entity", &self.admin_entity),
            ("tenant", &self.tenant),
            ("trust-domain", &self.trust_domain),
        ] {
            validate_label(name, value)?;
        }
        if self.client_entity == self.admin_entity {
            bail!("client and admin entities must differ");
        }
        if self.client_entity == "policy-engine" || self.admin_entity == "policy-engine" {
            bail!("policy-engine is reserved for the engine's internal identity");
        }
        Ok(())
    }

    fn auth_config(&self) -> serde_json::Value {
        // JSON is valid YAML, and serialization quotes user-supplied keys and
        // values rather than interpolating them into a configuration document.
        serde_json::json!({
            "trust_domain": self.trust_domain,
            "default_tenant": self.tenant,
            "entities": {
                &self.client_entity: { "roles": ["reference-monitor-user", "observability-writer", "observability-reader"] },
                &self.admin_entity: { "roles": ["admin", "reference-monitor-user", "observability-writer", "observability-reader"] },
                "policy-engine": { "roles": ["observability-reader"] },
            }
        })
    }
}

fn validate_label(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        bail!("{name} must be nonempty and contain no control characters");
    }
    Ok(())
}

/// The same empty transform set as `config/transforms.example.json`.
pub const TRANSFORMS: &str = "{\"transforms\": {}}\n";

/// The client settings printed by `run`.
const CLIENT_FILE: &str = "client.json";

/// Files `sasy serve` reads, relative to the state directory.
const SERVE_FILES: [&str; 5] = [
    "auth/apikey.json",
    "auth_config.yaml",
    "transforms.json",
    "tls/server.crt",
    "tls/server.key",
];

/// Create the state in `dir` if needed, then print the client settings.
pub fn run(dir: &Path, options: &IdentityOptions) -> Result<()> {
    let client = ensure(dir, options)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(client.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

/// Return the client settings JSON, generating the state on first use.
///
/// The state is generated in a private staging directory and renamed into
/// place, so `dir` is either absent or complete, even when two containers
/// start on the same volume at once: the loser of the rename uses the
/// winner's state.
pub fn ensure(dir: &Path, options: &IdentityOptions) -> Result<String> {
    if !dir.exists() {
        let name = dir
            .file_name()
            .context("state directory needs a final path component")?;
        let staging = dir.with_file_name(format!(
            ".{}.{}",
            name.to_string_lossy(),
            &random_key()?[..16]
        ));
        let identity = options.first_start()?;
        let generated = generate(&staging, &identity).and_then(|_| {
            fs::rename(&staging, dir).with_context(|| format!("move state to {}", dir.display()))
        });
        if generated.is_err() {
            let _ = fs::remove_dir_all(&staging);
            if !dir.join(CLIENT_FILE).is_file() {
                generated?;
            }
        }
    }
    let client_path = dir.join(CLIENT_FILE);
    for file in SERVE_FILES.iter().chain([&CLIENT_FILE]) {
        if !dir.join(file).is_file() {
            bail!(
                "{} is missing from {}; remove the engine's data volume to start over",
                file,
                dir.display()
            );
        }
    }
    let stored_client: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&client_path)
            .with_context(|| format!("read {}", client_path.display()))?,
    )?;
    let identity = existing_identity(dir, &stored_client)?;
    identity.validate()?;
    options.check_existing(&identity)?;
    // Output only the public launcher contract, never arbitrary stored fields
    // or the admin API key from the engine configuration.
    let mut client = serde_json::json!({
        "api_key": stored_client["api_key"].as_str().context("client settings have no API key")?,
        "ca_cert_pem": stored_client["ca_cert_pem"].as_str().context("client settings have no CA certificate")?,
    });
    add_identity(&mut client, &identity);
    Ok(serde_json::to_string_pretty(&client)? + "\n")
}

fn existing_identity(dir: &Path, client: &serde_json::Value) -> Result<Identity> {
    let apikey: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("auth/apikey.json"))?)?;
    let keys = apikey["static_keys"]
        .as_object()
        .context("local engine configuration has no static API keys")?;
    let client_key = client["api_key"]
        .as_str()
        .context("client settings have no API key")?;
    let client_entity = keys
        .get(client_key)
        .and_then(|value| value.as_str())
        .context("client API key is absent from the engine configuration")?;
    if keys.len() != 2 {
        bail!("local engine configuration must have separate client and admin API keys");
    }
    let admin_entity = keys
        .iter()
        .find(|(key, _)| key.as_str() != client_key)
        .and_then(|(_, value)| value.as_str())
        .context("local engine configuration has no admin entity")?;
    let config_path = dir.join("auth_config.yaml");
    let config = sasy_auth::AuthConfig::load(
        config_path
            .to_str()
            .context("configuration path is not UTF-8")?,
    )?;
    for entity in [client_entity, admin_entity] {
        if !config.has_entity(entity) || config.get_tenant(entity) != config.default_tenant {
            bail!("local engine entity does not match its configured default tenant");
        }
    }
    Ok(Identity {
        client_entity: client_entity.into(),
        admin_entity: admin_entity.into(),
        tenant: config.default_tenant,
        trust_domain: config
            .trust_domain
            .context("local engine has no trust domain")?,
    })
}

fn add_identity(client: &mut serde_json::Value, identity: &Identity) {
    client["client_entity"] = identity.client_entity.clone().into();
    client["admin_entity"] = identity.admin_entity.clone().into();
    client["tenant"] = identity.tenant.clone().into();
    client["trust_domain"] = identity.trust_domain.clone().into();
}

fn generate(dir: &Path, identity: &Identity) -> Result<()> {
    fs::create_dir(dir).with_context(|| format!("create {}", dir.display()))?;
    fs::create_dir(dir.join("auth"))?;
    fs::create_dir(dir.join("tls"))?;
    set_mode(dir, 0o700)?;

    let client_key = random_key()?;
    let admin_key = random_key()?;
    let apikey = serde_json::json!({
        "type": "api_key",
        "metadata_key": "x-api-key",
        "static_keys": { &client_key: &identity.client_entity, &admin_key: &identity.admin_entity },
    });
    write_private(
        &dir.join("auth/apikey.json"),
        &serde_json::to_string_pretty(&apikey)?,
    )?;
    write_private(
        &dir.join("auth_config.yaml"),
        &serde_json::to_string_pretty(&identity.auth_config())?,
    )?;
    write_private(&dir.join("transforms.json"), TRANSFORMS)?;

    let (ca_pem, server_pem, server_key_pem) = certificates()?;
    write_private(&dir.join("tls/ca.crt"), &ca_pem)?;
    write_private(&dir.join("tls/server.crt"), &server_pem)?;
    write_private(&dir.join("tls/server.key"), &server_key_pem)?;

    let mut client = serde_json::json!({
        "api_key": client_key,
        "ca_cert_pem": ca_pem,
    });
    add_identity(&mut client, identity);
    let client = serde_json::to_string_pretty(&client)? + "\n";
    write_private(&dir.join(CLIENT_FILE), &client)
}

/// A private CA and a server certificate for loopback names, valid ten years.
/// The engine is meant to be published on the host's loopback interface only.
fn certificates() -> Result<(String, String, String)> {
    let now = OffsetDateTime::now_utc();
    let mut ca = params("SASY local engine CA", now)?;
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
    Ok((ca_cert.pem(), server_cert.pem(), server_key.serialize_pem()))
}

fn params(name: &str, now: OffsetDateTime) -> Result<CertificateParams> {
    let mut params = CertificateParams::new(Vec::<String>::new())?;
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(DnType::CommonName, name);
    params.not_before = now - Duration::minutes(5);
    params.not_after = now + Duration::days(3650);
    params.is_ca = IsCa::ExplicitNoCa;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.use_authority_key_identifier_extension = true;
    Ok(params)
}

/// 32 random bytes from the OS, hex-encoded.
fn random_key() -> Result<String> {
    let mut bytes = [0u8; 32];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .context("read /dev/urandom")?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn write_private(path: &Path, contents: &str) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options
        .open(path)
        .with_context(|| format!("write {}", path.display()))?;
    file.write_all(contents.as_bytes())?;
    set_mode(path, 0o600)
}

fn set_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))
            .with_context(|| format!("set permissions on {}", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = (path, mode);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_run_generates_and_later_runs_reuse() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let first = ensure(&dir, &IdentityOptions::default()).unwrap();
        for file in SERVE_FILES {
            assert!(dir.join(file).is_file(), "{file} missing");
        }
        assert_eq!(ensure(&dir, &IdentityOptions::default()).unwrap(), first);

        let client: serde_json::Value = serde_json::from_str(&first).unwrap();
        let key = client["api_key"].as_str().unwrap();
        assert_eq!(key.len(), 64);
        let apikey: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("auth/apikey.json")).unwrap())
                .unwrap();
        assert_eq!(apikey["static_keys"][key], "client");
        assert_eq!(apikey["static_keys"].as_object().unwrap().len(), 2);
        let ca = client["ca_cert_pem"].as_str().unwrap();
        assert_eq!(ca, fs::read_to_string(dir.join("tls/ca.crt")).unwrap());
    }

    #[test]
    fn custom_identity_is_serialized_without_yaml_injection() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let options = IdentityOptions {
            client_entity: Some("client: [reader] # \"quoted\"".into()),
            admin_entity: Some("admin: {roles: []}".into()),
            tenant: Some("tenant: &alias *alias".into()),
            trust_domain: Some("domain: # local".into()),
        };
        let first = ensure(&dir, &options).unwrap();
        let client: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert_eq!(
            client["client_entity"].as_str(),
            options.client_entity.as_deref()
        );
        assert_eq!(
            client["admin_entity"].as_str(),
            options.admin_entity.as_deref()
        );
        assert_eq!(client["tenant"].as_str(), options.tenant.as_deref());
        assert_eq!(
            client["trust_domain"].as_str(),
            options.trust_domain.as_deref()
        );
        let config =
            sasy_auth::AuthConfig::load(dir.join("auth_config.yaml").to_str().unwrap()).unwrap();
        assert_eq!(config.entities.len(), 3);
        assert_eq!(
            config
                .get_roles(options.client_entity.as_ref().unwrap())
                .len(),
            3
        );
        assert_eq!(
            config
                .get_roles(options.admin_entity.as_ref().unwrap())
                .len(),
            4
        );
        assert_eq!(
            config.default_tenant,
            options.tenant.as_ref().unwrap().as_str()
        );
        assert_eq!(
            config.trust_domain.as_deref(),
            options.trust_domain.as_deref()
        );
        assert_eq!(ensure(&dir, &options).unwrap(), first);
        assert_eq!(ensure(&dir, &IdentityOptions::default()).unwrap(), first);
        assert!(client.get("admin_key").is_none());
    }

    #[test]
    fn invalid_identities_do_not_create_state() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        for options in [
            IdentityOptions {
                client_entity: Some("admin".into()),
                ..Default::default()
            },
            IdentityOptions {
                client_entity: Some("policy-engine".into()),
                ..Default::default()
            },
            IdentityOptions {
                admin_entity: Some("policy-engine".into()),
                ..Default::default()
            },
            IdentityOptions {
                client_entity: Some("  ".into()),
                ..Default::default()
            },
            IdentityOptions {
                admin_entity: Some("admin\nroles: []".into()),
                ..Default::default()
            },
            IdentityOptions {
                tenant: Some("".into()),
                ..Default::default()
            },
            IdentityOptions {
                trust_domain: Some("domain\0".into()),
                ..Default::default()
            },
        ] {
            assert!(ensure(&dir, &options).is_err(), "accepted {options:?}");
            assert!(!dir.exists());
        }
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 0);
    }

    #[test]
    fn explicit_identity_mismatch_preserves_existing_state() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let first = ensure(&dir, &IdentityOptions::default()).unwrap();
        let originals: Vec<_> = SERVE_FILES
            .iter()
            .chain([&CLIENT_FILE])
            .map(|file| (file, fs::read(dir.join(file)).unwrap()))
            .collect();
        for options in [
            IdentityOptions {
                client_entity: Some("another-client".into()),
                ..Default::default()
            },
            IdentityOptions {
                admin_entity: Some("another-admin".into()),
                ..Default::default()
            },
            IdentityOptions {
                tenant: Some("another-tenant".into()),
                ..Default::default()
            },
            IdentityOptions {
                trust_domain: Some("another-domain".into()),
                ..Default::default()
            },
        ] {
            let err = ensure(&dir, &options).unwrap_err().to_string();
            assert!(err.contains("separate data volume"), "{err}");
        }
        for (file, contents) in originals {
            assert_eq!(fs::read(dir.join(file)).unwrap(), contents);
        }
        assert_eq!(ensure(&dir, &IdentityOptions::default()).unwrap(), first);
    }

    #[test]
    fn old_client_settings_gain_metadata_without_rotating_credentials() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let first = ensure(&dir, &IdentityOptions::default()).unwrap();
        let mut old: serde_json::Value = serde_json::from_str(&first).unwrap();
        for field in ["client_entity", "admin_entity", "tenant", "trust_domain"] {
            old.as_object_mut().unwrap().remove(field);
        }
        old["admin_key"] = "must never be returned".into();
        let old_json = serde_json::to_string_pretty(&old).unwrap();
        fs::write(dir.join(CLIENT_FILE), &old_json).unwrap();
        // The old initializer wrote YAML, rather than JSON in a YAML file.
        fs::write(dir.join("auth_config.yaml"), "trust_domain: sasy.local\ndefault_tenant: default\nentities:\n  client:\n    roles: [reference-monitor-user, observability-writer, observability-reader]\n  admin:\n    roles: [admin, reference-monitor-user, observability-writer, observability-reader]\n  policy-engine:\n    roles: [observability-reader]\n").unwrap();
        assert_eq!(ensure(&dir, &IdentityOptions::default()).unwrap(), first);
        assert_eq!(fs::read_to_string(dir.join(CLIENT_FILE)).unwrap(), old_json);
        assert_eq!(
            ensure(
                &dir,
                &IdentityOptions {
                    client_entity: Some("client".into()),
                    ..Default::default()
                }
            )
            .unwrap(),
            first
        );
    }

    #[test]
    fn server_certificate_chains_to_the_printed_ca() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let client: serde_json::Value =
            serde_json::from_str(&ensure(&dir, &IdentityOptions::default()).unwrap()).unwrap();
        let ca_pem = client["ca_cert_pem"].as_str().unwrap();
        let server_pem = fs::read_to_string(dir.join("tls/server.crt")).unwrap();
        let ca_der = rustls_pemfile::certs(&mut ca_pem.as_bytes())
            .next()
            .unwrap()
            .unwrap();
        let server_der = rustls_pemfile::certs(&mut server_pem.as_bytes())
            .next()
            .unwrap()
            .unwrap();
        let (_, ca) = x509_parser::parse_x509_certificate(&ca_der).unwrap();
        let (_, server) = x509_parser::parse_x509_certificate(&server_der).unwrap();
        assert_eq!(server.issuer(), ca.subject());
        assert!(ca.is_ca());
        assert!(!server.is_ca());
        let names = server.subject_alternative_name().unwrap().unwrap();
        assert!(format!("{:?}", names.value).contains("localhost"));
    }

    #[test]
    fn a_marker_without_its_files_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        ensure(&dir, &IdentityOptions::default()).unwrap();
        fs::remove_file(dir.join("tls/server.key")).unwrap();
        let err = ensure(&dir, &IdentityOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("server.key"), "got: {err}");
    }

    #[test]
    fn concurrent_first_starts_agree_on_one_state() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let results: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| ensure(&dir, &IdentityOptions::default()).unwrap()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(results.iter().all(|r| r == &results[0]));
        let entries = fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(entries, 1, "staging directories were left behind");
    }

    #[test]
    fn concurrent_different_identities_do_not_return_each_others_credentials() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        let barrier = std::sync::Barrier::new(2);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = ["first", "second"]
                .into_iter()
                .map(|entity| {
                    let dir = &dir;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        let result = ensure(
                            dir,
                            &IdentityOptions {
                                client_entity: Some(entity.into()),
                                ..Default::default()
                            },
                        );
                        (entity, result)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            results.iter().filter(|(_, result)| result.is_ok()).count(),
            1
        );
        for (entity, result) in results {
            match result {
                Ok(client) => {
                    let client: serde_json::Value = serde_json::from_str(&client).unwrap();
                    assert_eq!(client["client_entity"], entity);
                }
                Err(error) => assert!(error.to_string().contains("separate data volume")),
            }
        }
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 1);
    }

    #[test]
    fn embedded_config_matches_the_repository_examples() {
        let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
        let expected =
            sasy_auth::AuthConfig::load(config.join("auth_config.example.yaml").to_str().unwrap())
                .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("local");
        ensure(&dir, &IdentityOptions::default()).unwrap();
        let actual =
            sasy_auth::AuthConfig::load(dir.join("auth_config.yaml").to_str().unwrap()).unwrap();
        assert_eq!(actual.trust_domain, expected.trust_domain);
        assert_eq!(actual.default_tenant, expected.default_tenant);
        assert_eq!(actual.entities.len(), expected.entities.len());
        for entity in expected.entities.keys() {
            assert_eq!(actual.get_roles(entity), expected.get_roles(entity));
            assert_eq!(actual.get_tenant(entity), expected.get_tenant(entity));
        }
        let transforms: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(config.join("transforms.example.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            transforms,
            serde_json::from_str::<serde_json::Value>(TRANSFORMS).unwrap()
        );
    }
}
