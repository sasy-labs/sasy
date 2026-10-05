//! Trusted package-provided Nix runtime closure, never a policy/request option.
//!
//! A launcher may set `SASY_NIX_RUNTIME_MANIFEST` to an immutable, root-owned
//! store JSON file. Only explicitly listed canonical store items are mounted;
//! neither `/nix/store` nor a caller-selected directory is an admissible root.
//! The selection is read once per process, like the sandbox capability probe.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub(crate) const MANIFEST_ENV: &str = "SASY_NIX_RUNTIME_MANIFEST";
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_STORE_PATHS: usize = 4096;
const MAX_PATH_ENTRIES: usize = 128;
const TOOL_NAMES: &[&str] = &[
    "bwrap",
    "prlimit",
    "sh",
    "true",
    "python3",
    "souffle",
    "g++",
    "souffle-interpreted",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    store_paths: Vec<String>,
    path: Vec<String>,
    tools: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(crate) struct NixRuntime {
    pub(crate) store_paths: Vec<PathBuf>,
    pub(crate) search_path: String,
    tools: BTreeMap<String, PathBuf>,
    digest: String,
}

fn error(message: impl std::fmt::Display) -> String {
    format!("invalid {MANIFEST_ENV}: {message}")
}

/// Lexically recognize one concrete store item. No normalization may turn an
/// ancestor or traversal spelling into an allowed dependency.
fn store_item(path: &Path) -> Result<PathBuf, String> {
    let value = path.to_str().ok_or_else(|| error("non-UTF-8 store path"))?;
    if value.len() > 4096 || value.contains(['\0', '\n', '\r', ':']) {
        return Err(error("invalid store path characters or length"));
    }
    let rest = value
        .strip_prefix("/nix/store/")
        .ok_or_else(|| error("path is outside /nix/store"))?;
    let parts: Vec<_> = rest.split('/').collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || *part == "." || *part == "..")
    {
        return Err(error("non-normal store path"));
    }
    let name = parts[0];
    let (hash, suffix) = name
        .split_once('-')
        .ok_or_else(|| error("not a concrete Nix store item"))?;
    if hash.len() != 32
        || !hash
            .bytes()
            .all(|byte| b"0123456789abcdfghijklmnpqrsvwxyz".contains(&byte))
        || suffix.is_empty()
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"+-._?=".contains(&byte))
    {
        return Err(error("not a concrete Nix store item"));
    }
    Ok(Path::new("/nix/store").join(name))
}

fn trusted_ownership(uid: u32, mode: u32) -> bool {
    uid == 0 && mode & 0o222 == 0
}

fn immutable_root_owned(metadata: &Metadata) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if !trusted_ownership(metadata.uid(), metadata.mode()) {
            return Err(error("package files must be root-owned and read-only"));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err(error("Nix runtime manifests require Unix ownership checks"))
    }
}

/// Check each real ancestor below the store item too: read-only file contents
/// alone are insufficient if a writable parent permits replacing that file.
fn trusted_store_path(path: &Path) -> Result<PathBuf, String> {
    let item = store_item(path)?;
    let canonical =
        fs::canonicalize(path).map_err(|e| error(format!("cannot resolve package path: {e}")))?;
    if canonical != path {
        return Err(error("package path must be canonical (no symlinks)"));
    }
    let mut current = path;
    loop {
        let metadata = fs::symlink_metadata(current)
            .map_err(|e| error(format!("cannot inspect package path: {e}")))?;
        immutable_root_owned(&metadata)?;
        if current == item {
            break;
        }
        current = current
            .parent()
            .ok_or_else(|| error("missing store item parent"))?;
    }
    Ok(item)
}

/// Validate the spelling used for execution as well as its resolved target.
/// A symlink itself normally has mode0777, but its owner and immutable parent
/// directory protect replacement. Preserve the name: multicall binaries such
/// as coreutils dispatch on argv[0] (`true` is not equivalent to `coreutils`).
fn trusted_launcher_path(path: &Path, roots: &BTreeSet<PathBuf>) -> Result<(), String> {
    let item = store_item(path)?;
    let mut current = path;
    loop {
        let metadata = fs::symlink_metadata(current)
            .map_err(|e| error(format!("cannot inspect launcher path: {e}")))?;
        if metadata.file_type().is_symlink() {
            // An intermediate link may otherwise hide a writable target
            // directory even when the final executable is immutable.
            let target = fs::canonicalize(current)
                .map_err(|e| error(format!("cannot resolve launcher ancestor: {e}")))?;
            if !roots.contains(&store_item(&target)?) {
                return Err(error("launcher ancestor resolves outside selected closure"));
            }
            trusted_store_path(&target)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.uid() != 0 {
                    return Err(error("package launcher symlinks must be root-owned"));
                }
            }
            #[cfg(not(unix))]
            return Err(error(
                "package launcher symlinks require Unix ownership checks",
            ));
        } else {
            immutable_root_owned(&metadata)?;
        }
        if current == item {
            break;
        }
        current = current
            .parent()
            .ok_or_else(|| error("missing launcher parent"))?;
    }
    Ok(())
}

impl NixRuntime {
    fn parse(bytes: &[u8]) -> Result<Manifest, String> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(error("manifest exceeds 1 MiB"));
        }
        let manifest: Manifest =
            serde_json::from_slice(bytes).map_err(|e| error(format!("invalid JSON: {e}")))?;
        if manifest.version != 1 {
            return Err(error("unsupported manifest version"));
        }
        if manifest.store_paths.is_empty()
            || manifest.store_paths.len() > MAX_STORE_PATHS
            || manifest.path.is_empty()
            || manifest.path.len() > MAX_PATH_ENTRIES
        {
            return Err(error("empty or oversized closure/PATH"));
        }
        if manifest
            .tools
            .keys()
            .any(|key| !TOOL_NAMES.contains(&key.as_str()))
        {
            return Err(error("unknown tool role"));
        }
        for name in TOOL_NAMES {
            if *name == "souffle-interpreted" {
                continue;
            }
            if (cfg!(target_os = "linux") || !matches!(*name, "bwrap" | "prlimit"))
                && !manifest.tools.contains_key(*name)
            {
                return Err(error(format!("missing pinned tool {name}")));
            }
        }
        Ok(manifest)
    }

    fn load(path: &Path) -> Result<Self, String> {
        trusted_store_path(path)?;
        let file = File::open(path).map_err(|e| error(format!("cannot open manifest: {e}")))?;
        let metadata = file
            .metadata()
            .map_err(|e| error(format!("cannot inspect manifest: {e}")))?;
        immutable_root_owned(&metadata)?;
        if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES as u64 {
            return Err(error("manifest must be a bounded regular file"));
        }
        let mut bytes = Vec::new();
        file.take((MAX_MANIFEST_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| error(format!("cannot read manifest: {e}")))?;
        let manifest = Self::parse(&bytes)?;
        let mut roots = BTreeSet::new();
        for entry in &manifest.store_paths {
            let root = PathBuf::from(entry);
            if store_item(&root)? != root || !roots.insert(root.clone()) {
                return Err(error(
                    "closure entries must be unique complete store-item roots",
                ));
            }
            trusted_store_path(&root)?;
            let metadata = fs::metadata(&root)
                .map_err(|e| error(format!("cannot inspect closure item: {e}")))?;
            if !metadata.is_file() && !metadata.is_dir() {
                return Err(error("closure item is not a regular file or directory"));
            }
        }
        let resolve = |value: &str| -> Result<PathBuf, String> {
            let path = Path::new(value);
            if !roots.contains(&store_item(path)?) {
                return Err(error("path is outside selected closure"));
            }
            trusted_launcher_path(path, &roots)?;
            let canonical = fs::canonicalize(path)
                .map_err(|e| error(format!("cannot resolve selected path: {e}")))?;
            if !roots.contains(&store_item(&canonical)?) {
                return Err(error("symlink resolves outside selected closure"));
            }
            trusted_store_path(&canonical)?;
            Ok(canonical)
        };
        let mut search_paths = Vec::new();
        for entry in &manifest.path {
            let resolved = resolve(entry)?;
            if !resolved.is_dir() {
                return Err(error("PATH entry is not a directory"));
            }
            search_paths.push(resolved.to_string_lossy().into_owned());
        }
        let mut tools = BTreeMap::new();
        for (name, value) in &manifest.tools {
            let path = resolve(value)?;
            let metadata =
                fs::metadata(&path).map_err(|e| error(format!("cannot inspect tool: {e}")))?;
            if !metadata.is_file() || metadata.len() > 64 << 20 {
                return Err(error("pinned tool is not a bounded regular file"));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if metadata.permissions().mode() & 0o111 == 0 {
                    return Err(error("pinned tool is not executable"));
                }
            }
            // Validation above covers both the immutable launcher spelling
            // and its canonical target; invocation must retain argv[0].
            tools.insert(name.clone(), PathBuf::from(value));
        }
        Ok(Self {
            store_paths: roots.into_iter().collect(),
            search_path: search_paths.join(":"),
            tools,
            digest: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    pub(crate) fn tool(&self, name: &str) -> Option<&Path> {
        self.tools.get(name).map(PathBuf::as_path)
    }

    /// A separately requested store mount may only narrow the fixed closure.
    pub(crate) fn check_mount(&self, path: &Path) -> Result<(), String> {
        if path.starts_with("/nix") {
            let root = store_item(path)?;
            let canonical =
                fs::canonicalize(path).map_err(|e| error(format!("cannot resolve mount: {e}")))?;
            if self.store_paths.binary_search(&root).is_err()
                || self
                    .store_paths
                    .binary_search(&store_item(&canonical)?)
                    .is_err()
            {
                return Err(error("additional mount is outside selected closure"));
            }
        }
        Ok(())
    }

    /// Dependency selection is part of compiler identity, even when a tool's
    /// executable bytes themselves have not changed.
    pub(crate) fn cache_identity(&self, executable_digest: &str) -> String {
        format!(
            "{:x}",
            Sha256::digest(format!(
                "sasy-nix-runtime-v1\n{}\n{}",
                self.digest, executable_digest
            ))
        )
    }
}

pub(crate) fn configured() -> Result<Option<&'static NixRuntime>, String> {
    static RUNTIME: OnceLock<Result<Option<NixRuntime>, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| match std::env::var_os(MANIFEST_ENV) {
            None => Ok(None),
            Some(path) if path.is_empty() => Err(error("empty manifest path")),
            Some(path) => NixRuntime::load(Path::new(&path)).map(Some),
        })
        .as_ref()
        .map(Option::as_ref)
        .map_err(Clone::clone)
}

#[cfg(test)]
impl NixRuntime {
    pub(crate) fn fixture() -> Self {
        let root = PathBuf::from("/nix/store/00000000000000000000000000000000-selected-runtime");
        Self {
            store_paths: vec![root.clone()],
            search_path: root.join("bin").to_string_lossy().into_owned(),
            tools: TOOL_NAMES
                .iter()
                .map(|name| (name.to_string(), root.join("bin").join(name)))
                .collect(),
            digest: "selected-manifest".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> serde_json::Value {
        let root = "/nix/store/00000000000000000000000000000000-runtime";
        serde_json::json!({
            "version": 1,
            "store_paths": [root],
            "path": [format!("{root}/bin")],
            "tools": TOOL_NAMES.iter().map(|name| (name.to_string(), format!("{root}/bin/{name}"))).collect::<BTreeMap<_, _>>()
        })
    }

    #[test]
    fn accepts_only_concrete_normal_store_items() {
        let root = "/nix/store/00000000000000000000000000000000-tool-1.0";
        assert_eq!(store_item(Path::new(root)).unwrap(), Path::new(root));
        assert_eq!(
            store_item(&Path::new(root).join("bin/c++")).unwrap(),
            Path::new(root)
        );
        for rejected in [
            "/",
            "/nix",
            "/nix/store",
            "/nix/store/",
            "/etc",
            "relative",
            "/nix/store/not-a-store-item",
            "/nix/store/00000000000000000000000000000000-",
            "/nix/store/00000000000000000000000000000000-x/../secret",
            "/nix/store/00000000000000000000000000000000-x//bin",
            "/nix/store/00000000000000000000000000000000-x/./bin",
            "/nix/store/00000000000000000000000000000000-x/bin:",
            "/nix/store/00000000000000000000000000000000-x/\n",
        ] {
            assert!(
                store_item(Path::new(rejected)).is_err(),
                "accepted {rejected:?}"
            );
        }
    }

    #[test]
    fn manifest_schema_is_versioned_bounded_and_closed() {
        assert!(NixRuntime::parse(&serde_json::to_vec(&manifest()).unwrap()).is_ok());
        for (field, value) in [
            ("version", serde_json::json!(2)),
            ("version", serde_json::json!("1")),
            ("store_paths", serde_json::json!([])),
            (
                "store_paths",
                serde_json::json!(vec!["x"; MAX_STORE_PATHS + 1]),
            ),
            ("path", serde_json::json!([])),
            ("path", serde_json::json!(vec!["x"; MAX_PATH_ENTRIES + 1])),
            ("environment", serde_json::json!({"PATH": "/untrusted"})),
        ] {
            let mut value_to_test = manifest();
            value_to_test[field] = value;
            assert!(
                NixRuntime::parse(&serde_json::to_vec(&value_to_test).unwrap()).is_err(),
                "accepted {field}"
            );
        }
        let mut value = manifest();
        value["tools"]["custom-request-tool"] = serde_json::json!("/tmp/executable");
        assert!(NixRuntime::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        value = manifest();
        value["tools"].as_object_mut().unwrap().remove("souffle");
        assert!(NixRuntime::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        assert!(NixRuntime::parse(&vec![b' '; MAX_MANIFEST_BYTES + 1]).is_err());
        assert!(NixRuntime::parse(b"{\"version\":1,\"version\":1}").is_err());
        assert!(NixRuntime::parse(b"{malformed}").is_err());
    }

    #[test]
    fn package_ownership_cannot_be_replaced_by_file_readonly_alone() {
        assert!(trusted_ownership(0, 0o100444));
        assert!(trusted_ownership(0, 0o040555));
        assert!(!trusted_ownership(1000, 0o100444));
        for writable in [0o100644, 0o100464, 0o100446, 0o040755] {
            assert!(!trusted_ownership(0, writable));
        }
    }

    #[test]
    fn caller_manifest_files_are_rejected_even_with_valid_json() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("runtime.json");
        fs::write(&path, serde_json::to_vec(&manifest()).unwrap()).unwrap();
        assert!(NixRuntime::load(&path)
            .unwrap_err()
            .contains("outside /nix/store"));
        assert!(NixRuntime::load(Path::new("/nix/store")).is_err());
    }

    #[test]
    fn dependency_manifest_changes_invalidate_compiler_identity() {
        let first = NixRuntime::fixture();
        let mut second = NixRuntime::fixture();
        second.digest = "different-dependency-closure".into();
        assert_ne!(
            first.cache_identity("same-tool-bytes"),
            second.cache_identity("same-tool-bytes")
        );
        assert_ne!(
            first.cache_identity("compiler-a"),
            first.cache_identity("compiler-b")
        );
        assert_eq!(
            first.cache_identity("compiler-a"),
            first.cache_identity("compiler-a")
        );
    }

    #[test]
    fn no_mount_may_expand_the_store_ancestor() {
        let runtime = NixRuntime::fixture();
        for path in ["/nix", "/nix/store", "/nix/store/not-an-item"] {
            assert!(runtime.check_mount(Path::new(path)).is_err());
        }
        // Existing per-spawn policy workspaces are independent of toolchain roots.
        assert!(runtime
            .check_mount(Path::new("/tmp/sasy-policy-workspace"))
            .is_ok());
    }

    #[test]
    fn invalid_manifest_fails_closed_in_a_fresh_process() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("untrusted.json");
        fs::write(&path, serde_json::to_vec(&manifest()).unwrap()).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "nix_runtime::tests::invalid_manifest_child",
                "--nocapture",
            ])
            .env(MANIFEST_ENV, &path)
            .env("DISABLE_BWRAP", "1")
            .env("SASY_EVALUATOR_BWRAP", "0")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }

    #[test]
    #[ignore = "subprocess-only environment/cache boundary control"]
    fn invalid_manifest_child() {
        assert!(configured().is_err());
        assert!(crate::sandbox::check_sandbox().is_err());
        assert!(crate::sandbox::build_bwrap_argv(
            "true",
            &[],
            Path::new("/tmp/work"),
            false,
            &[],
            &[],
            None,
            true,
            None
        )
        .is_err());
        #[cfg(feature = "compiler")]
        assert!(
            crate::sandbox::sandboxed_command("true", &[], Path::new("/tmp/work"), false, &[])
                .is_err()
        );
    }
}
