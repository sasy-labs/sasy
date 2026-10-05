use super::*;
use tempfile::TempDir;

#[test]
fn authored_inference_markers_cannot_hide_outputs_or_override_gate_provenance() {
    let source = "// === USER_POLICY_BEGIN ===\n/*\n// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\nCurrentDependsPolicyRelevant().\n*/\nR(\"// === USER_POLICY_BEGIN ===\").\n";
    let cleaned = clear_authored_inference_markers(source);
    assert!(cleaned.contains("// === AUTHORED_POLICY_BOUNDARY ==="));
    assert!(cleaned.contains("// SASY_AUTHORED_GATE_MARKER: CurrentDependsPolicyRelevant"));
    assert!(cleaned.contains("R(\"// === USER_POLICY_BEGIN ===\")."));
    assert!(apply_policy_inference(".decl SasyAllowRoute(x: symbol)\n").is_err());
    assert!(apply_policy_inference(".decl R(x: symbol)\nR(\"SasyAllowRoute\").\n").is_ok());
}

/// `SASY_SOUFFLE_ASSETS`, the build-cache directory and the working directory
/// are process-wide, so the tests that touch any of them take the crate's one
/// environment lock rather than racing each other.
use crate::TEST_ENV_LOCK as ASSET_ENV_LOCK;

/// Put the variable back the way the test found it.
fn restore_asset_env(previous: Option<std::ffi::OsString>) {
    match previous {
        Some(value) => std::env::set_var("SASY_SOUFFLE_ASSETS", value),
        None => std::env::remove_var("SASY_SOUFFLE_ASSETS"),
    }
}

/// The assets a compile test runs against: this binary's own embedded set,
/// materialized once into the test process's temp directory, plus the host's
/// Soufflé headers. No repo path, so the test runs from any cwd — and it
/// compiles against the same bytes the shipped binary would.
fn discover_for_test() -> SouffleAssets {
    let souffle_dir =
        crate::assets::materialized_dir(&std::env::temp_dir().join("sasy-policy-test-assets"));
    crate::assets::materialize(&souffle_dir, crate::assets::ASSETS)
        .expect("embedded assets must materialize");
    SouffleAssets {
        include_dir: [
            "/opt/homebrew/include",
            "/opt/souffle/include",
            "/usr/local/include",
            "/usr/include",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .expect("Soufflé include dir not found"),
        sugar_py: souffle_dir.join("sugar.py"),
        evaluator_shim: souffle_dir.join("evaluator_shim.cpp"),
        evaluator_protocol: souffle_dir.join("evaluator_protocol.h"),
        json_string_codec: souffle_dir.join("json_string_codec.h"),
        functors_common: souffle_dir.join("functors_common.cpp"),
        common_policy: Some(souffle_dir.join("common_policy.dl")),
        exclusive_dir: Some(souffle_dir),
        word_size: detect_souffle_word_size().unwrap_or(32),
    }
}

/// A directory holding the embedded set, for tests that point
/// `SASY_SOUFFLE_ASSETS` at one.
fn materialized_assets(root: &Path) -> PathBuf {
    let dir = crate::assets::materialized_dir(root);
    crate::assets::materialize(&dir, crate::assets::ASSETS).unwrap();
    dir
}

#[test]
fn an_asset_directory_named_by_the_environment_is_used_as_it_stands() {
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = TempDir::new().unwrap();
    let dir = materialized_assets(root.path());

    let previous = std::env::var_os("SASY_SOUFFLE_ASSETS");
    std::env::set_var("SASY_SOUFFLE_ASSETS", &dir);
    let resolved = assets_dir().expect("a complete directory must be accepted");
    restore_asset_env(previous);

    assert_eq!(resolved, dir);
}

#[test]
fn an_asset_directory_missing_a_file_is_refused_by_name() {
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = TempDir::new().unwrap();
    let dir = materialized_assets(root.path());
    std::fs::remove_file(dir.join("functors_common.cpp")).unwrap();

    let previous = std::env::var_os("SASY_SOUFFLE_ASSETS");
    std::env::set_var("SASY_SOUFFLE_ASSETS", &dir);
    let error = assets_dir().expect_err("an incomplete directory must be refused");
    restore_asset_env(previous);

    let message = error.to_string();
    assert!(
        message.contains("functors_common.cpp"),
        "the error must name the missing file: {message}"
    );
    assert!(message.contains("SASY_SOUFFLE_ASSETS"), "{message}");
}

#[test]
fn discovery_without_an_override_materializes_under_the_build_cache_root() {
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = TempDir::new().unwrap();

    let previous_assets = std::env::var_os("SASY_SOUFFLE_ASSETS");
    let previous_cache = std::env::var_os("SASY_SOUFFLE_BUILD_CACHE_DIR");
    std::env::remove_var("SASY_SOUFFLE_ASSETS");
    std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", root.path());
    let resolved = assets_dir();
    restore_asset_env(previous_assets);
    match previous_cache {
        Some(v) => std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", v),
        None => std::env::remove_var("SASY_SOUFFLE_BUILD_CACHE_DIR"),
    }

    let resolved = resolved.expect("materialization must succeed");
    assert_eq!(
        resolved,
        root.path()
            .join("assets")
            .join(crate::assets::asset_set_hash())
    );
    assert!(resolved.join("sugar.py").is_file());
}

/// Set `SASY_SOUFFLE_BUILD_CACHE_DIR`, `SASY_SOUFFLE_ASSETS`, `XDG_CACHE_HOME`
/// and `HOME` for the duration of the closure, then put all four back.
fn with_asset_env<T>(values: [(&str, Option<&std::ffi::OsStr>); 4], body: impl FnOnce() -> T) -> T {
    let previous: Vec<(String, Option<std::ffi::OsString>)> = values
        .iter()
        .map(|(name, _)| ((*name).to_string(), std::env::var_os(name)))
        .collect();
    for (name, value) in values {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
    let out = body();
    for (name, value) in previous {
        match value {
            Some(value) => std::env::set_var(&name, value),
            None => std::env::remove_var(&name),
        }
    }
    out
}

#[test]
fn a_relatively_named_build_cache_still_gives_an_absolute_assets_directory() {
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let relative = std::ffi::OsString::from("relative-cache-root");
    let resolved = with_asset_env(
        [
            ("SASY_SOUFFLE_BUILD_CACHE_DIR", Some(relative.as_os_str())),
            ("SASY_SOUFFLE_ASSETS", None),
            ("XDG_CACHE_HOME", None),
            ("HOME", None),
        ],
        assets_root,
    );

    assert!(
        resolved.is_absolute(),
        "a relative cache root must be resolved: {}",
        resolved.display()
    );
    assert_eq!(
        resolved,
        std::env::current_dir().unwrap().join(&relative),
        "resolved against the working directory, once"
    );
}

#[test]
fn with_the_cache_off_the_assets_land_under_the_user_cache_directory() {
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cache_home = TempDir::new().unwrap();
    let off = std::ffi::OsString::from("off");
    let resolved = with_asset_env(
        [
            ("SASY_SOUFFLE_BUILD_CACHE_DIR", Some(off.as_os_str())),
            ("SASY_SOUFFLE_ASSETS", None),
            ("XDG_CACHE_HOME", Some(cache_home.path().as_os_str())),
            ("HOME", None),
        ],
        || assets_dir().expect("materialization must succeed with the cache off"),
    );

    assert!(
        resolved.starts_with(cache_home.path().join("sasy")),
        "{} is not under the user cache directory",
        resolved.display()
    );
    assert!(resolved.join("sugar.py").is_file());
}

/// Stands the process in a directory and puts it back when dropped — on a
/// panic too, so a failed assertion does not leave every later test running in
/// a temp directory that is about to be deleted.
struct CwdGuard(PathBuf);

impl CwdGuard {
    fn enter(dir: &Path) -> Self {
        let previous = std::env::current_dir().expect("a working directory");
        std::env::set_current_dir(dir).expect("move into the directory");
        Self(previous)
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn the_bootstrap_build_directory_hangs_off_the_data_directory() {
    // The working directory is read-only, and the data directory is absolute:
    // the case a systemd unit with no `WorkingDirectory=` is in. The process
    // actually stands in that directory while the call runs — otherwise
    // "nothing was written beside the working directory" asserts about a
    // directory nothing was ever going to write to. The move is process-wide,
    // so this holds the same lock as the tests that set the asset environment.
    use std::os::unix::fs::PermissionsExt;
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let read_only_cwd = TempDir::new().unwrap();
    let read_only_cwd = std::fs::canonicalize(read_only_cwd.path()).unwrap();
    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o500)).unwrap();

    let data = TempDir::new().unwrap();
    let data_dir = std::fs::canonicalize(data.path()).unwrap();

    let cwd = CwdGuard::enter(&read_only_cwd);
    let dir = bootstrap_build_dir(&data_dir).expect("an absolute data dir must be usable");
    let entries = std::fs::read_dir(&read_only_cwd).unwrap().count();
    drop(cwd);
    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o700)).unwrap();

    assert_eq!(dir, data_dir.join("souffle-build").join("bootstrap"));
    assert!(dir.starts_with(&data_dir), "{}", dir.display());
    assert!(dir.is_absolute() && dir.is_dir());
    assert_eq!(
        entries, 0,
        "nothing may be written beside the working directory"
    );
    let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700, "the build directory is the compile's own");

    // Idempotent: a restart reuses it.
    assert_eq!(bootstrap_build_dir(&data_dir).unwrap(), dir);
}

#[test]
fn an_unwritable_data_directory_is_named_in_the_bootstrap_error() {
    use std::os::unix::fs::PermissionsExt;
    if unsafe { libc::geteuid() } == 0 {
        return; // root writes into a 0o500 directory regardless.
    }
    let td = TempDir::new().unwrap();
    let data_dir = td.path().join("unwritable");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::set_permissions(&data_dir, std::fs::Permissions::from_mode(0o500)).unwrap();

    let error = bootstrap_build_dir(&data_dir).expect_err("an unwritable data dir must fail");
    let message = error.to_string();
    std::fs::set_permissions(&data_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        message.contains(
            &data_dir
                .join("souffle-build")
                .join("bootstrap")
                .display()
                .to_string()
        ),
        "the error must name the directory it could not create: {message}"
    );
}

#[test]
fn the_compile_sandbox_binds_the_asset_directory_once() {
    let root = TempDir::new().unwrap();
    let dir = materialized_assets(root.path());
    let assets = SouffleAssets {
        include_dir: PathBuf::from("/usr/include"),
        sugar_py: dir.join("sugar.py"),
        evaluator_shim: dir.join("evaluator_shim.cpp"),
        evaluator_protocol: dir.join("evaluator_protocol.h"),
        json_string_codec: dir.join("json_string_codec.h"),
        functors_common: dir.join("functors_common.cpp"),
        common_policy: Some(dir.join("common_policy.dl")),
        exclusive_dir: Some(dir.clone()),
        word_size: 64,
    };

    let binds = compile_ro_binds(&assets);
    assert_eq!(
        binds,
        vec![Path::new("/usr/include"), dir.as_path()],
        "the headers and the one asset directory, nothing else"
    );

    let build_dir = TempDir::new().unwrap();
    let argv = crate::sandbox::build_bwrap_argv(
        "g++",
        &["-c", "policy_program.cpp"],
        build_dir.path(),
        true,
        &binds,
        &[],
        None,
        true,
        // The compile chain's bwrap carries no seccomp descriptor; only the
        // evaluator spawn does.
        None,
    )
    .unwrap();
    let asset_binds = argv
        .windows(2)
        .filter(|w| w[0] == "--ro-bind" && w[1] == dir.to_string_lossy())
        .count();
    assert_eq!(asset_binds, 1, "the asset directory is bound exactly once");
}

#[test]
fn discovery_standing_in_a_checkout_takes_the_embedded_set_not_the_checkout() {
    // A decoy `souffle` tree under the working directory, holding
    // files with the right names and the wrong contents: what discovery used to
    // find by walking repo-relative paths. What it must return instead is the
    // materialized set, under the build cache.
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let decoy_root = TempDir::new().unwrap();
    let decoy = decoy_root.path().join(".").join("souffle");
    std::fs::create_dir_all(&decoy).unwrap();
    for (name, _) in crate::assets::ASSETS {
        std::fs::write(decoy.join(name), "# planted by another checkout\n").unwrap();
    }
    let cache = TempDir::new().unwrap();

    let previous_assets = std::env::var_os("SASY_SOUFFLE_ASSETS");
    let previous_cache = std::env::var_os("SASY_SOUFFLE_BUILD_CACHE_DIR");
    let previous_cwd = std::env::current_dir().expect("a working directory");
    std::env::remove_var("SASY_SOUFFLE_ASSETS");
    std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", cache.path());
    std::env::set_current_dir(decoy_root.path()).expect("stand in the decoy checkout");
    let discovered = SouffleAssets::discover();
    std::env::set_current_dir(previous_cwd).expect("come back");
    restore_asset_env(previous_assets);
    match previous_cache {
        Some(v) => std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", v),
        None => std::env::remove_var("SASY_SOUFFLE_BUILD_CACHE_DIR"),
    }

    let discovered = match discovered {
        Ok(assets) => assets,
        // Discovery also resolves the host's Soufflé headers and word size;
        // without a Soufflé install there is nothing here to say.
        Err(error) => {
            eprintln!("skipped: no host Soufflé toolchain ({error})");
            return;
        }
    };
    let expected = cache
        .path()
        .join("assets")
        .join(crate::assets::asset_set_hash());
    for path in [
        &discovered.sugar_py,
        &discovered.evaluator_shim,
        &discovered.evaluator_protocol,
        &discovered.json_string_codec,
        &discovered.functors_common,
        discovered.common_policy.as_ref().unwrap(),
    ] {
        assert_eq!(
            path.parent(),
            Some(expected.as_path()),
            "{} is not the materialized set",
            path.display()
        );
        assert!(!path.starts_with(decoy_root.path()), "{}", path.display());
    }
    assert_eq!(
        std::fs::read_to_string(&discovered.sugar_py).unwrap(),
        crate::assets::ASSETS
            .iter()
            .find(|(name, _)| *name == "sugar.py")
            .unwrap()
            .1,
        "the embedded bytes, not the decoy's"
    );
}

// The deployment-image test remains in the private deployment repository.

#[test]
fn assured_assets_sharing_one_directory_are_still_bound_file_by_file() {
    // The assured-runtime path forces the shim and its two headers into one
    // directory, and the usual layout puts the fourth file there too — but
    // that directory is the source tree's `souffle/`, which also holds
    // functor and shim sources. Shape is not provenance: only a directory
    // this binary or `sasy install-assets` created is bound whole.
    let root = TempDir::new().unwrap();
    let dir = root.path().join("souffle");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("functors.cpp"), "not the compiler's").unwrap();

    let assets = SouffleAssets {
        include_dir: PathBuf::from("/usr/include"),
        sugar_py: dir.join("sugar.py"),
        evaluator_shim: dir.join("evaluator_shim.cpp"),
        evaluator_protocol: dir.join("evaluator_protocol.h"),
        json_string_codec: dir.join("json_string_codec.h"),
        functors_common: dir.join("functors_common.cpp"),
        common_policy: Some(dir.join("common_policy.dl")),
        exclusive_dir: None,
        word_size: 64,
    };

    let binds = compile_ro_binds(&assets);
    assert_eq!(
        binds,
        vec![
            Path::new("/usr/include"),
            assets.evaluator_shim.as_path(),
            assets.evaluator_protocol.as_path(),
            assets.json_string_codec.as_path(),
            assets.functors_common.as_path(),
        ],
        "one directory by accident is not one directory by provenance"
    );
    assert!(
        !binds.contains(&dir.as_path()),
        "the directory holding functors.cpp may not be bound: {binds:?}"
    );
}

#[test]
fn an_operator_installed_asset_directory_is_bound_whole() {
    // `SASY_SOUFFLE_ASSETS` names a directory written by `sasy install-assets`
    // — the set and nothing else — so discovery marks it exclusive and the
    // sandbox binds it once.
    let _guard = ASSET_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = TempDir::new().unwrap();
    let dir = materialized_assets(root.path());

    let previous = std::env::var_os("SASY_SOUFFLE_ASSETS");
    std::env::set_var("SASY_SOUFFLE_ASSETS", &dir);
    let discovered = SouffleAssets::discover();
    restore_asset_env(previous);

    let discovered = match discovered {
        Ok(assets) => assets,
        // Discovery also resolves the host's Soufflé headers and word size;
        // without a Soufflé install there is nothing here to say.
        Err(error) => {
            eprintln!("skipped: no host Soufflé toolchain ({error})");
            return;
        }
    };
    assert_eq!(discovered.exclusive_dir.as_deref(), Some(dir.as_path()));
    let binds = compile_ro_binds(&discovered);
    assert_eq!(
        binds,
        vec![discovered.include_dir.as_path(), dir.as_path()],
        "the headers and the installed directory, nothing else"
    );
}

#[test]
fn assets_spread_over_several_directories_are_bound_file_by_file() {
    // The assured-runtime path takes each asset path from a signed manifest,
    // and those directories may hold anything else. Bind the files, not the
    // directories they happen to sit in.
    let root = TempDir::new().unwrap();
    let shim_dir = root.path().join("shim");
    let functors_dir = root.path().join("functors");
    std::fs::create_dir_all(&shim_dir).unwrap();
    std::fs::create_dir_all(&functors_dir).unwrap();
    std::fs::write(functors_dir.join("a-secret"), "not the compiler's").unwrap();

    let assets = SouffleAssets {
        include_dir: PathBuf::from("/usr/include"),
        sugar_py: shim_dir.join("sugar.py"),
        evaluator_shim: shim_dir.join("evaluator_shim.cpp"),
        evaluator_protocol: shim_dir.join("evaluator_protocol.h"),
        json_string_codec: shim_dir.join("json_string_codec.h"),
        functors_common: functors_dir.join("functors_common.cpp"),
        common_policy: Some(shim_dir.join("common_policy.dl")),
        exclusive_dir: None,
        word_size: 64,
    };

    let binds = compile_ro_binds(&assets);
    assert_eq!(
        binds,
        vec![
            Path::new("/usr/include"),
            assets.evaluator_shim.as_path(),
            assets.evaluator_protocol.as_path(),
            assets.json_string_codec.as_path(),
            assets.functors_common.as_path(),
        ],
        "each asset on its own, no directory of them"
    );
    assert!(
        !binds.contains(&functors_dir.as_path()) && !binds.contains(&shim_dir.as_path()),
        "no directory holding other files may be bound: {binds:?}"
    );
}

/// End-to-end: compile a trivial policy twice with the build
/// cache enabled. The second compile must hit the cache, which
/// we verify two ways:
///   1. The cache directory contains an entry after the first
///      compile, and the second compile reuses it (binary appears
///      in the second build dir).
///   2. On the second compile (a cache hit) the souffle/g++
///      pipeline is skipped, so `policy_program.cpp` is *not*
///      created in the second build dir.
///
/// Ignored by default because it shells out to souffle + g++
/// (~5–10 s on a fresh build). Run with
/// `cargo test --release -p sasy-policy compile_then_cache_hit_skips_souffle_gxx -- --ignored --nocapture`.
#[test]
#[ignore]
fn compile_then_cache_hit_skips_souffle_gxx() {
    let assets = discover_for_test();
    // The cache directory is handed to the compile calls rather than named in
    // the environment: the environment is process-wide, and the other ignored
    // tests compile in parallel in this same process, so a sibling compile
    // would deposit its own entry here and the count below would be wrong.
    let cache_dir = TempDir::new().unwrap();
    let cache_root = Some(cache_dir.path());

    let policy = ".decl Hello(x: symbol)\nHello(\"world\").\n";

    // First compile: cold cache.
    let build_a = TempDir::new().unwrap();
    let r1 = compile_souffle_into_cache(policy, None, build_a.path(), &assets, cache_root)
        .expect("first compile must succeed");
    assert!(r1.binary_path.exists(), "first-compile binary must exist");
    assert!(
        build_a.path().join("policy_program.cpp").exists(),
        "first compile should run souffle -g (policy_program.cpp present)",
    );
    let cached: Vec<_> = std::fs::read_dir(cache_dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(cached.len(), 1, "exactly one cache entry expected");

    // Second compile: warm cache, in a fresh build dir.
    let build_b = TempDir::new().unwrap();
    let r2 = compile_souffle_into_cache(policy, None, build_b.path(), &assets, cache_root)
        .expect("second compile must succeed via cache");
    assert!(
        r2.binary_path.exists(),
        "cached binary must be materialized"
    );
    assert!(
        !build_b.path().join("policy_program.cpp").exists(),
        "cache hit must skip souffle -g (no policy_program.cpp)",
    );
    // Desugared file is still expected — load_rule_metadata reads it.
    assert!(build_b.path().join("desugared.dl").exists());
}

/// End-to-end proof that per-action `action_metadata` reaches the
/// evaluator: the shim seeds `ActionMetadata(idx, rel, a, b)` per query,
/// a policy projects it (`PackageVerdict`), and a `MAL` verdict flips
/// an otherwise-allowed action to denied. Exercises the full pipeline
/// (sugar.py → souffle -g → g++ shim → IPC query). Ignored by default
/// (shells out to the toolchain); run with `--ignored --nocapture`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn action_metadata_seeds_per_query_edb() {
    use crate::evaluator::manager::EvaluatorProcess;
    use crate::evaluator::types::{
        ActionMetadataEntry, EvalAction, EvalAuthRequest, PolicyMetadataFact,
    };
    use crate::evaluator::Evaluator;

    let assets = discover_for_test();
    let build = TempDir::new().unwrap();
    // Allow every tool call; a block-kind DenialReason fires (→
    // Unauthorized via common_policy) iff THIS action carries a
    // pkg_verdict=MAL fact in its per-action metadata.
    let policy = r#"#include "common_policy.dl"

.decl PackageVerdict(pkg: symbol, verdict: symbol)
PackageVerdict(p, v) :- ActionMetadata(_, "pkg_verdict", p, v).

IsAuthorized(idx) :- Actions(idx, _).
DenialReason(idx, "block", "malicious package (OSV MAL-)", "remove the dependency") :-
    Actions(idx, _), PackageVerdict(_, "MAL").
"#;
    let r = compile_souffle_with_assets(policy, None, build.path(), &assets)
        .expect("policy must compile");
    let ev = EvaluatorProcess::souffle(r.binary_path, "policy_program".into())
        .expect("evaluator must spawn");

    let mk_req = |meta: Vec<ActionMetadataEntry>| EvalAuthRequest {
        current_node_ids: vec![],
        actions: vec![EvalAction::ToolCall {
            fn_name: "Bash".into(),
            args: r#"{"command":"npm install foo"}"#.into(),
        }],
        entity: None,
        roles: vec![],
        tenant_id: Some("default".into()),
        session_id: None,
        principal: Some("tester".into()),
        action_metadata: meta,
    };

    // MAL verdict on this action's package → denied.
    let denied = ev
        .query(mk_req(vec![ActionMetadataEntry {
            index: 0,
            facts: vec![PolicyMetadataFact {
                rel: "pkg_verdict".into(),
                a: "foo".into(),
                b: "MAL".into(),
            }],
        }]))
        .await
        .expect("query must succeed");
    assert_eq!(denied.results.len(), 1);
    assert!(
        !denied.results[0].authorized,
        "a MAL pkg_verdict in action_metadata must deny"
    );

    // No metadata → the projection is empty, the action is allowed.
    let allowed = ev.query(mk_req(vec![])).await.expect("query must succeed");
    assert!(
        allowed.results[0].authorized,
        "no action_metadata → ActionMetadata empty → allowed"
    );
}

/// Edge attribution reaches the policy language: a policy that
/// refuses an action whose current node has an incoming dependency
/// edge asserted by a low-trust principal, one whose entity is
/// low-trust, or one that still carries instrumentation metadata.
/// Driven through the real compiled shim over
/// [`ATTRIBUTION_ROWS`], the same table the interpreted backend runs
/// in `edge_attribution_reaches_the_interpreted_backend`.
/// `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn edge_principal_decides_an_action() {
    use crate::evaluator::manager::EvaluatorProcess;

    let assets = discover_for_test();
    let build = TempDir::new().unwrap();
    let r = compile_souffle_with_assets(EDGE_PRINCIPAL_POLICY, None, build.path(), &assets)
        .expect("policy must compile");
    let ev = EvaluatorProcess::souffle(r.binary_path, "policy_program".into())
        .expect("evaluator must spawn");

    run_attribution_rows(&ev, Backend::Compiled).await;
}

/// The same table on the interpreted backend, which reaches Soufflé
/// through fact files rather than a compiled program: the two shims
/// must give the same answer on every row they both cover.
///
/// The interpreted adapter is a build product
/// (`bash souffle/build-test-runtime.sh`), so this test needs it built.
/// The assets are named from `CARGO_MANIFEST_DIR`, so the working
/// directory does not matter. Without the adapter the test FAILS —
/// set `SASY_TEST_SKIP_INTERPRETED=1` to skip it instead, which is
/// for hosts that cannot build it at all. `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn edge_attribution_reaches_the_interpreted_backend() {
    let work = TempDir::new().unwrap();
    let policy_path = work.path().join("edge_principal.dl");
    std::fs::write(&policy_path, EDGE_PRINCIPAL_POLICY).unwrap();

    let skip = std::env::var("SASY_TEST_SKIP_INTERPRETED").is_ok_and(|v| v == "1");
    let build = crate::evaluator::factory::build_souffle_factory_with_artifacts(
        sasy_common::Backend::SouffleInterpreted,
        work.path(),
        &policy_path,
        None,
        Some(&interpreted_assets_for_test()),
    );
    // Both steps have to honour the flag. Building the factory
    // succeeds without the adapter binary on disk — it is the spawn
    // that finds it missing — so a skip that guards only the build
    // still fails the test on a host that cannot build the adapter.
    let unavailable = |error: String| -> ! {
        panic!(
            "the interpreted backend must be available: {error}. \
             Build it with `bash souffle/build-test-runtime.sh`, or set \
             SASY_TEST_SKIP_INTERPRETED=1 to skip this test."
        )
    };
    let built = match build {
        Ok(built) => built,
        Err(error) if skip => {
            eprintln!("skipping on SASY_TEST_SKIP_INTERPRETED=1: {error}");
            return;
        }
        Err(error) => unavailable(error.to_string()),
    };
    let ev = match (built.factory)() {
        Ok(ev) => ev,
        Err(error) if skip => {
            eprintln!("skipping on SASY_TEST_SKIP_INTERPRETED=1: {error}");
            return;
        }
        Err(error) => unavailable(error.to_string()),
    };

    run_attribution_rows(ev.as_ref(), Backend::Interpreted).await;
}

/// The interpreted backend's assets, named from the crate's own
/// directory so the test does not depend on the working directory.
///
/// The compile-chain assets come from the embedded set (see
/// [`discover_for_test`]), which deliberately does not carry the
/// interpreted backend's adapter or its `functors.cpp`: those stay
/// build-time artifacts of the repository's `souffle/` directory
/// (`bash souffle/build-test-runtime.sh`), so they are named there.
fn interpreted_assets_for_test() -> crate::evaluator::factory::AuthenticatedSouffleAssets {
    let assets = discover_for_test();
    let souffle_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../souffle"));
    crate::evaluator::factory::AuthenticatedSouffleAssets {
        compiler: assets,
        interpreted_adapter: souffle_dir.join("souffle-interpreted"),
        interpreted_functors: souffle_dir.join("functors.cpp"),
    }
}

/// Which shim is under test. The interpreted adapter keeps no edge
/// metadata at all — it writes an empty `EdgeData.facts` — so rows
/// that turn on `EdgeData` are asserted on the compiled shim only.
/// Every other row must answer the same on both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    Compiled,
    Interpreted,
}

/// One row of the shared table: a sequence of graph updates, then
/// what each backend must answer afterwards.
struct AttributionRow {
    what: &'static str,
    /// Reset the evaluator before applying the updates, as a resync
    /// does: the shim throws its state away and the store replays.
    reset_first: bool,
    updates: fn() -> Vec<crate::engine::GraphUpdate>,
    /// (session, current node, expected authorization, why).
    expect: &'static [(&'static str, &'static str, bool, &'static str)],
    /// `false` when the row's expectations only hold on the compiled
    /// shim; its updates are still applied on both, so the two runs
    /// stay on the same state.
    both_backends: bool,
}

/// Every update sequence both shims are driven through, in order.
/// The rows are cumulative: each applies to the state the previous
/// ones left.
const ATTRIBUTION_ROWS: &[AttributionRow] = &[
    AttributionRow {
        what: "three edges recorded once: by the gateway, by the intern, by nobody",
        reset_first: false,
        updates: seeded_attributed_edges,
        expect: &[
            (
                "",
                "dst-gateway",
                true,
                "an edge asserted by the gateway principal is not low-trust",
            ),
            (
                "",
                "dst-intern",
                false,
                "an edge asserted by the intern principal denies the action",
            ),
            (
                "",
                "dst-anonymous",
                true,
                "an edge recorded with no principal has no EdgePrincipal tuple, \
                 so it matches neither the allow nor the deny side",
            ),
        ],
        both_backends: true,
    },
    AttributionRow {
        what: "the same edges re-recorded under the other principal",
        reset_first: false,
        updates: swapped_attributed_edges,
        expect: &[
            (
                "",
                "dst-gateway",
                false,
                "the edge now names the intern, so the gateway's superseded \
                 principal must not keep the action authorized",
            ),
            (
                "",
                "dst-intern",
                true,
                "the intern's principal was replaced by the gateway's",
            ),
        ],
        both_backends: true,
    },
    AttributionRow {
        what: "a re-record that carries no principal at all",
        reset_first: false,
        updates: unattributed_re_record,
        expect: &[(
            "",
            "dst-gateway",
            true,
            "the newest recording carries no principal, so the edge has none \
             and matches neither side",
        )],
        both_backends: true,
    },
    AttributionRow {
        what: "an edge whose client-named entity is low-trust",
        reset_first: false,
        updates: contractor_entity_edge,
        expect: &[(
            "",
            "dst-entity",
            false,
            "the edge names the contractor entity, which the policy denies",
        )],
        both_backends: true,
    },
    AttributionRow {
        what: "the same edge re-recorded naming a different entity",
        reset_first: false,
        updates: entity_replaced_edge,
        expect: &[(
            "",
            "dst-entity",
            true,
            "a recording that names an entity replaces the stored one, so the \
             contractor is gone and the deny rule no longer matches",
        )],
        both_backends: true,
    },
    AttributionRow {
        what: "a re-record that omits the entity: the store carries the stored \
               one forward, so the update still names it",
        reset_first: false,
        updates: entity_carried_forward_edge,
        expect: &[(
            "",
            "dst-keep",
            false,
            "omitting the entity does not clear it — the store merges the \
             stored entity into the recording and sends it, so the contractor \
             is still there to deny on",
        )],
        both_backends: true,
    },
    AttributionRow {
        what: "shim lifecycle only: an update that arrives carrying no entity \
               at all retracts the row (the store does not emit this for a \
               re-record that omits the entity — it sends the stored one)",
        reset_first: false,
        updates: entity_retracted_edge,
        expect: &[(
            "",
            "dst-keep",
            true,
            "an update with no entity is the shim's retraction path: the row \
             goes away and nothing remains to deny on",
        )],
        both_backends: true,
    },
    AttributionRow {
        what: "two sessions recording the same (m1, m2) edge, in both orders",
        reset_first: false,
        updates: two_session_updates,
        expect: TWO_SESSION_EXPECTATIONS,
        both_backends: true,
    },
    AttributionRow {
        what: "a resync: the shim's state is dropped and the store replays it",
        reset_first: true,
        updates: replayed_after_resync,
        expect: REPLAY_EXPECTATIONS,
        both_backends: true,
    },
    AttributionRow {
        what: "an edge that carries a message index, then one that dropped it",
        reset_first: false,
        updates: edge_metadata_then_dropped,
        expect: &[(
            "",
            "dst-meta",
            true,
            "the second recording dropped the message index, so the EdgeData \
             row is retracted and the metadata rule no longer denies",
        )],
        // The interpreted adapter carries no edge metadata: it writes
        // an empty EdgeData.facts, so this row can only be asserted
        // on the compiled shim.
        both_backends: false,
    },
    AttributionRow {
        what: "edges recorded only inside one session, seen from that session, \
               from a session that recorded nothing, and from no session",
        reset_first: false,
        updates: session_only_edges,
        expect: SESSION_ONLY_EXPECTATIONS,
        both_backends: true,
    },
    AttributionRow {
        what: "the same edge re-announced under a second session",
        reset_first: false,
        updates: re_announced_in_second_session,
        expect: RE_ANNOUNCED_EXPECTATIONS,
        both_backends: true,
    },
];

/// Drive one evaluator through [`ATTRIBUTION_ROWS`].
async fn run_attribution_rows(ev: &dyn crate::evaluator::Evaluator, backend: Backend) {
    for row in ATTRIBUTION_ROWS {
        if row.reset_first {
            ev.reset().await.expect("reset");
        }
        ev.update((row.updates)())
            .await
            .unwrap_or_else(|e| panic!("{backend:?} / {}: update failed: {e}", row.what));
        if !row.both_backends && backend != Backend::Compiled {
            continue;
        }
        for (session, current, expected, why) in row.expect {
            let response = ev
                .query(edge_principal_request_in(session, current))
                .await
                .unwrap_or_else(|e| panic!("{backend:?} / {}: query failed: {e}", row.what));
            let result = response.results.into_iter().next().unwrap();
            assert_eq!(
                result.authorized, *expected,
                "{backend:?} / {}: {why}",
                row.what
            );
        }
    }
}

/// Row 1: three destinations, each with one incoming edge — one
/// asserted by "gateway", one by "intern", one by nobody.
fn seeded_attributed_edges() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("", "src-gateway"),
        session_node("", "src-intern"),
        session_node("", "src-anonymous"),
        session_node("", "dst-gateway"),
        session_node("", "dst-intern"),
        session_node("", "dst-anonymous"),
        attributed_edge("src-gateway", "dst-gateway", Some("gateway")),
        attributed_edge("src-intern", "dst-intern", Some("intern")),
        attributed_edge("src-anonymous", "dst-anonymous", None),
    ]
}

/// Row 2: each of the two attributed edges changes hands.
fn swapped_attributed_edges() -> Vec<crate::engine::GraphUpdate> {
    vec![
        attributed_edge("src-gateway", "dst-gateway", Some("intern")),
        attributed_edge("src-intern", "dst-intern", Some("gateway")),
    ]
}

/// Row 3: the intern's edge is re-recorded with no principal.
fn unattributed_re_record() -> Vec<crate::engine::GraphUpdate> {
    vec![attributed_edge("src-gateway", "dst-gateway", None)]
}

/// Row 4: an edge whose recording client named a low-trust entity.
fn contractor_entity_edge() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("", "src-entity"),
        session_node("", "dst-entity"),
        entity_edge("src-entity", "dst-entity", Some("contractor")),
    ]
}

/// Row 5: the same edge re-recorded naming a different entity, which
/// replaces the stored one.
fn entity_replaced_edge() -> Vec<crate::engine::GraphUpdate> {
    vec![entity_edge("src-entity", "dst-entity", Some("auditor"))]
}

/// Row 6: a second edge recorded with the contractor entity, then
/// re-recorded by a request that omitted the entity. `store.rs` merges
/// an omitted entity with the stored edge's — only `principal` is
/// exempt — so what the store announces for that second recording
/// still carries the contractor, which is why this row's update does.
fn entity_carried_forward_edge() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("", "src-keep"),
        session_node("", "dst-keep"),
        entity_edge("src-keep", "dst-keep", Some("contractor")),
        entity_edge("src-keep", "dst-keep", Some("contractor")),
    ]
}

/// Row 7: an update that carries no entity at all. This is the shim's
/// retraction path, kept pinned because the shim must honour it — but
/// the store does not emit it for a re-record that omits the entity,
/// which is what the row above covers.
fn entity_retracted_edge() -> Vec<crate::engine::GraphUpdate> {
    vec![entity_edge("src-keep", "dst-keep", None)]
}

/// The resync row: everything the store still holds after the resync — the
/// three seeded edges as they now stand, and both sessions' edges.
fn replayed_after_resync() -> Vec<crate::engine::GraphUpdate> {
    let mut updates = seeded_attributed_edges();
    // As row 3 left it: the gateway's edge carries no principal.
    updates.push(attributed_edge("src-gateway", "dst-gateway", None));
    updates.extend(two_session_updates());
    updates
}

/// The metadata row: an edge recorded with a message index, then re-recorded
/// without one. The second recording retracts the EdgeData row.
fn edge_metadata_then_dropped() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("", "src-meta"),
        session_node("", "dst-meta"),
        indexed_edge("src-meta", "dst-meta", Some(2)),
        indexed_edge("src-meta", "dst-meta", None),
    ]
}

/// Two edges recorded inside session `s-zz` and nowhere else: one
/// carries the gateway's principal, the other none. Their
/// destinations demand attestation, so a backend that shows an edge
/// to a session that never recorded it — with no principal, since
/// the principal is scoped — denies where the compiled shim allows.
fn session_only_edges() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("s-zz", "gap-src"),
        session_node("s-zz", "gap-dst"),
        session_node("s-zz", "gap2-src"),
        session_node("s-zz", "gap2-dst"),
        session_edge("s-zz", "gap-src", "gap-dst", Some("gateway")),
        session_edge("s-zz", "gap2-src", "gap2-dst", None),
    ]
}

/// Session `s-yy` records the same (gap-src, gap-dst) pair, naming
/// the intern. The edge now belongs to both sessions, each with its
/// own principal.
fn re_announced_in_second_session() -> Vec<crate::engine::GraphUpdate> {
    vec![
        session_node("s-yy", "gap-src"),
        session_node("s-yy", "gap-dst"),
        session_edge("s-yy", "gap-src", "gap-dst", Some("intern")),
    ]
}

/// What [`session_only_edges`] must answer. The two rows asked from
/// outside `s-zz` are the ones that catch a backend which scopes
/// attribution but not the edges it attributes.
const SESSION_ONLY_EXPECTATIONS: &[(&str, &str, bool, &str)] = &[
    (
        "s-zz",
        "gap-dst",
        true,
        "the recording session sees the edge together with the gateway \
         principal recorded on it",
    ),
    (
        "s-zz",
        "gap2-dst",
        false,
        "in the recording session the edge is there and carries no principal, \
         so the attestation rule denies — the rule is live, not vacuous",
    ),
    (
        "",
        "gap-dst",
        true,
        "a query with no session sees nothing of another session's edge: not \
         the edge, and so not a missing principal either",
    ),
    (
        "",
        "gap2-dst",
        true,
        "same for the unattributed edge — an unsessioned query answers from \
         the global partition, which recorded neither",
    ),
    (
        "s-none",
        "gap2-dst",
        true,
        "a session that recorded nothing has no edges to answer from, so no \
         edge of another session's can look unattested to it",
    ),
];

/// What [`re_announced_in_second_session`] must answer: an edge
/// announced under two sessions belongs to both, and each answers
/// from the principal its own recording carried.
const RE_ANNOUNCED_EXPECTATIONS: &[(&str, &str, bool, &str)] = &[
    (
        "s-yy",
        "gap-dst",
        false,
        "the second session recorded the intern's principal on the same pair \
         of ids, and is denied on it",
    ),
    (
        "s-zz",
        "gap-dst",
        true,
        "the first session keeps the gateway's principal; the second \
         session's record of the same ids does not reach it",
    ),
    (
        "",
        "gap-dst",
        true,
        "neither session's copy of the edge reaches a query with no session",
    ),
];

/// What the sessions of [`two_session_updates`] must answer, and
/// what the three seeded edges must still answer beside them.
const TWO_SESSION_EXPECTATIONS: &[(&str, &str, bool, &str)] = &[
    (
        "s-a",
        "m2",
        true,
        "s-a recorded the gateway's edge; the intern's record of the same ids \
         in s-b must not reach it",
    ),
    (
        "s-b",
        "m2",
        false,
        "s-b recorded the intern's edge and is denied on its own principal",
    ),
    (
        "s-c",
        "m2",
        false,
        "s-c recorded the intern's edge first; s-d's later gateway record must \
         not clear it",
    ),
    (
        "s-d",
        "m2",
        true,
        "s-d recorded the gateway's edge after s-c recorded the intern's",
    ),
];

/// After the resync the same answers must come back — the sessions
/// stay apart, and nothing the reset threw away comes back.
const REPLAY_EXPECTATIONS: &[(&str, &str, bool, &str)] = &[
    (
        "",
        "dst-gateway",
        true,
        "the replayed edge carries no principal",
    ),
    (
        "",
        "dst-intern",
        false,
        "the replay records the intern's edge as the seed did",
    ),
    ("s-a", "m2", true, "s-a is the gateway's session again"),
    ("s-b", "m2", false, "s-b is the intern's session again"),
    ("s-c", "m2", false, "s-c is the intern's session again"),
    ("s-d", "m2", true, "s-d is the gateway's session again"),
];

/// One edge update carrying an entity and no principal.
fn entity_edge(src: &str, dst: &str, entity: Option<&str>) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::EdgeCreated {
        source: src.into(),
        destination: dst.into(),
        message_index: None,
        proximal: None,
        principal: None,
        entity: entity.map(str::to_string),
        session_id: String::new(),
    }
}

/// One edge update carrying instrumentation metadata, or none.
fn indexed_edge(src: &str, dst: &str, message_index: Option<u32>) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::EdgeCreated {
        source: src.into(),
        destination: dst.into(),
        message_index,
        proximal: None,
        principal: None,
        entity: None,
        session_id: String::new(),
    }
}

/// Allow every tool call, except from a node reached by an edge that
/// a low-trust principal asserted, an edge whose recording client
/// named a low-trust entity, an edge that still carries an
/// instrumentation message index, or — for the handful of
/// destinations that demand attestation — an incoming edge with no
/// principal at all.
///
/// That last rule is the shape the reference page recommends for
/// demanding attestation: a join on `Edge` with the attribution
/// negated. It is scoped to `AttestationRequired` destinations so it
/// stays out of the way of the rows above it, which record edges
/// with no principal on purpose. Its point is that a backend must
/// never show a policy an edge without also showing the principal
/// recorded with it — an edge from another session must be invisible
/// entirely, not visible and unattributed.
const EDGE_PRINCIPAL_POLICY: &str = r#"#include "common_policy.dl"

.decl LowTrustPrincipal(p: symbol)
LowTrustPrincipal("intern").

.decl LowTrustEntity(e: symbol)
LowTrustEntity("contractor").

.decl AttestationRequired(id: symbol)
AttestationRequired("gap-dst").
AttestationRequired("gap2-dst").

.decl EdgeHasPrincipal(src: symbol, dst: symbol)
EdgeHasPrincipal(src, dst) :- EdgePrincipal(src, dst, _).

IsAuthorized(idx) :- Actions(idx, _).
DenialReason(idx, "block", "incoming edge asserted by a low-trust principal",
             "re-record the dependency through the gateway") :-
    Actions(idx, _),
    Current(id),
    EdgePrincipal(_, id, p),
    LowTrustPrincipal(p).
DenialReason(idx, "block", "incoming edge names a low-trust entity",
             "re-record the dependency without that entity") :-
    Actions(idx, _),
    Current(id),
    EdgeEntity(_, id, e),
    LowTrustEntity(e).
DenialReason(idx, "block", "incoming edge still carries a message index",
             "re-record the dependency without instrumentation metadata") :-
    Actions(idx, _),
    Current(id),
    EdgeData(_, id, [_, mi]),
    mi >= 0.
DenialReason(idx, "block", "incoming edge carries no principal",
             "re-record the dependency through an authenticated client") :-
    Actions(idx, _),
    Current(id),
    AttestationRequired(id),
    Edge(src, id),
    !EdgeHasPrincipal(src, id).
"#;

/// Row 1 of the shared table, applied to one evaluator.
async fn seed_attributed_edges(ev: &dyn crate::evaluator::Evaluator) {
    ev.update(seeded_attributed_edges())
        .await
        .expect("seed attributed edges");
}

/// One edge update, carrying a principal or none.
fn attributed_edge(src: &str, dst: &str, principal: Option<&str>) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::EdgeCreated {
        source: src.into(),
        destination: dst.into(),
        message_index: None,
        proximal: None,
        principal: principal.map(str::to_string),
        entity: None,
        session_id: String::new(),
    }
}

/// A dependency's attribution is authoritative per record: the store
/// stamps the principal from the recording request and keeps only the
/// latest one, so re-recording an edge under a different principal —
/// or under none — must move the policy's answer with it. Soufflé
/// relations are sets and have no tuple delete, so the compiled shim
/// can only honour that by rebuilding the session's relations; before
/// it did, the superseded tuple stayed and the live stream answered
/// differently from a bootstrap of the same stored graph. Compiled
/// backend. `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn a_re_recorded_edge_replaces_the_attribution_the_policy_sees() {
    use crate::evaluator::manager::EvaluatorProcess;
    use crate::evaluator::types::EvalAuthResponse;
    use crate::evaluator::Evaluator;

    let assets = discover_for_test();
    let build = TempDir::new().unwrap();
    let r = compile_souffle_with_assets(EDGE_PRINCIPAL_POLICY, None, build.path(), &assets)
        .expect("policy must compile");
    let ev = EvaluatorProcess::souffle(r.binary_path, "policy_program".into())
        .expect("evaluator must spawn");

    seed_attributed_edges(&ev).await;

    let first = |resp: EvalAuthResponse| resp.results.into_iter().next().unwrap();
    macro_rules! authorized {
        ($dest:expr) => {
            first(ev.query(edge_principal_request($dest)).await.unwrap()).authorized
        };
    }

    // The intern takes over an edge the gateway had asserted.
    ev.update(vec![attributed_edge(
        "src-gateway",
        "dst-gateway",
        Some("intern"),
    )])
    .await
    .unwrap();
    assert!(
        !authorized!("dst-gateway"),
        "the edge now names the intern, so the gateway's superseded principal \
         must not keep the action authorized"
    );

    // The dependency is re-recorded through the gateway.
    ev.update(vec![attributed_edge(
        "src-intern",
        "dst-intern",
        Some("gateway"),
    )])
    .await
    .unwrap();
    assert!(
        authorized!("dst-intern"),
        "the intern's principal was replaced, so the deny rule no longer matches"
    );

    // An attribution can also go away: a client with no auth-derived
    // identity re-records the edge and the store keeps no principal.
    ev.update(vec![attributed_edge(
        "src-anonymous",
        "dst-anonymous",
        Some("intern"),
    )])
    .await
    .unwrap();
    assert!(
        !authorized!("dst-anonymous"),
        "the edge was just asserted by the intern"
    );
    ev.update(vec![attributed_edge(
        "src-anonymous",
        "dst-anonymous",
        None,
    )])
    .await
    .unwrap();
    assert!(
        authorized!("dst-anonymous"),
        "the re-record carries no principal, so the edge has none and matches \
         neither side"
    );

    // A resync clears the shim's state and replays the store. The
    // replayed edge carries no principal, and no later rebuild may
    // resurrect the one the reset threw away.
    ev.reset().await.unwrap();
    let node = |id: &str| crate::engine::GraphUpdate::NodeCreated {
        id: id.into(),
        content: Some("payload".into()),
        role: Some("user".into()),
        agent: None,
        tools: vec![],
        entity: None,
        principal: None,
        derived_from: None,
        metadata: None,
        session_id: String::new(),
    };
    ev.update(vec![
        node("src-gateway"),
        node("dst-gateway"),
        node("spare"),
        attributed_edge("src-gateway", "dst-gateway", None),
    ])
    .await
    .unwrap();
    assert!(
        authorized!("dst-gateway"),
        "the replayed edge carries no principal at all"
    );
    ev.update(vec![crate::engine::GraphUpdate::NodeDeleted(
        "spare".into(),
    )])
    .await
    .unwrap();
    assert!(
        authorized!("dst-gateway"),
        "an unrelated delete rebuilds the session's relations out of the shim's \
         own maps, which must not resurrect the principal from before the resync"
    );
}

fn edge_principal_request(current: &str) -> crate::evaluator::types::EvalAuthRequest {
    edge_principal_request_in("", current)
}

/// The same request, asked inside one session. The empty string is
/// the global partition, which is what `session_id: None` means on
/// the wire.
fn edge_principal_request_in(
    session: &str,
    current: &str,
) -> crate::evaluator::types::EvalAuthRequest {
    use crate::evaluator::types::{EvalAction, EvalAuthRequest};

    EvalAuthRequest {
        current_node_ids: vec![current.into()],
        actions: vec![EvalAction::ToolCall {
            fn_name: "Bash".into(),
            args: r#"{"command":"ls -la"}"#.into(),
        }],
        entity: None,
        roles: vec![],
        tenant_id: Some("default".into()),
        session_id: Some(session.to_string()),
        principal: Some("tester".into()),
        action_metadata: vec![],
    }
}

/// One session's record of an edge between two message ids, with the
/// principal that session's recording request carried.
fn session_edge(
    session: &str,
    src: &str,
    dst: &str,
    principal: Option<&str>,
) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::EdgeCreated {
        source: src.into(),
        destination: dst.into(),
        message_index: None,
        proximal: None,
        principal: principal.map(str::to_string),
        entity: None,
        session_id: session.to_string(),
    }
}

/// One message node inside a session.
fn session_node(session: &str, id: &str) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::NodeCreated {
        id: id.into(),
        content: Some("payload".into()),
        role: Some("user".into()),
        agent: None,
        tools: vec![],
        entity: None,
        principal: None,
        derived_from: None,
        metadata: None,
        session_id: session.to_string(),
    }
}

/// The four sessions of [`two_sessions_keep_their_own_edge_attribution`],
/// each recording the same (m1, m2) pair: two sessions record the
/// gateway's edge first, two record the intern's first.
///
/// Message ids are unique within a session, not across sessions, so
/// two sessions in one shim routinely name the same pair. Each
/// session must answer from the principal it recorded.
fn two_session_updates() -> Vec<crate::engine::GraphUpdate> {
    let mut updates = Vec::new();
    for session in ["s-a", "s-b", "s-c", "s-d"] {
        updates.push(session_node(session, "m1"));
        updates.push(session_node(session, "m2"));
    }
    // Gateway first, then intern.
    updates.push(session_edge("s-a", "m1", "m2", Some("gateway")));
    updates.push(session_edge("s-b", "m1", "m2", Some("intern")));
    // Intern first, then gateway.
    updates.push(session_edge("s-c", "m1", "m2", Some("intern")));
    updates.push(session_edge("s-d", "m1", "m2", Some("gateway")));
    updates
}

/// What each of those four sessions must answer for `m2`.
const TWO_SESSION_CASES: &[(&str, bool, &str)] = &[
    (
        "s-a",
        true,
        "s-a recorded the gateway's edge, and the intern's later record of the \
         same ids in s-b must not reach it",
    ),
    (
        "s-b",
        false,
        "s-b recorded the intern's edge and must be denied on its own principal",
    ),
    (
        "s-c",
        false,
        "s-c recorded the intern's edge first; the gateway's later record in \
         s-d must not clear it",
    ),
    (
        "s-d",
        true,
        "s-d recorded the gateway's edge after the intern's record in s-c, \
         which belongs to that session alone",
    ),
];

/// Attribution is per session, not per pair of message ids. Two
/// sessions that record the same (m1, m2) edge under different
/// principals each answer from their own, in either order of
/// recording, and again after a resync replays the same stream.
/// Compiled backend. `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn two_sessions_keep_their_own_edge_attribution() {
    use crate::evaluator::manager::EvaluatorProcess;
    use crate::evaluator::types::EvalAuthResponse;
    use crate::evaluator::Evaluator;

    let assets = discover_for_test();
    let build = TempDir::new().unwrap();
    let r = compile_souffle_with_assets(EDGE_PRINCIPAL_POLICY, None, build.path(), &assets)
        .expect("policy must compile");
    let ev = EvaluatorProcess::souffle(r.binary_path, "policy_program".into())
        .expect("evaluator must spawn");

    let first = |resp: EvalAuthResponse| resp.results.into_iter().next().unwrap();
    ev.update(two_session_updates()).await.unwrap();
    for (session, expected, why) in TWO_SESSION_CASES {
        let result = first(
            ev.query(edge_principal_request_in(session, "m2"))
                .await
                .unwrap(),
        );
        assert_eq!(result.authorized, *expected, "{why}");
    }

    // A resync throws the shim's state away and the store replays it.
    // The sessions must come back apart, not merged.
    ev.reset().await.unwrap();
    ev.update(two_session_updates()).await.unwrap();
    for (session, expected, why) in TWO_SESSION_CASES {
        let result = first(
            ev.query(edge_principal_request_in(session, "m2"))
                .await
                .unwrap(),
        );
        assert_eq!(
            result.authorized, *expected,
            "after a resync and re-bootstrap: {why}"
        );
    }
}

/// A policy that reads `Event.metadata` through `MessageMetadata`:
/// publishing is refused when a message the call depends on carries a
/// marker inside its recorded metadata.
///
/// The path spells the adapters' canonical JSON, whose object keys carry
/// a type prefix (`s:` for a string key). A path is split on `.` and `[`
/// only, so the colon inside a key needs no escaping.
///
/// Two more tools read the same relation differently, so a test can say
/// what a row IS rather than only that the marker is missing: `probe` is
/// refused by ANY row for a dependency, whatever it holds, and `audit`
/// only by the unmarked value. Without them, an implementation that
/// emitted an empty row, or that dropped a value it was asked to store,
/// would answer every question here the same way a correct one does.
const MESSAGE_METADATA_POLICY: &str = r#"#include "common_policy.dl"

CurrentDependsPolicyRelevant() :- IsTool(_, "publish").
CurrentDependsPolicyRelevant() :- IsTool(_, "probe").
CurrentDependsPolicyRelevant() :- IsTool(_, "audit").

.decl MarkedDependency(id: symbol)
MarkedDependency(id) :-
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".

.decl DependencyWithMetadata(id: symbol)
DependencyWithMetadata(id) :-
    CurrentDepends(id),
    MessageMetadata(id, _).

.decl UnmarkedDependency(id: symbol)
UnmarkedDependency(id) :-
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "ordinary".

IsAuthorized(idx) :- Actions(idx, _).
DenialReason(idx, "block", "a message this call depends on is marked in its metadata",
             "record the message without the marker") :-
    Actions(idx, a),
    IsTool(a, "publish"),
    MarkedDependency(_).
DenialReason(idx, "block", "a message this call depends on has a metadata row",
             "record the message without metadata") :-
    Actions(idx, a),
    IsTool(a, "probe"),
    DependencyWithMetadata(_).
DenialReason(idx, "block", "a message this call depends on is marked ordinary",
             "record the message with other metadata") :-
    Actions(idx, a),
    IsTool(a, "audit"),
    UnmarkedDependency(_).
"#;

/// One node carrying (or not carrying) the adapter's record of its
/// message, in the global partition.
fn metadata_node(id: &str, metadata: Option<&str>) -> crate::engine::GraphUpdate {
    metadata_node_in("", id, metadata)
}

/// The same node, recorded under a named session.
fn metadata_node_in(session: &str, id: &str, metadata: Option<&str>) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::NodeCreated {
        id: id.into(),
        content: Some("payload".into()),
        role: Some("user".into()),
        agent: None,
        tools: vec![],
        entity: None,
        principal: None,
        derived_from: None,
        metadata: metadata.map(|m| m.to_string()),
        session_id: session.into(),
    }
}

/// A dependency edge, in the global partition or a named session.
fn metadata_edge(session: &str, src: &str, dst: &str) -> crate::engine::GraphUpdate {
    crate::engine::GraphUpdate::EdgeCreated {
        source: src.into(),
        destination: dst.into(),
        message_index: None,
        proximal: None,
        principal: None,
        entity: None,
        session_id: session.into(),
    }
}

/// The adapters' shape: every object key carries its type prefix.
const MARKED_METADATA: &str =
    r#"{"s:additional_kwargs":{"s:marker":"exfil"},"s:content":"payload","s:type":"ai"}"#;
const UNMARKED_METADATA: &str =
    r#"{"s:additional_kwargs":{"s:marker":"ordinary"},"s:content":"payload","s:type":"ai"}"#;

/// `Event.metadata` reaches a compiled policy as `MessageMetadata`: a
/// rule joining on it refuses a call whose dependency carries a marker,
/// leaves a dependency recorded without metadata alone, and keeps its
/// answer after the marked message is re-recorded by an update that says
/// nothing about metadata.
///
/// It also covers what happens to a row over a node's life: an explicit
/// value replaces the old one and is still the value the policy reads,
/// deleting the node takes the row with it — in every session holding
/// one for that id, not just the session the node was last recorded
/// under — a reset empties the store, and a second session that
/// re-records the same id inherits nothing.
/// `--ignored`, like every test here that compiles a policy: it needs
/// the Soufflé toolchain.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn message_metadata_reaches_a_compiled_policy() {
    use crate::evaluator::manager::EvaluatorProcess;
    use crate::evaluator::types::{EvalAction, EvalAuthRequest, EvalAuthResponse};
    use crate::evaluator::Evaluator;

    let assets = discover_for_test();
    let build = TempDir::new().unwrap();
    let r = compile_souffle_with_assets(MESSAGE_METADATA_POLICY, None, build.path(), &assets)
        .expect("policy must compile");
    let ev = EvaluatorProcess::souffle(r.binary_path, "policy_program".into())
        .expect("evaluator must spawn");

    let edge = |src: &str, dst: &str| metadata_edge("", src, dst);
    ev.update(vec![
        metadata_node("src-marked", Some(MARKED_METADATA)),
        metadata_node("src-plain", Some(UNMARKED_METADATA)),
        metadata_node("src-none", None),
        metadata_node("dst-marked", None),
        metadata_node("dst-plain", None),
        metadata_node("dst-none", None),
        edge("src-marked", "dst-marked"),
        edge("src-plain", "dst-plain"),
        edge("src-none", "dst-none"),
    ])
    .await
    .expect("seed messages and their dependencies");

    let call = |tool: &str, current: &str, session: Option<&str>| EvalAuthRequest {
        current_node_ids: vec![current.into()],
        actions: vec![EvalAction::ToolCall {
            fn_name: tool.into(),
            args: r#"{"destination":"external"}"#.into(),
        }],
        entity: None,
        roles: vec![],
        tenant_id: Some("default".into()),
        session_id: session.map(str::to_string),
        principal: Some("tester".into()),
        action_metadata: vec![],
    };
    let publish = |current: &str| call("publish", current, None);
    // `probe` is refused by ANY row for the dependency, `audit` only by
    // the unmarked value, so "no row" and "this value" are each testable
    // rather than inferred from the marker rule staying quiet.
    let probe = |current: &str| call("probe", current, None);
    let audit = |current: &str| call("audit", current, None);
    let first = |resp: EvalAuthResponse| resp.results.into_iter().next().unwrap();

    let marked = first(ev.query(publish("dst-marked")).await.unwrap());
    assert!(
        !marked.authorized,
        "the marker in the dependency's metadata must refuse the call; an empty \
         MessageMetadata relation would allow it, which is how a rule that \
         silently matches nothing reads. reasons={:?}",
        marked.denial_reasons,
    );
    assert!(
        first(ev.query(publish("dst-plain")).await.unwrap()).authorized,
        "metadata without the marker does not refuse"
    );
    assert!(
        first(ev.query(publish("dst-none")).await.unwrap()).authorized,
        "a message recorded without metadata has no row, so the rule cannot match it"
    );
    assert!(
        !first(ev.query(probe("dst-plain")).await.unwrap()).authorized,
        "the any-row rule must refuse a dependency that does have a row"
    );
    assert!(
        first(ev.query(probe("dst-none")).await.unwrap()).authorized,
        "a message recorded without metadata must have NO row at all, not an empty one"
    );

    // An update that carries no metadata says nothing about it. Erasing
    // on absence would empty every rule that joins on MessageMetadata
    // the first time a message is re-recorded.
    ev.update(vec![metadata_node("src-marked", None)])
        .await
        .expect("re-record the marked message without metadata");
    assert!(
        !first(ev.query(publish("dst-marked")).await.unwrap()).authorized,
        "a re-record that carries no metadata must keep the existing record"
    );

    // An explicit value does replace it — and the new value is what the
    // policy reads, not nothing.
    ev.update(vec![metadata_node("src-marked", Some(UNMARKED_METADATA))])
        .await
        .expect("re-record the marked message with metadata of its own");
    assert!(
        first(ev.query(publish("dst-marked")).await.unwrap()).authorized,
        "a re-record that carries metadata replaces what was there"
    );
    assert!(
        !first(ev.query(audit("dst-marked")).await.unwrap()).authorized,
        "the replacement value must be the row the policy reads; discarding it \
         would also stop the marker rule from refusing"
    );

    // Deleting the node takes its row with it: a message recorded again
    // under the same id, carrying no metadata, must not find the old
    // value waiting for it.
    ev.update(vec![crate::engine::GraphUpdate::NodeDeleted(
        "src-marked".into(),
    )])
    .await
    .expect("delete the marked message");
    ev.update(vec![metadata_node("src-marked", None)])
        .await
        .expect("record the id again, with no metadata");
    assert!(
        first(ev.query(probe("dst-marked")).await.unwrap()).authorized,
        "a deleted message leaves no metadata row behind"
    );

    // A reset empties the store, so a message re-recorded afterwards
    // without metadata has none — the value it carried before is gone.
    ev.reset().await.expect("reset the graph");
    ev.update(vec![
        metadata_node("src-plain", None),
        metadata_node("dst-plain", None),
        edge("src-plain", "dst-plain"),
    ])
    .await
    .expect("re-seed after the reset");
    // The second recording of the id is the one that rebuilds the
    // relations from what the shim holds, which is where a value the
    // reset failed to drop would come back.
    ev.update(vec![metadata_node("src-plain", None)])
        .await
        .expect("record the re-seeded message once more");
    assert!(
        first(ev.query(probe("dst-plain")).await.unwrap()).authorized,
        "a reset must drop the stored metadata, not leave it to reattach"
    );

    // Two sessions, one message id. The session that recorded a value
    // keeps it; a session that re-records the id without one inherits
    // nothing, or a scoped recording could borrow another session's
    // metadata (or hide behind its absence).
    ev.update(vec![
        metadata_node_in("a", "shared", Some(MARKED_METADATA)),
        metadata_node_in("a", "sink-a", None),
        metadata_edge("a", "shared", "sink-a"),
    ])
    .await
    .expect("record the shared message in session a");
    assert!(
        !first(
            ev.query(call("publish", "sink-a", Some("a")))
                .await
                .unwrap()
        )
        .authorized,
        "the session that recorded the metadata reads it"
    );
    ev.update(vec![
        metadata_node_in("b", "shared", None),
        metadata_node_in("b", "sink-b", None),
        metadata_edge("b", "shared", "sink-b"),
    ])
    .await
    .expect("re-record the same id in session b, with no metadata");
    assert!(
        first(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()).authorized,
        "session b recorded no metadata, so it must see no row for the id"
    );

    // Session b now records a value of its own for the same id. It is
    // b's value, and only b's: a's value must survive it, including
    // after the id comes back to a on an update that says nothing about
    // metadata. Holding one value per id, whichever session recorded it
    // last, loses a's marker here.
    ev.update(vec![metadata_node_in(
        "b",
        "shared",
        Some(UNMARKED_METADATA),
    )])
    .await
    .expect("record the shared id in session b, with metadata of its own");
    assert!(
        first(
            ev.query(call("publish", "sink-b", Some("b")))
                .await
                .unwrap()
        )
        .authorized,
        "session b reads its own value, which carries no marker"
    );
    assert!(
        !first(ev.query(call("audit", "sink-b", Some("b"))).await.unwrap()).authorized,
        "and it is b's value the policy reads, not a's"
    );
    ev.update(vec![metadata_node_in("a", "shared", None)])
        .await
        .expect("bring the id back to session a, with no metadata");
    assert!(
        !first(
            ev.query(call("publish", "sink-a", Some("a")))
                .await
                .unwrap()
        )
        .authorized,
        "session a keeps the value it recorded: another session recording \
         under the same id must not take it away"
    );

    // A deletion names an id and nothing else, so it takes the node out
    // of every session at once — and every session's value for the id
    // has to go with it, not just the one the node was last recorded
    // under. Both sessions hold a value for `shared` here: a's marker
    // and b's unmarked value. Each half below re-records the id twice,
    // because the compiled shim rebuilds a session's relations on a
    // re-record, which is where a retained value would reappear.
    ev.update(vec![crate::engine::GraphUpdate::NodeDeleted(
        "shared".into(),
    )])
    .await
    .expect("delete the shared message");
    for _ in 0..2 {
        ev.update(vec![metadata_node_in("a", "shared", None)])
            .await
            .expect("record the id again in session a, with no metadata");
    }
    assert!(
        first(ev.query(call("probe", "sink-a", Some("a"))).await.unwrap()).authorized,
        "the delete must take session a's value for the id with it"
    );
    // And the row is absent, not unreachable: the same query refuses
    // again as soon as a value comes back.
    for _ in 0..2 {
        ev.update(vec![metadata_node_in(
            "a",
            "shared",
            Some(UNMARKED_METADATA),
        )])
        .await
        .expect("give the id a value in session a again");
    }
    assert!(
        !first(ev.query(call("probe", "sink-a", Some("a"))).await.unwrap()).authorized,
        "session a reaches the id's row when there is one to reach"
    );
    for _ in 0..2 {
        ev.update(vec![metadata_node_in("b", "shared", None)])
            .await
            .expect("record the id again in session b, with no metadata");
    }
    assert!(
        first(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()).authorized,
        "and session b's value too: a delete is not scoped to the session \
         the node happened to be recorded under last"
    );
    for _ in 0..2 {
        ev.update(vec![metadata_node_in(
            "b",
            "shared",
            Some(UNMARKED_METADATA),
        )])
        .await
        .expect("give the id a value in session b again");
    }
    assert!(
        !first(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()).authorized,
        "session b reaches the id's row when there is one to reach"
    );
}

/// The same relation on the interpreted backend, which reaches Soufflé
/// through fact files rather than a compiled program, and the same
/// lifecycle: omission keeps the value, an explicit value replaces it,
/// deletion and reset remove it — deletion in every session holding a
/// value for the id — and a second session inherits nothing.
/// Same skip rule as `edge_attribution_reaches_the_interpreted_backend`:
/// without the adapter built the test fails, and
/// `SASY_TEST_SKIP_INTERPRETED=1` skips it. `--ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn message_metadata_reaches_the_interpreted_backend() {
    use crate::evaluator::types::{EvalAction, EvalAuthRequest};

    let work = TempDir::new().unwrap();
    let policy_path = work.path().join("message_metadata.dl");
    std::fs::write(&policy_path, MESSAGE_METADATA_POLICY).unwrap();

    let skip = std::env::var("SASY_TEST_SKIP_INTERPRETED").is_ok_and(|v| v == "1");
    let built = match crate::evaluator::factory::build_souffle_factory_with_artifacts(
        sasy_common::Backend::SouffleInterpreted,
        work.path(),
        &policy_path,
        None,
        Some(&interpreted_assets_for_test()),
    ) {
        Ok(built) => built,
        Err(error) if skip => {
            eprintln!("skipping on SASY_TEST_SKIP_INTERPRETED=1: {error}");
            return;
        }
        Err(error) => panic!("the interpreted backend must be available: {error}"),
    };
    let ev = match (built.factory)() {
        Ok(ev) => ev,
        Err(error) if skip => {
            eprintln!("skipping on SASY_TEST_SKIP_INTERPRETED=1: {error}");
            return;
        }
        Err(error) => panic!("the interpreted backend must be available: {error}"),
    };

    ev.update(vec![
        metadata_node("src-marked", Some(MARKED_METADATA)),
        metadata_node("src-plain", Some(UNMARKED_METADATA)),
        metadata_node("dst-marked", None),
        metadata_node("dst-plain", None),
        metadata_node("src-none", None),
        metadata_node("dst-none", None),
        metadata_edge("", "src-marked", "dst-marked"),
        metadata_edge("", "src-plain", "dst-plain"),
        metadata_edge("", "src-none", "dst-none"),
    ])
    .await
    .expect("seed messages and their dependencies");

    let call = |tool: &str, current: &str, session: Option<&str>| EvalAuthRequest {
        current_node_ids: vec![current.into()],
        actions: vec![EvalAction::ToolCall {
            fn_name: tool.into(),
            args: r#"{"destination":"external"}"#.into(),
        }],
        entity: None,
        roles: vec![],
        tenant_id: Some("default".into()),
        session_id: session.map(str::to_string),
        principal: Some("tester".into()),
        action_metadata: vec![],
    };
    let publish = |current: &str| call("publish", current, None);
    let probe = |current: &str| call("probe", current, None);
    let audit = |current: &str| call("audit", current, None);
    let authorized = |resp: crate::evaluator::types::EvalAuthResponse| {
        resp.results.into_iter().next().unwrap().authorized
    };

    assert!(
        !authorized(ev.query(publish("dst-marked")).await.unwrap()),
        "the interpreted backend must write the metadata fact too; an empty \
         MessageMetadata.facts would allow this"
    );
    assert!(
        authorized(ev.query(publish("dst-plain")).await.unwrap()),
        "metadata without the marker does not refuse"
    );
    assert!(
        !authorized(ev.query(probe("dst-plain")).await.unwrap()),
        "the any-row rule must refuse a dependency that does have a row"
    );
    assert!(
        authorized(ev.query(probe("dst-none")).await.unwrap()),
        "a message recorded without metadata must have NO row at all, not an empty one"
    );

    // Omission says nothing about the value; an explicit value replaces
    // it, and the replacement is what the policy reads.
    ev.update(vec![metadata_node("src-marked", None)])
        .await
        .expect("re-record the marked message without metadata");
    assert!(
        !authorized(ev.query(publish("dst-marked")).await.unwrap()),
        "a re-record that carries no metadata must keep the existing record"
    );
    ev.update(vec![metadata_node("src-marked", Some(UNMARKED_METADATA))])
        .await
        .expect("re-record the marked message with metadata of its own");
    assert!(
        authorized(ev.query(publish("dst-marked")).await.unwrap()),
        "a re-record that carries metadata replaces what was there"
    );
    assert!(
        !authorized(ev.query(audit("dst-marked")).await.unwrap()),
        "the replacement value must be the row the policy reads"
    );

    // Deletion and reset both take the row away — a re-record of the same
    // id afterwards, with no metadata, must not bring the old value back.
    ev.update(vec![crate::engine::GraphUpdate::NodeDeleted(
        "src-marked".into(),
    )])
    .await
    .expect("delete the marked message");
    ev.update(vec![metadata_node("src-marked", None)])
        .await
        .expect("record the id again, with no metadata");
    assert!(
        authorized(ev.query(probe("dst-marked")).await.unwrap()),
        "a deleted message leaves no metadata row behind"
    );
    ev.reset().await.expect("reset the graph");
    ev.update(vec![
        metadata_node("src-plain", None),
        metadata_node("dst-plain", None),
        metadata_edge("", "src-plain", "dst-plain"),
    ])
    .await
    .expect("re-seed after the reset");
    assert!(
        authorized(ev.query(probe("dst-plain")).await.unwrap()),
        "a reset must drop the stored metadata, not leave it to reattach"
    );

    // One id under two sessions: the recording session keeps the value,
    // the other inherits nothing.
    ev.update(vec![
        metadata_node_in("a", "shared", Some(MARKED_METADATA)),
        metadata_node_in("a", "sink-a", None),
        metadata_edge("a", "shared", "sink-a"),
    ])
    .await
    .expect("record the shared message in session a");
    assert!(
        !authorized(
            ev.query(call("publish", "sink-a", Some("a")))
                .await
                .unwrap()
        ),
        "the session that recorded the metadata reads it"
    );
    ev.update(vec![
        metadata_node_in("b", "shared", None),
        metadata_node_in("b", "sink-b", None),
        metadata_edge("b", "shared", "sink-b"),
    ])
    .await
    .expect("re-record the same id in session b, with no metadata");
    assert!(
        authorized(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()),
        "session b recorded no metadata, so it must see no row for the id"
    );

    // b's own value for the id is b's alone: a keeps the marker it
    // recorded, and still reads it once the id returns to a on an update
    // that says nothing about metadata.
    ev.update(vec![metadata_node_in(
        "b",
        "shared",
        Some(UNMARKED_METADATA),
    )])
    .await
    .expect("record the shared id in session b, with metadata of its own");
    assert!(
        authorized(
            ev.query(call("publish", "sink-b", Some("b")))
                .await
                .unwrap()
        ),
        "session b reads its own value, which carries no marker"
    );
    assert!(
        !authorized(ev.query(call("audit", "sink-b", Some("b"))).await.unwrap()),
        "and it is b's value the policy reads, not a's"
    );
    ev.update(vec![metadata_node_in("a", "shared", None)])
        .await
        .expect("bring the id back to session a, with no metadata");
    assert!(
        !authorized(
            ev.query(call("publish", "sink-a", Some("a")))
                .await
                .unwrap()
        ),
        "session a keeps the value it recorded: another session recording \
         under the same id must not take it away"
    );

    // The same delete across sessions: both sessions hold a value for
    // `shared`, and the delete takes both, not only the one the node was
    // last recorded under. Each half checks the row is absent and then
    // that the query reaches one when it exists.
    ev.update(vec![crate::engine::GraphUpdate::NodeDeleted(
        "shared".into(),
    )])
    .await
    .expect("delete the shared message");
    ev.update(vec![metadata_node_in("a", "shared", None)])
        .await
        .expect("record the id again in session a, with no metadata");
    assert!(
        authorized(ev.query(call("probe", "sink-a", Some("a"))).await.unwrap()),
        "the delete must take session a's value for the id with it"
    );
    ev.update(vec![metadata_node_in(
        "a",
        "shared",
        Some(UNMARKED_METADATA),
    )])
    .await
    .expect("give the id a value in session a again");
    assert!(
        !authorized(ev.query(call("probe", "sink-a", Some("a"))).await.unwrap()),
        "session a reaches the id's row when there is one to reach"
    );
    ev.update(vec![metadata_node_in("b", "shared", None)])
        .await
        .expect("record the id again in session b, with no metadata");
    assert!(
        authorized(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()),
        "and session b's value too: a delete is not scoped to the session \
         the node happened to be recorded under last"
    );
    ev.update(vec![metadata_node_in(
        "b",
        "shared",
        Some(UNMARKED_METADATA),
    )])
    .await
    .expect("give the id a value in session b again");
    assert!(
        !authorized(ev.query(call("probe", "sink-b", Some("b"))).await.unwrap()),
        "session b reaches the id's row when there is one to reach"
    );
}

#[path = "compiler_inference_tests.rs"]
mod inference_tests;
