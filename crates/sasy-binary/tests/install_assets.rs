//! The downloaded-binary story, end to end: a `sasy` executable standing on
//! its own outside the source tree installs its compile-chain assets and
//! compiles a policy.
//!
//! Ignored by default because it needs the real toolchain — `souffle`, a C++
//! compiler and `python3` on PATH — and pays a full compile (~10 s). Run it
//! with:
//!
//! ```text
//! cargo test -p sasy-binary --test install_assets -- --ignored --nocapture
//! ```

#![cfg(feature = "compiler")]

use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

/// The working directory is process-wide, so the tests that move it take this
/// lock rather than racing each other.
static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The trivial policy: enough for the pipeline to run end to end, nothing more.
const POLICY: &str = ".decl Hello(x: symbol)\nHello(\"world\").\n";

/// `install-assets` writes where it is told and touches nothing else.
///
/// The directory named on the command line is the operator's: a relative name
/// resolves against their working directory (whose parent path is the empty
/// string), and the directory it lands in may be one they do not own — `/tmp`,
/// or a system prefix. So the command creates and hardens the target itself
/// and leaves everything above it as it found it. No toolchain needed, so this
/// one is not ignored.
#[test]
fn install_assets_writes_the_named_directory_and_nothing_above_it() {
    use std::os::unix::fs::PermissionsExt;

    let home = TempDir::new().unwrap();
    let home = home.path();
    std::fs::set_permissions(home, std::fs::Permissions::from_mode(0o755)).unwrap();

    // A relative name, resolved against the working directory.
    let output = Command::new(env!("CARGO_BIN_EXE_sasy"))
        .arg("install-assets")
        .arg("chain")
        .current_dir(home)
        .output()
        .expect("run sasy install-assets");
    assert!(
        output.status.success(),
        "install-assets failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(home.join("chain").join("sugar.py").is_file());

    let mode = std::fs::metadata(home).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o755,
        "the operator's working directory must not be re-moded"
    );
}

#[test]
#[ignore]
fn a_binary_outside_the_repo_installs_its_assets_and_compiles_a_policy() {
    let home = TempDir::new().unwrap();
    let home = home.path();

    // The executable, copied away from the target directory it was built in, so
    // nothing it finds can be reached by walking up from where it stands.
    let exe = home.join("sasy");
    std::fs::copy(env!("CARGO_BIN_EXE_sasy"), &exe).expect("copy the built binary");

    let installed = home.join("assets");
    let output = Command::new(&exe)
        .arg("install-assets")
        .arg(&installed)
        .current_dir(home)
        .output()
        .expect("run sasy install-assets");
    assert!(
        output.status.success(),
        "install-assets failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = String::from_utf8_lossy(&output.stdout);
    for name in [
        "sugar.py",
        "evaluator_shim.cpp",
        "evaluator_protocol.h",
        "json_string_codec.h",
        "functors_common.cpp",
        "common_policy.dl",
    ] {
        assert!(installed.join(name).is_file(), "{name} was not installed");
        assert!(printed.contains(name), "{name} was not printed: {printed}");
    }

    // Compile from a working directory that is not the repo, with the installed
    // directory named — the operator's escape hatch.
    let policy = home.join("trivial.dl");
    std::fs::write(&policy, POLICY).unwrap();
    compile_in(home, &policy, Some(&installed));

    // And again with nothing named, so the compile has to materialize the set
    // the binary carries. This is the case a downloaded binary is in.
    compile_in(home, &policy, None);
}

/// Compile `policy` from `cwd` as the working directory, optionally pointing
/// `SASY_SOUFFLE_ASSETS` at `assets`.
///
/// The compile runs in this process, through the same entry point the server
/// uses. Moving the process's working directory out of the repo is the point:
/// the compile must not depend on where it is run from. The move is
/// process-wide, so every test that makes it holds [`CWD_LOCK`].
fn compile_in(cwd: &Path, policy: &Path, assets: Option<&Path>) {
    // Held across the environment writes too: both variables are process-wide.
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let build = cwd.join(match assets {
        Some(_) => "build-installed",
        None => "build-embedded",
    });
    std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", cwd.join("cache"));
    match assets {
        Some(dir) => std::env::set_var("SASY_SOUFFLE_ASSETS", dir),
        None => std::env::remove_var("SASY_SOUFFLE_ASSETS"),
    }
    let previous_cwd = std::env::current_dir().expect("a working directory");
    std::env::set_current_dir(cwd).expect("move out of the repo");
    let compiled = sasy_policy::compiler::compile_policy_file(policy, &build, None);
    std::env::set_current_dir(previous_cwd).expect("come back");

    let compiled = compiled.unwrap_or_else(|error| panic!("compile failed ({assets:?}): {error}"));
    assert!(
        compiled.binary_path.is_file(),
        "no evaluator was produced in {}",
        build.display()
    );
}

/// A whole server boots from a working directory it cannot write to, told
/// nothing but an absolute data directory and a policy.
///
/// This is the systemd unit that runs at `/`, or any read-only container
/// rootfs. Nothing on the startup path may fall back to a relative path: with
/// no `--credentials-db`, the sqlite backend puts its database under the data
/// directory, and the server reaches the point where it is listening.
///
/// Ignored by default: it needs the real toolchain and pays a policy compile.
#[test]
#[ignore]
fn the_server_boots_from_a_read_only_working_directory() {
    use std::io::Read;
    use std::os::unix::fs::PermissionsExt;

    let home = TempDir::new().unwrap();
    let home = std::fs::canonicalize(home.path()).unwrap();
    let read_only_cwd = home.join("cwd");
    std::fs::create_dir(&read_only_cwd).unwrap();
    let data_dir = home.join("data");
    let policy = home.join("trivial.dl");
    std::fs::write(&policy, POLICY).unwrap();
    let log = home.join("serve.log");

    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o500)).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_sasy"))
        .arg("serve")
        .arg("--policy")
        .arg(&policy)
        // A port of this test's own, so a server already running on the
        // default one is not what it measures.
        .args(["--addr", "127.0.0.1:50251"])
        .arg("--data-dir")
        .arg(&data_dir)
        .current_dir(&read_only_cwd)
        .env("SASY_ALLOW_NO_AUTH", "1")
        .env("SASY_SOUFFLE_BUILD_CACHE_DIR", home.join("cache"))
        .env_remove("SASY_SOUFFLE_ASSETS")
        .stdout(std::fs::File::create(&log).unwrap())
        .stderr(std::fs::File::options().append(true).open(&log).unwrap())
        .spawn()
        .expect("start sasy serve");

    // Poll rather than sleep: the compile dominates and its length varies.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    let mut listening = false;
    while std::time::Instant::now() < deadline {
        let mut printed = String::new();
        std::fs::File::open(&log)
            .unwrap()
            .read_to_string(&mut printed)
            .unwrap();
        if printed.contains("starting gRPC server") {
            listening = true;
            break;
        }
        if let Some(status) = child.try_wait().expect("poll the server") {
            panic!("the server exited ({status}):\n{printed}");
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    let _ = child.kill();
    let _ = child.wait();
    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o700)).unwrap();

    let printed = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(listening, "the server never started listening:\n{printed}");
    assert!(
        data_dir.join("credentials.db").is_file(),
        "the credential database must land under the data dir:\n{printed}"
    );
    assert_eq!(
        std::fs::read_dir(&read_only_cwd).unwrap().count(),
        0,
        "nothing may be written beside the working directory"
    );
}

/// The bootstrap compile a server does at startup, from a working directory it
/// cannot write to and with an absolute data directory: exactly the systemd
/// unit that runs at `/`. Everything the compile produces lands under the data
/// directory, and nothing is asked of the working one.
#[test]
#[ignore]
fn the_bootstrap_compile_survives_a_read_only_working_directory() {
    use std::os::unix::fs::PermissionsExt;

    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let home = TempDir::new().unwrap();
    let read_only_cwd = home.path().join("cwd");
    std::fs::create_dir(&read_only_cwd).unwrap();
    let data_dir = std::fs::canonicalize(home.path()).unwrap().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let policy = home.path().join("trivial.dl");
    std::fs::write(&policy, POLICY).unwrap();

    let build =
        sasy_policy::compiler::bootstrap_build_dir(&data_dir).expect("bootstrap build directory");
    assert!(build.starts_with(&data_dir), "{}", build.display());

    std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", data_dir.join("cache"));
    std::env::remove_var("SASY_SOUFFLE_ASSETS");
    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o500)).unwrap();

    let previous_cwd = std::env::current_dir().expect("a working directory");
    std::env::set_current_dir(&read_only_cwd).expect("move to the read-only directory");
    let compiled = sasy_policy::compiler::compile_policy_file(&policy, &build, None);
    std::env::set_current_dir(previous_cwd).expect("come back");
    std::fs::set_permissions(&read_only_cwd, std::fs::Permissions::from_mode(0o700)).unwrap();

    let compiled = compiled.unwrap_or_else(|error| panic!("bootstrap compile failed: {error}"));
    assert!(compiled.binary_path.starts_with(&data_dir));
    assert!(compiled.binary_path.is_file());
    assert_eq!(
        std::fs::read_dir(&read_only_cwd).unwrap().count(),
        0,
        "nothing may be written beside the working directory"
    );
}
