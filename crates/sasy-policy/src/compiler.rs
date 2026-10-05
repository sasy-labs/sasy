//! Policy compilation.
//!
//! Compiles Soufflé policies at startup or runtime, producing
//! an evaluator binary that can be spawned via IPC.
//!
//! The three stages (python3 sugar.py / souffle -g / g++) run on
//! caller-supplied policy text and C++ functor source. When
//! bubblewrap is available we run each stage inside a stripped
//! sandbox so the compiler toolchain can't read credentials / config
//! via `#include` tricks. The sandbox helper lives in
//! `crate::sandbox` and is shared with `evaluator::ipc` (which uses
//! it to confine `__attribute__((constructor))` code in the produced
//! evaluator binary). Bubblewrap isn't present on macOS / local dev,
//! so we fall back to direct subprocess calls.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use sha2::{Digest, Sha256};
use tracing::info;

use crate::sandbox::sandboxed_command;
use crate::souffle_cache;

/// Discovered build asset paths.
#[derive(Debug, Clone)]
pub struct SouffleAssets {
    pub include_dir: PathBuf,
    pub sugar_py: PathBuf,
    pub evaluator_shim: PathBuf,
    pub evaluator_protocol: PathBuf,
    pub json_string_codec: PathBuf,
    pub functors_common: PathBuf,
    pub common_policy: Option<PathBuf>,
    /// A directory that holds the asset set and nothing else, safe to bind
    /// into the compile sandbox whole.
    ///
    /// Only two paths set it: the set this binary materializes (a directory it
    /// creates and owns) and the `SASY_SOUFFLE_ASSETS` directory an operator
    /// installs with `sasy install-assets`. Anything that names asset files
    /// wherever they happen to lie — the assured runtime's signed manifest, a
    /// test fixture — leaves it `None`, and the sandbox binds the files one by
    /// one rather than hand the compiler whatever else shares a directory with
    /// them.
    pub exclusive_dir: Option<PathBuf>,
    /// Soufflé RamDomain word size (32 or 64).
    pub word_size: u8,
}

/// The read-only sandbox binds the g++ stage needs: the Soufflé headers, and
/// the compile-chain assets.
///
/// A directory is bound whole only when it is known to hold nothing but the
/// asset set — [`SouffleAssets::exclusive_dir`], set by the materialized set
/// and by an operator's `SASY_SOUFFLE_ASSETS`. Otherwise each asset file is
/// bound on its own, whatever their layout: where the paths come from is what
/// says a directory is safe to hand over, never where they happen to lie. The
/// assured-runtime path takes every path from a signed manifest, and its
/// files usually sit in the source tree's `souffle/` directory next to
/// functor and shim sources the compiler has no business reading.
fn compile_ro_binds(assets: &SouffleAssets) -> Vec<&Path> {
    let files = [
        assets.evaluator_shim.as_path(),
        assets.evaluator_protocol.as_path(),
        assets.json_string_codec.as_path(),
        assets.functors_common.as_path(),
    ];
    let mut binds: Vec<&Path> = vec![assets.include_dir.as_path()];
    if let Some(dir) = assets.exclusive_dir.as_deref() {
        binds.push(dir);
        return binds;
    }
    for asset in files {
        if !binds.contains(&asset) {
            binds.push(asset);
        }
    }
    binds
}

/// Maximum duration of one compiler invocation. Bounds hung toolchain processes
/// while allowing cold compilation on slower machines.
fn compile_timeout() -> Duration {
    std::env::var("SASY_COMPILE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(300))
}

/// Run `cmd` to completion like `Command::output`, but kill it if it exceeds
/// [`compile_timeout`].
///
/// `what` names the step in the timeout error. The pipes are drained on their
/// own threads: polling `try_wait` while a child fills a pipe it cannot flush
/// would deadlock the child and then report it as a timeout, blaming the clock
/// for a plumbing mistake.
pub(crate) fn output_bounded(mut cmd: Command, what: &str) -> std::io::Result<Output> {
    let timeout = compile_timeout();
    // stdin explicitly null. `Command::output()` did that for us; `spawn()`
    // INHERITS it, which would hand the toolchain the server's own stdin on
    // fd 0. Inside bwrap `--dev /dev` keeps `/dev/stdin` usable, so functor
    // source containing `#include "/dev/stdin"` — exactly what the read-only
    // bind list exists to contain — would read the server's input instead of
    // hitting immediate EOF, and could surface those bytes in the compile
    // error returned to the client.
    // Its own process group, so the timeout can signal the whole tree.
    // `child.kill()` reaps the direct child only: outside bwrap — macOS, and
    // the documented `DISABLE_BWRAP=1` trusted-local mode — the command is a
    // bare `g++`, and the process that actually wedges is `cc1plus` or `ld`
    // underneath it. Killing the driver left the wedge running, holding the
    // pipes open (so the reader threads below never finish) and still writing
    // into the build directory the retry is about to reuse.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    #[cfg(unix)]
    let child_group = child.id() as i32;
    let mut out_pipe = child.stdout.take().expect("stdout piped above");
    let mut err_pipe = child.stderr.take().expect("stderr piped above");
    let out_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut out_pipe, &mut buf);
        buf
    });
    let err_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut err_pipe, &mut buf);
        buf
    });

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            // The group first — `process_group(0)` made the child its own
            // leader, so its pid is the group id and a negative pid signals
            // every descendant. Then the direct child, in case the group
            // signal did not apply.
            #[cfg(unix)]
            unsafe {
                libc::kill(-child_group, libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("{what} exceeded {}s and was killed", timeout.as_secs()),
            ));
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    Ok(Output {
        status,
        stdout: out_reader.join().unwrap_or_default(),
        stderr: err_reader.join().unwrap_or_default(),
    })
}

/// Locate the compile-chain assets.
///
/// `SASY_SOUFFLE_ASSETS` selects an operator-managed directory containing all six
/// required files. Otherwise, materialize the binary's embedded assets under
/// `<build-cache root>/assets/<asset set hash>`. Never discover assets from the
/// current working directory or another checkout.
pub(crate) fn assets_dir() -> Result<PathBuf, CompileError> {
    if let Some(dir) = std::env::var_os("SASY_SOUFFLE_ASSETS").filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        let missing: Vec<&str> = crate::assets::ASSETS
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !dir.join(name).is_file())
            .collect();
        if !missing.is_empty() {
            return Err(CompileError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "SASY_SOUFFLE_ASSETS={} is missing: {}",
                    dir.display(),
                    missing.join(", ")
                ),
            )));
        }
        return Ok(dir);
    }
    Ok(crate::assets::materialize_under(&assets_root())?)
}

/// The root the materialized assets live under: the build cache's directory
/// when one is configured, so both on-disk products of a compile sit together.
///
/// With the cache switched off there is no configured root, so we fall back to
/// the user's cache directory (`XDG_CACHE_HOME`, else `~/.cache`, else the
/// system temp dir).
///
/// Whatever the source, the answer is absolute: a relative root is joined to
/// the working directory here. Under the server that is a fixed place, because
/// the server resolves `--data-dir` — whose default `data/graph` is relative to
/// the directory it was started in — to an absolute path at startup, before it
/// derives the cache root from it.
///
/// The one root that is not fixed by construction is a relative
/// `SASY_SOUFFLE_BUILD_CACHE_DIR`. Nothing memoizes it: this function re-reads
/// the environment through `souffle_cache::cache_root()` on every compile and
/// resolves it against the working directory of that moment. It stays stable
/// only because the server never changes its working directory — a caller that
/// does (a test, an embedding process) moves the assets with it.
fn assets_root() -> PathBuf {
    let configured = souffle_cache::cache_root().unwrap_or_else(|| {
        std::env::var_os("XDG_CACHE_HOME")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|d| !d.is_empty())
                    .map(|home| PathBuf::from(home).join(".cache"))
            })
            .unwrap_or_else(std::env::temp_dir)
            .join("sasy")
    });
    if configured.is_absolute() {
        configured
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&configured))
            .unwrap_or(configured)
    }
}

impl SouffleAssets {
    /// Resolve the compile-chain assets: the six files this binary carries
    /// (see [`assets_dir`]) plus the Soufflé C++ headers, which are part of the
    /// runtime toolchain and are found on the host like `souffle` and `g++`.
    pub fn discover() -> Result<Self, CompileError> {
        let dir = assets_dir()?;
        // The include dir is the one that actually CONTAINS the Soufflé headers,
        // not merely the first directory that exists — on Linux `/usr/local/include`
        // usually exists but is empty, which would shadow `/usr/include` where the
        // Ubuntu `.deb` installs `souffle/SouffleInterface.h`.
        let find_include = |candidates: &[&str]| -> Result<PathBuf, CompileError> {
            for c in candidates {
                if Path::new(c).join("souffle/SouffleInterface.h").exists() {
                    return Ok(PathBuf::from(c));
                }
            }
            Err(CompileError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Soufflé headers (souffle/SouffleInterface.h) not found under any of: {:?}",
                    candidates
                ),
            )))
        };

        if let Some(runtime) = crate::nix_runtime::configured().map_err(CompileError::Sandbox)? {
            runtime.check_mount(&dir).map_err(CompileError::Sandbox)?;
            if let Some(include) = std::env::var_os("SASY_SOUFFLE_INCLUDE") {
                runtime
                    .check_mount(Path::new(&include))
                    .map_err(CompileError::Sandbox)?;
            }
        }
        Ok(Self {
            // Honor an explicit override (escape hatch), else search the usual
            // prefixes for the dir that actually contains the headers
            // (`find_include`), then de-duplicate a doubled header install
            // (`crate::souffle_include`) so a broken Homebrew bottle still
            // compiles without users hand-fixing their toolchain.
            include_dir: crate::souffle_include::normalize(
                match std::env::var_os("SASY_SOUFFLE_INCLUDE").filter(|d| !d.is_empty()) {
                    Some(dir) => PathBuf::from(dir),
                    None => find_include(&[
                        "/opt/homebrew/include",
                        "/opt/homebrew/opt/souffle/include",
                        "/opt/souffle/include",
                        "/usr/local/include",
                        "/usr/include",
                    ])?,
                },
            )?,
            sugar_py: dir.join("sugar.py"),
            evaluator_shim: dir.join("evaluator_shim.cpp"),
            evaluator_protocol: dir.join("evaluator_protocol.h"),
            json_string_codec: dir.join("json_string_codec.h"),
            functors_common: dir.join("functors_common.cpp"),
            common_policy: Some(dir.join("common_policy.dl")),
            // Both directories `assets_dir` can return hold the set and
            // nothing else: one this binary materializes, or one an operator
            // installed with `sasy install-assets`.
            exclusive_dir: Some(dir),
            word_size: detect_souffle_word_size()?,
        })
    }
}

/// Detect Soufflé RamDomain word size from `souffle --version`.
///
/// Parses "Word size: 64 bits" or "Word size: 32 bits".
pub(crate) fn detect_souffle_word_size() -> Result<u8, CompileError> {
    let (_, version) =
        selected_tool_version("souffle", "souffle").map_err(CompileError::AssuredTool)?;
    if version.contains("Word size: 64") {
        info!("Detected Soufflé 64-bit word size");
        Ok(64)
    } else if version.contains("Word size: 32") || !assured_mode() {
        Ok(32)
    } else {
        Err(CompileError::AssuredTool(
            "assured souffle version did not report its word size".to_string(),
        ))
    }
}

fn assured_mode() -> bool {
    std::env::var("SASY_ASSURED_MODE").ok().as_deref() == Some("1")
}

struct SelectedTool {
    path: PathBuf,
    sha256: String,
}

pub(crate) fn resolve_path_tool(role: &str, fallback: &str) -> Result<PathBuf, String> {
    if let Some(runtime) = crate::nix_runtime::configured()? {
        return runtime
            .tool(fallback)
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("Nix runtime manifest has no pinned {role} tool"));
    }
    let fallback = Path::new(fallback);
    let candidate = if fallback.components().count() > 1 {
        fallback.to_path_buf()
    } else {
        std::env::var_os("PATH")
            .and_then(|path| {
                std::env::split_paths(&path)
                    .map(|directory| directory.join(fallback))
                    .find(|candidate| candidate.is_file())
            })
            .ok_or_else(|| format!("{role} tool is unavailable on PATH"))?
    };
    std::fs::canonicalize(&candidate)
        .map_err(|error| format!("cannot resolve {role} tool: {error}"))
}

fn selected_tool_identity(role: &str, fallback: &str) -> Result<SelectedTool, String> {
    let runtime = crate::nix_runtime::configured()?;
    if !assured_mode() {
        let path = resolve_path_tool(role, fallback)?;
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("cannot read selected {role} tool: {error}"))?;
        return Ok(SelectedTool {
            path,
            sha256: crate::hash::hex(&Sha256::digest(&bytes)),
        });
    }
    let prefix = format!("SASY_TOOL_{}", role.to_ascii_uppercase());
    let configured = std::env::var(format!("{prefix}_PATH"))
        .map_err(|_| format!("{prefix}_PATH is required in assured mode"))?;
    let expected = std::env::var(format!("{prefix}_SHA256"))
        .map_err(|_| format!("{prefix}_SHA256 is required in assured mode"))?;
    let path = PathBuf::from(&configured);
    if !path.is_absolute()
        || expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("invalid assured {role} tool pin"));
    }
    let resolved = std::fs::canonicalize(&path)
        .map_err(|error| format!("assured {role} tool is unavailable: {error}"))?;
    if let Some(runtime) = runtime {
        let launcher = runtime
            .tool(fallback)
            .ok_or_else(|| format!("Nix runtime manifest has no pinned {role} tool"))?;
        let target = std::fs::canonicalize(launcher)
            .map_err(|error| format!("cannot resolve Nix {role} tool: {error}"))?;
        if target != resolved {
            return Err(format!(
                "assured {role} tool differs from Nix runtime manifest"
            ));
        }
    }
    let metadata = std::fs::metadata(&resolved)
        .map_err(|error| format!("cannot inspect assured {role} tool: {error}"))?;
    if resolved != path || !metadata.is_file() || metadata.len() > 64 << 20 {
        return Err(format!(
            "assured {role} tool is not a bounded resolved file"
        ));
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!("assured {role} tool is not executable"));
    }
    let bytes = std::fs::read(&resolved)
        .map_err(|error| format!("cannot read assured {role} tool: {error}"))?;
    let actual = crate::hash::hex(&Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!("assured {role} tool bytes changed"));
    }
    Ok(SelectedTool {
        // The authenticated target bytes still match the pin. Preserve the
        // package launcher name for compiler aliases/multicall executables.
        path: runtime
            .and_then(|r| r.tool(fallback))
            .map(Path::to_path_buf)
            .unwrap_or(resolved),
        sha256: actual,
    })
}

pub(crate) fn selected_tool(role: &str, fallback: &str) -> Result<PathBuf, String> {
    selected_tool_identity(role, fallback).map(|tool| tool.path)
}

pub(crate) fn selected_tool_version(
    role: &str,
    fallback: &str,
) -> Result<(PathBuf, String), String> {
    selected_tool_version_and_digest(role, fallback).map(|(path, version, _)| (path, version))
}

pub(crate) fn selected_tool_version_and_digest(
    role: &str,
    fallback: &str,
) -> Result<(PathBuf, String, String), String> {
    let mut tool = selected_tool_identity(role, fallback)?;
    if let Some(runtime) = crate::nix_runtime::configured()? {
        tool.sha256 = runtime.cache_identity(&tool.sha256);
    }
    let output = match Command::new(&tool.path).arg("--version").output() {
        Ok(output) => output,
        Err(_) if !assured_mode() => {
            return Ok((tool.path, String::new(), tool.sha256));
        }
        Err(error) => return Err(format!("assured {role} version failed: {error}")),
    };
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if assured_mode() && (!output.status.success() || version.is_empty()) {
        return Err(format!("assured {role} returned no successful version"));
    }
    Ok((tool.path, version, tool.sha256))
}

/// Auto-detect companion functor C++ file for a policy.
///
/// For `airline_policy.dl`, checks in order:
///   1. `functors.cpp` (generic, in same directory)
///   2. `airline_functors.cpp` (replace `_policy` with `_functors`)
///   3. `airline_functors.cpp` (base prefix before `_policy`)
///   4. `airline_policy_functors.cpp` (full stem + `_functors`)
pub fn find_functors(policy_path: &Path) -> Option<String> {
    let dir = policy_path.parent()?;
    let stem = policy_path.file_stem()?.to_str()?;

    let mut candidates = vec![
        dir.join("functors.cpp"),
        dir.join(format!("{}.cpp", stem.replace("_policy", "_functors"))),
    ];

    // Base prefix before "_policy"
    if let Some(base) = stem.split("_policy").next() {
        candidates.push(dir.join(format!("{}_functors.cpp", base)));
    }
    // Full stem + _functors
    candidates.push(dir.join(format!("{}_functors.cpp", stem)));

    for path in &candidates {
        if path.exists() {
            info!("Found custom functors: {}", path.display());
            return std::fs::read_to_string(path).ok();
        }
    }
    None
}

/// Create the startup compile directory at `<data-dir>/souffle-build/bootstrap`.
///
/// The server resolves the data directory to an absolute path. Keeping builds
/// here avoids depending on a writable working directory. Mode 0o700 protects
/// intermediate sources and evaluator binaries from other users; failures name
/// the affected path.
pub fn bootstrap_build_dir(data_dir: &Path) -> Result<PathBuf, CompileError> {
    let dir = data_dir.join("souffle-build").join("bootstrap");
    std::fs::create_dir_all(&dir).map_err(|error| {
        CompileError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to create the bootstrap build directory {}: {error}",
                dir.display()
            ),
        ))
    })?;
    #[cfg(unix)]
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).map_err(|error| {
        CompileError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to restrict the bootstrap build directory {}: {error}",
                dir.display()
            ),
        ))
    })?;
    Ok(dir)
}

/// Compile a Soufflé policy file with auto-discovered assets and functors.
pub fn compile_policy_file(
    policy_path: &Path,
    build_dir: &Path,
    functor_source: Option<&str>,
) -> Result<CompileResult, CompileError> {
    let assets = SouffleAssets::discover()?;
    // Auto-detect companion functor file if none provided
    let auto_functors = if functor_source.is_none() {
        find_functors(policy_path)
    } else {
        None
    };
    let effective_functors = functor_source.or(auto_functors.as_deref());
    compile_souffle_from_file(policy_path, effective_functors, build_dir, &assets)
}

/// Compile a policy from a file path.
///
/// Reads the source and delegates to `compile_souffle_with_assets`,
/// which prepends `common_policy.dl` automatically.
pub fn compile_souffle_from_file(
    policy_path: &Path,
    functor_source: Option<&str>,
    build_dir: &Path,
    assets: &SouffleAssets,
) -> Result<CompileResult, CompileError> {
    let source = std::fs::read_to_string(policy_path)?;
    compile_souffle_with_assets(&source, functor_source, build_dir, assets)
}

/// Result of a policy compilation.
pub struct CompileResult {
    /// Path to the compiled evaluator binary.
    pub binary_path: PathBuf,
    /// Backend name (e.g., "souffle", "souffle-interpreted").
    pub backend: String,
    /// Desugared policy path (for rule metadata loading).
    pub policy_path: Option<PathBuf>,
}

/// Errors from policy compilation.
#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("Preprocessor failed: {0}")]
    Preprocess(String),
    #[error("Soufflé compilation failed: {0}")]
    SouffleCompile(String),
    #[error("C++ compilation failed: {0}")]
    CppCompile(String),
    #[error("Cargo build failed: {0}")]
    CargoBuild(String),
    #[error("Sandbox unavailable: {0}")]
    Sandbox(String),
    #[error("Assured tool invalid: {0}")]
    AssuredTool(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Prepend `common_policy.dl` to policy source and strip existing `#include` for it.
///
/// This ensures the base type/relation declarations are always present
/// without requiring each `.dl` file to include them manually.
pub fn prepend_common_policy(
    policy_source: &str,
    assets: &SouffleAssets,
) -> Result<String, CompileError> {
    let cleaned = clear_authored_inference_markers(policy_source);
    let policy_source = cleaned.as_str();
    if let Some(ref cp) = assets.common_policy {
        // Idempotency: if the caller (e.g. a test harness or an SDK that
        // inlines #include directives before uploading) has already baked
        // common_policy's declarations into the source, don't prepend a
        // second copy — Soufflé treats duplicated .type / .decl as a hard
        // error. We look for `.decl IsAuthorized`, a signature marker
        // unique to common_policy.
        let already_inlined = policy_source
            .lines()
            .any(|l| l.trim_start().starts_with(".decl IsAuthorized"));
        if already_inlined {
            return Ok(policy_source
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    !(trimmed.starts_with("#include") && trimmed.contains("common_policy"))
                })
                .collect::<Vec<_>>()
                .join("\n"));
        }
        let common = std::fs::read_to_string(cp)?;
        // Inject a marker between common_policy and the user policy
        // so ``sugar.py``'s default-rule injection can scan only the
        // user portion when deciding whether to emit gating
        // defaults. See ``_default_rules`` in ``sugar.py``.
        let combined = format!(
            "{}\n\n// === USER_POLICY_BEGIN ===\n\n{}",
            common, policy_source,
        );
        Ok(combined
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !(trimmed.starts_with("#include") && trimmed.contains("common_policy"))
            })
            .collect::<Vec<_>>()
            .join("\n"))
    } else {
        Ok(policy_source.to_string())
    }
}

/// Provenance comments are compiler-owned. Neutralize authored copies before
/// inserting the real common/user boundary; quoted policy content is unchanged.
fn clear_authored_inference_markers(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len());
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'"' {
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == b'"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
        } else if bytes[index..].starts_with(b"//") || bytes[index..].starts_with(b"/*") {
            let end = if bytes[index..].starts_with(b"//") {
                source[index..]
                    .find('\n')
                    .map_or(source.len(), |offset| index + offset)
            } else {
                source[index + 2..]
                    .find("*/")
                    .map_or(source.len(), |offset| index + 4 + offset)
            };
            output.push_str(&source[start..index]);
            output.push_str(
                &source[index..end]
                    .replace("SASY_AUTO_GATE_DEFAULT:", "SASY_AUTHORED_GATE_MARKER:")
                    .replace(
                        "=== USER_POLICY_BEGIN ===",
                        "=== AUTHORED_POLICY_BOUNDARY ===",
                    ),
            );
            start = end;
            index = end;
        } else {
            index += 1;
        }
    }
    output.push_str(&source[start..]);
    output
}

/// Compile Soufflé policy source with discovered assets.
///
/// Pipeline: prepend common_policy → sugar.py → souffle -g → g++ → evaluator binary
///
/// Once sugar.py has produced the desugared source, the cache key is
/// computed and (if [`souffle_cache`] is enabled) the cached ELF is
/// hardlinked in place of running souffle/g++. The desugared file is
/// always written so [`crate::engine::Engine::load_rule_metadata`]
/// has source to parse, regardless of cache hit/miss.
pub fn compile_souffle_with_assets(
    policy_source: &str,
    functor_source: Option<&str>,
    build_dir: &Path,
    assets: &SouffleAssets,
) -> Result<CompileResult, CompileError> {
    compile_souffle_into_cache(policy_source, functor_source, build_dir, assets, None)
}

/// Shared additive attribution and conservative action-gate inference. Keep
/// this after Python desugaring and before hashing, checking or compiling, so
/// compiled, interpreted, validation and baked policy packs see the same rules.
pub(crate) fn apply_policy_inference(desugared: &str) -> Result<String, String> {
    use sasy_policy_analysis::{action_gates, allow_attribution};
    if allow_attribution::reserved_namespace_collision(desugared) {
        return Err("policy uses the reserved SasyAllow compiler namespace".into());
    }
    let attribution = allow_attribution::transform_desugared(desugared);
    if let Some(reason) = &attribution.fallback_reason {
        tracing::debug!(
            reason,
            "allow-route attribution unavailable; retaining author hints"
        );
    }
    // The marker in the desugared text retains the complete user portion,
    // including resolved includes. Generated diagnostic outputs are roots too.
    Ok(action_gates::infer_action_gates(&attribution.source, &attribution.source).source)
}

/// [`compile_souffle_with_assets`] with the build cache named outright.
///
/// `cache_root` is the directory the compiled binary is stored in and looked
/// up from; `None` means the one `SASY_SOUFFLE_BUILD_CACHE_DIR` names, which
/// is what the server uses. Naming it here is for callers that must not share
/// a cache with the rest of the process — a test that counts the entries its
/// own compiles produced cannot do that in a directory another compile
/// running at the same moment can also write to, and pointing the environment
/// variable at it would do exactly that, since the variable is process-wide.
pub fn compile_souffle_into_cache(
    policy_source: &str,
    functor_source: Option<&str>,
    build_dir: &Path,
    assets: &SouffleAssets,
    cache_root: Option<&Path>,
) -> Result<CompileResult, CompileError> {
    let sugar_py = &assets.sugar_py;
    let souffle_include = &assets.include_dir;
    let evaluator_shim = &assets.evaluator_shim;
    let evaluator_protocol = &assets.evaluator_protocol;
    let json_string_codec = &assets.json_string_codec;
    let functors_common = &assets.functors_common;
    let python = selected_tool("python3", "python3").map_err(CompileError::AssuredTool)?;
    let (souffle, souffle_version, souffle_sha256) =
        selected_tool_version_and_digest("souffle", "souffle")
            .map_err(CompileError::AssuredTool)?;
    let (cxx, cxx_version, cxx_sha256) =
        selected_tool_version_and_digest("cxx", "g++").map_err(CompileError::AssuredTool)?;

    std::fs::create_dir_all(build_dir)?;

    // Fail closed if a sandbox is expected but can't run (bwrap present yet
    // unable to create namespaces, and not explicitly disabled) — don't silently
    // compile this policy's C++ unsandboxed. Trusted callers
    // set DISABLE_BWRAP=1; macOS/dev without bwrap passes through.
    crate::sandbox::check_sandbox().map_err(CompileError::Sandbox)?;

    // Prepend common_policy.dl and strip existing #include for it
    let full_source = prepend_common_policy(policy_source, assets)?;

    // Write combined policy source
    let policy_path = build_dir.join("policy.dl");
    std::fs::write(&policy_path, &full_source)?;

    // Run sugar preprocessor — sandboxed so a malicious .dl can't
    // exploit a python3 / sugar.py bug to read server state.
    let preprocess = sandboxed_command(
        python.to_str().unwrap(),
        &[
            sugar_py.to_str().unwrap(),
            "--resolve-includes",
            policy_path.to_str().unwrap(),
        ],
        build_dir,
        true,
        &[sugar_py.as_path()],
    )
    .map_err(CompileError::Sandbox)?;
    let preprocess = output_bounded(preprocess, "sugar.py preprocessing")?;
    if !preprocess.status.success() {
        return Err(CompileError::Preprocess(
            String::from_utf8_lossy(&preprocess.stderr).to_string(),
        ));
    }

    let desugared_path = build_dir.join("desugared.dl");
    let desugared_source = apply_policy_inference(&String::from_utf8_lossy(&preprocess.stdout))
        .map_err(CompileError::Preprocess)?;
    std::fs::write(&desugared_path, &desugared_source)?;

    let output_bin = build_dir.join("souffle-evaluator");

    // Cache lookup: a hit lets us skip the ~7s souffle/g++ pipeline
    // entirely. Reading the shim/header/functors_common sources keeps a
    // toolchain-asset change (e.g. evaluator_shim.cpp edit) from
    // serving a stale binary.
    let magic_set_env = std::env::var("SASY_SOUFFLE_MAGIC_SET").ok();
    let magic_set = magic_set_env
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let shim_src = std::fs::read_to_string(evaluator_shim).unwrap_or_default();
    let evaluator_protocol_src = std::fs::read_to_string(evaluator_protocol)?;
    let json_string_codec_src = std::fs::read_to_string(json_string_codec)?;
    let functors_common_src = std::fs::read_to_string(functors_common).unwrap_or_default();
    let cache_key = souffle_cache::cache_key(&souffle_cache::CacheKeyInputs {
        desugared_source: &desugared_source,
        word_size: assets.word_size,
        magic_set,
        functor_source,
        evaluator_shim_source: &shim_src,
        evaluator_protocol_source: &evaluator_protocol_src,
        json_string_codec_source: &json_string_codec_src,
        functors_common_source: &functors_common_src,
        souffle_version: &souffle_version,
        souffle_sha256: &souffle_sha256,
        cxx_version: &cxx_version,
        cxx_sha256: &cxx_sha256,
        // Tenant id isn't plumbed through this code path yet; we
        // hash a single global namespace, matching today's
        // shared-tenant semantics.
        tenant_id: None,
    });
    if souffle_cache::try_get_in(cache_root, &cache_key, &output_bin) {
        info!(
            key = %cache_key,
            "Soufflé policy served from cache: {}",
            output_bin.display()
        );
        return Ok(CompileResult {
            binary_path: output_bin,
            backend: "souffle".to_string(),
            policy_path: Some(desugared_path),
        });
    }

    // Generate C++ from Soufflé — sandboxed so a crafted .dl can't
    // leverage souffle's parser for side-effects beyond the build dir.
    let cpp_path = build_dir.join("policy_program.cpp");
    let mut souffle_args: Vec<String> = vec!["-g".into(), cpp_path.to_str().unwrap().into()];
    // Magic-set transform: opt-in via env, since it can occasionally
    // pessimize already-tight rules. Pass through verbatim — Soufflé
    // accepts a comma-separated relation list or `*` for all.
    if let Some(m) = magic_set {
        souffle_args.push(format!("-m{}", m));
    }
    souffle_args.push(desugared_path.to_str().unwrap().into());
    let souffle_args_ref: Vec<&str> = souffle_args.iter().map(String::as_str).collect();
    let gen = sandboxed_command(
        souffle.to_str().unwrap(),
        &souffle_args_ref,
        build_dir,
        true,
        &[souffle_include.as_path()],
    )
    .map_err(CompileError::Sandbox)?;
    let gen = output_bounded(gen, "souffle codegen")?;
    if !gen.status.success() {
        let err = format!(
            "{}{}",
            String::from_utf8_lossy(&gen.stderr),
            String::from_utf8_lossy(&gen.stdout),
        );
        return Err(CompileError::SouffleCompile(err));
    }

    // Compile evaluator binary
    let ram_domain_def = format!("-DRAM_DOMAIN_SIZE={}", assets.word_size);
    // g++ output flags come from the single shared const that the cache
    // key + build signature also fold in (so they can't drift). The
    // word-size define is added separately — it's keyed via `word_size`.
    let mut args: Vec<&str> = Vec::new();
    args.extend(souffle_cache::GXX_OUTPUT_FLAGS.iter().copied());
    args.push(&ram_domain_def);
    args.push("-I");
    args.push(souffle_include.to_str().unwrap());
    args.push(evaluator_shim.to_str().unwrap());
    args.push(cpp_path.to_str().unwrap());
    args.push(functors_common.to_str().unwrap());

    // Add custom functor source if provided
    let functor_path = build_dir.join("custom_functors.cpp");
    if let Some(src) = functor_source {
        std::fs::write(&functor_path, src)?;
        args.push(functor_path.to_str().unwrap());
    }

    args.extend_from_slice(&["-o", output_bin.to_str().unwrap(), "-lpthread"]);

    // g++ runs caller-supplied .cpp — the most dangerous stage
    // because `#include "/path/to/secret"` would normally expose file
    // contents via compile errors. Inside bwrap the only paths
    // reachable are the toolkit headers and the build dir; `#include`
    // to absolute paths outside those binds fails with "file not
    // found", leaking nothing.
    let compile = sandboxed_command(
        cxx.to_str().unwrap(),
        &args,
        build_dir,
        true,
        &compile_ro_binds(assets),
    )
    .map_err(CompileError::Sandbox)?;
    let compile = output_bounded(compile, "C++ compile")?;
    if !compile.status.success() {
        return Err(CompileError::CppCompile(
            String::from_utf8_lossy(&compile.stderr).to_string(),
        ));
    }

    info!("Soufflé policy compiled: {}", output_bin.display());
    souffle_cache::put_in(cache_root, &cache_key, &output_bin);

    Ok(CompileResult {
        binary_path: output_bin,
        backend: "souffle".to_string(),
        policy_path: Some(desugared_path),
    })
}

#[cfg(test)]
#[path = "compiler_tests.rs"]
mod tests;
