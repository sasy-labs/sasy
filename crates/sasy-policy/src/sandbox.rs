//! Bubblewrap wrapper shared by the compile chain and the evaluator
//! spawn site.
//!
//! Both call sites face the same threat: caller-supplied C++ that links
//! into a process running under the server's uid. For the compile chain
//! the danger is `#include "/etc/passwd"`-style file-read oracles in
//! g++; for the evaluator it is `__attribute__((constructor))` /
//! static-storage initializers that run *before* `main`, ahead of the
//! in-process seccomp filter.
//!
//! On Linux with `bwrap` on PATH, both call sites should run inside a
//! stripped sandbox with only the toolchain / binary readable, no
//! network, and `--clearenv`. On macOS / local dev where `bwrap` is
//! absent, both fall back to direct invocation and rely on
//! developer-environment trust.

use std::env;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Once, OnceLock};

#[cfg(unix)]
use std::os::unix::io::RawFd;
#[cfg(not(unix))]
type RawFd = i32;

/// Env vars forwarded into sandboxed processes. Everything else is
/// stripped so the child can't see ANTHROPIC_API_KEY, OPENAI_API_KEY,
/// or other server secrets. Header search paths are deliberately NOT
/// forwarded: the caller's asset set decides what a compile may include.
pub const FORWARDED_ENV: &[&str] = &["LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"];

/// Resource limits applied via `prlimit(1)` so a malicious functor
/// or a template-bomb policy can't trigger memory/CPU/file-size DoS
/// against the server. Picked to be generous for legitimate compiles
/// (g++ on a small policy uses ~100 MB and finishes in seconds) but
/// to stop a runaway in well under a minute of wall-clock.
pub struct ResourceLimits {
    /// Address-space cap (bytes). Stops template-bomb / recursive
    /// macro / `#pragma GCC optimize` memory blowup.
    pub address_space_bytes: u64,
    /// CPU-second cap. `Some` for compile-time (one-shot, finite),
    /// `None` for the long-running evaluator (per-query timeouts
    /// belong at the IPC layer).
    pub cpu_seconds: Option<u32>,
    /// Max single-file size (bytes). Stops `g++ -o /work/huge` from
    /// filling /tmp.
    pub fsize_bytes: u64,
}

impl ResourceLimits {
    /// Defaults for the compile chain (g++, sugar.py, souffle).
    pub fn compile() -> Self {
        Self {
            address_space_bytes: 4 * 1024 * 1024 * 1024,
            cpu_seconds: Some(600),
            fsize_bytes: 256 * 1024 * 1024,
        }
    }
    /// Defaults for the runtime evaluator. No CPU cap because the evaluator is
    /// long-running and serves many queries: `RLIMIT_CPU` counts cumulative
    /// CPU time, so any finite value is a countdown to SIGKILL on a healthy
    /// evaluator, reached sooner the busier the session. What bounds a runaway
    /// is the stall window in [`crate::session_evaluator`]: the process is
    /// killed and a fresh one re-bootstraps from the graph store.
    ///
    /// That argument only holds because the kill reaches the process doing the
    /// spinning. Under bwrap the evaluator runs inside a new PID namespace
    /// with no PDEATHSIG, so signalling the handle we hold would leave it
    /// alive; `IpcChild` puts the child in its own process group and kills the
    /// group. Without that, every stall would leak a process burning a core.
    pub fn evaluator() -> Self {
        Self {
            address_space_bytes: 2 * 1024 * 1024 * 1024,
            // Deliberately no CPU cap. `RLIMIT_CPU` counts CUMULATIVE CPU
            // time, and this process is long-lived and serves every query for
            // its session — so any finite cap is a countdown to SIGKILL on a
            // perfectly healthy evaluator, reached sooner the busier the
            // session is.
            cpu_seconds: None,
            fsize_bytes: 64 * 1024 * 1024,
        }
    }
}

fn path_lookup(name: &str) -> bool {
    match crate::nix_runtime::configured() {
        Ok(Some(runtime)) => return runtime.tool(name).is_some(),
        Err(_) => return false,
        Ok(None) => {}
    }
    let path = match env::var_os("PATH") {
        Some(p) => p,
        None => return false,
    };
    env::split_paths(&path).any(|d| d.join(name).exists())
}

fn bwrap_disabled() -> bool {
    env::var("DISABLE_BWRAP").ok().as_deref() == Some("1")
}

/// Whether `SASY_EVALUATOR_BWRAP`'s value switches the evaluator's bwrap
/// wrapper off. Split from the env read so the rule is testable without
/// mutating the process environment, which would race every other test that
/// asks about the sandbox.
fn evaluator_bwrap_off(value: Option<&str>) -> bool {
    value == Some("0")
}

/// True unless the evaluator's bwrap wrapper is switched off with
/// `SASY_EVALUATOR_BWRAP=0`, the documented escape hatch for running the
/// evaluator naked (debugging, or a platform where bwrap misbehaves).
///
/// Separate from [`bwrap_disabled`]: `DISABLE_BWRAP=1` turns the sandbox off
/// everywhere, this one only for the evaluator, leaving the compile chain
/// confined.
fn evaluator_bwrap_enabled() -> bool {
    !evaluator_bwrap_off(env::var("SASY_EVALUATOR_BWRAP").ok().as_deref())
}

/// True if bwrap should be used: not explicitly disabled, on PATH, and a probe
/// confirms it can actually create namespaces. Set `DISABLE_BWRAP=1` to bypass
/// the sandbox (e.g. trusted policy, or attaching strace to the child). A
/// present-but-broken bwrap returns false here — but the compile chain calls
/// [`check_sandbox`] first and FAILS CLOSED in that case, so this never silently
/// downgrades an untrusted compile to unsandboxed.
pub fn has_bubblewrap() -> bool {
    !bwrap_disabled() && path_lookup("bwrap") && bwrap_usable()
}

/// Argv for the usability probe: a no-op `true` run with the SAME isolation
/// flags as a real sandboxed compile/evaluator run, built via the shared
/// [`build_bwrap_argv`]. Sharing the builder is the point: a hand-rolled subset
/// can pass while the real run fails — e.g. a kernel that creates the network
/// namespace but rejects bwrap's loopback setup (`RTM_NEWADDR: Operation not
/// permitted`, common on Ubuntu / hardened kernels / containers). A non-faithful
/// probe lets such a bwrap slip past [`check_sandbox`] and die mid-compile with
/// a cryptic `bwrap: loopback …` error instead of the actionable guidance.
fn probe_argv(work_dir: &Path, seccomp_fd: Option<RawFd>) -> Result<Vec<String>, String> {
    build_bwrap_argv(
        "true",
        &[],
        work_dir,
        false,
        &[],
        &[],
        None,
        true,
        seccomp_fd,
    )
}

/// Run one probe. `Ok` iff bwrap exited 0; `Err` carries what it printed on
/// stderr, which is the only account of why a filter was refused.
///
/// When a descriptor is given it is passed the way the evaluator spawn passes
/// it — `--seccomp FD`, `FD_CLOEXEC` cleared in this child alone, from a
/// `pre_exec` hook — because a probe that hands bwrap the flag differently
/// from the real spawn is a probe that can disagree with it.
fn run_probe(work_dir: &Path, program: Option<&crate::seccomp::ProgramFd>) -> Result<(), String> {
    let argv = probe_argv(work_dir, program.map(|p| p.raw()))?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    if let Some(fd) = program.map(|p| p.raw()) {
        use std::os::unix::process::CommandExt;
        // SAFETY: the closure runs between `fork` and `execve` in the probe's
        // own child; the parent holds the descriptor open across the run, and
        // neither the call nor the error it returns allocates or locks.
        unsafe {
            cmd.pre_exec(move || {
                if crate::seccomp::inherit_in_child(fd) {
                    Ok(())
                } else {
                    Err(std::io::Error::from_raw_os_error(libc::EBADF))
                }
            });
        }
    }
    let out = match cmd.output() {
        Ok(out) => out,
        Err(e) => return Err(e.to_string()),
    };
    if out.status.success() {
        return Ok(());
    }
    let said = String::from_utf8_lossy(&out.stderr).trim().to_string();
    Err(if said.is_empty() {
        format!("bwrap exited {}", out.status)
    } else {
        said
    })
}

/// Namespace isolation and pre-exec seccomp support are probed separately.
/// Failure to create namespaces stops the compile chain. If only seccomp is
/// unavailable, ordinary mode can spawn without it and reports
/// `seccomp_pre_exec: false`; assured mode requires the filter.
struct BwrapProbe {
    /// The full spawn, `--seccomp FD` included, ran.
    with_seccomp: bool,
    /// The same run without the flag ran.
    without_seccomp: bool,
    /// What bwrap said when it refused the filter, for the one startup
    /// warning. `None` when the flag was never in play (no program for this
    /// architecture, a bubblewrap too old for it, or the debug switch).
    seccomp_failure: Option<String>,
}

impl BwrapProbe {
    /// The decision from the two runs: the second one is only made, and only
    /// needed, when the first refused the filter.
    fn from_runs(
        with_seccomp: Option<Result<(), String>>,
        without_seccomp: impl FnOnce() -> bool,
    ) -> Self {
        match with_seccomp {
            Some(Ok(())) => Self {
                with_seccomp: true,
                without_seccomp: true,
                seccomp_failure: None,
            },
            Some(Err(said)) => Self {
                with_seccomp: false,
                without_seccomp: without_seccomp(),
                seccomp_failure: Some(said),
            },
            None => Self {
                with_seccomp: false,
                without_seccomp: without_seccomp(),
                seccomp_failure: None,
            },
        }
    }

    /// bwrap runs on this host at all — with the filter, or without it.
    fn bwrap_usable(&self) -> bool {
        self.with_seccomp || self.without_seccomp
    }

    /// The deny-list can be handed to bwrap before the `execve`.
    fn seccomp_flag_usable(&self) -> bool {
        self.with_seccomp
    }
}

/// One-time (process-cached) probe: run `true` through the real sandbox argv.
/// Containers / WSL / hardened kernels block unprivileged user/network
/// namespaces (or reject loopback setup after `--unshare-net`), so bwrap exits
/// non-zero here; the cached result lets the compile chain decide once. Because
/// the probe shares [`build_bwrap_argv`] with real runs, a bwrap that fails the
/// real compile also fails here — so [`check_sandbox`] can fail closed with
/// actionable guidance instead of the compile dying on a raw bwrap error.
fn probe() -> &'static BwrapProbe {
    static PROBE: OnceLock<BwrapProbe> = OnceLock::new();
    PROBE.get_or_init(|| {
        // build_bwrap_argv always binds and chdirs into work_dir, so it must
        // exist on the host; a private subdir of the temp dir is enough.
        let work_dir = env::temp_dir().join("sasy-bwrap-probe");
        if std::fs::create_dir_all(&work_dir).is_err() {
            return BwrapProbe::from_runs(None, || false);
        }
        // The descriptor only where the real spawn would build one, so the
        // probe asks the question the spawn asks and no other.
        let takes_the_flag = bwrap_version().is_some_and(|v| v >= BWRAP_SECCOMP_FD_MIN_VERSION);
        let program = if pre_exec_seccomp_disabled() || !takes_the_flag {
            None
        } else {
            crate::seccomp::deny_list_program_fd()
        };
        let with_seccomp = program
            .as_ref()
            .map(|handle| run_probe(&work_dir, Some(handle)));
        BwrapProbe::from_runs(with_seccomp, || run_probe(&work_dir, None).is_ok())
    })
}

/// One-time (process-cached) probe: run `true` through the real sandbox argv.
/// Containers / WSL / hardened kernels block unprivileged user/network
/// namespaces (or reject loopback setup after `--unshare-net`), so bwrap exits
/// non-zero here; the cached result lets the compile chain decide once. Because
/// the probe shares [`build_bwrap_argv`] with real runs, a bwrap that fails the
/// real compile also fails here — so [`check_sandbox`] can fail closed with
/// actionable guidance instead of the compile dying on a raw bwrap error.
///
/// A host that runs bwrap but refuses the seccomp filter is usable: the
/// evaluator is spawned without the flag (see [`BwrapProbe`]).
fn bwrap_usable() -> bool {
    probe().bwrap_usable()
}

/// The first release of bubblewrap that takes a seccomp program on a file
/// descriptor (`--seccomp FD`). Older ones reject the flag, so the evaluator
/// would fail to spawn at all rather than start unfiltered.
const BWRAP_SECCOMP_FD_MIN_VERSION: (u32, u32) = (0, 4);

/// The version `bwrap --version` reports, as (major, minor).
///
/// Its output is one line, `bubblewrap 0.8.0`; anything else — a wrapper that
/// prints something of its own, a build with no version — is `None`, and the
/// caller then treats the flag as unavailable rather than guessing. Split from
/// the process spawn so the parsing rule is testable on a host with no bwrap.
fn parse_bwrap_version(output: &str) -> Option<(u32, u32)> {
    let field = output.split_whitespace().find(|w| {
        w.split('.')
            .next()
            .is_some_and(|first| !first.is_empty() && first.chars().all(|c| c.is_ascii_digit()))
    })?;
    let mut parts = field.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// One-time (process-cached) `bwrap --version`. Cached for the same reason
/// [`bwrap_usable`] is: the answer decides how every evaluator in this process
/// is spawned, and re-running a subprocess per spawn to learn a constant is
/// waste. A bwrap replaced under a running binary keeps the old answer until
/// the binary restarts.
fn bwrap_version() -> Option<(u32, u32)> {
    static VERSION: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *VERSION.get_or_init(|| {
        let runtime = crate::nix_runtime::configured().ok()?;
        let bwrap = runtime
            .and_then(|r| r.tool("bwrap"))
            .unwrap_or(Path::new("bwrap"));
        let out = Command::new(bwrap)
            .arg("--version")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        parse_bwrap_version(&String::from_utf8_lossy(&out.stdout))
    })
}

/// Whether `SASY_SECCOMP_PRE_EXEC_DISABLE=1` is set. Debug-only: it puts the
/// evaluator install the deny-list itself, after its static constructors have
/// already run, instead of having it installed before the exec. It exists so a
/// test can show the difference between the two spawns on one host.
fn pre_exec_seccomp_disabled() -> bool {
    env::var("SASY_SECCOMP_PRE_EXEC_DISABLE").ok().as_deref() == Some("1")
}

/// True iff the evaluator's syscall deny-list is installed BEFORE its first
/// instruction — that is, iff bwrap will wrap the spawn, its version takes
/// `--seccomp FD`, this build has a program for the architecture, and the
/// debug switch is not set.
///
/// Distinct from [`sandbox_available`], which answers whether caller-supplied
/// C++ runs confined at all. Both can be true with only namespaces in place
/// before `execve`: on an older bubblewrap a functor's
/// `__attribute__((constructor))` still runs inside the namespaces with the
/// full syscall surface. Nothing that reports the sandbox's state should
/// conflate the two.
pub fn seccomp_pre_exec_available() -> bool {
    seccomp_pre_exec_with(probe())
}

/// [`seccomp_pre_exec_available`] against a given probe result, so the
/// host-independent half of the answer — including that a probe which could
/// not hand bwrap the filter turns it off — can be tested on any host.
fn seccomp_pre_exec_with(probe: &BwrapProbe) -> bool {
    sandbox_available()
        && !pre_exec_seccomp_disabled()
        && crate::seccomp::deny_list_program().is_some()
        && bwrap_version().is_some_and(|v| v >= BWRAP_SECCOMP_FD_MIN_VERSION)
        && probe.seccomp_flag_usable()
}

/// The startup warning owed to a host whose bwrap runs but will not take the
/// deny-list: what was decided, and what bwrap said. `None` when the filter
/// was never refused.
fn seccomp_fallback_warning(probe: &BwrapProbe) -> Option<String> {
    let said = probe.seccomp_failure.as_ref()?;
    Some(format!(
        "this host cannot take the evaluator's syscall deny-list before `execve`: bwrap \
         refused `--seccomp` ({said}). Every evaluator is spawned without the flag, and \
         the deny-list is installed in-process instead, after the static constructors of \
         any functor C++ linked into it have already run."
    ))
}

/// Say once, at startup, that this host confines the evaluator with namespaces
/// but installs the syscall deny-list only after the evaluator's static
/// constructors have run. Only for the cases an operator can act on — a
/// working bwrap too old for `--seccomp FD`, or one that refused the filter —
/// since a host with no bwrap at all is already reported by the functor-gate
/// warnings.
///
/// Once per process: it describes a fact about the host, and repeating it per
/// evaluator spawn would bury it.
pub fn log_seccomp_pre_exec_status() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if !sandbox_available() || seccomp_pre_exec_available() {
            return;
        }
        if let Some(reason) = seccomp_fallback_warning(probe()) {
            tracing::warn!(seccomp_pre_exec = false, "{reason}");
            return;
        }
        if let Some((major, minor)) = bwrap_version() {
            if (major, minor) < BWRAP_SECCOMP_FD_MIN_VERSION {
                tracing::warn!(
                    bwrap_version = format!("{major}.{minor}"),
                    seccomp_pre_exec = false,
                    "bubblewrap {major}.{minor} does not take a seccomp program on a file \
                     descriptor (needs {}.{} or newer), so the evaluator's syscall deny-list \
                     is installed in-process, after the static constructors of any functor \
                     C++ linked into it have already run",
                    BWRAP_SECCOMP_FD_MIN_VERSION.0,
                    BWRAP_SECCOMP_FD_MIN_VERSION.1,
                );
            }
        }
    });
}

/// Guard the compile chain. The policy/functor C++ runs through the toolchain,
/// which a MALICIOUS policy could exploit (an `#include` file-read oracle in
/// g++, a functor `constructor` that reads creds). So when a sandbox is expected
/// but can't run, we REFUSE to compile rather than silently fall back — an
/// untrusted policy must never degrade to unsandboxed.
///
/// - `Ok(())`  — bwrap absent (platform doesn't sandbox; trusted dev), explicitly
///   disabled via `DISABLE_BWRAP=1` (trusted opt-out, e.g. a local deployment's
///   curated profile), or bwrap is usable.
/// - `Err(..)` — bwrap is installed but cannot create namespaces and is NOT
///   disabled: fail closed with actionable guidance.
pub fn check_sandbox() -> Result<(), String> {
    // Invalid package configuration must never masquerade as absent bwrap.
    crate::nix_runtime::configured()?;
    if bwrap_disabled() || !path_lookup("bwrap") || bwrap_usable() {
        return Ok(());
    }
    Err(
        "bubblewrap is installed but cannot create namespaces — unprivileged \
         user/network namespaces are blocked (common in containers, WSL, and \
         hardened kernels; bwrap reports EPERM / RTM_NEWADDR). Refusing to compile \
         the policy UNSANDBOXED, since an untrusted policy could exploit the \
         compiler toolchain. Set DISABLE_BWRAP=1 to compile without the sandbox \
         (safe only for TRUSTED policies, e.g. a local deployment's curated \
         profile — which sets it automatically), or enable unprivileged namespaces."
            .into(),
    )
}

/// True iff caller-supplied C++ actually runs confined on this host.
///
/// [`check_sandbox`] answers a narrower question — "may the compile proceed?"
/// — and says `Ok` in two very different situations: the sandbox works, and
/// there is no sandbox at all (macOS, a Linux host without the package,
/// `DISABLE_BWRAP=1`). This one separates them, because an authorization
/// decision about whose C++ may be compiled and loaded needs to know whether
/// the confinement it is counting on exists.
///
/// It answers for the runtime that will actually LOAD the code. The process
/// that dlopens a caller's functor is the evaluator, and the evaluator has a
/// switch of its own (`SASY_EVALUATOR_BWRAP=0`) on top of the host-wide one:
/// with it set, the compile is still confined but the evaluator is spawned
/// naked, and a `__attribute__((constructor))` in the caller's C++ runs
/// unconfined as the service user. Asking only whether bwrap exists would
/// admit that upload as sandboxed. `IpcChild::spawn` calls THIS function to
/// decide whether to wrap, so the gate's answer and the spawn cannot drift.
///
/// Asked per request by the functor gate, not cached in the service: the PATH
/// lookup and the env read are re-done on every call (only the
/// namespace-usability probe is process-cached), so a `bwrap` that disappears
/// under a long-lived process is seen the next time somebody uploads a
/// functor.
pub fn sandbox_available() -> bool {
    has_bubblewrap() && evaluator_bwrap_enabled()
}

fn has_prlimit() -> bool {
    path_lookup("prlimit")
}

fn prlimit_prefix(limits: &ResourceLimits) -> Vec<String> {
    let mut argv = vec![
        "prlimit".into(),
        format!("--as={}", limits.address_space_bytes),
        format!("--fsize={}", limits.fsize_bytes),
    ];
    if let Some(cpu) = limits.cpu_seconds {
        argv.push(format!("--cpu={cpu}"));
    }
    argv.push("--".into());
    argv
}

/// Construct the argv for a bwrap-sandboxed run of `program args…`.
///
/// `work_dir` is bound at the same path inside the sandbox and used
/// as the child's cwd. `work_writable` controls whether the bind is
/// `--bind` (RW) or `--ro-bind`. `ro_binds` are additional read-only
/// host paths to expose (toolchain headers, the evaluator binary's
/// directory, etc.). `extra_env` are spawn-scoped vars added via
/// `--setenv` *after* `--clearenv` strips the inherited environment.
/// `limits` are applied via `prlimit(1)` *outside* bwrap so they
/// propagate down to the program; skipped on hosts without prlimit.
///
/// `die_with_parent` controls whether `--die-with-parent` is emitted.
/// Pass `true` for short-lived subprocesses whose creating thread is
/// guaranteed to outlive them (the compile chain — `.output()` blocks
/// to completion). Pass `false` for long-lived subprocesses spawned
/// from `tokio::task::spawn_blocking` or any other context where the
/// creating thread may exit before the child does: bwrap implements
/// `--die-with-parent` via `prctl(PR_SET_PDEATHSIG, SIGKILL)`, which
/// fires on **thread** exit, not process exit. Tokio's
/// blocking-pool default `thread_keep_alive = 10s` would then SIGKILL
/// the child ~10 s after the spawning task finishes.
///
/// `seccomp_fd` is a file descriptor bwrap reads a seccomp BPF program from
/// (`--seccomp FD`) and installs after the namespaces are set up and before
/// the `execve`. Pass `Some` for the evaluator, whose binary links
/// caller-supplied functor C++ whose static constructors would otherwise run
/// unfiltered; the caller owns the descriptor and keeps `FD_CLOEXEC` **set**
/// on it, so no unrelated fork in this process inherits it, and clears the
/// flag only in this spawn's own `pre_exec` hook, between the fork and the
/// `execve` (see `crate::seccomp::inherit_in_child`, called from
/// `evaluator/ipc.rs`). `None` leaves the flag off entirely — an old
/// bubblewrap that does not know the flag would refuse to run at all.
///
/// The parameter order is part of this function's contract: every call site
/// passes the arguments positionally, and the two trailing flags
/// (`die_with_parent`, then `seccomp_fd`) are the ones added last.
///
/// Kept separate from `sandboxed_command` so the argv is unit-testable
/// without bwrap on the host.
#[allow(clippy::too_many_arguments)] // inherent: bwrap argv needs program, args, work dir + flags, binds, env, limits
pub fn build_bwrap_argv(
    program: &str,
    args: &[&str],
    work_dir: &Path,
    work_writable: bool,
    ro_binds: &[&Path],
    extra_env: &[(&str, &str)],
    limits: Option<&ResourceLimits>,
    die_with_parent: bool,
    seccomp_fd: Option<RawFd>,
) -> Result<Vec<String>, String> {
    let runtime = crate::nix_runtime::configured()?;
    build_bwrap_argv_with_runtime(
        program,
        args,
        work_dir,
        work_writable,
        ro_binds,
        extra_env,
        limits,
        die_with_parent,
        seccomp_fd,
        runtime,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_bwrap_argv_with_runtime(
    program: &str,
    args: &[&str],
    work_dir: &Path,
    work_writable: bool,
    ro_binds: &[&Path],
    extra_env: &[(&str, &str)],
    limits: Option<&ResourceLimits>,
    die_with_parent: bool,
    seccomp_fd: Option<RawFd>,
    runtime: Option<&crate::nix_runtime::NixRuntime>,
) -> Result<Vec<String>, String> {
    if let Some(runtime) = runtime {
        runtime.check_mount(work_dir)?;
        for path in ro_binds {
            runtime.check_mount(path)?;
        }
        if extra_env
            .iter()
            .any(|(key, value)| *key == "PATH" && *value != runtime.search_path)
        {
            return Err("Nix sandbox PATH cannot be overridden by a spawn".into());
        }
    }
    let mut argv: Vec<String> = Vec::new();
    if let Some(l) = limits {
        if has_prlimit() || runtime.is_some_and(|r| r.tool("prlimit").is_some()) {
            argv.extend(prlimit_prefix(l));
            if let Some(path) = runtime.and_then(|r| r.tool("prlimit")) {
                argv[0] = path.to_string_lossy().into_owned();
            }
        }
    }
    let bwrap = runtime
        .and_then(|r| r.tool("bwrap"))
        .unwrap_or(Path::new("bwrap"));
    argv.extend([bwrap.to_string_lossy().into_owned(), "--unshare-all".into()]);
    if let Some(fd) = seccomp_fd {
        argv.push("--seccomp".into());
        argv.push(fd.to_string());
    }
    if die_with_parent {
        argv.push("--die-with-parent".into());
    }
    argv.push("--clearenv".into());
    let push_ro = |v: &mut Vec<String>, host: &str| {
        if Path::new(host).exists() {
            v.push("--ro-bind".into());
            v.push(host.into());
            v.push(host.into());
        }
    };
    for host in &["/usr", "/bin", "/lib", "/lib64", "/opt"] {
        if runtime.is_some() && *host == "/bin" {
            continue;
        }
        push_ro(&mut argv, host);
    }
    for host in &[
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/resolv.conf",
        "/etc/ld.so.conf.d",
    ] {
        push_ro(&mut argv, host);
    }
    if let Some(runtime) = runtime {
        // libc popen/system invoke /bin/sh regardless of PATH. The host
        // alias may target an unrelated Nix image closure, so expose only
        // the explicitly pinned package shell at that conventional name.
        let shell = runtime
            .tool("sh")
            .ok_or("Nix runtime manifest has no pinned sh tool")?;
        argv.extend([
            "--dir".into(),
            "/bin".into(),
            "--ro-bind".into(),
            shell.to_string_lossy().into_owned(),
            "/bin/sh".into(),
        ]);
        for path in &runtime.store_paths {
            let path = path.to_string_lossy().into_owned();
            argv.extend(["--ro-bind".into(), path.clone(), path]);
        }
    }
    argv.extend_from_slice(&[
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--tmpfs".into(),
        "/tmp".into(),
        "--tmpfs".into(),
        "/run".into(),
    ]);
    if let Some(wd) = work_dir.to_str() {
        let bind_flag = if work_writable { "--bind" } else { "--ro-bind" };
        argv.extend_from_slice(&[
            bind_flag.into(),
            wd.into(),
            wd.into(),
            "--chdir".into(),
            wd.into(),
        ]);
    }
    for p in ro_binds {
        if let Some(s) = p.to_str() {
            if Path::new(s).exists() {
                argv.push("--ro-bind".into());
                argv.push(s.into());
                argv.push(s.into());
            }
        }
    }
    argv.extend_from_slice(&[
        "--setenv".into(),
        "PATH".into(),
        // /opt/souffle/bin covers container images that install the
        // souffle binary under /opt. It is read-only-bound via the
        // /opt mount above, so adding it here just makes it
        // discoverable on the inner PATH.
        runtime
            .map(|r| r.search_path.clone())
            .unwrap_or_else(|| "/usr/local/bin:/usr/bin:/bin:/opt/souffle/bin".into()),
        "--setenv".into(),
        "HOME".into(),
        "/tmp".into(),
    ]);
    for k in FORWARDED_ENV {
        if let Ok(v) = env::var(k) {
            argv.push("--setenv".into());
            argv.push((*k).into());
            argv.push(v);
        }
    }
    for (k, v) in extra_env {
        argv.push("--setenv".into());
        argv.push((*k).into());
        argv.push((*v).into());
    }
    argv.push("--".into());
    let program = runtime
        .and_then(|r| r.tool(program))
        .unwrap_or(Path::new(program));
    argv.push(program.to_string_lossy().into_owned());
    argv.extend(args.iter().map(|s| (*s).into()));
    Ok(argv)
}

/// Build a `Command` for `program args…` that runs inside bwrap +
/// prlimit when available, falling back to direct invocation
/// otherwise.
///
/// Compile-chain defaults (4 GiB AS, 600 s CPU, 256 MB FSIZE) are
/// applied automatically. Use `build_bwrap_argv` directly if you
/// need a different limit profile (e.g. the long-running evaluator).
#[cfg(feature = "compiler")]
pub fn sandboxed_command(
    program: &str,
    args: &[&str],
    work_dir: &Path,
    work_writable: bool,
    ro_binds: &[&Path],
) -> Result<Command, String> {
    crate::nix_runtime::configured()?;
    check_sandbox()?;
    if !has_bubblewrap() {
        let mut cmd = Command::new(program);
        cmd.args(args);
        return Ok(cmd);
    }
    let limits = ResourceLimits::compile();
    let argv = build_bwrap_argv(
        program,
        args,
        work_dir,
        work_writable,
        ro_binds,
        &[],
        Some(&limits),
        // Compile chain blocks on `.output()` to completion; the
        // creating thread is still alive when the child exits.
        true,
        // No pre-exec filter on the compile chain: the toolchain's own
        // syscall needs are not the evaluator's, and the pre-exec filter
        // applies only to the evaluator spawn.
        None,
    )?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Return the values that follow each occurrence of `flag` in `argv`.
    fn flag_values<'a>(argv: &'a [String], flag: &str, arity: usize) -> Vec<&'a [String]> {
        let mut hits = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            if argv[i] == flag && i + arity < argv.len() {
                hits.push(&argv[i + 1..i + 1 + arity]);
                i += arity + 1;
            } else {
                i += 1;
            }
        }
        hits
    }

    fn argv_for(work_writable: bool) -> Vec<String> {
        build_bwrap_argv(
            "/tmp/work/souffle-evaluator",
            &["policy_program"],
            &PathBuf::from("/tmp/work"),
            work_writable,
            &[],
            &[],
            None,
            true,
            None,
        )
        .unwrap()
    }

    /// Only the exact `0` switches the evaluator's sandbox off — an unset
    /// variable, or any other value, leaves it on. The functor gate reads the
    /// same rule through [`sandbox_available`], so an operator who takes the
    /// escape hatch is not also silently admitting non-admin C++ as confined.
    #[test]
    fn only_a_literal_zero_switches_the_evaluator_sandbox_off() {
        assert!(evaluator_bwrap_off(Some("0")));
        assert!(!evaluator_bwrap_off(None));
        assert!(!evaluator_bwrap_off(Some("1")));
        assert!(!evaluator_bwrap_off(Some("")));
        assert!(!evaluator_bwrap_off(Some("false")));
    }

    #[test]
    fn evaluator_argv_uses_ro_bind_for_workspace() {
        let argv = argv_for(false);
        let ro = flag_values(&argv, "--ro-bind", 2);
        assert!(
            ro.iter()
                .any(|v| v[0] == "/tmp/work" && v[1] == "/tmp/work"),
            "evaluator workspace must be RO-bound at the same path",
        );
        assert!(
            !argv
                .windows(2)
                .any(|w| w[0] == "--bind" && w[1] == "/tmp/work"),
            "evaluator workspace must not be RW-bound",
        );
    }

    #[test]
    fn compile_argv_uses_rw_bind_for_workspace() {
        let argv = argv_for(true);
        let bind = flag_values(&argv, "--bind", 2);
        assert!(
            bind.iter()
                .any(|v| v[0] == "/tmp/work" && v[1] == "/tmp/work"),
            "compile workspace must be RW-bound at the same path",
        );
        let chdir = flag_values(&argv, "--chdir", 1);
        assert_eq!(chdir.len(), 1);
        assert_eq!(chdir[0][0], "/tmp/work");
    }

    #[test]
    fn argv_has_hardening_flags() {
        let argv = argv_for(false);
        for flag in ["--unshare-all", "--die-with-parent", "--clearenv"] {
            assert!(argv.contains(&flag.to_string()), "missing {flag}");
        }
        // Server-controlled paths must not appear in the sandbox.
        assert!(!argv.iter().any(|a| a == "/home"));
        assert!(!argv.iter().any(|a| a == "/root"));
    }

    /// The evaluator's spawn hands bwrap the deny-list on a descriptor, and
    /// the flag has to name that exact number for bwrap to read it.
    #[test]
    fn argv_names_the_seccomp_descriptor_it_was_given() {
        let argv = build_bwrap_argv(
            "/tmp/work/souffle-evaluator",
            &[],
            &PathBuf::from("/tmp/work"),
            false,
            &[],
            &[],
            None,
            false,
            Some(7),
        )
        .unwrap();
        let at = argv
            .iter()
            .position(|a| a == "--seccomp")
            .expect("--seccomp");
        assert_eq!(argv[at + 1], "7");
        assert!(
            at < argv.iter().rposition(|a| a == "--").unwrap(),
            "the flag belongs to bwrap, before the program terminator",
        );
    }

    /// Without a descriptor the flag must be absent entirely: a bubblewrap
    /// that predates `--seccomp` refuses an argv carrying it, which would
    /// turn a missing pre-exec filter into an evaluator that cannot start.
    #[test]
    fn argv_omits_the_seccomp_flag_when_there_is_no_descriptor() {
        assert!(!argv_for(false).iter().any(|a| a == "--seccomp"));
    }

    #[test]
    fn the_bwrap_version_is_read_out_of_its_own_line() {
        assert_eq!(parse_bwrap_version("bubblewrap 0.8.0\n"), Some((0, 8)));
        assert_eq!(parse_bwrap_version("bubblewrap 0.4\n"), Some((0, 4)));
        assert_eq!(parse_bwrap_version("bubblewrap 1.0.0-rc1\n"), Some((1, 0)));
    }

    /// Anything that is not a version is not read as one: the caller then
    /// leaves `--seccomp` off rather than guessing that an unknown build
    /// takes it.
    #[test]
    fn an_unreadable_version_line_is_no_version() {
        assert_eq!(parse_bwrap_version(""), None);
        assert_eq!(parse_bwrap_version("bubblewrap unknown\n"), None);
    }

    /// The minimum is the release that added `--seccomp FD`; 0.3.x must not
    /// be admitted by an off-by-one in the comparison.
    #[test]
    fn only_bubblewrap_0_4_and_newer_takes_the_program_on_a_descriptor() {
        assert!((0, 4) >= BWRAP_SECCOMP_FD_MIN_VERSION);
        assert!((0, 8) >= BWRAP_SECCOMP_FD_MIN_VERSION);
        assert!((1, 0) >= BWRAP_SECCOMP_FD_MIN_VERSION);
        assert!((0, 3) < BWRAP_SECCOMP_FD_MIN_VERSION);
    }

    #[test]
    fn argv_omits_die_with_parent_when_disabled() {
        // The long-lived evaluator path must NOT emit
        // `--die-with-parent`.
        let argv = build_bwrap_argv(
            "/tmp/work/souffle-evaluator",
            &[],
            &PathBuf::from("/tmp/work"),
            false,
            &[],
            &[],
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            !argv.iter().any(|a| a == "--die-with-parent"),
            "die_with_parent=false must suppress --die-with-parent",
        );
        // Other hardening flags still emitted.
        assert!(argv.contains(&"--unshare-all".to_string()));
        assert!(argv.contains(&"--clearenv".to_string()));
    }

    #[test]
    fn argv_terminator_precedes_program() {
        let argv = argv_for(false);
        let term = argv.iter().rposition(|a| a == "--").unwrap();
        assert_eq!(
            &argv[term + 1..],
            &["/tmp/work/souffle-evaluator", "policy_program"]
        );
    }

    #[test]
    fn probe_argv_matches_real_sandbox_flags() {
        // Probe loopback setup with the real isolation flags, so unsupported
        // sandbox configurations fail before compilation starts.
        let argv = probe_argv(&PathBuf::from("/tmp/work"), Some(7)).unwrap();
        assert_eq!(argv.first().map(String::as_str), Some("bwrap"));
        assert_eq!(argv.last().map(String::as_str), Some("true"));
        for flag in [
            "--unshare-all",
            "--die-with-parent",
            "--clearenv",
            "--proc",
            "--dev",
        ] {
            assert!(argv.contains(&flag.to_string()), "probe missing {flag}");
        }
        // The probe must not bind the whole host root: that form passes on
        // kernels where the real sandbox fails.
        assert!(
            !argv
                .windows(3)
                .any(|w| w[0] == "--dev-bind" && w[1] == "/" && w[2] == "/"),
            "probe must not bind the whole host root",
        );
        // And the flag the real evaluator spawn passes: without it the probe
        // passes on a host where every evaluator spawn dies inside bwrap,
        // which treats any seccomp problem as fatal.
        assert!(
            argv.windows(2).any(|w| w[0] == "--seccomp" && w[1] == "7"),
            "probe must hand bwrap the deny-list descriptor it was given",
        );
        assert!(
            !probe_argv(&PathBuf::from("/tmp/work"), None)
                .unwrap()
                .contains(&"--seccomp".to_string()),
            "and must leave the flag off when there is no descriptor",
        );
    }

    /// The host this exists for: bwrap creates namespaces but will not take
    /// the deny-list — no `CONFIG_SECCOMP_FILTER`, or a container or LSM
    /// policy refusing `prctl(PR_SET_SECCOMP)`. Before the pre-exec filter
    /// that host ran fine, so it must keep running: spawn without the flag,
    /// say so in the status, and name the reason once.
    #[test]
    fn a_host_that_refuses_the_filter_still_spawns_the_evaluator_without_it() {
        let refused = "bwrap: Can't read seccomp data: Bad file descriptor";
        let probe = BwrapProbe::from_runs(Some(Err(refused.into())), || true);

        assert!(
            probe.bwrap_usable(),
            "bwrap itself works here, so the compile chain must not fail closed"
        );
        assert!(!probe.seccomp_flag_usable());
        assert!(
            !seccomp_pre_exec_with(&probe),
            "every status the binary reports must say the filter is not installed before exec"
        );

        let argv = build_bwrap_argv(
            "/opt/evaluator",
            &[],
            &PathBuf::from("/tmp/work"),
            false,
            &[],
            &[],
            None,
            false,
            probe.seccomp_flag_usable().then_some(7),
        )
        .unwrap();
        assert!(
            !argv.contains(&"--seccomp".to_string()),
            "the spawn must not pass a flag this bwrap dies on: {argv:?}"
        );

        let warning = seccomp_fallback_warning(&probe).expect("one warning naming the reason");
        assert!(
            warning.contains(refused),
            "the warning must quote bwrap: {warning}"
        );

        // A probe that never had a descriptor to offer (old bubblewrap, no
        // program for the architecture) is a different case and warns
        // elsewhere, in the version branch.
        assert!(seccomp_fallback_warning(&BwrapProbe::from_runs(None, || true)).is_none());
    }

    /// A host where bwrap cannot create namespaces at all is still unusable,
    /// filter or no filter: the fallback must not turn a broken sandbox into
    /// a working one.
    #[test]
    fn the_fallback_does_not_rescue_a_bwrap_that_cannot_run() {
        let probe = BwrapProbe::from_runs(Some(Err("bwrap: No permissions".into())), || false);
        assert!(!probe.bwrap_usable());
        assert!(!probe.seccomp_flag_usable());
    }

    /// The real probe against a descriptor bwrap cannot read: it must fail
    /// exactly where the evaluator spawn would, while the same probe without
    /// the flag succeeds — which is the pair the fallback decision is made
    /// from.
    ///
    /// Ignored by default: it needs a Linux host with a working bubblewrap,
    /// (Linux only). Run it by name.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "needs Linux and a working bubblewrap"]
    fn the_probe_fails_on_a_descriptor_bwrap_cannot_read() {
        let work_dir = std::env::temp_dir().join("sasy-bwrap-probe-unreadable");
        fs::create_dir_all(&work_dir).expect("a work dir");
        if run_probe(&work_dir, None).is_err() {
            eprintln!("skipped: bwrap cannot create namespaces here at all");
            return;
        }
        // Write-only: bwrap's read of the program fails, which is the shape
        // of every seccomp problem it treats as fatal.
        let fd = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
        assert!(fd >= 0, "opening /dev/null write-only");
        let unreadable = crate::seccomp::ProgramFd::from_raw_for_test(fd);
        let refused = run_probe(&work_dir, Some(&unreadable))
            .expect_err("bwrap must refuse a program it cannot read");
        let probe =
            BwrapProbe::from_runs(Some(Err(refused)), || run_probe(&work_dir, None).is_ok());
        assert!(probe.bwrap_usable());
        assert!(!probe.seccomp_flag_usable());
        assert!(!seccomp_pre_exec_with(&probe));
        eprintln!(
            "fallback warning: {}",
            seccomp_fallback_warning(&probe).expect("a warning")
        );
    }

    /// A fake bwrap accepts a minimal namespace probe but rejects the real
    /// sandbox arguments at loopback setup (RTM_NEWADDR). The usability check
    /// must use the same arguments as an actual compile and fail closed.
    #[cfg(unix)]
    #[test]
    fn probe_fails_on_a_bwrap_a_whole_root_probe_would_pass() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join("sasy-fake-bwrap-probe-test");
        fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("bwrap");
        // Exit 0 only for a whole-root `--dev-bind / /` argv; otherwise mimic
        // the kernel that rejects loopback setup for the real sandbox flags.
        fs::write(
            &fake,
            "#!/bin/sh\n\
             case \" $* \" in\n\
             \x20 *\" --dev-bind / / \"*) exit 0 ;;\n\
             esac\n\
             echo 'bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted' >&2\n\
             exit 1\n",
        )
        .unwrap();
        let mut perm = fs::metadata(&fake).unwrap().permissions();
        perm.set_mode(0o755);
        fs::set_permissions(&fake, perm).unwrap();

        // Run an argv (argv[0] == "bwrap") against the fake binary.
        let run = |argv: &[String]| {
            Command::new(&fake)
                .args(&argv[1..])
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };

        // A whole-root probe passes against this bwrap, which is why the real
        // probe must not use that form.
        let whole_root_probe = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--dev-bind",
            "/",
            "/",
            "true",
        ]
        .map(String::from);
        assert!(
            run(&whole_root_probe),
            "a whole-root probe passes against this bwrap",
        );

        // The real probe uses the real sandbox flags, so it FAILS like the
        // real run.
        let real_probe = probe_argv(&PathBuf::from("/tmp/work"), None).unwrap();
        assert!(
            !run(&real_probe),
            "the real probe must fail like the real compile does on this kernel",
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn argv_sets_restricted_path_and_home() {
        let argv = argv_for(false);
        let setenv: Vec<_> = flag_values(&argv, "--setenv", 2)
            .into_iter()
            .map(|v| (v[0].clone(), v[1].clone()))
            .collect();
        // PATH + HOME are replaced — no inheritance from the server.
        assert!(setenv.iter().any(|(k, _)| k == "PATH"));
        assert!(setenv.iter().any(|(k, _)| k == "HOME"));
        // Sensitive env must never appear — not in FORWARDED_ENV.
        assert!(!setenv.iter().any(|(k, _)| k == "ANTHROPIC_API_KEY"));
    }

    #[test]
    fn argv_with_limits_emits_prlimit_prefix_when_available() {
        // Skip on hosts without prlimit (e.g. macOS) — the prefix is
        // gated on prlimit being on PATH.
        if !path_lookup("prlimit") {
            return;
        }
        let limits = ResourceLimits::compile();
        let argv = build_bwrap_argv(
            "g++",
            &[],
            &PathBuf::from("/tmp/work"),
            true,
            &[],
            &[],
            Some(&limits),
            true,
            None,
        )
        .unwrap();
        assert_eq!(argv[0], "prlimit");
        // Must terminate prlimit args before bwrap.
        let term_pos = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(argv[term_pos + 1], "bwrap");
        // The expected limits show up as flags.
        assert!(argv.iter().any(|a| a.starts_with("--as=")));
        assert!(argv.iter().any(|a| a.starts_with("--cpu=")));
        assert!(argv.iter().any(|a| a.starts_with("--fsize=")));
    }

    #[test]
    fn evaluator_limits_omit_cpu_cap() {
        let l = ResourceLimits::evaluator();
        assert!(
            l.cpu_seconds.is_none(),
            "evaluator is long-running; no CPU cap"
        );
        assert!(l.address_space_bytes > 0);
        assert!(l.fsize_bytes > 0);
    }

    #[test]
    fn argv_propagates_extra_env() {
        let argv = build_bwrap_argv(
            "/tmp/work/souffle-evaluator",
            &[],
            &PathBuf::from("/tmp/work"),
            false,
            &[],
            &[("LD_LIBRARY_PATH", "/usr/local/lib")],
            None,
            true,
            None,
        )
        .unwrap();
        let setenv: Vec<_> = flag_values(&argv, "--setenv", 2)
            .into_iter()
            .map(|v| (v[0].clone(), v[1].clone()))
            .collect();
        assert!(
            setenv
                .iter()
                .any(|(k, v)| k == "LD_LIBRARY_PATH" && v == "/usr/local/lib"),
            "extra_env entries must be passed via --setenv inside bwrap; got {:?}",
            setenv,
        );
    }

    #[test]
    fn argv_propagates_extra_ro_binds() {
        let tmp = std::env::temp_dir().join("sasy-sandbox-test-toolkit");
        fs::create_dir_all(&tmp).unwrap();
        let toolkit_file = tmp.join("fake-shim.cpp");
        fs::write(&toolkit_file, b"// toolkit\n").unwrap();
        let argv = build_bwrap_argv(
            "g++",
            &[],
            &PathBuf::from("/tmp/work"),
            true,
            &[toolkit_file.as_path()],
            &[],
            None,
            true,
            None,
        )
        .unwrap();
        let tf = toolkit_file.to_str().unwrap().to_string();
        let ro: Vec<_> = flag_values(&argv, "--ro-bind", 2)
            .into_iter()
            .map(|v| (v[0].clone(), v[1].clone()))
            .collect();
        assert!(
            ro.iter().any(|(src, dst)| src == &tf && dst == &tf),
            "toolkit file should be RO-bound at the same path; got {:?}",
            ro,
        );
    }

    #[test]
    fn nix_closure_is_read_only_and_retains_every_isolation_boundary() {
        let runtime = crate::nix_runtime::NixRuntime::fixture();
        let argv = build_bwrap_argv_with_runtime(
            "true",
            &["an argument"],
            Path::new("/tmp/policy-work"),
            false,
            &[],
            &[("SASY_SECCOMP_PRE_EXEC", "1")],
            Some(&ResourceLimits::evaluator()),
            false,
            Some(19),
            Some(&runtime),
        )
        .unwrap();
        for flag in ["--unshare-all", "--clearenv", "--proc", "--dev", "--tmpfs"] {
            assert!(argv.iter().any(|arg| arg == flag), "missing {flag}");
        }
        assert_eq!(flag_values(&argv, "--seccomp", 1)[0][0], "19");
        let mounts = flag_values(&argv, "--ro-bind", 2);
        let root = runtime.store_paths[0].to_str().unwrap();
        assert!(mounts.iter().any(|pair| pair[0] == root && pair[1] == root));
        assert!(!mounts.iter().any(|pair| pair[0] == "/bin"));
        assert!(mounts.iter().any(
            |pair| pair[0] == runtime.tool("sh").unwrap().to_str().unwrap() && pair[1] == "/bin/sh"
        ));
        assert!(!mounts
            .iter()
            .any(|pair| pair[0] == "/nix" || pair[0] == "/nix/store"));
        assert!(flag_values(&argv, "--bind", 2).is_empty());
        let env = flag_values(&argv, "--setenv", 2);
        assert!(env
            .iter()
            .any(|pair| pair[0] == "PATH" && pair[1] == runtime.search_path));
        assert!(!env
            .iter()
            .any(|pair| pair[0] == "SASY_API_KEY" || pair[0] == "SASY_NIX_RUNTIME_MANIFEST"));
        assert_eq!(argv[0], runtime.tool("prlimit").unwrap().to_str().unwrap());
        assert!(argv
            .iter()
            .any(|arg| arg == runtime.tool("bwrap").unwrap().to_str().unwrap()));
        assert_eq!(
            argv[argv.len() - 2],
            runtime.tool("true").unwrap().to_str().unwrap()
        );
        assert_eq!(argv.last().unwrap(), "an argument");
    }

    #[test]
    fn nix_mount_and_path_expansion_fail_before_constructing_a_command() {
        let runtime = crate::nix_runtime::NixRuntime::fixture();
        assert!(build_bwrap_argv_with_runtime(
            "true",
            &[],
            Path::new("/tmp/work"),
            true,
            &[Path::new("/nix/store")],
            &[],
            None,
            true,
            None,
            Some(&runtime)
        )
        .is_err());
        assert!(build_bwrap_argv_with_runtime(
            "true",
            &[],
            Path::new("/tmp/work"),
            true,
            &[],
            &[("PATH", "/untrusted/bin")],
            None,
            true,
            None,
            Some(&runtime)
        )
        .is_err());
    }

    /// Nix coreutils' `true` is a
    /// symlink to its multicall executable. Resolving the invocation name to
    /// `coreutils` makes the real capability probe exit1 despite working
    /// namespaces. This test deliberately cannot turn failure into a skip.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires an installed trusted Nix runtime manifest and usable namespaces"]
    fn the_package_probe_preserves_its_multicall_launcher_name() {
        let runtime = crate::nix_runtime::configured()
            .unwrap()
            .expect("package manifest required");
        let launcher = runtime.tool("true").expect("pinned true launcher");
        assert_eq!(launcher.file_name().unwrap(), "true");
        let target = fs::canonicalize(launcher).unwrap();
        assert_ne!(
            launcher, target,
            "fixture must exercise a multicall symlink"
        );
        let direct = Command::new(launcher).output().unwrap();
        assert!(
            direct.status.success(),
            "{}",
            String::from_utf8_lossy(&direct.stderr)
        );
        let work = tempfile::tempdir().unwrap();
        run_probe(work.path(), None).expect("real Nix closure probe must execute true");
        let filter = crate::seccomp::deny_list_program_fd().expect("Linux deny-list descriptor");
        run_probe(work.path(), Some(&filter))
            .expect("real Nix closure probe must execute true with the pre-exec deny-list");
    }

    /// Exercise libc's hardcoded /bin/sh preprocessor path and the following
    /// native stages through the production builder, not a hand-built bwrap.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires an installed trusted Nix runtime manifest and usable namespaces"]
    fn the_package_generates_compiles_and_interprets_a_policy() {
        let runtime = crate::nix_runtime::configured()
            .unwrap()
            .expect("package manifest required");
        let work = tempfile::tempdir().unwrap();
        let source = work.path().join("policy.dl");
        fs::write(
            &source,
            ".decl Value(x:number)\nValue(41).\n.output Value\n",
        )
        .unwrap();
        let cpp = work.path().join("policy.cpp");
        let binary = work.path().join("policy");
        let stage = |program: &str, args: &[&str]| {
            let argv = build_bwrap_argv(
                program,
                args,
                work.path(),
                true,
                &[],
                &[],
                Some(&ResourceLimits::compile()),
                true,
                None,
            )
            .unwrap();
            let out = Command::new(&argv[0]).args(&argv[1..]).output().unwrap();
            assert!(
                out.status.success(),
                "{program}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            out.stdout
        };
        stage("/bin/sh", &["-c", "true"]);
        stage(
            "souffle",
            &["-g", cpp.to_str().unwrap(), source.to_str().unwrap()],
        );
        let include = runtime
            .tool("souffle")
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("include");
        assert!(include.join("souffle/CompiledSouffle.h").is_file());
        stage(
            "g++",
            &[
                "-std=c++17",
                "-O1",
                "-pthread",
                "-I",
                include.to_str().unwrap(),
                cpp.to_str().unwrap(),
                "-o",
                binary.to_str().unwrap(),
            ],
        );
        for (backend, program) in [
            ("compiled", binary.to_str().unwrap()),
            ("interpreted", "souffle"),
        ] {
            let output = work.path().join(backend);
            fs::create_dir(&output).unwrap();
            let mut args = vec![
                "-F",
                work.path().to_str().unwrap(),
                "-D",
                output.to_str().unwrap(),
            ];
            if backend == "interpreted" {
                args.push(source.to_str().unwrap());
            }
            stage(program, &args);
            assert_eq!(
                fs::read_to_string(output.join("Value.csv")).unwrap(),
                "41\n",
                "{backend}"
            );
        }

        // Use the installed common schema and native JSON FFI too: a plain
        // Datalog program alone does not exercise SASY's C++ support assets.
        let assets = runtime
            .tool("souffle-interpreted")
            .unwrap()
            .parent()
            .unwrap();
        let common = fs::read_to_string(assets.join("common_policy.dl")).unwrap();
        let common = common
            .lines()
            .filter(|line| !line.starts_with(".input ") && !line.starts_with(".output "))
            .collect::<Vec<_>>()
            .join("\n");
        let raw = work.path().join("common-raw.dl");
        let common_source = work.path().join("common.dl");
        fs::write(&raw, format!(r#"{common}
// USER_POLICY_BEGIN
Principal("test-principal").
IsAuthorized(idx) :- Actions(idx, $CallTool("Read", args)), @json_get_str(args, "file_path") = "notes.txt".
IsAuthorized(idx) :- Actions(idx, action), IsTool(action, "Write").
Unauthorized(idx) :- Actions(idx, action), IsTool(action, "Write").
Actions(0, $CallTool("Read", "{{\"file_path\":\"notes.txt\"}}" )).
Actions(1, $CallTool("Read", "{{\"file_path\":\"other.txt\"}}" )).
Actions(2, $CallTool("Write", "{{}}" )).
.output IsAuthorized
.output Authorized
.output Unauthorized
"#)).unwrap();
        let desugared = stage(
            "python3",
            &[
                assets.join("sugar.py").to_str().unwrap(),
                raw.to_str().unwrap(),
            ],
        );
        fs::write(&common_source, desugared).unwrap();
        let common_cpp = work.path().join("common.cpp");
        let common_binary = work.path().join("common");
        stage(
            "souffle",
            &[
                "-g",
                common_cpp.to_str().unwrap(),
                common_source.to_str().unwrap(),
            ],
        );
        stage(
            "g++",
            &[
                "-std=c++17",
                "-O1",
                "-pthread",
                "-DRAM_DOMAIN_SIZE=64",
                "-I",
                include.to_str().unwrap(),
                common_cpp.to_str().unwrap(),
                assets.join("functors_common.cpp").to_str().unwrap(),
                "-o",
                common_binary.to_str().unwrap(),
            ],
        );
        // Compile the production IPC shim too. Its runtime protocol is covered
        // by the installed-engine qualification, not this CSV smoke fixture.
        stage(
            "g++",
            &[
                "-std=c++17",
                "-O2",
                "-D__EMBEDDED_SOUFFLE__",
                "-DRAM_DOMAIN_SIZE=64",
                "-I",
                include.to_str().unwrap(),
                assets.join("evaluator_shim.cpp").to_str().unwrap(),
                common_cpp.to_str().unwrap(),
                assets.join("functors_common.cpp").to_str().unwrap(),
                "-o",
                work.path().join("evaluator").to_str().unwrap(),
                "-lpthread",
            ],
        );
        for (backend, program) in [
            ("compiled-common", common_binary.to_str().unwrap()),
            ("interpreted-common", "souffle"),
        ] {
            let output = work.path().join(backend);
            fs::create_dir(&output).unwrap();
            let mut args = vec![
                "-F",
                work.path().to_str().unwrap(),
                "-D",
                output.to_str().unwrap(),
            ];
            if backend == "interpreted-common" {
                args.extend([
                    "-L",
                    assets.to_str().unwrap(),
                    "-l",
                    "functors",
                    common_source.to_str().unwrap(),
                ]);
            }
            stage(program, &args);
            assert_eq!(
                fs::read_to_string(output.join("IsAuthorized.csv")).unwrap(),
                "0\n2\n",
                "{backend}"
            );
            assert_eq!(
                fs::read_to_string(output.join("Authorized.csv")).unwrap(),
                "0\n",
                "{backend}"
            );
            assert_eq!(
                fs::read_to_string(output.join("Unauthorized.csv")).unwrap(),
                "2\n",
                "{backend}"
            );
        }
    }
}
