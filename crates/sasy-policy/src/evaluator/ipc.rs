//! Length-prefixed JSON IPC over stdin/stdout pipes.
//!
//! Wire format: 4-byte big-endian payload length, 8-byte big-endian request
//! id, then the JSON payload. All messages are newline-free JSON
//! (serde_json::to_vec).
//!
//! The id is what makes a request abandonable. A caller that gives up on a
//! reply — the query budget in [`crate::session_evaluator`] — leaves the
//! evaluator still computing, and its answer arrives later; without an id the
//! next call would read that late answer as its own and decide one action on
//! another action's evaluation. The evaluator echoes the id it was sent, and
//! [`IpcChild::read_reply`] discards (and logs) every frame that is not the
//! one it is waiting for.
//!
//! The Soufflé shim on the other end of this pipe is compiled per policy by
//! this binary from `souffle`, so the format needs no
//! compatibility story: a mismatched pair is a build to redo, not a version to
//! negotiate. An external FlowLog evaluator (`--evaluator flowlog`, not part of
//! this repository) speaks the same protocol.

use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::{debug, error, info, warn};

use super::types::{EvalRequest, EvalResponse};
use super::{EvaluatorError, OracleClock};
use std::env;

use crate::sandbox::{build_bwrap_argv, sandbox_available, ResourceLimits};

/// Hard cap on a single length-prefixed IPC frame from the evaluator
/// subprocess. The evaluator is the *confined / untrusted* side — it links
/// caller-supplied functor C++ — so the host must not allocate an
/// attacker-chosen size: a malicious functor could emit a 4 GiB length
/// prefix and OOM the host. 64 MiB is far above any legitimate eval frame.
const MAX_IPC_FRAME: usize = 64 * 1024 * 1024;

/// Width of the request-id header that follows every length prefix.
const FRAME_ID_BYTES: usize = 8;

/// How much is read from the child's pipe per syscall while a frame is being
/// assembled. Frames are small (a query and its answer are kilobytes), so this
/// is one read for the common case and a loop for a bulk update.
const READ_CHUNK: usize = 64 * 1024;

/// Scrub the two strings an `LlmQuery` frame carries, before anything reads
/// them.
///
/// `@llm_check_fn(prompt, context)` is the one place the engine sends recorded
/// content off the host: the context is usually a message's contents, and both
/// strings travel to an external model provider. Whatever the redactor
/// recognizes is replaced by a marker here, at the frame, so the log preview,
/// the cache key and the provider all see the scrubbed text and no caller can
/// forget to ask.
///
/// Best effort by construction: a secret in a shape
/// [`sasy_redaction`] does not know goes through. With the switch off
/// ([`crate::oracle_redaction`]) the strings pass through untouched.
///
/// The counts are logged, never the values.
fn redact_oracle_strings(prompt: &str, context: &str) -> (String, String) {
    if !crate::oracle_redaction::oracle_redaction_enabled() {
        return (prompt.to_string(), context.to_string());
    }
    let (prompt, prompt_hits) = sasy_redaction::redact_text(prompt);
    let (context, context_hits) = sasy_redaction::redact_text(context);
    debug!(
        prompt_redactions = prompt_hits,
        context_redactions = context_hits,
        "[LLM oracle] scrubbed the query before it left the process"
    );
    (prompt, context)
}

/// Wraps a child process with framed IPC over stdin/stdout.
///
/// Both buffers live on the struct rather than inside the read/write calls so
/// that abandoning a call mid-frame is survivable: the query budget drops the
/// future at whatever await it happens to be sitting on, and the bytes already
/// moved are recorded here, so the next call resumes the frame instead of
/// reading a payload as a length prefix.
pub struct IpcChild {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    /// Frame bytes serialized but not yet accepted by the pipe. Non-empty only
    /// after a write was abandoned; the next write drains it first, which
    /// finishes the abandoned request so the evaluator can answer (and the
    /// answer be discarded) rather than reading two frames spliced together.
    out_buf: Vec<u8>,
    /// Bytes read from the child that do not yet form a whole frame.
    in_buf: Vec<u8>,
    /// Landing buffer for one `read` syscall, reused so the per-query path
    /// allocates nothing. Its contents are meaningful only inside the read
    /// that filled it.
    scratch: Vec<u8>,
    /// Set once the byte stream can no longer be trusted to start on a frame
    /// boundary. A framing violation is fatal to the channel, not to the one
    /// call that saw it: the bad header stays at the head of `in_buf` (or has
    /// already been half-consumed), so every later read would fail on the same
    /// bytes forever and the session would be refused for good — alive by
    /// every check the supervisor makes, so never killed and never respawned.
    /// Latched here so every subsequent call reports `ProcessDied` instead,
    /// which is the path that already evicts and respawns.
    poisoned: bool,
    /// Test seam: the answer [`IpcChild::liveness`] gives instead of asking the
    /// kernel. `try_wait` cannot be made to fail on demand, so `Unknown` — the
    /// case in which nothing was reaped and the pid is still ours — is
    /// otherwise unreachable from a test, and the code that has to treat it
    /// differently from a reap could not be exercised at all.
    #[cfg(test)]
    forced_liveness: Option<ChildLiveness>,
}

/// The answer to "is the child still running?", kept in three parts.
///
/// `Reaped` is the only outcome in which the child's pid has been returned to
/// the kernel and may be recycled; `Unknown` means the question failed and the
/// pid is unchanged. Anything that retires a pid from the shutdown kill list
/// keys on `Reaped` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildLiveness {
    /// The child is still running.
    Running,
    /// The child exited and this call reaped it; its pid is now free.
    Reaped,
    /// Asking failed. Nothing was reaped, so the pid is still ours.
    Unknown,
}

impl IpcChild {
    /// Spawn a child process with stdin/stdout pipes.
    ///
    /// When bwrap is available, the evaluator runs inside a
    /// stripped sandbox from `_start` onward —
    /// `__attribute__((constructor))` and static-storage initializers
    /// in caller-supplied functor code see the same restricted FS
    /// view that the in-process seccomp filter imposes post-`main`.
    /// Set `SASY_EVALUATOR_BWRAP=0` to opt out (the evaluator then
    /// runs with only its in-process sandbox). The functor gate reads
    /// that switch too — with it
    /// set, the host counts as having no sandbox and a non-admin's
    /// C++ is refused even under `--allow-user-functors`.
    ///
    /// `--die-with-parent` is intentionally **off** for this bwrap
    /// (passed as `false` to `build_bwrap_argv`). bwrap implements
    /// the flag via `prctl(PR_SET_PDEATHSIG, SIGKILL)`, which fires
    /// on **thread** exit, not process exit. The evaluator is
    /// spawned from `tokio::task::spawn_blocking` (the
    /// `SetPolicy` install path), and tokio's blocking-pool
    /// `thread_keep_alive` defaults to 10 s — so PDEATHSIG would
    /// SIGKILL the wrapped evaluator ~10 s after each upload.
    /// `Child::kill_on_drop(true)` already handles Rust-side
    /// cleanup on normal shutdown.
    ///
    /// The compile-chain bwrap (the file-read-oracle fix) keeps
    /// `--die-with-parent` on — those subprocesses block on
    /// `.output()` to completion, well before the creating thread
    /// can park.
    ///
    /// `env_overrides` are applied scoped to the child only — used
    /// for backends that need `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`
    /// without leaking the override into the server's own env.
    pub fn spawn(
        program: &str,
        args: &[&str],
        env_overrides: &[(String, String)],
    ) -> Result<Self, EvaluatorError> {
        // Default: wrap in bwrap when bwrap is on PATH.
        // Escape hatch: SASY_EVALUATOR_BWRAP=0 disables (e.g. for
        // debugging or for a platform where bwrap misbehaves).
        let allow_bwrap = env::var("SASY_EVALUATOR_BWRAP").ok().as_deref() != Some("0");
        Self::spawn_maybe_wrapped(program, args, env_overrides, allow_bwrap)
    }

    /// Admit only the backend's explicitly selected runtime files. A policy
    /// filename in argv or a library search path is not itself a sandbox bind.
    pub(crate) fn spawn_with_read_only_files(
        program: &str,
        args: &[&str],
        env_overrides: &[(String, String)],
        files: &[std::path::PathBuf],
    ) -> Result<Self, EvaluatorError> {
        if files.is_empty() {
            return Self::spawn(program, args, env_overrides);
        }
        crate::sandbox::log_seccomp_pre_exec_status();
        let allow_bwrap = env::var("SASY_EVALUATOR_BWRAP").ok().as_deref() != Some("0");
        let paths: Vec<&Path> = files.iter().map(std::path::PathBuf::as_path).collect();
        Self::spawn_with_runtime_files(
            program,
            args,
            env_overrides,
            allow_bwrap,
            crate::sandbox::seccomp_pre_exec_available(),
            &paths,
        )
    }

    /// [`IpcChild::spawn`] with the wrapper decision made by the caller.
    ///
    /// The tests need both shapes on one host: the direct spawn is the only
    /// one whose child pids are host pids (under bwrap they are pids in the
    /// sandbox's own namespace), and the wrapped spawn is the one the kill
    /// path was actually built for. Taking the decision as an argument keeps
    /// both reachable without mutating the process's environment, which is
    /// shared by every test in the binary.
    fn spawn_maybe_wrapped(
        program: &str,
        args: &[&str],
        env_overrides: &[(String, String)],
        allow_bwrap: bool,
    ) -> Result<Self, EvaluatorError> {
        crate::sandbox::log_seccomp_pre_exec_status();
        Self::spawn_maybe_wrapped_with_seccomp(
            program,
            args,
            env_overrides,
            allow_bwrap,
            crate::sandbox::seccomp_pre_exec_available(),
        )
    }

    /// [`IpcChild::spawn_maybe_wrapped`] with the pre-exec filter decided by
    /// the caller too.
    ///
    /// Same reason the wrapper decision is an argument: the probe test needs
    /// both spawns on one host, and the alternative — setting
    /// `SASY_SECCOMP_PRE_EXEC_DISABLE` for one of them — mutates an
    /// environment every other test in the binary shares.
    fn spawn_maybe_wrapped_with_seccomp(
        program: &str,
        args: &[&str],
        env_overrides: &[(String, String)],
        allow_bwrap: bool,
        pre_exec_seccomp: bool,
    ) -> Result<Self, EvaluatorError> {
        Self::spawn_with_runtime_files(
            program,
            args,
            env_overrides,
            allow_bwrap,
            pre_exec_seccomp,
            &[],
        )
    }

    fn spawn_with_runtime_files(
        program: &str,
        args: &[&str],
        env_overrides: &[(String, String)],
        allow_bwrap: bool,
        pre_exec_seccomp: bool,
        read_only_files: &[&Path],
    ) -> Result<Self, EvaluatorError> {
        // A malformed package manifest must not turn sandbox_available=false
        // into an unconfined evaluator launch, even for an explicit opt-out.
        if crate::nix_runtime::configured()
            .map_err(EvaluatorError::InitError)?
            .is_some()
        {
            crate::sandbox::check_sandbox().map_err(EvaluatorError::InitError)?;
        }
        info!("Spawning evaluator: {} {:?}", program, args);

        let env_pairs: Vec<(&str, &str)> = env_overrides
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Default: wrap in bwrap when bwrap is on PATH and the caller has not
        // asked for the naked spawn.
        //
        // `sandbox_available` is the same question the functor gate asks
        // before admitting a non-admin's C++, and asking it through the same
        // function is deliberate: two switches read in two places drift, and
        // the drift admits an upload as confined that this line then runs
        // naked. It carries the `SASY_EVALUATOR_BWRAP=0` escape hatch, so the
        // env read in [`IpcChild::spawn`] and this call agree by construction.
        let wrap_in_bwrap = allow_bwrap && sandbox_available();

        // Held until after the spawn: bwrap reads the program off this
        // descriptor in the child, and the parent has no further use for it
        // once the child exists.
        let mut seccomp_program: Option<crate::seccomp::ProgramFd> = None;

        let mut cmd = if wrap_in_bwrap {
            // Wrap the evaluator in bwrap + pass SASY_SKIP_IN_PROC_SANDBOX=1
            // via bwrap's --setenv. The evaluator's own in-process
            // chroot + netns-unshare + seccomp BPF filter was sized
            // for the naked-spawn environment, and layering it on top
            // of bwrap's altered glibc environment hits syscalls
            // outside the 35-entry allowlist
            // (SECCOMP_RET_KILL_PROCESS → mid-traffic SIGKILL). bwrap
            // provides strictly stronger isolation than chroot +
            // unshare anyway, so the in-process layer is redundant
            // when we're already confining from the outside.
            //
            // The one thing bwrap cannot delegate back is the syscall filter's
            // TIMING. The shim installs its copy from `main`, and the functor
            // C++ linked into the evaluator has already run its static
            // constructors by then. So the filter is built here and handed to
            // bwrap on a descriptor, which installs it after the namespaces and
            // before the `execve` — see [`crate::seccomp`]. On a bubblewrap too
            // old for `--seccomp FD` (or a host with no program for the
            // architecture) `seccomp_fd` is `None`, the shim's copy is all
            // there is, and [`crate::sandbox::log_seccomp_pre_exec_status`]
            // says so once at startup.
            seccomp_program = if pre_exec_seccomp {
                crate::seccomp::deny_list_program_fd()
            } else {
                None
            };
            let seccomp_fd = seccomp_program.as_ref().map(|p| p.raw());
            let mut bwrap_env = env_pairs.clone();
            bwrap_env.push(("SASY_SKIP_IN_PROC_SANDBOX", "1"));
            if seccomp_fd.is_some() {
                // Tells the shim which of the two cases it is in, so its log
                // line does not read as if it were the first filter installed.
                bwrap_env.push(("SASY_SECCOMP_PRE_EXEC", "1"));
            }
            let work_dir = Path::new(program)
                .parent()
                .unwrap_or_else(|| Path::new("/"));
            let argv = build_bwrap_argv(
                program,
                args,
                work_dir,
                false,
                read_only_files,
                &bwrap_env,
                // Cap address space (2 GB) + max file size (64 MB) via prlimit
                // so a hostile user functor can't OOM or fill the disk on the
                // host. No CPU cap here by design — the evaluator is long-lived
                // and serves many queries; CPU-spin protection belongs at the
                // IPC layer (a per-query watchdog), tracked separately.
                Some(&ResourceLimits::evaluator()),
                // Long-lived evaluator: PDEATHSIG would fire when the
                // tokio blocking-pool thread that spawned us parks.
                // Rely on kill_on_drop(true) for cleanup.
                false,
                seccomp_fd,
            )
            .map_err(EvaluatorError::InitError)?;
            let mut c = Command::new(&argv[0]);
            c.args(&argv[1..]);
            c
        } else {
            // No host-side bwrap, so the evaluator's in-process
            // sandbox is the only runtime isolation available.
            // Leave SASY_SKIP_IN_PROC_SANDBOX unset so it fires.
            let mut c = Command::new(program);
            c.args(args);
            for (k, v) in env_overrides {
                c.env(k, v);
            }
            c
        };
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        // Its own process group, so [`IpcChild::kill`] can signal the whole
        // tree rather than the process we happen to hold a handle to. Under
        // bwrap that handle is the OUTER wrapper: `--unshare-all` includes
        // `--unshare-pid`, so the evaluator runs as init inside a new PID
        // namespace and, with `--die-with-parent` off (see above), survives
        // the wrapper's death — a functor spinning in a loop never reads its
        // stdin either, so the pipe closing does not stop it. Left alone it
        // would burn a core and hold its 2 GB of address space for as long as
        // the host lives, one leaked process per stall. The group signal is
        // what makes the stall window an actual bound, which is what
        // `ResourceLimits::evaluator` points at when it declines to set a
        // cumulative CPU cap.
        #[cfg(unix)]
        cmd.process_group(0);

        // The program descriptor is `FD_CLOEXEC` in the parent, so no other
        // fork in this process — a concurrent evaluator spawn, anything else
        // the binary starts — ever sees it. This child is the one that needs
        // it, so the flag comes off here, after the fork and before the
        // `execve`, where nothing else can observe the number. A failure to
        // clear it means bwrap would find no program on that descriptor: fail
        // the spawn rather than exec an evaluator whose constructors run
        // unfiltered.
        let carries_seccomp_program = seccomp_program.is_some();
        #[cfg(unix)]
        if let Some(fd) = seccomp_program.as_ref().map(|p| p.raw()) {
            // SAFETY: the closure runs between `fork` and `execve` in the
            // child, which is where `inherit_in_child` may be called; the
            // parent holds the descriptor open across `spawn`, and neither
            // the call nor the error it returns allocates or locks —
            // `from_raw_os_error` stores the number and nothing else, which
            // is why the reason is worded by the parent below rather than
            // built here.
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

        let spawned = cmd.spawn();
        // The child has it (or the spawn failed); either way the parent's copy
        // goes now rather than at the end of the function.
        drop(seccomp_program);
        let mut child = spawned.map_err(|e| {
            // The child cannot say more than an errno without allocating, so
            // `EBADF` out of a spawn that carried a deny-list descriptor is
            // read here as the `pre_exec` hook above failing to clear
            // `FD_CLOEXEC`.
            let reason = if carries_seccomp_program && e.raw_os_error() == Some(libc::EBADF) {
                "could not hand the seccomp deny-list to bwrap".to_string()
            } else {
                e.to_string()
            };
            EvaluatorError::InitError(format!("Failed to spawn {}: {}", program, reason))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EvaluatorError::InitError("No stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EvaluatorError::InitError("No stdout".into()))?;

        Ok(Self {
            child,
            stdin,
            stdout,
            out_buf: Vec::new(),
            in_buf: Vec::new(),
            scratch: vec![0u8; READ_CHUNK],
            poisoned: false,
            #[cfg(test)]
            forced_liveness: None,
        })
    }

    /// Make [`IpcChild::liveness`] answer `liveness` from now on. Tests only.
    #[cfg(test)]
    pub(crate) fn force_liveness(&mut self, liveness: ChildLiveness) {
        self.forced_liveness = Some(liveness);
    }

    /// Send a request under `id` and wait for the reply carrying that id
    /// (no oracle support).
    pub async fn call(
        &mut self,
        id: u64,
        request: &EvalRequest,
    ) -> Result<EvalResponse, EvaluatorError> {
        self.check_not_poisoned()?;
        self.write_message(id, request).await?;
        self.read_reply(id).await
    }

    /// Refuse to use a channel that lost frame sync. See `poisoned`.
    fn check_not_poisoned(&self) -> Result<(), EvaluatorError> {
        if self.poisoned {
            return Err(EvaluatorError::ProcessDied);
        }
        Ok(())
    }

    /// Send a request and wait for the response, servicing LLM oracle
    /// callbacks during query evaluation.
    ///
    /// When the C++ evaluator's `@llm_check_fn` functor hits a cache miss,
    /// it sends an `LlmQuery` response instead of the final `QueryResult`.
    /// This method calls `llm_check_fn` to resolve it, sends the result
    /// back, and continues reading until the final response arrives.
    ///
    /// `oracle` measures how long each callback spends with the provider. The
    /// session's query budget stops for that time — see
    /// [`crate::evaluator::Evaluator::take_oracle_wait`] — while the stall
    /// window keeps running.
    pub async fn call_with_oracle<F>(
        &mut self,
        id: u64,
        request: &EvalRequest,
        llm_check_fn: F,
        oracle: &Arc<OracleClock>,
    ) -> Result<EvalResponse, EvaluatorError>
    where
        F: Fn(&str, &str) -> bool + Send + Sync + 'static,
    {
        // The callback goes to the blocking pool rather than running here.
        // `block_in_place` would run it on the thread polling this future, and
        // a future that does not return from `poll` cannot be preempted by a
        // timer: the session evaluator's job budget would neither bound nor
        // deny an oracle round trip, only observe it after the fact. Off the
        // polling thread, the budget's `timeout_at` fires on time and the
        // caller gets the fail-closed deny.
        self.check_not_poisoned()?;
        let llm_check_fn = Arc::new(llm_check_fn);
        self.write_message(id, request).await?;
        loop {
            let response: EvalResponse = self.read_reply(id).await?;
            match response {
                EvalResponse::LlmQuery { prompt, context } => {
                    let check = Arc::clone(&llm_check_fn);
                    // The callback closes the wait itself, so it is recorded
                    // even when this future is dropped at its budget: the
                    // blocking task outlives the drop, and a wait left open
                    // would be handed out again and again as fresh extension.
                    let clock = Arc::clone(oracle);
                    // Tests only. A test installs its subscriber for its own
                    // thread, and the two oracle lines are emitted on the
                    // blocking pool, which has no thread-local subscriber of
                    // its own — so a test's capture carries this thread's
                    // dispatcher over. In a shipped binary the subscriber is
                    // global and the pool already finds it, while the scoped
                    // guard below would raise tracing's scoped-dispatcher
                    // count for the whole oracle call, taking every log
                    // statement on every thread of the process off its fast
                    // path for as long as the provider takes to answer.
                    #[cfg(test)]
                    let dispatch = tracing::dispatcher::get_default(|d| d.clone());
                    // The scrub rides along into the blocking pool. It walks
                    // both strings, `context` is agent-controlled and a frame
                    // runs to `MAX_IPC_FRAME`, so on the polling thread it
                    // would hold the worker for as long as the walk takes and
                    // the session's job budget could not fire during it.
                    let handed_over = tokio::task::spawn_blocking(move || {
                        let query = move || {
                            // The first thing done with the frame. Everything
                            // below — the log lines, the callback, and through
                            // the callback the cache key and the provider —
                            // sees the scrubbed strings and never the ones the
                            // evaluator sent.
                            let (prompt, context) = redact_oracle_strings(&prompt, &context);
                            // Truncate on a UTF-8 char boundary — `context` is
                            // agent-controlled, so a byte slice at a fixed
                            // index (`&context[..80]`) would panic
                            // mid-codepoint.
                            let ctx_preview = match context.char_indices().nth(80) {
                                Some((idx, _)) => format!("{}...", &context[..idx]),
                                None => context.clone(),
                            };
                            info!(
                                "[LLM oracle] query: prompt={:?} context={:?}",
                                prompt, ctx_preview
                            );
                            // Counted from here, not from the hand-off: what
                            // the budget stops for is the provider, and the
                            // scrub above is this process's own work.
                            clock.start();
                            let answer = check(&prompt, &context);
                            clock.finish();
                            info!("[LLM oracle] result={} for prompt={:?}", answer, prompt);
                            answer
                        };
                        #[cfg(test)]
                        {
                            tracing::dispatcher::with_default(&dispatch, query)
                        }
                        #[cfg(not(test))]
                        {
                            query()
                        }
                    })
                    .await;
                    // No-op unless the callback panicked before closing it.
                    oracle.finish();
                    // `JoinError`'s Display prints the panic payload verbatim,
                    // and a panic raised inside the closure can carry the
                    // strings it was still holding — a `&str` sliced off a
                    // char boundary renders up to 256 bytes of the string it
                    // was cut from. So the join failure is named by kind and
                    // the payload is dropped with the error.
                    let result = handed_over.map_err(|e| {
                        let cause = if e.is_panic() {
                            "panicked"
                        } else if e.is_cancelled() {
                            "was cancelled"
                        } else {
                            "did not finish"
                        };
                        EvaluatorError::IpcError(format!("the LLM oracle callback {cause}"))
                    })?;
                    // Under the query's own id: the oracle round trip is part
                    // of answering that query, and the evaluator echoes the id
                    // of the request it is working on either way.
                    self.write_message(id, &EvalRequest::LlmResult { result })
                        .await?;
                }
                other => return Ok(other),
            }
        }
    }

    /// Frame `msg` under `id` and hand it to the pipe.
    ///
    /// Anything left over from an abandoned write goes out first, so the
    /// evaluator always reads whole requests in order. Progress is written
    /// back to `out_buf` after every accepted chunk, which is what makes the
    /// method resumable rather than merely restartable: `write` moves nothing
    /// when its future is dropped, so the buffer and the pipe agree even if
    /// this returns at an await it never came back from.
    async fn write_message<T: serde::Serialize>(
        &mut self,
        id: u64,
        msg: &T,
    ) -> Result<(), EvaluatorError> {
        let payload = serde_json::to_vec(msg)
            .map_err(|e| EvaluatorError::IpcError(format!("Serialize: {}", e)))?;

        let len = payload.len() as u32;
        self.out_buf.extend_from_slice(&len.to_be_bytes());
        self.out_buf.extend_from_slice(&id.to_be_bytes());
        self.out_buf.extend_from_slice(&payload);

        while !self.out_buf.is_empty() {
            match self.stdin.write(&self.out_buf).await {
                Ok(0) => {
                    error!("Evaluator stdin accepted no bytes");
                    return Err(EvaluatorError::ProcessDied);
                }
                // Dropped from the buffer only once the pipe has taken them,
                // so an abandoned write leaves exactly the unsent remainder.
                Ok(n) => {
                    self.out_buf.drain(..n);
                }
                Err(e) => {
                    error!("Write frame failed: {}", e);
                    return Err(EvaluatorError::ProcessDied);
                }
            }
        }
        self.stdin.flush().await.map_err(|e| {
            error!("Flush failed: {}", e);
            EvaluatorError::ProcessDied
        })?;

        debug!(id, bytes = payload.len(), "sent request frame");
        Ok(())
    }

    /// Read frames until one carries `id`, discarding the rest.
    ///
    /// A frame with another id is the late answer to a request whose caller
    /// already gave up (see the module docs). Reading it as this call's reply
    /// would decide one action from another action's evaluation, so it is
    /// dropped and logged; the evaluator is sequential, so the reply this call
    /// is waiting for is behind it.
    async fn read_reply(&mut self, id: u64) -> Result<EvalResponse, EvaluatorError> {
        loop {
            let (frame_id, response) = self.read_frame().await?;
            if frame_id == id {
                return Ok(response);
            }
            warn!(
                discarded_id = frame_id,
                awaiting_id = id,
                "discarded a late evaluator reply for an abandoned request"
            );
        }
    }

    /// Read one whole frame: its id and its decoded payload.
    ///
    /// All partial state lives in `in_buf`, so a caller that drops this future
    /// mid-frame loses nothing — the next call picks the frame up where the
    /// bytes stopped.
    async fn read_frame(&mut self) -> Result<(u64, EvalResponse), EvaluatorError> {
        const HEADER: usize = 4 + FRAME_ID_BYTES;
        loop {
            if self.in_buf.len() >= HEADER {
                let len = u32::from_be_bytes(self.in_buf[..4].try_into().unwrap()) as usize;
                if len > MAX_IPC_FRAME {
                    // Not a frame this channel can skip past: the length that
                    // would say where the next one starts is the field that is
                    // wrong, so there is no resynchronizing. Reported as a
                    // dead process — which is what it amounts to, since the
                    // only recovery is the respawn that path performs — rather
                    // than as an error for this one call, which would leave
                    // the same bad header in place for every call after it.
                    error!("IPC frame too large: {len} bytes (max {MAX_IPC_FRAME}); channel lost frame sync");
                    self.poisoned = true;
                    return Err(EvaluatorError::ProcessDied);
                }
                if self.in_buf.len() >= HEADER + len {
                    let id = u64::from_be_bytes(self.in_buf[4..HEADER].try_into().unwrap());
                    let payload: Vec<u8> = self.in_buf.drain(..HEADER + len).skip(HEADER).collect();
                    debug!(id, bytes = len, "read reply frame");
                    let response = serde_json::from_slice(&payload)
                        .map_err(|e| EvaluatorError::IpcError(format!("Deserialize: {}", e)))?;
                    return Ok((id, response));
                }
            }

            // Into the scratch buffer and only then into `in_buf`: `read`
            // moves no bytes when its future is dropped, so growing `in_buf`
            // before the await would leave zero padding behind in a buffer
            // that outlives this call.
            match self.stdout.read(&mut self.scratch).await {
                Ok(0) => {
                    error!("Evaluator stdout closed");
                    return Err(EvaluatorError::ProcessDied);
                }
                Ok(n) => self.in_buf.extend_from_slice(&self.scratch[..n]),
                Err(e) => {
                    error!("Read frame failed: {}", e);
                    return Err(EvaluatorError::ProcessDied);
                }
            }
        }
    }

    /// The child's OS pid, while it is still ours to signal. Also its
    /// process-group id: `process_group(0)` at spawn made it the leader.
    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Check if the child process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.liveness(), ChildLiveness::Running)
    }

    /// What asking about the child actually established.
    ///
    /// `try_wait` has three outcomes, not two, and the difference matters to
    /// anyone tracking the pid: `Ok(None)` — running; `Ok(Some(_))` — exited
    /// AND reaped, so the kernel may now hand that number to an unrelated
    /// process; `Err(_)` — the question could not be answered and NOTHING was
    /// reaped, so the pid is still ours. Retiring a pid on the error case
    /// would take a group that may still be running off the shutdown kill
    /// list, which is the same leak in the other direction.
    pub fn liveness(&mut self) -> ChildLiveness {
        #[cfg(test)]
        if let Some(forced) = self.forced_liveness {
            return forced;
        }
        match self.child.try_wait() {
            Ok(None) => ChildLiveness::Running,
            Ok(Some(_)) => ChildLiveness::Reaped,
            Err(_) => ChildLiveness::Unknown,
        }
    }

    /// Kill the child process — and, under bwrap, the sandboxed evaluator
    /// inside it.
    ///
    /// The group first: `process_group(0)` at spawn made the child its own
    /// leader, so its pid is the group id and a negative pid signals every
    /// descendant, including a process that our `Child` handle cannot even
    /// name because it lives in the wrapper's PID namespace. Signalled before
    /// the reap, while the pid is still ours: after `wait()` the number could
    /// belong to somebody else's group.
    pub async fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // SAFETY: `kill(2)` with a pid we own and have not reaped.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        let _ = self.child.kill().await;
    }
}

// ── Standalone framing helpers (used in tests) ───────────────────

/// Write one framed JSON message, carrying `id`, to any async writer.
#[cfg(test)]
pub(crate) async fn write_framed<W: AsyncWriteExt + Unpin, T: serde::Serialize>(
    writer: &mut W,
    id: u64,
    msg: &T,
) -> Result<(), EvaluatorError> {
    let payload = serde_json::to_vec(msg)
        .map_err(|e| EvaluatorError::IpcError(format!("Serialize: {}", e)))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_be_bytes()).await.map_err(|e| {
        error!("Write length failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    writer.write_all(&id.to_be_bytes()).await.map_err(|e| {
        error!("Write id failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    writer.write_all(&payload).await.map_err(|e| {
        error!("Write payload failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    writer.flush().await.map_err(|e| {
        error!("Flush failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    Ok(())
}

/// Read one framed JSON message from any async reader, with the id it carries.
#[cfg(test)]
pub(crate) async fn read_framed<R: AsyncReadExt + Unpin, T: serde::de::DeserializeOwned>(
    reader: &mut R,
) -> Result<(u64, T), EvaluatorError> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await.map_err(|e| {
        error!("Read length failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_IPC_FRAME {
        return Err(EvaluatorError::IpcError(format!(
            "IPC frame too large: {len} bytes exceeds cap {MAX_IPC_FRAME}"
        )));
    }
    let mut id_buf = [0u8; FRAME_ID_BYTES];
    reader.read_exact(&mut id_buf).await.map_err(|e| {
        error!("Read id failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await.map_err(|e| {
        error!("Read payload failed: {}", e);
        EvaluatorError::ProcessDied
    })?;
    let msg = serde_json::from_slice(&payload)
        .map_err(|e| EvaluatorError::IpcError(format!("Deserialize: {}", e)))?;
    Ok((u64::from_be_bytes(id_buf), msg))
}

/// Run the oracle loop on generic streams, under request id `id`.
///
/// Sends `request`, then reads responses, discarding any frame carrying
/// another id (the late answer to an abandoned request). On `LlmQuery`, calls
/// `llm_check_fn` and writes `LlmResult` back. Returns on any other response
/// variant.
#[cfg(test)]
pub(crate) async fn oracle_loop<R, W, F>(
    reader: &mut R,
    writer: &mut W,
    id: u64,
    request: &EvalRequest,
    llm_check_fn: F,
) -> Result<EvalResponse, EvaluatorError>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
    F: Fn(&str, &str) -> bool,
{
    write_framed(writer, id, request).await?;
    loop {
        let (frame_id, response): (u64, EvalResponse) = read_framed(reader).await?;
        if frame_id != id {
            warn!(
                discarded_id = frame_id,
                awaiting_id = id,
                "discarded a late evaluator reply for an abandoned request"
            );
            continue;
        }
        match response {
            EvalResponse::LlmQuery { prompt, context } => {
                // Same rule as the other handler: scrub at the frame, before
                // the callback that carries these strings to the provider.
                let (prompt, context) = redact_oracle_strings(&prompt, &context);
                let result = llm_check_fn(&prompt, &context);
                write_framed(writer, id, &EvalRequest::LlmResult { result }).await?;
            }
            other => return Ok(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::types::{EvalActionResult, EvalAuthResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Simulate the "C++ side": read a request, send back a sequence of
    /// responses, each echoing the id the request carried.
    async fn mock_evaluator(
        mut reader: tokio::io::DuplexStream,
        mut writer: tokio::io::DuplexStream,
        responses: Vec<EvalResponse>,
    ) {
        // Read the initial request (keep its id, discard the body)
        let (id, _req): (u64, EvalRequest) = read_framed(&mut reader).await.unwrap();

        for resp in responses {
            write_framed(&mut writer, id, &resp).await.unwrap();

            // If we sent an LlmQuery, read the LlmResult back
            if matches!(resp, EvalResponse::LlmQuery { .. }) {
                let _llm_result: (u64, EvalRequest) = read_framed(&mut reader).await.unwrap();
            }
        }
    }

    fn query_result_ok() -> EvalResponse {
        EvalResponse::QueryResult(EvalAuthResponse {
            results: vec![EvalActionResult {
                index: 0,
                authorized: true,
                is_authenticated: true,
                is_denylisted: false,
                is_allowlisted: true,
                allow_routes: vec![],
                transform_ids: vec![],
                deny_if_unauthorized: false,
                allow_passthrough: false,
                requires_approval: false,
                denial_reasons: vec![],
            }],
        })
    }

    fn dummy_query() -> EvalRequest {
        EvalRequest::Query(crate::evaluator::types::EvalAuthRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: None,
            roles: vec![],
            tenant_id: None,
            session_id: None,
            principal: None,
            action_metadata: vec![],
        })
    }

    fn query_result_denied() -> EvalResponse {
        let mut denied = query_result_ok();
        if let EvalResponse::QueryResult(ref mut r) = denied {
            r.results[0].authorized = false;
            r.results[0].is_allowlisted = false;
        }
        denied
    }

    /// One wire frame — length, id, payload — as the shell would have to
    /// print it.
    #[cfg(unix)]
    fn frame_bytes(id: u64, response: &EvalResponse) -> Vec<u8> {
        let payload = serde_json::to_vec(response).unwrap();
        let mut out = (payload.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&id.to_be_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// Every byte as `\0NNN`, the one escape `printf %b` reads the same way in
    /// every POSIX shell.
    #[cfg(unix)]
    fn octal_escaped(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("\\0{:03o}", b)).collect()
    }

    /// Killing an evaluator kills what it wrapped, not just the process we
    /// hold a handle to. Under bwrap the handle is the wrapper and the
    /// evaluator is init in a new PID namespace, so a signal to the handle
    /// alone leaves a spinning functor running forever — the stall window
    /// would kill nothing and leak a process per breach. A shell that forks
    /// and waits stands in for that shape here.
    #[cfg(unix)]
    #[tokio::test]
    async fn kill_takes_down_what_the_child_wrapped_too() {
        // Deliberately the direct spawn, on every host: the pid the shell
        // prints is a host pid only when nothing wrapped it (under bwrap it is
        // a pid in the sandbox's own namespace). Asking for the unwrapped
        // spawn rather than skipping where bwrap exists is what keeps this
        // assertion running everywhere; the wrapped shape is pinned by
        // `kill_reaches_through_the_bwrap_wrapper` below.
        let mut child = IpcChild::spawn_maybe_wrapped(
            "/bin/sh",
            &["-c", "sleep 30 & printf '%s\\n' \"$!\"; wait"],
            &[],
            false,
        )
        .expect("spawn a shell");

        // Read the pid the shell forked, one chunk at a time — the child's
        // stdout is a raw pipe here, not the framed protocol.
        let mut line = String::new();
        while !line.contains('\n') {
            let mut chunk = [0u8; 64];
            let n = child
                .stdout
                .read(&mut chunk)
                .await
                .expect("the shell reports the pid it forked");
            assert!(n > 0, "the shell exited without reporting a pid");
            line.push_str(&String::from_utf8_lossy(&chunk[..n]));
        }
        let wrapped: i32 = line.trim().parse().expect("a pid");
        // SAFETY: signal 0 only tests for the process's existence.
        assert_eq!(
            unsafe { libc::kill(wrapped, 0) },
            0,
            "the wrapped process should be running before the kill"
        );

        child.kill().await;

        let mut gone = false;
        for _ in 0..100 {
            // SAFETY: as above — signal 0 asks, it does not signal.
            if unsafe { libc::kill(wrapped, 0) } != 0 {
                gone = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            gone,
            "the wrapped process outlived the kill — the signal went to the wrapper only"
        );
    }

    /// Host pids whose argv contains `marker`, read from `/proc`. Linux only:
    /// it is the one place a process started inside a PID namespace can be
    /// found from outside it, which is the whole difficulty the bwrap case
    /// poses. Compiled everywhere so the test below stays type-checked on
    /// hosts that cannot run it.
    #[cfg(unix)]
    fn host_pids_matching(marker: &str) -> Vec<i32> {
        let mut found = Vec::new();
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return found;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(pid) = name.to_str().and_then(|s| s.parse::<i32>().ok()) else {
                continue;
            };
            if let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) {
                if String::from_utf8_lossy(&cmdline).contains(marker) {
                    found.push(pid);
                }
            }
        }
        found
    }

    /// The case the process group exists for. Under bwrap the handle we hold
    /// is the wrapper: `--unshare-all` gives the evaluator its own PID
    /// namespace, so it survives the wrapper's death and a signal to the
    /// handle alone would leave it spinning. The claim that `kill(-pgid)`
    /// still reaches it — bwrap neither `setsid`s nor `setpgid`s, and
    /// `prlimit` execs in place — is asserted by the code and demonstrated
    /// here, on a host that actually has bwrap. Elsewhere there is nothing to
    /// demonstrate and the test returns.
    #[cfg(unix)]
    #[tokio::test]
    async fn kill_reaches_through_the_bwrap_wrapper() {
        if !cfg!(target_os = "linux") {
            eprintln!(
                "skipped: the host-side /proc scan this needs exists only on Linux, so there \
                 is no way here to see a process inside the sandbox's own PID namespace"
            );
            return;
        }
        if !crate::sandbox::has_bubblewrap() {
            eprintln!(
                "skipped: bubblewrap is not on PATH, so the evaluator spawns unwrapped and \
                 there is no wrapper for the kill to reach through"
            );
            return;
        }
        // A duration no other process on the host is sleeping for, so the
        // /proc scan cannot match somebody else's sleeper.
        let marker = format!("31337.{}", std::process::id());
        let script = format!("sleep {marker} & wait");
        let mut child = IpcChild::spawn_maybe_wrapped("/bin/sh", &["-c", &script], &[], true)
            .expect("spawn a shell under bwrap");
        // The handle's own pid: the bwrap wrapper, which is what the kill is
        // sent through. Both ends of the pair have to be gone afterwards —
        // the wrapper because it is the process this side is holding, and the
        // sleeper because it is the one doing the work the stall window is
        // trying to stop.
        let wrapper = child.pid().expect("a pid for the wrapper") as i32;

        let mut sleeper = Vec::new();
        for _ in 0..200 {
            sleeper = host_pids_matching(&marker);
            if !sleeper.is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert!(
            !sleeper.is_empty(),
            "the sandboxed sleeper never appeared on the host — nothing to kill"
        );

        child.kill().await;

        let mut gone = false;
        for _ in 0..200 {
            if host_pids_matching(&marker).is_empty() {
                gone = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert!(
            gone,
            "a process bwrap wrapped outlived the kill: the group signal does not reach \
             into the sandbox, so the stall window bounds nothing"
        );

        // SAFETY: `kill(2)` with signal 0 only tests for the process. The
        // child is reaped by `kill`, so the pid is unambiguous.
        let wrapper_alive = unsafe { libc::kill(wrapper, 0) } == 0;
        assert!(
            !wrapper_alive,
            "the bwrap wrapper {wrapper} outlived the kill: the session respawns alongside \
             a process still holding its memory"
        );
    }

    /// The point of the request id. A query whose caller gave up is still
    /// being evaluated, and its answer arrives on the pipe later; without the
    /// id the next query would take that answer as its own and authorize one
    /// action from another action's evaluation. Here the child speaks first
    /// with a stale id, and the two calls must each get their own frame.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_reply_carrying_another_request_id_is_discarded() {
        let mut script = String::from("printf '%b' '");
        // The abandoned request's answer, arriving before either call's.
        script.push_str(&octal_escaped(&frame_bytes(
            3,
            &EvalResponse::Error {
                message: "the abandoned request".into(),
            },
        )));
        script.push_str(&octal_escaped(&frame_bytes(7, &query_result_ok())));
        script.push_str(&octal_escaped(&frame_bytes(8, &query_result_denied())));
        // Stay alive so the requests have somewhere to be written; the frames
        // are already in the pipe either way.
        script.push_str("'; sleep 30");

        let mut child = IpcChild::spawn("/bin/sh", &["-c", &script], &[]).expect("spawn a shell");

        match child.call(7, &dummy_query()).await.expect("a reply") {
            EvalResponse::QueryResult(r) => assert!(
                r.results[0].authorized,
                "call 7 was answered with another request's frame"
            ),
            other => panic!("the stale frame was read as the reply: {:?}", other),
        }
        match child.call(8, &dummy_query()).await.expect("a reply") {
            EvalResponse::QueryResult(r) => assert!(
                !r.results[0].authorized,
                "call 8 got call 7's answer, not its own"
            ),
            other => panic!("expected call 8's own answer, got {:?}", other),
        }

        child.kill().await;
    }

    /// A length prefix past the cap is not a bad answer to one call, it is a
    /// lost frame boundary: the field that would say where the next frame
    /// starts is the field that is wrong. Reported as an error for that one
    /// call, the bad header stays at the head of the buffer and every later
    /// call fails on the same bytes — for ever, on a child that answers every
    /// aliveness check, so it is never killed, never respawned, and the
    /// session fails closed until the server restarts. A functor writing a
    /// stray line to stdout is enough to produce one.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_length_past_the_cap_ends_the_channel_instead_of_failing_every_call_forever() {
        let mut script = String::from("printf '%b' '");
        // A whole header: a length of 0xFFFFFFFF and an id of 0. Short of a
        // full header the reader would simply wait for more bytes.
        script.push_str(&octal_escaped(&[
            0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0,
        ]));
        // Alive afterwards, which is exactly the trap: nothing about this
        // child looks dead.
        script.push_str("'; sleep 30");

        let mut child = IpcChild::spawn("/bin/sh", &["-c", &script], &[]).expect("spawn a shell");

        for attempt in 1..=3u64 {
            match child.call(attempt, &dummy_query()).await {
                Err(EvaluatorError::ProcessDied) => {}
                other => panic!(
                    "attempt {attempt} reported {other:?}; a channel that lost frame sync must \
                     report a dead process, which is the only path that respawns it"
                ),
            }
        }

        child.kill().await;
    }

    /// The oracle round trip is inside the query, so the query budget has to
    /// be able to end it. A callback run on the polling thread cannot be
    /// preempted by a timer — the future never returns from `poll`, the
    /// budget's timeout never fires, and the check is neither bounded nor
    /// denied but eventually ALLOWED, late. Run off that thread it is just a
    /// slow future, and the budget cuts it.
    ///
    /// Multi-threaded on purpose: on a current-thread runtime the shape this
    /// pins is not even reachable.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_blocking_oracle_callback_does_not_outlive_the_query_budget() {
        let mut script = String::from("printf '%b' '");
        script.push_str(&octal_escaped(&frame_bytes(
            11,
            &EvalResponse::LlmQuery {
                prompt: "is this exfiltration?".into(),
                context: "…".into(),
            },
        )));
        // Nothing after the oracle question: the only thing that can end this
        // call is the budget.
        script.push_str("'; sleep 30");

        let mut child = IpcChild::spawn("/bin/sh", &["-c", &script], &[]).expect("spawn a shell");

        let budget = std::time::Duration::from_millis(200);
        let clock = Arc::new(OracleClock::default());
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let entered_tx = std::sync::Mutex::new(Some(entered_tx));
        let request = dummy_query();
        let (started, outcome) = {
            let query = child.call_with_oracle(
                11,
                &request,
                move |_prompt: &str, _ctx: &str| {
                    entered_tx
                        .lock()
                        .unwrap()
                        .take()
                        .expect("only one oracle callback")
                        .send(std::time::Instant::now())
                        .expect("test is waiting for callback entry");
                    // An oracle that takes far longer than the budget — an LLM
                    // client under its own 30 s timeout, or a wedged one.
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    true
                },
                &clock,
            );
            tokio::pin!(query);
            // Process startup, IPC and blocking-pool scheduling are not
            // provider time. Poll the query until the callback actually starts
            // before arming the measured budget, so slow CI startup cannot
            // consume most of the time asserted against OracleClock below.
            let started = tokio::select! {
                started = entered_rx => started.expect("oracle callback entered"),
                outcome = &mut query => panic!("query returned before callback entry: {outcome:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                    panic!("oracle callback did not start")
                }
            };
            // Use the timestamp sent FROM the callback, not Instant::now()
            // here: an inline/block_in_place regression would keep this task
            // stuck in poll for three seconds even though entry was signaled.
            let outcome = tokio::time::timeout_at(
                tokio::time::Instant::from_std(started + budget),
                &mut query,
            )
            .await;
            (started, outcome)
        };

        assert!(
            outcome.is_err(),
            "the oracle callback ran to completion inside the budget's own future: \
             a {:?} budget did not bound it",
            budget
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "the budget fired {:?} late — the callback was blocking the poll",
            started.elapsed().saturating_sub(budget)
        );

        // The wait is reported while the callback is still running, not only
        // when it returns. The session's budget stops for it (M3), and a
        // budget expires mid-call by definition — a clock that counted only
        // finished calls would report zero exactly when it is read.
        let counted = clock.take();
        assert!(
            counted >= budget / 2,
            "the oracle clock reported {counted:?} for a callback that had been running \
             for at least {:?}",
            started.elapsed()
        );

        child.kill().await;
    }

    #[tokio::test]
    async fn oracle_no_llm_queries() {
        // C++ returns QueryResult directly — no oracle callbacks.
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);

        let responses = vec![query_result_ok()];
        tokio::spawn(mock_evaluator(server_read, server_write, responses));

        let mut reader = client_read;
        let mut writer = client_write;
        let result = oracle_loop(&mut reader, &mut writer, 1, &dummy_query(), |_, _| {
            panic!("should not be called")
        })
        .await
        .unwrap();

        match result {
            EvalResponse::QueryResult(r) => assert!(r.results[0].authorized),
            other => panic!("Expected QueryResult, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn oracle_single_llm_query() {
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);

        let responses = vec![
            EvalResponse::LlmQuery {
                prompt: "Contains PII?".into(),
                context: "SSN: 123".into(),
            },
            query_result_ok(),
        ];
        tokio::spawn(mock_evaluator(server_read, server_write, responses));

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let mut reader = client_read;
        let mut writer = client_write;
        let result = oracle_loop(
            &mut reader,
            &mut writer,
            1,
            &dummy_query(),
            move |prompt, context| {
                cc.fetch_add(1, Ordering::Relaxed);
                assert_eq!(prompt, "Contains PII?");
                assert_eq!(context, "SSN: 123");
                true
            },
        )
        .await
        .unwrap();

        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        assert!(matches!(result, EvalResponse::QueryResult(_)));
    }

    #[tokio::test]
    async fn oracle_multiple_llm_queries() {
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);

        let responses = vec![
            EvalResponse::LlmQuery {
                prompt: "check1".into(),
                context: "ctx1".into(),
            },
            EvalResponse::LlmQuery {
                prompt: "check2".into(),
                context: "ctx2".into(),
            },
            EvalResponse::LlmQuery {
                prompt: "check3".into(),
                context: "ctx3".into(),
            },
            query_result_ok(),
        ];
        tokio::spawn(mock_evaluator(server_read, server_write, responses));

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let mut reader = client_read;
        let mut writer = client_write;
        let result = oracle_loop(&mut reader, &mut writer, 1, &dummy_query(), move |_, _| {
            cc.fetch_add(1, Ordering::Relaxed);
            true
        })
        .await
        .unwrap();

        assert_eq!(call_count.load(Ordering::Relaxed), 3);
        assert!(matches!(result, EvalResponse::QueryResult(_)));
    }

    #[tokio::test]
    async fn oracle_callback_result_false() {
        // Verify the callback's return value is correctly sent back.
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);

        // Custom mock: reads request, sends LlmQuery, reads LlmResult
        // and verifies it's false, then sends QueryResult.
        tokio::spawn(async move {
            let mut reader = server_read;
            let mut writer = server_write;

            let (id, _req): (u64, EvalRequest) = read_framed(&mut reader).await.unwrap();

            write_framed(
                &mut writer,
                id,
                &EvalResponse::LlmQuery {
                    prompt: "test".into(),
                    context: "ctx".into(),
                },
            )
            .await
            .unwrap();

            let (_, llm_result): (u64, EvalRequest) = read_framed(&mut reader).await.unwrap();
            match llm_result {
                EvalRequest::LlmResult { result } => assert!(!result, "Expected false"),
                other => panic!("Expected LlmResult, got {:?}", other),
            }

            write_framed(&mut writer, id, &query_result_ok())
                .await
                .unwrap();
        });

        let mut reader = client_read;
        let mut writer = client_write;
        let result = oracle_loop(
            &mut reader,
            &mut writer,
            1,
            &dummy_query(),
            |_, _| false, // always deny
        )
        .await
        .unwrap();

        assert!(matches!(result, EvalResponse::QueryResult(_)));
    }

    // ── Oracle redaction ─────────────────────────────

    /// The three values a redacted query must not carry, in the shapes an
    /// agent actually reads them in: a `.env` file it opened, and a request
    /// header quoted into the question the policy asks.
    const ORACLE_PASSWORD: &str = "s3cr3t-pg-9f2a1c";
    const ORACLE_API_KEY: &str = "sk-proj-AAAAAAAAAAAAAAAAAAAA1234";
    const ORACLE_BEARER: &str = "abcdefghijklmnopqrstuvwxyz012345";

    fn oracle_context() -> String {
        format!(
            "# staging\nDB_PASSWORD={ORACLE_PASSWORD}\nOPENAI_API_KEY={ORACLE_API_KEY}\n\
             REGION=us-east-1\n"
        )
    }

    fn oracle_prompt() -> String {
        format!(
            "Does this call exfiltrate data? The request carried\n\
             authorization: Bearer {ORACLE_BEARER}\n\
             and its body named {ORACLE_API_KEY} outright.\n"
        )
    }

    /// What the oracle callback is allowed to have been given.
    fn assert_scrubbed(prompt: &str, context: &str) {
        for (label, text) in [("prompt", prompt), ("context", context)] {
            assert!(
                text.contains("[redacted"),
                "the {label} reached the callback with no marker in it: {text:?}"
            );
        }
        // The two marker forms the model ever sees: the plain one, where the
        // name said what the value was, and the shaped one, where the value
        // itself gave it away.
        assert!(
            context.contains("DB_PASSWORD=[redacted]"),
            "the name is kept and the value replaced: {context:?}"
        );
        assert!(
            prompt.contains("authorization: [redacted]"),
            "the header kept its name and lost its value: {prompt:?}"
        );
        assert!(
            prompt.contains("[redacted:openai-key]"),
            "a value that gives its own shape away is marked with it: {prompt:?}"
        );
        for value in [ORACLE_PASSWORD, ORACLE_API_KEY, ORACLE_BEARER] {
            assert!(
                !prompt.contains(value) && !context.contains(value),
                "{value:?} reached the oracle callback — prompt {prompt:?}, context {context:?}"
            );
        }
        // The text around the secrets is what makes the question answerable,
        // so it has to survive.
        assert!(
            context.contains("REGION=us-east-1") && prompt.contains("exfiltrate"),
            "scrubbing took more than the secrets: prompt {prompt:?}, context {context:?}"
        );
    }

    /// Records what the oracle callback was handed.
    type Seen = Arc<std::sync::Mutex<Vec<(String, String)>>>;

    fn recorder(seen: &Seen) -> impl Fn(&str, &str) -> bool + Send + Sync + 'static {
        let seen = Arc::clone(seen);
        move |prompt: &str, context: &str| {
            seen.lock()
                .unwrap()
                .push((prompt.to_string(), context.to_string()));
            true
        }
    }

    /// A child that asks one oracle question carrying the three secrets, then
    /// answers the query.
    #[cfg(unix)]
    fn secret_bearing_child(id: u64) -> IpcChild {
        let mut script = String::from("printf '%b' '");
        script.push_str(&octal_escaped(&frame_bytes(
            id,
            &EvalResponse::LlmQuery {
                prompt: oracle_prompt(),
                context: oracle_context(),
            },
        )));
        script.push_str(&octal_escaped(&frame_bytes(id, &query_result_ok())));
        // Stay alive: the oracle's answer is written back to this child.
        script.push_str("'; sleep 30");
        IpcChild::spawn("/bin/sh", &["-c", &script], &[]).expect("spawn a shell")
    }

    /// The query path the server runs. The callback is the last hop before
    /// the provider — it is what `crate::llm::check` is in production, and
    /// through it the cache key — so what it is handed is what leaves the
    /// host.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_oracle_callback_is_handed_markers_not_the_secrets_the_frame_carried() {
        let _switch = crate::oracle_redaction::test_switch::force(true);
        let mut child = secret_bearing_child(21);
        let seen: Seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let clock = Arc::new(OracleClock::default());

        let answer = child
            .call_with_oracle(21, &dummy_query(), recorder(&seen), &clock)
            .await
            .expect("a reply");
        child.kill().await;

        assert!(matches!(answer, EvalResponse::QueryResult(_)));
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1, "the oracle was asked once");
        assert_scrubbed(&calls[0].0, &calls[0].1);
    }

    /// A panic in the callback is reported by kind, never by payload.
    ///
    /// The closure holds the strings the frame carried while it runs, so a
    /// panic raised inside it can carry them in its message — the standard
    /// library's own char-boundary panic quotes up to 256 bytes of the string
    /// it was slicing. `JoinError`'s Display prints that message verbatim, so
    /// interpolating the join error would put the payload into an error
    /// string that is logged and returned to the caller.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_panicking_callback_is_reported_without_its_panic_payload() {
        const SENTINEL: &str = "sk-live-do-not-log-this-one";

        let _switch = crate::oracle_redaction::test_switch::force(true);
        let mut child = secret_bearing_child(22);
        let clock = Arc::new(OracleClock::default());
        // The panic is expected, so keep the default hook's report off the
        // test's output; the hook is global, hence restored right after.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let error = child
            .call_with_oracle(
                22,
                &dummy_query(),
                move |_prompt: &str, _context: &str| -> bool {
                    panic!("the callback tripped over {SENTINEL}")
                },
                &clock,
            )
            .await
            .expect_err("a panicking callback is not an answer");
        std::panic::set_hook(hook);
        child.kill().await;

        let text = error.to_string();
        assert!(
            text.contains("panicked"),
            "the error does not say the callback panicked: {text}"
        );
        assert!(
            !text.contains(SENTINEL),
            "the panic payload was copied into the error: {text}"
        );
    }

    /// The other handler, on the same rule. A frame read here reaches a
    /// callback too, so redacting in one place and not the other would leave
    /// the secrets a path out.
    #[tokio::test]
    async fn the_second_handler_scrubs_the_frame_the_same_way() {
        let _switch = crate::oracle_redaction::test_switch::force(true);
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);
        tokio::spawn(mock_evaluator(
            server_read,
            server_write,
            vec![
                EvalResponse::LlmQuery {
                    prompt: oracle_prompt(),
                    context: oracle_context(),
                },
                query_result_ok(),
            ],
        ));

        let seen: Seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let check = recorder(&seen);
        let mut reader = client_read;
        let mut writer = client_write;
        let result = oracle_loop(&mut reader, &mut writer, 1, &dummy_query(), check)
            .await
            .unwrap();

        assert!(matches!(result, EvalResponse::QueryResult(_)));
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1, "the oracle was asked once");
        assert_scrubbed(&calls[0].0, &calls[0].1);
    }

    /// With the switch off the two strings are handed over exactly as the
    /// evaluator wrote them. An operator who turns redaction off gets the
    /// oracle they had before it existed, not a differently-mangled one.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_switch_off_hands_the_callback_the_strings_as_written() {
        let _switch = crate::oracle_redaction::test_switch::force(false);
        let mut child = secret_bearing_child(22);
        let seen: Seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let clock = Arc::new(OracleClock::default());

        child
            .call_with_oracle(22, &dummy_query(), recorder(&seen), &clock)
            .await
            .expect("a reply");
        child.kill().await;

        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, oracle_prompt());
        assert_eq!(calls[0].1, oracle_context());
    }

    /// The same, through the second handler.
    #[tokio::test]
    async fn the_switch_off_is_pass_through_in_the_second_handler_too() {
        let _switch = crate::oracle_redaction::test_switch::force(false);
        let (client_read, server_write) = tokio::io::duplex(4096);
        let (server_read, client_write) = tokio::io::duplex(4096);
        tokio::spawn(mock_evaluator(
            server_read,
            server_write,
            vec![
                EvalResponse::LlmQuery {
                    prompt: oracle_prompt(),
                    context: oracle_context(),
                },
                query_result_ok(),
            ],
        ));

        let seen: Seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let check = recorder(&seen);
        let mut reader = client_read;
        let mut writer = client_write;
        oracle_loop(&mut reader, &mut writer, 1, &dummy_query(), check)
            .await
            .unwrap();

        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, oracle_prompt());
        assert_eq!(calls[0].1, oracle_context());
    }

    /// A sink a `tracing` subscriber writes formatted events into, so a test
    /// can read what was logged.
    #[derive(Clone, Default)]
    struct LogSink(Arc<std::sync::Mutex<Vec<u8>>>);

    impl LogSink {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }

        fn clear(&self) {
            self.0.lock().unwrap().clear();
        }
    }

    impl std::io::Write for LogSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogSink {
        type Writer = LogSink;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// The log preview is taken from the frame too, so redacting after it
    /// would put the secret in the server's own log — the place an operator
    /// is least likely to be watching for one. Ordering is what this pins:
    /// the query line is written from the scrubbed strings.
    ///
    /// The subscriber here is thread-local, installed for the test's own
    /// thread, and the two oracle lines are not emitted there: they are
    /// emitted on the blocking pool. What carries them back is the test-only
    /// dispatcher capture in `call_with_oracle` — it clones the dispatcher of
    /// the thread that polls the query and re-installs it around the blocking
    /// task, so the lines this test reads are the lines that call wrote. The
    /// runtime is single-threaded so that thread is this one.
    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn the_log_line_is_written_from_the_scrubbed_strings() {
        let _switch = crate::oracle_redaction::test_switch::force(true);
        let sink = LogSink::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(sink.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let _log = tracing::subscriber::set_default(subscriber);
        let clock = Arc::new(OracleClock::default());

        // One call first, then a rebuild, then the call this test reads.
        // `tracing` caches whether a log statement is worth evaluating, once
        // per statement for the whole process, and computes it from whichever
        // thread reached the statement first — a sibling test with no
        // subscriber, when the suite runs in parallel, and then the answer is
        // "no one is listening". The warm-up makes sure both statements have
        // been reached, and the rebuild recomputes the answer here, where the
        // sink is listening.
        let mut warmup = secret_bearing_child(23);
        warmup
            .call_with_oracle(23, &dummy_query(), |_: &str, _: &str| true, &clock)
            .await
            .expect("a reply");
        warmup.kill().await;
        tracing::callsite::rebuild_interest_cache();
        sink.clear();

        let mut child = secret_bearing_child(24);
        child
            .call_with_oracle(24, &dummy_query(), |_: &str, _: &str| true, &clock)
            .await
            .expect("a reply");
        child.kill().await;

        let logged = sink.text();
        assert!(
            logged.contains("[LLM oracle] query:"),
            "the query line was not logged at all: {logged}"
        );
        assert!(
            logged.contains("[redacted"),
            "the query line carries no marker, so it was written before the scrub: {logged}"
        );
        for value in [ORACLE_PASSWORD, ORACLE_API_KEY, ORACLE_BEARER] {
            assert!(
                !logged.contains(value),
                "{value:?} was written to the log: {logged}"
            );
        }
    }

    /// A stand-in for the evaluator binary: functor C++ whose
    /// `__attribute__((constructor))` runs before `main`, tries one of the
    /// syscalls on the deny-list, and remembers what happened. The frame loop
    /// then answers one request with that result, which is also what shows
    /// the process is alive and serving after the filter was installed.
    ///
    /// `unshare(0)` is a no-op that needs no namespace capabilities. Creating
    /// a mount namespace instead would require CAP_SYS_ADMIN, which bwrap
    /// can drop even without our filter, giving an unrelated EPERM on CI.
    /// The filter rejects the syscall number regardless of flags, so the
    /// no-op still distinguishes a pre-exec denial from an unfiltered call.
    #[cfg(target_os = "linux")]
    const PRE_EXEC_PROBE_SOURCE: &str = r#"
#include <sched.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unistd.h>

static long g_rc = -999;
static int g_errno = -999;

__attribute__((constructor)) static void probe_before_main() {
    errno = 0;
    g_rc = ::unshare(0);
    g_errno = errno;
}

static bool read_exact(char* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(0, buf + got, n - got);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

int main() {
    char header[12];
    if (!read_exact(header, sizeof(header))) return 0;
    uint32_t len = 0;
    for (int i = 0; i < 4; ++i) len = (len << 8) | static_cast<unsigned char>(header[i]);
    std::string payload(len, '\0');
    if (len && !read_exact(&payload[0], len)) return 0;

    char message[128];
    std::snprintf(message, sizeof(message), "probe rc=%ld errno=%d", g_rc, g_errno);
    std::string reply = std::string("{\"Error\":{\"message\":\"") + message + "\"}}";
    uint32_t out_len = static_cast<uint32_t>(reply.size());
    char out_header[12];
    for (int i = 0; i < 4; ++i) out_header[i] = static_cast<char>((out_len >> (8 * (3 - i))) & 0xff);
    for (int i = 0; i < 8; ++i) out_header[4 + i] = header[4 + i];
    ::write(1, out_header, sizeof(out_header));
    ::write(1, reply.data(), reply.size());
    return 0;
}
"#;

    /// Compile [`PRE_EXEC_PROBE_SOURCE`] into `dir` and answer the binary's
    /// path, or `None` when there is no g++ to compile it with.
    #[cfg(target_os = "linux")]
    fn build_pre_exec_probe(dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let source = dir.join("probe.cpp");
        std::fs::write(&source, PRE_EXEC_PROBE_SOURCE).expect("write the probe source");
        let binary = dir.join("probe-evaluator");
        let out = std::process::Command::new("g++")
            .args(["-O0", "-o"])
            .arg(&binary)
            .arg(&source)
            .output()
            .ok()?;
        assert!(
            out.status.success(),
            "compiling the probe: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(binary)
    }

    /// Spawn the probe under bwrap with the pre-exec filter on or off and
    /// return what its constructor saw.
    #[cfg(target_os = "linux")]
    async fn pre_exec_probe_result(binary: &std::path::Path, pre_exec_seccomp: bool) -> String {
        let mut child = IpcChild::spawn_maybe_wrapped_with_seccomp(
            binary.to_str().expect("a utf-8 path"),
            &[],
            &[],
            true,
            pre_exec_seccomp,
        )
        .expect("spawn the probe under bwrap");
        let reply = child.call(1, &EvalRequest::Reset).await;
        child.kill().await;
        match reply.expect("the probe answers after its constructor ran") {
            EvalResponse::Error { message } => message,
            other => panic!("the probe answers with its constructor's result, not {other:?}"),
        }
    }

    /// The window the pre-exec filter closes, shown from both sides on one
    /// host: a functor's
    /// static constructor calling a denied syscall is refused when bwrap
    /// installed the deny-list before the exec, and is not refused when it
    /// did not — which is what the evaluator's own copy, installed from
    /// `main`, can never reach in time.
    ///
    /// Ignored by default: it needs a Linux host with a working bubblewrap
    /// 0.4.0 or newer and a g++ (Linux only). Run it by name.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "needs Linux, a working bubblewrap 0.4.0+, and g++"]
    async fn a_constructor_cannot_reach_a_denied_syscall_when_bwrap_filters_before_exec() {
        if !crate::sandbox::sandbox_available() {
            eprintln!("skipped: no working bubblewrap, so nothing installs a filter before exec");
            return;
        }
        if !crate::sandbox::seccomp_pre_exec_available() {
            eprintln!("skipped: this bubblewrap does not take the program on a descriptor");
            return;
        }
        let dir = tempfile::tempdir().expect("a temp dir");
        let Some(binary) = build_pre_exec_probe(dir.path()) else {
            eprintln!("skipped: no g++ to compile the probe with");
            return;
        };

        let filtered = pre_exec_probe_result(&binary, true).await;
        assert_eq!(
            filtered,
            format!("probe rc=-1 errno={}", libc::EPERM),
            "the constructor reached a denied syscall despite the pre-exec filter"
        );

        let unfiltered = pre_exec_probe_result(&binary, false).await;
        assert_eq!(
            unfiltered, "probe rc=0 errno=0",
            "without the pre-exec filter the no-op syscall must succeed"
        );
    }

    /// The shim's log line tells the two cases apart from the environment, so
    /// the variable has to survive `--clearenv` and reach the child — and only
    /// when a descriptor was actually passed, or the line would claim a
    /// pre-exec filter that is not there.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "needs Linux and a working bubblewrap"]
    async fn the_child_is_told_whether_bwrap_installed_the_filter_before_exec() {
        if !crate::sandbox::seccomp_pre_exec_available() {
            eprintln!("skipped: no bubblewrap that takes the program on a descriptor");
            return;
        }
        for (pre_exec, expected) in [(true, true), (false, false)] {
            let mut child = IpcChild::spawn_maybe_wrapped_with_seccomp(
                "/usr/bin/env",
                &[],
                &[],
                true,
                pre_exec,
            )
            .expect("spawn env under bwrap");
            let mut printed = String::new();
            loop {
                let mut chunk = [0u8; 512];
                let n = child.stdout.read(&mut chunk).await.expect("env's output");
                if n == 0 {
                    break;
                }
                printed.push_str(&String::from_utf8_lossy(&chunk[..n]));
            }
            child.kill().await;
            assert_eq!(
                printed.contains("SASY_SECCOMP_PRE_EXEC=1"),
                expected,
                "spawn with pre_exec={pre_exec} passed the wrong environment: {printed}"
            );
            assert!(
                printed.contains("SASY_SKIP_IN_PROC_SANDBOX=1"),
                "the bwrap spawn always tells the evaluator its chroot is delegated"
            );
        }
    }
}
