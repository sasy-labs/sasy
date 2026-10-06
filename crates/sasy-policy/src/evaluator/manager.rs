//! Evaluator process manager.
//!
//! Manages the lifecycle of an out-of-process evaluator child,
//! exposing it through the Evaluator trait. Handles spawning,
//! restart on crash, and clean shutdown.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::engine::GraphUpdate;

use super::ipc::{ChildLiveness, IpcChild};
use super::protocol::validate_transportable_request;
use super::types::{EvalAuthRequest, EvalAuthResponse, EvalRequest, EvalResponse};
use super::{Evaluator, EvaluatorError, OracleClock};

fn validate_query_response(
    expected_actions: usize,
    response: EvalAuthResponse,
) -> Result<EvalAuthResponse, EvaluatorError> {
    if response.results.len() != expected_actions {
        return Err(EvaluatorError::QueryError(format!(
            "evaluator returned {} results for {expected_actions} actions",
            response.results.len(),
        )));
    }
    for (expected, result) in response.results.iter().enumerate() {
        if result.index != expected as u32 {
            return Err(EvaluatorError::QueryError(format!(
                "evaluator result at position {expected} has index {}",
                result.index,
            )));
        }
    }
    Ok(response)
}

/// Name a response frame that arrived where another variant was expected.
///
/// The result goes into an error string that is logged and returned to the
/// caller, so every variant is named without its payload. Two variants carry
/// recorded content: `LlmQuery` holds the prompt a policy wrote and the
/// context it was given, which is agent-controlled; `QueryResult` holds the
/// denial reasons a policy built, and a policy may build one out of decoded
/// message text. Neither is
/// formatted whole. The match is exhaustive on purpose — there is no Debug
/// fallback arm, so no future variant can start copying a payload into a log
/// line by inheriting one. A count or a length stands in where it tells an
/// operator which frame this was.
fn describe_unexpected(response: &EvalResponse) -> String {
    match response {
        EvalResponse::UpdateOk => "UpdateOk".to_string(),
        EvalResponse::ResetOk => "ResetOk".to_string(),
        EvalResponse::Error { message } => {
            format!("Error {{ message: {} bytes }}", message.len())
        }
        EvalResponse::QueryResult(response) => {
            let denial_reasons: usize = response
                .results
                .iter()
                .map(|result| result.denial_reasons.len())
                .sum();
            format!(
                "QueryResult {{ results: {}, denial reasons: {denial_reasons} }}",
                response.results.len(),
            )
        }
        EvalResponse::LlmQuery { prompt, context } => format!(
            "LlmQuery {{ prompt: {} bytes, context: {} bytes }}",
            prompt.len(),
            context.len()
        ),
    }
}

/// Return the LLM check callback for the oracle loop.
///
/// When the `llm` feature is enabled, delegates to `crate::llm::check`.
/// Otherwise returns a stub that always returns `false` (fail-safe deny).
fn llm_check_fn() -> impl Fn(&str, &str) -> bool + Send + 'static {
    #[cfg(feature = "llm")]
    {
        |prompt: &str, context: &str| crate::llm::check(prompt, context)
    }
    #[cfg(not(feature = "llm"))]
    {
        |_prompt: &str, _context: &str| {
            tracing::warn!("LLM oracle callback but llm feature not enabled — returning false");
            false
        }
    }
}

/// Configuration for an out-of-process evaluator.
#[derive(Debug, Clone, Default)]
pub struct EvaluatorProcessConfig {
    pub program: String,
    pub args: Vec<String>,
    pub backend: String,
    /// Per-spawn env overrides, applied to the child only. Used for
    /// the souffle-interpreted backend's `LD_LIBRARY_PATH` so the
    /// override doesn't leak into the server's own env (and from
    /// there into every subsequent subprocess).
    pub env: Vec<(String, String)>,
}

/// Manages an out-of-process evaluator communicating over stdin/stdout.
pub struct EvaluatorProcess {
    config: EvaluatorProcessConfig,
    read_only_files: Vec<PathBuf>,
    child: Mutex<Option<IpcChild>>,
    /// Source of the request ids the frame protocol carries. Monotonic across
    /// restarts as well as requests, so a frame from a killed child can never
    /// be mistaken for an answer from its replacement.
    next_request_id: AtomicU64,
    /// The id written and not yet answered; ``0`` for none. A caller that
    /// abandons a call — the query budget expiring — leaves it set, which is
    /// how the session evaluator knows this process still owes a reply and
    /// which request it owes. Read without the mutex on purpose: the whole
    /// point is to ask while a call is in flight.
    outstanding_id: AtomicU64,
    /// The live child's pid (also its process-group id), or ``0`` when there
    /// is none. Kept outside the mutex so `kill` can signal the group without
    /// waiting for whatever IPC is currently holding it — see `kill`.
    child_pid: AtomicU32,
    /// Time spent waiting on the LLM oracle, drained by
    /// [`Evaluator::take_oracle_wait`]. See that method for why the session's
    /// query budget stops for it.
    oracle: Arc<OracleClock>,
}

impl EvaluatorProcess {
    pub fn new(config: EvaluatorProcessConfig) -> Result<Self, EvaluatorError> {
        Self::new_with_read_only_files(config, Vec::new())
    }

    pub(crate) fn new_with_read_only_files(
        config: EvaluatorProcessConfig,
        read_only_files: Vec<PathBuf>,
    ) -> Result<Self, EvaluatorError> {
        let args_refs: Vec<&str> = config.args.iter().map(|s| s.as_str()).collect();
        let child = IpcChild::spawn_with_read_only_files(
            &config.program,
            &args_refs,
            &config.env,
            &read_only_files,
        )?;
        info!(
            "Evaluator process started: {} (backend={})",
            config.program, config.backend
        );
        let pid = child.pid().unwrap_or(0);
        // Track the group so it dies with this server even if nothing ever
        // calls `kill` on it — see `crate::evaluator::reaper`.
        crate::evaluator::reaper::install_shutdown_handler();
        crate::evaluator::reaper::register(pid);
        Ok(Self {
            config,
            read_only_files,
            child: Mutex::new(Some(child)),
            next_request_id: AtomicU64::new(0),
            outstanding_id: AtomicU64::new(0),
            child_pid: AtomicU32::new(pid),
            oracle: Arc::new(OracleClock::default()),
        })
    }

    /// Spawn a compiled-Soufflé evaluator: the `souffle-evaluator` shim
    /// binary `evaluator_bin` with the policy `program_name` as its arg.
    pub fn souffle(evaluator_bin: PathBuf, program_name: String) -> Result<Self, EvaluatorError> {
        info!(
            "Starting Soufflé evaluator: bin={}, program={}",
            evaluator_bin.display(),
            program_name
        );
        Self::new(EvaluatorProcessConfig {
            program: evaluator_bin.to_string_lossy().to_string(),
            args: vec![program_name],
            backend: "souffle".to_string(),
            env: Vec::new(),
        })
    }

    /// The live child's pid, or `0`. Also its process-group id.
    #[cfg(test)]
    pub(crate) fn child_pid(&self) -> u32 {
        self.child_pid.load(Ordering::Relaxed)
    }

    /// Claim the next request id and record it as owed.
    ///
    /// Called with the child's mutex held, so the id the evaluator is working
    /// on is always the one recorded here: ids allocated ahead of the lock
    /// would let a queued caller overwrite the mark belonging to the request
    /// actually in flight.
    fn begin_request(&self) -> u64 {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.outstanding_id.store(id, Ordering::Relaxed);
        id
    }

    async fn call(&self, request: EvalRequest) -> Result<EvalResponse, EvaluatorError> {
        validate_transportable_request(&request)?;
        let mut guard = self.child.lock().await;
        let child = guard.as_mut().ok_or(EvaluatorError::ProcessDied)?;
        let id = self.begin_request();

        match child.call(id, &request).await {
            Ok(EvalResponse::Error { message }) => {
                self.outstanding_id.store(0, Ordering::Relaxed);
                Err(EvaluatorError::QueryError(message))
            }
            Ok(resp) => {
                self.outstanding_id.store(0, Ordering::Relaxed);
                Ok(resp)
            }
            Err(EvaluatorError::ProcessDied) => {
                error!("Evaluator IPC failed, retaining child handle for reaping");
                self.outstanding_id.store(0, Ordering::Relaxed);
                // Dropping here delegates reaping to Tokio's background
                // reaper. A concurrent kill would then see no handle and
                // return before the child has actually been reaped.
                child.stop_unusable();
                Err(EvaluatorError::ProcessDied)
            }
            Err(e) => {
                self.outstanding_id.store(0, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Kill the child and spawn a replacement in its place.
    ///
    /// The outstanding-request mark is cleared: it belonged to the process
    /// that was just killed. Left standing it would mark a brand-new child as
    /// owing a reply it was never asked for, and the session's stall clock
    /// would run against silence that is nobody's fault — up to a kill of a
    /// healthy evaluator.
    pub async fn restart(&self) -> Result<(), EvaluatorError> {
        let mut guard = self.child.lock().await;
        if let Some(child) = guard.as_mut() {
            warn!("Killing existing evaluator process");
            child.kill().await;
        }
        crate::evaluator::reaper::forget(self.child_pid.swap(0, Ordering::Relaxed));
        let args_refs: Vec<&str> = self.config.args.iter().map(|s| s.as_str()).collect();
        let new_child = IpcChild::spawn_with_read_only_files(
            &self.config.program,
            &args_refs,
            &self.config.env,
            &self.read_only_files,
        )?;
        let new_pid = new_child.pid().unwrap_or(0);
        crate::evaluator::reaper::register(new_pid);
        self.child_pid.store(new_pid, Ordering::Relaxed);
        self.outstanding_id.store(0, Ordering::Relaxed);
        *guard = Some(new_child);
        info!("Evaluator process restarted");
        Ok(())
    }

    /// Is the child still running?
    ///
    /// Answering reaps a child that has already exited (`try_wait`), which
    /// frees its pid for the kernel to hand to an unrelated process. So a
    /// "no" also takes the group off the reaper's list: the invariant that
    /// list depends on is that a pid is registered only between spawn and
    /// reap, and a later `kill_all` signalling a recycled number would
    /// `SIGKILL` a process group that has nothing to do with this server.
    ///
    /// Only a CONFIRMED reap retires the pid. Asking can also simply fail, and
    /// that answers "not running" without having reaped anything: forgetting
    /// there would drop a group that may still be alive off the kill list, so
    /// a shutdown would no longer stop it — the same leak, mirrored.
    pub async fn is_alive(&self) -> bool {
        let mut guard = self.child.lock().await;
        match guard.as_mut() {
            Some(child) => {
                let liveness = child.liveness();
                if Self::liveness_retires_pid(liveness) {
                    crate::evaluator::reaper::forget(self.child_pid.swap(0, Ordering::Relaxed));
                }
                liveness == ChildLiveness::Running
            }
            None => false,
        }
    }

    /// Does this answer mean the pid may leave the reaper's shutdown list?
    ///
    /// Yes for a reap and nothing else. A child that is still running must
    /// stay on the list because a shutdown has to kill it; a child whose state
    /// could not be read must stay on it because it may still be running.
    fn liveness_retires_pid(liveness: ChildLiveness) -> bool {
        matches!(liveness, ChildLiveness::Reaped)
    }

    /// Make the child answer `liveness` to every later liveness question, so a
    /// test can drive [`EvaluatorProcess::is_alive`] through an outcome the
    /// kernel will not produce on demand. Tests only.
    #[cfg(test)]
    pub(crate) async fn force_child_liveness(&self, liveness: ChildLiveness) {
        let mut guard = self.child.lock().await;
        guard
            .as_mut()
            .expect("the child handle is still held")
            .force_liveness(liveness);
    }
}

impl Drop for EvaluatorProcess {
    /// Signal the child's group on the way out.
    ///
    /// The last line of defence for the eviction paths: whoever released the
    /// final handle to this evaluator no longer wants it, and an evaluator
    /// nobody wants must not keep running. `kill_on_drop` on the `Child`
    /// handle does not cover it, because under `bwrap` that handle names the
    /// wrapper and the evaluator inside outlives it.
    fn drop(&mut self) {
        self.kill_group_now();
    }
}

#[tonic::async_trait]
impl Evaluator for EvaluatorProcess {
    async fn update(&self, updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
        match self.call(EvalRequest::Update { updates }).await? {
            EvalResponse::UpdateOk => Ok(()),
            other => Err(EvaluatorError::UpdateError(format!(
                "Unexpected response: {}",
                describe_unexpected(&other)
            ))),
        }
    }

    async fn query(&self, request: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
        let expected_actions = request.actions.len();
        let request = EvalRequest::Query(request);
        validate_transportable_request(&request)?;
        let mut guard = self.child.lock().await;
        let child = guard.as_mut().ok_or(EvaluatorError::ProcessDied)?;
        let id = self.begin_request();

        let llm_check = llm_check_fn();
        let result = child
            .call_with_oracle(id, &request, llm_check, &self.oracle)
            .await;
        // Any outcome here means the child answered (or is gone): a reply is
        // no longer owed. An abandoned call never reaches this line, which is
        // exactly the case that must leave the mark standing.
        self.outstanding_id.store(0, Ordering::Relaxed);

        match result {
            Ok(EvalResponse::Error { message }) => Err(EvaluatorError::QueryError(message)),
            Ok(EvalResponse::QueryResult(result)) => {
                validate_query_response(expected_actions, result)
            }
            Ok(other) => Err(EvaluatorError::QueryError(format!(
                "Unexpected response: {}",
                describe_unexpected(&other)
            ))),
            Err(EvaluatorError::ProcessDied) => {
                error!("Evaluator IPC failed, retaining child handle for reaping");
                // Dropping here delegates reaping to Tokio's background
                // reaper. A concurrent kill would then see no handle and
                // return before the child has actually been reaped.
                child.stop_unusable();
                Err(EvaluatorError::ProcessDied)
            }
            Err(e) => Err(e),
        }
    }

    async fn reset(&self) -> Result<(), EvaluatorError> {
        match self.call(EvalRequest::Reset).await? {
            EvalResponse::ResetOk => Ok(()),
            other => Err(EvaluatorError::UpdateError(format!(
                "Unexpected response: {}",
                describe_unexpected(&other)
            ))),
        }
    }

    async fn set_metadata(
        &self,
        facts: Vec<crate::evaluator::types::PolicyMetadataFact>,
    ) -> Result<(), EvaluatorError> {
        // The shim replies `"UpdateOk"` to SetMetadata (same ack as
        // Update). Empty is filtered by the caller, but guard anyway.
        if facts.is_empty() {
            return Ok(());
        }
        match self.call(EvalRequest::SetMetadata { facts }).await? {
            EvalResponse::UpdateOk => Ok(()),
            other => Err(EvaluatorError::UpdateError(format!(
                "Unexpected response: {}",
                describe_unexpected(&other)
            ))),
        }
    }

    async fn kill(&self) {
        // The signal goes out FIRST, from the pid kept outside the mutex.
        // Taking the lock first would mean a concurrent IPC — and one is
        // usually in flight, since a wedged evaluator is exactly what brings
        // us here — swallows the whole kill: the caller's reap deadline
        // expires having sent no signal at all, the handle is never cleared,
        // and the map respawns while the old process is still running and
        // still holding its memory. Under bubblewrap nothing else reaches it
        // either: that process is init in its own PID namespace, so
        // `kill_on_drop` on the wrapper does not touch it. Negative pid to
        // signal the whole group, which `process_group(0)` at spawn made this
        // child the leader of.
        #[cfg(unix)]
        {
            let pid = self.child_pid.load(Ordering::Relaxed);
            if pid != 0 {
                warn!("Killing evaluator process group after the stall window");
                // SAFETY: `kill(2)` on a pid this process spawned. The reap
                // below is what makes it ours to name; nothing else reaps it.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
        }
        // Now the handle. With the process already signalled, any call holding
        // the mutex fails and releases it; if this wait is abandoned, only the
        // reap is lost, never the signal.
        let mut guard = self.child.lock().await;
        if let Some(child) = guard.as_mut() {
            child.kill().await;
        }
        // Clear the handle rather than leaving a dead one: every later `call`
        // then reports `ProcessDied` immediately instead of blocking on a pipe
        // nobody is reading.
        *guard = None;
        self.outstanding_id.store(0, Ordering::Relaxed);
        crate::evaluator::reaper::forget(self.child_pid.swap(0, Ordering::Relaxed));
    }

    /// Kill the child's process group without waiting for anything.
    ///
    /// The synchronous door into the same cleanup `kill` performs, for callers
    /// that have no `await` available — destructors. Eviction reaches an
    /// evaluator this way: a session dropped mid-evaluation (a forced policy
    /// rollout, `EndSession`, a rebind, the idle sweep) never reaches the
    /// stall window that would otherwise call `kill`, and the sandboxed
    /// process behind the wrapper would keep spinning.
    fn kill_group_now(&self) {
        crate::evaluator::reaper::kill_group(self.child_pid.swap(0, Ordering::Relaxed));
    }

    fn outstanding_request(&self) -> Option<u64> {
        match self.outstanding_id.load(Ordering::Relaxed) {
            0 => None,
            id => Some(id),
        }
    }

    fn take_oracle_wait(&self) -> Duration {
        self.oracle.take()
    }

    fn backend_name(&self) -> &str {
        &self.config.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "requires working Linux bubblewrap; mandatory in core release CI"]
    async fn runtime_asset_binds_stay_read_only_and_survive_restart() {
        assert!(
            crate::sandbox::sandbox_available(),
            "working bubblewrap is required"
        );
        let directory = tempfile::tempdir().unwrap();
        let selected = directory.path().join("selected-policy");
        let neighbor = directory.path().join("unselected-private-file");
        std::fs::write(&selected, "selected fixture").unwrap();
        std::fs::write(&neighbor, "private fixture").unwrap();
        let script = r#"
import json, pathlib, struct, sys
selected, neighbor = map(pathlib.Path, sys.argv[1:])
while True:
    header = sys.stdin.buffer.read(12)
    if not header:
        break
    length, ident = struct.unpack('>IQ', header)
    sys.stdin.buffer.read(length)
    try:
        assert selected.read_text() == 'selected fixture'
        assert not neighbor.exists()
        try:
            selected.write_text('changed')
        except OSError:
            pass
        else:
            raise AssertionError('selected file was writable')
        response = 'ResetOk'
    except (OSError, AssertionError):
        response = {'Error': {'message': 'runtime assets were not isolated'}}
    data = json.dumps(response).encode()
    sys.stdout.buffer.write(struct.pack('>IQ', len(data), ident) + data)
    sys.stdout.buffer.flush()
"#;
        let evaluator = EvaluatorProcess::new_with_read_only_files(
            EvaluatorProcessConfig {
                program: "/usr/bin/python3".into(),
                args: vec![
                    "-c".into(),
                    script.into(),
                    selected.display().to_string(),
                    neighbor.display().to_string(),
                ],
                backend: "asset-isolation-probe".into(),
                env: Vec::new(),
            },
            vec![selected.clone()],
        )
        .unwrap();
        for restart in [false, true] {
            if restart {
                evaluator.restart().await.unwrap();
            }
            tokio::time::timeout(Duration::from_secs(10), evaluator.reset())
                .await
                .unwrap()
                .unwrap();
        }
        evaluator.kill().await;
        assert_eq!(
            std::fs::read_to_string(selected).unwrap(),
            "selected fixture"
        );
    }

    /// A stalled evaluator is one that is inside a call, and a call holds the
    /// child mutex for as long as the child stays silent. If the kill waited
    /// for that mutex the signal would never be sent at all: the caller's reap
    /// deadline would expire having done nothing, the handle would stay live,
    /// and the map would respawn alongside a process still holding its core
    /// and its memory — under bubblewrap a process nothing else can reach,
    /// since it is init in its own PID namespace.
    ///
    /// `/bin/sleep` is the wedged evaluator here: it takes the request frame
    /// and never answers, so the call holding the mutex is unfinishable until
    /// the process is signalled.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_kill_reaches_the_process_while_another_call_holds_the_child() {
        let evaluator = Arc::new(
            EvaluatorProcess::new(EvaluatorProcessConfig {
                program: "/bin/sleep".to_string(),
                args: vec!["918273645".to_string()],
                backend: "wedged".to_string(),
                env: Vec::new(),
            })
            .expect("spawn /bin/sleep"),
        );
        let pid = evaluator.child_pid.load(Ordering::Relaxed) as i32;
        assert!(pid > 0, "no pid was recorded for the spawned child");

        // Occupy the mutex the way real traffic does.
        let holder = Arc::clone(&evaluator);
        let holding = tokio::spawn(async move {
            let _ = holder
                .set_metadata(vec![crate::evaluator::types::PolicyMetadataFact {
                    rel: "TrustedDomain".to_string(),
                    a: "example.test".to_string(),
                    b: String::new(),
                }])
                .await;
        });
        // The request mark is set with the child mutex held. Wait for that
        // state rather than assuming the task ran during a fixed sleep.
        tokio::time::timeout(Duration::from_secs(5), async {
            while evaluator.outstanding_request().is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the holder must enter the IPC call");
        assert!(
            !holding.is_finished(),
            "the holder was expected to be stuck"
        );

        let killed =
            tokio::time::timeout(std::time::Duration::from_secs(5), evaluator.kill()).await;
        assert!(
            killed.is_ok(),
            "the kill never got past the mutex the wedged call is holding, so no signal was \
             sent — the process the stall window exists to remove is still running"
        );

        // Signalled AND reaped: `kill(pid, 0)` finding nothing is proof, and
        // the reap is what makes the pid unambiguous.
        // SAFETY: `kill(2)` with signal 0 only tests for the process.
        let still_there = unsafe { libc::kill(pid, 0) } == 0;
        assert!(
            !still_there,
            "the evaluator process {pid} outlived its kill"
        );

        let _ = holding.await;
    }

    /// A lost IPC channel must remain unusable, but its OS handle is still
    /// needed so kill can reap synchronously instead of relying on Tokio's
    /// background reaper after dropping the Child.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_ipc_retains_the_child_until_kill_reaps_it() {
        for query in [false, true] {
            let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
                program: "/usr/bin/python3".into(),
                args: vec![
                    "-c".into(),
                    r#"
import struct, sys, time
length = struct.unpack('>I', sys.stdin.buffer.read(4))[0]
ident = sys.stdin.buffer.read(8)
sys.stdin.buffer.read(length)
sys.stdout.buffer.write(struct.pack('>I', 0xffffffff) + ident)
sys.stdout.buffer.flush()
time.sleep(600)
"#
                    .into(),
                ],
                backend: "broken-ipc".into(),
                env: Vec::new(),
            })
            .expect("spawn the malformed-frame evaluator");
            let pid = evaluator.child_pid();
            let result = tokio::time::timeout(Duration::from_secs(5), async {
                if query {
                    evaluator
                        .query(EvalAuthRequest {
                            current_node_ids: vec![],
                            actions: vec![],
                            entity: None,
                            roles: vec![],
                            tenant_id: Some("synthetic".into()),
                            session_id: None,
                            principal: None,
                            action_metadata: vec![],
                        })
                        .await
                        .map(|_| ())
                } else {
                    evaluator.reset().await
                }
            })
            .await
            .expect("the malformed frame must fail promptly");
            assert!(matches!(result, Err(EvaluatorError::ProcessDied)));
            assert!(
                evaluator.child.lock().await.is_some(),
                "IPC failure dropped the handle before kill could reap it"
            );
            assert!(
                crate::evaluator::reaper::is_registered(pid),
                "the unreaped child must stay on the shutdown list"
            );
            assert!(
                crate::evaluator::reaper::group_is_empty(pid),
                "fatal IPC must stop the child without waiting for another dispatch"
            );
            let retry = tokio::time::timeout(Duration::from_secs(5), evaluator.reset())
                .await
                .expect("later calls must fail without another IPC exchange");
            assert!(matches!(retry, Err(EvaluatorError::ProcessDied)));
            tokio::time::timeout(Duration::from_secs(5), evaluator.kill())
                .await
                .expect("kill must reap the retained child");
            // SAFETY: signal zero checks whether the child PID still exists.
            assert_ne!(
                unsafe { libc::kill(pid as i32, 0) },
                0,
                "the broken evaluator was not reaped"
            );
            assert!(!crate::evaluator::reaper::is_registered(pid));
        }
    }

    /// A live evaluator that nobody holds a handle to any more must not keep
    /// running — and killing the process we hold a handle to is not enough.
    ///
    /// Under `bwrap` that handle names the outer wrapper; the evaluator runs
    /// inside its own PID namespace and outlives the wrapper's death, which is
    /// why the kill goes to the process GROUP. The stand-in here is a shell
    /// that leaves a background child in the same group: `kill_on_drop` reaps
    /// the shell, and only the group signal reaches what it left behind.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_evaluator_stops_the_whole_process_group() {
        let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), "sleep 918273646 & wait".to_string()],
            backend: "wedged".to_string(),
            env: Vec::new(),
        })
        .expect("spawn /bin/sh");
        let pgid = evaluator.child_pid();
        assert!(pgid > 0, "no pid was recorded for the spawned child");
        // Give the shell time to start its background child.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(
            !crate::evaluator::reaper::group_is_empty(pgid),
            "the background child never started, so this pins nothing"
        );

        drop(evaluator);
        assert!(
            crate::evaluator::reaper::group_is_empty(pgid),
            "process group {pgid} outlived the handle to it"
        );
    }

    /// Asking whether a child is alive reaps it if it has already exited, and
    /// a reaped pid must leave the reaper's list at that moment.
    ///
    /// The list is what a shutdown signal walks, sending `SIGKILL` to each
    /// entry's process group. A pid the kernel has taken back can be handed to
    /// any process on the host, so an entry left behind after the reap is an
    /// unrelated process group waiting to be killed by this server's shutdown.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn asking_a_dead_child_whether_it_is_alive_unregisters_its_group() {
        let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
            // Exits on its own the moment it is spawned — an evaluator that
            // died without anybody killing it.
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), "exit 0".to_string()],
            backend: "short-lived".to_string(),
            env: Vec::new(),
        })
        .expect("spawn /bin/sh");
        let pgid = evaluator.child_pid();
        assert!(pgid > 0, "no pid was recorded for the spawned child");
        assert!(
            crate::evaluator::reaper::is_registered(pgid),
            "the spawn must register the group in the first place"
        );

        // Let it exit, then ask — which is what reaps it.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(!evaluator.is_alive().await, "the child was told to exit");

        assert!(
            !crate::evaluator::reaper::is_registered(pgid),
            "pid {pgid} was reaped but is still on the shutdown kill list, so a \
             later shutdown would signal whatever process reuses that number"
        );
    }

    /// Only a confirmed reap retires a pid from the shutdown kill list.
    ///
    /// "Not running" has two causes: the child was reaped, which frees its pid
    /// for the kernel to recycle, and asking failed, which changes nothing. The
    /// first must leave the list, the second must stay on it — a group taken
    /// off the list while its processes may still be running is a group no
    /// shutdown will kill.
    /// The same rule, observed where it acts: through `is_alive`, on the
    /// reaper's list.
    ///
    /// `is_alive` answers "not running" for two different facts. Only one of
    /// them — a confirmed reap — returns the pid to the kernel, and only that
    /// one may take the group off the shutdown kill list. The other, a
    /// question that could not be answered, reaped nothing: the group may
    /// still be running, and a group off the list is a group no shutdown will
    /// kill. Both answers are forced here, because a failing `try_wait` is not
    /// something a test can arrange for real.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unanswerable_liveness_question_leaves_the_group_on_the_kill_list() {
        let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
            // Exits by itself, so nothing is left running once the test drops
            // the handle — the forced answers stand in for its real state.
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), "exit 0".to_string()],
            backend: "short-lived".to_string(),
            env: Vec::new(),
        })
        .expect("spawn /bin/sh");
        let pgid = evaluator.child_pid();
        assert!(pgid > 0, "no pid was recorded for the spawned child");
        assert!(
            crate::evaluator::reaper::is_registered(pgid),
            "the spawn must register the group in the first place"
        );

        evaluator.force_child_liveness(ChildLiveness::Unknown).await;
        assert!(
            !evaluator.is_alive().await,
            "an unanswerable question is not a claim that the child is running"
        );
        assert!(
            crate::evaluator::reaper::is_registered(pgid),
            "pid {pgid} was not reaped — the question just failed — so the group \
             may still be running and must stay on the shutdown kill list"
        );

        evaluator.force_child_liveness(ChildLiveness::Reaped).await;
        assert!(!evaluator.is_alive().await, "a reaped child is not running");
        assert!(
            !crate::evaluator::reaper::is_registered(pgid),
            "pid {pgid} was reaped and can be recycled by the kernel, so a later \
             shutdown must not signal it"
        );
    }

    #[test]
    fn only_a_confirmed_reap_takes_a_group_off_the_kill_list() {
        assert!(
            EvaluatorProcess::liveness_retires_pid(ChildLiveness::Reaped),
            "a reaped pid can be recycled by the kernel, so it must leave the list"
        );
        assert!(
            !EvaluatorProcess::liveness_retires_pid(ChildLiveness::Unknown),
            "asking failed and nothing was reaped, so the group may still be \
             running and a shutdown still has to be able to kill it"
        );
        assert!(
            !EvaluatorProcess::liveness_retires_pid(ChildLiveness::Running),
            "a running child is exactly what the kill list is for"
        );
    }

    /// A restart replaces the child; the unanswered request of the process it
    /// killed does not follow it.
    ///
    /// The mark is what the session's stall clock runs against. Left standing
    /// across a restart it says a brand-new child owes a reply it was never
    /// asked for, and the session would kill a healthy evaluator for silence
    /// that is nobody's.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_restart_clears_the_outstanding_request_of_the_process_it_replaced() {
        let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
            program: "/bin/sleep".to_string(),
            args: vec!["918273645".to_string()],
            backend: "wedged".to_string(),
            env: Vec::new(),
        })
        .expect("spawn /bin/sleep");

        // `/bin/sleep` never answers, so the caller gives up — which is
        // exactly the case that leaves the mark standing on purpose.
        let asked = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            evaluator.query(EvalAuthRequest {
                current_node_ids: vec![],
                actions: vec![],
                entity: None,
                roles: vec![],
                tenant_id: Some("t".to_string()),
                session_id: Some("s".to_string()),
                principal: None,
                action_metadata: vec![],
            }),
        )
        .await;
        assert!(asked.is_err(), "the wedged evaluator answered after all");
        assert!(
            evaluator.outstanding_request().is_some(),
            "an abandoned call must leave the process marked as owing a reply"
        );

        evaluator.restart().await.expect("restart the evaluator");
        assert_eq!(
            evaluator.outstanding_request(),
            None,
            "the new child was marked as owing the dead child's reply"
        );

        evaluator.kill().await;
    }
    #[test]
    fn evaluator_response_requires_one_ordered_result_per_action() {
        let empty = EvalAuthResponse {
            results: Vec::new(),
        };
        let error = validate_query_response(1, empty).unwrap_err();
        assert!(error.to_string().contains("0 results for 1 actions"));

        let mut result: super::super::types::EvalActionResult =
            serde_json::from_value(serde_json::json!({
                "allow_passthrough": false,
                "authorized": false,
                "denial_reasons": [],
                "deny_if_unauthorized": true,
                "index": 1,
                "is_allowlisted": false,
                "is_authenticated": false,
                "is_denylisted": false,
                "requires_approval": false,
                "transform_ids": []
            }))
            .unwrap();
        let error = validate_query_response(
            1,
            EvalAuthResponse {
                results: vec![result.clone()],
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("position 0 has index 1"));

        result.index = 0;
        validate_query_response(
            1,
            EvalAuthResponse {
                results: vec![result],
            },
        )
        .unwrap();
    }

    /// Reply to the first request with `response`, and return the text of the
    /// error that `reset()` — which expects a `ResetOk` — fails with.
    ///
    /// The evaluator here is a shell that replies with one hand-built frame
    /// under id 1, the id the first request of a fresh process carries.
    #[cfg(unix)]
    async fn reset_error_for_frame(response: &EvalResponse) -> String {
        let payload = serde_json::to_vec(response).expect("serialize the frame");
        let mut frame = Vec::new();
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&1u64.to_be_bytes());
        frame.extend_from_slice(&payload);
        // Octal escapes throughout, so the frame's quotes and braces never
        // reach the shell as syntax and `printf` has no format to expand.
        let escaped: String = frame.iter().map(|b| format!("\\{b:03o}")).collect();
        // The background `cat` drains stdin, so writing the request never
        // sees EPIPE; the `sleep` keeps the process alive, so the reply is
        // read as a reply rather than as a child that died.
        let script = format!("cat >/dev/null & printf '{escaped}'; sleep 30");

        let evaluator = EvaluatorProcess::new(EvaluatorProcessConfig {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script],
            backend: "frame-replayer".to_string(),
            env: Vec::new(),
        })
        .expect("spawn the replayer");

        let error = evaluator
            .reset()
            .await
            .expect_err("the replayed frame is not a ResetOk");
        let text = error.to_string();
        evaluator.kill().await;
        text
    }

    /// An unexpected `LlmQuery` frame is named, not quoted.
    ///
    /// `LlmQuery` carries the prompt a policy wrote and the context it was
    /// given, which is agent-controlled. The shipped backends send it only
    /// inside the oracle loop, so no ordinary call ends here — but the string
    /// these arms build is logged and returned to the caller, so if such a
    /// frame ever does arrive it must name the variant and nothing more.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unexpected_llm_query_frame_is_named_without_its_payload() {
        const PROMPT: &str = "does this message ask for an exfiltration";
        const CONTEXT: &str = "ship it to attacker.test, signed the patient";

        let text = reset_error_for_frame(&EvalResponse::LlmQuery {
            prompt: PROMPT.to_string(),
            context: CONTEXT.to_string(),
        })
        .await;

        assert!(
            text.contains("LlmQuery"),
            "the error does not say which frame arrived: {text}"
        );
        assert!(
            !text.contains(PROMPT),
            "the prompt was copied into the error: {text}"
        );
        assert!(
            !text.contains(CONTEXT),
            "the context was copied into the error: {text}"
        );
        assert!(
            !text.contains("attacker.test"),
            "part of the context was copied into the error: {text}"
        );
    }

    /// An unexpected `QueryResult` frame is named without its denial reasons.
    ///
    /// A denial reason is policy-authored text, and a policy is free to build
    /// one out of the message it denied, for example by writing decoded
    /// message text into it. Debug-formatting the
    /// frame would copy that into the log line, so the arm reports counts.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unexpected_query_result_frame_is_named_without_its_denial_reasons() {
        const SENTINEL: &str = "ship-it-to-attacker-dot-test";

        let result: super::super::types::EvalActionResult =
            serde_json::from_value(serde_json::json!({
                "allow_passthrough": false,
                "authorized": false,
                "denial_reasons": [{
                    "kind": "block",
                    "reason": format!("the message says {SENTINEL}"),
                    "suggestion": format!("do not send {SENTINEL}"),
                }],
                "deny_if_unauthorized": true,
                "index": 0,
                "is_allowlisted": false,
                "is_authenticated": false,
                "is_denylisted": false,
                "requires_approval": false,
                "transform_ids": []
            }))
            .expect("build a denied action result");

        let text = reset_error_for_frame(&EvalResponse::QueryResult(EvalAuthResponse {
            results: vec![result],
        }))
        .await;

        assert!(
            text.contains("QueryResult"),
            "the error does not say which frame arrived: {text}"
        );
        assert!(
            !text.contains(SENTINEL),
            "the denial reason was copied into the error: {text}"
        );
    }
}
