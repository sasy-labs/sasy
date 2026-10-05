//! Evaluator abstraction for Datalog backends.
//!
//! Defines a backend-agnostic interface for policy evaluation.
//! Implementations are out-of-process evaluators communicating over
//! stdin/stdout pipes (Soufflé, FlowLog).

#[cfg(feature = "compiler")]
pub(crate) mod factory;
pub mod ipc;
pub mod manager;
pub mod protocol;
pub mod reaper;
pub mod sandbox;
pub mod types;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::engine::GraphUpdate;
use types::{EvalAuthRequest, EvalAuthResponse, PolicyMetadataFact};

/// Measures the time a query spends waiting on the LLM oracle.
///
/// An `@llm_check` cache miss suspends the evaluation while an external
/// provider is asked, and the answer comes back through the same pipe as the
/// query. That wait is somebody else's API being slow, not this evaluator
/// spinning, so the session's query budget stops for it — see
/// [`Evaluator::take_oracle_wait`], which is how the session task reads this.
///
/// The clock counts the part of a call already spent as well as calls that
/// have finished, because the budget it feeds can expire in the middle of one:
/// counting only completed calls would make an in-flight oracle look like a
/// wedged evaluator.
#[derive(Default)]
pub struct OracleClock {
    /// Waiting that is over and has not been drained yet.
    accrued_us: AtomicU64,
    /// When the callback now running last started counting. Reset by a drain,
    /// so the same microsecond is never handed out twice.
    running_since: Mutex<Option<Instant>>,
}

impl OracleClock {
    /// A callback is about to be handed to the provider.
    pub fn start(&self) {
        *self.running_since.lock() = Some(Instant::now());
    }

    /// The provider answered (or failed): fold the wait into the total.
    pub fn finish(&self) {
        if let Some(started) = self.running_since.lock().take() {
            self.accrued_us
                .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    /// Everything waited since the last drain, including the part of a call
    /// that is still running.
    pub fn take(&self) -> Duration {
        let mut running = self.running_since.lock();
        let mut total = self.accrued_us.swap(0, Ordering::Relaxed);
        if let Some(started) = running.as_mut() {
            total += started.elapsed().as_micros() as u64;
            *started = Instant::now();
        }
        Duration::from_micros(total)
    }
}

/// Errors from evaluator operations.
#[derive(thiserror::Error, Debug)]
pub enum EvaluatorError {
    #[error("Evaluator initialization failed: {0}")]
    InitError(String),
    #[error("Update failed: {0}")]
    UpdateError(String),
    #[error("Query failed: {0}")]
    QueryError(String),
    #[error("Evaluator process died")]
    ProcessDied,
    #[error("IPC error: {0}")]
    IpcError(String),
}

/// Backend-agnostic interface for Datalog policy evaluation.
///
/// Graph state (Edge, SentMessage, ToolResult) is maintained by the
/// evaluator between calls. Authorization queries are stateless from
/// the caller's perspective — instance relations (Current, Actions, etc.)
/// are parameters of the query, not persistent state.
#[tonic::async_trait]
pub trait Evaluator: Send + Sync {
    /// Apply a batch of graph updates (insert/delete of edges and nodes).
    async fn update(&self, updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError>;

    /// Evaluate an authorization query against the current graph state.
    async fn query(&self, request: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError>;

    /// Clear all graph state.
    async fn reset(&self) -> Result<(), EvaluatorError>;

    /// Seed the static `PolicyMetadata(rel, a, b)` EDB. Sent once
    /// after spawn, before bootstrap; constant for the evaluator's
    /// life (config decoupled from the compiled policy, so a
    /// precompiled/restricted evaluator accepts it without
    /// recompiling). Default no-op for backends without config facts.
    async fn set_metadata(&self, _facts: Vec<PolicyMetadataFact>) -> Result<(), EvaluatorError> {
        Ok(())
    }

    /// Stop this evaluator for good.
    ///
    /// Called when the evaluator has owed a reply for longer than the stall
    /// window. A compiled Datalog program has no cancellation point to ask it
    /// to stop at, and abandoning the call left a frame half-read on its pipe
    /// anyway, so killing it is what lets the session respawn onto a clean
    /// one. Default no-op for backends with no process behind them.
    async fn kill(&self) {}

    /// Stop this evaluator from a destructor.
    ///
    /// Same intent as [`Evaluator::kill`], minus the wait: a destructor cannot
    /// await, and the wait is only the reap. What matters is that the signal
    /// goes out, because the process behind the sandbox wrapper does not stop
    /// on its own — see [`crate::evaluator::reaper`].
    ///
    /// Called when a session evaluator is dropped, which is how eviction
    /// reaches an evaluator that is still evaluating: a forced policy rollout,
    /// `EndSession`, a session rebind or the idle sweep all drop the session
    /// without ever reaching the stall window that calls `kill`. Default no-op
    /// for backends with no process behind them.
    fn kill_group_now(&self) {}

    /// The id of a request this evaluator was given and has not answered, if
    /// any.
    ///
    /// Set while a call is in flight and, crucially, left set when a caller
    /// abandons one: the session evaluator reads it to tell "still working"
    /// from "answered and idle" after a query blew its budget, and names it in
    /// the log when the stall window runs out. `None` for a backend with no
    /// process behind it, which cannot owe anything.
    fn outstanding_request(&self) -> Option<u64> {
        None
    }

    /// Time this evaluator has spent waiting on the LLM oracle since the last
    /// call, which resets the count to zero.
    ///
    /// The session's query budget is a bound on the evaluator, and an oracle
    /// round trip is not the evaluator: `@llm_check` hands the prompt to a
    /// provider whose own timeout (`LLM_TIMEOUT_SECS`, 30s by default) is
    /// longer than the budget, so a healthy check against a slow provider would
    /// otherwise be denied every time the prompt missed the cache. The session
    /// task stops its budget clock for exactly the time reported here, up to
    /// `EvaluationDeadlines::oracle_bound`. The stall window does NOT stop: an
    /// oracle path that never comes back is still a wedged evaluator and is
    /// still killed.
    ///
    /// Default zero: a backend with no oracle never waits on one.
    fn take_oracle_wait(&self) -> Duration {
        Duration::ZERO
    }

    /// Name of the backend (for logging/status).
    fn backend_name(&self) -> &str;
}
