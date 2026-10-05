//! Per-session policy evaluator.
//!
//! Each live `(tenant, session)` scope has its own
//! [`SessionEvaluator`] task wrapping a single [`Evaluator`]
//! (Soufflé subprocess). The task:
//!
//! 1. Bootstraps the evaluator with the scope's existing graph
//!    state via [`GraphStore::get_session_state`] (so resumption
//!    after eviction sees the same EDB it would have if the
//!    evaluator had been alive the whole time).
//! 2. Subscribes to graph store broadcasts and processes only
//!    updates carrying its own scope; cross-scope (and
//!    cross-tenant) broadcasts are dropped at the filter and
//!    never reach the evaluator. ``Sequence`` markers are applied
//!    unconditionally so the seq fence still works.
//! 3. Buffers updates and flushes them to the evaluator either on a
//!    short background tick or just before each query.
//! 4. Picks queries from a bounded async-channel and runs them.
//!
//! Compared with the previous `WorkerPool` model:
//! * No cross-scope broadcast catch-up — flush only ships this
//!   scope's events.
//! * No session-affinity hashing or work-stealing — scopes own
//!   their evaluators directly; the dispatch map looks up the
//!   evaluator by `(tenant, session)`.
//! * The sequence fence is preserved because intra-session parallel
//!   tool calls still need it.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use sasy_common::SessionScope;
use sasy_graph::GraphStore;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use crate::engine::GraphUpdate;
use crate::evaluator::protocol::validate_transportable_request;
use crate::evaluator::types::{EvalAuthRequest, EvalAuthResponse, EvalRequest, PolicyMetadataFact};
use crate::evaluator::{Evaluator, EvaluatorError};
use crate::policy_registry::{PolicyId, PolicyRegistry};
use crate::sync::{broadcast_to_updates, full_state_scoped_to_updates, full_state_to_updates};

/// Default seconds a session evaluator may sit idle before the
/// eviction sweep drops its subprocess. Graph state in the store
/// is preserved either way, so the next query for the same
/// session re-spawns and re-bootstraps. ``0`` disables eviction.
///
/// 30 minutes, because an interactive agent session idles for many
/// minutes between tool calls (model thinking time, the user reading)
/// while still being alive, and each eviction costs a re-bootstrap that
/// replays the whole session graph, which grows with session length.
/// Re-bootstrap is lossless (RocksDB-backed), so the only cost of a
/// longer TTL is holding an idle subprocess; memory-bound multi-session
/// servers can lower this (or cap count) via `SASY_SESSION_IDLE_TTL_SECS`
/// / `SASY_MAX_LIVE_SESSIONS`.
const DEFAULT_IDLE_TTL_SECS: u64 = 1800; // 30 minutes

/// Default seconds between eviction sweeps. ``0`` disables the
/// sweep regardless of TTL.
const DEFAULT_SWEEP_INTERVAL_SECS: u64 = 60;

/// Upper bound on how long a query's sequence-fence waits for the broadcast
/// marker that would satisfy `min_sequence` before giving up. The fence normally
/// clears in microseconds (the awaited event's marker is already in the ring);
/// this only bites the pathology where the marker is NEVER broadcast — e.g. a
/// writer bumped the shard sequence but its `commit_batch` errored before the
/// send, then the session went idle. An unbounded `recv().await` there hangs the
/// CheckToolCall RPC indefinitely; a bounded wait ends in a DENY, which is what
/// "the evaluator cannot yet see the write you just made" honestly is. The
/// evaluator is left running and consuming updates, so a later query for the
/// same session is answered normally. Well under typical agent-hook timeouts.
const SEQ_FENCE_TIMEOUT: Duration = Duration::from_secs(2);

/// Default wall-clock budget for one dispatched job — the query and the
/// evaluator IPCs the same caller is blocked on behind it. See
/// [`EvaluationDeadlines`] and the `--query-timeout-secs` CLI flag.
pub const DEFAULT_QUERY_TIMEOUT_SECS: u64 = 10;

/// Default stall window: how long the evaluator may owe a reply before it is
/// presumed unable to answer anything and killed. See [`EvaluationDeadlines`].
pub const DEFAULT_EVALUATOR_STALL_SECS: u64 = 60;

/// Default bound on how long one job's budget clock may be stopped for the
/// LLM oracle.
///
/// The provider client carries its own timeout — `LLM_TIMEOUT_SECS`, 30s by
/// default — and this is that number, read from the same variable when it is
/// set. It exists separately because the provider's timeout lives inside an
/// optional feature and inside a callback the session task cannot see, and a
/// bound the task does not hold is not a bound on the task.
pub const DEFAULT_ORACLE_WAIT_SECS: u64 = 30;

/// How many kills a session may take inside [`KILL_WINDOW`] before its
/// evaluator stops being respawned.
///
/// Not a flag: it is a shape, not a tuning knob. A deterministic runaway — a
/// rule that does not terminate on an input the agent keeps producing — kills
/// the process again on every repetition, and each kill costs a spawn and a
/// re-bootstrap of the session's whole graph. Past the cap the session is
/// simply refused until the window passes, which is the same fail-closed
/// answer at a fraction of the cost. Three is enough to ride out a transient
/// wedge and few enough that a runaway stops churning quickly.
const KILL_CAP: usize = 3;

/// The window [`KILL_CAP`] counts kills in. Long enough that three kills
/// inside it really is a pattern rather than three unrelated bad queries, and
/// short enough that a session recovers on its own after a genuine incident.
const KILL_WINDOW: Duration = Duration::from_secs(600);

/// How much longer than the query budget a caller waits on `dispatch` before
/// answering itself.
///
/// The task is what normally answers: it bounds the query, the fence, the
/// flush and the re-sync, and its deny names what was being decided. But the
/// task's own bounds only apply to work it started for THIS job — a caller
/// whose job is still queued behind a bootstrap, a re-sync or a probe is
/// bounded by nothing on that path, and `dispatch` would wait for it
/// indefinitely. The grace is what keeps the task's answer the one callers
/// normally get, while making the budget a real bound on every path into the
/// engine rather than on the query alone.
const DISPATCH_GRACE: Duration = Duration::from_secs(1);

/// How long the kill path waits for the killed evaluator to be reaped.
///
/// `Evaluator::kill` signals the process group and then awaits `wait()`. The
/// SIGKILL is delivered by the time that blocks, so a child that cannot be
/// reaped promptly (uninterruptible sleep on a slow mount, a wrapper waiting on
/// a forked grandchild) costs only a zombie if the reap is abandoned — whereas
/// waiting forever would hold this task exactly as long as the wedge it is
/// clearing.
const KILL_REAP_DEADLINE: Duration = Duration::from_secs(2);

/// Default idle TTL for unreferenced *policy variants* in the
/// [`PolicyRegistry`]. After this long with no
/// `last_referenced` activity, a non-default policy entry is
/// dropped by the orphan sweep. Defaults are exempt.
const DEFAULT_POLICY_IDLE_TTL_SECS: u64 = 3600;

/// The two wall-clock bounds on one session evaluator, set by
/// `--query-timeout-secs` and `--evaluator-stall-secs`.
///
/// * **query budget** — how long a caller waits for a decision. A job that
///   overruns it is denied (fail closed) and the evaluator is left running:
///   it is busy, not broken, and a functor that is slow on some inputs then
///   costs exactly the queries that touch them.
/// * **stall window** — how long the evaluator may owe a reply before it is
///   presumed unable to answer anything. Past it the process is killed and
///   respawned, because a compiled Datalog program has no cancellation point
///   to ask it to stop at.
///
/// The stall window is never shorter than the query budget: a query that has
/// not overrun its own budget yet cannot be evidence that the process is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationDeadlines {
    pub query_budget: Duration,
    pub stall_window: Duration,
    /// How much of one job's budget the LLM oracle may consume without
    /// counting against it.
    ///
    /// An `@llm_check` cache miss suspends the evaluation while an external
    /// provider is asked, over the same pipe. That wait is not the evaluator
    /// spinning, so the budget clock stops for it — but only up to this much,
    /// so a provider that never answers cannot hold a caller for ever. Past it
    /// the query is denied and the evaluator is kept, exactly as a budget
    /// miss — or sooner, when the stall deadline arrives first: the allowance
    /// ends at whichever of the two comes first, and when it is the stall
    /// deadline the evaluator is killed rather than kept, because an oracle
    /// path that never comes back is a wedged evaluator.
    pub oracle_bound: Duration,
}

impl Default for EvaluationDeadlines {
    fn default() -> Self {
        Self {
            query_budget: Duration::from_secs(DEFAULT_QUERY_TIMEOUT_SECS),
            stall_window: Duration::from_secs(DEFAULT_EVALUATOR_STALL_SECS),
            oracle_bound: Duration::from_secs(DEFAULT_ORACLE_WAIT_SECS),
        }
    }
}

impl EvaluationDeadlines {
    /// Build a pair, refusing a zero query budget, or a stall window below it.
    ///
    /// The binary calls this while parsing its flags so the refusal happens at
    /// startup, where an operator sees it, rather than as a surprise kill on
    /// the first slow query.
    pub fn new(query_budget: Duration, stall_window: Duration) -> Result<Self, String> {
        if query_budget.is_zero() {
            return Err(
                "a query budget of 0 is not \"no timeout\": every evaluation would overrun \
                 it and every check would be denied. Set a positive number of seconds; \
                 there is no way to switch the budget off"
                    .to_string(),
            );
        }
        if stall_window < query_budget {
            return Err(format!(
                "evaluator stall window ({}s) is below the query budget ({}s): \
                 a query that has not yet overrun its budget would be treated \
                 as a dead process",
                stall_window.as_secs_f64(),
                query_budget.as_secs_f64(),
            ));
        }
        Ok(Self {
            query_budget,
            stall_window,
            oracle_bound: oracle_bound_from_env(),
        })
    }

    /// The same pair with the documented invariant made true.
    ///
    /// The fields are public, so a caller can build a pair directly and skip
    /// the check in [`EvaluationDeadlines::new`]. A stall window below the
    /// query budget means the evaluator is killed for a query that has not
    /// even overrun its own budget yet, which the type's own documentation
    /// says cannot happen. Rather than let that through, the window is raised
    /// to the budget — neither bound the caller asked for is shortened — and
    /// the correction is logged.
    ///
    /// Applied where a pair is put to work, so it holds however the pair was
    /// built.
    #[must_use]
    pub fn normalized(self) -> Self {
        if self.stall_window >= self.query_budget {
            return self;
        }
        warn!(
            "evaluator stall window ({}) is below the query budget ({}) — using the query \
             budget as the stall window",
            human_bound(self.stall_window),
            human_bound(self.query_budget),
        );
        Self {
            stall_window: self.query_budget,
            ..self
        }
    }

    /// The pair a library caller gets with no configuration: the defaults,
    /// each overridable by its environment variable. The binary's
    /// `--query-timeout-secs` / `--evaluator-stall-secs` flags read the same
    /// variables, so a server started through `sasy serve` and one embedding
    /// this crate agree.
    pub fn from_env() -> Self {
        let default = Self::default();
        let query_budget = env_secs("SASY_QUERY_TIMEOUT_SECS", default.query_budget);
        let stall_window = env_secs("SASY_EVALUATOR_STALL_SECS", default.stall_window);
        match Self::new(query_budget, stall_window) {
            Ok(d) => d,
            Err(e) => {
                // A library default cannot refuse to start. Take the larger of
                // the two as the stall window: it keeps the invariant without
                // shortening either bound the operator asked for.
                warn!("{e} — using the query budget as the stall window");
                Self {
                    query_budget,
                    stall_window: query_budget,
                    oracle_bound: oracle_bound_from_env(),
                }
            }
        }
    }
}

/// The oracle bound: the LLM client's own timeout when one is configured,
/// [`DEFAULT_ORACLE_WAIT_SECS`] otherwise. Read from `LLM_TIMEOUT_SECS`, the
/// same variable `LlmConfig::from_env` reads, so the budget stops for exactly
/// as long as the provider is allowed to take.
fn oracle_bound_from_env() -> Duration {
    env_secs(
        "LLM_TIMEOUT_SECS",
        Duration::from_secs(DEFAULT_ORACLE_WAIT_SECS),
    )
}

/// A bound as it is written into a message a caller reads.
///
/// Seconds to one decimal at a second and above, milliseconds below it: a
/// deadline of 800ms printed with `{:.0}s` reads "1s", and one of 80ms reads
/// "0s" — a bound of zero, which is not what happened and not something an
/// operator can act on.
fn human_bound(d: Duration) -> String {
    if d < Duration::from_secs(1) {
        format!("{}ms", d.as_millis())
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

fn env_secs(var: &str, default: Duration) -> Duration {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(default)
}

fn resolve_idle_ttl() -> Duration {
    let secs = std::env::var("SASY_SESSION_IDLE_TTL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_IDLE_TTL_SECS);
    Duration::from_secs(secs)
}

/// Interval between empty-update keepalive IPCs to each session
/// evaluator, in milliseconds. ``0`` (default) disables the
/// keepalive entirely. It exists because long idle gaps
/// between queries (typical at low concurrency where
/// agents spend seconds in LLM thinking time) let the OS demote
/// the Soufflé subprocess into a deep C-state / lower P-state
/// and evict its working set, contributing to the p99 tail. A
/// periodic empty ``evaluator.update`` keeps the process scheduled
/// and the IPC code path warm without touching graph state.
fn resolve_keepalive_interval() -> Duration {
    let ms = std::env::var("SASY_KEEPALIVE_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    Duration::from_millis(ms)
}

fn resolve_sweep_interval() -> Duration {
    let secs = std::env::var("SASY_SESSION_SWEEP_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SWEEP_INTERVAL_SECS);
    Duration::from_secs(secs)
}

/// Idle TTL for unreferenced policy variants in the per-tenant
/// [`PolicyRegistry`]. ``0`` disables the orphan sweep.
fn resolve_policy_idle_ttl() -> Duration {
    let secs = std::env::var("SASY_POLICY_VARIANT_IDLE_TTL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_POLICY_IDLE_TTL_SECS);
    Duration::from_secs(secs)
}

/// Hard cap on the number of live per-session evaluators. ``None``
/// (the default) leaves the map unbounded — current behavior. When
/// set, [`SessionEvaluatorMap::get_or_spawn`] evicts the least
/// recently active entry before inserting a new one if the map is
/// already at the cap. Eviction preserves graph state in the store,
/// so an evicted session that gets later traffic transparently
/// re-bootstraps. ``0`` is treated as unbounded (consistent with how
/// ``0`` disables the idle-TTL sweep).
fn resolve_max_live_sessions() -> Option<usize> {
    std::env::var("SASY_MAX_LIVE_SESSIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// One session's record of evaluator kills.
///
/// Owned by the [`SessionEvaluatorMap`] and shared with the session's task,
/// because the task cannot count these itself: the kill it would be counting
/// is the same kill that ends it, so a per-task counter resets to zero exactly
/// when it would start to matter.
#[derive(Default)]
pub struct KillLedger {
    kills: Mutex<std::collections::VecDeque<Instant>>,
}

impl KillLedger {
    /// Record a kill, and report how many are now inside the window.
    fn record(&self) -> usize {
        let mut kills = self.kills.lock();
        kills.push_back(Instant::now());
        Self::prune(&mut kills);
        kills.len()
    }

    /// `Some(remaining)` while the window still holds [`KILL_CAP`] kills —
    /// how long until the oldest of them ages out and the session may spawn
    /// an evaluator again.
    fn blocked_for(&self) -> Option<Duration> {
        let mut kills = self.kills.lock();
        Self::prune(&mut kills);
        if kills.len() < KILL_CAP {
            return None;
        }
        kills
            .front()
            .map(|first| KILL_WINDOW.saturating_sub(first.elapsed()))
    }

    /// True once every recorded kill has aged out — the ledger holds nothing
    /// and the map can forget it.
    fn is_spent(&self) -> bool {
        let mut kills = self.kills.lock();
        Self::prune(&mut kills);
        kills.is_empty()
    }

    fn prune(kills: &mut std::collections::VecDeque<Instant>) {
        while kills.front().is_some_and(|t| t.elapsed() >= KILL_WINDOW) {
            kills.pop_front();
        }
    }

    /// Test-only: take the ledger's lock, which stops the next [`Self::record`]
    /// inside itself. A test holds this to see what the rest of the process can
    /// observe while a kill is only half-way through being published.
    #[cfg(test)]
    fn block_records(&self) -> parking_lot::MutexGuard<'_, std::collections::VecDeque<Instant>> {
        self.kills.lock()
    }

    /// Test-only: how many kills are recorded right now.
    #[cfg(test)]
    fn recorded(&self) -> usize {
        self.kills.lock().len()
    }
}

/// Every session's kill ledger, keyed by scope.
///
/// Held in its own `Arc` rather than inline in the [`SessionEvaluatorMap`] so
/// a session's task can reach the table itself, not only the one ledger it was
/// handed at spawn. That difference is what makes the cap work: the ledger is
/// resolved by scope at the moment a kill is recorded, so the record always
/// lands in the ledger the next spawn's cap check will read, even when the
/// entry the task started with is no longer the one in the table.
#[derive(Default)]
pub struct KillLedgers {
    inner: RwLock<HashMap<SessionScope, Arc<KillLedger>>>,
}

impl KillLedgers {
    /// This session's ledger, created on first use.
    fn for_scope(&self, scope: &SessionScope) -> Arc<KillLedger> {
        if let Some(l) = self.inner.read().get(scope) {
            return Arc::clone(l);
        }
        Arc::clone(self.inner.write().entry(scope.clone()).or_default())
    }

    /// Drop the ledgers that say nothing any more: every kill aged out AND no
    /// session task still holding the entry.
    ///
    /// The second condition is not housekeeping, it is the cap. A ledger is
    /// "spent" from the moment it is created — a session that has never been
    /// killed has no kills in the window — so a sweep that pruned on
    /// emptiness alone would drop the entry of every healthy live session on
    /// its first tick, and each later kill would be recorded into a ledger the
    /// respawn's cap check no longer reads. Under the production defaults (a
    /// 60s sweep inside a 60s stall window) that is every kill.
    fn prune_spent(&self) {
        self.inner
            .write()
            .retain(|_, l| Arc::strong_count(l) > 1 || !l.is_spent());
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.read().len()
    }
}

/// A session task's route to its kill ledger.
///
/// Carries the table and the scope, so the ledger is resolved when a kill is
/// recorded, and an `Arc` to the ledger the session spawned against, so
/// [`KillLedgers::prune_spent`] can see that a task still holds it.
#[derive(Clone)]
pub struct KillLedgerRef {
    ledgers: Arc<KillLedgers>,
    scope: SessionScope,
    /// Never read: held only so its reference count is above one for as long
    /// as this session's task lives.
    _spawned_with: Arc<KillLedger>,
}

impl KillLedgerRef {
    fn new(ledgers: Arc<KillLedgers>, scope: SessionScope) -> Self {
        let spawned_with = ledgers.for_scope(&scope);
        Self {
            ledgers,
            scope,
            _spawned_with: spawned_with,
        }
    }

    /// Record a kill in the table's current ledger for this scope, and report
    /// how many are now inside the window.
    fn record(&self) -> usize {
        self.ledgers.for_scope(&self.scope).record()
    }
}

/// Builds fresh per-session [`Evaluator`] instances on demand.
///
/// Each invocation must produce an *independent* evaluator (its own
/// Soufflé subprocess for production backends), since per-session
/// isolation depends on each session having its own state. The
/// upload pipeline captures the compiled-policy config in this
/// closure so a new session can spawn a fresh process at any time.
pub type EvaluatorFactory =
    Arc<dyn Fn() -> Result<Arc<dyn Evaluator>, EvaluatorError> + Send + Sync>;

/// Result returned by a per-session evaluator for a query.
pub struct QueryResult {
    pub eval_response: EvalAuthResponse,
    /// Microseconds spent draining broadcasts at the seq fence
    /// before evaluation. Usually 0 once intra-session events have
    /// caught up.
    pub sync_wait_us: u64,
    /// Microseconds spent shipping buffered ``GraphUpdate`` batches
    /// to the evaluator (background tick + pre-query flush). Per
    /// session, so cross-session traffic does not contribute.
    pub flush_us: u64,
    /// Microseconds spent in the evaluator query.
    pub eval_us: u64,
    /// Snapshot of this session's node/edge counts at evaluation
    /// time (this evaluator's view, not the global graph). Kept on
    /// every result so the timing log and tests have one shape.
    pub graph_nodes: u64,
    pub graph_edges: u64,
    /// Per-session counts pulled from the graph store (matches
    /// the legacy timing-log shape so the SDK and analyses keep
    /// reading the same numbers). Always populated for non-global
    /// scopes; ``None`` only when the request scope was global.
    pub session_nodes: Option<u64>,
    pub session_edges: Option<u64>,
    /// The policy the dispatch actually resolved to — the binding the
    /// map selected (explicit pin → session binding → tenant
    /// default). `None` on the inner per-task result before the map
    /// attaches it in [`SessionEvaluatorMap::dispatch`]. The
    /// denial-trace builder selects per-policy rule metadata by this
    /// id, so a tenant running multiple policies attributes each
    /// deny message to the policy that actually decided it.
    pub resolved_policy_id: Option<PolicyId>,
}

/// What a [`SessionEvaluatorTask::catch_up_from_replay`] attempt achieved.
enum CatchUp {
    /// The gap was closed incrementally. No reload needed.
    Applied,
    /// The gap reaches back further than the store retains, or this scope
    /// cannot be replayed at all. Reload.
    NeedsResync,
    /// The evaluator subprocess died. Exit the task.
    Dead,
}

struct QueryJob {
    request: EvalAuthRequest,
    min_sequence: i64,
    respond: tokio::sync::oneshot::Sender<Result<QueryResult, String>>,
}

/// Per-session policy evaluator handle. Owns a tokio task and an
/// evaluator; dropping aborts the task and (when the inner Arc
/// drops to zero) closes the underlying evaluator subprocess.
pub struct SessionEvaluator {
    scope: SessionScope,
    backend_name: String,
    query_tx: async_channel::Sender<QueryJob>,
    /// Shared with the task so the map can re-seed PolicyMetadata into the
    /// live evaluator mid-session (dynamic facts via `UpdatePolicyMetadata`).
    /// IPC calls serialize on the evaluator's child mutex, so this races
    /// safely with the task's queries.
    evaluator: Arc<dyn Evaluator>,
    /// Shared with the task so both `dispatch` (call-site) and the
    /// task's broadcast loop bump last activity. Otherwise a session
    /// that's only receiving events (no auth queries yet) would look
    /// idle to the eviction sweep and churn through evict→prewarm
    /// cycles every TTL period.
    last_active: Arc<Mutex<Instant>>,
    /// Set to `true` by the task right before it exits due to a
    /// dead subprocess. [`SessionEvaluatorMap::dispatch`] reads it
    /// after each call and evicts the live entry on death so the
    /// next request respawns and re-bootstraps from RocksDB.
    process_died: Arc<std::sync::atomic::AtomicBool>,
    /// The same bounds the task runs under, so `dispatch` can answer a caller
    /// whose job the task has not reached yet. See [`DISPATCH_GRACE`].
    deadlines: EvaluationDeadlines,
    /// Budget clock the task has stopped for the LLM oracle, in microseconds,
    /// cumulative for this evaluator's life. Read by `dispatch` so a caller's
    /// bound stops for the same wait — see [`OracleCredit`].
    oracle_credit_us: Arc<std::sync::atomic::AtomicU64>,
    /// Test seam: how many callers have read `oracle_credit_us` as their
    /// starting point. A caller counts only credit granted AFTER that read —
    /// anything already on the counter is somebody else's wait and is folded
    /// into its base — so a test that grants credit by hand has to know the
    /// read has happened. Without this it can only guess, and a grant that
    /// lands first is silently invisible.
    #[cfg(test)]
    credit_base_reads: Arc<std::sync::atomic::AtomicU64>,
    /// Count of full reloads this task has performed. A reload is O(shard) and
    /// is what the replay catch-up exists to avoid, so tests assert on it
    /// directly rather than on a log line. Read only there.
    #[cfg_attr(not(test), allow(dead_code))]
    full_resyncs: Arc<AtomicU64>,
    /// Count of gaps this task closed from the store's replay log. Read only
    /// by tests, which need it to prove the lag path ran at all.
    #[cfg_attr(not(test), allow(dead_code))]
    catch_ups: Arc<AtomicU64>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for SessionEvaluator {
    /// Stop the task, and the subprocess it was driving.
    ///
    /// Aborting the task alone is not enough. The task is the only thing that
    /// ever calls [`Evaluator::kill`] (from `kill_stalled`), so aborting it
    /// while an evaluation is in flight destroys the one thing that would have
    /// killed the evaluator. Every eviction path lands here — a forced policy
    /// rollout, `EndSession`, a session rebind, the idle sweep — and none of
    /// them waits for the stall window first. The subprocess would keep
    /// spinning on a query whose answer nobody is waiting for, holding a core
    /// and its 2 GB address-space limit for the life of the host.
    ///
    /// An idle evaluator is not affected either way: it exits on its own when
    /// its stdin closes.
    fn drop(&mut self) {
        self.evaluator.kill_group_now();
        self.task.abort();
    }
}

impl SessionEvaluator {
    /// Spawn a new per-session evaluator task. Bootstraps from the
    /// graph store's current per-session state so a resumed (or
    /// freshly-created) session sees its full history immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        scope: SessionScope,
        evaluator: Arc<dyn Evaluator>,
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
        metadata: Vec<PolicyMetadataFact>,
        deadlines: EvaluationDeadlines,
        policy_hash: String,
        kills: KillLedgerRef,
    ) -> Self {
        // However this pair was built — `new`, `from_env`, or a struct
        // literal that skipped both — the stall window is not shorter than the
        // query budget by the time it bounds anything. See
        // [`EvaluationDeadlines::normalized`].
        let deadlines = deadlines.normalized();
        let backend_name = evaluator.backend_name().to_string();
        let (qtx, qrx) = async_channel::bounded(query_capacity.max(1));
        let broadcast_rx = graph_store.subscribe();
        let last_active = Arc::new(Mutex::new(Instant::now()));
        let process_died = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let oracle_credit_us = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let full_resyncs = Arc::new(AtomicU64::new(0));
        let catch_ups = Arc::new(AtomicU64::new(0));
        let keepalive_interval = resolve_keepalive_interval();

        let task_scope = scope.clone();
        let task_last_active = Arc::clone(&last_active);
        let task_process_died = Arc::clone(&process_died);
        let task_oracle_credit_us = Arc::clone(&oracle_credit_us);
        let task_full_resyncs = Arc::clone(&full_resyncs);
        let task_catch_ups = Arc::clone(&catch_ups);
        let handle_evaluator = Arc::clone(&evaluator);
        let task = tokio::spawn(async move {
            let task = SessionEvaluatorTask {
                scope: task_scope,
                evaluator,
                graph_store,
                broadcast_rx,
                query_rx: qrx,
                sequence: 0,
                snapshot_seq: 0,
                graph_incomplete: false,
                node_count: 0,
                edge_count: 0,
                pending_updates: Vec::new(),
                flush_interval,
                metadata,
                last_active: task_last_active,
                process_died: task_process_died,
                oracle_credit_us: task_oracle_credit_us,
                full_resyncs: task_full_resyncs,
                catch_ups: task_catch_ups,
                keepalive_interval,
                deadlines,
                outstanding: None,
                resync_owed: false,
                policy_hash,
                kills,
            };
            task.run().await;
        });

        Self {
            scope,
            backend_name,
            query_tx: qtx,
            evaluator: handle_evaluator,
            last_active,
            process_died,
            deadlines,
            oracle_credit_us,
            #[cfg(test)]
            credit_base_reads: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            full_resyncs,
            catch_ups,
            task,
        }
    }

    /// Re-seed this session's live evaluator with the given PolicyMetadata
    /// facts (the full assembled set). Serializes against queries on the
    /// evaluator's IPC mutex.
    pub async fn push_metadata(
        &self,
        facts: Vec<PolicyMetadataFact>,
    ) -> Result<(), EvaluatorError> {
        // Bounded like every other pre-serving IPC in this file. The caller
        // (UpdatePolicyMetadata) holds the scope lock across this call, so an
        // unbounded wait here against a wedged evaluator would block the scope
        // — EndSession and every later metadata write for it — for good.
        match tokio::time::timeout(
            self.deadlines.stall_window,
            self.evaluator.set_metadata(facts),
        )
        .await
        {
            Ok(result) => result,
            Err(_elapsed) => Err(EvaluatorError::UpdateError(format!(
                "the policy evaluator did not accept the metadata re-seed within {}",
                human_bound(self.deadlines.stall_window),
            ))),
        }
    }

    /// How many times this task has reloaded its whole graph from the store.
    /// A replay catch-up leaves this untouched.
    #[cfg(test)]
    pub(crate) fn full_resync_count(&self) -> u64 {
        self.full_resyncs.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many broadcast gaps this task closed from the store's replay log.
    #[cfg(test)]
    pub(crate) fn catch_up_count(&self) -> u64 {
        self.catch_ups.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// True iff the underlying subprocess died and the task has
    /// exited. The map evicts these entries lazily on the next
    /// dispatch so a respawn re-bootstraps from RocksDB.
    pub fn process_died(&self) -> bool {
        self.process_died.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Send a query to the session's evaluator. Updates the
    /// last-activity timestamp, used by the eviction sweep to keep
    /// idle sessions alive while they are getting queries.
    pub async fn dispatch(
        &self,
        request: EvalAuthRequest,
        min_sequence: i64,
    ) -> Result<QueryResult, String> {
        *self.last_active.lock() = Instant::now();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let job = QueryJob {
            request,
            min_sequence,
            respond: tx,
        };
        // The caller's own bound covers the enqueue as well as the wait. The
        // queue is small (SESSION_QUERY_CAPACITY) and the task drains it only
        // between jobs, so while the task is inside a stall-window-bounded
        // operation (bootstrap, re-sync, catch-up probe) the queue stays full
        // and `send` blocks with no bound of its own. One deadline spans both
        // halves so a caller past the queue's last slot waits no longer than a
        // caller that got one.
        let wait = self.deadlines.query_budget + DISPATCH_GRACE;
        let deadline = tokio::time::Instant::now() + wait;
        // The caller's clock stops for the LLM oracle exactly as the task's
        // does. Without this the task would still be legitimately waiting on
        // the provider — its own budget stopped — while the caller was
        // answered with a deny that says nothing answered in time.
        let credit_base = self
            .oracle_credit_us
            .load(std::sync::atomic::Ordering::Relaxed);
        // The base is now fixed for this caller; see `credit_base_reads`.
        #[cfg(test)]
        self.credit_base_reads
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        // One credit for both waits, and the running total travels with it:
        // what the enqueue spent is what the response wait starts from, and
        // either refusal can then say how much of this caller's wait was the
        // oracle's rather than the evaluator's.
        let mut credit = OracleCredit {
            counter: &self.oracle_credit_us,
            base_us: credit_base,
            bound: self.deadlines.oracle_bound,
            used: Duration::ZERO,
        };
        match credit.wait_for(self.query_tx.send(job), deadline).await {
            Ok(sent) => sent.map_err(|_| "session evaluator task dead".to_string())?,
            Err(_elapsed) => {
                return Err(format!(
                    "the policy evaluator's queue stayed full for {} (its {} budget and \
                     {} of grace){}, so the check fails closed",
                    human_bound(wait),
                    human_bound(self.deadlines.query_budget),
                    human_bound(DISPATCH_GRACE),
                    credit.spent_note(),
                ));
            }
        }
        // The task answers first in every case it has started working on this
        // job, and its deny is the more specific one; this catches the cases
        // where the job never got that far.
        match credit.wait_for(rx, deadline).await {
            Ok(answered) => {
                answered.map_err(|_| "session evaluator dropped response".to_string())?
            }
            Err(_elapsed) => Err(format!(
                "the policy evaluator did not answer within {} (its {} budget and \
                 {} of grace){}, so the check fails closed",
                human_bound(wait),
                human_bound(self.deadlines.query_budget),
                human_bound(DISPATCH_GRACE),
                credit.spent_note(),
            )),
        }
    }

    pub fn scope(&self) -> &SessionScope {
        &self.scope
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn last_active(&self) -> Instant {
        *self.last_active.lock()
    }
}

struct SessionEvaluatorTask {
    scope: SessionScope,
    evaluator: Arc<dyn Evaluator>,
    graph_store: Arc<GraphStore>,
    broadcast_rx: broadcast::Receiver<sasy_graph::GraphUpdate>,
    query_rx: async_channel::Receiver<QueryJob>,
    sequence: i64,
    /// Store-wide sequence as of the last AUTHORITATIVE snapshot (bootstrap /
    /// full_resync). Written ONLY there — never by a broadcast marker — which is
    /// what makes it trustworthy where `sequence` is not: it asserts "the graph I
    /// hold provably includes every write up to this counter". The global-scope
    /// path uses it as a resync TRIGGER (skip the reload when the store hasn't
    /// moved past it), which is sound even though the same counter is useless as a
    /// fence CLEARANCE signal (markers reorder across shards).
    snapshot_seq: i64,
    /// Set when a bootstrap RESET the evaluator and then failed to load the
    /// snapshot back in, so the subprocess holds a graph we know is missing
    /// state. Distinct from `snapshot_seq` being stale: that only makes the
    /// global path *retry*, while any query landing in between would still be
    /// answered — silently — from the hollow graph. A deny-by-absence is
    /// indistinguishable from a legitimate deny at the call site, so queries
    /// fail CLOSED until a reload actually lands.
    graph_incomplete: bool,
    node_count: u64,
    edge_count: u64,
    pending_updates: Vec<GraphUpdate>,
    flush_interval: Duration,
    /// The `PolicyMetadata(rel, a, b)` facts this evaluator is seeded with,
    /// assembled at spawn and sent once, before bootstrap. Not only static
    /// config: `assemble_metadata` appends the session's dynamic facts (its
    /// recorded detaint decisions), which is what carries them across a
    /// respawn. Empty for a session with neither. Drained (via `mem::take`)
    /// on first send; not re-sent on `full_resync` (the subprocess keeps its
    /// EDB — only a respawn re-seeds).
    metadata: Vec<PolicyMetadataFact>,
    /// Shared with the [`SessionEvaluator`] handle so the task can
    /// bump activity when broadcasts arrive (not just when queries
    /// dispatch). Read by the eviction sweep.
    last_active: Arc<Mutex<Instant>>,
    /// Shared with the [`SessionEvaluator`] handle. The task sets it
    /// before returning when an evaluator IPC reports
    /// [`EvaluatorError::ProcessDied`]; the map reads it after
    /// dispatch and evicts so the next request respawns.
    process_died: Arc<std::sync::atomic::AtomicBool>,
    /// Shared with the [`SessionEvaluator`] handle: the budget clock this task
    /// has stopped for the LLM oracle, in microseconds, cumulative. Bumped
    /// whenever the query arm extends a deadline; read by `dispatch` so the
    /// caller's own bound stops for the same wait.
    oracle_credit_us: Arc<std::sync::atomic::AtomicU64>,
    /// Shared with the [`SessionEvaluator`] handle. Bumped on every full
    /// reload so tests can tell a replay catch-up from a reload.
    full_resyncs: Arc<AtomicU64>,
    /// Shared with the handle. Bumped on every gap closed from the replay log.
    catch_ups: Arc<AtomicU64>,
    /// Period between empty-update keepalive IPCs. Zero disables.
    /// See [`resolve_keepalive_interval`] for the motivation.
    keepalive_interval: Duration,
    /// Wall-clock bounds on one job and on the evaluator's silence.
    deadlines: EvaluationDeadlines,
    /// Set while the evaluator owes a reply that nobody is still waiting for —
    /// a job that overran its budget and was denied. Cleared by the first
    /// later IPC that completes, which is proof the evaluator caught up.
    outstanding: Option<Outstanding>,
    /// Set when a re-sync was abandoned at a caller's budget, and cleared only
    /// by one that completes.
    ///
    /// A reload is O(the scope's graph) and a budget is a fixed number of
    /// seconds, so a graph large enough that its reload does not fit is a
    /// graph whose reload NEVER fits: every later check re-triggers the same
    /// re-sync, spends the same budget and is denied again, and the scope is
    /// locked out for good. While this is set the checks are denied without
    /// starting another doomed reload, and the task drives one to completion
    /// under the stall window instead — the same bound its bootstrap runs
    /// under — so the lockout ends with the reload rather than with the
    /// traffic.
    resync_owed: bool,
    /// The policy this session is bound to, named in the log a kill emits so
    /// the query that wedged the evaluator can be tied to the rules that were
    /// evaluating it.
    policy_hash: String,
    /// This session's route to its kill record, which lives in the map so it
    /// survives the respawn each kill causes.
    kills: KillLedgerRef,
}

/// One waiter's view of the budget clock the session has stopped for the LLM
/// oracle.
///
/// The session task grants itself extension when its own deadline expires and
/// the evaluator turns out to have been waiting on the provider; `counter` is
/// the running total of what it has granted, in microseconds, for the life of
/// this evaluator. A waiter records the total it started at and extends its own
/// deadline by whatever has been granted since — the same wait, measured the
/// same way, capped at the same `bound`.
struct OracleCredit<'a> {
    counter: &'a std::sync::atomic::AtomicU64,
    base_us: u64,
    bound: Duration,
    /// What this caller has spent of `bound`, across every wait it makes.
    ///
    /// `dispatch` makes two — the enqueue and the response — under one
    /// deadline, and the running total is carried between them so a refusal
    /// can say how much of the caller's wait was the oracle's. It does not
    /// move the bound: both waits measure `counter` against the same
    /// `base_us` and the same `deadline`, so each lands it at
    /// `deadline + min(granted, bound)` whether the total is carried or not —
    /// one bound for the dispatch, never one per wait.
    used: Duration,
}

impl OracleCredit<'_> {
    /// Await `fut` until `deadline`, stopping the clock for oracle wait as it
    /// is granted. `Err(())` once the deadline passes with no new credit.
    async fn wait_for<F: std::future::Future>(
        &mut self,
        fut: F,
        deadline: tokio::time::Instant,
    ) -> Result<F::Output, ()> {
        tokio::pin!(fut);
        // Where an earlier wait under this deadline had already reached.
        let mut deadline = deadline + self.used;
        loop {
            match tokio::time::timeout_at(deadline, &mut fut).await {
                Ok(v) => return Ok(v),
                Err(_elapsed) => {
                    let granted = Duration::from_micros(
                        self.counter
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .saturating_sub(self.base_us),
                    );
                    let extra = granted
                        .saturating_sub(self.used)
                        .min(self.bound.saturating_sub(self.used));
                    if extra.is_zero() {
                        return Err(());
                    }
                    self.used += extra;
                    deadline += extra;
                }
            }
        }
    }

    /// What the refusals add when the oracle is part of why the caller waited.
    /// Empty when no credit was spent, so the common refusal is unchanged.
    fn spent_note(&self) -> String {
        if self.used.is_zero() {
            String::new()
        } else {
            format!(
                " and {} more that the evaluation spent waiting on the LLM oracle",
                human_bound(self.used)
            )
        }
    }
}

/// A request the evaluator has been given and has not answered.
struct Outstanding {
    /// The IPC id it is owed under, when the backend reports one.
    id: Option<u64>,
    /// When the evaluator was handed the OLDEST request it still owes. Later
    /// queries deliberately do NOT reset this: the stall window has to measure
    /// how long the process has been silent, and an evaluator asked something
    /// new every few seconds would otherwise never look stalled.
    since: Instant,
    /// The action shape and tool name that request carried, for the log.
    action: String,
    tool: String,
}

/// The host an HTTP action names, which is what stands in for the tool name
/// the log carries.
///
/// Not the whole URL: it is unbounded, and its path and query string carry
/// caller-controlled values — the kind of thing a policy exists to keep out of
/// places it will be read from. The host is what identifies the call site.
fn request_host(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let host = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    host.to_string()
}

/// The action shape and tool name a log line names for a request. The
/// first action only: a multi-action request is one evaluation, and the first
/// action is enough to find the call site.
fn describe_request(request: &EvalAuthRequest) -> (String, String) {
    match request.actions.first() {
        Some(crate::evaluator::types::EvalAction::ToolCall { fn_name, .. }) => {
            ("tool_call".to_string(), fn_name.clone())
        }
        Some(crate::evaluator::types::EvalAction::HttpRequest { url, .. }) => {
            ("http_request".to_string(), request_host(url))
        }
        Some(crate::evaluator::types::EvalAction::SendMessage { agent, .. }) => {
            ("send_message".to_string(), agent.clone())
        }
        None => ("none".to_string(), String::new()),
    }
}

/// `full_resync` on the per-request path, under the job's budget.
///
/// Only usable inside the query arm, where a caller is waiting on `$respond`:
/// it expands to a `return` when the subprocess is gone (the existing "task
/// exits" signal) and to a `continue` when the budget runs out. A re-sync is
/// an evaluator IPC with no bound of its own, issued with the check blocked on
/// it, so from the caller's side there is no difference between it overrunning
/// and the query overrunning — both are answered with the same deny.
macro_rules! resync_within_budget {
    ($task:expr, $respond:expr, $deadline:expr) => {
        let started = Instant::now();
        let outcome = tokio::time::timeout_at($deadline, $task.full_resync()).await;
        match outcome {
            Ok(true) => {
                $task.outstanding = None;
                $task.resync_owed = false;
            }
            Ok(false) => return,
            Err(_elapsed) => {
                // The reload is owed until one lands: this one emptied the
                // evaluator with its `reset` and was dropped before the
                // snapshot went back in (`full_resync` marks the graph
                // incomplete for exactly that reason), and re-running it on
                // the next caller's budget would only repeat the failure.
                $task.resync_owed = true;
                $task.deny_over_budget(
                    $respond,
                    "the graph re-sync",
                    "update".to_string(),
                    String::new(),
                    started,
                );
                continue;
            }
        }
    };
}

impl SessionEvaluatorTask {
    /// Clear the mark, but only on evidence that the child was spoken to.
    ///
    /// An IPC can fail before the child hears anything — a request the
    /// transport refuses is rejected ahead of the child mutex, so no frame is
    /// written and `begin_request` is never reached. That is an `Err` from a
    /// still-silent evaluator, and clearing the mark on it would disarm the
    /// stall window: the wedged evaluator would never be killed, never
    /// respawned and never counted. `outstanding_request` is exactly that
    /// evidence — a completed call stores 0 (so the healthy path clears),
    /// while a call that never reached the child leaves the abandoned id set.
    fn clear_outstanding_if_child_spoke(&mut self) {
        if self.evaluator.outstanding_request().is_none() {
            self.outstanding = None;
        }
    }

    /// Record that the evaluator owes a reply.
    ///
    /// An existing mark is kept as it stands, only gaining the newer request
    /// id: `since` must stay at the FIRST request the evaluator went silent
    /// on, or an evaluator asked something new every few seconds would keep
    /// resetting its own stall clock and never look stalled.
    fn mark_busy(&mut self, action: String, tool: String, since: Instant) {
        let id = self.evaluator.outstanding_request();
        match self.outstanding.as_mut() {
            Some(o) => o.id = id.or(o.id),
            None => {
                self.outstanding = Some(Outstanding {
                    id,
                    since,
                    action,
                    tool,
                })
            }
        }
    }

    /// A job ran out of budget with a caller waiting: log what was being
    /// evaluated, remember that the evaluator still owes a reply, and
    /// answer the caller.
    ///
    /// The answer is an `Err`, the shape this task already produces for a dead
    /// subprocess and for a graph it knows is incomplete: no decision, which
    /// every caller of the engine treats as a refusal. The evaluator is left
    /// running — it is busy, not broken, and killing it here would throw away
    /// a session's loaded graph over one slow query.
    fn deny_over_budget(
        &mut self,
        respond: tokio::sync::oneshot::Sender<Result<QueryResult, String>>,
        phase: &str,
        action: String,
        tool: String,
        started: Instant,
    ) {
        self.mark_busy(action.clone(), tool.clone(), started);
        warn!(
            scope = %self.scope,
            phase,
            policy = %self.policy_hash,
            action = %action,
            tool = %tool,
            elapsed_ms = started.elapsed().as_millis() as u64,
            budget_secs = self.deadlines.query_budget.as_secs_f64(),
            outstanding_id = self.outstanding.as_ref().and_then(|o| o.id),
            "policy evaluation exceeded its budget — denying the check, keeping the evaluator"
        );
        let _ = respond.send(Err(format!(
            "policy evaluation exceeded its {} budget during {phase}: \
             the evaluator has not answered yet, so the check fails closed",
            human_bound(self.deadlines.query_budget),
        )));
    }

    /// When this evaluator's silence becomes a stall.
    ///
    /// The stall window measures silence since the child last spoke, whatever
    /// phase happens to be waiting on it. While a reply is owed that is
    /// `Outstanding::since` — the moment the oldest unanswered request was
    /// handed over — so every bound the task takes on its own account
    /// (a re-sync, a flush, the catch-up probe) expires at the SAME instant
    /// the select's liveness arm would have. A phase that started its own full
    /// window from wherever it happened to begin would let the kill land at up
    /// to twice the window. With nothing owed there is no silence to measure:
    /// the clock starts with the operation about to be issued.
    fn stall_deadline(&self) -> tokio::time::Instant {
        let since = self
            .outstanding
            .as_ref()
            .map(|o| o.since)
            .unwrap_or_else(Instant::now);
        tokio::time::Instant::from_std(since + self.deadlines.stall_window)
    }

    /// True when the evaluator has owed a reply for longer than the stall
    /// window — the point at which it is presumed unable to answer anything
    /// rather than merely slow on this query.
    fn stalled(&self) -> bool {
        self.outstanding
            .as_ref()
            .is_some_and(|o| o.since.elapsed() >= self.deadlines.stall_window)
    }

    /// The evaluator has been silent past the stall window: kill it, answer
    /// everyone waiting, and leave the task so the map respawns onto a fresh
    /// process that re-bootstraps from the store.
    ///
    /// The order is deliberate:
    ///
    /// * The kill is recorded before `mark_process_died`, because that flag is
    ///   what tells the map to respawn: a dispatcher that read it in between
    ///   would run the cap check against a count that did not yet include this
    ///   kill, and the cap would be one respawn late every time.
    /// * `mark_process_died` before anyone is answered, because `dispatch`
    ///   returns the instant a response lands and the map reads that flag
    ///   immediately after.
    /// * The answers before the kill, because the kill awaits the child's
    ///   reap. A child that cannot be reaped promptly would otherwise leave
    ///   the caller with no answer at all — the hang this exists to remove.
    /// * The queued jobs with it: they have been waiting behind the wedged one
    ///   for the whole window, and dropping their senders hands each of them
    ///   "session evaluator dropped response", which reads as a lost request
    ///   rather than a decision.
    async fn kill_stalled(
        &mut self,
        waiting: Option<tokio::sync::oneshot::Sender<Result<QueryResult, String>>>,
    ) {
        let (action, tool, id, elapsed) = match self.outstanding.as_ref() {
            Some(o) => (o.action.clone(), o.tool.clone(), o.id, o.since.elapsed()),
            // Nothing outstanding means the wedge is in the session's own
            // start-up, before any query: the bootstrap IPC never came back.
            None => (
                "bootstrap".to_string(),
                String::new(),
                self.evaluator.outstanding_request(),
                self.deadlines.stall_window,
            ),
        };
        error!(
            scope = %self.scope,
            policy = %self.policy_hash,
            action = %action,
            tool = %tool,
            outstanding_id = id,
            elapsed_ms = elapsed.as_millis() as u64,
            stall_secs = self.deadlines.stall_window.as_secs_f64(),
            "evaluator owed a reply past the stall window — killing its process group"
        );
        // Recorded before the death is published, and before anyone is
        // answered. `mark_process_died` is what makes the map respawn and
        // `dispatch` returns the moment a response lands, so a dispatcher
        // reaching the cap check between the two would read a count that is
        // one kill short and hand out a respawn the cap should have refused.
        let in_window = self.kills.record();
        self.mark_process_died();
        if in_window >= KILL_CAP {
            error!(
                scope = %self.scope,
                kills = in_window,
                window_secs = KILL_WINDOW.as_secs(),
                "kill cap reached — this session will be refused until the window passes"
            );
        }
        let reason = format!(
            "the policy evaluator stopped answering ({} with no reply); it was killed \
             and the session re-bootstraps on the next request, so the check fails closed",
            human_bound(elapsed),
        );
        if let Some(tx) = waiting {
            let _ = tx.send(Err(reason.clone()));
        }
        // Closed before the drain so a job cannot land in the buffer between
        // the drain and this task's exit, where nothing would ever answer it.
        // Buffered jobs stay receivable after a close; a sender arriving later
        // gets a clean "task dead" from `dispatch`, and the map has already
        // been told to respawn.
        self.query_rx.close();
        let mut queued = 0usize;
        while let Ok(job) = self.query_rx.try_recv() {
            queued += 1;
            let _ = job.respond.send(Err(reason.clone()));
        }
        if queued > 0 {
            warn!(
                scope = %self.scope,
                queued,
                "denied the checks that were queued behind the stalled evaluator"
            );
        }
        if tokio::time::timeout(KILL_REAP_DEADLINE, self.evaluator.kill())
            .await
            .is_err()
        {
            // The signal is delivered by now; only the reap is abandoned.
            warn!(
                scope = %self.scope,
                "the killed evaluator did not exit within {}s — leaving it to be reaped",
                KILL_REAP_DEADLINE.as_secs_f64()
            );
        }
        self.outstanding = None;
    }

    /// Mark the subprocess as dead so the map evicts us on the next
    /// dispatch and a fresh evaluator respawns from graph state.
    fn mark_process_died(&self) {
        self.process_died
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Returns `true` iff the error is [`EvaluatorError::ProcessDied`].
    /// Side effect: records the death on `self.process_died`.
    fn note_if_dead(&self, err: &EvaluatorError) -> bool {
        if matches!(err, EvaluatorError::ProcessDied) {
            self.mark_process_died();
            error!(scope = %self.scope, "evaluator subprocess died — task exiting");
            true
        } else {
            false
        }
    }

    /// Bootstrap from the graph store. Returns `false` if the
    /// evaluator subprocess died during the bootstrap update — the
    /// caller should exit the task in that case.
    async fn bootstrap(&mut self) -> bool {
        // A global-scope evaluator answers tenant-wide queries, and
        // its live broadcast path accepts EVERY session's events
        // under the tenant (`SessionScope::matches` is tenant-wide
        // for a global scope). Bootstrap must mirror that: read all
        // the tenant's shards, preserving each event's own session
        // attribution, so a cold start or full re-sync reproduces
        // the state the live stream converges to — rather than only
        // the global shard, which would make the cold-start view
        // diverge from the steady state and flip decisions for a
        // tenant-wide policy after a restart/resync. A non-global
        // scope reads just its own shard and flattens to its own
        // session, as before. Both fence read-your-writes on the
        // scope's own shard sequence (the global shard, for a global
        // scope).
        let loaded = if self.scope.is_global() {
            self.graph_store
                .get_full_state_for_tenant_scoped(self.scope.tenant())
                .map(|(events, edges, seq)| {
                    let n_events = events.len();
                    let n_edges = edges.len();
                    let updates = full_state_scoped_to_updates(&events, &edges);
                    (updates, n_events, n_edges, seq)
                })
        } else {
            self.graph_store
                .get_session_state(&self.scope)
                .map(|(events, edges, seq)| {
                    let n_events = events.len();
                    let n_edges = edges.len();
                    let updates = full_state_to_updates(&events, &edges, self.scope.session());
                    (updates, n_events, n_edges, seq)
                })
        };
        match loaded {
            Ok((updates, n_events, n_edges, seq)) => {
                self.node_count = n_events as u64;
                self.edge_count = n_edges as u64;
                let mut applied = true;
                if !updates.is_empty() {
                    if let Err(e) = self.evaluator.update(updates).await {
                        warn!(
                            scope = %self.scope,
                            "bootstrap update failed: {}",
                            e
                        );
                        if self.note_if_dead(&e) {
                            return false;
                        }
                        applied = false;
                    }
                }
                self.sequence = seq;
                // Only claim snapshot coverage if the state actually LANDED in the
                // evaluator. A non-fatal update error (oversized IPC frame, decode
                // failure) after `full_resync`'s `reset()` leaves an EMPTY graph; if
                // we advanced the snapshot anyway, the global path would skip its
                // reload until the counter moved again and stay empty in the
                // meantime. Leaving it stale makes the next query retry.
                self.graph_incomplete = !applied;
                if applied {
                    self.snapshot_seq = seq;
                }
                info!(
                    scope = %self.scope,
                    seq,
                    events = n_events,
                    edges = n_edges,
                    "session evaluator bootstrapped"
                );
                true
            }
            Err(e) => {
                warn!(
                    scope = %self.scope,
                    "bootstrap failed: {}",
                    e
                );
                // Same hazard as a failed update, one path over: `full_resync` already
                // `reset()` the subprocess, so an unreadable store leaves an EMPTY
                // graph that queries would silently answer from. Both store readers
                // are infallible today (a missing shard reads as empty), so this is
                // defense-in-depth — but the invariant is "flag ⟺ the graph may be
                // hollow", and a `Result` signature is an invitation for some future
                // fallible read to land here and quietly break it.
                self.graph_incomplete = true;
                true
            }
        }
    }

    fn process_broadcast(&mut self, gu: sasy_graph::GraphUpdate) {
        // Fence-cursor advance, routed by scope so it stays in the SAME counter
        // space as the dispatch `min_sequence` (see evaluator_engine):
        //   · A NON-global evaluator tracks its per-scope shard counter, advancing
        //     on its own scope's `SessionSequence` marker. Per-scope counters from
        //     different sessions are incomparable, so a global evaluator must NOT
        //     advance on these: an unrelated counter would defeat its fence.
        //   · A GLOBAL evaluator tracks the store-wide `Sequence` counter (which
        //     bootstrap also loads) so its cursor stays meaningful, but it does NOT
        //     fence on it: that counter is observable out of order across shards, so
        //     a global query reconciles from the durable store instead (see the
        //     is_global() branch in the query arm). Store-wide, not per-tenant —
        //     content stays tenant-isolated via matches() below.
        //
        // The cursor only ever moves FORWARD. The task subscribes before it
        // bootstraps, so the ring can still hold markers stamped before the
        // snapshot the bootstrap loaded; taking one at face value rewound the
        // cursor and made the next query wait out its fence on progress the
        // evaluator already held. A replay catch-up sets the cursor through the
        // same rule, to a sequence it has provably applied.
        if let sasy_graph::GraphUpdate::SessionSequence { scope, seq } = &gu {
            if !self.scope.is_global() && self.scope.matches(scope) {
                self.sequence = (*seq).max(self.sequence);
            }
            return;
        }
        if let sasy_graph::GraphUpdate::Sequence(seq) = &gu {
            // Tenant-wide counter: the fence cursor for a global evaluator only.
            // Non-global evaluators ignore it (kept for neo4j_sync / observability).
            if self.scope.is_global() {
                self.sequence = (*seq).max(self.sequence);
            }
            return;
        }

        // Filter out updates that don't belong to this scope. A global
        // subscriber matches every update under its tenant via
        // SessionScope::matches. A DropSession for our scope is logged
        // below but not acted on here — teardown is driven by
        // SessionEvaluatorMap eviction (which drops this task), not by
        // the broadcast stream.
        let belongs = match &gu {
            sasy_graph::GraphUpdate::NodeCreated { scope, .. } => self.scope.matches(scope),
            sasy_graph::GraphUpdate::EdgeCreated { scope, .. } => self.scope.matches(scope),
            // Deletions carry their scope for exactly this filter: without one
            // every EdgeDeleted fell through to `false` below and no evaluator
            // ever saw a removal, live or replayed.
            sasy_graph::GraphUpdate::EdgeDeleted { scope, .. } => self.scope.matches(scope),
            sasy_graph::GraphUpdate::DropSession(scope) => self.scope.matches(scope),
            _ => false,
        };
        if !belongs {
            return;
        }

        // Belongs to this session — count it as activity for the
        // eviction sweep. Otherwise a session that's only receiving
        // events (no queries yet) would look idle and get evicted
        // immediately, churning through evict→prewarm cycles.
        *self.last_active.lock() = Instant::now();

        let updates = broadcast_to_updates(&gu);
        for u in updates {
            match &u {
                GraphUpdate::NodeCreated { .. } => self.node_count += 1,
                GraphUpdate::NodeDeleted(_) => self.node_count = self.node_count.saturating_sub(1),
                GraphUpdate::EdgeCreated { .. } => self.edge_count += 1,
                GraphUpdate::EdgeDeleted { .. } => {
                    self.edge_count = self.edge_count.saturating_sub(1)
                }
                GraphUpdate::DropSession(_) => {
                    // Informational only: this task is torn down by the
                    // map's eviction path, not from the broadcast.
                    debug!(scope = %self.scope, "drop session received");
                    continue;
                }
            }
            self.pending_updates.push(u);
        }
    }

    /// Returns `(elapsed_us, alive)`. `alive=false` means the
    /// subprocess died during the flush; the caller should exit.
    async fn flush_pending(&mut self) -> (u64, bool) {
        if self.pending_updates.is_empty() {
            return (0, true);
        }
        let batch = std::mem::take(&mut self.pending_updates);
        let n = batch.len();
        let t0 = Instant::now();
        let mut alive = true;
        if let Err(e) = self.evaluator.update(batch).await {
            warn!(
                scope = %self.scope,
                n,
                "apply batch failed: {}",
                e
            );
            if self.note_if_dead(&e) {
                alive = false;
            } else {
                // The subprocess is alive and the batch is gone — `mem::take`
                // consumed it and nothing resends it. Nothing else notices
                // either: the sequence marker that would force a resync is
                // consumed when an update is QUEUED, not when it applies, so
                // the fence is already satisfied and the next query runs
                // against an EDB silently missing those records. A dropped
                // `EdgeCreated` is a missing dependency, and a provenance rule
                // that cannot see it under-matches — authorizing what should
                // be denied.
                //
                // Same treatment a failed bootstrap gets: the next query
                // reloads and, if the reload does not fix it, fails closed.
                // That also keeps the two paths consistent, since a record
                // that fails to apply here fails the same way on re-bootstrap.
                self.graph_incomplete = true;
            }
        }
        (t0.elapsed().as_micros() as u64, alive)
    }

    /// Close a broadcast gap from the store's retained change log instead of
    /// reloading the whole shard.
    ///
    /// The updates go through [`Self::process_broadcast`] — the same code the
    /// live stream uses — so the scope filter, the counters and the buffering
    /// behave identically and the fence advances the same way.
    async fn catch_up_from_replay(&mut self) -> CatchUp {
        // A global evaluator's cursor is the STORE-WIDE counter while the
        // replay logs are per scope with per-scope sequences: different counter
        // spaces, and a global scope spans every session in the tenant besides.
        // There is nothing here to compare against, so it reloads, as before.
        if self.scope.is_global() {
            return CatchUp::NeedsResync;
        }
        // Read the target sequence BEFORE the log. A write landing between the
        // two is then absent from the target rather than absent from the
        // updates: the cursor stays behind and the fence waits for a marker,
        // instead of clearing on a write this task has not applied.
        let target = self.graph_store.session_sequence(&self.scope);
        let Some(updates) = self.graph_store.updates_since(&self.scope, self.sequence) else {
            return CatchUp::NeedsResync;
        };
        let n = updates.len();
        for u in updates {
            self.process_broadcast(u);
        }
        let (flush_us, alive) = self.flush_pending().await;
        if !alive {
            return CatchUp::Dead;
        }
        // `process_broadcast` counts each update it applies, which over-counts
        // when the ring later redelivers one of these. The store knows the
        // real per-scope totals and they are O(1) to read. Its edge total is
        // the message-dependency count — the same quantity `bootstrap` loads,
        // computation edges excluded — so the two paths install the same
        // number for the same shard.
        let (nodes, edges) = self.graph_store.session_counts(&self.scope);
        self.node_count = nodes;
        self.edge_count = edges;
        self.sequence = target.max(self.sequence);
        self.catch_ups
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug!(
            scope = %self.scope,
            updates = n,
            flush_us,
            "caught up from the replay log instead of reloading"
        );
        CatchUp::Applied
    }

    /// Close a gap in the broadcast stream, from the store's retained change
    /// log where it reaches back far enough and from the whole shard where it
    /// does not.
    ///
    /// The retained log is tried first because the missed updates cost one
    /// incremental apply, where a reload costs the whole shard — and a reload
    /// long enough for the ring to overflow again is how one session reloaded
    /// 176 times. `pending_updates` is NOT cleared on the catch-up path: those
    /// are legitimate buffered updates and the catch-up flushes them alongside
    /// the replayed ones. The reload path drops them, because a reload is what
    /// makes them redundant.
    ///
    /// The catch-up is skipped when a reload is owed. `resync_owed` says a
    /// previous `full_resync` emptied the evaluator with its `reset` and was
    /// dropped before the snapshot went back in, so there is no graph left for
    /// an incremental apply to sit on top of — only a reload refills it. That
    /// is why the decision lives here rather than at the two call sites: the
    /// drain loop and the select's Lagged arm are two ways of noticing the
    /// same gap, and a rule written twice is a rule that can drift.
    /// `needs_resync` is the drain loop's observation of the gap; the Lagged
    /// arm passes `true`, the lag being the observation itself.
    ///
    /// Every wait is under the stall window, like the bootstrap and the
    /// metadata seed: this runs with no caller waiting and nothing else
    /// watching, so an evaluator that wedges here would take the task with it
    /// — the loop would never reach the select, the stall arm would never be
    /// armed, and every check for this session would queue behind a process
    /// nothing is going to kill. The bound turns that into the kill it should
    /// be. The catch-up and the reload share it: they are two ways of closing
    /// the same gap, and the silence that has to be killed is the same silence
    /// in either.
    ///
    /// Returns `false` when the task must exit — the subprocess is gone, or it
    /// was just killed for stalling.
    async fn close_broadcast_gap(&mut self, needs_resync: bool) -> bool {
        let bound = self.stall_deadline();
        let caught_up = if needs_resync && !self.resync_owed {
            match tokio::time::timeout_at(bound, self.catch_up_from_replay()).await {
                Ok(CatchUp::Applied) => true,
                Ok(CatchUp::Dead) => return false,
                Ok(CatchUp::NeedsResync) => false,
                Err(_elapsed) => {
                    self.kill_stalled(None).await;
                    return false;
                }
            }
        } else {
            false
        };
        if !caught_up {
            self.pending_updates.clear();
            match tokio::time::timeout_at(bound, self.full_resync()).await {
                Ok(true) => self.resync_owed = false,
                Ok(false) => return false,
                Err(_elapsed) => {
                    self.kill_stalled(None).await;
                    return false;
                }
            }
        }
        true
    }

    /// Returns `false` if the evaluator subprocess died during
    /// reset or re-bootstrap — caller should exit the task.
    async fn full_resync(&mut self) -> bool {
        // Marked incomplete for the duration, because that is what the graph
        // becomes: `reset` empties the evaluator and only a bootstrap that
        // LANDS refills it. Anything that ends this function early — a dropped
        // future at a caller's budget, most of all — leaves the subprocess
        // holding an empty EDB, and an empty EDB answers checks: no denylist
        // fact, no taint edge, no provenance, so absence reads as innocence
        // and the answer can be an ALLOW. `bootstrap` writes the flag again
        // from what it actually applied, so a completed reload clears it.
        self.graph_incomplete = true;
        self.full_resyncs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Err(e) = self.evaluator.reset().await {
            warn!(
                scope = %self.scope,
                "reset failed: {}",
                e
            );
            if self.note_if_dead(&e) {
                return false;
            }
        }
        self.bootstrap().await
    }

    async fn run(mut self) {
        // Seed the assembled facts (static config plus the session's dynamic
        // ones) into the PolicyMetadata EDB once, before the first query and
        // before bootstrap loads graph state. Drained
        // so it isn't re-sent on a later full_resync (same subprocess
        // keeps its EDB); only a respawn re-runs `run` and re-seeds.
        let facts = std::mem::take(&mut self.metadata);
        if !facts.is_empty() {
            let n = facts.len();
            // Under the stall window, like every other IPC issued before this
            // session can serve: this is the FIRST thing sent to a brand-new
            // subprocess, so a functor that wedges on startup wedges here, and
            // an unbounded wait would leave the session's checks queued behind
            // it with nothing to time out.
            let seeded = match tokio::time::timeout(
                self.deadlines.stall_window,
                self.evaluator.set_metadata(facts),
            )
            .await
            {
                Ok(r) => r,
                Err(_elapsed) => {
                    self.kill_stalled(None).await;
                    return;
                }
            };
            if let Err(e) = seeded {
                error!(scope = %self.scope, "set_metadata failed: {}", e);
                // Marked dead unconditionally, not just when the error says
                // the subprocess died. The task is about to exit, so the
                // handle is a shell either way; leaving the flag false means
                // the map's fast path keeps handing that shell out and every
                // dispatch bumps `last_active`, so the idle sweep never reaps
                // it either — the session is refused forever and never gets a
                // second spawn. Marking it dead is what makes the map evict
                // and respawn on the next request, under the kill cap.
                self.mark_process_died();
                // Exit either way. Serving without the seed means serving with
                // an empty `PolicyMetadata` EDB for this evaluator's whole
                // lifetime: a policy that derives its sink taxonomy from these
                // facts then has no sinks and denies nothing, and one shaped
                // like the shipped profile loses its persisted detaint
                // decisions. The facts are drained by now, so nothing resends
                // them — only a respawn re-seeds.
                //
                // This also matches `assemble_metadata`, which already refuses
                // to spawn when it cannot READ the config. Refusing to serve
                // when it cannot DELIVER it is the same judgement.
                return;
            } else {
                debug!(scope = %self.scope, facts = n, "seeded policy metadata");
            }
        }
        // Bounded by the stall window like everything else: a functor that
        // spins in a `constructor` wedges the very first Update, and an
        // unbounded bootstrap would leave every check for this session queued
        // behind it with nothing to time out.
        match tokio::time::timeout(self.deadlines.stall_window, self.bootstrap()).await {
            Ok(true) => {}
            Ok(false) => {
                // Same contract as the seed above: every exit from `run()`
                // leaves the handle marked dead so the map respawns.
                self.mark_process_died();
                return;
            }
            Err(_elapsed) => {
                self.kill_stalled(None).await;
                return;
            }
        }

        // Background flushes accumulate here and get attributed to
        // whichever query observes the catch-up cost next, matching
        // the worker.rs accounting shape so timing logs stay
        // comparable across architectures.
        let mut pending_flush_us: u64 = 0;

        let mut flush_tick: Option<tokio::time::Interval> = if self.flush_interval.is_zero() {
            None
        } else {
            let mut iv = tokio::time::interval(self.flush_interval);
            iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            Some(iv)
        };

        // Optional keepalive: periodically issue an empty-update IPC
        // to the evaluator so the subprocess stays scheduled and
        // doesn't drop into a deep C-state / lower P-state during
        // long idle gaps (typical at low concurrency where agents
        // spend seconds waiting on LLM responses). Enabled via
        // ``SASY_KEEPALIVE_INTERVAL_MS``, default off.
        let mut keepalive_tick: Option<tokio::time::Interval> = if self.keepalive_interval.is_zero()
        {
            None
        } else {
            let mut iv = tokio::time::interval(self.keepalive_interval);
            iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Skip the immediate first tick; otherwise an empty
            // IPC fires at startup before there's anything to
            // keep warm.
            iv.tick().await;
            Some(iv)
        };

        loop {
            // Drain any buffered broadcasts; only this session's
            // updates land in pending_updates.
            let mut needs_resync = false;
            loop {
                match self.broadcast_rx.try_recv() {
                    Ok(gu) => self.process_broadcast(gu),
                    Err(broadcast::error::TryRecvError::Empty) => break,
                    Err(broadcast::error::TryRecvError::Lagged(n)) => {
                        warn!(
                            scope = %self.scope,
                            "broadcast lagged {n}, reconciling"
                        );
                        needs_resync = true;
                        break;
                    }
                    Err(broadcast::error::TryRecvError::Closed) => {
                        info!(
                            scope = %self.scope,
                            "broadcast channel closed"
                        );
                        return;
                    }
                }
            }
            // Nothing background-initiated goes down the pipe while the
            // evaluator still owes a reply. The shim answers one frame at a
            // time, so an Update sent now would block this task until the
            // abandoned query finishes — with no caller waiting on it and no
            // budget over it, that is the hang the budget exists to remove,
            // moved one step away from the query arm. Updates keep buffering
            // and go out with the first job that finds the evaluator caught
            // up; an evaluator that never catches up is the stall window's.
            if self.outstanding.is_none() {
                // Under the stall window, like the bootstrap and the metadata
                // seed: these run with no caller waiting and nothing else
                // watching, so an evaluator that wedges here would take the
                // task with it — the loop would never reach the select, the
                // stall arm would never be armed, and every check for this
                // session would queue behind a process nothing is going to
                // kill. The bound turns that into the kill it should be — for
                // the flush here, and for the catch-up and the reload inside
                // `close_broadcast_gap`.
                if needs_resync || self.resync_owed {
                    if !self.close_broadcast_gap(needs_resync).await {
                        return;
                    }
                } else {
                    let bound = self.stall_deadline();
                    let flushed = tokio::time::timeout_at(bound, self.flush_pending()).await;
                    let (flush_us, alive) = match flushed {
                        Ok(v) => v,
                        Err(_elapsed) => {
                            self.kill_stalled(None).await;
                            return;
                        }
                    };
                    pending_flush_us = pending_flush_us.saturating_add(flush_us);
                    if !alive {
                        return;
                    }
                }
            } else if needs_resync {
                // The ring dropped events for this scope, so the buffer is no
                // longer a complete picture: drop it and fail closed until a
                // reload lands, which the next job does once the evaluator is
                // answering again.
                self.pending_updates.clear();
                self.graph_incomplete = true;
            }

            // Computed before the select so the branch below borrows a copy
            // rather than `self`, which the other branches need.
            let stall_at = self
                .outstanding
                .as_ref()
                .map(|o| tokio::time::Instant::from_std(o.since + self.deadlines.stall_window));

            // One bounded IPC while a reply is owed, when nothing else is
            // queued: the catch-up probe.
            //
            // A query denied at its budget leaves the evaluator computing, and
            // its answer lands in the pipe with nobody reading it. Every IPC
            // the task issues on its own is suppressed while a reply is owed,
            // so without this the ONLY thing that can observe that answer is a
            // brand-new query — and a session that goes quiet after one slow
            // check would have its evaluator killed at the stall window, its
            // graph thrown away and a kill put on its ledger, for a process
            // that had already caught up. An empty update is the cheapest
            // question that needs the child's attention: it can only be
            // answered behind whatever the child is still doing, so returning
            // IS the proof, and a wedged evaluator simply never answers it and
            // is killed exactly as before.
            //
            // Empty on purpose: the probe is dropped at its bound, and a
            // dropped update loses the batch it took (see `flush_pending`),
            // which is nothing when there is nothing in it.
            if let Some(probe_until) = stall_at.filter(|_| self.query_rx.is_empty()) {
                // Bounded by the stall deadline, which is the same deadline
                // the select's liveness arm holds: a probe that does not come
                // back IS the silence that kills. Only issued with nothing
                // queued, so it is never what a waiting check is stuck behind.
                match tokio::time::timeout_at(probe_until, self.evaluator.update(Vec::new())).await
                {
                    Ok(Ok(())) => {
                        debug!(
                            scope = %self.scope,
                            "the evaluator answered again — it owes nothing"
                        );
                        self.outstanding = None;
                    }
                    Ok(Err(e)) => {
                        if self.note_if_dead(&e) {
                            return;
                        }
                        // It spoke, which is what the mark was about; what it
                        // said is the failing IPC's own business.
                        self.outstanding = None;
                    }
                    Err(_elapsed) => {
                        if self.stalled() {
                            self.kill_stalled(None).await;
                            return;
                        }
                    }
                }
                continue;
            }

            tokio::select! {
                biased;

                // First, and deliberately: this is the liveness guard, and a
                // busy broadcast ring would starve it from any later position.
                _ = async {
                    match stall_at {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    // Nobody is waiting on this one — the query that went
                    // unanswered was denied at its budget — but the process is
                    // still burning whatever it is burning, and the graph it
                    // holds is only recoverable by re-bootstrapping a new one.
                    self.kill_stalled(None).await;
                    return;
                }

                result = self.broadcast_rx.recv() => {
                    match result {
                        Ok(gu) => self.process_broadcast(gu),
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!(
                                scope = %self.scope,
                                "broadcast lagged {n}, reconciling"
                            );
                            if self.outstanding.is_some() {
                                // As above: no IPC while a reply is owed. The
                                // buffer is no longer a complete picture, so
                                // drop it and fail closed until a job can
                                // close the gap.
                                self.pending_updates.clear();
                                self.graph_incomplete = true;
                            } else {
                                // The lag IS the observation of a gap, so the
                                // same gap-closing rule the drain loop uses,
                                // under the same stall window: nobody is
                                // waiting on this work, so nothing else would
                                // ever end it.
                                if !self.close_broadcast_gap(true).await {
                                    return;
                                }
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!(
                                scope = %self.scope,
                                "broadcast channel closed"
                            );
                            return;
                        }
                    }
                }

                _ = async {
                    match flush_tick.as_mut() {
                        Some(iv) => { iv.tick().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    // Drain + flush will run on next iteration.
                }

                _ = async {
                    match keepalive_tick.as_mut() {
                        Some(iv) => { iv.tick().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    // Empty-update IPC. Doesn't change graph state
                    // but exercises the pipe + Soufflé dispatch
                    // path so the OS keeps the process at a high
                    // P-state and the working set hot. A failed
                    // keepalive that reports ProcessDied is still
                    // fatal — without exit-on-death, the task would
                    // burn cycles re-failing forever and the map
                    // would keep handing this evaluator to callers.
                    // Skipped while a reply is owed: the catch-up probe above
                    // is already asking that evaluator the same question under
                    // a bound, and a second one would only queue behind it.
                    // Bounded by the stall window like every other IPC the
                    // task issues on its own, so a keepalive cannot be the
                    // thing that wedges the task.
                    if self.outstanding.is_none() {
                        match tokio::time::timeout(
                            self.deadlines.stall_window,
                            self.evaluator.update(Vec::new()),
                        )
                        .await
                        {
                            Ok(Err(e)) => {
                                if self.note_if_dead(&e) {
                                    return;
                                }
                            }
                            Ok(Ok(())) => {}
                            Err(_elapsed) => {
                                self.kill_stalled(None).await;
                                return;
                            }
                        }
                    }
                }

                result = self.query_rx.recv() => {
                    match result {
                        Ok(job) => {
                            // The evaluator has been silent past the stall
                            // window: no point spending this caller's budget
                            // asking it again.
                            if self.stalled() {
                                self.kill_stalled(Some(job.respond)).await;
                                return;
                            }
                            if self.resync_owed {
                                // A reload was abandoned at some caller's
                                // budget and the task is driving one to
                                // completion. Starting another on THIS
                                // caller's budget would abandon it at the same
                                // point, for the same reason, and answer no
                                // check either way — the reload is O(the
                                // graph) and the budget is a constant. Fail
                                // this one closed and let the reload land.
                                warn!(
                                    scope = %self.scope,
                                    policy = %self.policy_hash,
                                    "a graph reload is still outstanding — denying the check"
                                );
                                let _ = job.respond.send(Err(
                                    "the evaluator's graph is being reloaded after a re-sync \
                                     that did not fit in the query budget, so the check fails \
                                     closed until it lands"
                                        .to_string(),
                                ));
                                // Answered first, then the reload — with
                                // nobody waiting on it, and under the stall
                                // window, which is the bound for the work the
                                // task does on its own account. Driven from
                                // here and not only from the background block
                                // because a scope under steady traffic never
                                // has an idle moment: something has to finish
                                // the reload while checks keep arriving, or
                                // the deny above is the permanent answer.
                                self.pending_updates.clear();
                                // Anchored on the silence, not on when this
                                // reload happens to start: the evaluator
                                // already owes a reply here, and a fresh
                                // window would put the kill at up to twice the
                                // stall window from the moment it went quiet.
                                let until = self.stall_deadline();
                                match tokio::time::timeout_at(until, self.full_resync()).await {
                                    Ok(true) => {
                                        // It spoke, so it owes nothing, and
                                        // the graph is whole again.
                                        self.clear_outstanding_if_child_spoke();
                                        self.resync_owed = false;
                                    }
                                    Ok(false) => return,
                                    Err(_elapsed) => {
                                        self.kill_stalled(None).await;
                                        return;
                                    }
                                }
                                continue;
                            }
                            let sync_start = Instant::now();
                            // ONE budget for the whole job, not just the query.
                            // The fence, a re-sync and the pre-query flush all
                            // issue evaluator IPCs with no bound of their own
                            // and with this caller already blocked on
                            // `job.respond` — a functor that spins inside a
                            // `constructor` wedges the very first Update, and a
                            // budget that covered only the query would never be
                            // reached.
                            //
                            // And it is clamped to the stall deadline. A job
                            // dequeued while the evaluator already owes a reply
                            // gets what is LEFT of that silence, not a fresh
                            // full budget: with a fresh one the task would sit
                            // in this arm past the moment the child's silence
                            // became a kill, and under steady traffic — where
                            // there is always a next job to dequeue — the kill
                            // landed up to one query budget late. A healthy
                            // first query is untouched: with nothing
                            // outstanding the cap is now plus the stall window,
                            // which is never below the budget.
                            //
                            // The cap is read ONCE, here, for the whole job.
                            // With nothing owed `stall_deadline` starts a fresh
                            // window from the moment it is CALLED, so a bound
                            // re-read inside a loop would push itself ahead of
                            // itself and stop bounding anything.
                            let stall_cap = self.stall_deadline();
                            let job_deadline = (tokio::time::Instant::now()
                                + self.deadlines.query_budget)
                                .min(stall_cap);
                            // Captured before the request moves into the query,
                            // so a budget miss can name what was being decided.
                            let (action_kind, tool_name) = describe_request(&job.request);

                            // GLOBAL scope: reconcile from the durable store rather
                            // than fencing on the broadcast. The store-wide
                            // `Sequence` counter can be observed OUT OF ORDER across
                            // shards — each shard serializes its own sends under its
                            // own lock, but two shards racing can emit `Sequence(2)`
                            // before shard-A's content + `Sequence(1)` reaches this
                            // receiver. A marker-based fence would then clear at 2
                            // with A's earlier content still missing: a
                            // read-your-writes FAIL-OPEN. `get_full_state_for_tenant_scoped`
                            // is a consistent cross-shard snapshot, so a resync can't
                            // be fooled by stream order.
                            //
                            // But global is NOT a rare path — the refmon forwards
                            // EVERY proxied request under `SessionScope::global`, and
                            // an absent `session_id` on CheckToolCall /
                            // CheckAuthorization lands here too — so reloading
                            // unconditionally makes per-request cost grow with the
                            // tenant's entire graph (O(N) per request, O(N²) over a
                            // deployment). Gate it on `snapshot_seq`, which only an
                            // authoritative load writes: if the store-wide counter
                            // hasn't moved past our snapshot, NO write landed since,
                            // so there is nothing this query could miss and the
                            // reload is pure waste. Unsound as a fence *clearance*
                            // signal, sound as a resync *trigger*.
                            // Set by any resync the fence/global block already did, so
                            // the incomplete-graph guard below reuses it instead of
                            // paying a SECOND full-tenant reload in the same query.
                            let mut just_resynced = false;
                            if self.scope.is_global() {
                                if self.snapshot_seq < job.min_sequence {
                                    // Buffered per-node updates are superseded by the
                                    // snapshot (every other full_resync call site
                                    // clears first; keep that invariant so a future
                                    // deletion-bearing update can't be re-applied).
                                    self.pending_updates.clear();
                                    resync_within_budget!(self, job.respond, job_deadline);
                                    just_resynced = true;
                                }
                            } else if self.sequence < job.min_sequence {
                                // A single WALL-CLOCK deadline for the whole fence, not
                                // a per-recv timer. The broadcast ring is store-wide, so
                                // on a busy server `recv()` keeps returning OTHER
                                // sessions' updates within microseconds — those are
                                // filtered without advancing self.sequence, and a
                                // per-recv timeout would re-arm every iteration and never
                                // elapse under sustained cross-session traffic (the check
                                // would still hang). `timeout_at` a fixed deadline bounds
                                // the total wait regardless of how many foreign updates
                                // arrive. The normal path clears in microseconds.
                                //
                                // Clamped to the job's own deadline. The fence
                                // runs inside a job a caller is blocked on, and
                                // its own two seconds on top of a budget
                                // already part-spent would let one job wait
                                // longer than its budget in total — which is
                                // the bound the caller was promised.
                                let fence_start = tokio::time::Instant::now();
                                let deadline =
                                    (fence_start + SEQ_FENCE_TIMEOUT).min(job_deadline);
                                let fence_window =
                                    deadline.saturating_duration_since(fence_start);
                                // Decided inside the loop, acted on outside it:
                                // the reload is an evaluator IPC under the job
                                // budget, and abandoning the job from in here
                                // would only abandon the fence loop.
                                let mut resync_after_fence = false;
                                let mut fence_missed = false;
                                // Set when the replay catch-up below is
                                // abandoned at that deadline, carrying the
                                // moment it was issued so the deny can say how
                                // long the caller waited.
                                let mut catch_up_over_budget: Option<Instant> = None;
                                while self.sequence < job.min_sequence {
                                    match tokio::time::timeout_at(deadline, self.broadcast_rx.recv())
                                        .await
                                    {
                                        Ok(Ok(gu)) => self.process_broadcast(gu),
                                        Ok(Err(broadcast::error::RecvError::Lagged(n))) => {
                                            warn!(
                                                scope = %self.scope,
                                                "lagged {n} during seq wait"
                                            );
                                            // The retained change log first,
                                            // the whole shard only when the
                                            // gap reaches back further than
                                            // the store keeps. A catch-up
                                            // advances the cursor the same way
                                            // a marker does, so re-test the
                                            // fence rather than assuming it
                                            // cleared: if the log did not
                                            // reach min_sequence the wait
                                            // continues, still under the one
                                            // deadline.
                                            //
                                            // Bounded by that same deadline,
                                            // already clamped to the job's
                                            // budget: this is an evaluator IPC
                                            // issued with the caller blocked
                                            // on it, so it may not outlast
                                            // what the caller was promised.
                                            let catch_up_start = Instant::now();
                                            match tokio::time::timeout_at(
                                                deadline,
                                                self.catch_up_from_replay(),
                                            )
                                            .await
                                            {
                                                Ok(CatchUp::Applied) => continue,
                                                Ok(CatchUp::Dead) => return,
                                                Ok(CatchUp::NeedsResync) => {}
                                                Err(_elapsed) => {
                                                    // The catch-up flushes what
                                                    // it replayed, and
                                                    // `flush_pending` took that
                                                    // batch out of
                                                    // `pending_updates` before
                                                    // the IPC just abandoned:
                                                    // nothing resends those
                                                    // records. Fail closed
                                                    // until a reload puts the
                                                    // graph back.
                                                    self.graph_incomplete = true;
                                                    catch_up_over_budget = Some(catch_up_start);
                                                    break;
                                                }
                                            }
                                            // Decided here, acted on outside
                                            // the loop: the reload is an
                                            // evaluator IPC under the job
                                            // budget, and abandoning the job
                                            // from in here would only abandon
                                            // the fence loop.
                                            resync_after_fence = true;
                                            break;
                                        }
                                        Ok(Err(broadcast::error::RecvError::Closed)) => break,
                                        Err(_elapsed) => {
                                            // The marker for min_sequence never arrived
                                            // (writer commit failed before its broadcast,
                                            // then idle — possibly masked by cross-session
                                            // ring traffic).
                                            fence_missed = true;
                                            break;
                                        }
                                    }
                                }
                                if let Some(started) = catch_up_over_budget {
                                    // Not the fence's deny: that one says the
                                    // evaluator is behind and leaves it alone,
                                    // while this one abandoned an IPC the child
                                    // may still be inside. `deny_over_budget`
                                    // marks it as owing a reply, which is what
                                    // starts the silence the stall window
                                    // measures.
                                    self.deny_over_budget(
                                        job.respond,
                                        "the replay catch-up",
                                        "update".to_string(),
                                        String::new(),
                                        started,
                                    );
                                    continue;
                                }
                                if fence_missed {
                                    // Deny this query while the evaluator catches
                                    // up. Keep consuming updates rather than
                                    // reloading the entire graph per request.
                                    warn!(
                                        scope = %self.scope,
                                        policy = %self.policy_hash,
                                        min_sequence = job.min_sequence,
                                        sequence = self.sequence,
                                        fence_secs = fence_window.as_secs_f64(),
                                        "sequence fence not cleared — denying the check"
                                    );
                                    let _ = job.respond.send(Err(format!(
                                        "the evaluator had not caught up to sequence {} \
                                         within {} (it is at {}), so the check fails closed",
                                        job.min_sequence,
                                        human_bound(fence_window),
                                        self.sequence,
                                    )));
                                    continue;
                                }
                                if resync_after_fence {
                                    self.pending_updates.clear();
                                    resync_within_budget!(self, job.respond, job_deadline);
                                    just_resynced = true;
                                }
                            }

                            // A previous bootstrap reset the evaluator and then failed
                            // to load the snapshot back in. Flushing buffered per-node
                            // updates on top of that would build a graph that LOOKS
                            // populated while missing everything older than the failure,
                            // and the query would be answered from it. Retry the reload
                            // once — a genuinely transient cause clears on the retry; a
                            // payload-shaped one won't, and then we fail this check
                            // closed instead of guessing (see below for why the
                            // payload-shaped case is NOT escalated).
                            if self.graph_incomplete {
                                // Skip if the block above already reloaded — retrying
                                // immediately would just repeat the failure at double
                                // the cost, on the refmon's per-request path.
                                if !just_resynced {
                                    self.pending_updates.clear();
                                    resync_within_budget!(self, job.respond, job_deadline);
                                }
                                if self.graph_incomplete {
                                    warn!(
                                        scope = %self.scope,
                                        "graph incomplete after reload — failing check closed"
                                    );
                                    let _ = job.respond.send(Err(
                                        "evaluator graph incomplete: snapshot reload failed".to_string(),
                                    ));
                                    // Deliberately NOT escalating to `mark_process_died()`
                                    // after N failures: the failure count would live on
                                    // THIS task, which the respawn destroys, so it resets
                                    // every time. The deterministic case — the common one,
                                    // since the non-fatal update errors are payload-shaped
                                    // — would stay just as unbounded while gaining a
                                    // subprocess spawn every N requests. Bounding this
                                    // for real needs
                                    // per-scope state in `SessionEvaluatorMap` that
                                    // survives a respawn; until then retrying is strictly
                                    // the cheaper of the two, and both fail closed.
                                    //
                                    // `continue`, not `return`: the subprocess is alive
                                    // (a dead one already returned via `note_if_dead`), so
                                    // exiting would drop a healthy task and orphan its
                                    // entry in the evaluator map.
                                    continue;
                                }
                            }

                            let flush_start = Instant::now();
                            // Whether this flush is about to SAY anything to
                            // the evaluator. An empty one returns without an
                            // IPC, and clearing the mark on that would clear it
                            // on no evidence at all: under steady traffic every
                            // query would reset the stall clock — which
                            // `Outstanding::since` exists to keep still — and
                            // an evaluator that answers nothing, ever, would
                            // never look stalled and never be killed.
                            let flush_speaks = !self.pending_updates.is_empty();
                            let flushed =
                                tokio::time::timeout_at(job_deadline, self.flush_pending()).await;
                            let (pre_query_flush_us, alive) = match flushed {
                                Ok(v) => {
                                    if flush_speaks {
                                        self.clear_outstanding_if_child_spoke();
                                    }
                                    v
                                }
                                Err(_elapsed) => {
                                    // `flush_pending` took the batch out of
                                    // `pending_updates` before the IPC that was
                                    // just abandoned, and the sequence markers
                                    // for those records were consumed when they
                                    // were queued: nothing resends them and no
                                    // fence records their absence. Fail closed
                                    // until a reload puts the graph back.
                                    self.graph_incomplete = true;
                                    self.deny_over_budget(
                                        job.respond,
                                        "the pre-query flush",
                                        "update".to_string(),
                                        String::new(),
                                        flush_start,
                                    );
                                    continue;
                                }
                            };
                            let flush_us = std::mem::take(&mut pending_flush_us)
                                .saturating_add(pre_query_flush_us);
                            let sync_wait_us = sync_start.elapsed().as_micros() as u64;

                            if !alive {
                                let _ = job.respond.send(Err("evaluator subprocess died".to_string()));
                                return;
                            }

                            let (session_nodes, session_edges) = if self.scope.is_global() {
                                (None, None)
                            } else {
                                let (n, e) = self.graph_store.session_counts(&self.scope);
                                (Some(n), Some(e))
                            };

                            let eval_start = Instant::now();
                            // The Arc is cloned so the query future borrows IT
                            // and not `self`, which leaves `&mut self` usable
                            // in the arm that abandons the query.
                            let evaluator = Arc::clone(&self.evaluator);
                            let query = evaluator.query(job.request);
                            tokio::pin!(query);
                            // The clock stops for the LLM oracle. An
                            // `@llm_check` cache miss suspends the evaluation
                            // while an external provider is asked, and that
                            // provider's own timeout is longer than this
                            // budget — so without this, every prompt that
                            // missed the cache would be denied although the
                            // evaluator is healthy. Waiting on somebody else's
                            // API is not the evaluator spinning, which is what
                            // the budget is for. The extension is what the
                            // wait measured, capped at `oracle_bound` AND at
                            // `stall_cap`, so the whole wait ends at whichever
                            // of the two comes first: an oracle that never
                            // answers cannot hold the caller for ever, and it
                            // cannot buy a silent child one oracle bound of
                            // extra life either — an evaluator wedged on the
                            // oracle path is killed one stall window after it
                            // went quiet, like any other.
                            let mut deadline = job_deadline;
                            let mut oracle_credit = Duration::ZERO;
                            let answered = loop {
                                match tokio::time::timeout_at(deadline, &mut query).await {
                                    Ok(r) => break Some(r),
                                    Err(_elapsed) => {
                                        let waited = evaluator.take_oracle_wait();
                                        let room = self
                                            .deadlines
                                            .oracle_bound
                                            .saturating_sub(oracle_credit);
                                        let extend = waited.min(room);
                                        if extend.is_zero() {
                                            break None;
                                        }
                                        // Whatever is left of the credit once
                                        // the stall deadline has taken its
                                        // share. Nothing past `stall_cap` is
                                        // granted, and nothing past it is
                                        // handed to the waiting callers
                                        // either: `oracle_credit_us` is what
                                        // `OracleCredit` extends THEIR
                                        // deadlines by, so it may only carry
                                        // what this wait actually took.
                                        let extended = (deadline + extend).min(stall_cap);
                                        let taken =
                                            extended.saturating_duration_since(deadline);
                                        if taken.is_zero() {
                                            break None;
                                        }
                                        oracle_credit += taken;
                                        deadline = extended;
                                        self.oracle_credit_us.fetch_add(
                                            taken.as_micros() as u64,
                                            std::sync::atomic::Ordering::Relaxed,
                                        );
                                        debug!(
                                            scope = %self.scope,
                                            oracle_wait_us = waited.as_micros() as u64,
                                            oracle_credit_us = oracle_credit.as_micros() as u64,
                                            "the budget clock stopped for the LLM oracle"
                                        );
                                    }
                                }
                            };
                            let result = match answered {
                                Some(r) => {
                                    // The evaluator spoke, so it owes nothing:
                                    // any earlier mark is stale. Only on
                                    // evidence it was reached, though — see
                                    // `clear_outstanding_if_child_spoke`.
                                    self.clear_outstanding_if_child_spoke();
                                    r
                                }
                                None => {
                                    self.deny_over_budget(
                                        job.respond,
                                        "the query",
                                        action_kind,
                                        tool_name,
                                        eval_start,
                                    );
                                    continue;
                                }
                            };
                            let eval_us = eval_start.elapsed().as_micros() as u64;

                            // Per-query timing (off by default; RUST_LOG=sasy_policy=debug).
                            // eval_us dominates once a long session's graph is large.
                            tracing::debug!(scope = %self.scope, sync_wait_us, flush_us, eval_us, edges = self.edge_count, nodes = self.node_count, "query-timing");

                            let died = result.as_ref().err().map(|e| self.note_if_dead(e)).unwrap_or(false);

                            let _ = job.respond.send(
                                result
                                    .map(|eval_response| QueryResult {
                                        eval_response,
                                        sync_wait_us,
                                        flush_us,
                                        eval_us,
                                        graph_nodes: self.node_count,
                                        graph_edges: self.edge_count,
                                        session_nodes,
                                        session_edges,
                                        resolved_policy_id: None,
                                    })
                                    .map_err(|e| e.to_string())
                            );

                            if died {
                                return;
                            }
                        }
                        Err(_) => {
                            info!(
                                scope = %self.scope,
                                "query channel closed"
                            );
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// How long a caller waits for another caller's in-flight spawn
/// before failing closed. A spawn is a (possibly cold) policy
/// compile plus a subprocess fork: 30 s is comfortably past the
/// slowest healthy cold compile (5-7 s) and short enough that a
/// wedged toolchain returns an error to the waiting dispatcher
/// instead of parking it forever.
const SPAWN_DEADLINE: Duration = Duration::from_secs(30);

/// One evaluator spawn per scope.
///
/// [`SessionEvaluatorMap::get_or_spawn`] deliberately calls the
/// factory with no map lock held — a cold Soufflé compile must not
/// freeze every other session's dispatch — so without this gate two
/// callers for the same scope (typically the prewarm listener
/// reacting to the first graph update while the first dispatch
/// arrives) both fork an evaluator and the loser's is dropped. The
/// result is correct either way, but every race costs a subprocess
/// and a bootstrap.
///
/// This is a side table of per-scope locks rather than an in-flight
/// marker inside the map's value, because it is the smaller change:
/// the map's value is an `Arc<SessionEvaluator>` and every reader of
/// it (the idle sweep, the LRU cap, the tenant evictions, the live
/// count) would otherwise have to learn about a variant that carries
/// no evaluator yet. The table holds one small entry per scope being
/// spawned or live: the eviction paths drop the entry with the scope,
/// and a spawner drops its own on the way out, so a spawn that fails
/// leaves nothing behind.
///
/// Lock discipline: `locks` is a leaf lock. It is taken to look an
/// entry up or to drop one and released immediately — never held
/// while waiting for a per-scope lock, for `inner`, or for the
/// factory. It is therefore safe to take while `inner` is held (the
/// eviction paths do exactly that) with no lock-order cycle.
#[derive(Default)]
struct SpawnGate {
    locks: Mutex<HashMap<SessionScope, Arc<Mutex<()>>>>,
}

impl SpawnGate {
    /// The lock for `scope`, created on first use.
    fn handle(&self, scope: &SessionScope) -> Arc<Mutex<()>> {
        Arc::clone(
            self.locks
                .lock()
                .entry(scope.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    /// Reclaim `scope`'s entry if nobody is using it.
    ///
    /// Reclamation is by refcount, not unconditional: the entry goes
    /// only when the table holds the last `Arc`, i.e. no caller is
    /// inside — or waiting to enter — the factory for this scope.
    /// Dropping an entry that a spawner still holds would orphan that
    /// spawner's lock, and the next caller would make a fresh one and
    /// run the factory concurrently with it — the duplicate spawn the
    /// gate exists to remove. Because every clone of the table's `Arc`
    /// is made under `locks` (see [`Self::handle`]), a count of one
    /// observed here cannot grow while this lock is held, so the
    /// removal is not racing a caller that is about to wait.
    ///
    /// Called from every path that removes the scope from the map, and
    /// from the spawner itself once its guard is dropped — the latter
    /// is what keeps a spawn that FAILS (an unresolvable policy id, a
    /// factory error) from leaving an entry behind for a scope that
    /// never becomes live. An entry a spawner still holds survives this
    /// call and is reclaimed by that spawner's own release instead, so
    /// the table cannot outgrow the live sessions plus the spawns in
    /// flight.
    fn forget(&self, scope: &SessionScope) {
        let mut locks = self.locks.lock();
        if locks.get(scope).is_some_and(|l| Arc::strong_count(l) == 1) {
            locks.remove(scope);
        }
    }

    /// Reclaim every unused lock belonging to `tenant`, for the
    /// tenant-wide evictions. In-flight spawns are kept for the same
    /// reason as in [`Self::forget`].
    fn forget_tenant(&self, tenant: &str) {
        self.locks
            .lock()
            .retain(|scope, lock| scope.tenant() != tenant || Arc::strong_count(lock) > 1);
    }

    /// Live entry count. Tests assert the table does not grow past
    /// the sessions that are actually live.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.locks.lock().len()
    }
}

/// One spawn attempt's hold on a scope's gate entry.
///
/// The entry is taken on construction. On drop — every exit path from
/// the spawn: the successful insert, the timeout, an unresolvable
/// policy id, a factory error, the retry budget running out — the
/// lease reclaims it unless the attempt left a live evaluator in the
/// map for the scope, in which case the entry stays for the eviction
/// paths to drop with the scope.
///
/// Reclaiming a failed attempt is what keeps the table bounded. A
/// spawn can fail before anything reaches the map (`peek_policy_for_scope`,
/// `ensure_installed`, `registry.resolve` and the factory all return
/// `Err` after the entry exists), and the scope never becomes live, so
/// no eviction path would ever see it. Both the session id and the
/// policy id come off the wire, so without this a client could add one
/// permanently retained entry per refused request.
///
/// Declaration order matters at the use site: the caller declares the
/// lease before the guard it takes from [`Self::lock`], so the guard
/// is dropped first and this reclamation runs with the per-scope lock
/// already released.
struct SpawnLease<'a> {
    map: &'a SessionEvaluatorMap,
    scope: &'a SessionScope,
    /// `None` only inside [`Drop::drop`], which takes the `Arc` out
    /// so the table's is the last one when `forget` counts.
    handle: Option<Arc<Mutex<()>>>,
}

impl<'a> SpawnLease<'a> {
    fn take(map: &'a SessionEvaluatorMap, scope: &'a SessionScope) -> Self {
        let handle = map.spawn_gate.handle(scope);
        Self {
            map,
            scope,
            handle: Some(handle),
        }
    }

    /// The per-scope lock this attempt serialises on.
    fn lock(&self) -> &Mutex<()> {
        self.handle
            .as_ref()
            .expect("the lease's handle is taken only by its own drop")
    }
}

impl Drop for SpawnLease<'_> {
    fn drop(&mut self) {
        // Ours goes first: `forget` reclaims the entry only when the
        // table holds the last `Arc`, which is exactly the case where
        // no other caller is spawning or waiting for this scope.
        drop(self.handle.take());
        // No lock is held here — the guard above us dropped first —
        // and the order taken is `inner` then `locks`, the same order
        // the eviction paths take.
        let live = self.map.inner.read().contains_key(self.scope);
        if !live {
            self.map.spawn_gate.forget(self.scope);
        }
    }
}

/// What a caller does when another caller is already spawning the
/// evaluator for this scope.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpawnWait {
    /// Wait for the in-flight spawn (bounded by the spawn deadline)
    /// and then take the entry it installed. What a dispatch does: it
    /// has nothing to answer with until the evaluator exists.
    Wait,
    /// Give up immediately. What the prewarm listener does: it must
    /// never block a runtime worker, and the spawn it would have done
    /// is already being done.
    Skip,
}

/// Lazy registry of [`SessionEvaluator`]s keyed by [`SessionScope`].
///
/// On first traffic for a scope, calls the registered factory to
/// spawn a fresh evaluator (which the spawned task then bootstraps
/// from [`GraphStore::get_session_state`]). Subsequent queries for
/// the same scope land on the same evaluator. Eviction (idle TTL
/// or explicit end-session) drops the [`SessionEvaluator`] but
/// leaves the graph state in [`GraphStore`] intact, so a resumed
/// scope reconstructs an evaluator whose EDB matches its full
/// history.
pub struct SessionEvaluatorMap {
    inner: RwLock<HashMap<SessionScope, Arc<SessionEvaluator>>>,
    /// Per-scope policy binding. Set on first traffic from
    /// the request's `policy_id` (or the tenant default), and
    /// rotated by [`Self::set_session_policy`]. Sessions present
    /// in `inner` but absent from this map are unbound — that
    /// only happens transiently during eviction.
    session_to_policy: RwLock<HashMap<SessionScope, PolicyId>>,
    /// Per-tenant registry of installed policies. Resolves
    /// `(tenant, policy_id) → factory`.
    registry: Arc<PolicyRegistry>,
    graph_store: Arc<GraphStore>,
    flush_interval: Duration,
    query_capacity: usize,
    /// Hard cap on live evaluator count. ``None`` = unbounded.
    max_live_sessions: Option<usize>,
    /// Wall-clock bounds every session evaluator this map spawns runs under.
    deadlines: EvaluationDeadlines,
    /// Per-session kill records. Kept here, not on the session's task, because
    /// the kill destroys the task that would be counting it. Entries are
    /// dropped by the eviction sweep and by every evict path once their kills
    /// have aged out AND no task still holds them.
    kill_ledgers: Arc<KillLedgers>,
    /// Serialises the factory call per scope, so a scope's evaluator
    /// is spawned once however many callers arrive at once.
    spawn_gate: SpawnGate,
    /// How long a caller waits for another caller's in-flight spawn
    /// before failing closed. [`SPAWN_DEADLINE`] outside tests.
    spawn_deadline: Duration,
    /// The operator settings the lazy install consults before it compiles
    /// persisted functor source. Set once at startup by the binary (see
    /// [`Self::set_policy_service_config`]); the conservative default —
    /// refuse user-admitted functor source — applies until it is.
    policy_service_config: RwLock<crate::service::PolicyServiceConfig>,
}

impl SessionEvaluatorMap {
    /// Construct a new map and spawn the pre-warm listener task.
    /// Returns an [`Arc`] because the listener holds a weak handle
    /// back to the map.
    pub fn new(
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
    ) -> Arc<Self> {
        Self::with_deadlines(
            graph_store,
            flush_interval,
            query_capacity,
            EvaluationDeadlines::from_env(),
        )
    }

    /// Like [`Self::new`] but with the evaluation deadlines given rather than
    /// read from the environment — what the binary's `--query-timeout-secs` /
    /// `--evaluator-stall-secs` flags reach, and what a test constructs to put
    /// a budget or a stall window within its own runtime.
    pub fn with_deadlines(
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
        deadlines: EvaluationDeadlines,
    ) -> Arc<Self> {
        Self::with_eviction(
            graph_store,
            flush_interval,
            query_capacity,
            resolve_idle_ttl(),
            resolve_sweep_interval(),
            resolve_max_live_sessions(),
            deadlines,
        )
    }

    /// Like [`Self::new`] but with explicit eviction parameters.
    /// ``idle_ttl == 0`` (or ``sweep_interval == 0``) disables the
    /// eviction sweep — useful in tests that don't want a
    /// background task interfering with timing.
    /// ``max_live_sessions == None`` leaves the map unbounded.
    #[allow(clippy::too_many_arguments)]
    pub fn with_eviction(
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
        idle_ttl: Duration,
        sweep_interval: Duration,
        max_live_sessions: Option<usize>,
        deadlines: EvaluationDeadlines,
    ) -> Arc<Self> {
        Self::build(
            graph_store,
            flush_interval,
            query_capacity,
            idle_ttl,
            sweep_interval,
            max_live_sessions,
            deadlines,
            SPAWN_DEADLINE,
        )
    }

    /// Like [`Self::with_eviction`] but with an explicit spawn
    /// deadline. A test seam: the production deadline is 30 s, and a
    /// test that pins the fail-closed behaviour of a wedged factory
    /// must not sleep for it.
    #[cfg(test)]
    fn with_spawn_deadline(
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
        spawn_deadline: Duration,
    ) -> Arc<Self> {
        Self::build(
            graph_store,
            flush_interval,
            query_capacity,
            Duration::ZERO,
            Duration::ZERO,
            None,
            // The spawn-deadline seam is about a wedged factory, not about the
            // evaluation bounds, so those stay what the process runs with.
            EvaluationDeadlines::from_env(),
            spawn_deadline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        graph_store: Arc<GraphStore>,
        flush_interval: Duration,
        query_capacity: usize,
        idle_ttl: Duration,
        sweep_interval: Duration,
        max_live_sessions: Option<usize>,
        deadlines: EvaluationDeadlines,
        spawn_deadline: Duration,
    ) -> Arc<Self> {
        let map = Arc::new(Self {
            inner: RwLock::new(HashMap::new()),
            session_to_policy: RwLock::new(HashMap::new()),
            registry: Arc::new(PolicyRegistry::new()),
            graph_store,
            flush_interval,
            query_capacity,
            max_live_sessions,
            deadlines,
            kill_ledgers: Arc::new(KillLedgers::default()),
            spawn_gate: SpawnGate::default(),
            spawn_deadline,
            policy_service_config: RwLock::new(Default::default()),
        });
        Self::spawn_prewarm_listener(&map);
        Self::spawn_eviction_sweep(&map, idle_ttl, sweep_interval);
        // Orphan sweep for the policy registry runs on the same
        // cadence as the session sweep — both are housekeeping
        // tasks and there's no reason to give policies a separate
        // interval knob today. The idle threshold *is* separate
        // (policies live longer than sessions). Zero disables.
        Self::spawn_policy_orphan_sweep(&map, resolve_policy_idle_ttl(), sweep_interval);
        map
    }

    /// Periodically call [`PolicyRegistry::evict_orphans`] so
    /// unreferenced policy variants don't accumulate in
    /// long-running tenants. Defaults are exempt from the sweep.
    fn spawn_policy_orphan_sweep(map: &Arc<Self>, idle_ttl: Duration, sweep_interval: Duration) {
        if idle_ttl.is_zero() || sweep_interval.is_zero() {
            return;
        }
        let weak = Arc::downgrade(map);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(sweep_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(map) = weak.upgrade() else { return };
                let n = map.registry.evict_orphans(idle_ttl);
                if n > 0 {
                    debug!(count = n, "policy orphan sweep dropped variants");
                }
            }
        });
    }

    /// Access the per-tenant policy registry. Service handlers use
    /// this directly when uploading a variant policy
    /// (`as_variant=true`) so they can install without the legacy
    /// "evict every session in the tenant" semantics that
    /// [`Self::swap_factory`] applies.
    pub fn registry(&self) -> &Arc<PolicyRegistry> {
        &self.registry
    }

    /// Spawn the idle-TTL eviction sweep. Periodically scans the
    /// session map for evaluators whose ``last_active`` is older
    /// than ``idle_ttl`` and drops them. The drop tears down the
    /// per-session subprocess but preserves graph state in the
    /// store, so a later query for the same scope transparently
    /// re-spawns and re-bootstraps.
    ///
    /// The listener holds a [`Weak`] back to the map so the sweep
    /// stops automatically when the map is dropped. ``idle_ttl == 0``
    /// or ``sweep_interval == 0`` disables the sweep entirely.
    fn spawn_eviction_sweep(map: &Arc<Self>, idle_ttl: Duration, sweep_interval: Duration) {
        if idle_ttl.is_zero() || sweep_interval.is_zero() {
            return;
        }
        let weak = Arc::downgrade(map);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(sweep_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Skip the immediate first tick; otherwise the sweep
            // fires before any session has had a chance to record
            // activity.
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(map) = weak.upgrade() else { return };
                let now = Instant::now();
                let stale: Vec<SessionScope> = map
                    .inner
                    .read()
                    .iter()
                    .filter_map(|(scope, se)| {
                        let elapsed = now.duration_since(se.last_active());
                        if elapsed >= idle_ttl {
                            Some(scope.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                for scope in stale {
                    // Drop the evaluator only — keep the
                    // session→policy binding so a resumed
                    // session comes back under the same variant.
                    if map.evict_evaluator_only(&scope) {
                        debug!(scope = %scope, "evicted idle session evaluator (binding preserved)");
                    }
                }
                // Kill ledgers that no task holds and whose kills have all aged
                // out say nothing any more; keeping one per session ever seen
                // would be a leak.
                map.kill_ledgers.prune_spent();
            }
        });
    }

    /// Spawn a background task that subscribes to graph store
    /// broadcasts and lazily spawns a [`SessionEvaluator`] for any
    /// previously-unseen scope. This keeps first-auth
    /// latency low: by the time a query arrives for a new scope,
    /// the evaluator is already up and the bootstrap has been
    /// amortized across the events that arrived before the query.
    ///
    /// Sessions that exist only in the persisted RocksDB graph
    /// (loaded into memory at startup) do *not* trigger pre-warm:
    /// startup loads do not broadcast, so persisted-but-idle
    /// sessions remain dormant until they see live traffic.
    /// The listener holds a [`Weak`] back to the map so the map
    /// can drop normally when no longer referenced.
    fn spawn_prewarm_listener(map: &Arc<Self>) {
        let weak = Arc::downgrade(map);
        let mut rx = map.graph_store.subscribe();
        tokio::spawn(async move {
            loop {
                let result = rx.recv().await;
                let Some(map) = weak.upgrade() else { return };
                match result {
                    Ok(gu) => {
                        let scope: Option<SessionScope> = match &gu {
                            sasy_graph::GraphUpdate::NodeCreated { scope, .. } => {
                                if scope.is_global() {
                                    None
                                } else {
                                    Some(scope.clone())
                                }
                            }
                            sasy_graph::GraphUpdate::EdgeCreated { scope, .. } => {
                                if scope.is_global() {
                                    None
                                } else {
                                    Some(scope.clone())
                                }
                            }
                            _ => None,
                        };
                        if let Some(scope) = scope {
                            // Prewarm uses the session's existing
                            // binding (or the tenant default if
                            // unbound). Live broadcasts never carry
                            // policy_id — that lives on the auth
                            // request — so passing `None` is correct.
                            //
                            // Non-blocking: prewarm never competes with a
                            // request. If a dispatch is already spawning this
                            // scope's evaluator, `try_get_or_spawn` returns
                            // immediately instead of waiting behind a compile
                            // — prewarm has nothing to add to a spawn that is
                            // already happening, and blocking here would park
                            // a runtime worker on it.
                            if let Err(e) = map.try_get_or_spawn(&scope, None) {
                                debug!(scope = %scope, "prewarm failed: {}", e);
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("prewarm listener lagged {n}");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                }
            }
        });
    }

    /// Install a new evaluator factory as the default policy for
    /// `tenant`, and drop every existing per-session evaluator in
    /// that tenant. The next query for any session in `tenant`
    /// re-spawns one via the new factory and re-bootstraps from the
    /// graph store. Mirrors the worker-pool swap shape so hot-reload
    /// of policies stays atomic.
    ///
    /// Tenant-unaware callers go through [`Self::swap_factory`],
    /// which targets the `"default"` tenant.
    pub fn install_policy_default_for_tenant(
        &self,
        tenant: &str,
        factory: EvaluatorFactory,
        backend_name: String,
    ) -> PolicyId {
        let policy_id = self.registry.install(tenant, factory, backend_name, true);

        // Evict every live session in the affected tenant so they
        // re-bootstrap under the new default. Sessions in other
        // tenants are untouched — that's the point of multi-tenant
        // isolation. Bindings are dropped too; the next request
        // re-binds to the new default (or whichever variant the
        // request specifies).
        let mut sessions = self.inner.write();
        let mut bindings = self.session_to_policy.write();
        sessions.retain(|scope, _| scope.tenant() != tenant);
        bindings.retain(|scope, _| scope.tenant() != tenant);
        // The spawn gate follows the evaluators out, so its table
        // never outgrows the live sessions.
        self.spawn_gate.forget_tenant(tenant);

        policy_id
    }

    /// Backward-compatible shim: installs `factory` as the default
    /// for the `"default"` tenant. Callers that need multi-tenant
    /// or multi-policy semantics should call
    /// [`Self::install_policy_default_for_tenant`] or use
    /// [`Self::registry`] directly.
    /// Install the operator's deployment decisions. Called once by the binary
    /// at startup, before anything is served; the lazy install reads them
    /// every time it is about to compile persisted functor source.
    pub fn set_policy_service_config(&self, config: crate::service::PolicyServiceConfig) {
        *self.policy_service_config.write() = config;
    }

    /// The decisions in force now — what the lazy install would consult on the
    /// next dispatch. Lets a caller check that the config it meant to install
    /// is the one this map holds.
    pub fn policy_service_config(&self) -> crate::service::PolicyServiceConfig {
        self.policy_service_config.read().clone()
    }

    pub fn swap_factory(&self, factory: EvaluatorFactory, backend_name: String) -> PolicyId {
        self.install_policy_default_for_tenant("default", factory, backend_name)
    }

    /// Resolve the policy id for `scope`, given an optional client-
    /// supplied id from the request. Rules:
    ///
    /// * If the session already has a binding:
    ///     - `requested = None` → use the existing binding.
    ///     - `requested = Some(b) && b == binding` → ok.
    ///     - `requested = Some(b) && b != binding` → error
    ///       (`Status::FailedPrecondition` upstream). Switching
    ///       requires the explicit `set_session_policy` RPC.
    /// * If no binding exists:
    ///     - `requested = Some(id)` → bind the session to `id`.
    ///     - `requested = None` → bind to the tenant default
    ///       (errors if the tenant has no default).
    ///
    /// Validation is deliberately deferred to
    /// [`Self::ensure_installed`] (called from [`Self::get_or_spawn`]
    /// after this returns): a binding may legitimately point at a
    /// `content_hash` whose `PolicyEntry` hasn't been compiled into
    /// this process yet (the lazy-replay case). The presence /
    /// absence of an installed entry is checked at compile-on-demand
    /// time, not here.
    fn resolve_policy_for_scope(
        &self,
        scope: &SessionScope,
        requested: Option<&PolicyId>,
    ) -> Result<PolicyId, String> {
        self.resolve_policy_for_scope_inner(scope, requested, true)
    }

    /// Resolve WITHOUT recording a first-touch binding.
    ///
    /// `get_or_spawn`'s unlocked phase only needs to know what to compile.
    /// Binding there records a row derived from a tenant default read outside
    /// the map lock, so a force update landing in between is undone by a
    /// binding to the policy it replaced — and the re-check under the lock
    /// then reads that same stale row and agrees with it, installing an
    /// evaluator for the superseded policy that nothing later corrects.
    /// Binding happens under the lock instead, where the rollout cannot
    /// interleave.
    fn peek_policy_for_scope(
        &self,
        scope: &SessionScope,
        requested: Option<&PolicyId>,
    ) -> Result<PolicyId, String> {
        self.resolve_policy_for_scope_inner(scope, requested, false)
    }

    fn resolve_policy_for_scope_inner(
        &self,
        scope: &SessionScope,
        requested: Option<&PolicyId>,
        bind: bool,
    ) -> Result<PolicyId, String> {
        let bindings = self.session_to_policy.read();
        if let Some(existing) = bindings.get(scope) {
            return match requested {
                Some(r) if r != existing => Err(format!(
                    "session {} is bound to policy {} but request asked for {}; \
                     use SetSessionPolicy to switch",
                    scope, existing, r
                )),
                _ => Ok(existing.clone()),
            };
        }
        drop(bindings);

        // First-touch binding: pick the explicit request or fall
        // through to the tenant default. The default itself may be
        // a lazy reservation pointing at a content hash that
        // hasn't been compiled yet — that's fine, ensure_installed
        // materializes it.
        let resolved = match requested {
            Some(r) => r.clone(),
            None => self
                .registry
                .default_for(scope.tenant())
                .ok_or_else(|| format!("no policy registered for tenant '{}'", scope.tenant()))?,
        };
        if !bind {
            // Peek only. See `get_or_spawn`.
            return Ok(resolved);
        }
        self.session_to_policy
            .write()
            .insert(scope.clone(), resolved.clone());
        debug!(
            scope = %scope,
            policy = %resolved,
            requested = ?requested.map(|p| p.to_string()),
            "session bound to policy"
        );
        Ok(resolved)
    }

    /// Make sure the policy identified by `content_hash` is
    /// installed in the registry. If it already is, fast-return.
    /// Otherwise fetch the source bundle from the graph store and
    /// compile + install it via [`PolicyRegistry::install_dedup`].
    /// Returns Err if no graph store is wired (unit-test case
    /// without persistence), the source is missing, or compile
    /// fails.
    ///
    /// This is the lazy-replay hot path: at boot we only rehydrate
    /// the binding tables; the actual compile happens here on first
    /// dispatch for each session that wakes up. The souffle build
    /// cache makes the per-session hit ≈150 ms warm.
    fn ensure_installed(&self, tenant: &str, content_hash: &str) -> Result<(), String> {
        let policy_id = PolicyId::from_string(content_hash);
        if self.registry.contains(tenant, &policy_id) {
            return Ok(());
        }
        // The functor gate, run again. This source was admitted at upload
        // time under the settings of the process that took the upload; the
        // settings in force now are the ones that count, because it is this
        // process that is about to compile the C++ and load it. The store
        // keeps the two admission classes as separate records, so this asks
        // within the class in force rather than trusting whichever uploader
        // wrote last. Source with no functors is unaffected — there is nothing
        // to admit.
        let config = self.policy_service_config.read().clone();
        let source =
            match crate::replay::admitted_policy_source(&self.graph_store, content_hash, &config) {
                // The class it was found under has already decided the gate;
                // only the bundle is needed from here.
                Ok(Some((_, s))) => s,
                Ok(None) => {
                    return Err(format!(
                        "no persisted source for content_hash={content_hash}: \
                         re-upload via SetPolicy"
                    ))
                }
                Err(reason) if reason.starts_with(crate::service::FUNCTOR_REFUSED_AT_LOAD) => {
                    warn!(
                        tenant,
                        content_hash, "refusing to lazily compile a persisted policy: {reason}"
                    );
                    // The refusal leads the message. Every reader of this
                    // marker matches it as a PREFIX (see
                    // `crate::service::FUNCTOR_REFUSED_AT_LOAD`), so that a
                    // policy source which merely quotes the marker text and
                    // then fails to compile cannot be reported as a functor
                    // refusal.
                    return Err(format!(
                        "{reason} (policy {content_hash} is not installed for tenant {tenant})"
                    ));
                }
                Err(e) => return Err(e),
            };
        let (factory, backend) = crate::replay::compile_factory_from_source(
            &source.policy_source,
            &source.functor_source,
            &source.backend,
        )?;
        // make_default=false here: if this hash is the tenant
        // default, the `defaults` map already points at it from the
        // boot rehydration. install_dedup just needs to materialize
        // the entry; flipping make_default would re-run the
        // demote-prev-default branch and clobber that pointer.
        self.registry
            .install_dedup(tenant, content_hash, factory, backend, false);
        // ...but the fresh entry now has `is_default=false` even
        // when it *is* the tenant default. The orphan sweep checks
        // `entries[…].is_default`, so without this fix-up the
        // tenant default could be swept on first idle pass. Sync
        // the entry's flag with the `defaults` map.
        if self.registry.default_for(tenant).as_ref() == Some(&policy_id) {
            self.registry.mark_entry_default(tenant, &policy_id);
        }
        debug!(
            tenant,
            content_hash, "lazy-installed persisted policy on first dispatch"
        );
        Ok(())
    }

    /// Rehydrate a `(scope → content_hash)` binding from disk.
    /// Used by the boot replay path; never validates that the
    /// policy is installed (that's [`Self::ensure_installed`]'s
    /// job, run on first dispatch).
    pub fn lazy_bind_session(&self, scope: SessionScope, content_hash: &str) {
        let policy_id = PolicyId::from_string(content_hash);
        self.session_to_policy.write().insert(scope, policy_id);
    }

    /// Rehydrate a tenant default from disk. Same shape as
    /// [`Self::lazy_bind_session`] — registers the pointer without
    /// requiring the underlying policy to be compiled yet.
    pub fn lazy_set_default(&self, tenant: &str, content_hash: &str) {
        self.registry.reserve_default(tenant, content_hash);
    }

    /// Look up or lazily spawn the evaluator for ``scope``,
    /// resolving the policy variant per [`Self::resolve_policy_for_scope`].
    /// Returns the evaluator together with the [`PolicyId`] the scope
    /// actually resolved to, so the caller can attribute per-policy
    /// rule metadata to the deciding policy.
    fn get_or_spawn(
        &self,
        scope: &SessionScope,
        requested_policy: Option<&PolicyId>,
    ) -> Result<(Arc<SessionEvaluator>, PolicyId), String> {
        self.get_or_spawn_waiting(scope, requested_policy, SpawnWait::Wait)
    }

    /// [`Self::get_or_spawn`] for a caller that must not block: if
    /// another caller is already spawning this scope's evaluator it
    /// returns an error instead of waiting. Used by the prewarm
    /// listener.
    fn try_get_or_spawn(
        &self,
        scope: &SessionScope,
        requested_policy: Option<&PolicyId>,
    ) -> Result<(Arc<SessionEvaluator>, PolicyId), String> {
        self.get_or_spawn_waiting(scope, requested_policy, SpawnWait::Skip)
    }

    fn get_or_spawn_waiting(
        &self,
        scope: &SessionScope,
        requested_policy: Option<&PolicyId>,
        wait: SpawnWait,
    ) -> Result<(Arc<SessionEvaluator>, PolicyId), String> {
        // Fast path. Skip if the cached evaluator's subprocess died —
        // we fall through to respawn under the write lock.
        if let Some(se) = self.inner.read().get(scope) {
            if !se.process_died() {
                // Even on cache hit we have to validate the requested
                // policy against the binding — otherwise a client could
                // send a stale id and silently get the wrong policy.
                let policy_id = self.resolve_policy_for_scope(scope, requested_policy)?;
                return Ok((Arc::clone(se), policy_id));
            }
        }

        // Compile and spawn without holding the map lock, so cold sessions do
        // not block unrelated live sessions. A per-scope gate admits one factory
        // call at a time; waiters time out after spawn_deadline and fail closed.
        // Recheck binding, occupancy, and the LRU cap when reacquiring the lock:
        // all may change during compilation. Dropping the lease removes the
        // gate on every exit path, including invalid-policy failures.
        let lease = SpawnLease::take(self, scope);
        let _spawn_guard = match wait {
            SpawnWait::Wait => match lease.lock().try_lock_for(self.spawn_deadline) {
                Some(guard) => guard,
                None => {
                    return Err(format!(
                        "timed out after {:?} waiting for an in-flight evaluator \
                         spawn for {scope}",
                        self.spawn_deadline
                    ));
                }
            },
            // Prewarm is an optimisation, never a requirement: if a spawn is
            // already in flight for this scope there is nothing to gain by
            // waiting for it (and blocking here would park a runtime worker
            // on a Soufflé compile), so skip. The dispatch that arrives next
            // finds the entry the other caller installed.
            SpawnWait::Skip => match lease.lock().try_lock() {
                Some(guard) => guard,
                None => {
                    return Err(format!("spawn already in flight for {scope}"));
                }
            },
        };
        //
        // Bounded, because the retry is driven by concurrent rebinds: each
        // pass either installs, hands back somebody else's evaluator, or
        // observes the binding move under it.
        const SPAWN_ATTEMPTS: usize = 4;
        for _ in 0..SPAWN_ATTEMPTS {
            // Re-check under the write lock, dropping a dead handle so the
            // work below spawns its replacement. The binding is preserved —
            // it lives in `session_to_policy`, not here.
            {
                let mut map = self.inner.write();
                if let Some(se) = map.get(scope) {
                    if se.process_died() {
                        map.remove(scope);
                    } else {
                        // Resolve while still holding `map`, so the evaluator
                        // and the id describing it are read at one instant. A
                        // rebind evicts under this same lock, so it cannot
                        // land between the two and leave us returning the old
                        // evaluator labelled with the new policy's id.
                        // (Lock order: `inner` then `session_to_policy`,
                        // matching `set_session_policy`.)
                        let policy_id = self.resolve_policy_for_scope(scope, requested_policy)?;
                        return Ok((Arc::clone(se), policy_id));
                    }
                }
            }

            // The kill cap, before any of the expensive work: a session
            // whose evaluator has been killed KILL_CAP times inside
            // KILL_WINDOW is not given another one. Every check is refused
            // with a reason saying so until the oldest kill ages out. Checked
            // here and not at dispatch, so a session that recovered is never
            // deprived of a live, healthy evaluator; checked before the
            // compile and the fork, so a deterministic runaway stops costing
            // them.
            let kills = self.kill_ledger_for(scope);
            if let Some(remaining) = kills.blocked_for() {
                return Err(format!(
                    "the policy evaluator for {scope} was killed {KILL_CAP} times within \
                     {} minutes and is not being respawned for another {}; \
                     checks fail closed until then",
                    KILL_WINDOW.as_secs() / 60,
                    human_bound(remaining),
                ));
            }

            // Unlocked. Lazy compile: if the binding/default points at a
            // policy whose entry has not been materialized in this process yet
            // (the post-restart case), compile and install it now. This is what
            // makes startup O(1) regardless of how many sessions were
            // persisted — only sessions that actually wake up pay the compile.
            let policy_id = self.peek_policy_for_scope(scope, requested_policy)?;
            self.ensure_installed(scope.tenant(), policy_id.as_str())?;
            let (factory, _backend, _) = self
                .registry
                .resolve(scope.tenant(), Some(&policy_id))
                .map_err(|e| e.to_string())?;
            let evaluator = factory().map_err(|e| format!("evaluator factory failed: {}", e))?;

            let mut map = self.inner.write();
            if let Some(se) = map.get(scope) {
                if !se.process_died() {
                    // Somebody else got there first. Theirs is live and
                    // bootstrapped; ours is surplus and is dropped here.
                    //
                    // Re-resolve rather than returning the id from before the
                    // lock: the binding can have moved while we were
                    // compiling, and handing back a stale id would answer a
                    // client that pinned the OLD policy without the
                    // failed-precondition it gets on every other path, and
                    // would attribute the denial trace to the wrong policy's
                    // rule metadata.
                    //
                    // Still under `map`, so the pair is read at one instant:
                    // a rebind evicts under this lock, and cannot slip
                    // between the lookup and the resolve.
                    let policy_id = self.resolve_policy_for_scope(scope, requested_policy)?;
                    return Ok((Arc::clone(se), policy_id));
                }
                map.remove(scope);
            }

            // The binding can have moved while we were unlocked — a
            // `SetPolicy(Session)` rebind takes this same lock and evicts.
            // Installing an evaluator built for the superseded policy would
            // enforce the old one until something next evicted it, so start
            // over instead.
            if self.resolve_policy_for_scope(scope, requested_policy)? != policy_id {
                drop(map);
                debug!(scope = %scope, "policy rebound mid-spawn; retrying");
                continue;
            }

            // Cap enforcement, under the same lock as the insert so the map
            // can never transiently exceed ``cap``. Deferred until the policy
            // has resolved and compiled — an unservable request returns its
            // error above without churning a healthy live session.
            //
            // Only the live evaluator is dropped, never the session→policy
            // binding: a resumed session must come back under the variant it
            // was bound to, or memory pressure would silently drop it to the
            // tenant default and break the multi-policy isolation contract.
            if let Some(cap) = self.max_live_sessions {
                if map.len() >= cap {
                    let lru = map
                        .iter()
                        .min_by_key(|(_, se)| se.last_active())
                        .map(|(scope, _)| scope.clone());
                    if let Some(scope) = lru {
                        map.remove(&scope);
                        // The evicted scope's gate entry goes with it. (Not
                        // this scope's: we are holding its gate.)
                        self.spawn_gate.forget(&scope);
                        debug!(
                            scope = %scope,
                            cap,
                            "evicted LRU session evaluator (live count at cap; binding preserved)"
                        );
                    }
                }
            }

            // Config for the resolved binding, read UNDER the lock. It is two
            // point reads, and holding the lock across them is what keeps a
            // concurrent `update_session_metadata` from landing between the
            // read and the insert — where its reseed would find no evaluator
            // and ours would be seeded with the value it just replaced.
            let metadata = self.assemble_metadata(scope, policy_id.as_str())?;

            let se = Arc::new(SessionEvaluator::spawn(
                scope.clone(),
                evaluator,
                Arc::clone(&self.graph_store),
                self.flush_interval,
                self.query_capacity,
                metadata,
                self.deadlines,
                policy_id.to_string(),
                KillLedgerRef::new(Arc::clone(&self.kill_ledgers), scope.clone()),
            ));
            map.insert(scope.clone(), Arc::clone(&se));
            return Ok((se, policy_id));
        }

        Err(format!(
            "gave up spawning an evaluator for {scope} after {SPAWN_ATTEMPTS} attempts: \
             the session's policy binding kept changing underneath it"
        ))
    }

    /// This session's kill ledger, created on first use.
    ///
    /// Lock order: the ledger table is a leaf. It is taken alone here and in
    /// the session task, and under `inner` nowhere — so it orders against
    /// nothing and cannot participate in a cycle.
    fn kill_ledger_for(&self, scope: &SessionScope) -> Arc<KillLedger> {
        self.kill_ledgers.for_scope(scope)
    }

    /// Combine this binding's static config with the session's dynamic facts
    /// into the `PolicyMetadata` seed set.
    ///
    /// Static config is read from the binding `(scope, policy_id)` first and
    /// only falls back to the policy-wide entry when the binding carries none.
    /// The two are alternatives rather than a merge: config named on a
    /// session-scoped bind is that binding's configuration in full, so merging
    /// could resurrect a tenant-default value the caller meant to drop. The
    /// fallback is what keeps a Default/Force bind, a precompiled profile, and
    /// any config written before this key existed working unchanged.
    ///
    /// Every read here fails closed, and the error is returned rather than
    /// absorbed. A read error is not the same as "no configuration", but
    /// mapping it to the absent case makes the two indistinguishable: the
    /// session spawns with the tenant default in place of its own pin, or with
    /// no facts at all, and keeps that fact set for the evaluator's lifetime —
    /// so one transient RocksDB error decides how that session is policed
    /// until it is next evicted. Refusing the authorization is the only safe
    /// answer to "we cannot tell what this session's configuration is".
    fn assemble_metadata(
        &self,
        scope: &SessionScope,
        policy_id: &str,
    ) -> Result<Vec<PolicyMetadataFact>, String> {
        let to_fact = |f: sasy_graph::PolicyMetadataFact| PolicyMetadataFact {
            rel: f.rel,
            a: f.a,
            b: f.b,
        };
        let binding: Option<Vec<PolicyMetadataFact>> =
            match self.graph_store.get_binding_metadata(scope, policy_id) {
                Ok(facts) => facts.map(|f| f.into_iter().map(to_fact).collect()),
                Err(e) => {
                    warn!(scope = %scope, policy_id, error = %e, "read binding metadata failed");
                    return Err(format!("could not read binding config for {scope}: {e}"));
                }
            };
        let mut out: Vec<PolicyMetadataFact> = match binding {
            // This session pinned itself: its config is authoritative, even when
            // it named none. Do not fall through.
            Some(facts) => facts,
            // No bind of its own — it is running on the tenant default, so the
            // policy-wide config is the right one to seed.
            None => match self
                .graph_store
                .get_policy_metadata(scope.tenant(), policy_id)
            {
                Ok(facts) => facts.into_iter().map(to_fact).collect(),
                Err(e) => {
                    warn!(policy_id, error = %e, "read policy metadata failed");
                    return Err(format!(
                        "could not read tenant config for {}: {e}",
                        scope.tenant()
                    ));
                }
            },
        };
        match self.graph_store.get_session_metadata(scope) {
            Ok(facts) => out.extend(facts.into_iter().map(to_fact)),
            Err(e) => {
                warn!(scope = %scope, error = %e, "read session metadata failed");
                return Err(format!("could not read session config for {scope}: {e}"));
            }
        }
        Ok(out)
    }

    /// Append session-scoped dynamic metadata (e.g. a detaint decision) and, if
    /// the session is live, re-seed its evaluator with the full assembled set.
    /// Append-only + persisted, so a respawn/bootstrap re-reads it.
    pub async fn update_session_metadata(
        &self,
        scope: &SessionScope,
        facts: Vec<PolicyMetadataFact>,
    ) -> Result<(), String> {
        if facts.is_empty() {
            return Ok(());
        }
        validate_transportable_request(&EvalRequest::SetMetadata {
            facts: facts.clone(),
        })
        .map_err(|e| e.to_string())?;

        let graph_facts: Vec<sasy_graph::PolicyMetadataFact> = facts
            .iter()
            .map(|f| sasy_graph::PolicyMetadataFact {
                rel: f.rel.clone(),
                a: f.a.clone(),
                b: f.b.clone(),
            })
            .collect();
        self.graph_store
            .append_session_metadata(scope, &graph_facts)
            .map_err(|e| e.to_string())?;

        // Reseed the live evaluator (if any) with static + session facts.
        let live = self.inner.read().get(scope).cloned();
        if let Some(se) = live {
            let policy_id = self.resolve_policy_for_scope(scope, None)?;
            let full = self.assemble_metadata(scope, policy_id.as_str())?;
            se.push_metadata(full).await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Dispatch an authorization query under ``scope``. Lazily
    /// spawns the per-scope evaluator on first traffic.
    /// `requested_policy` carries the wire-supplied
    /// [`AuthorizationRequest::policy_id`] (or `None` if absent).
    ///
    /// If the underlying subprocess died during this call (or had
    /// already died on a prior call), the live evaluator is evicted
    /// (binding preserved) so the next dispatch respawns and
    /// re-bootstraps from RocksDB. Without this, a single dead
    /// subprocess would brick the session permanently.
    pub async fn dispatch(
        &self,
        scope: &SessionScope,
        requested_policy: Option<&PolicyId>,
        request: EvalAuthRequest,
        min_sequence: i64,
    ) -> Result<QueryResult, String> {
        let (evaluator, resolved_policy_id) = self.get_or_spawn(scope, requested_policy)?;
        let mut result = evaluator.dispatch(request, min_sequence).await;
        if evaluator.process_died() {
            // Drop the live handle but keep the session→policy
            // binding — a respawn must come back under the same
            // variant the session was bound to.
            self.evict_evaluator_only(scope);
        }
        // Attach the resolved binding so the engine can pick this
        // policy's rule metadata for any denial trace.
        if let Ok(qr) = result.as_mut() {
            qr.resolved_policy_id = Some(resolved_policy_id);
        }
        result
    }

    /// Rebind `scope` to `policy_id`, evicting its current
    /// evaluator (so the next dispatch re-spawns under the new
    /// policy and re-bootstraps from the same preserved graph
    /// state). The `policy_id` must already exist in the scope's
    /// tenant. Returns `true` iff a live evaluator was evicted.
    pub fn set_session_policy(
        &self,
        scope: &SessionScope,
        policy_id: &PolicyId,
    ) -> Result<bool, String> {
        if !self.registry.contains(scope.tenant(), policy_id) {
            return Err(format!(
                "policy {} not found in tenant {}",
                policy_id,
                scope.tenant()
            ));
        }
        // Lock order: `inner` first, then `session_to_policy`.
        // `get_or_spawn` holds `inner.write()` while calling
        // `resolve_policy_for_scope`, which can acquire
        // `session_to_policy.write()` on a first-touch binding —
        // so any other code path must take the locks in the same
        // order to avoid a deadlock between a SetPolicy(scope=Session)
        // rebind and a concurrent first-touch authorization.
        //
        // Inside the critical section the WRITES still happen in
        // (binding, eviction) order so an external observer can't
        // see a stale binding paired with an evicted evaluator.
        let mut inner = self.inner.write();
        let mut bindings = self.session_to_policy.write();
        bindings.insert(scope.clone(), policy_id.clone());
        let evicted = inner.remove(scope).is_some();
        self.spawn_gate.forget(scope);
        debug!(
            scope = %scope,
            policy = %policy_id,
            evicted,
            "session rebound to new policy"
        );
        Ok(evicted)
    }

    /// Full eviction: drop both the live evaluator AND the
    /// session→policy binding. Used by `EndSession`, which
    /// signals the client is done with this session entirely. The
    /// graph state in the store is untouched, so a future query
    /// for the same scope re-spawns under the *tenant default*
    /// (the binding is gone).
    pub fn evict(&self, scope: &SessionScope) -> bool {
        // Lock order: `inner` first, matching `set_session_policy`
        // and `get_or_spawn`, so an `evict` racing with first-touch
        // authorization can't deadlock.
        let dropped = {
            let mut inner = self.inner.write();
            let mut bindings = self.session_to_policy.write();
            bindings.remove(scope);
            let evicted = inner.remove(scope).is_some();
            self.spawn_gate.forget(scope);
            evicted
        };
        // Locks released first: the ledger table orders against nothing, and
        // it stays that way. Pruned here as well as in the sweep so a
        // deployment with the sweep switched off does not accumulate one
        // ledger per session it has ever seen.
        self.kill_ledgers.prune_spent();
        dropped
    }

    /// Soft eviction: drop the live evaluator but keep the
    /// session→policy binding, so a resumed session re-bootstraps under the
    /// same variant it was bound to before and re-reads its configuration.
    /// Returns ``true`` iff a live evaluator was actually dropped.
    ///
    /// Callers: the idle-TTL sweep, the dead-subprocess path in `dispatch`,
    /// and — through `Engine::evict_session_evaluator` — the rollback of a
    /// configuration write, which needs any evaluator that seeded from the
    /// rolled-back row to be rebuilt.
    pub fn evict_evaluator_only(&self, scope: &SessionScope) -> bool {
        let dropped = self.inner.write().remove(scope).is_some();
        self.spawn_gate.forget(scope);
        self.kill_ledgers.prune_spent();
        dropped
    }

    /// Drop every live evaluator in `tenant` while leaving every binding in
    /// place, so each session rebuilds on its next call under the policy it is
    /// already bound to and re-reads its configuration.
    ///
    /// The tenant-wide counterpart of [`Self::evict_evaluator_only`], and used
    /// for the same reason: after a tenant-wide configuration write is rolled
    /// back, any evaluator that spawned while the rolled-back rows were in
    /// place is holding a seed that no longer matches the store, and a seed
    /// lasts the evaluator's lifetime.
    pub fn evict_tenant_evaluators_only(&self, tenant: &str) -> usize {
        let dropped = {
            let mut sessions = self.inner.write();
            let stale: Vec<SessionScope> = sessions
                .keys()
                .filter(|s| s.tenant() == tenant)
                .cloned()
                .collect();
            for scope in &stale {
                sessions.remove(scope);
            }
            self.spawn_gate.forget_tenant(tenant);
            stale.len()
        };
        self.kill_ledgers.prune_spent();
        dropped
    }

    /// Tenant-wide full eviction: drops every live evaluator AND
    /// session→policy binding under `tenant`. Sessions in other
    /// tenants are untouched. Used by `install_policy` when a new
    /// tenant default is installed — every existing session in the
    /// tenant must re-bootstrap under the new default on next
    /// traffic. Returns the count of evaluators dropped.
    pub fn evict_tenant(&self, tenant: &str) -> usize {
        let dropped = {
            let mut sessions = self.inner.write();
            let mut bindings = self.session_to_policy.write();
            let stale: Vec<SessionScope> = sessions
                .keys()
                .filter(|s| s.tenant() == tenant)
                .cloned()
                .collect();
            for scope in &stale {
                sessions.remove(scope);
                bindings.remove(scope);
            }
            // Also clear bindings for scopes that had no live evaluator
            // (idle/evicted sessions) but still carry a binding — they'd
            // resume under the *old* policy_id otherwise.
            bindings.retain(|s, _| s.tenant() != tenant);
            self.spawn_gate.forget_tenant(tenant);
            stale.len()
        };
        self.kill_ledgers.prune_spent();
        dropped
    }

    /// Live evaluator count.
    pub fn live_sessions(&self) -> usize {
        self.inner.read().len()
    }

    /// Backend name reported by health/status. Multi-policy
    /// deployments may have several backends co-installed; we
    /// surface the backend of the `"default"` tenant's default
    /// policy as the representative since that's what
    /// `swap_factory`-based callers (single-tenant, single-policy)
    /// see. Returns `"uninitialized"` if no policy has been
    /// installed yet.
    pub fn backend_name(&self) -> String {
        let default_id = match self.registry.default_for("default") {
            Some(id) => id,
            None => return "uninitialized".to_string(),
        };
        match self.registry.resolve("default", Some(&default_id)) {
            Ok((_, name, _)) => name,
            Err(_) => "uninitialized".to_string(),
        }
    }

    pub fn has_factory(&self) -> bool {
        self.registry.default_for("default").is_some()
    }

    /// Look up the policy currently bound to `scope`, if any.
    /// Used by the gRPC handler for `SetSessionPolicy` to confirm
    /// the rebind landed.
    pub fn current_policy(&self, scope: &SessionScope) -> Option<PolicyId> {
        self.session_to_policy.read().get(scope).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::types::{EvalActionResult, EvalAuthResponse};
    use parking_lot::Mutex as PMutex;
    use sasy_common::observability::{Event, Role};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const T: &str = "default";

    type ScopedEvaluators = Arc<PMutex<Vec<(SessionScope, Arc<LoggingEvaluator>)>>>;

    /// Mock that records each update batch and query's session.
    struct LoggingEvaluator {
        label: String,
        updates: PMutex<Vec<usize>>,
        queries: PMutex<Vec<Option<String>>>,
        update_count: AtomicUsize,
        /// Times `reset()` was called — one per full re-sync (reset + re-bootstrap).
        /// Lets a test assert a re-sync did (or did not) happen.
        reset_count: AtomicUsize,
        /// Facts received via `set_metadata` (config seeding).
        metadata: PMutex<Vec<PolicyMetadataFact>>,
    }

    impl LoggingEvaluator {
        fn new(label: &str) -> Arc<Self> {
            Arc::new(Self {
                label: label.to_string(),
                updates: PMutex::new(Vec::new()),
                queries: PMutex::new(Vec::new()),
                update_count: AtomicUsize::new(0),
                reset_count: AtomicUsize::new(0),
                metadata: PMutex::new(Vec::new()),
            })
        }
    }

    #[tonic::async_trait]
    impl Evaluator for LoggingEvaluator {
        async fn update(&self, updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            self.updates.lock().push(updates.len());
            self.update_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn query(&self, req: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            self.queries.lock().push(req.session_id.clone());
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            self.updates.lock().clear();
            self.queries.lock().clear();
            self.reset_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn set_metadata(&self, facts: Vec<PolicyMetadataFact>) -> Result<(), EvaluatorError> {
            *self.metadata.lock() = facts;
            Ok(())
        }

        fn backend_name(&self) -> &str {
            &self.label
        }
    }

    /// An evaluator whose query parks on an LLM oracle before answering, and
    /// reports that wait the way `EvaluatorProcess` does.
    ///
    /// The wait is a `sleep`, not work: it stands for the round trip to a
    /// provider, which is what the query budget stops for.
    struct OracleEvaluator {
        oracle_wait: Duration,
        clock: Arc<crate::evaluator::OracleClock>,
        queries: AtomicUsize,
    }

    impl OracleEvaluator {
        fn new(oracle_wait: Duration) -> Arc<Self> {
            Arc::new(Self {
                oracle_wait,
                clock: Arc::new(crate::evaluator::OracleClock::default()),
                queries: AtomicUsize::new(0),
            })
        }
    }

    #[tonic::async_trait]
    impl Evaluator for OracleEvaluator {
        async fn update(&self, _updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn query(&self, _req: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.clock.start();
            tokio::time::sleep(self.oracle_wait).await;
            self.clock.finish();
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }

        fn take_oracle_wait(&self) -> Duration {
            self.clock.take()
        }

        fn backend_name(&self) -> &str {
            "oracle"
        }
    }

    fn oracle_map(
        store: &Arc<GraphStore>,
        deadlines: EvaluationDeadlines,
        oracle_wait: Duration,
    ) -> (Arc<SessionEvaluatorMap>, Arc<OracleEvaluator>) {
        let ev = OracleEvaluator::new(oracle_wait);
        let handed = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&handed) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "oracle".to_string());
        (map, ev)
    }

    /// The budget clock stops for the LLM oracle.
    ///
    /// A cache miss on `@llm_check` suspends the evaluation while an external
    /// provider is asked, and that provider's own timeout is longer than the
    /// budget. The evaluator is not spinning and is not behind — it is waiting
    /// on somebody else's API — so the check is answered normally.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_oracle_round_trip_longer_than_the_budget_is_still_answered() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(150),
            stall_window: Duration::from_secs(30),
            oracle_bound: Duration::from_secs(5),
        };
        let (map, ev) = oracle_map(&store, deadlines, Duration::from_millis(900));
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let started = Instant::now();
        let answered = tokio::time::timeout(
            Duration::from_secs(10),
            map.dispatch(&scope, None, req_for(&scope), seq),
        )
        .await
        .expect("the check must not hang")
        .unwrap_or_else(|e| {
            panic!("a healthy evaluator waiting on the oracle should still answer: {e}")
        });
        assert!(answered.eval_response.results[0].authorized);
        assert!(
            started.elapsed() >= Duration::from_millis(900),
            "the answer came back before the oracle could have"
        );
        assert_eq!(
            ev.queries.load(Ordering::Relaxed),
            1,
            "the query was reissued instead of waited for"
        );
        assert_eq!(map.live_sessions(), 1, "the evaluator was not kept");
    }

    /// The stop has its own bound. Past it the check is denied — and the
    /// evaluator is kept, exactly as on a budget miss: an oracle that is slow
    /// is not an evaluator that is broken.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_oracle_round_trip_past_its_own_bound_is_denied_and_the_evaluator_kept() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(150),
            stall_window: Duration::from_secs(30),
            oracle_bound: Duration::from_millis(300),
        };
        // Far past the bound, and past anything this test waits for.
        let (map, _ev) = oracle_map(&store, deadlines, Duration::from_secs(20));
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let started = Instant::now();
        let answered = tokio::time::timeout(
            Duration::from_secs(10),
            map.dispatch(&scope, None, req_for(&scope), seq),
        )
        .await
        .expect("the check must not hang");
        let reason = match answered {
            Err(reason) => reason,
            Ok(_) => panic!("a query past the oracle bound must not be answered"),
        };
        assert!(
            reason.contains("during the query"),
            "the deny must be the task's own budget refusal, got: {reason}"
        );
        let elapsed = started.elapsed();
        assert!(
            elapsed >= deadlines.query_budget + deadlines.oracle_bound,
            "the check was denied after {elapsed:?}, before the oracle had its bound"
        );
        assert!(
            elapsed < deadlines.query_budget + deadlines.oracle_bound + DISPATCH_GRACE,
            "the check waited {elapsed:?} — past the budget, the oracle bound and the grace"
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "a slow oracle killed the evaluator; it is busy, not broken"
        );
    }

    /// A caller's oracle credit is one `oracle_bound` for the whole dispatch,
    /// not one per wait.
    ///
    /// `dispatch` makes two waits — the enqueue and the response — under one
    /// deadline, and hands the same credit to both. Both measure what has been
    /// granted since the SAME base against that SAME deadline, so whatever the
    /// second grants lands the deadline at `base + min(granted, bound)`:
    /// exactly where the first left it, whether or not the running total is
    /// carried between them. A caller cannot be held for two bounds.
    #[tokio::test]
    async fn the_oracle_credit_buys_one_bound_across_both_waits() {
        let counter = std::sync::atomic::AtomicU64::new(0);
        let bound = Duration::from_millis(300);
        let mut credit = OracleCredit {
            counter: &counter,
            base_us: 0,
            bound,
            used: Duration::ZERO,
        };
        // Far more credit than the bound, and granted before either wait
        // starts: the cap, not the grant, is what limits them.
        counter.store(5_000_000, Ordering::Relaxed);

        let base = Duration::from_millis(200);
        let started = tokio::time::Instant::now();
        let deadline = started + base;
        assert!(
            credit
                .wait_for(std::future::pending::<()>(), deadline)
                .await
                .is_err(),
            "a wait on a future that never completes must end at its deadline"
        );
        assert!(
            credit
                .wait_for(std::future::pending::<()>(), deadline)
                .await
                .is_err(),
            "the second wait must end too"
        );
        let elapsed = started.elapsed();
        assert!(
            elapsed >= base + bound,
            "the waits gave up after {elapsed:?}, before the {bound:?} of credit was spent"
        );
        assert!(
            elapsed < base + bound + Duration::from_millis(150),
            "the two waits took {elapsed:?}: together they spent more than the {bound:?} \
             bound, so a caller can be held for two of them"
        );
        assert_eq!(
            credit.used, bound,
            "the credit spent across both waits is the bound, once"
        );
    }

    /// A refusal says how much of the caller's wait was the oracle's.
    ///
    /// Without it the message names a budget and a grace that add up to less
    /// than the caller actually waited, and the extra time has no explanation
    /// anywhere the caller can see.
    ///
    /// The evaluator here wedges in its BOOTSTRAP, so the task never reaches
    /// the queue and the refusal is `dispatch`'s own rather than the task's.
    /// The credit is granted the way the query arm grants it — by bumping the
    /// counter the two share — because a task that is still bootstrapping has
    /// no query to grant it from.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_refusal_names_the_oracle_credit_the_caller_spent() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(100),
            // Far out: the bootstrap is bounded by the stall window, and
            // nothing here is about killing the evaluator.
            stall_window: Duration::from_secs(30),
            oracle_bound: Duration::from_millis(400),
        };
        let (map, ev) = paced_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        // The bootstrap IPC never comes back, so the task never dequeues.
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);

        let started = Instant::now();
        let call = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        // Grant the caller its credit while it waits, as the query arm would —
        // but only once the caller has read its base off the counter. A grant
        // that lands before that read is folded into the base and counts for
        // nothing, so waiting for the read is what makes the hand-off an
        // ordering rather than a race: after it, the grant is this caller's by
        // construction, whenever the loop gets to run.
        let waited = Instant::now();
        loop {
            let granted = map.inner.read().get(&scope).is_some_and(|se| {
                if se.credit_base_reads.load(Ordering::Acquire) == 0 {
                    return false;
                }
                se.oracle_credit_us.fetch_add(400_000, Ordering::Relaxed);
                true
            });
            assert!(
                granted || waited.elapsed() < Duration::from_secs(5),
                "the caller never reached the wait its credit is spent on"
            );
            if granted {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        let answered = tokio::time::timeout(Duration::from_secs(10), call)
            .await
            .expect("the check must not hang")
            .expect("the caller's task must not panic");
        let reason = match answered {
            Err(reason) => reason,
            Ok(_) => panic!("a bootstrap that never returns cannot answer a check"),
        };
        let elapsed = started.elapsed();
        assert!(
            reason.contains("LLM oracle"),
            "the refusal must say how much of the wait was the oracle's, got: {reason}"
        );
        assert!(
            reason.contains("400ms"),
            "the refusal must name the credit it spent, got: {reason}"
        );
        assert!(
            elapsed >= deadlines.query_budget + DISPATCH_GRACE + deadlines.oracle_bound,
            "the caller was refused after {elapsed:?}, before it had spent the credit the \
             message claims"
        );
    }

    /// An evaluator that parks on the LLM oracle and never comes back, holding
    /// the child while it does.
    ///
    /// The shim answers one frame at a time, so an evaluator wedged inside a
    /// query answers nothing else either: an update sent to it queues behind
    /// the reply that is not coming. Modelling that is what makes the mark the
    /// stall window is measured from stand — an update that returned here
    /// would be evidence the child had spoken, and the catch-up probe would
    /// clear it.
    struct WedgedOracleEvaluator {
        clock: Arc<crate::evaluator::OracleClock>,
        next_id: AtomicUsize,
        outstanding: AtomicUsize,
        kills: AtomicUsize,
    }

    #[tonic::async_trait]
    impl Evaluator for WedgedOracleEvaluator {
        async fn update(&self, _updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            while self.outstanding.load(Ordering::Relaxed) != 0 {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            Ok(())
        }

        async fn query(&self, _req: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
            self.outstanding.store(id, Ordering::Relaxed);
            // Handed to the provider and never taken back: every
            // `take_oracle_wait` from here on reports more wait, which is
            // credit that must not extend the query's deadline without limit.
            self.clock.start();
            std::future::pending::<()>().await;
            unreachable!("the wedged oracle never answers")
        }

        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn kill(&self) {
            self.kills.fetch_add(1, Ordering::Relaxed);
            self.outstanding.store(0, Ordering::Relaxed);
        }

        fn take_oracle_wait(&self) -> Duration {
            self.clock.take()
        }

        fn outstanding_request(&self) -> Option<u64> {
            match self.outstanding.load(Ordering::Relaxed) {
                0 => None,
                id => Some(id as u64),
            }
        }

        fn backend_name(&self) -> &str {
            "wedged-oracle"
        }
    }

    fn wedged_oracle_map(
        store: &Arc<GraphStore>,
        deadlines: EvaluationDeadlines,
    ) -> (Arc<SessionEvaluatorMap>, Arc<WedgedOracleEvaluator>) {
        let ev = Arc::new(WedgedOracleEvaluator {
            clock: Arc::new(crate::evaluator::OracleClock::default()),
            next_id: AtomicUsize::new(0),
            outstanding: AtomicUsize::new(0),
            kills: AtomicUsize::new(0),
        });
        let handed = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&handed) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "wedged-oracle".to_string());
        (map, ev)
    }

    /// Oracle wait credit pauses the query budget, but never extends the stall
    /// window. An evaluator that remains silent must still be terminated within
    /// that window, even while waiting indefinitely for its provider.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_wedged_oracle_cannot_carry_the_kill_past_the_stall_window() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(200),
            stall_window: Duration::from_millis(400),
            // Far past the window, as the shipped default is: the provider's
            // own timeout has nothing to do with this evaluator's liveness.
            oracle_bound: Duration::from_secs(3),
        };
        let (map, ev) = wedged_oracle_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        // The evaluator goes silent inside this check's query and says nothing
        // ever again.
        let silent = Instant::now();
        let call = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        while ev.kills.load(Ordering::Relaxed) == 0 && silent.elapsed() < Duration::from_secs(8) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let elapsed = silent.elapsed();
        call.abort();
        assert!(
            ev.kills.load(Ordering::Relaxed) > 0,
            "the evaluator owed a reply for the whole run and was never killed"
        );
        // The window, plus room for the spawn, the bootstrap and one probe.
        let bound = deadlines.stall_window + Duration::from_millis(600);
        assert!(
            elapsed < bound,
            "the wedged evaluator was killed {elapsed:?} after it went silent — past the \
             {bound:?} a {:?} stall window allows: the oracle credit carried the wait past \
             the stall deadline",
            deadlines.stall_window
        );
    }

    /// A stall window below the query budget is refused where an operator
    /// sees it.
    ///
    /// `sasy serve` builds its deadlines through this constructor and fails to
    /// start on the error, so a stall window below the query budget never
    /// reaches a running server — where it would show up as a kill of an
    /// evaluator whose query had not yet overrun its own budget.
    #[test]
    fn a_stall_window_below_the_query_budget_is_refused() {
        let refused = EvaluationDeadlines::new(Duration::from_secs(10), Duration::from_secs(5))
            .expect_err("a stall window below the query budget must not be accepted");
        assert!(
            refused.contains('5') && refused.contains("10"),
            "the refusal must name both numbers so the operator can see which to change, \
             got: {refused}"
        );
        let equal = EvaluationDeadlines::new(Duration::from_secs(10), Duration::from_secs(10))
            .expect("equal bounds are the boundary case and are allowed");
        assert_eq!(equal.query_budget, equal.stall_window);
    }

    /// A query budget of zero is refused at startup.
    ///
    /// `0` reads as "no timeout" — it means that for `--proxy-port` in the
    /// same CLI — but here every evaluation would overrun a zero budget and
    /// every check would be denied. The server would start clean and answer
    /// deny to everything, so the misreading is caught where the operator can
    /// still see it.
    #[test]
    fn a_zero_query_budget_is_refused() {
        let refused = EvaluationDeadlines::new(Duration::ZERO, Duration::from_secs(60))
            .expect_err("a zero query budget must not be accepted");
        assert!(
            refused.contains("no timeout"),
            "the refusal must say what an operator writing 0 was probably after, got: {refused}"
        );
    }

    /// The invariant holds even for a pair that skipped the constructor.
    ///
    /// The fields are public, so a caller can write the forbidden pair
    /// directly. `spawn` repairs it: the stall window is raised to the query
    /// budget, so no query is treated as a dead process before it has overrun
    /// its own budget.
    #[tokio::test]
    async fn a_hand_built_pair_is_repaired_before_it_bounds_anything() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "hand-built");
        let se = SessionEvaluator::spawn(
            scope.clone(),
            LoggingEvaluator::new("e") as Arc<dyn Evaluator>,
            Arc::clone(&store),
            Duration::from_millis(1),
            4,
            vec![],
            EvaluationDeadlines {
                query_budget: Duration::from_secs(10),
                stall_window: Duration::from_secs(1),
                ..EvaluationDeadlines::default()
            },
            "p".to_string(),
            KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
        );
        assert_eq!(
            se.deadlines.stall_window,
            Duration::from_secs(10),
            "the stall window stayed below the query budget, so a query still inside its \
             budget can be read as a dead process"
        );
    }

    /// Both flags fall back to their environment variables, so a server
    /// started through `sasy serve` and one embedding this crate agree.
    ///
    /// The variables are set and read back with no await in between: they are
    /// process-wide, and every other test in this binary that reads them takes
    /// whatever is set at that instant. The values are chosen so that a test
    /// unlucky enough to read them gets a working pair rather than a broken
    /// one.
    #[test]
    fn the_deadlines_fall_back_to_their_environment_variables() {
        std::env::set_var("SASY_QUERY_TIMEOUT_SECS", "7");
        std::env::set_var("SASY_EVALUATOR_STALL_SECS", "77");
        let from_env = EvaluationDeadlines::from_env();
        std::env::remove_var("SASY_QUERY_TIMEOUT_SECS");
        std::env::remove_var("SASY_EVALUATOR_STALL_SECS");
        assert_eq!(from_env.query_budget, Duration::from_secs(7));
        assert_eq!(from_env.stall_window, Duration::from_secs(77));

        // Equal bounds are the boundary case and are LEGAL — `new` accepts
        // them — so this pair goes through untouched. It is here to keep the
        // boundary pinned where the environment is what supplies it.
        std::env::set_var("SASY_QUERY_TIMEOUT_SECS", "9");
        std::env::set_var("SASY_EVALUATOR_STALL_SECS", "9");
        let boundary = EvaluationDeadlines::from_env();
        std::env::remove_var("SASY_QUERY_TIMEOUT_SECS");
        std::env::remove_var("SASY_EVALUATOR_STALL_SECS");
        assert_eq!(boundary.query_budget, Duration::from_secs(9));
        assert_eq!(boundary.stall_window, Duration::from_secs(9));

        // A stall window BELOW the budget is the impossible pair, and a
        // library default cannot refuse to start: it is repaired rather than
        // rejected, by raising the stall window to the budget — which keeps
        // the invariant without shortening either bound the operator asked
        // for. Any test unlucky enough to read the variables while they are
        // set gets the repaired pair, which is a working one.
        std::env::set_var("SASY_QUERY_TIMEOUT_SECS", "9");
        std::env::set_var("SASY_EVALUATOR_STALL_SECS", "4");
        let repaired = EvaluationDeadlines::from_env();
        std::env::remove_var("SASY_QUERY_TIMEOUT_SECS");
        std::env::remove_var("SASY_EVALUATOR_STALL_SECS");
        assert_eq!(repaired.query_budget, Duration::from_secs(9));
        assert_eq!(
            repaired.stall_window,
            Duration::from_secs(9),
            "the 4-second stall window was kept: a query still inside its 9-second budget \
             would be killed as a dead process"
        );

        // Unset: the shipped defaults.
        let defaults = EvaluationDeadlines::from_env();
        assert_eq!(
            defaults.query_budget,
            Duration::from_secs(DEFAULT_QUERY_TIMEOUT_SECS)
        );
        assert_eq!(
            defaults.stall_window,
            Duration::from_secs(DEFAULT_EVALUATOR_STALL_SECS)
        );
    }

    fn req_for(scope: &SessionScope) -> EvalAuthRequest {
        EvalAuthRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: None,
            roles: vec![],
            tenant_id: Some(scope.tenant().to_string()),
            session_id: if scope.is_global() {
                None
            } else {
                Some(scope.session().to_string())
            },
            principal: None,
            action_metadata: vec![],
        }
    }

    /// Test helper for building a stub [`Event`]. The session
    /// scope is now carried by the envelope (the `merge_*` call),
    /// so callers pass it directly to the store and `ev_for` only
    /// owns the per-event fields.
    fn ev_for(id: &str) -> Event {
        Event {
            text: Some("hi".into()),
            agent: None,
            role: Some(Role::User as i32),
            id: Some(id.to_string()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: None,
            metadata: None,
        }
    }

    /// Evaluator whose `update` fails non-fatally (an oversized IPC frame, a
    /// decode error) for the first `fail_updates` batches, then succeeds. The
    /// process stays ALIVE — that is the whole point: `ProcessDied` already exits
    /// the task, while this error is swallowed, leaving the subprocess holding
    /// whatever `reset()` left behind (nothing).
    struct FlakyUpdateEvaluator {
        fail_updates: AtomicUsize,
        queries: AtomicUsize,
        resets: AtomicUsize,
    }

    #[tonic::async_trait]
    impl Evaluator for FlakyUpdateEvaluator {
        async fn update(&self, _updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            if self
                .fail_updates
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    (n > 0).then(|| n - 1)
                })
                .is_ok()
            {
                return Err(EvaluatorError::UpdateError("frame too large".into()));
            }
            Ok(())
        }
        async fn query(&self, _req: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            self.queries.fetch_add(1, Ordering::Relaxed);
            Ok(EvalAuthResponse { results: vec![] })
        }
        async fn reset(&self) -> Result<(), EvaluatorError> {
            self.resets.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        async fn set_metadata(&self, _f: Vec<PolicyMetadataFact>) -> Result<(), EvaluatorError> {
            Ok(())
        }
        fn backend_name(&self) -> &str {
            "flaky"
        }
    }

    /// A bootstrap that resets the evaluator and then FAILS to load the snapshot
    /// back in must not have its query answered: the graph is empty, so every
    /// reachability-dependent rule silently under-matches and a deny-by-absence is
    /// indistinguishable from a real deny at the call site. The check fails closed
    /// (an `Err` the caller can surface / retry) until a reload actually lands.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn query_fails_closed_while_the_graph_is_knowingly_incomplete() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S1");
        // Non-empty store, so bootstrap actually has updates to (fail to) apply.
        store
            .merge_events(&scope, None, vec![ev_for("a"), ev_for("b")])
            .unwrap();

        let ev = Arc::new(FlakyUpdateEvaluator {
            // Enough to fail the spawn bootstrap AND the retry reload inside the
            // query arm, so the first dispatch has no way to recover.
            fail_updates: AtomicUsize::new(2),
            queries: AtomicUsize::new(0),
            resets: AtomicUsize::new(0),
        });
        let ev_for_factory = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&ev_for_factory) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        let first = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert!(
            first.is_err(),
            "a query against a knowingly-incomplete graph must fail closed"
        );
        assert_eq!(
            ev.queries.load(Ordering::Relaxed),
            0,
            "the hollow graph must never be QUERIED — that is the fail-open this guards"
        );

        // Updates now succeed: the retry reload lands and service resumes on the
        // same evaluator (the task must not have exited).
        let second = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert!(second.is_ok(), "must recover once the reload succeeds");
        assert_eq!(ev.queries.load(Ordering::Relaxed), 1);
        assert_eq!(map.live_sessions(), 1, "the healthy task must stay alive");
        // No `just_resynced` assertion here: this test dispatches a non-global scope
        // with min_sequence 0, so the fence block never resyncs and any such claim
        // would pass whether the guard existed or not. See
        // `an_incomplete_graph_after_a_fence_resync_does_not_reload_twice`.
    }

    /// A LIVE update that fails to apply must be treated exactly like a
    /// bootstrap that failed to apply.
    ///
    /// The batch is consumed by the flush and nothing resends it, and the
    /// sequence marker that would force a resync is consumed when an update is
    /// queued rather than when it applies — so the fence is already satisfied
    /// and, left alone, the evaluator keeps answering from an EDB silently
    /// missing those records. A dropped dependency edge makes a provenance
    /// rule under-match, which authorizes what should be denied. The same
    /// record fails CLOSED on re-bootstrap; the two paths have to agree.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_failed_live_update_fails_closed_like_a_failed_bootstrap() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S1");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();

        // Bootstrap succeeds: no failures budgeted yet.
        let ev = Arc::new(FlakyUpdateEvaluator {
            fail_updates: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            resets: AtomicUsize::new(0),
        });
        let ev_for_factory = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&ev_for_factory) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        let healthy = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert!(healthy.is_ok(), "precondition: a clean bootstrap answers");
        let answered_before = ev.queries.load(Ordering::Relaxed);

        // From here every apply fails — including the retry reload the query
        // arm attempts — so the evaluator cannot recover on its own.
        ev.fail_updates.store(1_000, Ordering::Relaxed);
        store.merge_events(&scope, None, vec![ev_for("b")]).unwrap();

        // Give the task time to drain the broadcast and fail the flush.
        for _ in 0..100 {
            if ev.resets.load(Ordering::Relaxed) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
            if map
                .dispatch(&scope, None, req_for(&scope), 0)
                .await
                .is_err()
            {
                break;
            }
        }

        let after = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert!(
            after.is_err(),
            "a query after an update that never applied must fail closed"
        );
        assert_eq!(
            ev.queries.load(Ordering::Relaxed),
            answered_before,
            "the incomplete graph must never be queried"
        );
    }

    /// Read the evaluator and its policy ID atomically under the map lock,
    /// including when another caller wins the spawn race. Hold one factory
    /// while a second installs the evaluator, then stall binding resolution to
    /// verify that a concurrent rebind cannot relabel the returned evaluator.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_spawn_race_loser_resolves_its_policy_id_under_the_map_lock() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "contested");

        // The FIRST factory call blocks; later ones do not.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(PMutex::new(release_rx));
        let calls = Arc::new(AtomicUsize::new(0));
        let factory: EvaluatorFactory = Arc::new(move || {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().recv();
            }
            Ok(Arc::new(FlakyUpdateEvaluator {
                fail_updates: AtomicUsize::new(0),
                queries: AtomicUsize::new(0),
                resets: AtomicUsize::new(0),
            }) as Arc<dyn Evaluator>)
        });
        let map = Arc::new(SessionEvaluatorMap::new(
            Arc::clone(&store),
            Duration::from_secs(60),
            0,
        ));
        map.swap_factory(factory, "mock".to_string());

        // The loser: into the factory, and stuck there.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let loser = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            std::thread::spawn(move || {
                let out = map.get_or_spawn(&scope, None);
                let _ = done_tx.send(out.is_ok());
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .unwrap()
            .expect("the first caller must reach the factory");

        // The winner's entry, installed directly while the loser is
        // held. It goes into the map by hand rather than through a
        // second `get_or_spawn` because the per-scope spawn gate now
        // makes that unstageable: the gate entry the loser holds
        // cannot be dropped underneath it (an eviction arriving now is
        // refused for exactly that reason), so a second caller would
        // wait rather than install. The branch under test stays
        // defensive — nothing else may install for a scope whose spawn
        // is in flight — and this is the state it must survive.
        {
            let winner = Arc::new(SessionEvaluator::spawn(
                scope.clone(),
                Arc::new(FlakyUpdateEvaluator {
                    fail_updates: AtomicUsize::new(0),
                    queries: AtomicUsize::new(0),
                    resets: AtomicUsize::new(0),
                }) as Arc<dyn Evaluator>,
                Arc::clone(&store),
                Duration::from_secs(60),
                1,
                Vec::new(),
                EvaluationDeadlines::default(),
                "p".to_string(),
                KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
            ));
            map.inner.write().insert(scope.clone(), winner);
        }

        // Stall every resolve, then let the loser out. It will find the map
        // already filled and take the winner's evaluator.
        let bindings_held = map.session_to_policy.write();
        let _ = release_tx.send(());
        // The loser needs a moment to get from the factory to the resolve;
        // this makes "it has not got there yet" implausible rather than
        // merely unlikely.
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            done_rx.try_recv().is_err(),
            "the loser must be stalled resolving the binding"
        );

        // From a plain OS thread, because a blocked probe must fail on a
        // timeout rather than park a runtime worker forever.
        let probe = {
            let map = Arc::clone(&map);
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _held = map.inner.write();
                let _ = tx.send(());
            });
            rx
        };
        assert!(
            probe.recv_timeout(Duration::from_millis(750)).is_err(),
            "the map lock must still be held while the binding is read — \
             releasing it first is what lets a rebind slip between the \
             evaluator and the id describing it"
        );

        drop(bindings_held);
        assert!(
            done_rx.recv_timeout(Duration::from_secs(3)).unwrap(),
            "the loser must finish once the bindings lock is free"
        );
        loser.join().unwrap();
        probe.recv_timeout(Duration::from_secs(3)).unwrap();
    }

    /// A cold session's spawn must not freeze dispatch for everyone else.
    ///
    /// The spawn's expensive half is a Soufflé compile plus a subprocess fork.
    /// Run under the evaluator map's write lock, it blocked the cache-hit fast
    /// path — which takes a read lock — for every session in every tenant, for
    /// the whole compile. Here the second factory call blocks until released,
    /// standing in for that compile, while an already-live unrelated session
    /// is dispatched: it has to answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cold_spawn_does_not_block_dispatch_for_other_sessions() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let live = SessionScope::new(T, "already-live");
        let cold = SessionScope::new(T, "cold");

        // Released by the test; the second factory call waits on it.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        // Signals that the blocking call has actually entered the factory.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(PMutex::new(release_rx));
        let calls = Arc::new(AtomicUsize::new(0));

        let factory: EvaluatorFactory = Arc::new(move || {
            if calls.fetch_add(1, Ordering::SeqCst) == 1 {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().recv();
            }
            Ok(Arc::new(FlakyUpdateEvaluator {
                fail_updates: AtomicUsize::new(0),
                queries: AtomicUsize::new(0),
                resets: AtomicUsize::new(0),
            }) as Arc<dyn Evaluator>)
        });
        let map = Arc::new(SessionEvaluatorMap::new(
            Arc::clone(&store),
            Duration::from_secs(60),
            0,
        ));
        map.swap_factory(factory, "mock".to_string());

        // First call: the live session, spawned without blocking.
        assert!(map.dispatch(&live, None, req_for(&live), 0).await.is_ok());

        // Second call: the cold session, stuck in the factory.
        let map_for_cold = Arc::clone(&map);
        let cold_for_task = cold.clone();
        let cold_task = tokio::spawn(async move {
            map_for_cold
                .dispatch(&cold_for_task, None, req_for(&cold_for_task), 0)
                .await
        });
        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .unwrap()
            .expect("the cold spawn must reach the factory");

        // Probe the map lock itself, from a plain OS thread. Awaiting a
        // dispatch here instead would regress into a HANG rather than a
        // failure: `get_or_spawn` is synchronous, so a task blocked on the map
        // lock parks its runtime worker, and with the workers gone the timeout
        // future has nothing left to poll it. A thread plus `recv_timeout`
        // fails cleanly and points at the actual invariant — that the map is
        // readable while a spawn is in its factory.
        let map_probe = Arc::clone(&map);
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = probe_tx.send(map_probe.live_sessions());
        });
        let live_count = probe_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("the evaluator map must stay readable while another session is in its factory");
        assert_eq!(
            live_count, 1,
            "the already-live session must still be there"
        );

        // And it must still answer.
        let served = tokio::time::timeout(
            Duration::from_secs(3),
            map.dispatch(&live, None, req_for(&live), 0),
        )
        .await
        .expect("a live session must not wait on another session's spawn");
        assert!(served.is_ok(), "live dispatch failed: {:?}", served.err());

        let _ = release_tx.send(());
        assert!(
            cold_task.await.unwrap().is_ok(),
            "the cold spawn must finish"
        );
    }

    /// When the fence/global block has ALREADY reloaded, the incomplete-graph guard
    /// must reuse that attempt, not run a second full reload in the same query. On
    /// the refmon's per-request global path a redundant reload is O(tenant graph).
    ///
    /// Needs a fence that actually fires: a global scope with `min_sequence` past
    /// `snapshot_seq` (which stays 0 precisely because no bootstrap lands), and a
    /// graph still hollow afterwards — otherwise the guard is never reached and the
    /// assertion is vacuous.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_incomplete_graph_after_a_fence_resync_does_not_reload_twice() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let global = SessionScope::new(T, "");
        // Non-empty global shard, so the bootstrap has updates to fail on.
        store
            .merge_events(&global, None, vec![ev_for("g1")])
            .unwrap();

        let ev = Arc::new(FlakyUpdateEvaluator {
            fail_updates: AtomicUsize::new(usize::MAX),
            queries: AtomicUsize::new(0),
            resets: AtomicUsize::new(0),
        });
        let ev_f = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&ev_f) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        assert!(
            map.dispatch(&global, None, req_for(&global), 1)
                .await
                .is_err(),
            "a hollow graph must fail closed"
        );
        assert_eq!(
            ev.resets.load(Ordering::Relaxed),
            1,
            "exactly one reload: the guard must not repeat the one the fence block just did"
        );
    }

    /// A permanently-failing reload keeps failing CLOSED — indefinitely, and without
    /// killing the task. Pins the decision not to escalate to `mark_process_died()`:
    /// the respawn would reset any per-task failure counter, so escalation cannot
    /// bound the deterministic case and only adds spawns. Should this become
    /// bounded for real, it belongs in `SessionEvaluatorMap`, where the state
    /// outlives the respawn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_permanently_failing_reload_keeps_failing_closed_without_respawn_churn() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S1");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();

        let spawns = Arc::new(AtomicUsize::new(0));
        let spawns_f = Arc::clone(&spawns);
        let factory: EvaluatorFactory = Arc::new(move || {
            spawns_f.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(FlakyUpdateEvaluator {
                // Never succeeds: the payload itself is what the shim rejects.
                fail_updates: AtomicUsize::new(usize::MAX),
                queries: AtomicUsize::new(0),
                resets: AtomicUsize::new(0),
            }) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        for _ in 0..5 {
            assert!(
                map.dispatch(&scope, None, req_for(&scope), 0)
                    .await
                    .is_err(),
                "every check must fail closed while the graph is hollow"
            );
        }
        assert_eq!(
            spawns.load(Ordering::Relaxed),
            1,
            "a hollow graph must not churn subprocesses — the respawn resets any \
             per-task bound, so it buys nothing"
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "and the task must stay alive to keep serving"
        );
    }

    /// Each new session gets a fresh evaluator from the factory.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn map_spawns_one_evaluator_per_session() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        map.swap_factory(factory, "mock".to_string());

        let s1 = SessionScope::new(T, "S1");
        let s2 = SessionScope::new(T, "S2");
        for _ in 0..3 {
            map.dispatch(&s1, None, req_for(&s1), 0).await.unwrap();
        }
        for _ in 0..2 {
            map.dispatch(&s2, None, req_for(&s2), 0).await.unwrap();
        }
        // Ping S1 again; should reuse, not spawn.
        map.dispatch(&s1, None, req_for(&s1), 0).await.unwrap();

        assert_eq!(counter.load(Ordering::Relaxed), 2, "expected 2 spawns");
        assert_eq!(map.live_sessions(), 2);
    }

    /// Persist-only config: facts stored with the policy (keyed by
    /// content hash) are read at spawn and seeded into the evaluator
    /// before the first query — proving the store → get_or_spawn →
    /// spawn → run → set_metadata path with no subprocess. Nothing is
    /// cached in the binding map; the store is the sole source.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn policy_metadata_seeded_at_spawn() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let captured: Arc<PMutex<Vec<Arc<LoggingEvaluator>>>> = Arc::new(PMutex::new(Vec::new()));
        let captured_for_factory = Arc::clone(&captured);
        let factory: EvaluatorFactory = Arc::new(move || {
            let e = LoggingEvaluator::new("e");
            captured_for_factory.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        let policy_id = map.swap_factory(factory, "mock".to_string());

        // Tenant-wide config travels with the policy, keyed by (tenant, hash).
        store
            .put_policy_metadata(
                T,
                policy_id.as_str(),
                &[sasy_graph::PolicyMetadataFact {
                    rel: "TrustedDomain".into(),
                    a: "registry.npmjs.org".into(),
                    b: String::new(),
                }],
            )
            .unwrap();

        let scope = SessionScope::new(T, "S1");
        // dispatch awaits the query, which the task only services
        // after set_metadata + bootstrap — so by return, seeding ran.
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();

        let evals = captured.lock();
        assert_eq!(evals.len(), 1, "exactly one spawn");
        let seeded = evals[0].metadata.lock();
        assert_eq!(seeded.len(), 1, "one config fact seeded before first query");
        assert_eq!(seeded[0].rel, "TrustedDomain");
        assert_eq!(seeded[0].a, "registry.npmjs.org");
    }

    #[tokio::test]
    async fn invalid_session_metadata_is_not_persisted() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        let scope = SessionScope::new(T, "S1");

        map.update_session_metadata(
            &scope,
            vec![PolicyMetadataFact {
                rel: "TrustedDomain".into(),
                a: "registry.npmjs.org".into(),
                b: String::new(),
            }],
        )
        .await
        .unwrap();

        let error = map
            .update_session_metadata(
                &scope,
                vec![PolicyMetadataFact {
                    rel: "TrustedDomain".into(),
                    a: "invalid\0domain".into(),
                    b: String::new(),
                }],
            )
            .await
            .unwrap_err();
        assert!(error.contains("embedded NUL"));

        let persisted = store.get_session_metadata(&scope).unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].rel, "TrustedDomain");
        assert_eq!(persisted[0].a, "registry.npmjs.org");
        assert_eq!(persisted[0].b, "");
    }

    /// A policy with no persisted config spawns with an empty EDB —
    /// the common case must not error or seed stale facts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_metadata_spawns_with_empty_config() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let captured: Arc<PMutex<Vec<Arc<LoggingEvaluator>>>> = Arc::new(PMutex::new(Vec::new()));
        let captured_for_factory = Arc::clone(&captured);
        let factory: EvaluatorFactory = Arc::new(move || {
            let e = LoggingEvaluator::new("e");
            captured_for_factory.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        let scope = SessionScope::new(T, "S1");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();

        let evals = captured.lock();
        assert_eq!(evals.len(), 1);
        assert!(
            evals[0].metadata.lock().is_empty(),
            "no facts seeded when none persisted"
        );
    }

    /// Cross-session events do not produce updates on a given
    /// session's evaluator.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_session_events_filter_out() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let evaluators: PMutex<Vec<Arc<LoggingEvaluator>>> = PMutex::new(Vec::new());
        let evaluators_for_factory: Arc<PMutex<Vec<Arc<LoggingEvaluator>>>> = Arc::new(evaluators);
        let evals_clone = Arc::clone(&evaluators_for_factory);
        let factory: EvaluatorFactory = Arc::new(move || {
            let label = format!("e{}", evals_clone.lock().len());
            let e = LoggingEvaluator::new(&label);
            evals_clone.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        let scope_a = SessionScope::new(T, "A");
        let scope_b = SessionScope::new(T, "B");

        // First touch each session so its evaluator spawns and
        // bootstraps (against an empty store at this point).
        map.dispatch(&scope_a, None, req_for(&scope_a), 0)
            .await
            .unwrap();
        map.dispatch(&scope_b, None, req_for(&scope_b), 0)
            .await
            .unwrap();

        // Now generate cross-session traffic.
        store
            .merge_events(&scope_a, None, vec![ev_for("a1"), ev_for("a2")])
            .unwrap();
        store
            .merge_events(&scope_b, None, vec![ev_for("b1")])
            .unwrap();
        // Per-session seq fence: each evaluator advances on its own
        // SessionSequence marker, so we must use the session-local
        // counter, not the global one.
        let seq_a = store.session_sequence(&scope_a);
        let seq_b = store.session_sequence(&scope_b);

        // Bump each evaluator past the seq fence so it has drained
        // and flushed.
        map.dispatch(&scope_a, None, req_for(&scope_a), seq_a)
            .await
            .unwrap();
        map.dispatch(&scope_b, None, req_for(&scope_b), seq_b)
            .await
            .unwrap();

        let evals = evaluators_for_factory.lock();
        assert_eq!(evals.len(), 2);
        let a = &evals[0];
        let b = &evals[1];

        // Sum of update batch sizes per evaluator. A's evaluator
        // should have seen ~2 events (a1, a2), never b1; B's
        // evaluator should have seen ~1 event (b1), never a1/a2.
        let a_total: usize = a.updates.lock().iter().sum();
        let b_total: usize = b.updates.lock().iter().sum();
        assert!(
            a_total >= 2,
            "session A evaluator should see its 2 events, got {}",
            a_total
        );
        assert!(
            b_total >= 1,
            "session B evaluator should see its 1 event, got {}",
            b_total
        );
        // Neither evaluator should have ingested the *other*
        // session's events. Check via total update count: at most
        // bootstrap + own events, never cross-session.
        assert!(
            a_total <= 4,
            "session A evaluator sees too many updates ({}) — cross-session leakage?",
            a_total
        );
        assert!(
            b_total <= 3,
            "session B evaluator sees too many updates ({}) — cross-session leakage?",
            b_total
        );
    }

    /// RegisterEvents-style live writes pre-warm a per-session
    /// evaluator before any auth query arrives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prewarm_on_broadcast_spawns_evaluator() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        // Live event for session "P" — should pre-warm an
        // evaluator without any dispatch yet.
        store
            .merge_events(&SessionScope::new(T, "P"), None, vec![ev_for("p1")])
            .unwrap();

        // Give the prewarm listener a chance to react.
        for _ in 0..50 {
            if map.live_sessions() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(
            map.live_sessions(),
            1,
            "expected prewarm to spawn an evaluator for session P"
        );
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    /// Persisted (disk-loaded) sessions do *not* pre-warm; they
    /// only spawn on subsequent live traffic. Modeled here by
    /// inserting events without going through the broadcast
    /// path — direct `merge_events` does broadcast, so we test
    /// the contrapositive: an idle map with no broadcasts has
    /// zero live sessions.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_map_has_no_live_sessions() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        // No traffic, no dispatches.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(map.live_sessions(), 0);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Idle-TTL sweep evicts evaluators with no activity for the
    /// configured duration; subsequent traffic re-spawns and
    /// re-bootstraps via the prewarm path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_ttl_sweep_evicts_inactive_sessions() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_millis(50), // idle TTL
            Duration::from_millis(20), // sweep interval
            None,                      // unbounded
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let scope_s = SessionScope::new(T, "S");

        // Spawn a session via dispatch. Counter should now be 1.
        map.dispatch(&scope_s, None, req_for(&scope_s), 0)
            .await
            .unwrap();
        assert_eq!(map.live_sessions(), 1);

        // Wait long enough for at least one sweep AFTER the TTL
        // has elapsed. The first tick is skipped, so we need at
        // least one full sweep_interval after the TTL has expired.
        for _ in 0..30 {
            if map.live_sessions() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(map.live_sessions(), 0, "expected idle session to be swept",);

        // A subsequent dispatch transparently re-spawns.
        map.dispatch(&scope_s, None, req_for(&scope_s), 0)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert_eq!(map.live_sessions(), 1);
    }

    /// Sessions receiving live broadcasts (prewarmed but never
    /// dispatched) must not be evicted while events are arriving.
    /// The task itself bumps last_active in process_broadcast.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn broadcast_activity_keeps_session_alive() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_millis(80), // idle TTL
            Duration::from_millis(20), // sweep
            None,                      // unbounded
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let scope_p = SessionScope::new(T, "P");

        // Pre-warm via a broadcast.
        store
            .merge_events(&scope_p, None, vec![ev_for("a1")])
            .unwrap();
        for _ in 0..30 {
            if map.live_sessions() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(map.live_sessions(), 1);

        // Keep firing events at sub-TTL cadence; the session must
        // not be swept because its task processes broadcasts and
        // bumps last_active.
        for i in 0..4 {
            tokio::time::sleep(Duration::from_millis(40)).await;
            store
                .merge_events(&scope_p, None, vec![ev_for(&format!("a{}", i + 2))])
                .unwrap();
        }
        assert_eq!(
            map.live_sessions(),
            1,
            "session receiving live broadcasts should not be evicted",
        );
    }

    /// Eviction drops the evaluator only; the next query
    /// re-spawns and the graph state survives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn evict_then_resume_re_bootstraps_from_store() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());

        let scope_s = SessionScope::new(T, "S");

        // Populate session S with a couple of events first so
        // resume has something to bootstrap.
        store
            .merge_events(&scope_s, None, vec![ev_for("s1"), ev_for("s2")])
            .unwrap();
        // That live write also reaches the prewarm listener, which spawns this
        // session's evaluator on its own. Let it land before dispatching:
        // raced, both paths spawn and the counts below are off by one for a
        // reason that has nothing to do with eviction.
        for _ in 0..50 {
            if map.live_sessions() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let seq = store.session_sequence(&scope_s);
        // Let the prewarm listener finish reacting to those events first. It
        // spawns for the same scope this dispatch is about to, and two callers
        // racing one cold scope deliberately both build an evaluator (the
        // loser's is dropped) — which is a second factory call, and says
        // nothing about the respawn this test is pinning.
        let settling = Instant::now();
        while map.live_sessions() == 0 && settling.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        map.dispatch(&scope_s, None, req_for(&scope_s), seq)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        assert!(map.evict(&scope_s));
        assert_eq!(map.live_sessions(), 0);

        // Resume — should spawn a NEW evaluator and bootstrap it
        // with the graph state that is still in the store.
        map.dispatch(&scope_s, None, req_for(&scope_s), seq)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert_eq!(map.live_sessions(), 1);
    }

    /// Evaluator whose queries hang until the test releases them, and which
    /// keeps the "still owes a reply" mark the way `EvaluatorProcess` does:
    /// set when a query starts, cleared only when one actually answers. A
    /// caller that abandons its call therefore leaves the mark standing, which
    /// is what the busy handling reads.
    struct BlockingEvaluator {
        label: String,
        released: Arc<std::sync::atomic::AtomicBool>,
        next_id: AtomicUsize,
        outstanding: AtomicUsize,
        started: AtomicUsize,
        answered: AtomicUsize,
        kills: AtomicUsize,
        /// Once this many queries have STARTED, every later one is refused
        /// without the child hearing it — the shape of a request the
        /// transport rejects ahead of the child mutex, which returns an error
        /// while the evaluator is still silently working on the last thing it
        /// was told.
        refuse_after_starts: AtomicUsize,
    }

    impl BlockingEvaluator {
        fn new(label: &str) -> (Arc<Self>, Arc<std::sync::atomic::AtomicBool>) {
            let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let e = Arc::new(Self {
                label: label.into(),
                released: Arc::clone(&released),
                next_id: AtomicUsize::new(0),
                outstanding: AtomicUsize::new(0),
                started: AtomicUsize::new(0),
                answered: AtomicUsize::new(0),
                kills: AtomicUsize::new(0),
                refuse_after_starts: AtomicUsize::new(usize::MAX),
            });
            (e, released)
        }

        /// Answering from here on, as a respawned evaluator does.
        fn already_released(label: &str) -> Arc<Self> {
            let (e, released) = Self::new(label);
            released.store(true, Ordering::Relaxed);
            e
        }
    }

    #[tonic::async_trait]
    impl Evaluator for BlockingEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            // Serialized behind an unanswered query, as the real evaluator is:
            // `EvaluatorProcess` holds one child and one mutex, and the shim
            // answers one frame at a time, so nothing can be told to an
            // evaluator that is still computing. Modelling this is what makes
            // the catch-up probe meaningful — an update that returned while a
            // query was wedged would be no evidence of anything.
            while self.outstanding.load(Ordering::Relaxed) != 0
                && !self.released.load(Ordering::Relaxed)
            {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            Ok(())
        }

        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            if self.started.load(Ordering::Relaxed)
                >= self.refuse_after_starts.load(Ordering::Relaxed)
            {
                // Refused before the child is spoken to: nothing is written,
                // nothing is marked, and whatever it already owes it still
                // owes.
                return Err(EvaluatorError::IpcError(
                    "request refused before the child was spoken to".into(),
                ));
            }
            let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
            self.outstanding.store(id, Ordering::Relaxed);
            self.started.fetch_add(1, Ordering::Relaxed);
            while !self.released.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            self.outstanding.store(0, Ordering::Relaxed);
            self.answered.fetch_add(1, Ordering::Relaxed);
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn kill(&self) {
            self.kills.fetch_add(1, Ordering::Relaxed);
            // A killed process answers nothing further, and its pending call
            // is gone with it.
            self.outstanding.store(0, Ordering::Relaxed);
        }

        fn outstanding_request(&self) -> Option<u64> {
            match self.outstanding.load(Ordering::Relaxed) {
                0 => None,
                id => Some(id as u64),
            }
        }

        fn backend_name(&self) -> &str {
            &self.label
        }
    }

    /// Budgets small enough to spend inside a test, with a stall window far
    /// enough away that nothing is killed while the budget is being pinned.
    fn short_budget() -> EvaluationDeadlines {
        EvaluationDeadlines {
            query_budget: Duration::from_millis(150),
            stall_window: Duration::from_secs(30),
            ..EvaluationDeadlines::default()
        }
    }

    fn blocking_map(
        store: &Arc<GraphStore>,
        deadlines: EvaluationDeadlines,
    ) -> (
        Arc<SessionEvaluatorMap>,
        Arc<BlockingEvaluator>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let (ev, released) = BlockingEvaluator::new("blocking");
        let factory_ev = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&factory_ev) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "blocking".to_string());
        (map, ev, released)
    }

    /// The margin between the task's own budget refusal and `dispatch`'s
    /// backstop, which fires a whole [`DISPATCH_GRACE`] later.
    ///
    /// A test that only checked "denied, and it mentions a budget" would pass
    /// with the query budget deleted outright, because the backstop answers
    /// with a message that says "budget" too. Both budget tests therefore
    /// assert the task's own refusal: its wording, and an elapsed time that
    /// leaves the backstop no room to have been the one that answered.
    fn well_inside_the_dispatch_backstop(deadlines: EvaluationDeadlines) -> Duration {
        deadlines.query_budget + DISPATCH_GRACE / 2
    }

    /// A query the evaluator does not answer in time is denied, and the
    /// evaluator is kept: it is busy, not broken. When its late answer finally
    /// arrives the session goes on serving — no respawn, no lost graph.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_query_over_its_budget_is_denied_and_the_evaluator_kept() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = short_budget();
        let (map, ev, released) = blocking_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let started = Instant::now();
        let denied = match map.dispatch(&scope, None, req_for(&scope), seq).await {
            Err(reason) => reason,
            Ok(_) => panic!("a query that outran its budget must not be answered"),
        };
        let elapsed = started.elapsed();
        assert!(
            denied.contains("during the query"),
            "the deny must be the task's own, which names the phase the budget ran out in; \
             `dispatch`'s backstop says only that nothing answered, got: {denied}"
        );
        assert!(
            elapsed < well_inside_the_dispatch_backstop(deadlines),
            "the caller waited {elapsed:?}, long enough that `dispatch`'s backstop \
             ({:?} budget + {DISPATCH_GRACE:?} grace) could have been what answered",
            deadlines.query_budget,
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "the evaluator was dropped over one slow query"
        );

        // The late reply lands; the next query gets its own answer.
        released.store(true, Ordering::Relaxed);
        let answered = map
            .dispatch(&scope, None, req_for(&scope), seq)
            .await
            .unwrap_or_else(|e| panic!("a query after the evaluator caught up is denied: {e}"));
        assert!(answered.eval_response.results[0].authorized);
        assert_eq!(
            ev.started.load(Ordering::Relaxed),
            2,
            "the second query went to the same evaluator, not a respawned one"
        );
    }

    /// A query that arrives while the evaluator still owes a reply waits its
    /// own budget for that reply and is then denied — it never inherits the
    /// wait already spent, and it never hangs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_query_arriving_while_the_evaluator_is_busy_is_denied_after_its_own_budget() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = short_budget();
        let budget = deadlines.query_budget;
        let (map, _ev, _released) = blocking_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let first = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        // Behind the first one, which is already wedged.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let started = Instant::now();
        let second = map.dispatch(&scope, None, req_for(&scope), seq).await;

        assert!(
            first.await.unwrap().is_err(),
            "the wedged query must be denied"
        );
        let elapsed = started.elapsed();
        let reason = match second {
            Err(reason) => reason,
            Ok(_) => panic!("the query behind a busy evaluator must be denied, not answered"),
        };
        assert!(
            elapsed >= budget,
            "the second query gave up before its own budget was spent"
        );
        assert!(
            reason.contains("during the query"),
            "the deny must be the task's own, which names the phase the budget ran out in; \
             `dispatch`'s backstop says only that nothing answered, got: {reason}"
        );
        assert!(
            elapsed < well_inside_the_dispatch_backstop(deadlines),
            "the caller waited {elapsed:?}, long enough that `dispatch`'s backstop \
             ({budget:?} budget + {DISPATCH_GRACE:?} grace) could have been what answered",
        );
    }

    /// An evaluator that takes a settable amount of time over each IPC and
    /// serves them one at a time, as the real one does.
    ///
    /// `EvaluatorProcess` holds one child behind one mutex and the shim answers
    /// one frame at a time, so nothing can be said to an evaluator that is
    /// still computing — a slow query delays every update behind it. The
    /// `busy` mutex here is that constraint; without it a test could not tell
    /// an evaluator that caught up from one that never will.
    struct PacedEvaluator {
        label: String,
        /// When the child is next free. Written when an IPC starts and never
        /// retracted, because a subprocess does not stop computing when the
        /// caller gives up: an abandoned query keeps the child to itself until
        /// its work is done, and only then can anything else be told anything.
        /// A mutex guard cannot model that — dropping the abandoned future
        /// would release it, and every probe would find a wedged evaluator
        /// idle.
        busy_until: PMutex<Instant>,
        update_ms: std::sync::atomic::AtomicU64,
        query_ms: std::sync::atomic::AtomicU64,
        updates: AtomicUsize,
        queries: AtomicUsize,
        /// One per `full_resync` — the reload a session does when its graph
        /// can no longer be trusted.
        resets: AtomicUsize,
        kills: AtomicUsize,
    }

    impl PacedEvaluator {
        fn new(label: &str) -> Arc<Self> {
            Arc::new(Self {
                label: label.into(),
                busy_until: PMutex::new(Instant::now()),
                update_ms: std::sync::atomic::AtomicU64::new(0),
                query_ms: std::sync::atomic::AtomicU64::new(0),
                updates: AtomicUsize::new(0),
                queries: AtomicUsize::new(0),
                resets: AtomicUsize::new(0),
                kills: AtomicUsize::new(0),
            })
        }

        fn set_update_ms(&self, ms: u64) {
            self.update_ms.store(ms, Ordering::Relaxed);
        }

        fn set_query_ms(&self, ms: u64) {
            self.query_ms.store(ms, Ordering::Relaxed);
        }

        /// Long enough that no test's own deadlines can pass it.
        const FOREVER_MS: u64 = 3_600_000;

        /// Take the child for `ms`, waiting first for whatever it is doing.
        async fn occupy(&self, ms: u64) {
            loop {
                let free_at = *self.busy_until.lock();
                let now = Instant::now();
                if free_at <= now {
                    *self.busy_until.lock() = now + Duration::from_millis(ms);
                    break;
                }
                tokio::time::sleep(std::cmp::min(free_at - now, Duration::from_millis(2))).await;
            }
            tokio::time::sleep(Duration::from_millis(ms)).await;
        }
    }

    #[tonic::async_trait]
    impl Evaluator for PacedEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            self.updates.fetch_add(1, Ordering::Relaxed);
            self.occupy(self.update_ms.load(Ordering::Relaxed)).await;
            Ok(())
        }

        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.occupy(self.query_ms.load(Ordering::Relaxed)).await;
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            self.resets.fetch_add(1, Ordering::Relaxed);
            self.occupy(0).await;
            Ok(())
        }

        async fn kill(&self) {
            self.kills.fetch_add(1, Ordering::Relaxed);
            // A killed process is computing nothing.
            *self.busy_until.lock() = Instant::now();
        }

        fn backend_name(&self) -> &str {
            &self.label
        }
    }

    fn paced_map(
        store: &Arc<GraphStore>,
        deadlines: EvaluationDeadlines,
    ) -> (Arc<SessionEvaluatorMap>, Arc<PacedEvaluator>) {
        let ev = PacedEvaluator::new("paced");
        let factory_ev = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&factory_ev) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "paced".to_string());
        (map, ev)
    }

    /// A caller is bounded by its own budget wherever its job is stuck, not
    /// only inside the query. A session whose evaluator wedges on the very
    /// first IPC has its check sitting in the queue behind a bootstrap that is
    /// bounded by the STALL window — a minute with the shipped defaults, six
    /// times the budget and far past any typical agent-hook timeout. The caller answers
    /// itself instead.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_check_waiting_on_a_wedged_bootstrap_is_denied_at_its_own_budget() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(200),
            stall_window: Duration::from_secs(20),
            ..EvaluationDeadlines::default()
        };
        let (map, ev) = paced_map(&store, deadlines);
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let started = Instant::now();
        let answered = map.dispatch(&scope, None, req_for(&scope), seq).await;
        let waited = started.elapsed();

        assert!(
            answered.is_err(),
            "a check whose evaluator never bootstrapped must fail closed"
        );
        assert!(
            waited < deadlines.query_budget + DISPATCH_GRACE + Duration::from_secs(2),
            "the caller waited {waited:?} on a {:?} budget — it was bounded by the stall \
             window, not by the budget",
            deadlines.query_budget
        );
    }

    /// Every caller is bounded, not only the first four.
    ///
    /// The channel in front of the task holds a handful of jobs (four in the
    /// binary, one in this map) and the task takes from it only between jobs,
    /// so while the task is inside a
    /// stall-window-bounded operation the queue stays full and a further caller
    /// blocks on the ENQUEUE. Bounding only the wait for the reply leaves that
    /// caller waiting the whole stall window — a minute on the shipped
    /// defaults, against the eleven seconds the callers holding the queue slots
    /// wait. `SessionScope::global` funnels every proxied request through one
    /// such queue, so this is the ordinary case under traffic, not a corner.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_check_past_the_queues_last_slot_is_bounded_like_one_that_got_a_slot() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(200),
            stall_window: Duration::from_secs(20),
            ..EvaluationDeadlines::default()
        };
        let (map, ev) = paced_map(&store, deadlines);
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        // Past the queue's capacity (one, in this map), so most of these
        // callers never get a slot at all.
        let n = 4;
        let mut waiting = Vec::new();
        for _ in 0..n {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            waiting.push(tokio::spawn(async move {
                let started = Instant::now();
                let answered = map.dispatch(&scope, None, req_for(&scope), seq).await;
                (answered.is_err(), started.elapsed())
            }));
        }

        let bound = deadlines.query_budget + DISPATCH_GRACE + Duration::from_secs(2);
        for (i, handle) in waiting.into_iter().enumerate() {
            let (denied, waited) = handle.await.unwrap();
            assert!(
                denied,
                "check {i} must fail closed against a wedged evaluator"
            );
            assert!(
                waited < bound,
                "check {i} waited {waited:?} on a {:?} budget — it was bounded by the stall \
                 window (or by nothing), not by the budget",
                deadlines.query_budget
            );
        }
    }

    /// The IPCs the task issues on its own — the background flush, the
    /// re-sync it does for a lagged ring, the keepalive — have no caller
    /// waiting on them and no budget over them. Left unbounded, one that never
    /// returns takes the task with it: the loop never reaches the select, so
    /// the stall window is never even armed, and every check for the session
    /// waits on a process nothing will kill.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_background_flush_that_never_returns_is_killed_at_the_stall_window() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(200),
                stall_window: Duration::from_millis(600),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        map.dispatch(&scope, None, req_for(&scope), seq)
            .await
            .expect("the healthy evaluator answers the first check");

        // From here the evaluator swallows updates. A new event gives the task
        // something to flush in the background, with nobody waiting on it.
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);
        store.merge_events(&scope, None, vec![ev_for("b")]).unwrap();

        let waited = Instant::now();
        while ev.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            ev.kills.load(Ordering::Relaxed),
            1,
            "the evaluator wedged a background flush and was never killed: the task is \
             stuck ahead of its own liveness guard"
        );
    }

    /// The stall clock measures how long the evaluator has been silent, so
    /// only something the evaluator SAID may reset it. A pre-query flush with
    /// nothing to flush says nothing — it returns without an IPC — and
    /// clearing the mark on it would hand a busy session a fresh clock on
    /// every check: an evaluator that answers nothing at all would look
    /// healthy forever, and the kill and respawn would never come.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn steady_traffic_does_not_reset_the_stall_clock() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(100),
                stall_window: Duration::from_millis(600),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        map.dispatch(&scope, None, req_for(&scope), seq)
            .await
            .expect("the healthy evaluator answers the first check");

        // Now it stops answering queries, while checks keep arriving — the
        // ordinary shape of a session under load whose policy hit an input it
        // cannot finish. Several callers at once, deliberately: a session with
        // a queue that never empties is the case where nothing but the clock
        // can notice the silence, and it is the case a busy server is in.
        ev.set_query_ms(PacedEvaluator::FOREVER_MS);
        let pumps: Vec<_> = (0..3)
            .map(|_| {
                let map = Arc::clone(&map);
                let scope = scope.clone();
                tokio::spawn(async move {
                    loop {
                        let _ = map.dispatch(&scope, None, req_for(&scope), seq).await;
                        // A yield point, so an `abort()` on this pump can land once
                        // the session is dead and every dispatch returns without
                        // parking; without one the runtime's shutdown waits for ever.
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        let waited = Instant::now();
        while ev.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for pump in pumps {
            pump.abort();
        }

        assert!(
            ev.kills.load(Ordering::Relaxed) > 0,
            "the evaluator owed a reply for the whole run and was never killed: each \
             query reset the stall clock, so the silence was never measured"
        );
    }

    /// The other half of "queries during the respawn are denied, not hung".
    ///
    /// The kill leaves the map holding an entry whose task has exited. A check
    /// arriving then is not denied and is not left waiting on a dead task: the
    /// map drops the entry, spawns a fresh evaluator, bootstraps it from the
    /// store and answers — inside a bound, which is the part that makes this a
    /// property and not a hope.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_check_arriving_after_the_task_exited_is_answered_by_a_fresh_evaluator() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        // The window is well clear of the budget, so the check is denied at
        // its budget and the kill lands later — the order that leaves the map
        // holding a dead entry rather than dropping it on the denied check's
        // way out.
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(300),
            stall_window: Duration::from_millis(900),
            ..EvaluationDeadlines::default()
        };
        let (map, wedged, fresh) = wedged_then_fresh_map_with(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        assert!(
            map.dispatch(&scope, None, req_for(&scope), seq)
                .await
                .is_err(),
            "the wedged check fails closed"
        );
        let waited = Instant::now();
        while wedged.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5)
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            wedged.kills.load(Ordering::Relaxed),
            1,
            "the wedged evaluator was never killed, so there is no respawn to test"
        );
        // The task is gone. The map is normally still holding its entry,
        // marked dead; a denied check that returned after the kill would have
        // dropped it on its way out instead. What must not be true is an entry
        // that still looks live.
        let entry = map.inner.read().get(&scope).map(|se| se.process_died());
        assert_ne!(
            entry,
            Some(false),
            "the killed session's entry is still marked live, so the next check is handed \
             a task that has exited"
        );

        // The respawn's own work — the fork and the bootstrap — is not inside
        // anybody's budget, so the bound is the caller's budget and grace plus
        // room for one spawn.
        let bound = deadlines.query_budget + DISPATCH_GRACE + Duration::from_secs(1);
        let started = Instant::now();
        let answered = tokio::time::timeout(
            Duration::from_secs(10),
            map.dispatch(&scope, None, req_for(&scope), seq),
        )
        .await
        .expect("the check after the respawn must not hang")
        .unwrap_or_else(|e| panic!("the respawned evaluator should answer: {e}"));
        let elapsed = started.elapsed();
        assert!(answered.eval_response.results[0].authorized);
        assert!(
            elapsed < bound,
            "the check waited {elapsed:?} for its respawn, past the {bound:?} it is owed"
        );
        assert_eq!(
            fresh.started.load(Ordering::Relaxed),
            1,
            "the answer came from the respawned evaluator"
        );
    }

    /// An evaluator that owes a reply is killed within the stall window even
    /// while the broadcast ring is busy and the queue never empties — the
    /// state that puts the task in the select, where its liveness arm is.
    ///
    /// That arm is first in a `biased` select so a busy ring cannot starve it.
    /// The stall check is deliberately made twice, though — the query arm
    /// makes it too, on the job it dequeues — so this test pins the property,
    /// not that arm's exclusive credit for it: with the arm removed and the
    /// ring flooded, the query arm still gets polled on this host and still
    /// kills.
    ///
    /// The kill is timed, not merely counted, because "eventually" is not the
    /// property. Steady traffic is exactly the state in which the task always
    /// has another job to dequeue, and a job that took a fresh full budget
    /// from wherever it happened to start held the task in the query arm past
    /// the instant the child's silence became a kill.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_stall_arm_kills_even_while_the_broadcast_ring_is_flooded() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        // A budget that does not divide the window, so the last job dequeued
        // before the stall deadline has only part of a budget left to it: a
        // fresh one would carry the task past that deadline, and that is the
        // overshoot this test bounds.
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(300),
            stall_window: Duration::from_millis(400),
            ..EvaluationDeadlines::default()
        };
        let (map, ev) = paced_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        map.dispatch(&scope, None, req_for(&scope), seq)
            .await
            .expect("the healthy evaluator answers the first check");

        ev.set_query_ms(PacedEvaluator::FOREVER_MS);
        // Checks that keep the queue non-empty, so the task never falls into
        // the catch-up probe and always reaches the select.
        let pumps: Vec<_> = (0..3)
            .map(|_| {
                let map = Arc::clone(&map);
                let scope = scope.clone();
                tokio::spawn(async move {
                    loop {
                        let _ = map.dispatch(&scope, None, req_for(&scope), seq).await;
                        // A yield point, so an `abort()` on this pump can land once
                        // the session is dead and every dispatch returns without
                        // parking; without one the runtime's shutdown waits for ever.
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();
        // The ring is store-wide, so other sessions' writes are what this
        // session's task keeps waking up to.
        let flood = {
            let store = Arc::clone(&store);
            tokio::spawn(async move {
                let mut n = 0u64;
                loop {
                    n += 1;
                    let other = SessionScope::new(T, format!("flood-{}", n % 8));
                    let _ = store.merge_events(&other, None, vec![ev_for(&format!("f{n}"))]);
                    tokio::task::yield_now().await;
                }
            })
        };

        // The moment the evaluator went silent: the first query it takes and
        // never answers. `queries` counts one already — the healthy check
        // above — so the second is the wedged one, and the poll is a
        // millisecond wide against a window of hundreds.
        let waited = Instant::now();
        while ev.queries.load(Ordering::Relaxed) < 2 && waited.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let silent = Instant::now();
        while ev.kills.load(Ordering::Relaxed) == 0 && silent.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let elapsed = silent.elapsed();
        flood.abort();
        for pump in pumps {
            pump.abort();
        }
        assert!(
            ev.kills.load(Ordering::Relaxed) > 0,
            "the evaluator owed a reply for the whole run and was never killed: the liveness \
             arm was starved by the broadcast traffic below it"
        );
        // The window, plus room for the flooded task to be scheduled and for
        // the kill itself.
        //
        // What holds the timing here is the catch-up probe and the liveness
        // arm, both anchored on the silence already: a saturated ring keeps
        // the broadcast arm ready, so the query arm below it is barely polled
        // and the job budget is hardly ever what the task is inside of when
        // the deadline passes. The job budget's own clamp is timed under
        // steady traffic instead, by
        // `a_job_dequeued_near_the_stall_deadline_cannot_outlast_it`.
        let bound = deadlines.stall_window + Duration::from_millis(150);
        assert!(
            elapsed < bound,
            "the evaluator was killed {elapsed:?} after it went silent — past the {bound:?} \
             its {:?} stall window allows",
            deadlines.stall_window
        );
    }

    /// A job dequeued while the evaluator already owes a reply gets what is
    /// LEFT of that silence, not a fresh full budget.
    ///
    /// Steady traffic is what makes this reachable: the queue always holds
    /// another job, so the task is back in the query arm the moment it has
    /// denied the last one. Taking a fresh budget from wherever that happened
    /// to fall left the task inside the query arm past the instant the child's
    /// silence became a kill — and neither the liveness arm nor the catch-up
    /// probe, the two places the kill is made, is reached from in there. The
    /// budget deliberately does not divide the window, so the last job before
    /// the stall deadline has only part of a budget's worth of silence left.
    ///
    /// Deliberately without the broadcast flood that
    /// `the_stall_arm_kills_even_while_the_broadcast_ring_is_flooded` runs
    /// under: a saturated ring keeps the broadcast arm ready and the query arm
    /// below it is never polled, so the task never dequeues the job whose
    /// budget this is about.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_job_dequeued_near_the_stall_deadline_cannot_outlast_it() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(300),
            stall_window: Duration::from_millis(400),
            ..EvaluationDeadlines::default()
        };
        let (map, ev) = paced_map(&store, deadlines);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        map.dispatch(&scope, None, req_for(&scope), seq)
            .await
            .expect("the healthy evaluator answers the first check");

        ev.set_query_ms(PacedEvaluator::FOREVER_MS);
        // The queue holds one job (capacity 1) with the others blocked in
        // `send`, so there is always something to dequeue the moment the task
        // is back at the select, and never the idle moment in which the
        // catch-up probe — anchored on the silence already — would be what
        // kills.
        let pumps: Vec<_> = (0..3)
            .map(|_| {
                let map = Arc::clone(&map);
                let scope = scope.clone();
                tokio::spawn(async move {
                    loop {
                        let _ = map.dispatch(&scope, None, req_for(&scope), seq).await;
                        // A yield point, so an `abort()` on this pump can land once
                        // the session is dead and every dispatch returns without
                        // parking; without one the runtime's shutdown waits for ever.
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        // The moment the evaluator went silent: the first query it takes and
        // never answers. `queries` counts one already — the healthy check
        // above — so the second is the wedged one, and the poll is a
        // millisecond wide against a window of hundreds.
        let waited = Instant::now();
        while ev.queries.load(Ordering::Relaxed) < 2 && waited.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let silent = Instant::now();
        while ev.kills.load(Ordering::Relaxed) == 0 && silent.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let elapsed = silent.elapsed();
        let queries = ev.queries.load(Ordering::Relaxed);
        for pump in pumps {
            pump.abort();
        }
        assert!(
            ev.kills.load(Ordering::Relaxed) > 0,
            "the evaluator owed a reply for the whole run and was never killed"
        );
        assert!(
            queries >= 3,
            "only {queries} queries were issued: no job was dequeued into the query arm \
             after the first was denied, so the budget this test is about was never taken \
             and the timing below proves nothing"
        );
        // The window, plus room for the task to be scheduled and for the kill
        // itself. Well short of the window plus a whole query budget, which is
        // what the job dequeued at the last deny takes without the clamp.
        let bound = deadlines.stall_window + Duration::from_millis(150);
        assert!(
            elapsed < bound,
            "the evaluator was killed {elapsed:?} after it went silent — past the {bound:?} \
             a {:?} stall window allows: a job dequeued near the stall deadline is still \
             taking a fresh {:?} budget",
            deadlines.stall_window,
            deadlines.query_budget
        );
    }

    /// The stall clock measures silence since the child last spoke, across
    /// query and reload phases. A query that resumes an abandoned reload must
    /// not restart that clock.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_evaluator_is_killed_one_stall_window_after_it_went_quiet() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let deadlines = EvaluationDeadlines {
            // A budget close to the window, so the phase that re-armed is
            // separated from the silence by enough to tell the two apart.
            query_budget: Duration::from_millis(800),
            stall_window: Duration::from_millis(1_000),
            ..EvaluationDeadlines::default()
        };
        let (map, ev) = paced_map(&store, deadlines);
        // Global scope: its fence reconciles from the store, so a check past
        // the snapshot triggers the reload this test needs.
        let scope = SessionScope::new(T, "");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        map.dispatch(
            &scope,
            None,
            req_for(&scope),
            store.session_sequence(&scope),
        )
        .await
        .expect("the first check bootstraps and is answered");

        // The write that puts the global scope past its snapshot, so the next
        // check has to reload. Flushed while the evaluator is still healthy:
        // the reload is what this test wedges, and a background flush caught
        // by the wedge instead would never let a caller's own budget be the
        // thing that abandons one.
        store.merge_events(&scope, None, vec![ev_for("b")]).unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let seq = store.session_sequence(&scope);

        // From here nothing the evaluator is asked ever comes back.
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);

        let quiet_since = Instant::now();
        let first = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        // Queued behind the first, which is what keeps the task out of the
        // catch-up probe and sends it into the reload it drives itself.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let second = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };

        // One window plus a margin for a loaded machine. Two windows is the
        // shape this rules out.
        let must_be_dead_by = quiet_since + Duration::from_millis(1_400);
        while ev.kills.load(Ordering::Relaxed) == 0 && Instant::now() < must_be_dead_by {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            ev.kills.load(Ordering::Relaxed) > 0,
            "the evaluator had been silent for {:?} and was still not killed: a phase that \
             started after the silence began was given a stall window of its own",
            quiet_since.elapsed()
        );

        // Both denies name the phase, which is what says the kill this test
        // timed was the one that came out of the reload the task drove.
        let first = tokio::time::timeout(Duration::from_secs(5), first)
            .await
            .expect("the first check must be answered")
            .unwrap()
            .err()
            .expect("the check whose reload was abandoned fails closed");
        assert!(
            first.contains("the graph re-sync"),
            "the first check should be denied at its budget during the reload, got: {first}"
        );
        let second = tokio::time::timeout(Duration::from_secs(5), second)
            .await
            .expect("the queued check must be answered")
            .unwrap()
            .err()
            .expect("the check queued behind it fails closed");
        assert!(
            second.contains("being reloaded"),
            "the second check should be the one that hands the reload to the task, got: {second}"
        );
    }

    /// A reload is O(the scope's graph) and the budget is a constant, so a
    /// graph big enough that its reload does not fit is a graph whose reload
    /// never fits. Running it on the caller's budget therefore denies the
    /// check AND abandons the reload, and the next check re-triggers exactly
    /// the same thing: the scope is locked out for good. The task has to drive
    /// one reload to completion under its own bound.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reload_too_big_for_the_budget_denies_and_then_lands() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(150),
                stall_window: Duration::from_secs(10),
                ..EvaluationDeadlines::default()
            },
        );
        // Global scope: the refmon's own path, and the one that reconciles
        // from the durable store rather than fencing on the broadcast. Its
        // `min_sequence` is its own shard's counter, exactly as the engine
        // passes it.
        let scope = SessionScope::new(T, "");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        map.dispatch(
            &scope,
            None,
            req_for(&scope),
            store.session_sequence(&scope),
        )
        .await
        .expect("the first check bootstraps and is answered");

        // Every reload from here takes longer than a check may wait for it.
        ev.set_update_ms(400);
        store.merge_events(&scope, None, vec![ev_for("b")]).unwrap();
        let seq = store.session_sequence(&scope);

        let denied = map
            .dispatch(&scope, None, req_for(&scope), seq)
            .await
            .err()
            .expect("a check that needs a reload it cannot wait for fails closed");
        assert!(
            !denied.is_empty(),
            "the deny must say something: {denied:?}"
        );

        // The reload the task owes lands on its own, and the scope serves
        // again. Without that, every later check re-triggers the same doomed
        // reload and this loop never sees an answer.
        let waited = Instant::now();
        let mut served = false;
        while !served && waited.elapsed() < Duration::from_secs(8) {
            served = map
                .dispatch(&scope, None, req_for(&scope), seq)
                .await
                .is_ok();
            if !served {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
        assert!(
            served,
            "every check for this scope was denied: a reload that does not fit the budget \
             locks the scope out permanently"
        );
    }

    /// A pre-query flush takes the buffered updates OUT of the buffer before
    /// the IPC that carries them, so a flush abandoned at the budget loses
    /// them: nothing resends the batch, and the sequence markers that would
    /// have forced a re-sync were consumed when those records were queued. The
    /// session must not go on answering checks from a graph that is silently
    /// missing them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_flush_dropped_at_the_budget_reloads_before_the_next_answer() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(150),
                stall_window: Duration::from_secs(5),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);
        map.dispatch(&scope, None, req_for(&scope), seq)
            .await
            .expect("the healthy evaluator answers the first check");
        assert_eq!(
            ev.resets.load(Ordering::Relaxed),
            0,
            "nothing to reload yet"
        );

        // Everything the evaluator does now takes longer than a check may wait
        // for it. The first check is denied and leaves a reply owed, which is
        // what suppresses the background flush; the second one arrives behind
        // it, so the records buffered in between go out on ITS pre-query
        // flush — the flush that is then dropped at its budget.
        ev.set_query_ms(400);
        ev.set_update_ms(400);
        let first = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        store.merge_events(&scope, None, vec![ev_for("b")]).unwrap();
        let newer = store.session_sequence(&scope);
        let second = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), newer).await })
        };
        assert!(first.await.unwrap().is_err(), "the slow query is denied");
        assert!(
            second.await.unwrap().is_err(),
            "the check whose flush outran the budget is denied"
        );

        // It answers queries again. Checks may still be denied while the
        // reload the session now owes is running, but the FIRST one that is
        // answered must come after that reload — never from the graph the
        // dropped batch left behind.
        ev.set_query_ms(0);
        let waited = Instant::now();
        let mut served = false;
        while !served && waited.elapsed() < Duration::from_secs(8) {
            served = map
                .dispatch(&scope, None, req_for(&scope), newer)
                .await
                .is_ok();
            if !served {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        }
        assert!(served, "the session never served again");
        assert!(
            ev.resets.load(Ordering::Relaxed) > 0,
            "a check was answered without reloading: the records the dropped flush took \
             with it are missing from the graph that answered it"
        );
    }

    /// A budget miss leaves the evaluator computing and its answer unread —
    /// the task suppresses every IPC it would otherwise issue while a reply is
    /// owed, so only a brand-new check could observe that answer. A session
    /// that goes quiet after one slow check would then lose a healthy
    /// evaluator, its loaded graph, and take a kill on the ledger it is capped
    /// by. One bounded probe is what turns "nobody asked again" back into
    /// "the process is fine".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_evaluator_that_answered_late_is_not_killed_when_the_session_goes_quiet() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(100),
                stall_window: Duration::from_millis(700),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        // Slower than the budget, faster than the stall window: the check is
        // denied, and the evaluator finishes shortly after.
        ev.set_query_ms(250);
        assert!(
            map.dispatch(&scope, None, req_for(&scope), seq)
                .await
                .is_err(),
            "a query over its budget must be denied"
        );

        // Nobody asks again — the session went quiet, as sessions do.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(
            ev.kills.load(Ordering::Relaxed),
            0,
            "a healthy evaluator that answered just after the budget was killed anyway, \
             because nothing ever read its reply"
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "the session lost its evaluator over one slow query"
        );
    }

    /// A short stall window, for tests that want the kill inside their own
    /// runtime rather than a minute later.
    fn short_stall() -> EvaluationDeadlines {
        EvaluationDeadlines {
            query_budget: Duration::from_millis(100),
            stall_window: Duration::from_millis(400),
            ..EvaluationDeadlines::default()
        }
    }

    /// A map whose first evaluator never answers and whose second one does —
    /// the shape of a stall followed by a respawn.
    fn wedged_then_fresh_map(
        store: &Arc<GraphStore>,
    ) -> (
        Arc<SessionEvaluatorMap>,
        Arc<BlockingEvaluator>,
        Arc<BlockingEvaluator>,
    ) {
        wedged_then_fresh_map_with(store, short_stall())
    }

    fn wedged_then_fresh_map_with(
        store: &Arc<GraphStore>,
        deadlines: EvaluationDeadlines,
    ) -> (
        Arc<SessionEvaluatorMap>,
        Arc<BlockingEvaluator>,
        Arc<BlockingEvaluator>,
    ) {
        let (wedged, _never) = BlockingEvaluator::new("wedged");
        let fresh = BlockingEvaluator::already_released("fresh");
        let (fw, ff) = (Arc::clone(&wedged), Arc::clone(&fresh));
        // Which evaluator a spawn gets turns on whether the wedged one has
        // been killed yet, not on how many times the factory has been called.
        // Two spawns can race one cold scope — the pre-warm listener and a
        // dispatch both reach for it — and the loser's evaluator is
        // constructed and dropped, so counting calls would sometimes hand the
        // surviving spawn the healthy evaluator and wedge nothing at all.
        let factory: EvaluatorFactory = Arc::new(move || {
            if fw.kills.load(Ordering::Relaxed) == 0 {
                Ok(Arc::clone(&fw) as Arc<dyn Evaluator>)
            } else {
                Ok(Arc::clone(&ff) as Arc<dyn Evaluator>)
            }
        });
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "blocking".to_string());
        (map, wedged, fresh)
    }

    /// An evaluator that owes a reply for longer than the stall window is
    /// killed, whether or not anybody is still waiting for it. The budget miss
    /// alone does not kill: the query is denied and the process is given the
    /// rest of the window to come back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_evaluator_silent_past_the_stall_window_is_killed() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, wedged, _fresh) = wedged_then_fresh_map(&store);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        assert!(
            map.dispatch(&scope, None, req_for(&scope), seq)
                .await
                .is_err(),
            "the query must be denied at its budget"
        );
        assert_eq!(
            wedged.kills.load(Ordering::Relaxed),
            0,
            "one slow query is not a reason to kill the evaluator"
        );

        // Nobody asks again; the stall window alone has to reach the process.
        let waited = Instant::now();
        while wedged.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5)
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            wedged.kills.load(Ordering::Relaxed),
            1,
            "the evaluator was left owing a reply past the stall window"
        );
    }

    /// An evaluator that is alive but refuses the metadata seed.
    ///
    /// The seed failing is not the same as the subprocess dying: an oversized
    /// frame or a decode failure comes back as an ordinary error from a
    /// process that is still there.
    struct SeedRefusingEvaluator {
        seeds: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl Evaluator for SeedRefusingEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn set_metadata(&self, _: Vec<PolicyMetadataFact>) -> Result<(), EvaluatorError> {
            self.seeds.fetch_add(1, Ordering::Relaxed);
            Err(EvaluatorError::UpdateError("frame too large".into()))
        }

        fn backend_name(&self) -> &str {
            "seed-refusing"
        }
    }

    /// A session whose seed fails gets another evaluator, not a permanent
    /// refusal.
    ///
    /// Refusing to serve without the config facts is the right call — an empty
    /// `PolicyMetadata` EDB denies nothing a metadata-driven policy means to
    /// deny. But the task exiting is only half of it: unless the handle is
    /// marked dead the map's fast path keeps handing that dead handle out, and
    /// every dispatch refreshes the session's last-activity stamp, so the idle
    /// sweep never reaps it either. One failed seed would refuse the session
    /// for the life of the process, with no second attempt ever made.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_session_whose_metadata_seed_fails_is_respawned_not_refused_forever() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let seeds = Arc::new(AtomicUsize::new(0));
        let spawns = Arc::new(AtomicUsize::new(0));
        let (fseeds, fspawns) = (Arc::clone(&seeds), Arc::clone(&spawns));
        let factory: EvaluatorFactory = Arc::new(move || {
            fspawns.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(SeedRefusingEvaluator {
                seeds: Arc::clone(&fseeds),
            }) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            short_budget(),
        );
        let policy_id = map.swap_factory(factory, "seed-refusing".to_string());
        // A fact to seed, so the seed is attempted at all.
        store
            .put_policy_metadata(
                T,
                policy_id.as_str(),
                &[sasy_graph::PolicyMetadataFact {
                    rel: "TrustedDomain".into(),
                    a: "registry.npmjs.org".into(),
                    b: String::new(),
                }],
            )
            .unwrap();

        let scope = SessionScope::new(T, "S");
        for attempt in 1..=2 {
            assert!(
                map.dispatch(&scope, None, req_for(&scope), 0)
                    .await
                    .is_err(),
                "attempt {attempt}: a session that could not be given its config must fail closed"
            );
        }

        assert_eq!(
            spawns.load(Ordering::Relaxed),
            2,
            "the second check reused the handle of a task that had already exited: nothing \
             marked it dead, so the map never evicted it and no second evaluator was ever tried"
        );
        assert_eq!(seeds.load(Ordering::Relaxed), 2, "each spawn seeds once");
    }

    /// An evaluator that never answers a metadata re-seed.
    ///
    /// Distinct from `BlockingEvaluator`, which a kill releases: this models
    /// the RPC path's own hazard, where the wedge outlives everything the
    /// session evaluator does about it.
    struct WedgedSeedEvaluator;

    #[tonic::async_trait]
    impl Evaluator for WedgedSeedEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            Ok(EvalAuthResponse {
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

        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }

        async fn set_metadata(&self, _: Vec<PolicyMetadataFact>) -> Result<(), EvaluatorError> {
            std::future::pending::<()>().await;
            unreachable!()
        }

        fn backend_name(&self) -> &str {
            "wedged-seed"
        }
    }

    /// The metadata re-seed is bounded like every other pre-serving IPC.
    ///
    /// `UpdatePolicyMetadata` reaches it while holding the scope lock, and it
    /// bridges into the runtime with `block_on`, so an unbounded wait there is
    /// not merely slow: the runtime worker is blocked, the future cannot be
    /// cancelled when the client goes away, and the scope lock is never given
    /// back — `EndSession` and every later metadata write for that session
    /// queue behind it for good.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_metadata_reseed_against_a_wedged_evaluator_returns_instead_of_hanging() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(Arc::new(WedgedSeedEvaluator) as Arc<dyn Evaluator>));
        let deadlines = EvaluationDeadlines {
            query_budget: Duration::from_millis(150),
            stall_window: Duration::from_millis(500),
            ..EvaluationDeadlines::default()
        };
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "wedged-seed".to_string());

        // Nothing to seed at spawn, so the session comes up serving; the
        // re-seed below is the first thing this evaluator is asked to accept.
        let scope = SessionScope::new(T, "S");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .expect("the session serves before the re-seed");

        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            map.update_session_metadata(
                &scope,
                vec![PolicyMetadataFact {
                    rel: "TrustedDomain".into(),
                    a: "registry.npmjs.org".into(),
                    b: String::new(),
                }],
            ),
        )
        .await;

        let returned = outcome.expect(
            "the re-seed never returned: an unbounded IPC on an RPC path that holds the scope \
             lock and cannot be cancelled",
        );
        assert!(
            returned.is_err(),
            "a re-seed the evaluator never accepted must be reported as a failure"
        );
    }

    /// An error that never reached the child is not evidence that the child
    /// spoke.
    ///
    /// A call can fail ahead of the child mutex — a request the transport
    /// refuses is rejected before a frame is written — so the evaluator hears
    /// nothing and still owes whatever it owed. Clearing the "owes a reply"
    /// mark on that error disarms the stall window outright: the wedged
    /// process is never killed, never respawned and never counted against the
    /// cap, and under traffic a refused check is the ordinary case, not a rare
    /// one, because such a check is queued behind the wedged one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_query_refused_before_the_child_hears_it_does_not_erase_the_stall_mark() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, wedged, _fresh) = wedged_then_fresh_map_with(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(150),
                stall_window: Duration::from_millis(600),
                ..EvaluationDeadlines::default()
            },
        );
        // The first query wedges; every one after it is refused before the
        // child hears it.
        wedged.refuse_after_starts.store(1, Ordering::Relaxed);
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        // The refused check is QUEUED behind the wedged one, which is where a
        // second check lands under traffic. Queued rather than arriving later
        // on purpose: with nothing queued the task spends the window on its
        // catch-up probe instead, and the query path this pins is never
        // reached.
        let wedging = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        let refused = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        assert!(
            wedging.await.unwrap().is_err(),
            "the wedged query must be denied at its budget"
        );
        assert!(
            refused.await.unwrap().is_err(),
            "the refused query must fail closed too"
        );

        let waited = Instant::now();
        while wedged.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5)
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            wedged.kills.load(Ordering::Relaxed),
            1,
            "the evaluator has been silent since the first query, but an error that never \
             reached it was taken as proof it had spoken, so the stall window never armed"
        );
    }

    /// Everyone waiting on a stalled evaluator is answered when it is killed —
    /// with a deny, not with a dropped response — and the session comes back
    /// on a fresh evaluator that re-bootstraps from the store.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_kill_answers_everyone_waiting_and_the_session_comes_back() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        // Budget and stall window together, so the second check is still
        // queued when the first one's budget runs out and the kill lands. Both
        // a second: long enough that a loaded machine cannot slip the second
        // dispatch past the kill and make the test race its own setup.
        let (map, wedged, fresh) = wedged_then_fresh_map_with(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_secs(1),
                stall_window: Duration::from_secs(1),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let first = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        // Only once the evaluator actually has the first check: from here the
        // second one's path into the queue is a map read and a channel send.
        let waited = Instant::now();
        while wedged.started.load(Ordering::Relaxed) == 0
            && waited.elapsed() < Duration::from_millis(500)
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let queued = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };

        let first = tokio::time::timeout(Duration::from_secs(10), first)
            .await
            .expect("the wedged check must be answered")
            .unwrap();
        let queued = tokio::time::timeout(Duration::from_secs(10), queued)
            .await
            .expect("the check queued behind it must be answered too")
            .unwrap();
        let queued_reason = match queued {
            Err(reason) => reason,
            Ok(_) => panic!("a check behind a stalled evaluator must not be authorized"),
        };
        assert!(first.is_err(), "the wedged check is denied");
        assert!(
            !queued_reason.contains("dropped response"),
            "the queued check lost its sender instead of being denied: {queued_reason}"
        );
        assert_eq!(
            wedged.kills.load(Ordering::Relaxed),
            1,
            "the stalled evaluator should have been killed exactly once"
        );

        let after = tokio::time::timeout(
            Duration::from_secs(5),
            map.dispatch(&scope, None, req_for(&scope), seq),
        )
        .await
        .expect("the check after the respawn must not hang")
        .unwrap_or_else(|e| panic!("the respawned evaluator should answer: {e}"));
        assert!(after.eval_response.results[0].authorized);
        assert_eq!(
            fresh.started.load(Ordering::Relaxed),
            1,
            "the answer came from the respawned evaluator"
        );
    }

    /// A session whose evaluator keeps being killed stops getting new ones.
    ///
    /// The runaway here is deterministic — every evaluator wedges — which is
    /// the case the cap exists for: without it each repetition of the call
    /// would buy another spawn and another full re-bootstrap, forever, for the
    /// same deny.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_session_past_the_kill_cap_is_refused_instead_of_respawned() {
        // The sweep is off here, so this pins the cap on its own.
        kill_cap_stops_the_respawns(Duration::ZERO).await;
    }

    /// The same cap, with the eviction sweep ticking many times inside the
    /// stall window — the production shape, where the 60s sweep and the 60s
    /// stall window overlap.
    ///
    /// A ledger with no kills in the window is "spent", which every healthy
    /// session's is, so a sweep that pruned on that alone would drop the entry
    /// while the session's task still pointed at it: every kill would then be
    /// recorded into an orphan, the respawn would create a fresh zero ledger,
    /// and the cap would never engage.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_kill_cap_survives_a_sweep_inside_the_stall_window() {
        kill_cap_stops_the_respawns(Duration::from_millis(5)).await;
    }

    /// Body of the two cap tests. `sweep_interval` of zero switches the
    /// eviction sweep off; anything else runs it at that cadence with an idle
    /// TTL far longer than the test, so the sweep ticks without evicting.
    async fn kill_cap_stops_the_respawns(sweep_interval: Duration) {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let spawns = Arc::new(AtomicUsize::new(0));
        let factory_spawns = Arc::clone(&spawns);
        let factory: EvaluatorFactory = Arc::new(move || {
            factory_spawns.fetch_add(1, Ordering::Relaxed);
            let (ev, _never) = BlockingEvaluator::new("wedged");
            Ok(ev as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_secs(3600),
            sweep_interval,
            None,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(80),
                stall_window: Duration::from_millis(80),
                ..EvaluationDeadlines::default()
            },
        );
        map.swap_factory(factory, "blocking".to_string());
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        // Keep asking. Every check is denied either way; what changes past the
        // cap is that no new evaluator is started to deny it.
        let mut capped = None;
        for _ in 0..12 {
            let answer = tokio::time::timeout(
                Duration::from_secs(5),
                map.dispatch(&scope, None, req_for(&scope), seq),
            )
            .await
            .expect("no check may hang, capped or not");
            if let Err(reason) = answer {
                if reason.contains("not being respawned") {
                    capped = Some(reason);
                    break;
                }
            }
            // Let the stall window pass so the kill lands before the next one.
            tokio::time::sleep(Duration::from_millis(120)).await;
        }

        let reason = capped.expect("the session should have hit the kill cap");
        assert!(
            reason.contains("killed 3 times"),
            "the deny must say why it is refusing, got: {reason}"
        );

        // Past the cap nothing new is started — that is the whole point. (The
        // count itself is not KILL_CAP: the pre-warm listener also spawns for
        // a scope it sees traffic on, which is exactly the traffic this test
        // generates.)
        let at_cap = spawns.load(Ordering::Relaxed);
        for _ in 0..2 {
            let answer = tokio::time::timeout(
                Duration::from_secs(5),
                map.dispatch(&scope, None, req_for(&scope), seq),
            )
            .await
            .expect("a capped check must not hang");
            match answer {
                Err(reason) => assert!(
                    reason.contains("not being respawned"),
                    "a capped check should say the session is not being respawned, got: {reason}"
                ),
                Ok(_) => panic!("a capped session must not authorize anything"),
            }
        }
        assert_eq!(
            spawns.load(Ordering::Relaxed),
            at_cap,
            "a capped session was handed another evaluator"
        );
    }

    /// A live session's ledger is not swept out from under it.
    ///
    /// It is "spent" — a healthy session has no kills in the window — so the
    /// only thing keeping it in the table is that the session's task still
    /// holds it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_live_sessions_kill_ledger_survives_the_sweep() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_secs(3600), // idle TTL: the session itself stays
            Duration::from_millis(5),  // sweep: many ticks inside this test
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let scope = SessionScope::new(T, "S");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(map.kill_ledgers.len(), 1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(map.live_sessions(), 1, "the session should still be live");
        assert_eq!(
            map.kill_ledgers.len(),
            1,
            "the sweep dropped the ledger of a live session: its next kill would be recorded \
             into an orphan and the cap would never see it"
        );
    }

    /// The kill reaches the ledger before it reaches the world.
    ///
    /// `process_died` is what makes the map respawn, so a dispatcher that read
    /// it while the kill was still on its way into the ledger would run the cap
    /// check one kill short. The test stops the kill exactly inside `record` by
    /// holding the ledger's own lock, and asserts the death is not published
    /// yet.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_kill_is_recorded_before_the_death_is_published() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, wedged, _fresh) = wedged_then_fresh_map_with(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(100),
                stall_window: Duration::from_millis(300),
                ..EvaluationDeadlines::default()
            },
        );
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        let seq = store.session_sequence(&scope);

        let _wedging = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), seq).await })
        };
        let waited = Instant::now();
        while wedged.started.load(Ordering::Relaxed) == 0
            && waited.elapsed() < Duration::from_secs(5)
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            wedged.started.load(Ordering::Relaxed) > 0,
            "the evaluator never received the query that wedges it"
        );
        let se = map
            .inner
            .read()
            .get(&scope)
            .map(Arc::clone)
            .expect("the session is live");
        let ledger = map.kill_ledger_for(&scope);

        // The observation runs on its own OS thread and waits with a blocking
        // sleep, because the kill it stops is stopped inside a runtime worker:
        // a `tokio::time::sleep` here would be waiting on a timer whose driver
        // that worker may be holding.
        let observer = {
            let ledger = Arc::clone(&ledger);
            let se = Arc::clone(&se);
            std::thread::spawn(move || {
                // Taken before the stall window elapses, so the kill blocks
                // inside `record` instead of getting through it.
                let held = ledger.block_records();
                std::thread::sleep(Duration::from_millis(1_200));
                let seen = (held.len(), se.process_died());
                drop(held);
                seen
            })
        };
        let (recorded_mid_kill, died_mid_kill) = observer.join().expect("observer thread");
        assert_eq!(
            recorded_mid_kill, 0,
            "the kill got into the ledger before the observer was in place — the test raced \
             its own setup and proves nothing"
        );
        assert!(
            !died_mid_kill,
            "the death was published while the kill had not reached the ledger yet: a \
             dispatcher reaching the cap check in that window is handed a respawn the cap \
             should have refused"
        );

        // Blocking waits for the same reason, until the worker is free again.
        let waited = Instant::now();
        while ledger.recorded() == 0 && waited.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            ledger.recorded(),
            1,
            "the kill was waiting on the observer's lock and must land once it is released"
        );
        let waited = Instant::now();
        while !se.process_died() && waited.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            se.process_died(),
            "once the record is in, the death is published"
        );
    }

    /// With the sweep switched off, an evict is what stops the ledger table
    /// growing by one entry per session the process has ever seen.
    ///
    /// The entry goes only once BOTH conditions hold: the kills have aged out
    /// (here there were none) and no task still holds it — which is why the
    /// evict that drops the evaluator is not necessarily the prune that
    /// collects its ledger, and the test drives evictions until it is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_evict_drops_a_spent_kill_ledger_with_the_sweep_off() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_secs(3600), // idle TTL: nothing ages out on its own
            Duration::ZERO,            // sweep off
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let scope = SessionScope::new(T, "S");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(
            map.kill_ledgers.len(),
            1,
            "a live session should have a ledger for its kills to land in"
        );

        assert!(map.evict(&scope), "the live evaluator should be dropped");
        // The task's Arc goes when the runtime drops the aborted task, which
        // is not synchronous with the evict. Keep evicting (a no-op on an
        // empty scope, except that it prunes) until the entry is collected.
        let mut collected = false;
        for _ in 0..100 {
            if map.kill_ledgers.len() == 0 {
                collected = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            map.evict(&scope);
        }
        assert!(
            collected,
            "with the sweep off, evicting must still collect a spent kill ledger"
        );
    }

    /// Build a task directly so the broadcast handling can be exercised
    /// without a live subscriber race.
    fn task_for(
        store: &Arc<GraphStore>,
        scope: &SessionScope,
        sequence: i64,
    ) -> SessionEvaluatorTask {
        task_with(store, scope, sequence, LoggingEvaluator::new("e"))
    }

    /// As [`task_for`], with the caller's own mock so it can read the mock's
    /// counters afterwards.
    fn task_with(
        store: &Arc<GraphStore>,
        scope: &SessionScope,
        sequence: i64,
        evaluator: Arc<LoggingEvaluator>,
    ) -> SessionEvaluatorTask {
        let (_qtx, qrx) = async_channel::bounded(1);
        SessionEvaluatorTask {
            scope: scope.clone(),
            evaluator: evaluator as Arc<dyn Evaluator>,
            graph_store: Arc::clone(store),
            broadcast_rx: store.subscribe(),
            query_rx: qrx,
            sequence,
            snapshot_seq: sequence,
            graph_incomplete: false,
            node_count: 0,
            edge_count: 0,
            pending_updates: Vec::new(),
            flush_interval: Duration::from_millis(1),
            metadata: vec![],
            last_active: Arc::new(PMutex::new(Instant::now())),
            process_died: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            oracle_credit_us: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            full_resyncs: Arc::new(AtomicU64::new(0)),
            catch_ups: Arc::new(AtomicU64::new(0)),
            keepalive_interval: Duration::ZERO,
            deadlines: EvaluationDeadlines::default(),
            outstanding: None,
            resync_owed: false,
            policy_hash: "test-policy".to_string(),
            kills: KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
        }
    }

    /// Catch-up and bootstrap must count the same message-dependency edges,
    /// excluding CHILD_OF / PRODUCES / CONSUMES computation edges. Reopen the
    /// store to exercise counts reconstructed from persistence.
    #[tokio::test]
    async fn a_catch_up_installs_the_edge_count_a_bootstrap_would_have() {
        use sasy_common::observability::{Computation, Edge};
        fn span_for(sid: &str, out: &str) -> Computation {
            Computation {
                span_id: sid.to_string(),
                trace_id: "t1".to_string(),
                parent_span_id: None,
                name: "test".to_string(),
                start_time_ns: 0,
                end_time_ns: 100,
                duration_ns: 100,
                status_code: 1,
                status_message: None,
                attributes_json: "{}".to_string(),
                events_json: "[]".to_string(),
                service_name: None,
                service_version: None,
                input_message_ids: vec![],
                output_message_id: Some(out.to_string()),
                linked_span_ids: vec![],
                principal: None,
                entity: None,
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let path = path.to_str().unwrap();
        let scope = SessionScope::new(T, "S");
        {
            let store = GraphStore::new(Some(path)).unwrap();
            store
                .merge_events(&scope, None, vec![ev_for("m1"), ev_for("m2")])
                .unwrap();
            store
                .merge_dependencies(
                    &scope,
                    None,
                    vec![Edge {
                        source: "m1".to_string(),
                        destination: "m2".to_string(),
                        message_index: None,
                        proximal: None,
                        principal: None,
                        entity: None,
                    }],
                )
                .unwrap();
            // Two computation edges: PRODUCES from each span to its message.
            store
                .merge_computations(
                    &scope,
                    None,
                    vec![span_for("sp1", "m1"), span_for("sp2", "m2")],
                )
                .unwrap();
        }
        // Reopen: the shard is now built by the loader, not by the write path.
        let store = Arc::new(GraphStore::new(Some(path)).unwrap());

        let mut booted = task_for(&store, &scope, 0);
        assert!(booted.bootstrap().await, "bootstrap failed");

        let mut caught_up = task_for(&store, &scope, store.session_sequence(&scope));
        assert!(
            matches!(caught_up.catch_up_from_replay().await, CatchUp::Applied),
            "the catch-up did not apply"
        );

        assert_eq!(
            caught_up.edge_count, booted.edge_count,
            "a catch-up installed {} edges where a bootstrap loads {} — the \
             computation edges are counted on one path and not the other",
            caught_up.edge_count, booted.edge_count
        );
        assert_eq!(booted.edge_count, 1, "the bootstrap count itself moved");
    }

    /// A gap noticed while a reload is OWED is closed by the reload, never by
    /// the replay log.
    ///
    /// `resync_owed` is set when a `full_resync` was abandoned at a caller's
    /// budget: its `reset` emptied the evaluator and the snapshot never went
    /// back in. Replaying the missed updates on top of that would build a
    /// graph that looks populated while holding nothing older than the
    /// failure, and every later check would be answered from it — the
    /// fail-open shape the guard exists to prevent. Both arms use the same
    /// store and the same gap, so the only thing that differs is the debt.
    #[tokio::test]
    async fn a_gap_with_a_reload_owed_reloads_instead_of_replaying() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S");
        store
            .merge_events(&scope, None, vec![ev_for("m1"), ev_for("m2")])
            .unwrap();

        // Control arm: nothing owed. The retained log reaches back far enough,
        // so the catch-up closes the gap and nothing is reloaded — which is
        // what makes the other arm's reload a decision rather than the only
        // option available.
        let replayed = LoggingEvaluator::new("replay");
        let mut task = task_with(&store, &scope, 0, Arc::clone(&replayed));
        assert!(
            task.close_broadcast_gap(true).await,
            "the task must keep running"
        );
        assert_eq!(
            task.catch_ups.load(Ordering::Relaxed),
            1,
            "this gap is replayable, so the arm below has something to refuse"
        );
        assert_eq!(task.full_resyncs.load(Ordering::Relaxed), 0);
        assert_eq!(
            replayed.reset_count.load(Ordering::Relaxed),
            0,
            "a replay must not reset the evaluator"
        );

        // The arm under test: the identical gap, with the reload owed.
        let reloaded = LoggingEvaluator::new("reload");
        let mut task = task_with(&store, &scope, 0, Arc::clone(&reloaded));
        task.resync_owed = true;
        assert!(
            task.close_broadcast_gap(true).await,
            "the task must keep running"
        );
        assert_eq!(
            task.catch_ups.load(Ordering::Relaxed),
            0,
            "an incremental apply would have sat on an evaluator holding no graph"
        );
        assert_eq!(
            task.full_resyncs.load(Ordering::Relaxed),
            1,
            "only a reload refills an evaluator a `reset` emptied"
        );
        assert_eq!(
            reloaded.reset_count.load(Ordering::Relaxed),
            1,
            "the reload is what puts the graph back"
        );
        assert!(!task.resync_owed, "a reload that lands clears the debt");
    }

    /// A removal has to reach the evaluator like any other change: the scope
    /// filter drops anything that does not name this session, so an
    /// An `EdgeDeleted` carrying no scope would be discarded by every
    /// subscriber, and the edge would live on in the policy engine's copy of
    /// the graph.
    #[tokio::test]
    async fn a_deleted_edge_reaches_the_evaluator_for_its_own_scope() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S");
        let other = SessionScope::new(T, "OTHER");
        let mut task = task_for(&store, &scope, 0);
        task.edge_count = 1;

        task.process_broadcast(sasy_graph::GraphUpdate::EdgeDeleted {
            source: "m1".to_string(),
            destination: "m2".to_string(),
            scope: other,
        });
        assert!(
            task.pending_updates.is_empty(),
            "another session's deletion was applied"
        );
        assert_eq!(task.edge_count, 1);

        task.process_broadcast(sasy_graph::GraphUpdate::EdgeDeleted {
            source: "m1".to_string(),
            destination: "m2".to_string(),
            scope: scope.clone(),
        });
        assert_eq!(
            task.pending_updates.len(),
            1,
            "this session's deletion never reached the evaluator"
        );
        assert!(matches!(
            task.pending_updates[0],
            GraphUpdate::EdgeDeleted { .. }
        ));
        assert_eq!(task.edge_count, 0);
    }

    /// The task subscribes BEFORE it bootstraps, so the ring can still hold
    /// markers stamped earlier than the snapshot it loaded. Taking one at face
    /// value rewound the fence cursor and made the next query wait out its
    /// deadline on progress the evaluator already had. The same rule is what
    /// lets a replay catch-up set the cursor.
    #[tokio::test]
    async fn a_sequence_marker_older_than_the_snapshot_does_not_rewind_the_fence() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S");
        let mut task = task_for(&store, &scope, 10);

        task.process_broadcast(sasy_graph::GraphUpdate::SessionSequence {
            scope: scope.clone(),
            seq: 3,
        });
        assert_eq!(task.sequence, 10, "a stale marker rewound the fence cursor");

        task.process_broadcast(sasy_graph::GraphUpdate::SessionSequence {
            scope: scope.clone(),
            seq: 12,
        });
        assert_eq!(task.sequence, 12, "a fresh marker must still advance it");
    }

    /// Eviction must stop the evaluator's process, not just its task.
    ///
    /// A session dropped while its evaluator is busy — a forced policy
    /// rollout, `EndSession`, a rebind, the idle sweep — never reaches the
    /// stall window that would kill the evaluator, and aborting the session's
    /// task destroys the only caller of `Evaluator::kill`. The evaluator
    /// handle is deliberately kept alive here (in the server it is shared with
    /// the task being aborted), so nothing but the session's own destructor
    /// can stop the process.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evicting_a_session_stops_its_evaluator_process() {
        use crate::evaluator::manager::{EvaluatorProcess, EvaluatorProcessConfig};

        let evaluator = Arc::new(
            EvaluatorProcess::new(EvaluatorProcessConfig {
                program: "/bin/sleep".to_string(),
                args: vec!["918273647".to_string()],
                backend: "wedged".to_string(),
                env: Vec::new(),
            })
            .expect("spawn /bin/sleep"),
        );
        let pid = evaluator.child_pid() as i32;
        assert!(pid > 0, "no pid was recorded for the spawned child");

        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "evicted");
        let session = SessionEvaluator::spawn(
            scope.clone(),
            Arc::clone(&evaluator) as Arc<dyn Evaluator>,
            Arc::clone(&store),
            Duration::from_millis(1),
            4,
            vec![],
            EvaluationDeadlines::default(),
            "p".to_string(),
            KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
        );
        tokio::task::yield_now().await;

        drop(session); // what every eviction path does
        assert!(
            crate::evaluator::reaper::stopped_running(pid),
            "the evicted session left evaluator process {pid} running"
        );
        // Holding this handle across the drop is the point of the test.
        drop(evaluator);
    }

    /// A gap the store can still replay is closed incrementally: the fence
    /// clears, the evaluator sees the missed updates, and no full reload runs.
    /// Single-threaded on purpose — the writer holds the thread for the whole
    /// burst, so the task is guaranteed to find itself past the ring.
    #[tokio::test(flavor = "current_thread")]
    async fn a_lagged_evaluator_catches_up_without_a_full_resync() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S");
        let eval = LoggingEvaluator::new("e");
        let se = SessionEvaluator::spawn(
            scope.clone(),
            Arc::clone(&eval) as Arc<dyn Evaluator>,
            Arc::clone(&store),
            Duration::from_millis(1),
            4,
            vec![],
            EvaluationDeadlines::default(),
            "p".to_string(),
            KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
        );
        tokio::task::yield_now().await;

        // More updates than the broadcast ring holds (32768) but fewer than the
        // replay log retains (65536), so the gap is recoverable.
        let events: Vec<Event> = (0..40_000).map(|i| ev_for(&format!("m{i}"))).collect();
        store.merge_events(&scope, None, events).unwrap();
        let seq = store.session_sequence(&scope);

        let res =
            tokio::time::timeout(Duration::from_secs(30), se.dispatch(req_for(&scope), seq)).await;
        assert!(res.is_ok(), "dispatch hung after a lag");
        assert!(res.unwrap().is_ok());
        assert!(
            se.catch_up_count() >= 1,
            "the evaluator never fell behind the ring, so this pins nothing"
        );
        assert_eq!(
            se.full_resync_count(),
            0,
            "a recoverable gap must not cost a full reload"
        );
        assert_eq!(
            eval.reset_count.load(Ordering::Relaxed),
            0,
            "the evaluator was reset, so a reload happened after all"
        );
        let applied: usize = eval.updates.lock().iter().sum();
        assert!(
            applied >= 40_000,
            "catch-up shipped {applied} updates, expected every missed one"
        );
    }

    /// A gap older than the retained log has no incremental answer, so the
    /// evaluator still falls back to a full reload.
    #[tokio::test(flavor = "current_thread")]
    async fn a_gap_beyond_retention_still_forces_a_full_resync() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "S");
        let eval = LoggingEvaluator::new("e");
        let se = SessionEvaluator::spawn(
            scope.clone(),
            Arc::clone(&eval) as Arc<dyn Evaluator>,
            Arc::clone(&store),
            Duration::from_millis(1),
            4,
            vec![],
            EvaluationDeadlines::default(),
            "p".to_string(),
            KillLedgerRef::new(Arc::new(KillLedgers::default()), scope.clone()),
        );
        tokio::task::yield_now().await;

        // Past the replay log's 65536-change bound.
        let events: Vec<Event> = (0..70_000).map(|i| ev_for(&format!("m{i}"))).collect();
        store.merge_events(&scope, None, events).unwrap();
        let seq = store.session_sequence(&scope);

        let res =
            tokio::time::timeout(Duration::from_secs(30), se.dispatch(req_for(&scope), seq)).await;
        assert!(res.is_ok(), "dispatch hung after an unrecoverable lag");
        assert!(res.unwrap().is_ok());
        assert!(
            se.full_resync_count() >= 1,
            "an unrecoverable gap must fall back to a full reload"
        );
    }

    /// A query whose `min_sequence` can never be satisfied by a broadcast marker
    /// (the marker was lost — e.g. a writer bumped the shard seq but its commit
    /// errored before the send) must NOT hang the fence forever. It times out
    /// and the check is DENIED: the evaluator cannot see the write the caller
    /// The fence cannot make a job outlast its budget.
    ///
    /// The fence has a bound of its own (`SEQ_FENCE_TIMEOUT`, two seconds),
    /// but it runs inside a job a caller is already waiting on. Unclamped,
    /// a job with a budget under two seconds would spend its budget and then
    /// two more seconds in the fence — the caller waiting longer than the
    /// bound it was promised.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_sequence_fence_is_clamped_to_the_job_budget() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let ev = LoggingEvaluator::new("e");
        let factory_ev = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&factory_ev) as Arc<dyn Evaluator>));
        let budget = Duration::from_millis(300);
        assert!(
            budget < SEQ_FENCE_TIMEOUT,
            "the test only says anything while the budget is the shorter of the two"
        );
        let deadlines = EvaluationDeadlines {
            query_budget: budget,
            stall_window: Duration::from_secs(30),
            ..EvaluationDeadlines::default()
        };
        let map = SessionEvaluatorMap::with_deadlines(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            deadlines,
        );
        map.swap_factory(factory, "mock".to_string());
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        // A marker that will never be broadcast: the fence can only expire.
        let unreachable = store.session_sequence(&scope) + 10_000;

        let started = Instant::now();
        let answer = tokio::time::timeout(
            Duration::from_secs(8),
            map.dispatch(&scope, None, req_for(&scope), unreachable),
        )
        .await
        .expect("the check must not hang");
        let elapsed = started.elapsed();
        let reason = match answer {
            Err(reason) => reason,
            Ok(_) => panic!("a fence the evaluator never cleared must not be answered"),
        };
        // The task's own fence deny, not `dispatch`'s backstop: unclamped, the
        // fence runs its full two seconds and the backstop is what answers.
        assert!(
            reason.contains("caught up"),
            "the deny must be the fence's own, got: {reason}"
        );
        assert!(
            elapsed < well_inside_the_dispatch_backstop(deadlines),
            "the caller waited {elapsed:?} on a {budget:?} budget: the fence ran its own \
             {SEQ_FENCE_TIMEOUT:?} instead of the time the job had left"
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "the evaluator was dropped over a fence it had not cleared"
        );
    }
    /// just made, which is not something to answer from an older graph.
    #[tokio::test]
    async fn seq_fence_times_out_instead_of_hanging() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let ev = LoggingEvaluator::new("e");
        let factory_ev = Arc::clone(&ev);
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(Arc::clone(&factory_ev) as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());
        let scope = SessionScope::new(T, "S");
        store.merge_events(&scope, None, vec![ev_for("a")]).unwrap();
        // min_sequence far beyond any marker that will ever be broadcast.
        let unreachable = store.session_sequence(&scope) + 10_000;
        // Bound the whole dispatch. Without SEQ_FENCE_TIMEOUT the fence
        // `recv().await` never returns; with it the dispatch denies and breaks.
        let res = tokio::time::timeout(
            Duration::from_secs(8),
            map.dispatch(&scope, None, req_for(&scope), unreachable),
        )
        .await;
        assert!(
            res.is_ok(),
            "dispatch hung on an unreachable min_sequence (fence not bounded)"
        );
        let answer = res.unwrap();
        let reason = match answer {
            Err(reason) => reason,
            Ok(_) => panic!("a fence the evaluator never cleared must not be answered"),
        };
        assert!(
            reason.contains("caught up"),
            "the deny must say the evaluator is behind, got: {reason}"
        );
        // `reset()` is called once per full re-sync, so this pins the other half
        // of the rule: a fence miss denies WITHOUT reloading the session.
        assert_eq!(
            ev.reset_count.load(Ordering::Relaxed),
            0,
            "a fence miss reloaded the session instead of denying"
        );
        assert_eq!(
            map.live_sessions(),
            1,
            "the evaluator was dropped over a fence it had not cleared"
        );
    }

    /// The fence bound must be a TOTAL wall-clock deadline, not a per-`recv()`
    /// timer: on a busy server the store-wide broadcast ring keeps delivering
    /// OTHER sessions' updates within microseconds, and a per-recv timeout would
    /// re-arm every iteration and never elapse — hanging the check under load.
    /// Here a background task floods the ring from a different session while the
    /// target's fence waits on an unreachable min_sequence; the dispatch must
    /// still return (with the fence's deny) within a bound comfortably above
    /// SEQ_FENCE_TIMEOUT.
    #[tokio::test]
    async fn seq_fence_deadline_bounded_under_cross_session_flood() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.swap_factory(factory, "mock".to_string());
        let target = SessionScope::new(T, "TARGET");
        store
            .merge_events(&target, None, vec![ev_for("t0")])
            .unwrap();
        let unreachable = store.session_sequence(&target) + 10_000;

        // Flood the shared ring from a DIFFERENT session for longer than the fence
        // deadline, so the target's recv() never blocks (it keeps getting foreign,
        // filtered updates). With a per-recv timer this starves the timeout.
        let flood_store = Arc::clone(&store);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_f = Arc::clone(&stop);
        let flood = tokio::spawn(async move {
            let other = SessionScope::new(T, "NOISY");
            let mut i = 0u64;
            while !stop_f.load(Ordering::Relaxed) {
                let _ = flood_store.merge_events(&other, None, vec![ev_for(&format!("n{i}"))]);
                i += 1;
                tokio::task::yield_now().await;
            }
        });

        let res = tokio::time::timeout(
            Duration::from_secs(8),
            map.dispatch(&target, None, req_for(&target), unreachable),
        )
        .await;
        stop.store(true, Ordering::Relaxed);
        let _ = flood.await;
        assert!(
            res.is_ok(),
            "dispatch hung: cross-session ring traffic starved a per-recv fence timeout"
        );
        assert!(
            res.unwrap().is_err(),
            "a fence the evaluator never cleared is denied, not answered"
        );
    }

    /// A replay catch-up abandoned inside the sequence fence denies through
    /// the BUDGET, not through the fence.
    ///
    /// The two denies are not interchangeable. The fence's own deny says the
    /// evaluator is merely behind and leaves it alone — no reply is owed, so
    /// no stall clock starts. But a catch-up dropped at the deadline abandoned
    /// an `Update` frame the child may still be inside, and the records it
    /// carried are gone from the buffer: the evaluator owes a reply and its
    /// graph is incomplete. Denying that through the fence would leave a
    /// wedged child with nothing measuring its silence.
    ///
    /// Single-threaded on purpose: the flood holds the thread for its whole
    /// burst, so the task is guaranteed to find itself past the ring rather
    /// than draining it as it fills.
    #[tokio::test(flavor = "current_thread")]
    async fn a_fence_catch_up_over_budget_denies_at_the_budget_and_starts_the_stall_clock() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let (map, ev) = paced_map(
            &store,
            EvaluationDeadlines {
                query_budget: Duration::from_millis(800),
                stall_window: Duration::from_millis(500),
                ..EvaluationDeadlines::default()
            },
        );
        let target = SessionScope::new(T, "TARGET");
        store
            .merge_events(&target, None, vec![ev_for("t0")])
            .unwrap();
        map.dispatch(
            &target,
            None,
            req_for(&target),
            store.session_sequence(&target),
        )
        .await
        .expect("the healthy evaluator bootstraps and answers");
        // Let the buffered broadcast of t0 flush while the evaluator is still
        // fast, so nothing is left for the pre-query flush to wedge on.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // From here every frame the child is sent never comes back.
        ev.set_update_ms(PacedEvaluator::FOREVER_MS);

        // A min_sequence no marker will ever carry, so the task sits in the
        // fence loop for the whole test.
        let unreachable = store.session_sequence(&target) + 10_000;
        let fenced = {
            let map = Arc::clone(&map);
            let target = target.clone();
            tokio::spawn(async move {
                map.dispatch(&target, None, req_for(&target), unreachable)
                    .await
            })
        };
        // Give the job time to reach the fence.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Records for the TARGET, so the catch-up below has a batch to flush
        // and therefore an IPC to wedge on.
        store
            .merge_events(&target, None, vec![ev_for("t1"), ev_for("t2")])
            .unwrap();
        // Then overflow the store-wide ring (32768) from a different session
        // in one burst, so the target's next `recv()` inside the fence reports
        // a lag rather than another foreign update.
        let noisy = SessionScope::new(T, "NOISY");
        let flood: Vec<Event> = (0..40_000).map(|i| ev_for(&format!("n{i}"))).collect();
        store.merge_events(&noisy, None, flood).unwrap();

        let denied = tokio::time::timeout(Duration::from_secs(10), fenced)
            .await
            .expect("the fenced check must be answered, not hang")
            .expect("the dispatch task must not panic")
            .err()
            .expect("a check whose catch-up was abandoned fails closed");
        assert!(
            denied.contains("the replay catch-up"),
            "the deny must name the phase that actually overran — the fence's own deny \
             leaves no reply owed, so nothing would ever measure the child's silence. \
             Got: {denied}"
        );

        // And the reply now owed is what the stall window measures: the child
        // is still inside the abandoned frame, so it is killed.
        let waited = Instant::now();
        while ev.kills.load(Ordering::Relaxed) == 0 && waited.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            ev.kills.load(Ordering::Relaxed) >= 1,
            "the abandoned catch-up left a wedged child that nothing killed"
        );
    }

    /// GLOBAL scope reconciles from the durable store (the store-wide `Sequence`
    /// stream reorders across shards, so it can't be trusted as a fence-clearance
    /// signal), but the reload is TRIGGERED by the tenant's GLOBAL SHARD counter —
    /// the shard session-less writers (and the refmon proxy) land on. That counter
    /// moves only under its own shard's write lock, so it is reorder-free, and it
    /// bounds cost: unrelated sessions' traffic must NOT force a full-tenant reload
    /// on a path the refmon takes per request. Other sessions' shards are documented
    /// best-effort eventually-consistent for a global evaluator. This pins all three
    /// halves of the contract.
    #[tokio::test]
    async fn global_scope_resync_is_triggered_by_global_shard_not_unrelated_traffic() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let ev = LoggingEvaluator::new("g");
        let ev_f = Arc::clone(&ev);
        let handed = Arc::new(AtomicUsize::new(0));
        // Every handed-out mock, so step (4) can assert on the SESSION evaluator's own
        // counter rather than the global one's.
        let handles: Arc<PMutex<Vec<Arc<LoggingEvaluator>>>> = Arc::new(PMutex::new(Vec::new()));
        let handles_f = Arc::clone(&handles);
        // ONLY the first spawn — the global scope, dispatched below before anything
        // else touches the map — gets the instrumented mock. Handing the same Arc to
        // every scope would let another task bump `reset_count`, and one does exist:
        // the prewarm listener spawns an evaluator for each session that RECEIVES a
        // write, so S1 gets a task here without ever being dispatched.
        let factory: EvaluatorFactory = Arc::new(move || {
            let e = if handed.fetch_add(1, Ordering::Relaxed) == 0 {
                Arc::clone(&ev_f)
            } else {
                LoggingEvaluator::new("other")
            };
            handles_f.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });
        // Cap well above the scopes in play so LRU eviction can't masquerade as a resync.
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 8);
        map.swap_factory(factory, "mock".to_string());

        let global = SessionScope::new(T, ""); // empty session = tenant-global scope
                                               // Mirrors what evaluator_engine passes as `min_sequence` for ANY scope —
                                               // `session_sequence(scope)`, which for the global scope IS the global shard's
                                               // own counter. (This test drives `dispatch` directly, so it pins the gate
                                               // LOGIC; the assertions below additionally pin the counter's premise, which
                                               // is what makes that uniform call the right cost-bounding trigger.)
        let seq_of = |sc: &SessionScope| store.session_sequence(sc);
        map.dispatch(&global, None, req_for(&global), seq_of(&global))
            .await
            .unwrap();

        // (1) An UNRELATED session's write must not force a global reload. It bumps
        // that session's shard (and the store-wide counter), but not the global shard.
        store
            .merge_events(&SessionScope::new(T, "S1"), None, vec![ev_for("s1")])
            .unwrap();
        assert_eq!(
            seq_of(&global),
            0,
            "premise: an unrelated session's write must NOT move the global shard counter"
        );
        assert!(
            store.get_sequence() > 0,
            "…even though it does move the store-wide one"
        );
        let before_unrelated = ev.reset_count.load(Ordering::Relaxed);
        map.dispatch(&global, None, req_for(&global), seq_of(&global))
            .await
            .unwrap();
        assert_eq!(
            ev.reset_count.load(Ordering::Relaxed),
            before_unrelated,
            "an unrelated session's write must not reload the whole tenant (hot-path cost)"
        );

        // (2) A write on the GLOBAL shard — where a session-less/proxy caller lands —
        // must reload, so that caller reads its own writes.
        store
            .merge_events(&global, None, vec![ev_for("g1")])
            .unwrap();
        assert!(
            seq_of(&global) > 0,
            "premise: a global-shard write MUST move the global shard counter"
        );
        let before_global = ev.reset_count.load(Ordering::Relaxed);
        map.dispatch(&global, None, req_for(&global), seq_of(&global))
            .await
            .unwrap();
        // Pins that the instrumented mock really is the GLOBAL evaluator (it answered
        // the global scope's queries), not some other scope's — the counter assertions
        // above are meaningless if that mapping ever drifts.
        {
            let qs = ev.queries.lock();
            // NOT just `all(...)`: that is vacuously true on an empty vec, which is
            // exactly what the drift being guarded against produces (the instrumented
            // mock landing on a scope nothing ever dispatches).
            assert!(
                !qs.is_empty(),
                "the instrumented mock must have answered queries"
            );
            assert!(
                qs.iter().all(|q| q.as_deref().unwrap_or("").is_empty()),
                "the instrumented mock must be the GLOBAL evaluator, not another scope's"
            );
        }
        assert!(
            ev.reset_count.load(Ordering::Relaxed) > before_global,
            "a global-shard write must reconcile from the store (read-your-writes)"
        );
        // The reload carries the whole tenant, i.e. the unrelated session's event too.
        let loaded: usize = ev.updates.lock().iter().sum();
        assert!(
            loaded >= 2,
            "global reload should carry the full tenant state, saw {loaded}"
        );

        // (3) Repeat with nothing new — no reload.
        let steady = ev.reset_count.load(Ordering::Relaxed);
        map.dispatch(&global, None, req_for(&global), seq_of(&global))
            .await
            .unwrap();
        assert_eq!(
            ev.reset_count.load(Ordering::Relaxed),
            steady,
            "an unchanged global shard must not trigger another reload"
        );

        // (4) A SESSION-scope query with an already-satisfied fence keeps the cheap
        // incremental path (no resync).
        let sess = SessionScope::new(T, "S1");
        map.dispatch(&sess, None, req_for(&sess), seq_of(&sess))
            .await
            .unwrap();
        // S1 runs on its OWN mock (the instrumented one is reserved for the global
        // scope), so assert on that instance. Asserting on `ev` here would be
        // trivially true — the global evaluator isn't dispatched between the two
        // reads, so its counter cannot move no matter what the session path does.
        let s1_ev = handles
            .lock()
            .iter()
            .find(|h| h.queries.lock().iter().any(|q| q.as_deref() == Some("S1")))
            .cloned()
            .expect("S1's evaluator should have answered a query (a resync would have cleared it)");
        let after_session_spawn = s1_ev.reset_count.load(Ordering::Relaxed);
        map.dispatch(&sess, None, req_for(&sess), seq_of(&sess))
            .await
            .unwrap();
        assert_eq!(
            s1_ev.reset_count.load(Ordering::Relaxed),
            after_session_spawn,
            "a satisfied session-scope fence must stay on the incremental fast path"
        );
    }

    /// Mock that flips to ProcessDied on demand so we can verify
    /// dead-subprocess recovery without spawning a real evaluator.
    struct FlakyEvaluator {
        label: String,
        dead: Arc<std::sync::atomic::AtomicBool>,
    }
    impl FlakyEvaluator {
        fn new(label: &str) -> (Arc<Self>, Arc<std::sync::atomic::AtomicBool>) {
            let dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let e = Arc::new(Self {
                label: label.into(),
                dead: Arc::clone(&dead),
            });
            (e, dead)
        }
    }
    #[tonic::async_trait]
    impl Evaluator for FlakyEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            if self.dead.load(Ordering::Relaxed) {
                Err(EvaluatorError::ProcessDied)
            } else {
                Ok(())
            }
        }
        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            if self.dead.load(Ordering::Relaxed) {
                Err(EvaluatorError::ProcessDied)
            } else {
                Ok(EvalAuthResponse {
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
        }
        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }
        fn backend_name(&self) -> &str {
            &self.label
        }
    }

    /// After the evaluator subprocess dies the next dispatch evicts
    /// the live handle, and the dispatch after that respawns a fresh
    /// evaluator from graph state. A `ProcessDied` entry must not stay
    /// in the map, or every later dispatch returns the same error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dispatch_recovers_after_subprocess_dies() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let dead_flags: Arc<PMutex<Vec<Arc<std::sync::atomic::AtomicBool>>>> =
            Arc::new(PMutex::new(Vec::new()));
        let dead_flags_for_factory = Arc::clone(&dead_flags);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            let (e, dead) = FlakyEvaluator::new(&format!("flaky{n}"));
            dead_flags_for_factory.lock().push(dead);
            Ok(e as Arc<dyn Evaluator>)
        });
        // Idle-TTL sweep disabled so eviction is purely caused by
        // the ProcessDied path under test.
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "flaky".to_string());

        let scope = SessionScope::new(T, "S");
        // First dispatch: live, succeeds.
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        // Kill the only subprocess. The next dispatch should report
        // the failure, evict the dead handle, and the one after that
        // should respawn cleanly.
        dead_flags.lock()[0].store(true, Ordering::Relaxed);
        let _err = map.dispatch(&scope, None, req_for(&scope), 0).await;

        // Give the task a tick to exit and set process_died.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let after = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert!(
            after.is_ok(),
            "dispatch after dead-process eviction should succeed; got err={:?}",
            after.err(),
        );
        assert!(
            counter.load(Ordering::Relaxed) >= 2,
            "factory should have been invoked a second time to respawn",
        );
    }

    /// With ``max_live_sessions = N``, the (N+1)th distinct session
    /// triggers LRU eviction of the least-recently-active entry.
    /// Live count plateaus at the cap; an evicted session that
    /// resumes re-bootstraps from the graph store (counter increments).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lru_cap_evicts_oldest_when_at_capacity() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        // Idle-TTL sweep disabled so the cap is the only eviction
        // signal under test.
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            Some(2),
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let s1 = SessionScope::new(T, "S1");
        let s2 = SessionScope::new(T, "S2");
        let s3 = SessionScope::new(T, "S3");

        // Spawn S1 (will become LRU), then S2.
        map.dispatch(&s1, None, req_for(&s1), 0).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        map.dispatch(&s2, None, req_for(&s2), 0).await.unwrap();
        assert_eq!(map.live_sessions(), 2);
        assert_eq!(counter.load(Ordering::Relaxed), 2);

        // Touch S2 again so S1 is the unambiguous LRU.
        tokio::time::sleep(Duration::from_millis(5)).await;
        map.dispatch(&s2, None, req_for(&s2), 0).await.unwrap();

        // Spawn S3 — at cap, S1 must be evicted to make room.
        map.dispatch(&s3, None, req_for(&s3), 0).await.unwrap();
        assert_eq!(map.live_sessions(), 2, "live count plateaus at cap");
        assert_eq!(counter.load(Ordering::Relaxed), 3);

        // Resuming S1 re-spawns (counter increments) and evicts the
        // new LRU (S2: last touched at t≈10ms, vs S3 at t≈15ms).
        map.dispatch(&s1, None, req_for(&s1), 0).await.unwrap();
        assert_eq!(map.live_sessions(), 2);
        assert_eq!(
            counter.load(Ordering::Relaxed),
            4,
            "evicted S1 re-spawns from store rather than reusing the dead Arc",
        );
    }

    /// Cross-tenant evaluators must not share broadcasts. Two
    /// scopes with the same session string but different tenants
    /// (acme/s1 vs orgb/s1) are distinct partitions; events
    /// pushed under one tenant must not leak to the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_tenant_evaluators_do_not_share_broadcasts() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let evaluators: Arc<PMutex<Vec<Arc<LoggingEvaluator>>>> = Arc::new(PMutex::new(Vec::new()));
        let evals_clone = Arc::clone(&evaluators);
        let factory: EvaluatorFactory = Arc::new(move || {
            let label = format!("e{}", evals_clone.lock().len());
            let e = LoggingEvaluator::new(&label);
            evals_clone.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.install_policy_default_for_tenant("acme", Arc::clone(&factory), "mock".to_string());
        map.install_policy_default_for_tenant("orgb", factory, "mock".to_string());

        let acme_s1 = SessionScope::new("acme", "s1");
        let orgb_s1 = SessionScope::new("orgb", "s1");

        // Touch each scope so its evaluator spawns + bootstraps
        // (against an empty store at this point).
        map.dispatch(&acme_s1, None, req_for(&acme_s1), 0)
            .await
            .unwrap();
        map.dispatch(&orgb_s1, None, req_for(&orgb_s1), 0)
            .await
            .unwrap();

        // Push events under acme/s1 only.
        store
            .merge_events(&acme_s1, None, vec![ev_for("a1"), ev_for("a2")])
            .unwrap();

        let acme_seq = store.session_sequence(&acme_s1);
        // Bump each evaluator past its own fence so any pending
        // updates have flushed into the LoggingEvaluator.
        map.dispatch(&acme_s1, None, req_for(&acme_s1), acme_seq)
            .await
            .unwrap();
        let orgb_seq = store.session_sequence(&orgb_s1);
        map.dispatch(&orgb_s1, None, req_for(&orgb_s1), orgb_seq)
            .await
            .unwrap();

        let evals = evaluators.lock();
        assert_eq!(evals.len(), 2);
        let acme_eval = &evals[0];
        let orgb_eval = &evals[1];

        let acme_total: usize = acme_eval.updates.lock().iter().sum();
        let orgb_total: usize = orgb_eval.updates.lock().iter().sum();
        assert!(
            acme_total >= 2,
            "acme evaluator should see its 2 events, got {}",
            acme_total
        );
        // orgb evaluator must NOT see acme's events. Bootstrap was
        // empty (no events under orgb yet), so the only updates it
        // could have received would be cross-tenant leakage.
        assert_eq!(
            orgb_total, 0,
            "orgb evaluator must not receive acme's events; got {} updates",
            orgb_total,
        );
    }

    /// A `SessionScope::global(tenant)` evaluator subscribes to
    /// every session within its tenant — but never to other
    /// tenants. Pushing events under acme/s1 and acme/s2 both
    /// reach the acme-global evaluator; an orgb-global evaluator
    /// sees neither.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tenant_global_scope_sees_all_sessions_in_tenant() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let evaluators: ScopedEvaluators = Arc::new(PMutex::new(Vec::new()));
        // The factory needs to know which scope it's spawning for.
        // We capture the scope by inspecting the dispatch sequence:
        // each call to the factory corresponds to the most recent
        // `get_or_spawn` invocation. Since the prewarm listener
        // also fires, we just label evaluators in spawn order and
        // resolve them after by querying the map.
        let evals_clone = Arc::clone(&evaluators);
        let factory: EvaluatorFactory = Arc::new(move || {
            let label = format!("e{}", evals_clone.lock().len());
            let e = LoggingEvaluator::new(&label);
            // We push a placeholder scope; updated by the
            // post-test inspector below.
            evals_clone
                .lock()
                .push((SessionScope::global("placeholder"), Arc::clone(&e)));
            Ok(e as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.install_policy_default_for_tenant("acme", Arc::clone(&factory), "mock".to_string());
        map.install_policy_default_for_tenant("orgb", factory, "mock".to_string());

        let acme_global = SessionScope::global("acme");
        let orgb_global = SessionScope::global("orgb");

        // Spawn each global evaluator.
        map.dispatch(&acme_global, None, req_for(&acme_global), 0)
            .await
            .unwrap();
        map.dispatch(&orgb_global, None, req_for(&orgb_global), 0)
            .await
            .unwrap();

        // Push events under acme/s1 and acme/s2. The prewarm
        // listener may also spawn per-session evaluators for those,
        // but they're not what we're measuring here. Two writes
        // now (one envelope = one session).
        let acme_s1 = SessionScope::new("acme", "s1");
        let acme_s2 = SessionScope::new("acme", "s2");
        store
            .merge_events(&acme_s1, None, vec![ev_for("a1")])
            .unwrap();
        store
            .merge_events(&acme_s2, None, vec![ev_for("a2")])
            .unwrap();

        // Give the broadcast a chance to reach each evaluator.
        // (The acme/s1 SessionSequence advances acme_global's
        // fence per matches() since global matches every session
        // in the tenant.)
        let acme_s1_seq = store.session_sequence(&SessionScope::new("acme", "s1"));
        let acme_s2_seq = store.session_sequence(&SessionScope::new("acme", "s2"));
        let acme_fence = acme_s1_seq.max(acme_s2_seq);
        map.dispatch(&acme_global, None, req_for(&acme_global), acme_fence)
            .await
            .unwrap();
        // Settle orgb side too.
        let orgb_seq = store.session_sequence(&orgb_global);
        map.dispatch(&orgb_global, None, req_for(&orgb_global), orgb_seq)
            .await
            .unwrap();

        // Brief settle for any prewarm-spawned evaluators to
        // process their cross-tenant noise (or rather: lack of it).
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Identify the global evaluators by their spawn order
        // (acme global first, orgb global second). Prewarm-spawned
        // s1/s2 evaluators come after.
        let evals = evaluators.lock();
        assert!(
            evals.len() >= 2,
            "expected at least 2 evaluators (got {})",
            evals.len()
        );
        let acme_eval = &evals[0].1;
        let orgb_eval = &evals[1].1;

        let acme_total: usize = acme_eval.updates.lock().iter().sum();
        let orgb_total: usize = orgb_eval.updates.lock().iter().sum();

        assert!(
            acme_total >= 2,
            "acme/global evaluator should see both s1 + s2 events, got {}",
            acme_total,
        );
        assert_eq!(
            orgb_total, 0,
            "orgb/global must not see acme's events; got {} updates",
            orgb_total,
        );
    }

    /// Mock evaluator that captures the actual `GraphUpdate`s it
    /// receives (node ids + edge src/dst/session_id), not just batch
    /// sizes — so a test can assert on cross-session attribution.
    struct RecordingEvaluator {
        nodes: PMutex<std::collections::HashSet<String>>,
        edges: PMutex<Vec<(String, String, String)>>, // (src, dst, session_id)
    }
    impl RecordingEvaluator {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                nodes: PMutex::new(std::collections::HashSet::new()),
                edges: PMutex::new(Vec::new()),
            })
        }
        fn node_ids(&self) -> std::collections::HashSet<String> {
            self.nodes.lock().clone()
        }
        fn edges(&self) -> Vec<(String, String, String)> {
            self.edges.lock().clone()
        }
    }
    #[tonic::async_trait]
    impl Evaluator for RecordingEvaluator {
        async fn update(&self, updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            let mut nodes = self.nodes.lock();
            let mut edges = self.edges.lock();
            for u in &updates {
                match u {
                    GraphUpdate::NodeCreated { id, .. } => {
                        nodes.insert(id.clone());
                    }
                    GraphUpdate::EdgeCreated {
                        source,
                        destination,
                        session_id,
                        ..
                    } => edges.push((source.clone(), destination.clone(), session_id.clone())),
                    _ => {}
                }
            }
            Ok(())
        }
        async fn query(&self, _req: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            Ok(EvalAuthResponse {
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
        async fn reset(&self) -> Result<(), EvaluatorError> {
            self.nodes.lock().clear();
            self.edges.lock().clear();
            Ok(())
        }
        fn backend_name(&self) -> &str {
            "recording"
        }
    }

    /// Global-session BOOTSTRAP path (c5): when graph state —
    /// including dependency edges — already exists across several
    /// non-global sessions BEFORE any global evaluator spawns, a
    /// global-scope query must bootstrap from ALL the tenant's
    /// shards, not just the `(tenant, "")` global shard. This
    /// complements `tenant_global_scope_sees_all_sessions_in_tenant`,
    /// which exercises the live broadcast path (global evaluator
    /// spawned first, events pushed after).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn global_session_bootstrap_aggregates_cross_session_dependencies() {
        use sasy_common::observability::Edge;

        let store = Arc::new(GraphStore::new(None).unwrap());

        // Recorder factory: each spawned evaluator captures the full
        // update batches it receives so we can confirm the global
        // evaluator saw cross-session nodes + edges with attribution.
        let recorders: Arc<PMutex<Vec<Arc<RecordingEvaluator>>>> =
            Arc::new(PMutex::new(Vec::new()));
        let rc = Arc::clone(&recorders);
        let factory: EvaluatorFactory = Arc::new(move || {
            let e = RecordingEvaluator::new();
            rc.lock().push(Arc::clone(&e));
            Ok(e as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.install_policy_default_for_tenant("acme", factory, "mock".to_string());

        // Manually build dependency chains in TWO non-global sessions
        // BEFORE any global evaluator exists: a1->a2 in s1, b1->b2 in
        // s2. Edges are intra-session; the global view aggregates
        // across sessions.
        let s1 = SessionScope::new("acme", "s1");
        let s2 = SessionScope::new("acme", "s2");
        let edge = |src: &str, dst: &str| Edge {
            source: src.to_string(),
            destination: dst.to_string(),
            message_index: None,
            proximal: None,
            principal: None,
            entity: None,
        };
        store
            .merge_events_with_dependencies(
                &s1,
                None,
                vec![ev_for("a1"), ev_for("a2")],
                vec![edge("a1", "a2")],
            )
            .unwrap();
        store
            .merge_events_with_dependencies(
                &s2,
                None,
                vec![ev_for("b1"), ev_for("b2")],
                vec![edge("b1", "b2")],
            )
            .unwrap();

        // The global trial: a global-scope query spawns a FRESH global
        // evaluator (no global-shard events were pushed, so the
        // prewarm listener never spawned one) that must bootstrap the
        // whole tenant.
        let global = SessionScope::global("acme");
        let fence = store.session_sequence(&global);
        let result = map
            .dispatch(&global, None, req_for(&global), fence)
            .await
            .unwrap();

        // The global evaluator's own counts reflect a bootstrap over
        // every tenant shard: 4 nodes (a1,a2,b1,b2) + 2 dependency
        // edges. Pre-c5 this read only the empty global shard → 0/0.
        assert_eq!(
            result.graph_nodes, 4,
            "global bootstrap must load all sessions' nodes"
        );
        assert_eq!(
            result.graph_edges, 2,
            "global bootstrap must load all sessions' dependency edges"
        );

        // Let any prewarm-spawned per-session evaluators settle.
        tokio::time::sleep(Duration::from_millis(10)).await;

        // The global evaluator is the only one whose received updates
        // reference nodes from BOTH sessions; per-session evaluators
        // each see just their own shard.
        let recs = recorders.lock();
        let global_rec = recs
            .iter()
            .find(|r| {
                let ids = r.node_ids();
                ids.contains("a1") && ids.contains("b1")
            })
            .expect("a global evaluator that bootstrapped both sessions");

        // Each dependency reached the global evaluator tagged with its
        // OWN session (attribution preserved, not flattened). Endpoint
        // ordering follows the store's DependsOn convention (tested
        // elsewhere), so match the pair unordered — what matters here
        // is presence + per-session attribution.
        let edges = global_rec.edges();
        let has_dep = |x: &str, y: &str, sid: &str| {
            edges
                .iter()
                .any(|(s, d, ss)| ss == sid && ((s == x && d == y) || (s == y && d == x)))
        };
        assert!(
            has_dep("a1", "a2", "s1"),
            "s1 dependency must reach the global evaluator tagged session=s1; got {edges:?}"
        );
        assert!(
            has_dep("b1", "b2", "s2"),
            "s2 dependency must reach the global evaluator tagged session=s2; got {edges:?}"
        );
    }

    // ────────────────────────────────────────────────────────────────
    // Integration tests for LRU + multi-policy + tenant-separation
    // interactions
    // ────────────────────────────────────────────────────────────────

    /// LRU eviction must not break the session→policy binding:
    /// an evicted session that resumes after eviction comes
    /// back under the same policy variant it was originally bound
    /// to, even though its evaluator process has been torn down and
    /// re-spawned in between. Without this, a tenant could quietly
    /// have its session "fall back" to the default policy when
    /// transient memory pressure evicted the evaluator.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multi_policy_lru_swap_preserves_policy_binding() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO, // disable idle TTL — only LRU matters here
            Duration::ZERO,
            Some(2), // hard cap = 2 live sessions
            EvaluationDeadlines::default(),
        );

        // Two policies installed in the same tenant; one default,
        // one variant. The factory stamps each spawned evaluator
        // with a label tied to the policy it was made from, so a
        // re-spawned evaluator's label tells us which factory the
        // map called — proving the binding survived eviction.
        let default_factory: EvaluatorFactory =
            Arc::new(|| Ok(LoggingEvaluator::new("p_default") as Arc<dyn Evaluator>));
        let variant_factory: EvaluatorFactory =
            Arc::new(|| Ok(LoggingEvaluator::new("p_variant") as Arc<dyn Evaluator>));
        let _default_id = map
            .registry()
            .install(T, default_factory, "souffle".into(), true);
        let variant_id = map
            .registry()
            .install(T, variant_factory, "souffle".into(), false);

        let s1 = SessionScope::new(T, "s1");
        let s2 = SessionScope::new(T, "s2");
        let s3 = SessionScope::new(T, "s3");

        // S1 binds to the variant; S2 and S3 to the default.
        map.dispatch(&s1, Some(&variant_id), req_for(&s1), 0)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        map.dispatch(&s2, None, req_for(&s2), 0).await.unwrap();
        assert_eq!(map.live_sessions(), 2);

        // Touching S3 forces LRU eviction; S1 (oldest) goes.
        tokio::time::sleep(Duration::from_millis(2)).await;
        map.dispatch(&s3, None, req_for(&s3), 0).await.unwrap();
        assert_eq!(map.live_sessions(), 2, "live count plateaus at cap");
        assert_eq!(
            map.current_policy(&s1),
            Some(variant_id.clone()),
            "S1's policy binding survived LRU eviction",
        );

        // Resume S1 — it should re-spawn under the *variant* factory
        // (not the default) because the binding persisted in
        // session_to_policy across eviction.
        map.dispatch(&s1, None, req_for(&s1), 0).await.unwrap();
        let backend = map.current_policy(&s1);
        assert_eq!(
            backend,
            Some(variant_id),
            "S1 resumed under the same variant after LRU eviction",
        );
    }

    /// Sessions in flight concurrently must each see only their own
    /// updates. This is the core multi-session enforcement
    /// invariant: a busy server with many simultaneous agents must
    /// not let one session's writes appear in another session's
    /// evaluator state, regardless of how the broadcasts interleave.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_sessions_dispatch_in_parallel() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let evaluators: ScopedEvaluators = Arc::new(PMutex::new(Vec::new()));
        let evals_for_factory = Arc::clone(&evaluators);
        let factory: EvaluatorFactory = Arc::new(move || {
            let label = format!("eval-{}", evals_for_factory.lock().len());
            let e = LoggingEvaluator::new(&label);
            evals_for_factory.lock().push((
                // Placeholder scope, filled in after the spawn
                // settles below by matching label order.
                SessionScope::new("default", ""),
                Arc::clone(&e),
            ));
            Ok(e as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            8,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        for t in ["acme", "orgb", "default"] {
            map.install_policy_default_for_tenant(t, Arc::clone(&factory), "mock".to_string());
        }

        // 12 sessions across 3 tenants × 4 sessions. Push
        // tenant-distinct events into the store concurrently; then
        // dispatch each session under its scope.
        let scopes: Vec<SessionScope> = ["acme", "orgb", "default"]
            .iter()
            .flat_map(|t| (0..4).map(move |i| SessionScope::new(*t, format!("s{i}"))))
            .collect();

        // Pre-warm by dispatching each scope so its evaluator
        // exists and subscribes before we push events.
        for scope in &scopes {
            map.dispatch(scope, None, req_for(scope), 0).await.unwrap();
        }
        assert_eq!(map.live_sessions(), 12);

        // Push 3 events into each scope concurrently.
        let mut handles = Vec::new();
        for scope in &scopes {
            let store = Arc::clone(&store);
            let scope = scope.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..3 {
                    let id = format!("{}-{}-{}", scope.tenant(), scope.session(), i);
                    let _ = store.merge_events(&scope, None, vec![ev_for(&id)]);
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        // Bump each evaluator past its scope-local seq so it has
        // drained its own broadcasts.
        for scope in &scopes {
            let seq = store.session_sequence(scope);
            map.dispatch(scope, None, req_for(scope), seq)
                .await
                .unwrap();
        }
        // Brief settle so background flushes complete.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Every evaluator should see *only* its own scope's events.
        // We compare update batch sizes summed: each scope pushed 3
        // events; an evaluator that's leaking would show >3.
        let evals = evaluators.lock();
        for (_, e) in evals.iter() {
            let total: usize = e.updates.lock().iter().sum();
            assert!(
                total <= 3 + 1, // +1 for any bootstrap update
                "{} saw {} updates — a leak from another scope?",
                e.label,
                total,
            );
        }
    }

    /// Cross-tenant probe: the user's specific concern — a policy
    /// that depends on a "global property" (here, the count of
    /// nodes visible to the evaluator) must produce different
    /// results in tenant A versus tenant B because each only sees
    /// its own subgraph. If isolation broke, the property would
    /// flip identically across tenants.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_tenant_isolation_under_global_property_probe() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let evaluators: ScopedEvaluators = Arc::new(PMutex::new(Vec::new()));
        let evals_for_factory = Arc::clone(&evaluators);
        let factory: EvaluatorFactory = Arc::new(move || {
            let label = format!("eval-{}", evals_for_factory.lock().len());
            let e = LoggingEvaluator::new(&label);
            evals_for_factory
                .lock()
                .push((SessionScope::new("default", ""), Arc::clone(&e)));
            Ok(e as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.install_policy_default_for_tenant("acme", Arc::clone(&factory), "mock".to_string());
        map.install_policy_default_for_tenant("orgb", factory, "mock".to_string());

        let acme_s1 = SessionScope::new("acme", "s1");
        let orgb_s1 = SessionScope::new("orgb", "s1");

        // Each tenant lights up its session evaluator.
        map.dispatch(&acme_s1, None, req_for(&acme_s1), 0)
            .await
            .unwrap();
        map.dispatch(&orgb_s1, None, req_for(&orgb_s1), 0)
            .await
            .unwrap();

        // Tenant acme writes 10 nodes; tenant orgb writes 1.
        // Under correct (tenant, session) sharding, a query in each
        // sees exactly its own writes.
        for i in 0..10 {
            store
                .merge_events(&acme_s1, None, vec![ev_for(&format!("a-{i}"))])
                .unwrap();
        }
        store
            .merge_events(&orgb_s1, None, vec![ev_for("b-0")])
            .unwrap();

        let acme_seq = store.session_sequence(&acme_s1);
        let orgb_seq = store.session_sequence(&orgb_s1);
        map.dispatch(&acme_s1, None, req_for(&acme_s1), acme_seq)
            .await
            .unwrap();
        map.dispatch(&orgb_s1, None, req_for(&orgb_s1), orgb_seq)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Verify per-evaluator update counts: acme saw 10, orgb 1.
        // The "global property" the policy evaluator would compute
        // (e.g. `NodeCount > 5`) flips for acme but not for orgb.
        let evals = evaluators.lock();
        let acme_eval = &evals[0].1;
        let orgb_eval = &evals[1].1;
        let acme_total: usize = acme_eval.updates.lock().iter().sum();
        let orgb_total: usize = orgb_eval.updates.lock().iter().sum();
        assert!(
            acme_total >= 10,
            "acme should see all 10 of its events, got {}",
            acme_total,
        );
        assert!(
            orgb_total <= 2,
            "orgb must not see acme's events; got {}",
            orgb_total,
        );
        assert!(
            acme_total > orgb_total + 5,
            "tenant-distinguishing property must differ across tenants \
             ({} vs {})",
            acme_total,
            orgb_total,
        );
    }

    /// Explicit policy switch via `set_session_policy` evicts the
    /// running evaluator and re-spawns under the new policy on the
    /// next dispatch. The graph state survives so the new evaluator
    /// bootstraps with the same EDB.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn set_session_policy_evicts_and_rebinds() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let v1_count = Arc::new(AtomicUsize::new(0));
        let v2_count = Arc::new(AtomicUsize::new(0));
        let v1c = Arc::clone(&v1_count);
        let v2c = Arc::clone(&v2_count);
        let v1_factory: EvaluatorFactory = Arc::new(move || {
            v1c.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new("v1") as Arc<dyn Evaluator>)
        });
        let v2_factory: EvaluatorFactory = Arc::new(move || {
            v2c.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new("v2") as Arc<dyn Evaluator>)
        });

        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        // v1 is the tenant default; v2 is a variant.
        let _v1 = map
            .registry()
            .install(T, v1_factory, "souffle".into(), true);
        let v2 = map
            .registry()
            .install(T, v2_factory, "souffle".into(), false);

        let scope = SessionScope::new(T, "s1");

        // Dispatch under default — v1 spawns.
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(v1_count.load(Ordering::Relaxed), 1);
        assert_eq!(v2_count.load(Ordering::Relaxed), 0);

        // Switch the binding; eviction is reported by set_session_policy.
        let evicted = map.set_session_policy(&scope, &v2).unwrap();
        assert!(evicted, "live evaluator was evicted by the rebind");
        assert_eq!(map.current_policy(&scope), Some(v2.clone()));

        // Next dispatch re-spawns under v2 — v2 factory called once.
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(v1_count.load(Ordering::Relaxed), 1, "v1 not re-spawned");
        assert_eq!(
            v2_count.load(Ordering::Relaxed),
            1,
            "v2 spawned after rebind"
        );

        // Sending the *old* policy_id mid-session is rejected to
        // prevent split-brain — set_session_policy is the only way
        // to switch.
        let old_v1_dispatch = map.dispatch(&scope, Some(&_v1), req_for(&scope), 0).await;
        assert!(
            old_v1_dispatch.is_err(),
            "post-rebind dispatch with the old policy_id must error",
        );
    }

    /// Concurrent first requests for one scope spawn one evaluator.
    ///
    /// Six callers are released at a single instant by a barrier and
    /// all ask for the same cold scope; the factory then holds its
    /// caller long enough that every other one is certainly inside
    /// `get_or_spawn` while the spawn is in flight. Exactly one
    /// factory call, and every caller comes back with that one
    /// evaluator.
    ///
    /// Without the gate: 6 factory calls, five of them surplus
    /// subprocesses that are spawned, bootstrapped and dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_first_requests_for_one_scope_spawn_one_evaluator() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::SeqCst);
            // Hold the spawn open: every other caller has already been
            // released by the barrier, so all of them are inside
            // `get_or_spawn` before this one returns.
            std::thread::sleep(Duration::from_millis(200));
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());
        let scope = SessionScope::new(T, "contested");

        const CALLERS: usize = 6;
        let barrier = Arc::new(std::sync::Barrier::new(CALLERS));
        let handle = tokio::runtime::Handle::current();
        let threads: Vec<_> = (0..CALLERS)
            .map(|_| {
                let map = Arc::clone(&map);
                let scope = scope.clone();
                let barrier = Arc::clone(&barrier);
                let handle = handle.clone();
                std::thread::spawn(move || {
                    // `SessionEvaluator::spawn` needs a runtime to
                    // spawn its task onto; these are plain OS threads
                    // so that the barrier release is simultaneous.
                    let _rt = handle.enter();
                    barrier.wait();
                    map.get_or_spawn(&scope, None).map(|(se, _)| se)
                })
            })
            .collect();
        let evaluators: Vec<Arc<SessionEvaluator>> = threads
            .into_iter()
            .map(|t| {
                t.join()
                    .unwrap()
                    .expect("every caller must come back with an evaluator")
            })
            .collect();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "a scope's evaluator is spawned once however many callers race for it"
        );
        for se in &evaluators {
            assert!(
                Arc::ptr_eq(se, &evaluators[0]),
                "every caller must be handed the one evaluator that was spawned"
            );
        }
        assert_eq!(map.live_sessions(), 1);
    }

    /// The prewarm listener never competes with a request: while a
    /// dispatch is spawning a scope's evaluator, the prewarm call for
    /// that scope returns immediately instead of spawning a second
    /// one, and the pair costs one factory call.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_prewarm_alongside_a_dispatch_spawns_one_evaluator() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "warm");

        // The first factory call blocks until released; later ones do not.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(PMutex::new(release_rx));
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().recv();
            }
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        // A dispatch, held inside the factory.
        let dispatcher = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), 0).await })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .unwrap()
            .expect("the dispatch must reach the factory");

        // The prewarm listener's own call, driven directly. It must
        // return at once rather than wait for the compile.
        let started = Instant::now();
        let prewarm = map.try_get_or_spawn(&scope, None);
        assert!(
            prewarm.is_err(),
            "prewarm must skip a scope whose spawn is already in flight"
        );
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "prewarm must not block behind the in-flight spawn"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "prewarm must not start a second spawn for the scope"
        );

        let _ = release_tx.send(());
        dispatcher.await.unwrap().unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "the dispatch and the prewarm together cost one spawn"
        );

        // And once the entry exists, prewarm takes it without spawning.
        assert!(map.try_get_or_spawn(&scope, None).is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(map.live_sessions(), 1);
    }

    /// After the subprocess dies, concurrent dispatchers produce one
    /// respawn between them — the respawn goes through the same
    /// per-scope gate as a cold spawn. Without the gate: seven factory
    /// calls for one death and six dispatchers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn after_a_death_concurrent_dispatchers_respawn_once() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let dead_flags: Arc<PMutex<Vec<Arc<std::sync::atomic::AtomicBool>>>> =
            Arc::new(PMutex::new(Vec::new()));
        let dead_flags_for_factory = Arc::clone(&dead_flags);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::SeqCst);
            if n > 0 {
                // Hold the respawn open so every other dispatcher is
                // certainly inside `get_or_spawn` while it runs.
                std::thread::sleep(Duration::from_millis(200));
            }
            let (e, dead) = FlakyEvaluator::new(&format!("flaky{n}"));
            dead_flags_for_factory.lock().push(dead);
            Ok(e as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "flaky".to_string());

        let scope = SessionScope::new(T, "S");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Kill it. A handle is only *known* dead once a dispatch has
        // failed on it, so one dispatch has to observe the death and
        // drop the handle before the racers below can see the empty
        // map and fall through to the respawn together. That dispatch
        // is awaited, so its eviction is ordered before them either
        // way; an eviction that lands *during* the respawn is the
        // harder case and has its own test below.
        dead_flags.lock()[0].store(true, Ordering::Relaxed);
        let _err = map.dispatch(&scope, None, req_for(&scope), 0).await;

        const DISPATCHERS: usize = 6;
        let mut tasks = Vec::new();
        for _ in 0..DISPATCHERS {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tasks.push(tokio::spawn(async move {
                map.dispatch(&scope, None, req_for(&scope), 0).await
            }));
        }
        for t in tasks {
            t.await.unwrap().expect("every dispatcher must be answered");
        }

        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "a death costs exactly one respawn, however many dispatchers see it"
        );
        assert_eq!(map.live_sessions(), 1);
    }

    /// An eviction that lands while a respawn is in flight must not
    /// start a second respawn.
    ///
    /// This is the shape a death actually produces: `dispatch` ends
    /// with `evict_evaluator_only` for every dispatcher that was
    /// holding the dead handle, so those evictions arrive while one
    /// dispatcher is already inside the factory. If an eviction could
    /// drop the gate entry the respawner is holding, the next
    /// dispatcher would make a fresh lock and run the factory
    /// alongside it. Without the refcount check in `SpawnGate::forget`:
    /// three factory calls for one death.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn an_evict_during_a_respawn_does_not_start_a_second_one() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let dead_flags: Arc<PMutex<Vec<Arc<std::sync::atomic::AtomicBool>>>> =
            Arc::new(PMutex::new(Vec::new()));
        let dead_flags_for_factory = Arc::clone(&dead_flags);

        // The respawn (the second factory call) is held open until the
        // test releases it, so the eviction and the next dispatcher
        // are certainly inside its window.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(PMutex::new(release_rx));
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::SeqCst);
            if n == 1 {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().recv();
            }
            let (e, dead) = FlakyEvaluator::new(&format!("flaky{n}"));
            dead_flags_for_factory.lock().push(dead);
            Ok(e as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "flaky".to_string());

        let scope = SessionScope::new(T, "S");
        map.dispatch(&scope, None, req_for(&scope), 0)
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        dead_flags.lock()[0].store(true, Ordering::Relaxed);
        // A handle is only known dead once a dispatch has failed on
        // it; this is that dispatch. It drops the handle without
        // respawning — it had already taken the evaluator before the
        // call failed.
        let _err = map.dispatch(&scope, None, req_for(&scope), 0).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Dispatcher A takes the respawn and is held inside the factory.
        let a = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), 0).await })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .unwrap()
            .expect("the respawn must reach the factory");

        // What a second dispatcher holding the same dead handle does
        // at the end of its own `dispatch`.
        map.evict_evaluator_only(&scope);

        // Dispatcher C arrives during the respawn. It must wait for
        // A's spawn, not start its own.
        let c = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::spawn(async move { map.dispatch(&scope, None, req_for(&scope), 0).await })
        };
        // Give C time to run a factory call of its own if the gate let
        // it: without the gate the third call lands within a few ms.
        for _ in 0..50 {
            if counter.load(Ordering::SeqCst) > 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "an eviction during a respawn must not release the gate to a second spawner"
        );

        let _ = release_tx.send(());
        a.await.unwrap().expect("dispatcher A must be answered");
        c.await.unwrap().expect("dispatcher C must be answered");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "one death costs one respawn even with an eviction interleaved"
        );
        assert_eq!(map.live_sessions(), 1);
    }

    /// A wedged factory must not park every dispatcher for the scope
    /// forever: a caller that has waited out the spawn deadline fails
    /// closed with an error naming the scope and the wait.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_waiting_caller_fails_closed_when_the_spawn_deadline_elapses() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new(T, "wedged");

        // The first factory call never returns until the test releases it.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(PMutex::new(release_rx));
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_factory = Arc::clone(&calls);
        let factory: EvaluatorFactory = Arc::new(move || {
            if calls_for_factory.fetch_add(1, Ordering::SeqCst) == 0 {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().recv();
            }
            Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>)
        });
        // Short deadline via the test seam: the production wait is 30 s.
        let deadline = Duration::from_millis(300);
        let map = SessionEvaluatorMap::with_spawn_deadline(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            deadline,
        );
        map.swap_factory(factory, "mock".to_string());

        let wedged = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            let handle = tokio::runtime::Handle::current();
            std::thread::spawn(move || {
                // Once released it finishes the spawn, which needs a
                // runtime to put the evaluator's task on.
                let _rt = handle.enter();
                let _ = map.get_or_spawn(&scope, None);
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv())
            .await
            .unwrap()
            .expect("the first caller must reach the factory");

        let waiter = {
            let map = Arc::clone(&map);
            let scope = scope.clone();
            tokio::task::spawn_blocking(move || {
                let started = Instant::now();
                let outcome = map.get_or_spawn(&scope, None).map(|_| ());
                (outcome, started.elapsed())
            })
        };
        let (outcome, waited) = waiter.await.unwrap();
        let err = outcome.expect_err("the waiter must fail closed, not hang");
        assert!(
            err.contains("timed out") && err.contains("wedged"),
            "the error must name the wait and the scope; got {err}"
        );
        assert!(
            waited >= deadline && waited < deadline + Duration::from_secs(5),
            "the waiter must give up at the deadline, not before and not much after; \
             waited {waited:?}"
        );

        // Let the wedged caller out so the test leaves no parked thread.
        let _ = release_tx.send(());
        wedged.join().unwrap();
    }

    /// A spawn that fails leaves no gate entry behind. The failure
    /// happens before anything reaches the map, so no eviction path
    /// would ever see the scope; both halves of the scope come off the
    /// wire, so an entry retained per refusal is a growth vector any
    /// authenticated client could drive. Without the lease's reclamation:
    /// 500 entries for 500 refused spawns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_spawn_that_fails_leaves_no_gate_entry() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(|| Err(EvaluatorError::InitError("wedged toolchain".into())));
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "always-fails".to_string());

        const REFUSALS: usize = 500;
        for i in 0..REFUSALS {
            let scope = SessionScope::new(T, format!("s{i}"));
            assert!(
                map.get_or_spawn(&scope, None).is_err(),
                "the factory refuses every spawn"
            );
        }

        assert_eq!(map.live_sessions(), 0, "no evaluator was ever installed");
        assert_eq!(
            map.spawn_gate.len(),
            0,
            "a refused spawn must not retain a gate entry"
        );
    }

    /// Evicting a scope drops its gate entry, so the side table never
    /// outgrows the sessions that are actually live.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn evicting_a_scope_forgets_its_spawn_gate_entry() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = Arc::clone(&counter);
        let factory: EvaluatorFactory = Arc::new(move || {
            let n = counter_for_factory.fetch_add(1, Ordering::Relaxed);
            Ok(LoggingEvaluator::new(&format!("e{n}")) as Arc<dyn Evaluator>)
        });
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::ZERO,
            Duration::ZERO,
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let a = SessionScope::new(T, "A");
        let b = SessionScope::new(T, "B");
        map.dispatch(&a, None, req_for(&a), 0).await.unwrap();
        map.dispatch(&b, None, req_for(&b), 0).await.unwrap();
        assert_eq!(map.spawn_gate.len(), 2, "one gate entry per live scope");

        // The idle sweep's path.
        assert!(map.evict_evaluator_only(&a));
        assert_eq!(map.spawn_gate.len(), 1);
        // The end-session path.
        assert!(map.evict(&b));
        assert_eq!(
            map.spawn_gate.len(),
            0,
            "an evicted scope leaves no gate entry behind"
        );
    }

    /// One evict does BOTH housekeeping jobs: the spawn gate forgets the
    /// scope, and the kill-ledger table collects the scope's entry. Each of
    /// the two mechanisms has its own test above; this one pins that a single
    /// evict path does not do one and skip the other — separately for each of
    /// the two paths, since they are separate functions carrying separate
    /// copies of both duties.
    ///
    /// The sweep's path is checked on its own scope, with no `evict` call
    /// anywhere near it: `prune_spent` drains the whole table, so one `evict`
    /// would collect the sweep's ledger on the other path's behalf and hide a
    /// missing prune in `evict_evaluator_only` — the path the idle sweep, the
    /// dead-subprocess branch of `dispatch` and the configuration rollback
    /// all use.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_evicted_scope_leaves_neither_a_gate_entry_nor_a_spent_ledger() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let factory: EvaluatorFactory =
            Arc::new(move || Ok(LoggingEvaluator::new("e") as Arc<dyn Evaluator>));
        let map = SessionEvaluatorMap::with_eviction(
            Arc::clone(&store),
            Duration::from_millis(1),
            1,
            Duration::from_secs(3600), // idle TTL: nothing ages out on its own
            Duration::ZERO,            // sweep off, so only the evict paths run
            None,
            EvaluationDeadlines::default(),
        );
        map.swap_factory(factory, "mock".to_string());

        let a = SessionScope::new(T, "A");
        let b = SessionScope::new(T, "B");
        map.dispatch(&a, None, req_for(&a), 0).await.unwrap();
        map.dispatch(&b, None, req_for(&b), 0).await.unwrap();
        assert_eq!(map.spawn_gate.len(), 2, "one gate entry per live scope");
        assert_eq!(map.kill_ledgers.len(), 2, "one ledger per live scope");

        // The idle sweep's path, on A alone.
        assert!(map.evict_evaluator_only(&a));
        assert_eq!(map.spawn_gate.len(), 1, "the sweep's path forgets the gate");

        // A ledger goes only once no task still holds it, and the aborted
        // task's Arc drops on the runtime's schedule rather than the evict's.
        // Keep evicting — a no-op on an empty scope, except that it prunes —
        // through the sweep's path only, and only for A. B is still live, so
        // its own task holds its ledger and exactly one entry must remain.
        let mut collected = false;
        for _ in 0..100 {
            if map.kill_ledgers.len() == 1 {
                collected = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            map.evict_evaluator_only(&a);
        }
        assert!(
            collected,
            "the sweep's path must collect the ledger of the scope it evicts"
        );

        // Then the end-session path, on B: the gate entry AND the ledger.
        assert!(map.evict(&b));
        assert_eq!(
            map.spawn_gate.len(),
            0,
            "the end-session path forgets the gate"
        );
        let mut collected = false;
        for _ in 0..100 {
            if map.kill_ledgers.len() == 0 {
                collected = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            map.evict(&b);
        }
        assert!(
            collected,
            "the end-session path must collect its ledger too"
        );
    }

    // ── The functor gate on the lazy install ────────────────────────
    //
    // A session that wakes up after a restart names a content hash whose
    // policy has not been compiled in this process yet. `ensure_installed`
    // compiles it then — functor source included — so it runs the gate first,
    // against the settings this process was started with.

    /// The persisted bundle a session's binding points at: real functor
    /// source, admitted as `admission`.
    fn persisted_functor_policy(
        admission: sasy_graph::FunctorAdmission,
    ) -> sasy_graph::PersistedPolicy {
        sasy_graph::PersistedPolicy {
            policy_source: "IsAuthorized(idx) :- Actions(idx, _).".into(),
            functor_source: "extern \"C\" const char* my_functor() { return \"\"; }".into(),
            backend: "souffle".into(),
            functor_admission: admission,
        }
    }

    /// With the opt-in off, a session bound to user-admitted functor source
    /// does not get it compiled on the way in. The dispatch fails, saying so
    /// and naming the flag that would admit it — and the session is left
    /// exactly as it was: no evaluator spawned, no entry in the registry, so
    /// the next call re-runs the same check rather than finding half a policy
    /// installed.
    #[tokio::test]
    async fn a_lazy_install_of_user_functors_is_refused_when_the_opt_in_is_off() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let bundle = persisted_functor_policy(sasy_graph::FunctorAdmission::User);
        store.put_policy_source("hash-fn", &bundle).unwrap();
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.set_policy_service_config(crate::service::PolicyServiceConfig {
            user_functors: crate::service::UserFunctors::Refuse,
            sandbox_available: Some(true),
        });
        let scope = SessionScope::new(T, "S");
        map.lazy_bind_session(scope.clone(), "hash-fn");

        let err = match map.dispatch(&scope, None, req_for(&scope), 0).await {
            Err(e) => e,
            Ok(_) => panic!("the gate must stop this before anything is compiled"),
        };
        assert!(
            err.contains(crate::service::FUNCTOR_REFUSED_AT_LOAD),
            "the failure must be the functor gate's, recognisably: {err}"
        );
        assert!(
            err.contains("--allow-user-functors"),
            "the failure must name the flag that would admit it: {err}"
        );
        assert_eq!(
            map.live_sessions(),
            0,
            "a refused install must not leave an evaluator behind"
        );
        assert!(
            !map.registry.contains(T, &PolicyId::from_string("hash-fn")),
            "a refused install must not leave the policy in the registry"
        );
    }

    /// The same binding under the unsandboxed opt-in gets past the gate. What
    /// happens next is the compiler's business (souffle and g++ are not
    /// assumed here), so this asserts only that the refusal is gone.
    #[tokio::test]
    async fn a_lazy_install_of_user_functors_passes_the_gate_when_unsandboxed() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let bundle = persisted_functor_policy(sasy_graph::FunctorAdmission::User);
        store.put_policy_source("hash-fn", &bundle).unwrap();
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.set_policy_service_config(crate::service::PolicyServiceConfig {
            user_functors: crate::service::UserFunctors::Unsandboxed,
            sandbox_available: Some(false),
        });
        let scope = SessionScope::new(T, "S");
        map.lazy_bind_session(scope.clone(), "hash-fn");

        let failure = map
            .dispatch(&scope, None, req_for(&scope), 0)
            .await
            .err()
            .unwrap_or_default();
        assert!(
            !failure.contains(crate::service::FUNCTOR_REFUSED_AT_LOAD),
            "the unsandboxed opt-in admits it: {failure}"
        );
    }

    /// Admin-admitted functor source loads with the opt-in off: the gate that
    /// admitted it was the admin role, and that has not changed.
    #[tokio::test]
    async fn a_lazy_install_of_admin_functors_is_not_held_back_by_the_flag() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let bundle = persisted_functor_policy(sasy_graph::FunctorAdmission::Admin);
        store.put_policy_source("hash-fn", &bundle).unwrap();
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.set_policy_service_config(crate::service::PolicyServiceConfig {
            user_functors: crate::service::UserFunctors::Refuse,
            sandbox_available: Some(false),
        });
        let scope = SessionScope::new(T, "S");
        map.lazy_bind_session(scope.clone(), "hash-fn");

        let failure = map
            .dispatch(&scope, None, req_for(&scope), 0)
            .await
            .err()
            .unwrap_or_default();
        assert!(
            !failure.contains(crate::service::FUNCTOR_REFUSED_AT_LOAD),
            "an admin-admitted policy is never held back by the opt-in: {failure}"
        );
    }

    /// One content hash, both classes stored: an admin uploaded the source and
    /// a non-admin later uploaded the same bytes. With the opt-in off the user
    /// record is refused and the admin record is what the lazy install
    /// compiles — the same rule boot replay follows, reached here through
    /// `ensure_installed`.
    ///
    /// This is what makes the lazy path's use of
    /// [`crate::replay::admitted_policy_source`] load-bearing: a gate that
    /// looked at one record — whichever the store hands back first, which is
    /// the least-privileged one — would refuse this session even though an
    /// admin really did upload these exact bytes.
    #[tokio::test]
    async fn a_lazy_install_reaches_the_admin_record_under_a_hash_that_has_both() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        store
            .put_policy_source(
                "hash-fn",
                &persisted_functor_policy(sasy_graph::FunctorAdmission::Admin),
            )
            .unwrap();
        store
            .put_policy_source(
                "hash-fn",
                &persisted_functor_policy(sasy_graph::FunctorAdmission::User),
            )
            .unwrap();
        let map = SessionEvaluatorMap::new(Arc::clone(&store), Duration::from_millis(1), 1);
        map.set_policy_service_config(crate::service::PolicyServiceConfig {
            user_functors: crate::service::UserFunctors::Refuse,
            sandbox_available: Some(false),
        });
        let scope = SessionScope::new(T, "S");
        map.lazy_bind_session(scope.clone(), "hash-fn");

        let failure = map
            .dispatch(&scope, None, req_for(&scope), 0)
            .await
            .err()
            .unwrap_or_default();
        assert!(
            !failure.contains(crate::service::FUNCTOR_REFUSED_AT_LOAD),
            "the later non-admin upload must not take away the admin record \
             that admits this hash: {failure}"
        );
    }
}
