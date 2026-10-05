//! Generic engine wrapping a per-session evaluator map.
//!
//! Each session gets its own [`SessionEvaluator`] (one Soufflé
//! subprocess per live session), spawned lazily on first traffic
//! and bootstrapped from [`GraphStore::get_session_state`]. Hot
//! reload swaps the evaluator factory; existing per-session
//! evaluators are dropped and the next query re-spawns. Denial
//! trace generation and timing are applied after the per-session
//! task returns the raw eval result.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use sasy_common::SessionScope;
use sasy_graph::GraphStore;
use tracing::{debug, info};

use crate::engine::{Engine, GraphUpdate, SyncStatus};
use crate::evaluator::types::{
    ActionMetadataEntry, EvalAction, EvalActionResult, EvalAuthRequest, PolicyMetadataFact,
};
use crate::evaluator::Evaluator;
#[cfg(test)]
use crate::evaluator::EvaluatorError;
use crate::policy_registry::PolicyId;
use crate::policy_types::{self, AuthAction};
use crate::rule_metadata::RuleMetadataStore;
use crate::session_evaluator::{EvaluatorFactory, SessionEvaluatorMap};
use crate::trace::{
    denial_reason_to_proto, generate_contextual_suggestions, group_allow_routes,
    rule_matches_action, TraceBuilder,
};

/// Default microseconds between background flushes of buffered
/// graph updates to each session evaluator. Set to 0 via
/// ``SASY_FLUSH_INTERVAL_US=0`` to disable. tokio's timer wheel
/// has ~1 ms granularity, so values below 1000 don't fire faster
/// in practice; the knob is in micros for forward compatibility.
const DEFAULT_FLUSH_INTERVAL_US: u64 = 1_000;

/// Per-session query channel capacity. With cap=1, dispatch
/// queues at most one query behind any in-flight one before
/// awaiting; bumping it absorbs short bursts without cross-task
/// hand-off. Sessions are independent, so capacity only absorbs
/// intra-session bursts. 4 matches the typical agent's
/// parallel-tool fan-out.
const SESSION_QUERY_CAPACITY: usize = 4;

fn resolve_flush_interval() -> Duration {
    let us = std::env::var("SASY_FLUSH_INTERVAL_US")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FLUSH_INTERVAL_US);
    Duration::from_micros(us)
}

/// Wrap a single shared [`Evaluator`] as a factory. Every session
/// receives the *same* evaluator handle — useful for tests where
/// per-session isolation is unnecessary, but never appropriate for
/// production where independent subprocess state per session is
/// the whole point.
pub fn shared_evaluator_factory(eval: Arc<dyn Evaluator>) -> EvaluatorFactory {
    Arc::new(move || Ok(Arc::clone(&eval)))
}

/// Policy engine backed by a per-session evaluator map.
pub struct EvaluatorEngine {
    sessions: Arc<SessionEvaluatorMap>,
    graph_store: Arc<GraphStore>,
    /// Rule metadata (`// @deny_message:` / `// @suggestion:` /
    /// source locations) keyed by the resolved policy id, so a tenant
    /// running multiple policies attributes each denial trace to the
    /// rule metadata of the policy that actually decided it — not
    /// whichever policy happened to load last into one shared store.
    /// The [`legacy_metadata_key`] entry holds metadata loaded via
    /// the policy-id-less legacy path (single-policy boot) and is the
    /// fallback when no per-policy entry matches the resolved id.
    rule_metadata: RwLock<HashMap<PolicyId, RuleMetadataStore>>,
    connected: std::sync::atomic::AtomicBool,
    stopped: std::sync::atomic::AtomicBool,
}

/// Metadata-map key for rule metadata loaded without a policy id (the
/// legacy single-policy [`Engine::load_rule_metadata`] path). The
/// empty string is never a real content-hash policy id, so it can't
/// collide with a per-policy entry; it is the denial-trace fallback
/// when no per-policy metadata matches the resolved id.
fn legacy_metadata_key() -> PolicyId {
    PolicyId::from_string(String::new())
}

/// The `min_sequence` a dispatch carries: the per-scope shard counter.
///
/// The value must live in the SAME counter space the evaluator compares it
/// against, and that differs by scope:
///   · non-global → the session's own shard counter, so unrelated scopes'
///     broadcasts don't lengthen this scope's fence wait (its cursor advances
///     on matching `SessionSequence` markers).
///   · global → the SAME call yields the tenant's GLOBAL SHARD counter, since a
///     global scope's session id is empty. A global evaluator never fences on
///     the broadcast (the store-wide `Sequence` stream reorders across shards);
///     it re-syncs from the durable store when this counter has moved past its
///     last snapshot.
///
/// Deliberately NOT the store-wide `get_sequence()` for the global case: every
/// session of every tenant bumps that, so gating on it forces a full-tenant
/// reload on unrelated traffic — and the refmon takes the global path per
/// proxied request. Equally NOT a constant: returning 0 for a global scope would
/// permanently disable the re-sync and silently stop a session-less writer from
/// reading its own writes. Extracted so both mistakes are test-pinned.
fn dispatch_fence_sequence(graph_store: &GraphStore, scope: &SessionScope) -> i64 {
    graph_store.session_sequence(scope)
}

impl EvaluatorEngine {
    /// Construct with a factory that spawns a fresh
    /// per-session evaluator on demand. The factory is what
    /// lets a freshly-touched (or resumed) session bring up its
    /// own subprocess; the map then bootstraps that subprocess
    /// from the session's existing graph state.
    pub fn new(
        factory: EvaluatorFactory,
        backend_name: String,
        graph_store: Arc<GraphStore>,
    ) -> Self {
        Self::with_deadlines(
            factory,
            backend_name,
            graph_store,
            crate::session_evaluator::EvaluationDeadlines::from_env(),
        )
    }

    /// Like [`Self::new`] but with the evaluation deadlines given rather than
    /// read from the environment. This is the constructor the binary uses, so
    /// `--query-timeout-secs` and `--evaluator-stall-secs` reach the session
    /// evaluators as configuration rather than through a global.
    pub fn with_deadlines(
        factory: EvaluatorFactory,
        backend_name: String,
        graph_store: Arc<GraphStore>,
        deadlines: crate::session_evaluator::EvaluationDeadlines,
    ) -> Self {
        let sessions = SessionEvaluatorMap::with_deadlines(
            Arc::clone(&graph_store),
            resolve_flush_interval(),
            SESSION_QUERY_CAPACITY,
            deadlines,
        );
        sessions.swap_factory(factory, backend_name);
        Self {
            sessions,
            graph_store,
            rule_metadata: RwLock::new(HashMap::new()),
            connected: std::sync::atomic::AtomicBool::new(true),
            stopped: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Hand the operator's deployment decisions to the session map, which
    /// consults them before it compiles persisted functor source on a lazy
    /// install. Called once by the binary at startup; without it the
    /// conservative default applies (user-admitted functor source is
    /// refused).
    pub fn with_policy_service_config(self, config: crate::service::PolicyServiceConfig) -> Self {
        self.sessions.set_policy_service_config(config);
        self
    }

    /// The decisions this engine's lazy-install path will consult. An engine
    /// built without [`Self::with_policy_service_config`] reports the
    /// conservative default, so a caller can check that the load paths in its
    /// process — boot replay and lazy install — see one configuration.
    pub fn policy_service_config(&self) -> crate::service::PolicyServiceConfig {
        self.sessions.policy_service_config()
    }

    /// Convenience constructor: every session sees the same
    /// evaluator. Equivalent to
    /// ``new(shared_evaluator_factory(eval), eval.backend_name())``.
    /// For tests and the legacy single-shared-evaluator mode.
    pub fn with_shared_evaluator(
        evaluator: Arc<dyn Evaluator>,
        graph_store: Arc<GraphStore>,
    ) -> Self {
        let backend = evaluator.backend_name().to_string();
        Self::new(shared_evaluator_factory(evaluator), backend, graph_store)
    }

    fn build_denial_trace(
        &self,
        action: &AuthAction,
        eval_result: &EvalActionResult,
        resolved_policy_id: Option<&PolicyId>,
    ) -> sasy_common::policy_engine::DenialTrace {
        let mut builder = TraceBuilder::new(action);

        if !eval_result.is_authenticated {
            builder = builder.not_authenticated(None);
        }

        // Reasons from the DenialReason relation, routed by kind: "ask" → a
        // soft Ask reason (reason_type ASK), anything else → a hard Denylisted
        // reason. (Ask reasons appear on otherwise-authorized actions, so this
        // runs whenever the shim reported the action as not-authorized — which
        // it does for both denylisted and requires_approval.)
        let mut emitted_reason = false;
        for dr in &eval_result.denial_reasons {
            emitted_reason = true;
            builder = if dr.kind == "ask" {
                builder.ask(Some(dr.reason.clone()), None)
            } else {
                builder.denylisted(Some(dr.reason.clone()), None)
            };
            if !dr.suggestion.is_empty() {
                builder = builder.suggest(&dr.suggestion);
            }
        }
        if eval_result.is_denylisted && !emitted_reason {
            builder = builder.denylisted(None, None);
        }

        let allow_routes = if !eval_result.is_allowlisted {
            group_allow_routes(&eval_result.allow_routes)
        } else {
            Vec::new()
        };
        let metadata_map = self.rule_metadata.read();
        // Prefer the metadata of the policy this dispatch actually
        // resolved to; fall back to the legacy single-policy slot.
        let metadata_store = resolved_policy_id
            .and_then(|id| metadata_map.get(id))
            .or_else(|| metadata_map.get(&legacy_metadata_key()))
            .filter(|store| store.is_loaded());
        let allow_metadata = metadata_store
            .and_then(|store| store.get_metadata_for_relation("IsAuthorized"))
            .map(|list| list.as_slice())
            .unwrap_or(&[]);
        if !allow_routes.is_empty() {
            // `reasons` and `suggested_fixes` are read by the model, so they
            // carry the authored wording of the allow rules that apply to
            // this request: a rule whose `@url_pattern` / `@tool_pattern`
            // annotation names this action's target, or one with no such
            // annotation. A rule annotated for another target is no
            // explanation for this request, whatever the analysis decided
            // about it. Rule identifiers and analysis verdicts stay in
            // `allow_routes` for the application; a suggestion on a route
            // ruled out by fixed context was already dropped when the
            // routes were grouped.
            let mut applicable = false;
            for route in allow_routes
                .iter()
                .filter(|r| !r.details.is_empty() || !r.suggestions.is_empty())
            {
                // The route carries the rule's wording, not its annotations;
                // the metadata parsed from the policy file carries both. The
                // wording pairs them: the denial message, narrowed by the
                // suggestions when the route kept any (a route ruled out by
                // fixed context keeps none), so two rules with one message
                // but different hints are told apart.
                // A route carries the bounded form of the authored message, so
                // the metadata's own text is bounded the same way before it is
                // compared; a message past the bound would otherwise match
                // nothing and be taken for a rule with no annotation.
                let bounded =
                    |text: &str| sasy_policy_analysis::allow_attribution::bounded_hint(text);
                // Grouping keeps the first appearance of each suggestion and
                // drops them all from a route ruled out by fixed context, so
                // the rule's own list goes through the same steps before it is
                // compared with what the route ended up carrying. The order the
                // evaluator returns rows in is not the order they were authored
                // in, so both sides are sorted and the comparison is on the set.
                let as_route = |values: &[String]| {
                    let mut seen: Vec<String> = Vec::new();
                    for value in sasy_policy_analysis::allow_attribution::route_suggestions(values)
                    {
                        if !value.is_empty() && !seen.contains(&value) {
                            seen.push(value);
                        }
                    }
                    seen.sort();
                    seen
                };
                let carried = {
                    let mut carried = route.suggestions.clone();
                    carried.sort();
                    carried
                };
                let same_message: Vec<_> = allow_metadata
                    .iter()
                    .filter(|meta| {
                        (!route.details.is_empty()
                            && meta.deny_message.as_deref().map(&bounded).as_deref()
                                == Some(route.details.as_str()))
                            || (route.details.is_empty()
                                && meta.deny_message.is_none()
                                && !route.suggestions.is_empty()
                                && as_route(&meta.suggestions) == carried)
                    })
                    .collect();
                let same_hints: Vec<_> = same_message
                    .iter()
                    .copied()
                    .filter(|meta| {
                        route.suggestions.is_empty() || as_route(&meta.suggestions) == carried
                    })
                    .collect();
                let candidates = if same_hints.is_empty() {
                    same_message
                } else {
                    same_hints
                };
                let for_another_target = !candidates.is_empty()
                    && candidates.iter().all(|meta| {
                        !meta.custom.is_empty() && !rule_matches_action(action, &meta.custom)
                    });
                if for_another_target {
                    continue;
                }
                applicable = true;
                builder = builder.not_allowlisted(
                    (!route.details.is_empty()).then(|| route.details.clone()),
                    route.source_location.clone(),
                );
                for suggestion in &route.suggestions {
                    builder = builder.suggest(suggestion);
                }
            }
            if !applicable {
                builder = builder.not_allowlisted(None, None);
            }
        } else if !eval_result.is_allowlisted {
            let mut found = false;
            for meta in allow_metadata {
                if !rule_matches_action(action, &meta.custom) {
                    continue;
                }
                found = true;
                builder = builder
                    .not_allowlisted(meta.deny_message.clone(), meta.source_location.clone());
                builder = builder.with_rule_metadata(meta);
            }
            if !found {
                builder = builder.not_allowlisted(None, None);
            }
        }

        if builder.suggested_fixes_count() == 0 && allow_routes.is_empty() {
            for suggestion in generate_contextual_suggestions(action) {
                builder = builder.suggest(&suggestion);
            }
        }

        let trace = builder.build();
        sasy_common::policy_engine::DenialTrace {
            action_description: trace.action_description,
            reasons: trace.reasons.iter().map(denial_reason_to_proto).collect(),
            suggested_fixes: trace.suggested_fixes,
            allow_routes,
        }
    }

    /// Load `// @deny_message:` / `// @suggestion:` / source-location
    /// metadata from `path` into the per-policy slot `key`, then prune
    /// slots for policies no longer in the registry — bounding growth
    /// to live policies. Identical re-uploads dedup to one `key` (the
    /// content hash). The legacy fallback slot is always retained.
    fn load_metadata_keyed(
        &self,
        key: PolicyId,
        path: &std::path::Path,
    ) -> Result<(), anyhow::Error> {
        let mut new_store = RuleMetadataStore::new();
        new_store
            .load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load rule metadata: {}", e))?;

        let live: std::collections::HashSet<PolicyId> = self
            .sessions
            .registry()
            .list()
            .into_iter()
            .map(|(_, pid)| pid)
            .collect();
        let legacy = legacy_metadata_key();

        let mut map = self.rule_metadata.write();
        map.retain(|k, _| *k == legacy || *k == key || live.contains(k));
        map.insert(key, new_store);
        Ok(())
    }
}

impl Engine for EvaluatorEngine {
    fn apply_graph_updates(&self, _updates: Vec<GraphUpdate>) -> Result<(), anyhow::Error> {
        // No-op: per-session evaluators subscribe directly to the
        // graph store's broadcast and filter by session_id. This
        // method exists only for Engine trait compatibility.
        Ok(())
    }

    fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: &[sasy_common::policy_engine::Action],
        entity: Option<&str>,
        roles: &[String],
        scope: &SessionScope,
        principal: Option<&str>,
        policy_id: Option<&str>,
    ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(anyhow::anyhow!("engine stopped"));
        }

        let total_start = Instant::now();

        let mut eval_actions = Vec::with_capacity(actions.len());
        let mut auth_actions: Vec<AuthAction> = Vec::with_capacity(actions.len());
        let mut action_metadata: Vec<ActionMetadataEntry> = Vec::new();
        for action in actions {
            if let Some(ea) = EvalAction::from_proto(action) {
                // Index by position in the *kept* actions vec (from_proto
                // drops actions with no action_type), so ActionMetadata's
                // idx matches Actions(idx, _) in the evaluator.
                let idx = eval_actions.len() as u32;
                eval_actions.push(ea);
                auth_actions.push(policy_types::proto_action_to_auth_option(action).unwrap_or(
                    AuthAction::HttpRequest {
                        url: String::new(),
                        body: String::new(),
                        headers: vec![],
                    },
                ));
                if !action.metadata.is_empty() {
                    action_metadata.push(ActionMetadataEntry {
                        index: idx,
                        facts: action
                            .metadata
                            .iter()
                            .map(|f| PolicyMetadataFact {
                                rel: f.rel.clone(),
                                a: f.a.clone(),
                                b: f.b.clone(),
                            })
                            .collect(),
                    });
                }
            }
        }

        let eval_req = EvalAuthRequest {
            current_node_ids: current_node_ids.to_vec(),
            actions: eval_actions,
            entity: entity.map(|s| s.to_string()),
            roles: roles.to_vec(),
            tenant_id: Some(scope.tenant().to_string()),
            session_id: if scope.is_global() {
                None
            } else {
                Some(scope.session().to_string())
            },
            principal: principal.map(|s| s.to_string()),
            action_metadata,
        };

        // Sequence fence: still required because intra-session
        // parallel tool calls can land just after the auth request
        // they enable. See `dispatch_fence_sequence`.
        let min_sequence = dispatch_fence_sequence(&self.graph_store, scope);

        let sessions = Arc::clone(&self.sessions);
        let handle = tokio::runtime::Handle::current();
        let scope_for_dispatch = scope.clone();
        let policy_id_owned = policy_id.map(crate::policy_registry::PolicyId::from_string);
        let query_result = tokio::task::block_in_place(|| {
            handle.block_on(sessions.dispatch(
                &scope_for_dispatch,
                policy_id_owned.as_ref(),
                eval_req,
                min_sequence,
            ))
        })
        .map_err(|e| anyhow::anyhow!("{}", e))?;

        let sync_wait_us = query_result.sync_wait_us;
        let flush_us = query_result.flush_us;
        let eval_us = query_result.eval_us;
        let graph_nodes = query_result.graph_nodes;
        let graph_edges = query_result.graph_edges;
        let session_nodes = query_result.session_nodes;
        let session_edges = query_result.session_edges;
        let resolved_policy_id = query_result.resolved_policy_id.clone();
        let eval_resp = query_result.eval_response;

        let results = eval_resp
            .results
            .iter()
            .enumerate()
            .map(|(i, eval_result)| {
                let trace = if !eval_result.authorized {
                    let action = auth_actions.get(i);
                    action.map(|a| {
                        self.build_denial_trace(a, eval_result, resolved_policy_id.as_ref())
                    })
                } else {
                    None
                };

                sasy_common::policy_engine::ActionResult {
                    index: eval_result.index,
                    authorized: eval_result.authorized,
                    trace,
                    transform_ids: eval_result.transform_ids.clone(),
                    deny_if_unauthorized: eval_result.deny_if_unauthorized,
                }
            })
            .collect();

        let total_us = total_start.elapsed().as_micros() as u64;
        let backend = self.sessions.backend_name();

        let timing = sasy_common::policy_engine::PerformanceTiming {
            total_us,
            sync_wait_us,
            eval_us,
            backend: backend.clone(),
            graph_nodes,
            graph_edges,
        };

        // Optional JSONL latency log (no-op unless
        // ``SASY_SERVER_LATENCY_LOG_FILE`` is set). One record per
        // action — multi-action RPCs (e.g. SendMessage with N tool
        // calls) share a request_id so consumers can regroup. Each
        // record carries the inputs the engine evaluated against
        // (fn_name, args, current_node_ids), making the log a
        // sufficient replay source.
        if crate::latency_log::enabled() {
            let request_id = crate::latency_log::next_request_id();
            let ts = crate::latency_log::now_ts();
            let sid_str = scope.session();
            for (i, action) in actions.iter().enumerate() {
                let (fn_name, args) = match &action.action_type {
                    Some(sasy_common::policy_engine::action::ActionType::ToolCall(tc)) => {
                        (tc.fn_name.as_str(), tc.args.as_str())
                    }
                    Some(sasy_common::policy_engine::action::ActionType::HttpRequest(hr)) => {
                        (hr.url.as_str(), hr.body.as_str())
                    }
                    Some(sasy_common::policy_engine::action::ActionType::SendMessage(_)) => {
                        ("send_message", "")
                    }
                    None => ("", ""),
                };
                let authorized = eval_resp
                    .results
                    .iter()
                    .find(|r| r.index as usize == i)
                    .map(|r| r.authorized)
                    .unwrap_or(false);
                crate::latency_log::log_action(&crate::latency_log::ActionRecord {
                    ts,
                    request_id,
                    action_idx: i as u32,
                    fn_name,
                    args,
                    current_node_ids,
                    authorized,
                    total_us,
                    sync_us: sync_wait_us,
                    flush_us,
                    eval_us,
                    session_id: sid_str,
                    backend: &backend,
                });
            }
        }

        // ``sync`` is the seq-fence wait, ``flush`` is the IPC
        // time spent shipping this session's buffered
        // ``GraphUpdate`` batches to its evaluator since the
        // previous query, ``eval`` is the policy IPC for this
        // query alone. With per-session evaluators the flush
        // figure is bounded by intra-session traffic only:
        // cross-session bursts do not contribute.
        match (session_nodes, session_edges) {
            (Some(sn), Some(se)) => {
                debug!(
                    "Timing: total={}us sync={}us flush={}us eval={}us backend={} \
                     graph={}n/{}e session={}n/{}e",
                    total_us,
                    sync_wait_us,
                    flush_us,
                    eval_us,
                    backend,
                    graph_nodes,
                    graph_edges,
                    sn,
                    se,
                );
            }
            _ => {
                debug!(
                    "Timing: total={}us sync={}us flush={}us eval={}us backend={} graph={}n/{}e",
                    total_us, sync_wait_us, flush_us, eval_us, backend, graph_nodes, graph_edges
                );
            }
        }

        Ok(sasy_common::policy_engine::AuthorizationResponse {
            results,
            timing: Some(timing),
        })
    }

    fn reset(&self) -> Result<(), anyhow::Error> {
        // Evaluators re-bootstrap from graph store on next query.
        Ok(())
    }

    fn load_rule_metadata(&self, path: &std::path::Path) -> Result<(), anyhow::Error> {
        // Legacy policy-id-less path (single-policy boot): load into
        // the fallback slot keyed by the empty id.
        self.load_metadata_keyed(legacy_metadata_key(), path)
    }

    fn load_rule_metadata_for_policy(
        &self,
        policy_id: &str,
        path: &std::path::Path,
    ) -> Result<(), anyhow::Error> {
        self.load_metadata_keyed(PolicyId::from_string(policy_id.to_string()), path)
    }

    fn get_sync_status(&self) -> SyncStatus {
        let seq = self.graph_store.get_sequence();
        let (node_count, edge_count) = self.graph_store.get_counts();
        SyncStatus {
            current_sequence: seq,
            node_count,
            edge_count,
            connected: self.connected.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    fn set_connected(&self, connected: bool) {
        self.connected
            .store(connected, std::sync::atomic::Ordering::Relaxed);
    }

    fn set_sequence(&self, _seq: i64) {
        // No-op: each session evaluator tracks sequence from
        // broadcasts.
    }

    fn backend_name(&self) -> String {
        self.sessions.backend_name()
    }

    fn worker_count(&self) -> usize {
        // The number of currently bootstrapped sessions.
        self.sessions.live_sessions()
    }

    fn swap_evaluator(&self, new_evaluator: Arc<dyn Evaluator>) -> Result<(), anyhow::Error> {
        let backend = new_evaluator.backend_name().to_string();
        info!("Swapping evaluator (single-shared) to: {}", backend);
        self.sessions
            .swap_factory(shared_evaluator_factory(new_evaluator), backend);
        info!("Evaluator swap complete");
        Ok(())
    }

    fn swap_evaluator_factory(
        &self,
        factory: EvaluatorFactory,
        backend_name: String,
    ) -> Result<(), anyhow::Error> {
        info!("Swapping evaluator factory to: {}", backend_name);
        self.sessions.swap_factory(factory, backend_name);
        info!("Evaluator factory swap complete");
        Ok(())
    }

    fn evict_tenant_evaluators(&self, tenant: &str) -> usize {
        self.sessions.evict_tenant_evaluators_only(tenant)
    }

    fn evict_session_evaluator(&self, scope: &SessionScope) -> bool {
        self.sessions.evict_evaluator_only(scope)
    }

    fn end_session(&self, scope: &SessionScope) -> bool {
        self.sessions.evict(scope)
    }

    fn install_policy(
        &self,
        tenant: &str,
        content_hash: &str,
        factory: EvaluatorFactory,
        backend_name: String,
        mode: crate::engine::InstallMode,
    ) -> Result<String, anyhow::Error> {
        use crate::engine::InstallMode;
        let force = matches!(mode, InstallMode::Force);
        // On Force, install WITHOUT publishing. The cleanup below can fail,
        // and publishing first would leave the default already changed with
        // no eviction performed: new and unbound sessions on the new policy,
        // pinned sessions still on the old one, and an error telling the
        // caller nothing happened. Nothing is published until every fallible
        // step has succeeded.
        let make_default = matches!(mode, InstallMode::Default);
        let policy_id = self.sessions.registry().install_dedup(
            tenant,
            content_hash,
            factory,
            backend_name,
            make_default,
        );
        if force {
            // The persisted per-session config was already cleared by the
            // caller, before any of this ran. It has to be gone before the
            // eviction below: eviction releases the evaluator and binding
            // locks, so an authorization arriving right after it can respawn
            // under the new default and seed from the old binding row,
            // keeping stale configuration for the evaluator's lifetime — the
            // rollout this force update exists to apply. The caller owns the
            // delete because it is the only side of this that cannot be
            // undone from here if the promotion below fails.
            self.sessions
                .registry()
                .promote_to_default(tenant, content_hash)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // Only now: every session re-bootstraps under the new default.
            let n = self.sessions.evict_tenant(tenant);
            if n > 0 {
                info!(tenant, evicted = n, "force-update evicted tenant sessions");
            }
        }
        info!(
            tenant,
            policy_id = %policy_id,
            mode = ?mode,
            "policy installed"
        );
        Ok(policy_id.to_string())
    }

    fn lookup_policy_by_content(&self, tenant: &str, content_hash: &str) -> Option<String> {
        self.sessions
            .registry()
            .lookup_by_content(tenant, content_hash)
            .map(|p| p.to_string())
    }

    fn lookup_policy_by_name(&self, tenant: &str, name: &str) -> Option<String> {
        self.sessions
            .registry()
            .lookup_by_name(tenant, name)
            .map(|p| p.to_string())
    }

    fn register_profile_name(&self, tenant: &str, name: &str, policy_id: &str) {
        let pid = crate::policy_registry::PolicyId::from_string(policy_id);
        self.sessions.registry().register_name(tenant, name, &pid);
    }

    fn set_session_policy(
        &self,
        scope: &SessionScope,
        policy_id: &str,
    ) -> Result<bool, anyhow::Error> {
        let pid = crate::policy_registry::PolicyId::from_string(policy_id);
        self.sessions
            .set_session_policy(scope, &pid)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn update_session_metadata(
        &self,
        scope: &SessionScope,
        facts: Vec<crate::evaluator::types::PolicyMetadataFact>,
    ) -> Result<(), anyhow::Error> {
        // Persist (sync) + re-seed any live evaluator (async, hence
        // the block_in_place bridge — same as the dispatch path).
        let sessions = Arc::clone(&self.sessions);
        let handle = tokio::runtime::Handle::current();
        let scope_owned = scope.clone();
        tokio::task::block_in_place(|| {
            handle.block_on(sessions.update_session_metadata(&scope_owned, facts))
        })
        .map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn lazy_bind_session(&self, scope: SessionScope, content_hash: &str) {
        self.sessions.lazy_bind_session(scope, content_hash);
    }

    fn lazy_set_default(&self, tenant: &str, content_hash: &str) {
        self.sessions.lazy_set_default(tenant, content_hash);
    }

    fn promote_to_default(
        &self,
        tenant: &str,
        content_hash: &str,
        force: bool,
    ) -> Result<(), anyhow::Error> {
        // The caller has already cleared the persisted per-session config for
        // a force rollout, for the reason given in `install_policy`.
        self.sessions
            .registry()
            .promote_to_default(tenant, content_hash)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        if force {
            // Force semantics: every live session in the tenant
            // re-bootstraps under the new default on next traffic.
            let n = self.sessions.evict_tenant(tenant);
            if n > 0 {
                info!(
                    tenant,
                    evicted = n,
                    "promote_to_default(force) evicted sessions"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, GraphUpdate};
    use crate::evaluator::types::{EvalActionResult, EvalAuthRequest, EvalAuthResponse};
    use parking_lot::Mutex;

    struct MockEvaluator {
        updates: Mutex<Vec<Vec<GraphUpdate>>>,
        name: String,
    }

    impl MockEvaluator {
        fn new(name: &str) -> Arc<Self> {
            Arc::new(Self {
                updates: Mutex::new(Vec::new()),
                name: name.to_string(),
            })
        }
    }

    #[tonic::async_trait]
    impl crate::evaluator::Evaluator for MockEvaluator {
        async fn update(&self, updates: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            self.updates.lock().push(updates);
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
            self.updates.lock().clear();
            Ok(())
        }

        fn backend_name(&self) -> &str {
            &self.name
        }
    }

    /// c2: a denial trace must carry the rule metadata of the policy
    /// the dispatch actually resolved to, not whichever policy loaded
    /// last. Two policies share the same allowlist rule but differ
    /// only in their `@deny_message`; the trace selects by the
    /// resolved policy id, and a `None` id falls back to the generic
    /// message (the never-loaded legacy slot).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn denial_trace_selects_metadata_by_resolved_policy() {
        use crate::policy_registry::PolicyId;
        use crate::policy_types::AuthAction;

        let dir = std::env::temp_dir();
        let pid_a = format!("c2-deny-a-{}", std::process::id());
        let pid_b = format!("c2-deny-b-{}", std::process::id());
        let path_a = dir.join(format!("{pid_a}.dl"));
        let path_b = dir.join(format!("{pid_b}.dl"));
        std::fs::write(
            &path_a,
            "// @deny_message: DENY_FROM_A\n\
             IsAuthorized(idx) :- Actions(idx, a), IsToolCall(a).\n",
        )
        .unwrap();
        std::fs::write(
            &path_b,
            "// @deny_message: DENY_FROM_B\n\
             IsAuthorized(idx) :- Actions(idx, a), IsToolCall(a).\n",
        )
        .unwrap();

        let store = Arc::new(GraphStore::new(None).unwrap());
        let engine = EvaluatorEngine::with_shared_evaluator(MockEvaluator::new("mock"), store);

        // Register both policies so load_metadata_keyed's prune (which
        // retains only registry-live policies) keeps both entries.
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        let reg = engine.sessions.registry();
        reg.install_dedup("t", &pid_a, Arc::clone(&factory), "mock".into(), false);
        reg.install_dedup("t", &pid_b, factory, "mock".into(), false);

        engine
            .load_metadata_keyed(PolicyId::from_string(pid_a.clone()), &path_a)
            .unwrap();
        engine
            .load_metadata_keyed(PolicyId::from_string(pid_b.clone()), &path_b)
            .unwrap();

        let action = AuthAction::ToolCall {
            fn_name: "send_email".into(),
            args: "{}".into(),
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            allow_routes: vec![],
            transform_ids: vec![],
            deny_if_unauthorized: false,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
        };

        let id_a = PolicyId::from_string(pid_a.clone());
        let id_b = PolicyId::from_string(pid_b.clone());
        let trace_a = engine.build_denial_trace(&action, &denied, Some(&id_a));
        let trace_b = engine.build_denial_trace(&action, &denied, Some(&id_b));
        let trace_none = engine.build_denial_trace(&action, &denied, None);

        let has = |t: &sasy_common::policy_engine::DenialTrace, msg: &str| {
            t.reasons.iter().any(|r| r.details == msg)
        };
        assert!(
            has(&trace_a, "DENY_FROM_A"),
            "A trace should carry A's message: {:?}",
            trace_a.reasons
        );
        assert!(
            !has(&trace_a, "DENY_FROM_B"),
            "A trace must not carry B's message"
        );
        assert!(
            has(&trace_b, "DENY_FROM_B"),
            "B trace should carry B's message: {:?}",
            trace_b.reasons
        );
        assert!(
            !has(&trace_b, "DENY_FROM_A"),
            "B trace must not carry A's message"
        );
        assert!(
            !has(&trace_none, "DENY_FROM_A") && !has(&trace_none, "DENY_FROM_B"),
            "no-policy trace must use the generic fallback, got {:?}",
            trace_none.reasons,
        );

        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inferred_routes_filter_hints_without_hiding_explicit_denials() {
        use crate::evaluator::types::{EvalAllowRoute, EvalDenialReason};
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let action = AuthAction::ToolCall {
            fn_name: "transfer".into(),
            args: "{}".into(),
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: true,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![EvalDenialReason {
                kind: "block".into(),
                reason: "immutable deny".into(),
                suggestion: String::new(),
            }],
            allow_routes: vec![
                EvalAllowRoute {
                    rule_id: "wrong-principal".into(),
                    status: "blocked".into(),
                    details: "principal differs".into(),
                    suggestion: "irrelevant approval".into(),
                    source_location: "policy:1".into(),
                },
                EvalAllowRoute {
                    rule_id: "current-principal".into(),
                    status: "possible".into(),
                    details: "fixed principal matches".into(),
                    suggestion: "request approval".into(),
                    source_location: "policy:2".into(),
                },
            ],
        };
        let trace = engine.build_denial_trace(&action, &denied, None);
        assert_eq!(trace.allow_routes.len(), 2);
        assert!(trace.reasons.iter().any(|r| r.details == "immutable deny"));
        assert!(trace
            .suggested_fixes
            .iter()
            .any(|s| s.contains("request approval")));
        assert!(!trace
            .suggested_fixes
            .iter()
            .any(|s| s.contains("irrelevant approval")));
        // The model-facing reasons carry the routes' authored wording,
        // without rule identifiers or analysis verdicts. Neither rule is
        // annotated with a target, so both apply; the hint on the route
        // ruled out by fixed context was dropped when routes were grouped.
        let reasons: Vec<_> = trace.reasons.iter().map(|r| r.details.clone()).collect();
        assert!(reasons.contains(&"fixed principal matches".to_string()));
        assert!(reasons.contains(&"principal differs".to_string()));
        assert!(!reasons.iter().any(|r| r.contains("current-principal")
            || r.contains("wrong-principal")
            || r.contains("reachability")));
        assert_eq!(trace.suggested_fixes, vec!["request approval".to_string()]);
        assert!(!denied.authorized);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn denial_trace_leaves_out_routes_whose_annotation_names_another_target() {
        use crate::evaluator::types::EvalAllowRoute;
        use crate::policy_registry::PolicyId;

        let dir = std::env::temp_dir();
        let pid = format!("c2-target-{}", std::process::id());
        let path = dir.join(format!("{pid}.dl"));
        std::fs::write(
            &path,
            "// @deny_message: OpenAI API access requires openai-access role\n\
             // @url_pattern: api.openai.com\n\
             IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.openai.com\"), HasRole(\"openai-access\").\n\
             // @deny_message: FDA API access requires approved registration\n\
             // @url_pattern: api.fda.gov\n\
             IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.fda.gov\"), HasRole(\"fda-access\").\n",
        )
        .unwrap();
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        engine
            .sessions
            .registry()
            .install_dedup("t", &pid, factory, "mock".into(), false);
        engine
            .load_metadata_keyed(PolicyId::from_string(pid.clone()), &path)
            .unwrap();

        let action = AuthAction::HttpRequest {
            url: "https://api.fda.gov/drug/label.json".into(),
            body: String::new(),
            headers: vec![],
        };
        let route = |rule_id: &str, details: &str, suggestion: &str| EvalAllowRoute {
            rule_id: rule_id.into(),
            // QueriesHost is not a condition the analysis decides.
            status: "unknown".into(),
            details: details.into(),
            suggestion: suggestion.into(),
            source_location: String::new(),
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
            allow_routes: vec![
                route(
                    "openai",
                    "OpenAI API access requires openai-access role",
                    "Request the openai-access role from an administrator",
                ),
                route(
                    "fda",
                    "FDA API access requires approved registration",
                    "Call the `register_fda_usage` tool",
                ),
            ],
        };
        let trace = engine.build_denial_trace(&action, &denied, Some(&PolicyId::from_string(pid)));
        let _ = std::fs::remove_file(&path);
        assert_eq!(trace.allow_routes.len(), 2);
        let reasons: Vec<_> = trace.reasons.iter().map(|r| r.details.clone()).collect();
        assert_eq!(
            reasons,
            vec!["FDA API access requires approved registration".to_string()]
        );
        assert_eq!(
            trace.suggested_fixes,
            vec!["Call the `register_fda_usage` tool".to_string()]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rule_with_more_suggestions_than_a_route_carries_still_pairs() {
        use crate::evaluator::types::EvalAllowRoute;
        use crate::policy_registry::PolicyId;

        // Five suggestions and no denial message: the route carries the first
        // four, so the rule is recognised only if the comparison applies the
        // same cap.
        let dir = std::env::temp_dir();
        let pid = format!("c2-hints-{}", std::process::id());
        let path = dir.join(format!("{pid}.dl"));
        std::fs::write(
            &path,
            "// @suggestion: Ask an approver\n\
             // @suggestion: Use the internal endpoint\n\
             // @suggestion: Retry with fewer fields\n\
             // @suggestion: Request the openai-access role\n\
             // @suggestion: Contact the operator\n\
             // @url_pattern: api.openai.com\n\
             IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.openai.com\").\n",
        )
        .unwrap();
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        engine
            .sessions
            .registry()
            .install_dedup("t", &pid, factory, "mock".into(), false);
        engine
            .load_metadata_keyed(PolicyId::from_string(pid.clone()), &path)
            .unwrap();
        let action = AuthAction::HttpRequest {
            url: "https://api.fda.gov/drug/label.json".into(),
            body: String::new(),
            headers: vec![],
        };
        // The evaluator returns its rows in its own order, not the order the
        // suggestions were authored in, so the route is built shuffled here.
        let hints = [
            "Retry with fewer fields",
            "Ask an approver",
            "Request the openai-access role",
            "Use the internal endpoint",
        ];
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
            allow_routes: hints
                .iter()
                .map(|hint| EvalAllowRoute {
                    rule_id: "openai".into(),
                    status: "unknown".into(),
                    details: String::new(),
                    suggestion: (*hint).into(),
                    source_location: String::new(),
                })
                .collect(),
        };
        let trace = engine.build_denial_trace(&action, &denied, Some(&PolicyId::from_string(pid)));
        let _ = std::fs::remove_file(&path);
        // The rule names another host, so none of its guidance belongs here.
        assert!(
            trace.suggested_fixes.is_empty(),
            "{:?}",
            trace.suggested_fixes
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_denial_message_past_the_hint_bound_still_pairs_with_its_rule() {
        use crate::evaluator::types::EvalAllowRoute;
        use crate::policy_registry::PolicyId;

        // Longer than the bound a route's text is held to, so the route carries
        // a truncated copy of it.
        let long_message = format!(
            "Access to the payments API requires approval. {}",
            "x".repeat(1200)
        );
        let dir = std::env::temp_dir();
        let pid = format!("c2-bound-{}", std::process::id());
        let path = dir.join(format!("{pid}.dl"));
        std::fs::write(
            &path,
            format!(
                "// @deny_message: {long_message}\n\
                 // @url_pattern: api.openai.com\n\
                 IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.openai.com\").\n"
            ),
        )
        .unwrap();
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        engine
            .sessions
            .registry()
            .install_dedup("t", &pid, factory, "mock".into(), false);
        engine
            .load_metadata_keyed(PolicyId::from_string(pid.clone()), &path)
            .unwrap();
        let action = AuthAction::HttpRequest {
            url: "https://api.fda.gov/drug/label.json".into(),
            body: String::new(),
            headers: vec![],
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
            allow_routes: vec![EvalAllowRoute {
                rule_id: "openai".into(),
                status: "unknown".into(),
                details: sasy_policy_analysis::allow_attribution::bounded_hint(&long_message),
                suggestion: String::new(),
                source_location: String::new(),
            }],
        };
        let trace = engine.build_denial_trace(&action, &denied, Some(&PolicyId::from_string(pid)));
        let _ = std::fs::remove_file(&path);
        // The rule is annotated for another host, so its wording is not an
        // explanation for this request whatever its length.
        assert!(!trace
            .reasons
            .iter()
            .any(|r| r.details.contains("payments API")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn denial_trace_pairs_routes_by_message_and_hints_and_keeps_hint_only_rules() {
        use crate::evaluator::types::EvalAllowRoute;
        use crate::policy_registry::PolicyId;

        let dir = std::env::temp_dir();
        let pid = format!("c2-pairing-{}", std::process::id());
        let path = dir.join(format!("{pid}.dl"));
        // Two rules share one denial message but name different targets and
        // hints; a third has a hint and no message at all.
        std::fs::write(
            &path,
            "// @deny_message: Access requires a role\n\
             // @suggestion: Request the openai-access role\n\
             // @url_pattern: api.openai.com\n\
             IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.openai.com\"), HasRole(\"openai-access\").\n\
             // @deny_message: Access requires a role\n\
             // @suggestion: Call register_fda_usage\n\
             // @url_pattern: api.fda.gov\n\
             IsAuthorized(idx) :- Actions(idx, a), QueriesHost(a, \"api.fda.gov\"), HasRole(\"fda-access\").\n\
             // @suggestion: Ask an approver\n\
             IsAuthorized(idx) :- Actions(idx, a), Approved(a).\n",
        )
        .unwrap();
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        engine
            .sessions
            .registry()
            .install_dedup("t", &pid, factory, "mock".into(), false);
        engine
            .load_metadata_keyed(PolicyId::from_string(pid.clone()), &path)
            .unwrap();
        let action = AuthAction::HttpRequest {
            url: "https://api.fda.gov/drug/label.json".into(),
            body: String::new(),
            headers: vec![],
        };
        let route = |rule_id: &str, details: &str, suggestion: &str| EvalAllowRoute {
            rule_id: rule_id.into(),
            status: "unknown".into(),
            details: details.into(),
            suggestion: suggestion.into(),
            source_location: String::new(),
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
            allow_routes: vec![
                route(
                    "openai",
                    "Access requires a role",
                    "Request the openai-access role",
                ),
                route("fda", "Access requires a role", "Call register_fda_usage"),
                route("approved", "", "Ask an approver"),
            ],
        };
        let trace = engine.build_denial_trace(&action, &denied, Some(&PolicyId::from_string(pid)));
        let _ = std::fs::remove_file(&path);
        assert_eq!(trace.allow_routes.len(), 3);
        let reasons: Vec<_> = trace.reasons.iter().map(|r| r.details.clone()).collect();
        // The FDA rule's wording once, and the hint-only rule as a reason
        // without wording (its hint is what the model needs).
        assert_eq!(
            reasons
                .iter()
                .filter(|r| r.as_str() == "Access requires a role")
                .count(),
            1
        );
        assert_eq!(reasons.len(), 2);
        let mut hints = trace.suggested_fixes.clone();
        hints.sort();
        assert_eq!(
            hints,
            vec![
                "Ask an approver".to_string(),
                "Call register_fda_usage".to_string()
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn denial_trace_with_only_ruled_out_routes_gives_the_generic_reason() {
        use crate::evaluator::types::EvalAllowRoute;
        use crate::policy_registry::PolicyId;

        let dir = std::env::temp_dir();
        let pid = format!("c2-ruled-out-{}", std::process::id());
        let path = dir.join(format!("{pid}.dl"));
        std::fs::write(
            &path,
            "// @deny_message: Cannot send email - recipient clearance may be insufficient\n\
             // @suggestion: Pick an internal recipient\n\
             // @tool_pattern: send_email\n\
             IsAuthorized(idx) :- Actions(idx, a), IsTool(a, \"send_email\").\n",
        )
        .unwrap();
        let engine = EvaluatorEngine::with_shared_evaluator(
            MockEvaluator::new("mock"),
            Arc::new(GraphStore::new(None).unwrap()),
        );
        let factory: EvaluatorFactory =
            Arc::new(|| Ok(MockEvaluator::new("mock") as Arc<dyn Evaluator>));
        engine
            .sessions
            .registry()
            .install_dedup("t", &pid, factory, "mock".into(), false);
        engine
            .load_metadata_keyed(PolicyId::from_string(pid.clone()), &path)
            .unwrap();
        let action = AuthAction::ToolCall {
            fn_name: "list_files".into(),
            args: "{}".into(),
        };
        let denied = EvalActionResult {
            index: 0,
            authorized: false,
            is_authenticated: true,
            is_denylisted: false,
            is_allowlisted: false,
            transform_ids: vec![],
            deny_if_unauthorized: true,
            allow_passthrough: false,
            requires_approval: false,
            denial_reasons: vec![],
            allow_routes: vec![EvalAllowRoute {
                rule_id: "send-email".into(),
                status: "blocked".into(),
                details: "Cannot send email - recipient clearance may be insufficient".into(),
                suggestion: "Pick an internal recipient".into(),
                source_location: "policy:1".into(),
            }],
        };
        let trace = engine.build_denial_trace(&action, &denied, Some(&PolicyId::from_string(pid)));
        let _ = std::fs::remove_file(&path);
        assert_eq!(trace.allow_routes.len(), 1);
        assert_eq!(trace.reasons.len(), 1);
        assert!(!trace.reasons[0].details.contains("send email"));
        assert!(trace
            .suggested_fixes
            .iter()
            .all(|s| !s.contains("internal recipient")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn engine_dispatches_to_workers() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let engine = EvaluatorEngine::with_shared_evaluator(MockEvaluator::new("test"), store);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert_eq!(engine.backend_name(), "test");
        // No live sessions yet — none have been touched.
        assert_eq!(engine.worker_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn swap_evaluator_creates_new_pool() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let engine = EvaluatorEngine::with_shared_evaluator(MockEvaluator::new("v1"), store);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(engine.backend_name(), "v1");

        engine.swap_evaluator(MockEvaluator::new("v2")).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(engine.backend_name(), "v2");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn end_session_drops_only_evaluator() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let engine = EvaluatorEngine::with_shared_evaluator(MockEvaluator::new("test"), store);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let req = sasy_common::policy_engine::Action {
            ..Default::default()
        };
        let scope = SessionScope::new("default", "abc");
        // Touch session "abc" so it spawns.
        engine
            .check_authorization(
                &[],
                std::slice::from_ref(&req),
                None,
                &[],
                &scope,
                None,
                None,
            )
            .unwrap();
        assert_eq!(engine.worker_count(), 1);

        assert!(engine.end_session(&scope));
        assert_eq!(engine.worker_count(), 0);

        // Resuming the same session re-spawns lazily.
        engine
            .check_authorization(&[], &[req], None, &[], &scope, None, None)
            .unwrap();
        assert_eq!(engine.worker_count(), 1);
    }
}

#[cfg(test)]
mod fence_tests {
    use super::*;
    use crate::evaluator::types::EvalAuthResponse;
    use sasy_common::observability::Event;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ev(id: &str) -> Event {
        Event {
            text: Some("x".into()),
            agent: None,
            role: None,
            id: Some(id.to_string()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: None,
            metadata: None,
        }
    }

    /// Pins the counter a dispatch fences on. Constructed so BOTH regressions fail:
    /// the global shard counter is non-zero yet differs from the store-wide one, so
    /// returning `get_sequence()` (a full-tenant reload on unrelated traffic) and
    /// returning a constant `0` (silently disabling the global re-sync, i.e. a
    /// session-less writer stops reading its own writes) are each caught.
    #[test]
    fn dispatch_fence_sequence_uses_the_scope_shard_not_the_store_wide_counter() {
        let store = GraphStore::new(None).unwrap();
        let tenant = "default";
        let global = SessionScope::new(tenant, "");
        let sess = SessionScope::new(tenant, "S1");

        // Unrelated session traffic moves the store-wide counter, NOT the global shard.
        store
            .merge_events(&sess, None, vec![ev("s1"), ev("s2")])
            .unwrap();
        assert_eq!(
            dispatch_fence_sequence(&store, &global),
            0,
            "unrelated traffic must not move the global fence"
        );
        assert!(
            store.get_sequence() >= 2,
            "…though it does move the store-wide counter"
        );

        // A write on the global shard — where session-less writers land — does move it.
        store.merge_events(&global, None, vec![ev("g1")]).unwrap();
        let g = dispatch_fence_sequence(&store, &global);
        assert_eq!(g, store.session_sequence(&global));
        assert!(
            g > 0,
            "a global-shard write must move the global fence (else the re-sync never fires)"
        );
        assert_ne!(
            g,
            store.get_sequence(),
            "must be the shard counter, not the store-wide one"
        );

        // Non-global scopes keep their own shard counter. (Tautological against the
        // current one-line body — kept only as a guard for a future scope-dependent
        // rewrite; the discriminating assertions are the global ones above.)
        assert_eq!(
            dispatch_fence_sequence(&store, &sess),
            store.session_sequence(&sess)
        );
    }

    /// Recording evaluator: `reset()` is the observable signal that a full re-sync
    /// happened, which is what an over-broad fence counter causes.
    struct ResetCountingEvaluator {
        resets: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl Evaluator for ResetCountingEvaluator {
        async fn update(&self, _u: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            Ok(())
        }
        async fn query(&self, _r: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
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
            "reset-counting"
        }
    }

    /// The helper above is only worth having if `check_authorization` actually CALLS
    /// it, and a unit test of the helper cannot see that — reverting the call site to
    /// `get_sequence()` leaves it green. This drives the real entry point instead:
    /// an unrelated session's write moves the store-wide counter but not the global
    /// shard's, so a global check must NOT reload the whole tenant. With the revert,
    /// it does, and `reset()` fires.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn global_check_authorization_does_not_reload_on_unrelated_session_writes() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let global = SessionScope::new("default", "");
        let resets = Arc::new(AtomicUsize::new(0));

        // Only the FIRST spawn (the global scope, checked before any other traffic)
        // gets the counting mock — the prewarm listener spawns one per written-to
        // session, and a shared instance would attribute their resets to us.
        let handed = Arc::new(AtomicUsize::new(0));
        let resets_f = Arc::clone(&resets);
        let factory: EvaluatorFactory = Arc::new(move || {
            let r = if handed.fetch_add(1, Ordering::Relaxed) == 0 {
                Arc::clone(&resets_f)
            } else {
                Arc::new(AtomicUsize::new(0))
            };
            Ok(Arc::new(ResetCountingEvaluator { resets: r }) as Arc<dyn Evaluator>)
        });
        let engine = EvaluatorEngine::new(factory, "mock".to_string(), Arc::clone(&store));

        let check = || engine.check_authorization(&[], &[], None, &[], &global, None, None);
        check().expect("first global check");
        let before = resets.load(Ordering::Relaxed);

        store
            .merge_events(&SessionScope::new("default", "S1"), None, vec![ev("s1")])
            .unwrap();
        assert!(
            store.get_sequence() > 0,
            "premise: the store-wide counter moved"
        );
        assert_eq!(
            store.session_sequence(&global),
            0,
            "premise: the GLOBAL shard's counter did not"
        );

        check().expect("second global check");
        assert_eq!(
            resets.load(Ordering::Relaxed),
            before,
            "an unrelated session's write must not force a full-tenant reload on the \
             refmon's per-request global path"
        );
    }
}
