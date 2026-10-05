//! Core policy engine trait and stub implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use sasy_common::policy_engine::{ActionResult, AuthorizationResponse};
use sasy_common::SessionScope;
use tracing::info;

/// A tool call in a message (for graph updates).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub arguments: String,
}

/// Graph update in the format the engine consumes.
///
/// Mirrors the observability server's domain types,
/// independent of proto.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GraphUpdate {
    NodeCreated {
        id: String,
        content: Option<String>,
        role: Option<String>,
        agent: Option<String>,
        tools: Vec<ToolInfo>,
        /// User-supplied actor in the caller's domain (free-form,
        /// not validated). Distinct from `principal`.
        entity: Option<String>,
        /// Server-stamped principal — the auth-derived identity of
        /// the gRPC caller that wrote this node. Immutable;
        /// populated from `MessageNode.principal` at sync time.
        #[serde(default)]
        principal: Option<String>,
        /// If this message is a tool result, the tool
        /// it was derived from.
        derived_from: Option<ToolInfo>,
        /// The adapter's canonical record of the source message
        /// (`Event.metadata`), passed through verbatim and never
        /// interpreted here. Absent when the message was recorded
        /// without one; an update that carries `None` says nothing
        /// about it rather than clearing it, so it is left off the
        /// wire entirely.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<String>,
        /// Session/conversation partition. Empty string means
        /// the legacy "global" partition.
        #[serde(default)]
        session_id: String,
    },
    NodeDeleted(String),
    EdgeCreated {
        source: String,
        destination: String,
        /// Position of `source` in the destination's recorded
        /// input history. Absent for edges that weren't recorded
        /// via a history-indexed step (e.g. flip-copy derivation).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_index: Option<u32>,
        /// True when `source` was the immediate predecessor of
        /// `destination` in the recording step's input history.
        /// Absent for edges without recorded context.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proximal: Option<bool>,
        /// Server-stamped principal — the auth-derived identity of
        /// the gRPC caller that asserted this edge. Immutable;
        /// populated from `Edge.principal` at sync time. Absent when
        /// the recording client carried no auth-derived identity.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        principal: Option<String>,
        /// User-supplied actor in the caller's domain (free-form,
        /// not validated) that the recording client named for this
        /// edge. Distinct from `principal`; absent when not sent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entity: Option<String>,
        /// Session/conversation partition for this edge.
        #[serde(default)]
        session_id: String,
    },
    EdgeDeleted {
        source: String,
        destination: String,
    },
    /// Drop all nodes and edges in this session. Workers remove the
    /// session's tuples from their indices and any cached evaluator
    /// state.
    DropSession(String),
}

/// Synchronization status of the engine.
#[derive(Debug, Clone)]
pub struct SyncStatus {
    pub current_sequence: i64,
    pub node_count: usize,
    pub edge_count: usize,
    pub connected: bool,
}

/// Trait abstracting the policy engine.
///
/// This allows the binary to inject any backend
/// (Soufflé, FlowLog, plugin) while tests
/// and stubs use simpler impls.
pub trait Engine: Send + Sync {
    /// Apply graph updates (persistent facts).
    fn apply_graph_updates(&self, updates: Vec<GraphUpdate>) -> Result<(), anyhow::Error>;

    /// Check authorization for actions.
    ///
    /// ``scope`` is the `(tenant, session)` partition the evaluator
    /// scopes its view to. A global scope (`session.is_empty()`)
    /// covers every session within the tenant; a concrete scope
    /// restricts evaluation to its own session's tuples.
    /// ``principal`` is the server-stamped auth-derived identity,
    /// distinct from ``entity``, which stays user-supplied.
    /// ``policy_id`` is the variant id from
    /// [`AuthorizationRequest::policy_id`]. On the first
    /// request for a scope, this binds the scope→policy mapping;
    /// subsequent requests must either omit it or send the same
    /// value (else error). Unset = use the tenant default.
    #[allow(clippy::too_many_arguments)] // inherent: full authorization context (entity, principal, roles, scope, policy_id, ...)
    fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: &[sasy_common::policy_engine::Action],
        entity: Option<&str>,
        roles: &[String],
        scope: &SessionScope,
        principal: Option<&str>,
        policy_id: Option<&str>,
    ) -> Result<AuthorizationResponse, anyhow::Error>;

    /// Reset all state (for full-state refresh).
    fn reset(&self) -> Result<(), anyhow::Error>;

    /// Load rule metadata from a .dl file.
    fn load_rule_metadata(&self, path: &std::path::Path) -> Result<(), anyhow::Error>;

    /// Like [`Self::load_rule_metadata`], but associates the metadata
    /// with a specific resolved policy id so multi-policy tenants
    /// attribute each denial trace to the deciding policy's rule
    /// metadata. The default delegates to the policy-id-less
    /// [`Self::load_rule_metadata`] — correct for single-policy
    /// engines (stub, plugin/C-ABI, replay).
    fn load_rule_metadata_for_policy(
        &self,
        policy_id: &str,
        path: &std::path::Path,
    ) -> Result<(), anyhow::Error> {
        let _ = policy_id;
        self.load_rule_metadata(path)
    }

    /// Get current sync status.
    fn get_sync_status(&self) -> SyncStatus;

    /// Set connected flag.
    fn set_connected(&self, connected: bool);

    /// Set current sequence number.
    fn set_sequence(&self, seq: i64);

    /// Return the active evaluator backend name.
    fn backend_name(&self) -> String {
        "unknown".to_string()
    }

    /// Number of evaluator worker processes.
    fn worker_count(&self) -> usize {
        1
    }

    /// Swap the evaluator backend at runtime, sharing one
    /// instance across every live session. Useful for tests;
    /// per-session deployments use [`Self::swap_evaluator_factory`].
    fn swap_evaluator(
        &self,
        _new_evaluator: Arc<dyn crate::evaluator::Evaluator>,
    ) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!(
            "swap_evaluator not supported by this engine"
        ))
    }

    /// Swap the per-session evaluator factory at runtime. The
    /// factory is invoked once per session at first traffic; the
    /// resulting evaluator is bootstrapped with that session's
    /// existing graph state before serving queries.
    fn swap_evaluator_factory(
        &self,
        _factory: crate::session_evaluator::EvaluatorFactory,
        _backend_name: String,
    ) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!(
            "swap_evaluator_factory not supported by this engine"
        ))
    }

    /// Drop every live evaluator in `tenant`, leaving bindings in place.
    /// The tenant-wide form of [`Self::evict_session_evaluator`], for undoing
    /// a tenant-wide configuration write. Default: no-op.
    fn evict_tenant_evaluators(&self, _tenant: &str) -> usize {
        0
    }

    /// Drop a session's live evaluator while leaving its binding in place,
    /// so the next call rebuilds it and re-reads its configuration.
    ///
    /// Needed after a configuration write is rolled back: `check_authorization`
    /// takes no lock, so it can have spawned an evaluator that seeded from the
    /// row in the window before the rollback put the old one back, and that
    /// seed lasts the evaluator's lifetime.
    ///
    /// Default: no-op for engines with no per-session evaluators.
    fn evict_session_evaluator(&self, _scope: &SessionScope) -> bool {
        false
    }

    /// End a session: drop its live evaluator AND its policy binding. The
    /// graph state in the store is left untouched, so a future query for the
    /// same scope re-spawns — but unbound, so under the tenant default rather
    /// than whatever this session had been pinned to. Use
    /// [`Self::evict_session_evaluator`] to drop only the evaluator.
    ///
    /// Returns ``true`` iff a live evaluator was actually dropped. Default:
    /// no-op (false), so non-evaluator engines (Stub, plugin) ignore the call.
    fn end_session(&self, _scope: &SessionScope) -> bool {
        false
    }

    /// Append dynamic metadata facts to `scope` and, if a live
    /// evaluator exists, re-seed it so the new facts take effect on
    /// the next query without an evict/re-bootstrap. Facts persist
    /// (RocksDB session-metadata CF) so they survive evaluator
    /// respawns. Append-only with deny-precedence semantics is a
    /// policy concern; this just unions the tuples.
    ///
    /// Used by the detaint recorder: an `@ask` approval records
    /// `PolicyMetadata("detaint_approved", <node>, "")`; a denial
    /// records `detaint_denied`.
    ///
    /// Default: no-op, so non-evaluator engines (Stub, plugin)
    /// silently accept and ignore.
    fn update_session_metadata(
        &self,
        _scope: &SessionScope,
        _facts: Vec<crate::evaluator::types::PolicyMetadataFact>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Install a freshly-compiled policy for `tenant` and return
    /// its server-assigned id. `mode` controls the side
    /// effects:
    ///
    /// * [`InstallMode::Variant`] — install only. No default change,
    ///   no session eviction. The returned id is bindable via
    ///   [`Self::set_session_policy`].
    /// * [`InstallMode::Default`] — install + set as the tenant's
    ///   default for newly-spawned unpinned sessions. Existing
    ///   sessions keep their bindings (gradual rollout).
    /// * [`InstallMode::Force`] — install + set as default + evict
    ///   every live session in the tenant. They re-bind to the new
    ///   default on next traffic. Disruptive — break-glass / eager
    ///   rollout.
    ///
    /// `content_hash` is the upload's stable content fingerprint
    /// (sha256 of source + functors + backend + ...). Repeated
    /// uploads with the same `(tenant, content_hash)` collapse to
    /// the same `policy_id` so the registry doesn't bloat under
    /// identical-upload workloads.
    ///
    /// Default: errors out, so non-evaluator engines (Stub, plugin)
    /// don't accept policy uploads. EvaluatorEngine overrides.
    fn install_policy(
        &self,
        _tenant: &str,
        _content_hash: &str,
        _factory: crate::session_evaluator::EvaluatorFactory,
        _backend_name: String,
        _mode: InstallMode,
    ) -> Result<String, anyhow::Error> {
        Err(anyhow::anyhow!(
            "install_policy not supported by this engine"
        ))
    }

    /// Look up an existing policy id by its content hash.
    /// Returns the id if `(tenant, content_hash)` was already
    /// uploaded — letting the gRPC handler short-circuit a
    /// duplicate upload without recompiling. `None` means the
    /// content hasn't been seen before; the caller proceeds with
    /// the full compile path.
    ///
    /// Default: returns `None`, so non-evaluator engines never
    /// dedup (they don't have a registry).
    fn lookup_policy_by_content(&self, _tenant: &str, _content_hash: &str) -> Option<String> {
        None
    }

    /// Resolve a baked/curated profile NAME to its policy id (== content hash),
    /// for `SetPolicy(bind_profile_name)`. Default: `None` (only the restricted
    /// EvaluatorEngine, which pre-installs named baked profiles, overrides it).
    fn lookup_policy_by_name(&self, _tenant: &str, _name: &str) -> Option<String> {
        None
    }

    /// Register a `(tenant, name) → policy_id` mapping so the policy can later be
    /// bound by name. Default: no-op (only the registry-backed EvaluatorEngine
    /// records it; used at startup for baked curated profiles).
    fn register_profile_name(&self, _tenant: &str, _name: &str, _policy_id: &str) {}

    /// Rebind `scope` to `policy_id`. Evicts the current
    /// per-session evaluator; the next `check_authorization` call
    /// for `scope` re-spawns under the new policy and bootstraps
    /// from the same preserved graph state. Errors if `policy_id`
    /// is unknown in the scope's tenant.
    ///
    /// Default: errors. EvaluatorEngine overrides.
    fn set_session_policy(
        &self,
        _scope: &SessionScope,
        _policy_id: &str,
    ) -> Result<bool, anyhow::Error> {
        Err(anyhow::anyhow!(
            "set_session_policy not supported by this engine"
        ))
    }

    /// Apply the Default/Force side effects for an *already-
    /// installed* content hash. Used by the SetPolicy dedup
    /// short-circuit: a re-upload of a previously-seen source
    /// should still flip the tenant default (and, for Force,
    /// evict live sessions) even though no compile happens.
    /// Without this, `SetPolicy(scope=Default)` against a deduped
    /// source returns success but the runtime default is
    /// unchanged until restart — a real correctness gap.
    ///
    /// `force=true` evicts every live session in the tenant
    /// (matching `InstallMode::Force` semantics). Default: errors,
    /// so non-evaluator engines reject the call.
    fn promote_to_default(
        &self,
        _tenant: &str,
        _content_hash: &str,
        _force: bool,
    ) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!(
            "promote_to_default not supported by this engine"
        ))
    }

    /// Boot-replay hook: rehydrate a `(scope → content_hash)`
    /// binding read from RocksDB. Does NOT validate that the
    /// referenced policy is compiled — that happens lazily on the
    /// first dispatch for `scope`. Default: no-op (engines that
    /// don't persist anything just discard the rehydrate signal).
    fn lazy_bind_session(&self, _scope: SessionScope, _content_hash: &str) {}

    /// Boot-replay hook: rehydrate a `(tenant → content_hash)`
    /// default. Same lazy semantics as
    /// [`Self::lazy_bind_session`]; the entry is materialized on
    /// the first dispatch from an unbound session under `tenant`.
    fn lazy_set_default(&self, _tenant: &str, _content_hash: &str) {}
}

/// Side-effect mode for [`Engine::install_policy`].
///
/// Maps onto the proto `PolicyScope` oneof — the service handler
/// translates `SessionTarget` → `Variant`, `DefaultTarget` →
/// `Default`, `ForceTarget` → `Force`. SessionTarget is paired
/// with a follow-up [`Engine::set_session_policy`] call to bind
/// the freshly-installed policy to the requested session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMode {
    /// Install in the registry. No default change. No eviction.
    /// Bindable via [`Engine::set_session_policy`].
    Variant,
    /// Install + set as the tenant's default policy for new
    /// unpinned sessions. Existing sessions keep their bindings.
    Default,
    /// Install + set as the tenant default + evict every live
    /// session in the tenant. They re-bind on next traffic.
    Force,
}

/// Stub engine that allows everything.
///
/// Used when no policy backend is configured, or
/// for testing without a real evaluator.
pub struct StubEngine {
    connected: AtomicBool,
    sequence: RwLock<i64>,
    node_count: RwLock<usize>,
    edge_count: RwLock<usize>,
}

impl StubEngine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            connected: AtomicBool::new(false),
            sequence: RwLock::new(0),
            node_count: RwLock::new(0),
            edge_count: RwLock::new(0),
        })
    }
}

impl Default for StubEngine {
    fn default() -> Self {
        Self {
            connected: AtomicBool::new(false),
            sequence: RwLock::new(0),
            node_count: RwLock::new(0),
            edge_count: RwLock::new(0),
        }
    }
}

impl Engine for StubEngine {
    fn apply_graph_updates(&self, updates: Vec<GraphUpdate>) -> Result<(), anyhow::Error> {
        let mut nc = self.node_count.write();
        let mut ec = self.edge_count.write();
        for u in &updates {
            match u {
                GraphUpdate::NodeCreated { .. } => {
                    *nc += 1;
                }
                GraphUpdate::NodeDeleted(_) => {
                    *nc = nc.saturating_sub(1);
                }
                GraphUpdate::EdgeCreated { .. } => {
                    *ec += 1;
                }
                GraphUpdate::EdgeDeleted { .. } => {
                    *ec = ec.saturating_sub(1);
                }
                GraphUpdate::DropSession(_) => {
                    // StubEngine doesn't track per-session counts; no-op.
                }
            }
        }
        Ok(())
    }

    fn check_authorization(
        &self,
        _current_node_ids: &[String],
        actions: &[sasy_common::policy_engine::Action],
        _entity: Option<&str>,
        _roles: &[String],
        _scope: &SessionScope,
        _principal: Option<&str>,
        _policy_id: Option<&str>,
    ) -> Result<AuthorizationResponse, anyhow::Error> {
        info!("stub engine: allowing {} action(s)", actions.len());
        let results = actions
            .iter()
            .enumerate()
            .map(|(i, _)| ActionResult {
                index: i as u32,
                authorized: true,
                trace: None,
                transform_ids: vec![],
                deny_if_unauthorized: false,
            })
            .collect();
        Ok(AuthorizationResponse {
            results,
            timing: None,
        })
    }

    fn reset(&self) -> Result<(), anyhow::Error> {
        *self.node_count.write() = 0;
        *self.edge_count.write() = 0;
        Ok(())
    }

    fn load_rule_metadata(&self, _path: &std::path::Path) -> Result<(), anyhow::Error> {
        Ok(())
    }

    fn get_sync_status(&self) -> SyncStatus {
        SyncStatus {
            current_sequence: *self.sequence.read(),
            node_count: *self.node_count.read(),
            edge_count: *self.edge_count.read(),
            connected: self.connected.load(Ordering::Relaxed),
        }
    }

    fn set_connected(&self, connected: bool) {
        self.connected.store(connected, Ordering::Relaxed);
    }

    fn set_sequence(&self, seq: i64) {
        *self.sequence.write() = seq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_allows_everything() {
        let engine = StubEngine::new();
        let resp = engine
            .check_authorization(
                &["n1".into()],
                &[],
                Some("user"),
                &["admin".into()],
                &SessionScope::global("default"),
                None,
                None,
            )
            .unwrap();
        assert!(resp.results.is_empty());
    }

    /// The edge update serialises its attribution when present and
    /// omits both fields entirely when absent, so a shim reading the
    /// JSON sees `null` for an edge nobody's identity was recorded
    /// for.
    #[test]
    fn edge_update_round_trips_its_attribution() {
        let attributed = GraphUpdate::EdgeCreated {
            source: "n1".into(),
            destination: "n2".into(),
            message_index: Some(3),
            proximal: Some(true),
            principal: Some("gateway".into()),
            entity: Some("ingest-worker".into()),
            session_id: "s1".into(),
        };
        let json = serde_json::to_string(&attributed).unwrap();
        assert!(json.contains("\"principal\":\"gateway\""));
        assert!(json.contains("\"entity\":\"ingest-worker\""));
        let back: GraphUpdate = serde_json::from_str(&json).unwrap();
        match back {
            GraphUpdate::EdgeCreated {
                principal, entity, ..
            } => {
                assert_eq!(principal.as_deref(), Some("gateway"));
                assert_eq!(entity.as_deref(), Some("ingest-worker"));
            }
            other => panic!("expected EdgeCreated, got {other:?}"),
        }

        let anonymous = GraphUpdate::EdgeCreated {
            source: "n1".into(),
            destination: "n2".into(),
            message_index: None,
            proximal: None,
            principal: None,
            entity: None,
            session_id: String::new(),
        };
        let json = serde_json::to_string(&anonymous).unwrap();
        assert!(
            !json.contains("principal"),
            "absent principal is not written"
        );
        assert!(!json.contains("entity"), "absent entity is not written");
        // An update written before these fields existed still parses.
        let legacy: GraphUpdate = serde_json::from_str(
            r#"{"EdgeCreated":{"source":"n1","destination":"n2","session_id":""}}"#,
        )
        .unwrap();
        match legacy {
            GraphUpdate::EdgeCreated {
                principal, entity, ..
            } => {
                assert!(principal.is_none());
                assert!(entity.is_none());
            }
            other => panic!("expected EdgeCreated, got {other:?}"),
        }
    }

    #[test]
    fn stub_tracks_counts() {
        let engine = StubEngine::new();
        engine
            .apply_graph_updates(vec![
                GraphUpdate::NodeCreated {
                    id: "n1".into(),
                    content: None,
                    role: None,
                    agent: None,
                    tools: vec![],
                    entity: None,
                    principal: None,
                    derived_from: None,
                    metadata: None,
                    session_id: String::new(),
                },
                GraphUpdate::EdgeCreated {
                    source: "n1".into(),
                    destination: "n2".into(),
                    message_index: None,
                    proximal: None,
                    principal: None,
                    entity: None,
                    session_id: String::new(),
                },
            ])
            .unwrap();
        let status = engine.get_sync_status();
        assert_eq!(status.node_count, 1);
        assert_eq!(status.edge_count, 1);
    }
}
