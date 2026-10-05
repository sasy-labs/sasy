//! Broadcast-based sync manager.
//!
//! Subscribes to [`sasy_graph::GraphStore`]'s broadcast
//! channel and feeds updates into the policy [`Engine`].

use std::sync::Arc;

use sasy_common::observability::{Edge, Event};
use sasy_common::SessionScope;
use sasy_graph::GraphStore;
use tracing::{info, warn};

use crate::engine::{Engine, GraphUpdate, ToolInfo};

/// Subscribes to graph broadcast and feeds the engine.
pub struct SyncManager<E: Engine> {
    engine: Arc<E>,
    store: Arc<GraphStore>,
}

impl<E: Engine + 'static> SyncManager<E> {
    pub fn new(engine: Arc<E>, store: Arc<GraphStore>) -> Self {
        Self { engine, store }
    }

    /// Bootstrap from current state, then stream.
    ///
    /// Spawns a background task; call once at startup.
    pub fn start(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            // 1. Load full state
            match self.store.get_full_state() {
                Ok((events, edges, seq)) => {
                    let updates = full_state_to_updates(&events, &edges, "");
                    if let Err(e) = self.engine.apply_graph_updates(updates) {
                        warn!(
                            "failed to apply full state: \
                             {e}"
                        );
                    }
                    self.engine.set_sequence(seq);
                    self.engine.set_connected(true);
                    info!(
                        seq,
                        events = events.len(),
                        edges = edges.len(),
                        "policy engine synced full state"
                    );
                }
                Err(e) => {
                    warn!("failed to get full state: {e}");
                }
            }

            // 2. Subscribe to broadcast
            let mut rx = self.store.subscribe();

            loop {
                match rx.recv().await {
                    Ok(gu) => {
                        let updates = broadcast_to_updates(&gu);
                        if !updates.is_empty() {
                            if let Err(e) = self.engine.apply_graph_updates(updates) {
                                warn!(
                                    "apply update failed: \
                                     {e}"
                                );
                            }
                        }
                        let seq = self.store.get_sequence();
                        self.engine.set_sequence(seq);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            "broadcast lagged {n}, \
                             re-syncing"
                        );
                        if let Err(e) = self.engine.reset() {
                            warn!("reset failed: {e}");
                        }
                        if let Ok((events, edges, seq)) = self.store.get_full_state() {
                            let updates = full_state_to_updates(&events, &edges, "");
                            let _ = self.engine.apply_graph_updates(updates);
                            self.engine.set_sequence(seq);
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        info!("broadcast channel closed");
                        self.engine.set_connected(false);
                        break;
                    }
                }
            }
        })
    }
}

/// Convert a proto Tool to ToolInfo.
fn tool_to_info(tool: &sasy_common::observability::Tool) -> ToolInfo {
    ToolInfo {
        name: tool.name.clone().unwrap_or_default(),
        arguments: tool.arguments.clone().unwrap_or_default(),
    }
}

/// Convert Event to GraphUpdate::NodeCreated with all fields.
///
/// `entity` is user-supplied (free-form actor in the caller's
/// domain) and `principal` is the server-stamped auth-derived
/// identity (immutable, set at the gRPC boundary). Both flow
/// through to the Soufflé `Message` record so policies can match
/// on either dimension — see `common_policy.dl`.
fn event_to_node_created(ev: &Event, id: String, session_id: &str) -> GraphUpdate {
    GraphUpdate::NodeCreated {
        id,
        content: ev.text.clone(),
        // Canonical proto-Role -> lowercase string via the shared
        // helper (explicit mapping), not Debug-derive — so reordering
        // or renaming a Role variant can't silently shift the string
        // that's persisted + matched in policies.
        role: ev
            .role
            .and_then(sasy_common::MessageRole::from_proto_opt)
            .map(|m| m.as_str().to_string()),
        agent: ev.agent.clone(),
        tools: ev.tools.iter().map(tool_to_info).collect(),
        entity: ev.entity.clone(),
        principal: ev.principal.clone(),
        derived_from: ev.derived_from.as_ref().map(tool_to_info),
        metadata: ev.metadata.clone(),
        session_id: session_id.to_string(),
    }
}

pub(crate) fn full_state_to_updates(
    events: &[Event],
    edges: &[sasy_common::observability::Edge],
    session_id: &str,
) -> Vec<GraphUpdate> {
    let mut updates = Vec::with_capacity(events.len() + edges.len());
    for ev in events {
        let id = ev.id.clone().unwrap_or_default();
        updates.push(event_to_node_created(ev, id, session_id));
    }
    for e in edges {
        updates.push(GraphUpdate::EdgeCreated {
            source: e.source.clone(),
            destination: e.destination.clone(),
            message_index: e.message_index,
            proximal: e.proximal,
            principal: e.principal.clone(),
            entity: e.entity.clone(),
            session_id: session_id.to_string(),
        });
    }
    updates
}

/// Convert *scope-preserving* full state — each Event/Edge paired
/// with its owning [`SessionScope`] — into [`GraphUpdate`]s,
/// preserving each item's own `session_id`. This mirrors
/// [`broadcast_to_updates`] (which derives `session_id` from each
/// update's `scope.session()`) so a global-scope evaluator's
/// bootstrap reproduces exactly the attribution the live stream
/// would have produced. Contrast [`full_state_to_updates`], which
/// flattens every item to one `session_id` — correct for a
/// single-session shard, wrong for the tenant-wide global view.
pub(crate) fn full_state_scoped_to_updates(
    events: &[(SessionScope, Event)],
    edges: &[(SessionScope, Edge)],
) -> Vec<GraphUpdate> {
    let mut updates = Vec::with_capacity(events.len() + edges.len());
    for (scope, ev) in events {
        let id = ev.id.clone().unwrap_or_default();
        updates.push(event_to_node_created(ev, id, scope.session()));
    }
    for (scope, e) in edges {
        updates.push(GraphUpdate::EdgeCreated {
            source: e.source.clone(),
            destination: e.destination.clone(),
            message_index: e.message_index,
            proximal: e.proximal,
            principal: e.principal.clone(),
            entity: e.entity.clone(),
            session_id: scope.session().to_string(),
        });
    }
    updates
}

pub(crate) fn broadcast_to_updates(gu: &sasy_graph::GraphUpdate) -> Vec<GraphUpdate> {
    use sasy_graph::GraphUpdate as GU;
    match gu {
        GU::NodeCreated {
            id, event, scope, ..
        } => {
            vec![event_to_node_created(event, id.clone(), scope.session())]
        }
        GU::NodeDeleted(id) => {
            vec![GraphUpdate::NodeDeleted(id.clone())]
        }
        GU::EdgeCreated {
            source,
            destination,
            message_index,
            proximal,
            scope,
            principal,
            entity,
            ..
        } => vec![GraphUpdate::EdgeCreated {
            source: source.clone(),
            destination: destination.clone(),
            message_index: *message_index,
            proximal: *proximal,
            principal: principal.clone(),
            entity: entity.clone(),
            session_id: scope.session().to_string(),
        }],
        GU::EdgeDeleted {
            source,
            destination,
            ..
        } => vec![GraphUpdate::EdgeDeleted {
            source: source.clone(),
            destination: destination.clone(),
        }],
        GU::DropSession(scope) => {
            vec![GraphUpdate::DropSession(scope.session().to_string())]
        }
        GU::ComputationCreated { .. }
        | GU::ComputationDeleted(_)
        | GU::ComputationEdgeCreated { .. }
        | GU::ComputationMessageLinkCreated { .. }
        | GU::Heartbeat
        | GU::Sequence(_)
        | GU::SessionSequence { .. } => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::StubEngine;
    use sasy_common::observability::{Event, Role};

    fn ev(id: &str, text: &str) -> Event {
        Event {
            text: Some(text.to_string()),
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

    /// `full_state_scoped_to_updates` must keep each item's OWN
    /// session_id (mirroring the live `broadcast_to_updates`), not
    /// flatten everything to one session — this is what makes a
    /// global-scope evaluator's cold bootstrap match its live
    /// steady state (c5).
    #[test]
    fn scoped_full_state_preserves_per_item_session_id() {
        let s1 = SessionScope::new("acme", "s1");
        let s2 = SessionScope::new("acme", "s2");
        let g = SessionScope::global("acme");

        let events = vec![
            (s1.clone(), ev("a1", "one")),
            (s2.clone(), ev("a2", "two")),
            (g.clone(), ev("g1", "three")),
        ];
        let edges = vec![(
            s1.clone(),
            Edge {
                source: "a1".to_string(),
                destination: "a2".to_string(),
                ..Default::default()
            },
        )];

        let updates = full_state_scoped_to_updates(&events, &edges);

        let node_sid = |id: &str| -> Option<String> {
            updates.iter().find_map(|u| match u {
                GraphUpdate::NodeCreated {
                    id: nid,
                    session_id,
                    ..
                } if nid == id => Some(session_id.clone()),
                _ => None,
            })
        };
        // Each node keeps its own session_id — not flattened.
        assert_eq!(node_sid("a1").as_deref(), Some("s1"));
        assert_eq!(node_sid("a2").as_deref(), Some("s2"));
        assert_eq!(node_sid("g1").as_deref(), Some("")); // global = empty

        // The edge inherits its scope's session_id too.
        let edge_sid = updates.iter().find_map(|u| match u {
            GraphUpdate::EdgeCreated {
                source, session_id, ..
            } if source == "a1" => Some(session_id.clone()),
            _ => None,
        });
        assert_eq!(edge_sid.as_deref(), Some("s1"));
    }

    /// The live path: an edge broadcast by the graph store arrives
    /// at the engine with the principal that asserted it and the
    /// entity the recording client named.
    #[test]
    fn broadcast_edge_forwards_principal_and_entity() {
        let gu = sasy_graph::GraphUpdate::EdgeCreated {
            source: "a1".to_string(),
            destination: "a2".to_string(),
            kind: sasy_graph::types::EdgeKindSer::DependsOn,
            message_index: Some(0),
            proximal: Some(true),
            scope: SessionScope::new("acme", "s1"),
            principal: Some("gateway".to_string()),
            entity: Some("ingest-worker".to_string()),
        };

        let updates = broadcast_to_updates(&gu);
        match updates.as_slice() {
            [GraphUpdate::EdgeCreated {
                principal, entity, ..
            }] => {
                assert_eq!(principal.as_deref(), Some("gateway"));
                assert_eq!(entity.as_deref(), Some("ingest-worker"));
            }
            other => panic!("expected one EdgeCreated, got {other:?}"),
        }
    }

    /// The replay path: a cold bootstrap rebuilt from stored state
    /// carries the same attribution the live stream would have,
    /// under both the flat and the scope-preserving conversion.
    #[test]
    fn replayed_edge_keeps_the_stored_principal() {
        let stored = Edge {
            source: "a1".to_string(),
            destination: "a2".to_string(),
            principal: Some("gateway".to_string()),
            entity: Some("ingest-worker".to_string()),
            ..Default::default()
        };

        let flat = full_state_to_updates(&[], std::slice::from_ref(&stored), "s1");
        let scoped =
            full_state_scoped_to_updates(&[], &[(SessionScope::new("acme", "s1"), stored.clone())]);

        for updates in [flat, scoped] {
            match updates.as_slice() {
                [GraphUpdate::EdgeCreated {
                    principal, entity, ..
                }] => {
                    assert_eq!(principal.as_deref(), Some("gateway"));
                    assert_eq!(entity.as_deref(), Some("ingest-worker"));
                }
                other => panic!("expected one EdgeCreated, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn sync_from_full_state() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        store
            .merge_events(
                &sasy_common::SessionScope::global("default"),
                None,
                vec![ev("a", "one"), ev("b", "two")],
            )
            .unwrap();

        let engine = StubEngine::new();
        let mgr = SyncManager::new(Arc::clone(&engine), Arc::clone(&store));
        let handle = mgr.start();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let status = engine.get_sync_status();
        assert_eq!(status.node_count, 2);
        assert!(status.connected);

        handle.abort();
    }

    #[tokio::test]
    async fn sync_receives_live_updates() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let engine = StubEngine::new();
        let mgr = SyncManager::new(Arc::clone(&engine), Arc::clone(&store));
        let handle = mgr.start();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        store
            .merge_events(
                &sasy_common::SessionScope::global("default"),
                None,
                vec![ev("c", "three")],
            )
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let status = engine.get_sync_status();
        assert_eq!(status.node_count, 1);

        handle.abort();
    }
}
