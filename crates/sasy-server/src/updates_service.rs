//! gRPC ObservabilityUpdates service (state + streaming).

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use sasy_auth::request_tenant;
use sasy_common::observability::{
    observability_updates_server::ObservabilityUpdates, stream_message::Message as StreamMsg,
    ComputationEdge, ComputationMessageEdge, ComputationMessageEdgeType, Edge, GraphState,
    GraphUpdate as ProtoGraphUpdate, GraphUpdates, StateRequest, StreamMessage, UpdateType,
};
use sasy_common::SessionScope;
use sasy_graph::GraphStore;
use tokio::sync::broadcast;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, warn};

use sasy_graph::GraphUpdate;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Wraps a [`GraphStore`] and implements the
/// `ObservabilityUpdates` gRPC service.
pub struct UpdatesService {
    store: Arc<GraphStore>,
}

impl UpdatesService {
    pub fn new(store: Arc<GraphStore>) -> Self {
        Self { store }
    }
}

#[tonic::async_trait]
impl ObservabilityUpdates for UpdatesService {
    async fn get_state(
        &self,
        _request: Request<StateRequest>,
    ) -> Result<Response<GraphState>, Status> {
        crate::rbac::check_role(&_request, sasy_common::roles::OBSERVABILITY_READER)?;
        let tenant = request_tenant(&_request, "default");
        let (events, edges, sequence) = self
            .store
            .get_full_state_for_tenant(&tenant)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(GraphState {
            events,
            edges,
            sequence,
        }))
    }

    type StreamUpdatesStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<GraphUpdates, Status>> + Send>>;

    async fn stream_updates(
        &self,
        request: Request<Streaming<StreamMessage>>,
    ) -> Result<Response<Self::StreamUpdatesStream>, Status> {
        crate::rbac::check_role(&request, sasy_common::roles::OBSERVABILITY_READER)?;
        let tenant = request_tenant(&request, "default");
        let tenant_filter = SessionScope::global(tenant.clone());
        let mut client_stream = request.into_inner();
        let store = Arc::clone(&self.store);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<GraphUpdates, Status>>(256);

        tokio::spawn(async move {
            // Wait for the initial StreamRequest
            let from_seq = match client_stream.next().await {
                Some(Ok(msg)) => match msg.message {
                    Some(StreamMsg::Subscribe(req)) => req.from_sequence,
                    _ => 0,
                },
                _ => return,
            };

            debug!(
                from_sequence = from_seq,
                tenant = %tenant,
                "stream_updates: client subscribed"
            );

            let mut rx_updates: broadcast::Receiver<GraphUpdate> = store.subscribe();

            // If client requests full state (from_seq 0),
            // send it
            if from_seq == 0 {
                if let Ok((events, edges, seq)) = store.get_full_state_for_tenant(&tenant) {
                    let full_state = ProtoGraphUpdate {
                        r#type: UpdateType::FullState as i32,
                        full_state: Some(GraphState {
                            events,
                            edges,
                            sequence: seq,
                        }),
                        ..Default::default()
                    };
                    let batch = GraphUpdates {
                        sequence: seq,
                        updates: vec![full_state],
                    };
                    if tx.send(Ok(batch)).await.is_err() {
                        return;
                    }
                }
            }

            let mut seq = store.get_sequence();
            let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
            // Skip the first immediate tick
            heartbeat.tick().await;

            loop {
                tokio::select! {
                    update = rx_updates.recv() => {
                        match update {
                            Ok(gu) => {
                                if !update_matches_tenant(&gu, &tenant_filter) {
                                    continue;
                                }
                                seq += 1;
                                let proto = graph_update_to_proto(&gu);
                                let batch = GraphUpdates {
                                    sequence: seq,
                                    updates: vec![proto],
                                };
                                if tx.send(Ok(batch)).await.is_err() {
                                    break;
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                warn!("stream lagged by {n}, re-sending full state");
                                // Re-send full state to catch up
                                if let Ok((events, edges, new_seq)) = store.get_full_state_for_tenant(&tenant) {
                                    seq = new_seq;
                                    let full_state = ProtoGraphUpdate {
                                        r#type: UpdateType::FullState as i32,
                                        full_state: Some(GraphState {
                                            events,
                                            edges,
                                            sequence: seq,
                                        }),
                                        ..Default::default()
                                    };
                                    let batch = GraphUpdates {
                                        sequence: seq,
                                        updates: vec![full_state],
                                    };
                                    if tx.send(Ok(batch)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                break;
                            }
                        }
                    }
                    msg = client_stream.next() => {
                        match msg {
                            Some(Ok(m)) => {
                                if let Some(StreamMsg::Ack(ack)) = m.message {
                                    debug!(
                                        position = ack.position,
                                        current = seq,
                                        lag = seq - ack.position,
                                        "client ack"
                                    );
                                }
                            }
                            Some(Err(e)) => {
                                warn!("client stream error: {e}");
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = heartbeat.tick() => {
                        let hb = ProtoGraphUpdate {
                            r#type: UpdateType::Heartbeat as i32,
                            ..Default::default()
                        };
                        let batch = GraphUpdates {
                            sequence: seq,
                            updates: vec![hb],
                        };
                        if tx.send(Ok(batch)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }
}

/// Per-tenant filter on the broadcast: forward only updates whose
/// scope (when present) matches `tenant_filter` (a global scope on
/// the caller's tenant). Variants without a scope (Heartbeat,
/// Sequence, ComputationEdge*, ComputationMessageLink*) carry no
/// tenant data and are forwarded unconditionally — they are status
/// markers, not data leaks.
fn update_matches_tenant(gu: &GraphUpdate, tenant_filter: &SessionScope) -> bool {
    match gu {
        GraphUpdate::NodeCreated { scope, .. }
        | GraphUpdate::EdgeCreated { scope, .. }
        | GraphUpdate::ComputationCreated { scope, .. } => tenant_filter.matches(scope),
        GraphUpdate::DropSession(scope) => tenant_filter.matches(scope),
        GraphUpdate::SessionSequence { scope, .. } => tenant_filter.matches(scope),
        // Computation linkages carry their scope, so cross-tenant
        // subscribers do not see another tenant's span/message
        // graph topology.
        GraphUpdate::ComputationEdgeCreated { scope, .. }
        | GraphUpdate::ComputationMessageLinkCreated { scope, .. } => tenant_filter.matches(scope),
        // A deletion carries its scope too, so it is filtered like the
        // creation it undoes rather than shown to every tenant.
        GraphUpdate::EdgeDeleted { scope, .. } => tenant_filter.matches(scope),
        GraphUpdate::NodeDeleted(_)
        | GraphUpdate::ComputationDeleted(_)
        | GraphUpdate::Heartbeat
        | GraphUpdate::Sequence(_) => true,
    }
}

/// Convert a domain [`GraphUpdate`] to a proto
/// [`ProtoGraphUpdate`].
///
/// Stamps `session_id` from the broadcast's [`SessionScope`] onto
/// the proto envelope so the receiver can route per-session
/// without re-deriving scope from the payload.
fn graph_update_to_proto(gu: &GraphUpdate) -> ProtoGraphUpdate {
    fn sid(scope: &SessionScope) -> Option<String> {
        if scope.is_global() {
            None
        } else {
            Some(scope.session().to_string())
        }
    }
    match gu {
        GraphUpdate::NodeCreated {
            id: _,
            event,
            scope,
            ..
        } => ProtoGraphUpdate {
            r#type: UpdateType::NodeCreated as i32,
            event: Some(event.clone()),
            session_id: sid(scope),
            ..Default::default()
        },
        GraphUpdate::NodeDeleted(id) => ProtoGraphUpdate {
            r#type: UpdateType::NodeDeleted as i32,
            node_id: Some(id.clone()),
            ..Default::default()
        },
        GraphUpdate::EdgeCreated {
            source,
            destination,
            message_index,
            proximal,
            principal,
            entity,
            scope,
            ..
        } => ProtoGraphUpdate {
            r#type: UpdateType::EdgeCreated as i32,
            edge: Some(Edge {
                source: source.clone(),
                destination: destination.clone(),
                message_index: *message_index,
                proximal: *proximal,
                principal: principal.clone(),
                entity: entity.clone(),
            }),
            session_id: sid(scope),
            ..Default::default()
        },
        GraphUpdate::EdgeDeleted {
            source,
            destination,
            scope,
        } => ProtoGraphUpdate {
            r#type: UpdateType::EdgeDeleted as i32,
            source_id: Some(source.clone()),
            destination_id: Some(destination.clone()),
            session_id: sid(scope),
            ..Default::default()
        },
        GraphUpdate::ComputationCreated {
            computation, scope, ..
        } => ProtoGraphUpdate {
            r#type: UpdateType::ComputationCreated as i32,
            computation: Some(computation.clone()),
            session_id: sid(scope),
            ..Default::default()
        },
        GraphUpdate::ComputationDeleted(id) => ProtoGraphUpdate {
            r#type: UpdateType::ComputationDeleted as i32,
            node_id: Some(id.clone()),
            ..Default::default()
        },
        GraphUpdate::ComputationEdgeCreated {
            parent_span_id,
            child_span_id,
            scope,
            ..
        } => ProtoGraphUpdate {
            r#type: UpdateType::ComputationEdgeCreated as i32,
            computation_edge: Some(ComputationEdge {
                parent_span_id: parent_span_id.clone(),
                child_span_id: child_span_id.clone(),
            }),
            session_id: sid(scope),
            ..Default::default()
        },
        GraphUpdate::ComputationMessageLinkCreated {
            span_id,
            message_id,
            is_produces,
            scope,
            ..
        } => {
            let edge_type = if *is_produces {
                ComputationMessageEdgeType::Produces
            } else {
                ComputationMessageEdgeType::Consumes
            };
            ProtoGraphUpdate {
                r#type: UpdateType::ComputationMessageLinkCreated as i32,
                computation_message_edge: Some(ComputationMessageEdge {
                    span_id: span_id.clone(),
                    message_id: message_id.clone(),
                    edge_type: edge_type as i32,
                    message_index: None,
                }),
                session_id: sid(scope),
                ..Default::default()
            }
        }
        GraphUpdate::Heartbeat | GraphUpdate::Sequence(_) | GraphUpdate::SessionSequence { .. } => {
            ProtoGraphUpdate {
                r#type: UpdateType::Heartbeat as i32,
                ..Default::default()
            }
        }
        GraphUpdate::DropSession(_) => ProtoGraphUpdate {
            // Session drops aren't surfaced over the public stream
            // protocol yet; emit a heartbeat to keep clients alive.
            r#type: UpdateType::Heartbeat as i32,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::observability::{Event, Role};

    fn make_store() -> Arc<GraphStore> {
        Arc::new(GraphStore::new(None).unwrap())
    }

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

    fn scope_for(tenant: &str) -> sasy_common::SessionScope {
        sasy_common::SessionScope::new(tenant, String::new())
    }

    /// The request the auth interceptor would hand a handler: a reader in the
    /// `default` tenant. `get_state` refuses a request carrying no
    /// `AuthResult` at all, so a test that omits it is testing the refusal
    /// rather than the read.
    fn reader_request() -> Request<StateRequest> {
        use sasy_auth::AuthResult;
        let mut req = Request::new(StateRequest {});
        req.extensions_mut().insert(AuthResult::success(
            "reader",
            vec![sasy_common::roles::OBSERVABILITY_READER.into()],
            "test",
        ));
        req
    }

    #[tokio::test]
    async fn get_state_empty() {
        let store = make_store();
        let svc = UpdatesService::new(store);
        let state = svc.get_state(reader_request()).await.unwrap().into_inner();
        assert_eq!(state.events.len(), 0);
        assert_eq!(state.edges.len(), 0);
        assert_eq!(state.sequence, 0);
    }

    #[tokio::test]
    async fn get_state_with_data() {
        let store = make_store();
        store
            .merge_events(
                &scope_for("default"),
                None,
                vec![ev("a", "one"), ev("b", "two")],
            )
            .unwrap();

        let svc = UpdatesService::new(store);
        let state = svc.get_state(reader_request()).await.unwrap().into_inner();
        assert_eq!(state.events.len(), 2);
        assert_eq!(state.sequence, 2);
    }

    /// `get_state` returns only the caller's tenant's events.
    /// Mirrors the broadcast filter applied in `stream_updates`.
    #[tokio::test]
    async fn get_state_filters_by_tenant() {
        use sasy_auth::AuthResult;

        let store = make_store();
        store
            .merge_events(&scope_for("acme"), None, vec![ev("a1", "from acme")])
            .unwrap();
        store
            .merge_events(&scope_for("orgb"), None, vec![ev("b1", "from orgb")])
            .unwrap();

        let svc = UpdatesService::new(store);

        // Alice (acme) sees only acme events.
        let mut alice_req = Request::new(StateRequest {});
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let state = svc.get_state(alice_req).await.unwrap().into_inner();
        let ids: Vec<_> = state.events.iter().filter_map(|e| e.id.clone()).collect();
        assert_eq!(ids, vec!["a1".to_string()]);

        // Bob (orgb) sees only orgb events.
        let mut bob_req = Request::new(StateRequest {});
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-reader".into()], "test")
                .with_tenant("orgb"),
        );
        let state = svc.get_state(bob_req).await.unwrap().into_inner();
        let ids: Vec<_> = state.events.iter().filter_map(|e| e.id.clone()).collect();
        assert_eq!(ids, vec!["b1".to_string()]);
    }

    /// The `update_matches_tenant` broadcast filter forwards only
    /// updates whose scope matches the caller's tenant; status
    /// markers (Heartbeat/Sequence) pass through unconditionally.
    #[test]
    fn update_filter_blocks_cross_tenant_payload_passes_status() {
        use sasy_common::SessionScope;
        use sasy_graph::types::EdgeKindSer;

        let acme = SessionScope::global("acme");

        // Cross-tenant NodeCreated is dropped.
        let orgb_node = GraphUpdate::NodeCreated {
            id: "x".into(),
            event: sasy_common::observability::Event {
                text: Some("orgb".into()),
                agent: None,
                role: None,
                id: Some("x".into()),
                tools: vec![],
                derived_from: None,
                principal: Some("bob".into()),
                entity: None,
                metadata: None,
            },
            scope: SessionScope::new("orgb", "s1"),
        };
        assert!(!update_matches_tenant(&orgb_node, &acme));

        // Same-tenant NodeCreated passes.
        let acme_node = GraphUpdate::NodeCreated {
            id: "y".into(),
            event: sasy_common::observability::Event {
                text: Some("acme".into()),
                agent: None,
                role: None,
                id: Some("y".into()),
                tools: vec![],
                derived_from: None,
                principal: Some("alice".into()),
                entity: None,
                metadata: None,
            },
            scope: SessionScope::new("acme", "s1"),
        };
        assert!(update_matches_tenant(&acme_node, &acme));

        // Status markers pass.
        assert!(update_matches_tenant(&GraphUpdate::Heartbeat, &acme));
        assert!(update_matches_tenant(&GraphUpdate::Sequence(1), &acme));

        // Cross-tenant EdgeCreated is dropped.
        let orgb_edge = GraphUpdate::EdgeCreated {
            source: "a".into(),
            destination: "b".into(),
            kind: EdgeKindSer::DependsOn,
            message_index: None,
            proximal: None,
            scope: SessionScope::new("orgb", "s1"),
            principal: Some("bob".into()),
            entity: None,
        };
        assert!(!update_matches_tenant(&orgb_edge, &acme));
    }

    // Note: stream_updates requires tonic::Streaming
    // which can't be constructed in unit tests without
    // a real gRPC transport. StreamUpdates is tested
    // via integration tests with a running server.

    #[tokio::test]
    async fn graph_update_to_proto_variants() {
        // Verify each variant converts correctly
        let gu = GraphUpdate::Heartbeat;
        let p = graph_update_to_proto(&gu);
        assert_eq!(p.r#type, UpdateType::Heartbeat as i32);

        let gu = GraphUpdate::NodeDeleted("x".into());
        let p = graph_update_to_proto(&gu);
        assert_eq!(p.r#type, UpdateType::NodeDeleted as i32);
        assert_eq!(p.node_id.as_deref(), Some("x"));

        let gu = GraphUpdate::EdgeCreated {
            source: "a".into(),
            destination: "b".into(),
            kind: sasy_graph::types::EdgeKindSer::DependsOn,
            message_index: Some(3),
            proximal: Some(true),
            scope: sasy_common::SessionScope::global("default"),
            principal: None,
            entity: None,
        };
        let p = graph_update_to_proto(&gu);
        assert_eq!(p.r#type, UpdateType::EdgeCreated as i32);
        let edge = p.edge.unwrap();
        assert_eq!(edge.source, "a");
        assert_eq!(edge.destination, "b");
        assert_eq!(edge.message_index, Some(3));
        assert_eq!(edge.proximal, Some(true));
    }

    /// Computation-edge / message-link updates carry the emitting
    /// scope. The stream filter must use that scope, or cross-tenant
    /// linkage metadata leaks through the filter.
    #[test]
    fn stream_filter_blocks_cross_tenant_computation_edges() {
        use sasy_common::SessionScope;
        let acme_filter = SessionScope::global("acme");
        let orgb_scope = SessionScope::new("orgb", "s1");
        let acme_scope = SessionScope::new("acme", "s1");

        let cross = GraphUpdate::ComputationEdgeCreated {
            parent_span_id: "p".into(),
            child_span_id: "c".into(),
            scope: orgb_scope.clone(),
        };
        assert!(
            !update_matches_tenant(&cross, &acme_filter),
            "cross-tenant computation edge must not pass the filter",
        );

        let same = GraphUpdate::ComputationEdgeCreated {
            parent_span_id: "p".into(),
            child_span_id: "c".into(),
            scope: acme_scope.clone(),
        };
        assert!(update_matches_tenant(&same, &acme_filter));

        let cross_link = GraphUpdate::ComputationMessageLinkCreated {
            span_id: "s".into(),
            message_id: "m".into(),
            is_produces: true,
            scope: orgb_scope,
        };
        assert!(!update_matches_tenant(&cross_link, &acme_filter));

        let same_link = GraphUpdate::ComputationMessageLinkCreated {
            span_id: "s".into(),
            message_id: "m".into(),
            is_produces: true,
            scope: acme_scope,
        };
        assert!(update_matches_tenant(&same_link, &acme_filter));
    }
}
