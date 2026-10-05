//! Protobuf conversion helpers for the C ABI boundary.
//!
//! Converts between `sasy_policy::engine::*` domain types
//! and the `policy_plugin` proto messages used for
//! serialization across the C ABI.

use sasy_common::policy_plugin::graph_update_entry::UpdateType;
use sasy_common::policy_plugin::{
    CheckAuthRequest, EdgeUpdate, GraphUpdateBatch, GraphUpdateEntry, NodeCreated, SyncStatusProto,
    ToolInfo,
};

use sasy_common::policy_engine::{Action, AuthorizationResponse};

/// Build a `CheckAuthRequest` proto from Engine trait
/// arguments. `tenant` and `principal` are the auth-derived
/// values from the host's request context — plugins use them
/// for tenant scoping and `HasPrincipal()` / `HasRole(...)`
/// rule evaluation. Empty / `None` is treated as anonymous /
/// default-tenant by plugins that read these fields; plugins
/// that don't read them are unaffected.
pub fn build_check_auth_request(
    current_node_ids: &[String],
    actions: &[Action],
    entity: Option<&str>,
    roles: &[String],
    session_id: Option<&str>,
    tenant: Option<&str>,
    principal: Option<&str>,
) -> CheckAuthRequest {
    CheckAuthRequest {
        current_node_ids: current_node_ids.to_vec(),
        actions: actions.to_vec(),
        entity: entity.map(|s| s.to_string()),
        roles: roles.to_vec(),
        session_id: session_id.map(|s| s.to_string()),
        tenant: tenant.map(|s| s.to_string()),
        principal: principal.map(|s| s.to_string()),
    }
}

/// Encode a `CheckAuthRequest` to protobuf bytes.
pub fn encode_check_auth_request(req: &CheckAuthRequest) -> Vec<u8> {
    use prost::Message;
    req.encode_to_vec()
}

/// Decode an `AuthorizationResponse` from protobuf bytes.
pub fn decode_authorization_response(
    buf: &[u8],
) -> Result<AuthorizationResponse, prost::DecodeError> {
    use prost::Message;
    AuthorizationResponse::decode(buf)
}

/// Encode an `AuthorizationResponse` to protobuf bytes.
pub fn encode_authorization_response(resp: &AuthorizationResponse) -> Vec<u8> {
    use prost::Message;
    resp.encode_to_vec()
}

/// Decode a `CheckAuthRequest` from protobuf bytes.
pub fn decode_check_auth_request(buf: &[u8]) -> Result<CheckAuthRequest, prost::DecodeError> {
    use prost::Message;
    CheckAuthRequest::decode(buf)
}

/// A graph update in the format the engine consumes.
///
/// Mirrors `sasy_policy::engine::GraphUpdate` but is
/// defined here so the SDK doesn't depend on `sasy-policy`.
#[derive(Debug, Clone)]
pub enum GraphUpdate {
    NodeCreated {
        id: String,
        content: Option<String>,
        role: Option<String>,
        agent: Option<String>,
        tools: Vec<(String, String)>,
        entity: Option<String>,
        derived_from: Option<(String, String)>,
        /// Session/conversation partition. ``None`` or empty means
        /// the legacy "global" partition.
        session_id: Option<String>,
    },
    NodeDeleted(String),
    EdgeCreated {
        source: String,
        destination: String,
        session_id: Option<String>,
        /// Server-stamped principal (auth-derived) that asserted
        /// this edge. ``None`` when the recording client carried no
        /// auth-derived identity.
        principal: Option<String>,
        /// User-supplied actor the recording client named for this
        /// edge. ``None`` when not sent.
        entity: Option<String>,
    },
    EdgeDeleted {
        source: String,
        destination: String,
    },
}

/// Build a `GraphUpdateBatch` proto from a vec of
/// GraphUpdate.
pub fn build_graph_update_batch(updates: &[GraphUpdate]) -> GraphUpdateBatch {
    let entries = updates
        .iter()
        .map(|u| match u {
            GraphUpdate::NodeCreated {
                id,
                content,
                role,
                agent,
                tools,
                entity,
                derived_from,
                session_id,
            } => GraphUpdateEntry {
                update_type: Some(UpdateType::NodeCreated(NodeCreated {
                    id: id.clone(),
                    content: content.clone(),
                    role: role.clone(),
                    agent: agent.clone(),
                    tools: tools
                        .iter()
                        .map(|(n, a)| ToolInfo {
                            name: n.clone(),
                            arguments: a.clone(),
                        })
                        .collect(),
                    entity: entity.clone(),
                    derived_from: derived_from.as_ref().map(|(n, a)| ToolInfo {
                        name: n.clone(),
                        arguments: a.clone(),
                    }),
                    session_id: session_id.clone(),
                })),
            },
            GraphUpdate::NodeDeleted(id) => GraphUpdateEntry {
                update_type: Some(UpdateType::NodeDeleted(id.clone())),
            },
            GraphUpdate::EdgeCreated {
                source,
                destination,
                session_id,
                principal,
                entity,
            } => GraphUpdateEntry {
                update_type: Some(UpdateType::EdgeCreated(EdgeUpdate {
                    source: source.clone(),
                    destination: destination.clone(),
                    session_id: session_id.clone(),
                    principal: principal.clone(),
                    entity: entity.clone(),
                })),
            },
            GraphUpdate::EdgeDeleted {
                source,
                destination,
            } => GraphUpdateEntry {
                update_type: Some(UpdateType::EdgeDeleted(EdgeUpdate {
                    source: source.clone(),
                    destination: destination.clone(),
                    session_id: None,
                    principal: None,
                    entity: None,
                })),
            },
        })
        .collect();

    GraphUpdateBatch { updates: entries }
}

/// Encode a `GraphUpdateBatch` to protobuf bytes.
pub fn encode_graph_update_batch(batch: &GraphUpdateBatch) -> Vec<u8> {
    use prost::Message;
    batch.encode_to_vec()
}

/// Decode a `GraphUpdateBatch` from protobuf bytes.
pub fn decode_graph_update_batch(buf: &[u8]) -> Result<GraphUpdateBatch, prost::DecodeError> {
    use prost::Message;
    GraphUpdateBatch::decode(buf)
}

/// Decode a `SyncStatusProto` from protobuf bytes.
pub fn decode_sync_status(buf: &[u8]) -> Result<SyncStatusProto, prost::DecodeError> {
    use prost::Message;
    SyncStatusProto::decode(buf)
}

/// Encode a `SyncStatusProto` to protobuf bytes.
pub fn encode_sync_status(status: &SyncStatusProto) -> Vec<u8> {
    use prost::Message;
    status.encode_to_vec()
}
