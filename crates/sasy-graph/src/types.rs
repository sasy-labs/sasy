//! Domain types for the graph store.

use sasy_common::{EdgeKind, SessionScope};
use serde::{Deserialize, Serialize};

// ── Node types (internal, serialisable) ─────────────

/// A message node in the dependency graph.
///
/// `PartialEq` is CONTENT equality over every field the store
/// persists — there is no server-stamped timestamp or receive
/// time on this node, so re-recording an identical event compares
/// equal. `upsert_event_into_shard` relies on that to stay silent
/// on a re-ingest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageNode {
    pub id: String,
    pub content: Option<String>,
    pub role: Option<String>,
    pub agent: Option<String>,
    pub tools_json: Option<String>,
    pub derived_from_json: Option<String>,
    /// `(tenant, session)` partition the node belongs to. Tenant is
    /// server-derived from auth at the gRPC boundary; session is
    /// the user-supplied `session_id` (empty string = the
    /// per-tenant global partition).
    pub scope: SessionScope,
    /// Server-stamped principal (auth-derived). Distinct from
    /// `agent` (the user-supplied actor); this is the immutable
    /// identity attribute used for write attribution.
    #[serde(default)]
    pub principal: Option<String>,
    /// User-supplied actor in the caller's domain (e.g. an end-user
    /// identity when a gateway relays events). Free-form, *not*
    /// overwritten by the server — distinct from `principal`
    /// (immutable auth-bound writer) and from `agent` (conversation
    /// role label).
    #[serde(default)]
    pub entity: Option<String>,
    /// Adapter-defined canonical record of the source message. Opaque
    /// here: the store keeps it and hands it back, and nothing reads it.
    /// It is part of the event the content hash is taken over, so a
    /// round trip that dropped it would invalidate every compact
    /// reference to the node.
    ///
    /// Left out of the serialized form when unset. An immutable version id
    /// is a digest of this node's JSON, so a node carrying no metadata has
    /// to serialize exactly as it did before the field existed, or the same
    /// snapshot would resolve to a different id than the one its writer
    /// already holds. Every serialization of a `MessageNode` is JSON, which
    /// names its fields, so an omitted key reads back as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
}

/// A computation (OTel span) node.
///
/// `PartialEq` is content equality, as on [`MessageNode`]. The
/// timestamps here are client-supplied span times, not server
/// stamps, so they repeat exactly on a re-ingest of the same span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputationNode {
    pub span_id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub duration_ns: u64,
    pub status_code: i32,
    pub status_message: Option<String>,
    pub attributes_json: String,
    pub events_json: String,
    pub service_name: Option<String>,
    pub service_version: Option<String>,
    pub input_message_ids: Vec<String>,
    pub output_message_id: Option<String>,
    /// `(tenant, session)` partition this computation belongs to.
    /// Same semantics as [`MessageNode::scope`].
    pub scope: SessionScope,
    /// Server-stamped principal (auth-derived). See
    /// [`MessageNode::principal`].
    #[serde(default)]
    pub principal: Option<String>,
    /// User-supplied actor in the caller's domain. See
    /// [`MessageNode::entity`].
    #[serde(default)]
    pub entity: Option<String>,
}

/// Unified node data stored in the petgraph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeData {
    Message(MessageNode),
    Computation(ComputationNode),
}

impl NodeData {
    pub fn id(&self) -> &str {
        match self {
            Self::Message(m) => &m.id,
            Self::Computation(c) => &c.span_id,
        }
    }
}

// ── Edge types ──────────────────────────────────────

/// Data stored on each edge in the petgraph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeData {
    pub kind: EdgeKindSer,
    pub message_index: Option<u32>,
    pub proximal: Option<bool>,
    /// `(tenant, session)` partition this edge belongs to. Edges
    /// only connect nodes within the same scope — cross-session
    /// (and cross-tenant) edges are not honoured by the evaluator.
    pub scope: SessionScope,
    /// Server-stamped principal (auth-derived). See
    /// [`MessageNode::principal`].
    #[serde(default)]
    pub principal: Option<String>,
    /// User-supplied actor in the caller's domain. See
    /// [`MessageNode::entity`].
    #[serde(default)]
    pub entity: Option<String>,
}

/// Serialisable mirror of [`EdgeKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKindSer {
    DependsOn,
    ChildOf,
    Produces,
    Consumes,
}

impl From<EdgeKind> for EdgeKindSer {
    fn from(k: EdgeKind) -> Self {
        match k {
            EdgeKind::DependsOn => Self::DependsOn,
            EdgeKind::ChildOf => Self::ChildOf,
            EdgeKind::Produces => Self::Produces,
            EdgeKind::Consumes => Self::Consumes,
        }
    }
}

impl From<EdgeKindSer> for EdgeKind {
    fn from(k: EdgeKindSer) -> Self {
        match k {
            EdgeKindSer::DependsOn => Self::DependsOn,
            EdgeKindSer::ChildOf => Self::ChildOf,
            EdgeKindSer::Produces => Self::Produces,
            EdgeKindSer::Consumes => Self::Consumes,
        }
    }
}

// ── GraphUpdate (broadcast channel payload) ─────────

/// A change notification emitted on every mutation.
/// Uses proto types so consumers (`sasy-server`) can
/// forward them directly to gRPC streams.
#[derive(Debug, Clone)]
pub enum GraphUpdate {
    NodeCreated {
        id: String,
        event: sasy_common::observability::Event,
        scope: SessionScope,
    },
    NodeDeleted(String),
    EdgeCreated {
        source: String,
        destination: String,
        kind: EdgeKindSer,
        message_index: Option<u32>,
        proximal: Option<bool>,
        /// `(tenant, session)` of this edge. Subscribers use it to
        /// filter their per-session indices via
        /// [`SessionScope::matches`].
        scope: SessionScope,
        /// Server-stamped principal (auth-derived). See
        /// [`super::types::MessageNode::principal`].
        principal: Option<String>,
        /// User-supplied actor (free-form). See
        /// [`super::types::MessageNode::entity`].
        entity: Option<String>,
    },
    EdgeDeleted {
        source: String,
        destination: String,
        /// `(tenant, session)` of the edge that was removed. Without it a
        /// deletion could not be routed: every per-session subscriber
        /// filters on scope, so an update that carries none is dropped by
        /// all of them and the removal never reaches an evaluator.
        scope: SessionScope,
    },
    ComputationCreated {
        computation: sasy_common::observability::Computation,
        scope: SessionScope,
    },
    ComputationDeleted(String),
    ComputationEdgeCreated {
        parent_span_id: String,
        child_span_id: String,
        scope: SessionScope,
    },
    ComputationMessageLinkCreated {
        span_id: String,
        message_id: String,
        is_produces: bool,
        scope: SessionScope,
    },
    /// Drop all nodes/edges associated with a `(tenant, session)`
    /// scope. Subscribers remove the scope's tuples from their
    /// indices and from any cached evaluator state.
    DropSession(SessionScope),
    Heartbeat,
    /// Global sequence marker emitted after a batch of mutations.
    /// Cross-cutting consumers (neo4j_sync, ObservabilityUpdates
    /// streaming) use this to track total graph progress.
    Sequence(i64),
    /// Per-scope sequence marker emitted after a batch of mutations
    /// on a single shard. Per-session evaluators fence on this so
    /// they don't drain unrelated sessions' broadcasts when waiting
    /// for their own writes to land.
    SessionSequence {
        scope: SessionScope,
        seq: i64,
    },
}

// ── EdgeKey (RocksDB persistence key) ───────────────

/// Persistent key for an edge in RocksDB.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeKey {
    pub source: String,
    pub destination: String,
    pub kind: EdgeKindSer,
}

impl EdgeKey {
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("EdgeKey serialization cannot fail")
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(b)
    }
}
