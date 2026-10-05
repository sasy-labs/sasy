//! In-memory graph backed by petgraph with RocksDB
//! persistence and broadcast change notifications.
//!
//! Sharded by `(tenant, session)` partition: each
//! [`SessionScope`] owns its own [`StableDiGraph`] under its own
//! [`RwLock`], so two scopes can mutate in parallel; only the
//! cross-shard reverse indices and the rocksdb WAL serialize
//! between them.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rocksdb::WriteBatch;
use sasy_common::observability::{self as proto, Computation, Edge, Event, TraceGraph};
use sasy_common::{EdgeKind, SessionScope};
use tokio::sync::broadcast;
use tracing::{debug, warn};
use uuid::Uuid;

mod snapshots;

use crate::convert;
use crate::error::GraphError;
use crate::persistence::RocksStore;
use crate::types::{EdgeData, EdgeKey, EdgeKindSer, GraphUpdate, NodeData};

/// Capacity of the store-wide broadcast ring.
///
/// A subscriber that falls further behind than this gets `Lagged` and has to
/// reconcile, and the reload a `Lagged` forces can itself take long enough for
/// a burst to overflow the ring again. Idempotent ingest keeps most re-ingest
/// traffic off the ring; this capacity absorbs what is left of a genuine
/// first-time burst.
///
/// The cost is memory, in two parts.
///
/// The first is paid up front and always resident: `tokio`'s broadcast channel
/// allocates every slot when the channel is created, and a [`GraphUpdate`]
/// slot is 416 bytes here, so the slot array is ~13.6 MB per `GraphStore`.
/// A process runs one store, so that is a one-off — but
/// it is ~12 MB of baseline RSS on an idle binary, in every build.
///
/// The second is paid only during a burst, and is not in that figure. A slot
/// holds its value until every live receiver has read it, so while a
/// subscriber lags, each un-consumed update keeps what it owns alive — a
/// `NodeCreated` owns its `Event`, message text included. Payload retention
/// therefore scales with this cap as well: raising it from 4096 to 32768
/// raised the ceiling on burst-time retention eightfold, and the total there
/// is bounded by the size of the records in flight, not by the 416-byte slot.
/// Nothing here caps that; a burst of large messages against a lagging
/// subscriber is the case to watch.
const CHANGE_CHANNEL_CAP: usize = 32768;

/// Changes retained per scope for [`GraphStore::updates_since`].
///
/// Compact records, not payloads: a full log is ~7 MB per scope (~72 bytes
/// inline per entry plus each entry's heap-allocated ids), and a subscriber
/// that missed less than this catches up incrementally instead of reloading
/// the whole shard.
///
/// The bound is PER SCOPE and a shard is only ever freed by
/// [`GraphStore::drop_session`], so the aggregate is bounded by the number of
/// sessions the process has served, not by any constant: a long-lived server
/// with twenty busy sessions holds ~140 MB of log, and nothing reclaims a
/// session's log while the process lives. Bounding it store-wide, or freeing
/// it when a session goes idle, is not implemented.
const REPLAY_LOG_CAP: usize = 65536;

/// One retained change, identifying WHAT changed rather than carrying the
/// change itself. [`GraphStore::updates_since`] materialises the current
/// content for each key when it replays, so the log never holds a stale
/// payload and its size does not track message size.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ChangeKey {
    /// A message node, by its id.
    Node(String),
    /// A dependency edge, by its persistence key.
    Edge(EdgeKey),
    /// A dependency edge that was removed.
    EdgeDeleted(EdgeKey),
}

/// A [`ChangeKey`] reduced to what identifies the THING it names, so that a
/// key's repeated records collapse to one replayed update. `Edge` and
/// `EdgeDeleted` name the same edge and so share an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DedupKey<'a> {
    Node(&'a str),
    Edge(&'a EdgeKey),
}

/// What [`GraphStore::claim_session_owner`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnerClaim {
    /// The row was absent and this call created it. Undo requires the token
    /// returned by `claim_session_owner_with_token`.
    Claimed,
    /// The caller already held it, or there was no principal to record.
    AlreadyHeld,
    /// Held by a different principal.
    HeldBy(String),
}

/// Permission to undo a newly created owner row, valid only until another
/// ownership operation touches the scope. The opaque allocation prevents ABA
/// reuse even when the same principal deletes and reclaims the session.
#[derive(Debug, Clone)]
pub struct OwnerClaimToken {
    scope: SessionScope,
    principal: String,
    generation: Arc<()>,
}

/// Thread-safe, persistent graph store.
///
/// Sharded by `(tenant, session)` scope. Each scope owns its own
/// [`StableDiGraph`] under its own [`RwLock`], so two scopes can
/// mutate in parallel; only the cross-shard reverse indices and
/// the rocksdb WAL serialize between them.
pub struct GraphStore {
    /// Per-scope shards. Created lazily on first write or
    /// bootstrap read. Empty session_id within a tenant is the
    /// per-tenant "global" partition.
    shards: RwLock<HashMap<SessionScope, Arc<RwLock<SessionShard>>>>,
    /// Reverse lookup for arbitrary-id reads (backward_slice,
    /// get_span, etc.): `(tenant, node_id) → scope` of the shard
    /// that owns it. Maintained whenever a shard upserts a node.
    /// Keying on `(tenant, id)` rather than `id` alone is what
    /// keeps two tenants that happen to mint the same node id
    /// (clients pick `Event.id`; collisions are easy) from
    /// clobbering each other's reverse entries — without this,
    /// the legitimate owner would silently lose `scope_for_node`
    /// resolution and `resolve_edge_scope` could route an edge
    /// write into the wrong tenant's shard.
    node_to_session: RwLock<HashMap<(String, String), SessionScope>>,
    /// `(tenant, trace_id) → scopes` that contain a span in that
    /// trace. Usually one scope per trace. Keyed on `(tenant,
    /// trace_id)` for the same reason as `node_to_session`.
    trace_to_sessions: RwLock<HashMap<(String, String), HashSet<SessionScope>>>,
    /// Single monotonic sequence across all shards.
    global_sequence: AtomicI64,
    rocks: RocksStore,
    tx: broadcast::Sender<GraphUpdate>,
    /// Serializes owner rows and pending rollback tokens across services.
    /// Matching claims invalidate the original creator's rollback, because
    /// the new caller may already be writing under that ownership.
    owner_claim_lock: parking_lot::Mutex<HashMap<SessionScope, Arc<()>>>,
}

/// One per-scope shard. Holds the petgraph plus the per-shard
/// indices. All edges connect endpoints within the same shard
/// (computations + Produces/Consumes/ChildOf live alongside the
/// messages they reference; cross-scope edges are not honoured).
struct SessionShard {
    graph: StableDiGraph<NodeData, EdgeData>,
    msg_idx: HashMap<String, NodeIndex>,
    /// Lazy immutable content fingerprints, reclaimed with this shard's nodes.
    content_hashes: HashMap<String, [u8; 32]>,
    comp_idx: HashMap<String, NodeIndex>,
    /// Mirrors ``msg_idx.keys()`` for O(1) ``events_for_session``.
    msg_set: HashSet<String>,
    /// Every edge in this shard, by its persistence key — the same
    /// `(source, destination, kind)` triple RocksDB's `CF_EDGES` is keyed
    /// by, so in-memory and durable state agree on what "the same edge"
    /// means.
    ///
    /// Without it, `add_edge` was called unconditionally and petgraph
    /// happily stores parallel edges, so every re-ingest of a dependency
    /// added another copy: the in-memory edge count inflated without bound
    /// across restarts while RocksDB (keyed) held exactly one row. The index
    /// is what lets a re-ingested edge be recognised and skipped.
    edge_idx: HashMap<EdgeKey, EdgeIndex>,
    /// Edge count for this scope (only counts edges within a
    /// non-global scope; legacy global-partition edges aren't
    /// tracked, matching the prior behaviour).
    edge_count: u64,
    /// Per-scope monotonic sequence. Bumped on every mutation in
    /// this shard; emitted as ``GraphUpdate::SessionSequence`` so
    /// per-scope evaluators can fence on their own progress
    /// without waiting on unrelated scopes' broadcasts to drain.
    sequence: i64,
    /// The last [`REPLAY_LOG_CAP`] changes made to this shard, oldest first,
    /// each tagged with the `sequence` value the change was assigned.
    ///
    /// Appended under the shard write lock, in the same critical section and
    /// the same order as the broadcast sends, so the log and the ring agree on
    /// what happened and in which order. A subscriber that fell behind the ring
    /// can replay from here instead of reloading the whole shard.
    replay: VecDeque<(i64, ChangeKey)>,
    /// Set when a write mutated this shard in memory and then failed to
    /// commit to RocksDB.
    ///
    /// The shard is then ahead of the durable copy with no way to reconcile:
    /// nothing rolls the mutation back, and there is no per-scope loader to
    /// re-read from disk. Left alone, evaluators keep bootstrapping and
    /// resyncing from the diverged state, so they answer with facts the write
    /// API reported as FAILED — and a restart erases them, so the same action
    /// is decided differently before and after.
    ///
    /// So the reads that SEED an evaluator refuse: the per-session state, the
    /// two slice reads, and both tenant-wide reads — the last because they are
    /// how a global-scope evaluator bootstraps, which would otherwise let one
    /// diverged session reach a decision through the tenant view. A tenant
    /// read refuses wholesale rather than skipping the bad shard, since a
    /// silently smaller graph is the under-match being guarded against.
    ///
    /// It does not stop an evaluator that is ALREADY live from deciding on
    /// the lineage it holds, nor stop later writes to the shard. It
    /// quarantines the state where it would be read afresh; there is no
    /// reconciliation short of a restart.
    diverged: bool,
}

/// True for an event with nothing to store, which `upsert_event_into_shard`
/// skips — returning an empty id and adding no node.
///
/// Public because callers that need to know whether a batch will actually
/// write anything must ask THIS function rather than approximate it. The
/// ingest handler's "a no-op batch must not claim session ownership" guard
/// cannot use `events.is_empty()`, which is a different question: a batch of
/// one content-free event is non-empty and still stores nothing, so it would
/// claim a victim's session id while writing nothing.
///
/// Shared with the combined call's edge pre-check, which has to agree with it
/// exactly. A pre-check that assumed every id in the batch would land would
/// pass an edge naming a skipped event, apply the other events, then reject in
/// the inner pre-pass and quarantine the session — a tenant-wide refusal
/// reachable by any writer sending one empty event and an edge to it.
/// The refusal a diverged shard gives, for reads and writes alike.
///
/// Writes are refused as well as reads because the in-memory index still
/// names the never-committed nodes: an edge recorded afterwards resolves
/// against them, commits durably, is broadcast naming a node no subscriber
/// ever saw, and is dropped by the loader at the next restart — success
/// reported for a taint link that does not survive.
fn diverged_error(scope: &SessionScope) -> GraphError {
    GraphError::InvalidEdge(format!(
        "session {scope} diverged from durable storage after a failed commit; \
         refusing to serve or extend its graph until the process restarts"
    ))
}

pub fn event_carries_nothing(ev: &Event) -> bool {
    ev.text.is_none()
        && ev.tools.is_empty()
        && ev.derived_from.is_none()
        && ev.role.is_none()
        && ev.agent.is_none()
        && ev.entity.is_none()
        && ev.metadata.is_none()
}

impl SessionShard {
    fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            msg_idx: HashMap::new(),
            content_hashes: HashMap::new(),
            comp_idx: HashMap::new(),
            msg_set: HashSet::new(),
            edge_idx: HashMap::new(),
            edge_count: 0,
            sequence: 0,
            replay: VecDeque::new(),
            diverged: false,
        }
    }

    fn resolve(&self, id: &str) -> Option<NodeIndex> {
        self.msg_idx
            .get(id)
            .or_else(|| self.comp_idx.get(id))
            .copied()
    }

    /// Collect every message as an `Event` and every `DependsOn` edge as an
    /// `Edge` from this shard. The shared inner walk behind the
    /// `get_full_state*` readers (which differ only in how they filter
    /// shards and whether they tag each item with its owning scope).
    fn collect_messages_and_edges(&self) -> (Vec<Event>, Vec<Edge>) {
        let mut events = Vec::new();
        for ni in self.msg_idx.values() {
            if let Some(NodeData::Message(m)) = self.graph.node_weight(*ni) {
                events.push(convert::message_to_event(m));
            }
        }
        let mut edges = Vec::new();
        for ei in self.graph.edge_indices() {
            let Some(data) = self.graph.edge_weight(ei) else {
                continue;
            };
            if data.kind != EdgeKindSer::DependsOn {
                continue;
            }
            if let Some((a, b)) = self.graph.edge_endpoints(ei) {
                // DependsOn edges are stored in petgraph as
                // (destination → source); swap back to the proto
                // convention (source = predecessor, destination =
                // successor) so this readback round-trips the recorded
                // edge — matching `backward_slice` and the live
                // `EdgeCreated` broadcast. Without the swap a cold
                // bootstrap reverses every dependency edge, flipping
                // multi-hop rule results after an evaluator respawn.
                let src = node_id(&self.graph, b);
                let dst = node_id(&self.graph, a);
                edges.push(convert::edge_data_to_proto(&src, &dst, data));
            }
        }
        (events, edges)
    }
}

impl GraphStore {
    /// Open a graph store. If `db_path` is `None`, uses
    /// a temporary directory (for tests).
    pub fn new(db_path: Option<&str>) -> Result<Self, GraphError> {
        match db_path {
            Some(p) => Self::open_at(p),
            None => Ok(Self::temp()),
        }
    }

    /// Subscribe to change notifications.
    pub fn subscribe(&self) -> broadcast::Receiver<GraphUpdate> {
        self.tx.subscribe()
    }

    /// Current global monotonic sequence (across all shards).
    pub fn get_sequence(&self) -> i64 {
        self.global_sequence.load(Ordering::SeqCst)
    }

    /// Per-scope monotonic sequence. Used by per-scope evaluators
    /// as the dispatch fence so they don't drain unrelated scopes'
    /// broadcasts when waiting for their own writes to land. Returns
    /// 0 for unknown / dropped scopes.
    pub fn session_sequence(&self, scope: &SessionScope) -> i64 {
        match self.shard_ro(scope) {
            Some(shard) => shard.read().sequence,
            None => 0,
        }
    }

    /// The MESSAGE-graph changes to `scope` with sequence greater than
    /// `from_seq` — or `None` when the retained log no longer reaches back that
    /// far and the caller has to reload instead.
    ///
    /// Computation changes (spans and their CHILD_OF / PRODUCES / CONSUMES
    /// edges) are not logged and so are never emitted here. That costs the one
    /// consumer nothing: the policy engine drops every computation update it
    /// receives on the live stream too (`sync::broadcast_to_updates`), so a
    /// catch-up and the live stream carry the same facts.
    ///
    /// The answer is in log order, EXCEPT that the dedupe below emits each key
    /// once, at the position of its LAST record — so an `EdgeCreated` can
    /// precede the `NodeCreated` of one of its endpoints (the node was
    /// recorded first, then re-recorded after the edge). Consumers must not
    /// depend on the order. It is harmless for the evaluator because its EDB is
    /// a SET of facts, not a stream of events: applying an edge fact before its
    /// endpoint's node fact leaves both present once the batch is applied, and
    /// every rule joins on the same set afterwards. Sorting the window
    /// nodes-first was considered and not done: it buys the one consumer
    /// nothing for a sort under the shard read lock, and a consumer that DID
    /// need event order could not be served from this log anyway, since the
    /// dedupe has already collapsed each key's history to one record.
    ///
    /// The updates are materialised from the shard's CURRENT state (a node's
    /// content as it stands now, an edge's data as it stands now), not from the
    /// payloads that were broadcast at the time. That is the right answer here
    /// because the consumer is converging on the store's state rather than
    /// auditing its history, and because applying an update is idempotent:
    /// replaying today's content for a key that changed three times lands in
    /// the same place as replaying all three versions in order, at a third of
    /// the cost.
    ///
    /// Because the content comes from current state, a key recorded more than
    /// once inside the window materialises the same value every time, so only
    /// its LAST record is emitted. That is what keeps a catch-up from costing
    /// more than the reload it replaces: the log tracks a scope's change
    /// HISTORY, which is unbounded relative to its size (an SDK that re-records
    /// a growing assistant message logs one record per turn), and undeduped a
    /// 200-node session with 40,000 retained changes shipped 40,000 updates in
    /// one IPC frame where the reload it avoids sends 200. Deduped, the answer
    /// is never longer than the number of distinct keys the window touched.
    ///
    /// `None` means "reconcile some other way". It is returned when the oldest
    /// retained record is newer than `from_seq + 1` (a change in the gap is
    /// unrecoverable from here), when `from_seq` is AHEAD of the scope's own
    /// sequence (a caller claiming progress the shard has never made cannot be
    /// served incrementally: answering "nothing changed" left it pinned above
    /// every sequence the shard is about to hand out, permanently short of the
    /// state it asked about, since cursors only move forward), when the scope
    /// has no shard at all, when the cursor predates a `drop_session` of the
    /// scope (the drop leaves the sequence counter behind, so a cursor from the
    /// dropped incarnation falls in the gap rather than being served the
    /// re-created shard's tail), and when the shard is quarantined as `diverged` —
    /// where nothing short of a restart is sound, so a caller must not be
    /// handed a partial catch-up.
    pub fn updates_since(&self, scope: &SessionScope, from_seq: i64) -> Option<Vec<GraphUpdate>> {
        let shard = self.shard_ro(scope)?;
        let s = shard.read();
        if s.diverged {
            return None;
        }
        if from_seq > s.sequence {
            return None;
        }
        match s.replay.front() {
            Some((oldest, _)) if from_seq + 1 < *oldest => return None,
            // Nothing retained. Only a caller already level with the shard can
            // be served, and only with an empty answer.
            None if from_seq < s.sequence => return None,
            _ => {}
        }

        // The log is sorted by sequence, so the window starts at a binary
        // search instead of a scan from the front: a one-update gap on a full
        // log costs a handful of comparisons under the read lock rather than
        // 65536.
        let start = s.replay.partition_point(|(seq, _)| *seq <= from_seq);
        // Walk the window backwards, keeping the first sighting of each key —
        // which is its last record. `Edge` and `EdgeDeleted` share one key, so
        // whichever came last wins and a create-then-delete replays as the
        // delete alone. Reversed afterwards to restore log order.
        let mut seen: HashSet<DedupKey<'_>> = HashSet::new();
        let mut window: Vec<&ChangeKey> = Vec::new();
        for (_, key) in s.replay.range(start..).rev() {
            let dedup = match key {
                ChangeKey::Node(id) => DedupKey::Node(id.as_str()),
                ChangeKey::Edge(ek) | ChangeKey::EdgeDeleted(ek) => DedupKey::Edge(ek),
            };
            if seen.insert(dedup) {
                window.push(key);
            }
        }
        window.reverse();

        let mut out = Vec::with_capacity(window.len());
        for key in window {
            match key {
                ChangeKey::Node(id) => {
                    // Absent means the node is gone from the shard, which today
                    // only happens when its whole session was dropped — and
                    // `DropSession` travels on its own.
                    if let Some(&ni) = s.msg_idx.get(id) {
                        if let Some(NodeData::Message(m)) = s.graph.node_weight(ni) {
                            out.push(GraphUpdate::NodeCreated {
                                id: id.clone(),
                                event: convert::message_to_event(m),
                                scope: m.scope.clone(),
                            });
                        }
                    }
                }
                ChangeKey::Edge(ek) => {
                    // Absent means it is gone from the shard. A later deletion
                    // is not this case — that record shares this key and wins
                    // the dedupe above, so the caller is told about the delete
                    // rather than a re-create it would have to undo.
                    if let Some(&ei) = s.edge_idx.get(ek) {
                        if let Some(d) = s.graph.edge_weight(ei) {
                            out.push(GraphUpdate::EdgeCreated {
                                source: ek.source.clone(),
                                destination: ek.destination.clone(),
                                kind: d.kind,
                                message_index: d.message_index,
                                proximal: d.proximal,
                                scope: d.scope.clone(),
                                principal: d.principal.clone(),
                                entity: d.entity.clone(),
                            });
                        }
                    }
                }
                ChangeKey::EdgeDeleted(ek) => out.push(GraphUpdate::EdgeDeleted {
                    source: ek.source.clone(),
                    destination: ek.destination.clone(),
                    scope: scope.clone(),
                }),
            }
        }
        Some(out)
    }

    /// Test-only view of a scope's retained change sequences, oldest first.
    /// Used to pin that the log's order and the broadcast's order agree.
    #[cfg(test)]
    pub(crate) fn replay_sequences(&self, scope: &SessionScope) -> Vec<i64> {
        match self.shard_ro(scope) {
            Some(shard) => shard.read().replay.iter().map(|(s, _)| *s).collect(),
            None => Vec::new(),
        }
    }

    // ── Policy persistence (passthroughs to RocksStore) ─────
    //
    // The graph store already owns the RocksDB handle and is
    // threaded through every layer that has both a tenant and a
    // session id, so it's the natural place to hang the policy /
    // session-ownership keyspaces too. The accessors here delegate
    // to [`RocksStore`] — the policy service uses them via this
    // GraphStore handle rather than reaching into the rocks layer.

    pub fn put_policy_binding(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        self.rocks.put_policy_binding(scope, content_hash)
    }

    pub fn delete_policy_binding(&self, scope: &SessionScope) -> Result<(), GraphError> {
        self.rocks.delete_policy_binding(scope)
    }

    pub fn get_policy_binding(&self, scope: &SessionScope) -> Result<Option<String>, GraphError> {
        self.rocks.get_policy_binding(scope)
    }

    pub fn all_policy_bindings(&self) -> Result<Vec<(SessionScope, String)>, GraphError> {
        self.rocks.all_policy_bindings()
    }

    pub fn put_tenant_default_policy(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        self.rocks.put_tenant_default_policy(tenant, content_hash)
    }

    pub fn get_tenant_default_policy(&self, tenant: &str) -> Result<Option<String>, GraphError> {
        self.rocks.get_tenant_default_policy(tenant)
    }

    pub fn delete_tenant_default_policy(&self, tenant: &str) -> Result<(), GraphError> {
        self.rocks.delete_tenant_default_policy(tenant)
    }

    pub fn all_tenant_defaults(&self) -> Result<Vec<(String, String)>, GraphError> {
        self.rocks.all_tenant_defaults()
    }

    pub fn put_policy_source(
        &self,
        content_hash: &str,
        value: &crate::persistence::PersistedPolicy,
    ) -> Result<(), GraphError> {
        self.rocks.put_policy_source(content_hash, value)
    }

    /// Store a source under an explicitly named admission class. See
    /// [`crate::persistence::RocksStore::put_policy_source_in_class`] — the
    /// key decides the class a record is read back in.
    pub fn put_policy_source_in_class(
        &self,
        content_hash: &str,
        admission: crate::persistence::FunctorAdmission,
        value: &crate::persistence::PersistedPolicy,
    ) -> Result<(), GraphError> {
        self.rocks
            .put_policy_source_in_class(content_hash, admission, value)
    }

    pub fn get_policy_source(
        &self,
        content_hash: &str,
    ) -> Result<Option<crate::persistence::PersistedPolicy>, GraphError> {
        self.rocks.get_policy_source(content_hash)
    }

    pub fn get_policy_source_in_class(
        &self,
        content_hash: &str,
        admission: crate::persistence::FunctorAdmission,
    ) -> Result<Option<crate::persistence::PersistedPolicy>, GraphError> {
        self.rocks
            .get_policy_source_in_class(content_hash, admission)
    }

    /// Every persisted record for `content_hash`, least-privileged admission
    /// class first, each paired with the class ITS KEY encodes — which is the
    /// authoritative one. See
    /// [`crate::persistence::RocksStore::policy_sources_by_least_privilege`].
    pub fn policy_sources_by_least_privilege(
        &self,
        content_hash: &str,
    ) -> Result<
        Vec<(
            crate::persistence::FunctorAdmission,
            crate::persistence::PersistedPolicy,
        )>,
        GraphError,
    > {
        self.rocks.policy_sources_by_least_privilege(content_hash)
    }

    pub fn put_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
        facts: &[crate::persistence::PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        self.rocks.put_policy_metadata(tenant, content_hash, facts)
    }

    #[cfg(test)]
    pub(crate) fn put_legacy_policy_metadata(
        &self,
        content_hash: &str,
        facts: &[crate::persistence::PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        self.rocks.put_legacy_policy_metadata(content_hash, facts)
    }

    pub fn get_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<Vec<crate::persistence::PolicyMetadataFact>, GraphError> {
        self.rocks.get_policy_metadata(tenant, content_hash)
    }

    pub fn append_session_metadata(
        &self,
        scope: &SessionScope,
        facts: &[crate::persistence::PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        self.rocks.append_session_metadata(scope, facts)
    }

    pub fn get_session_metadata(
        &self,
        scope: &SessionScope,
    ) -> Result<Vec<crate::persistence::PolicyMetadataFact>, GraphError> {
        self.rocks.get_session_metadata(scope)
    }

    pub fn delete_session_metadata(&self, scope: &SessionScope) -> Result<(), GraphError> {
        self.rocks.delete_session_metadata(scope)
    }

    pub fn put_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
        facts: &[crate::persistence::PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        self.rocks.put_binding_metadata(scope, content_hash, facts)
    }

    pub fn get_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<Option<Vec<crate::persistence::PolicyMetadataFact>>, GraphError> {
        self.rocks.get_binding_metadata(scope, content_hash)
    }

    pub fn clear_tenant_binding_metadata(
        &self,
        tenant: &str,
    ) -> Result<crate::persistence::ClearedBindingMetadata, GraphError> {
        self.rocks.clear_tenant_binding_metadata(tenant)
    }

    pub fn restore_binding_metadata(
        &self,
        rows: &crate::persistence::ClearedBindingMetadata,
    ) -> Result<(), GraphError> {
        self.rocks.restore_binding_metadata(rows)
    }

    pub fn delete_binding_metadata_for_policy(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        self.rocks
            .delete_binding_metadata_for_policy(scope, content_hash)
    }

    pub fn delete_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        self.rocks.delete_policy_metadata(tenant, content_hash)
    }

    pub fn delete_session_state(&self, scope: &SessionScope) -> Result<(), GraphError> {
        self.rocks.delete_session_state(scope)
    }

    pub fn delete_binding_metadata(&self, scope: &SessionScope) -> Result<(), GraphError> {
        self.rocks.delete_binding_metadata(scope)
    }

    pub fn put_session_owner(
        &self,
        scope: &SessionScope,
        principal: &str,
    ) -> Result<(), GraphError> {
        let mut claims = self.owner_claim_lock.lock();
        claims.remove(scope);
        self.rocks.put_session_owner(scope, principal)
    }

    pub fn get_session_owner(&self, scope: &SessionScope) -> Result<Option<String>, GraphError> {
        self.rocks.get_session_owner(scope)
    }

    pub fn delete_session_owner(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let mut claims = self.owner_claim_lock.lock();
        claims.remove(scope);
        self.rocks.delete_session_owner(scope)
    }

    /// Test-only accessor for poking the RocksDB layer directly
    /// (e.g. asserting `drop_session` cleared its edges). Not a
    /// production API.
    #[cfg(test)]
    pub(crate) fn rocks_for_test(&self) -> &crate::persistence::RocksStore {
        &self.rocks
    }

    /// Verify or claim ownership of `scope` for `principal`.
    ///
    /// * No owner recorded, `principal = Some(p)` → write `p` and
    ///   return `Ok(None)`.
    /// * No owner recorded, `principal = None` → no claim, return
    ///   `Ok(None)`. Anonymous callers can touch unowned sessions.
    /// * Owner matches `principal` → return `Ok(None)`.
    /// * Owner differs (or owner exists and `principal = None`)
    ///   → return `Ok(Some(existing_owner))`. The handler is
    ///   responsible for converting this into a deny (admin-bypass
    ///   excepted).
    ///
    /// The read + (conditional) write run under
    /// `owner_claim_lock` so two concurrent principals racing on
    /// the same fresh `(tenant, session)` produce a deterministic
    /// winner: whichever holds the lock first claims, the second
    /// observes that owner and gets denied (or matches if it's
    /// the same principal racing itself).
    pub fn check_or_claim_session_owner(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
    ) -> Result<Option<String>, GraphError> {
        Ok(match self.claim_session_owner(scope, principal)? {
            OwnerClaim::Claimed | OwnerClaim::AlreadyHeld => None,
            OwnerClaim::HeldBy(existing) => Some(existing),
        })
    }

    /// Claim ownership without rollback permission. Callers that may need to
    /// undo their claim must use [`Self::claim_session_owner_with_token`].
    pub fn claim_session_owner(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
    ) -> Result<OwnerClaim, GraphError> {
        let (claim, token) = self.claim_session_owner_with_token(scope, principal)?;
        if let Some(token) = token {
            self.commit_session_owner_claim(&token);
        }
        Ok(claim)
    }

    /// Claim ownership and capture rollback permission for a newly created row.
    /// Every subsequent claim, including `AlreadyHeld`, invalidates that
    /// permission before returning. Pending rollbacks never survive restart.
    /// Finish a returned token with `commit_session_owner_claim` on success or
    /// `release_session_owner_claim` on failure; dropping it alone is insufficient.
    pub fn claim_session_owner_with_token(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
    ) -> Result<(OwnerClaim, Option<OwnerClaimToken>), GraphError> {
        let mut claims = self.owner_claim_lock.lock();
        claims.remove(scope);
        let existing = self.get_session_owner(scope)?;
        match (existing, principal) {
            (Some(e), Some(p)) if e == p => Ok((OwnerClaim::AlreadyHeld, None)),
            (Some(e), _) => Ok((OwnerClaim::HeldBy(e), None)),
            (None, Some(p)) => {
                self.rocks.put_session_owner(scope, p)?;
                let generation = Arc::new(());
                claims.insert(scope.clone(), Arc::clone(&generation));
                Ok((
                    OwnerClaim::Claimed,
                    Some(OwnerClaimToken {
                        scope: scope.clone(),
                        principal: p.to_owned(),
                        generation,
                    }),
                ))
            }
            (None, None) => Ok((OwnerClaim::AlreadyHeld, None)),
        }
    }

    /// Keep the durable owner and discard this operation's rollback bookkeeping.
    /// A stale token cannot disarm a later claim, including delete/recreate ABA.
    pub fn commit_session_owner_claim(&self, token: &OwnerClaimToken) {
        let mut claims = self.owner_claim_lock.lock();
        if claims
            .get(&token.scope)
            .is_some_and(|generation| Arc::ptr_eq(generation, &token.generation))
        {
            claims.remove(&token.scope);
        }
    }

    /// Undo only an untouched claim made by this token's caller. An intervening
    /// writer may rely on the owner even before its graph or policy commit,
    /// so checking whether the session is empty cannot replace this guard.
    pub fn release_session_owner_claim(&self, token: &OwnerClaimToken) -> Result<bool, GraphError> {
        let mut claims = self.owner_claim_lock.lock();
        if !claims
            .get(&token.scope)
            .is_some_and(|generation| Arc::ptr_eq(generation, &token.generation))
        {
            return Ok(false);
        }
        if self.get_session_owner(&token.scope)?.as_deref() != Some(token.principal.as_str()) {
            claims.remove(&token.scope);
            return Ok(false);
        }
        self.rocks.delete_session_owner(&token.scope)?;
        claims.remove(&token.scope);
        Ok(true)
    }

    /// Aggregate node and edge counts across all shards.
    pub fn get_counts(&self) -> (usize, usize) {
        let r = self.shards.read();
        let mut nodes = 0usize;
        let mut edges = 0usize;
        for shard in r.values() {
            let s = shard.read();
            nodes += s.msg_idx.len();
            edges += s.graph.edge_count();
        }
        (nodes, edges)
    }

    // ── Session lifecycle ──────────────────────────

    /// Returns the set of currently-known scopes that have at least
    /// one message. Per-tenant "global" partitions (empty session)
    /// are included. Order is unspecified.
    pub fn list_sessions(&self) -> Vec<SessionScope> {
        self.shards
            .read()
            .iter()
            .filter_map(|(scope, v)| {
                if scope.is_global() {
                    return None;
                }
                let s = v.read();
                if s.msg_set.is_empty() {
                    None
                } else {
                    Some(scope.clone())
                }
            })
            .collect()
    }

    /// (node_count, edge_count) for a scope's tuples. Both zero for an
    /// unknown scope. `edge_count` counts message-dependency edges only — the
    /// same quantity an evaluator's bootstrap loads — not the computation
    /// edges (CHILD_OF / PRODUCES / CONSUMES) the same shard may hold.
    pub fn session_counts(&self, scope: &SessionScope) -> (u64, u64) {
        let shard = match self.shard_ro(scope) {
            Some(s) => s,
            None => return (0, 0),
        };
        let s = shard.read();
        (s.msg_set.len() as u64, s.edge_count)
    }

    /// Message ids in a scope. Empty if unknown.
    pub fn events_for_session(&self, scope: &SessionScope) -> Vec<String> {
        let shard = match self.shard_ro(scope) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let s = shard.read();
        s.msg_set.iter().cloned().collect()
    }

    /// Drop every node and edge belonging to ``scope``. Empties the shard in
    /// place (the entry and its sequence counter stay, as a tombstone), clears
    /// reverse indices, does a best-effort rocksdb cleanup, and broadcasts a
    /// [`GraphUpdate::DropSession`].
    pub fn drop_session(&self, scope: &SessionScope) -> Result<(), GraphError> {
        if scope.is_global() {
            return Ok(());
        }
        self.owner_claim_lock.lock().remove(scope);
        let shard = self.shards.read().get(scope).cloned();
        let shard = match shard {
            Some(s) => s,
            None => return Ok(()),
        };

        // The shard is emptied in place rather than removed from the map, and
        // its `sequence` survives as a tombstone. Removing it let a scope be
        // re-created starting at sequence 0, and a cursor left over from the
        // dropped incarnation is BEHIND that — so it passes every guard in
        // `updates_since` and gets served the NEW shard's tail: the consumer
        // keeps the deleted nodes, never sees the new shard's first records,
        // and still advances its fence. Keeping the counter means a scope
        // never re-uses a sequence, so such a cursor now trips the
        // retention guard (or the "nothing retained and you are behind" one)
        // and is told to reconcile some other way. The graph, the indices and
        // the replay log are freed, so the tombstone costs one map entry.
        //
        // The drop CONSUMES a sequence, exactly as the `DropSession` send
        // below consumes a global one. Without that bump the tombstone sits AT
        // the sequence a caught-up subscriber already holds — the ordinary
        // state of one — and `from_seq < s.sequence` is false, so it is served
        // `Some([])` now and the re-created shard's tail later: it keeps every
        // node the drop erased and never learns of the deletion. One sequence
        // spent here puts the tombstone strictly above every cursor of the
        // dropped incarnation.
        let (msg_ids, comp_ids): (Vec<String>, Vec<String>) = {
            let mut s = shard.write();
            let ids = (
                s.msg_set.iter().cloned().collect(),
                s.comp_idx.keys().cloned().collect(),
            );
            let sequence = s.sequence;
            *s = SessionShard::new();
            s.sequence = sequence + 1;
            ids
        };

        {
            // Reverse-index cleanup is last-writer-wins-aware: a
            // node id colliding across sessions in the same tenant
            // (clients pick `Event.id`, collisions happen) leaves
            // `node_to_session` pointing at the *latest* writer.
            // An unconditional `remove(&(tenant, id))` would erase
            // the surviving session's index entry whenever the
            // older session is dropped first. Only remove the
            // mapping if it still points at the scope we're
            // tearing down.
            let mut nts = self.node_to_session.write();
            let tenant = scope.tenant().to_string();
            for id in msg_ids.iter().chain(comp_ids.iter()) {
                let key = (tenant.clone(), id.clone());
                if nts.get(&key).map(|s| s == scope).unwrap_or(false) {
                    nts.remove(&key);
                }
            }
        }
        {
            let mut trs = self.trace_to_sessions.write();
            // A `(tenant, trace_id)` may span multiple sessions, so
            // only drop the trace entry when it has no remaining
            // session and only touch entries within this scope's
            // tenant — sibling tenants' traces are untouched.
            let tenant = scope.tenant();
            trs.retain(|(t, _), scopes| {
                if t != tenant {
                    return true;
                }
                scopes.remove(scope);
                !scopes.is_empty()
            });
        }

        for mid in &msg_ids {
            if let Err(e) = self.rocks.delete_message(scope, mid) {
                warn!("drop_session: rocks delete msg {} failed: {}", mid, e);
            }
        }
        for cid in &comp_ids {
            if let Err(e) = self.rocks.delete_computation(scope, cid) {
                warn!("drop_session: rocks delete comp {} failed: {}", cid, e);
            }
        }
        // Edges are scope-prefixed by the same length-prefix
        // encoding, so a single prefix scan + delete clears every
        // edge belonging to this scope. Without this, dropped
        // sessions left orphan edge rows behind that bloated
        // restart-time `all_edges()` walks.
        match self.rocks.delete_edges_for_scope(scope) {
            Ok(n) if n > 0 => {
                debug!(scope = %scope, n, "drop_session: cleared persisted edges");
            }
            Ok(_) => {}
            Err(e) => warn!(
                scope = %scope,
                error = %e,
                "drop_session: rocks delete edges failed"
            ),
        }

        let seq = self.global_sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.rocks.set_sequence(seq as u64);
        let _ = self.tx.send(GraphUpdate::DropSession(scope.clone()));
        let _ = self.tx.send(GraphUpdate::Sequence(seq));
        Ok(())
    }

    // ── Event (Message) operations ──────────────────

    /// Upsert messages (MERGE by ID). Returns the IDs.
    ///
    /// Events are grouped by `(tenant, session)` scope and processed
    /// shard-by-shard so cross-scope traffic doesn't serialize.
    /// `tenant` is the auth-derived tenant for the request — every
    /// event in `events` is scoped to this tenant; the per-event
    /// `session_id` field selects the per-tenant partition.
    /// `principal` is the auth-derived identity stamped onto every
    /// node; any value the client sent on `event.principal` is
    /// ignored.
    ///
    /// A re-ingest that changes nothing is SILENT: no sequence bump, no
    /// broadcast payload, no marker on the ring, and no RocksDB write — not
    /// even the sequence meta-key. The scope's sequence is still reported
    /// ([`GraphStore::session_sequence`] returns its current value), so a
    /// client fencing on read-your-writes (the `timeout_at` seq fence in
    /// `sasy-policy`'s session evaluator) is fenced on a sequence every
    /// subscriber has already been told about by the write that produced it,
    /// and clears at once.
    pub fn merge_events(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        events: Vec<Event>,
    ) -> Result<Vec<String>, GraphError> {
        let shard = self.get_or_create_shard(scope);
        let mut s = shard.write();
        if s.diverged {
            return Err(diverged_error(scope));
        }
        snapshots::validate_legacy_events(&s, scope, principal, &events)?;
        let mut updates: Vec<GraphUpdate> = Vec::new();
        let mut batch = self.rocks.new_batch();
        let mut new_nodes: Vec<String> = Vec::new();
        let mut replay: Vec<(i64, ChangeKey)> = Vec::new();

        let mut result: Vec<String> = Vec::with_capacity(events.len());
        for ev in &events {
            let id = self.upsert_event_into_shard(
                &mut s,
                ev,
                scope,
                principal,
                &mut batch,
                &mut updates,
                &mut new_nodes,
                &mut replay,
                false,
            )?;
            result.push(id);
        }

        let seq = self.global_sequence.load(Ordering::SeqCst);
        self.finalize_shard_write(s, batch, &new_nodes, scope, updates, replay, seq)?;
        Ok(result)
    }

    /// Upsert dependency edges between messages within `scope`.
    /// Cross-scope edges aren't a thing in the new model — the SDK
    /// emits one `Dependencies` envelope per session, so every
    /// edge in the batch routes to the same shard.
    pub fn merge_dependencies(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        edges: Vec<Edge>,
    ) -> Result<(), GraphError> {
        let shard = match self.shard_ro(scope) {
            Some(s) => s,
            None => {
                // Nothing has ever been recorded for this scope, so not one
                // of these edges can resolve. Returning `Ok` here reported
                // success for a batch that was discarded in full — and a
                // dropped `DependsOn` edge is a severed taint link, so a
                // provenance-gated denial silently stops firing. The caller
                // has to know, because nothing retries.
                return Err(GraphError::InvalidEdge(format!(
                    "no recorded events for {scope}, so none of the {} dependency \
                     edge(s) could be attached; record the events first, or use \
                     the combined events+dependencies call",
                    edges.len()
                )));
            }
        };

        let mut s = shard.write();
        if s.diverged {
            return Err(diverged_error(scope));
        }
        let mut updates: Vec<GraphUpdate> = Vec::new();
        let mut batch = self.rocks.new_batch();
        let mut replay: Vec<(i64, ChangeKey)> = Vec::new();

        snapshots::validate_legacy_edges(&s, scope, principal, &edges)?;
        self.merge_dependencies_into_shard(
            &mut s,
            &edges,
            principal,
            &mut batch,
            &mut updates,
            &mut replay,
            scope,
        )?;

        let seq = self.global_sequence.load(Ordering::SeqCst);
        self.finalize_shard_write(s, batch, &[], scope, updates, replay, seq)?;
        Ok(())
    }

    /// Remove one `DependsOn` edge from `scope`. Returns `true` iff an edge
    /// was actually removed.
    ///
    /// Deleting an edge the shard does not hold is a no-op: no RocksDB
    /// delete, no sequence bump and no broadcast. That is the same
    /// idempotency the add path has, and for the same reason — a client that
    /// replays a retraction must not make every subscriber reload.
    pub fn delete_dependency(
        &self,
        scope: &SessionScope,
        source: &str,
        destination: &str,
    ) -> Result<bool, GraphError> {
        let Some(shard) = self.shard_ro(scope) else {
            return Ok(false);
        };
        let mut s = shard.write();
        if s.diverged {
            return Err(diverged_error(scope));
        }
        let key = EdgeKey {
            source: source.to_string(),
            destination: destination.to_string(),
            kind: EdgeKindSer::DependsOn,
        };
        // The index is the only place that knows which petgraph edge this key
        // names; without it there was no way to delete an edge by key at all.
        let Some(&ei) = s.edge_idx.get(&key) else {
            return Ok(false);
        };
        if snapshots::is_immutable(destination) {
            return Err(GraphError::ImmutableViolation(
                "incoming dependencies cannot be deleted".into(),
            ));
        }
        // Durable before visible, as everywhere else on the write path: if the
        // row will not go away, the edge stays in memory and nobody is told
        // it went.
        self.rocks.delete_edge(scope, &key)?;
        s.edge_idx.remove(&key);
        s.graph.remove_edge(ei);
        if !scope.is_global() {
            s.edge_count = s.edge_count.saturating_sub(1);
        }
        s.sequence += 1;
        self.global_sequence.fetch_add(1, Ordering::SeqCst);

        let updates = vec![GraphUpdate::EdgeDeleted {
            source: source.to_string(),
            destination: destination.to_string(),
            scope: scope.clone(),
        }];
        let replay = vec![(s.sequence, ChangeKey::EdgeDeleted(key))];
        let batch = self.rocks.new_batch();
        let seq = self.global_sequence.load(Ordering::SeqCst);
        self.finalize_shard_write(s, batch, &[], scope, updates, replay, seq)?;
        Ok(true)
    }

    /// Atomically upsert events and dependency edges in a single
    /// shard write-lock hold. The whole batch shares `scope` so
    /// one lock acquisition covers both upserts.
    ///
    /// A re-ingest that changes nothing is SILENT: no sequence bump, no
    /// broadcast payload, no marker on the ring, and no RocksDB write — not
    /// even the sequence meta-key. The scope's sequence is still reported
    /// ([`GraphStore::session_sequence`] returns its current value), so a
    /// client fencing on read-your-writes (the `timeout_at` seq fence in
    /// `sasy-policy`'s session evaluator) is fenced on a sequence every
    /// subscriber has already been told about by the write that produced it,
    /// and clears at once.
    pub fn merge_events_with_dependencies(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        events: Vec<Event>,
        edges: Vec<Edge>,
    ) -> Result<Vec<String>, GraphError> {
        let shard = self.get_or_create_shard(scope);
        let mut s = shard.write();
        if s.diverged {
            return Err(diverged_error(scope));
        }
        snapshots::validate_legacy_events(&s, scope, principal, &events)?;
        let mut updates: Vec<GraphUpdate> = Vec::new();
        let mut batch = self.rocks.new_batch();
        let mut new_nodes: Vec<String> = Vec::new();
        let mut replay: Vec<(i64, ChangeKey)> = Vec::new();
        let mut ids: Vec<String> = Vec::with_capacity(events.len());

        snapshots::validate_legacy_edges(&s, scope, principal, &edges)?;
        // Validate the edges BEFORE touching the shard, against what will
        // exist once the events land: everything already recorded, plus the
        // ids this call carries. Letting the edge pre-pass inside
        // `merge_dependencies_into_shard` do it is too late — the events have
        // been added to the graph by then, and rejecting there unwinds
        // without committing or broadcasting, leaving the shard ahead of
        // RocksDB with nobody told. That is exactly the state `diverged`
        // exists to quarantine, and a client sending one bad edge should not
        // produce it.
        {
            let incoming: HashSet<&str> = events
                .iter()
                .filter(|e| !event_carries_nothing(e))
                .filter_map(|e| e.id.as_deref())
                .filter(|id| !id.is_empty())
                .collect();
            let unresolved: Vec<String> = edges
                .iter()
                .filter_map(|e| {
                    // Empty endpoints are rejected here too, and named the
                    // same way. Skipping them while the inner pre-pass
                    // rejects them let such an edge through this check, fail
                    // there, and take the fallback below — quarantining a
                    // shard nothing had touched.
                    if e.source.is_empty() || e.destination.is_empty() {
                        return Some(format!(
                            "{} -> {} (empty endpoint)",
                            e.source, e.destination
                        ));
                    }
                    let known = |id: &str| s.msg_idx.contains_key(id) || incoming.contains(id);
                    (!known(&e.source) || !known(&e.destination))
                        .then(|| format!("{} -> {}", e.source, e.destination))
                })
                .collect();
            if !unresolved.is_empty() {
                return Err(GraphError::InvalidEdge(format!(
                    "dependency edge(s) name nodes that are neither recorded in {scope} \
                     nor present in this batch: {}",
                    unresolved.join(", ")
                )));
            }
        }

        for ev in &events {
            ids.push(self.upsert_event_into_shard(
                &mut s,
                ev,
                scope,
                principal,
                &mut batch,
                &mut updates,
                &mut new_nodes,
                &mut replay,
                false,
            )?);
        }
        if let Err(e) = self.merge_dependencies_into_shard(
            &mut s,
            &edges,
            principal,
            &mut batch,
            &mut updates,
            &mut replay,
            scope,
        ) {
            // Quarantine ONLY if this call actually put something in the
            // shard. The pre-check above should have caught anything the
            // inner pre-pass rejects, so reaching here at all means the two
            // disagree — and marking the shard unconditionally turned that
            // disagreement into a permanent tenant-wide refusal, since the
            // tenant-wide reads refuse for any diverged session and a rejected
            // batch may have mutated nothing. Mutation is what the flag is
            // about; the disagreement is a bug to fix, not a reason to take a
            // clean session out of service.
            let applied_any = ids.iter().any(|id| !id.is_empty());
            if applied_any {
                s.diverged = true;
                tracing::error!(
                    scope = %scope, error = %e,
                    "edges rejected after their events were applied; marking the \
                     session diverged"
                );
            } else {
                tracing::error!(
                    scope = %scope, error = %e,
                    "edge pre-checks disagreed, but nothing was applied — rejecting \
                     without quarantining the session"
                );
            }
            return Err(e);
        }

        let seq = self.global_sequence.load(Ordering::SeqCst);
        self.finalize_shard_write(s, batch, &new_nodes, scope, updates, replay, seq)?;
        Ok(ids)
    }

    /// Upsert computations (OTel spans) into `scope`. Returns
    /// span_ids in input order.
    pub fn merge_computations(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        comps: Vec<Computation>,
    ) -> Result<Vec<String>, GraphError> {
        // Quarantined shards refuse computation writes too. Skipping this
        // check let a span be linked durably to a message that only ever
        // existed in memory: the write succeeds, and the loader drops the
        // dangling edge at the next restart — success reported for a link
        // that does not survive.
        if let Some(shard) = self.shard_ro(scope) {
            if shard.read().diverged {
                return Err(diverged_error(scope));
            }
        }
        let mut ids = Vec::with_capacity(comps.len());
        for c in &comps {
            ids.push(self.upsert_comp(scope, principal, c)?);
        }
        Ok(ids)
    }

    // ── Graph queries ───────────────────────────────

    /// Resolve the shard scope for an arbitrary-`id` read. Prefers an
    /// explicit `session` hint — the caller naming its own shard,
    /// which skips the `(tenant, id)` reverse lookup and its
    /// intra-tenant last-writer-wins ambiguity (two sessions in one
    /// tenant can mint the same id). Falls back to the reverse index
    /// when the hint is absent or empty (the empty/global sentinel is
    /// not a node-owning shard for these reads). The `tenant` always
    /// comes from the caller's auth context, so a hint cannot reach
    /// another tenant's shard.
    fn resolve_read_scope(
        &self,
        tenant: &str,
        id: &str,
        session: Option<&str>,
    ) -> Option<SessionScope> {
        match session {
            Some(s) if !s.is_empty() => Some(SessionScope::new(tenant, s)),
            _ => self.session_for_node(tenant, id),
        }
    }

    /// BFS backward slice returning proto types. The caller's
    /// auth-derived `tenant` scopes the lookup so a colliding `id` in
    /// a different tenant can't be reached; an explicit `session`
    /// names the shard directly (see [`Self::resolve_read_scope`]).
    pub fn backward_slice(
        &self,
        tenant: &str,
        id: &str,
        session: Option<&str>,
        max_depth: Option<u32>,
    ) -> Result<(Vec<Event>, Vec<Edge>), GraphError> {
        let scope = self
            .resolve_read_scope(tenant, id, session)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let shard = self
            .shard_ro(&scope)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let s = shard.read();
        // See `get_session_state`: a diverged shard must not decide anything.
        if s.diverged {
            return Err(GraphError::InvalidEdge(format!(
                "session {scope} diverged from durable storage after a failed \
                 commit; refusing to serve its graph until the process restarts"
            )));
        }
        let start = s
            .resolve(id)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let (nodes, edges) = bfs(&s.graph, start, Direction::Outgoing, max_depth);
        Ok(to_proto_graph(&nodes, &edges))
    }

    /// BFS forward slice returning proto types.
    pub fn forward_slice(
        &self,
        tenant: &str,
        id: &str,
        session: Option<&str>,
        max_depth: Option<u32>,
    ) -> Result<(Vec<Event>, Vec<Edge>), GraphError> {
        let scope = self
            .resolve_read_scope(tenant, id, session)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let shard = self
            .shard_ro(&scope)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let s = shard.read();
        // See `get_session_state`: a diverged shard must not decide anything.
        if s.diverged {
            return Err(GraphError::InvalidEdge(format!(
                "session {scope} diverged from durable storage after a failed \
                 commit; refusing to serve its graph until the process restarts"
            )));
        }
        let start = s
            .resolve(id)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        let (nodes, edges) = bfs(&s.graph, start, Direction::Incoming, max_depth);
        Ok(to_proto_graph(&nodes, &edges))
    }

    /// Scopes within `tenant` that hold at least one span of
    /// `trace_id`. Used by `get_trace` callers to apply
    /// per-principal ownership filtering before assembling the
    /// trace payload.
    pub fn scopes_for_trace(&self, tenant: &str, trace_id: &str) -> Vec<SessionScope> {
        self.trace_to_sessions
            .read()
            .get(&(tenant.to_string(), trace_id.to_string()))
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get a full trace by `(tenant, trace_id)`. Walks every shard
    /// in the tenant that holds a span for this trace (usually one).
    ///
    /// `allowed_scopes`: `None` walks every scope (legacy behaviour
    /// for in-process callers); `Some(set)` restricts the walk to
    /// those scopes — used by the gRPC handler to drop sessions the
    /// caller doesn't own without exposing their existence.
    ///
    /// A trace that genuinely spans multiple sessions (the same
    /// `trace_id` written under different `sasy.session(...)` contexts)
    /// is **intentionally flattened**: the returned `TraceGraph`
    /// merges spans/messages from every contributing (allowed) session
    /// without per-item session attribution. The `Computation`/`Event`
    /// items carry no `session_id` (it lives on the write envelope, not
    /// per item — see `proto/observability.proto`), and unlike the
    /// single-shard slice/span reads there is no one envelope session
    /// to attribute a cross-session trace to. This is safe — the
    /// caller only sees sessions it owns (`allowed_scopes`) — but it
    /// means consumers can't tell which session a given span came from.
    /// If that's ever needed, group the response by session rather than
    /// re-introducing a per-item `session_id`.
    pub fn get_trace(
        &self,
        tenant: &str,
        trace_id: &str,
        start_time_ns: Option<u64>,
        end_time_ns: Option<u64>,
        allowed_scopes: Option<&HashSet<SessionScope>>,
    ) -> Result<TraceGraph, GraphError> {
        let scopes: Vec<SessionScope> = self
            .trace_to_sessions
            .read()
            .get(&(tenant.to_string(), trace_id.to_string()))
            .map(|s| {
                s.iter()
                    .filter(|sc| allowed_scopes.is_none_or(|a| a.contains(*sc)))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        let mut comps: Vec<Computation> = Vec::new();
        let mut messages: Vec<Event> = Vec::new();
        let mut computation_edges: Vec<proto::ComputationEdge> = Vec::new();
        let mut message_edges: Vec<proto::ComputationMessageEdge> = Vec::new();

        for scope in &scopes {
            let shard = match self.shard_ro(scope) {
                Some(s) => s,
                None => continue,
            };
            let s = shard.read();

            let mut comp_nis: HashSet<NodeIndex> = HashSet::new();
            let mut msg_ids: HashSet<String> = HashSet::new();

            for &ni in s.comp_idx.values() {
                if let Some(NodeData::Computation(c)) = s.graph.node_weight(ni) {
                    if c.trace_id != trace_id {
                        continue;
                    }
                    if let Some(st) = start_time_ns {
                        if c.start_time_ns < st {
                            continue;
                        }
                    }
                    if let Some(et) = end_time_ns {
                        if c.end_time_ns > et {
                            continue;
                        }
                    }
                    comps.push(convert::comp_to_proto(c));
                    comp_nis.insert(ni);
                    for mid in &c.input_message_ids {
                        msg_ids.insert(mid.clone());
                    }
                    if let Some(ref mid) = c.output_message_id {
                        msg_ids.insert(mid.clone());
                    }
                }
            }

            for mid in &msg_ids {
                if let Some(&ni) = s.msg_idx.get(mid) {
                    if let Some(NodeData::Message(m)) = s.graph.node_weight(ni) {
                        messages.push(convert::message_to_event(m));
                    }
                }
            }

            for &ni in &comp_nis {
                for edge in s.graph.edges_directed(ni, Direction::Outgoing) {
                    let data = edge.weight();
                    let target_ni = edge.target();
                    match data.kind {
                        EdgeKindSer::ChildOf => {
                            let child_id = node_id(&s.graph, ni);
                            let parent_id = node_id(&s.graph, target_ni);
                            computation_edges.push(proto::ComputationEdge {
                                parent_span_id: parent_id,
                                child_span_id: child_id,
                            });
                        }
                        EdgeKindSer::Produces => {
                            let sid2 = node_id(&s.graph, ni);
                            let mid = node_id(&s.graph, target_ni);
                            message_edges.push(proto::ComputationMessageEdge {
                                span_id: sid2,
                                message_id: mid,
                                edge_type: proto::ComputationMessageEdgeType::Produces as i32,
                                message_index: None,
                            });
                        }
                        EdgeKindSer::Consumes => {
                            let sid2 = node_id(&s.graph, ni);
                            let mid = node_id(&s.graph, target_ni);
                            message_edges.push(proto::ComputationMessageEdge {
                                span_id: sid2,
                                message_id: mid,
                                edge_type: proto::ComputationMessageEdgeType::Consumes as i32,
                                message_index: data.message_index.map(|i| i as i32),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(TraceGraph {
            computations: comps,
            messages,
            computation_edges,
            message_edges,
        })
    }

    /// Get a single span by `(tenant, span_id)`. An explicit
    /// `session` names the shard directly (see
    /// [`Self::resolve_read_scope`]); otherwise the span_id is
    /// resolved via the reverse index.
    pub fn get_span(
        &self,
        tenant: &str,
        span_id: &str,
        session: Option<&str>,
    ) -> Result<Option<Computation>, GraphError> {
        let scope = match self.resolve_read_scope(tenant, span_id, session) {
            Some(s) => s,
            None => return Ok(None),
        };
        let shard = match self.shard_ro(&scope) {
            Some(s) => s,
            None => return Ok(None),
        };
        let s = shard.read();
        let ni = match s.comp_idx.get(span_id) {
            Some(ni) => *ni,
            None => return Ok(None),
        };
        match s.graph.node_weight(ni) {
            Some(NodeData::Computation(c)) => Ok(Some(convert::comp_to_proto(c))),
            _ => Ok(None),
        }
    }

    /// Mark a scope diverged, for tests. Production sets this only from
    /// `finalize_shard_write`, on a commit that failed after the shard was
    /// already mutated.
    #[cfg(test)]
    pub fn mark_diverged_for_test(&self, scope: &SessionScope) {
        if let Some(shard) = self.shard_ro(scope) {
            shard.write().diverged = true;
        }
    }

    /// Per-scope graph state: every message and DependsOn edge that
    /// belongs to ``scope``, plus the *scope-local* sequence.
    /// Returns empty vecs and zero seq for unknown / dropped scopes.
    pub fn get_session_state(
        &self,
        scope: &SessionScope,
    ) -> Result<(Vec<Event>, Vec<Edge>, i64), GraphError> {
        let shard = match self.shard_ro(scope) {
            Some(s) => s,
            None => return Ok((Vec::new(), Vec::new(), 0)),
        };
        let s = shard.read();
        // A diverged shard is ahead of RocksDB with no way to reconcile, so
        // an evaluator bootstrapped from it decides with facts the write API
        // reported as failed — until a restart erases them and the same
        // action is decided the other way. Refusing costs this session its
        // availability; answering costs the decision its meaning.
        if s.diverged {
            return Err(GraphError::InvalidEdge(format!(
                "session {scope} diverged from durable storage after a failed \
                 commit; refusing to serve its graph until the process restarts"
            )));
        }
        let seq = s.sequence;

        let events: Vec<Event> = s
            .msg_idx
            .values()
            .filter_map(|ni| match s.graph.node_weight(*ni)? {
                NodeData::Message(m) => Some(convert::message_to_event(m)),
                _ => None,
            })
            .collect();

        let edges: Vec<Edge> = s
            .graph
            .edge_indices()
            .filter_map(|ei| {
                let data = s.graph.edge_weight(ei)?;
                if data.kind != EdgeKindSer::DependsOn {
                    return None;
                }
                // DependsOn edges are stored in petgraph as
                // (destination → source); swap back to the proto
                // convention (source = predecessor, destination =
                // successor) so the bootstrap EDB round-trips the
                // recorded edge and matches the live `EdgeCreated`
                // broadcast. Without the swap a cold bootstrap reverses
                // every dependency edge, flipping multi-hop rule results
                // after an evaluator respawn.
                let (a, b) = s.graph.edge_endpoints(ei)?;
                let src = node_id(&s.graph, b);
                let dst = node_id(&s.graph, a);
                Some(convert::edge_data_to_proto(&src, &dst, data))
            })
            .collect();

        Ok((events, edges, seq))
    }

    /// Full graph state across all shards. Walks every shard.
    pub fn get_full_state(&self) -> Result<(Vec<Event>, Vec<Edge>, i64), GraphError> {
        let seq = self.global_sequence.load(Ordering::SeqCst);
        let shards = self.shards.read();
        let mut events: Vec<Event> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();

        for (scope, shard) in shards.iter() {
            let s = shard.read();
            // Display-only today (the Neo4j export), but the same rule: do not
            // hand out facts a write reported as failed and a restart erases.
            if s.diverged {
                return Err(diverged_error(scope));
            }
            let (e, d) = s.collect_messages_and_edges();
            events.extend(e);
            edges.extend(d);
        }

        Ok((events, edges, seq))
    }

    /// The same content as [`Self::get_full_state`], kept grouped by the
    /// scope that owns each part instead of flattened into one list.
    ///
    /// The Neo4j mirror needs this: it stamps a `tenant` on every node it
    /// writes, and once the shards are merged there is nothing left to read
    /// the tenant off. Same divergence rule as the flat version — a shard
    /// whose write failed is not exported, it is an error.
    #[allow(clippy::type_complexity)]
    pub fn get_full_state_by_scope(
        &self,
    ) -> Result<(Vec<(SessionScope, Vec<Event>, Vec<Edge>)>, i64), GraphError> {
        let seq = self.global_sequence.load(Ordering::SeqCst);
        let shards = self.shards.read();
        let mut out: Vec<(SessionScope, Vec<Event>, Vec<Edge>)> = Vec::new();

        for (scope, shard) in shards.iter() {
            let s = shard.read();
            if s.diverged {
                return Err(diverged_error(scope));
            }
            let (events, edges) = s.collect_messages_and_edges();
            out.push((scope.clone(), events, edges));
        }

        Ok((out, seq))
    }

    /// Per-tenant graph state: every message and DependsOn edge in
    /// shards belonging to `tenant`, plus the global sequence.
    /// Used by the external observability gRPC surface to enforce
    /// per-request tenant isolation on read paths.
    pub fn get_full_state_for_tenant(
        &self,
        tenant: &str,
    ) -> Result<(Vec<Event>, Vec<Edge>, i64), GraphError> {
        let seq = self.global_sequence.load(Ordering::SeqCst);
        let shards = self.shards.read();
        let mut events: Vec<Event> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();

        for (scope, shard) in shards.iter() {
            if scope.tenant() != tenant {
                continue;
            }
            let s = shard.read();
            // One diverged session poisons the whole tenant-wide answer: this
            // is a bootstrap read, so serving it would seed an evaluator with
            // facts a write reported as failed and a restart will erase. The
            // per-session reads refuse for the same reason.
            if s.diverged {
                return Err(GraphError::InvalidEdge(format!(
                    "session {scope} diverged from durable storage after a failed \
                     commit; refusing to serve tenant-wide state for {tenant} until \
                     the process restarts"
                )));
            }
            let (e, d) = s.collect_messages_and_edges();
            events.extend(e);
            edges.extend(d);
        }

        Ok((events, edges, seq))
    }

    /// Per-tenant *scope-preserving* graph state: every message and
    /// DependsOn edge in shards belonging to `tenant`, each paired
    /// with its owning [`SessionScope`]. Unlike
    /// [`Self::get_full_state_for_tenant`] (which flattens), this
    /// keeps per-session attribution so a global-scope per-session
    /// evaluator can bootstrap the same tenant-wide view its live
    /// broadcast stream feeds it — every session under the tenant,
    /// each event under its own session — rather than only the
    /// global shard.
    ///
    /// The returned sequence is the tenant's GLOBAL SHARD counter
    /// (`(tenant, "")`), read under that shard's own lock alongside its
    /// content — so it is exact, and it matches what the dispatch path
    /// passes as `min_sequence` (`session_sequence(scope)`, which for a
    /// global scope IS that shard). A global evaluator uses the pair as
    /// a resync TRIGGER: reload only when this counter has moved past
    /// the last snapshot.
    ///
    /// Deliberately NOT the store-wide `get_sequence()`. That counter is
    /// bumped by every session of every tenant, so gating on it would
    /// force a full-tenant reload whenever ANY unrelated writer moved it
    /// — no cost bound on a path the refmon takes per proxied request.
    /// The global shard is also exactly where session-less writers land,
    /// and its counter moves only under a single shard's write lock, so
    /// it is free of the cross-shard reordering that makes the
    /// store-wide counter unusable here. Per-session shards a global
    /// evaluator also holds are best-effort eventually-consistent,
    /// exactly as on the live path.
    #[allow(clippy::type_complexity)] // inherent: scoped full-state return is a triple of paired vecs + seq
    pub fn get_full_state_for_tenant_scoped(
        &self,
        tenant: &str,
    ) -> Result<(Vec<(SessionScope, Event)>, Vec<(SessionScope, Edge)>, i64), GraphError> {
        let shards = self.shards.read();
        let mut events: Vec<(SessionScope, Event)> = Vec::new();
        let mut edges: Vec<(SessionScope, Edge)> = Vec::new();
        // Fence value for a GLOBAL evaluator: the tenant's GLOBAL SHARD's own
        // counter, read under that shard's lock alongside its content (so it is
        // exact, not merely an understate). Deliberately NOT the store-wide
        // `global_sequence`: that one is bumped by every session of every tenant, so
        // comparing against it would force a full-tenant reload whenever ANY
        // unrelated writer moved it — no cost bound at all on the global path, which
        // the refmon takes for every proxied request. The global shard is exactly
        // where session-less writers land (`SessionScope::new(tenant, "")`), and its
        // counter is incremented + broadcast under a SINGLE shard's write lock, so it
        // is not subject to the cross-shard reordering that makes the store-wide
        // counter unusable here. Other sessions' shards remain best-effort
        // eventually-consistent for a global evaluator, as documented above.
        let mut global_seq: i64 = 0;

        for (scope, shard) in shards.iter() {
            if scope.tenant() != tenant {
                continue;
            }
            let s = shard.read();
            // Same refusal as the per-session reads. This is the bootstrap and
            // resync read for a GLOBAL-scope evaluator, so it is the path by
            // which a diverged session's unpersisted facts would otherwise
            // still reach an authorization decision.
            if s.diverged {
                return Err(GraphError::InvalidEdge(format!(
                    "session {scope} diverged from durable storage after a failed \
                     commit; refusing to serve tenant-wide state for {tenant} until \
                     the process restarts"
                )));
            }
            if scope.is_global() {
                global_seq = s.sequence;
            }
            let (e, d) = s.collect_messages_and_edges();
            events.extend(e.into_iter().map(|ev| (scope.clone(), ev)));
            edges.extend(d.into_iter().map(|ed| (scope.clone(), ed)));
        }

        Ok((events, edges, global_seq))
    }

    // ── Private helpers ─────────────────────────────

    /// Look up an existing shard for read; returns ``None`` if no
    /// scope has been registered with that key.
    fn shard_ro(&self, scope: &SessionScope) -> Option<Arc<RwLock<SessionShard>>> {
        self.shards.read().get(scope).cloned()
    }

    /// Get-or-create the shard for ``scope``.
    fn get_or_create_shard(&self, scope: &SessionScope) -> Arc<RwLock<SessionShard>> {
        {
            let r = self.shards.read();
            if let Some(s) = r.get(scope) {
                return s.clone();
            }
        }
        let mut w = self.shards.write();
        w.entry(scope.clone())
            .or_insert_with(|| Arc::new(RwLock::new(SessionShard::new())))
            .clone()
    }

    fn session_for_node(&self, tenant: &str, id: &str) -> Option<SessionScope> {
        self.node_to_session
            .read()
            .get(&(tenant.to_string(), id.to_string()))
            .cloned()
    }

    /// Public read accessor on the reverse index used by external
    /// gRPC handlers to enforce per-request tenant guards on
    /// arbitrary-id reads. Returns `Some(scope)` iff a node
    /// with `id` is owned by `tenant`; cross-tenant ids never leak.
    pub fn scope_for_node(&self, tenant: &str, id: &str) -> Option<SessionScope> {
        self.session_for_node(tenant, id)
    }

    /// True iff the given `(tenant, trace_id)` holds at least one
    /// span. Used by `get_trace` callers to short-circuit a
    /// not-found before the deeper walk.
    pub fn trace_has_tenant(&self, tenant: &str, trace_id: &str) -> bool {
        self.trace_to_sessions
            .read()
            .get(&(tenant.to_string(), trace_id.to_string()))
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }

    /// Upsert a single event into a held shard. Stages rocks puts
    /// into ``batch`` and broadcast updates into ``updates``.
    /// Returns the assigned/echoed message id (empty string for
    /// skipped empty events). The caller passes the
    /// `(tenant, session)` scope explicitly; every event in a
    /// batch shares it, so per-event session inference is no
    /// longer a thing.
    #[allow(clippy::too_many_arguments)] // inherent: shard + event + scope + principal + staging buffers
    fn upsert_event_into_shard(
        &self,
        s: &mut SessionShard,
        ev: &Event,
        scope: &SessionScope,
        principal: Option<&str>,
        batch: &mut WriteBatch,
        updates: &mut Vec<GraphUpdate>,
        new_nodes: &mut Vec<String>,
        replay: &mut Vec<(i64, ChangeKey)>,
        replace: bool,
    ) -> Result<String, GraphError> {
        // "Nothing worth storing" has to mean every field this node keeps, not
        // just text and tools. The merge below exists precisely so a re-record
        // can set `entity` or `derived_from` alone and have the rest
        // preserved — and policies match on `entity` — so gating on two fields
        // dropped exactly those updates on the floor, returning an empty id
        // and taking any edge that named it down with it.
        if !replace && event_carries_nothing(ev) {
            return Ok(String::new());
        }
        let id = ev.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let mut msg = convert::event_to_message(ev, &id, scope.clone(), principal);

        // Field-level merge with any prior node so a re-record that
        // sets only some fields preserves the others.
        if !replace {
            if let Some(&ni) = s.msg_idx.get(&id) {
                if let Some(NodeData::Message(prev)) = s.graph.node_weight(ni) {
                    if msg.derived_from_json.is_none() && prev.derived_from_json.is_some() {
                        msg.derived_from_json = prev.derived_from_json.clone();
                    }
                    if msg.tools_json.is_none() && prev.tools_json.is_some() {
                        msg.tools_json = prev.tools_json.clone();
                    }
                    if msg.role.is_none() && prev.role.is_some() {
                        msg.role = prev.role.clone();
                    }
                    if msg.agent.is_none() && prev.agent.is_some() {
                        msg.agent = prev.agent.clone();
                    }
                    if msg.content.is_none() && prev.content.is_some() {
                        msg.content = prev.content.clone();
                    }
                    // Policies match on `msg.entity`, so a re-record that
                    // omits it drops a field rules depend on — the same
                    // class as derived_from, one field over. `principal`
                    // is deliberately NOT merged: it is stamped per
                    // request at the auth boundary, so the incoming value
                    // is authoritative and a stale one must not survive.
                    if msg.entity.is_none() && prev.entity.is_some() {
                        msg.entity = prev.entity.clone();
                    }
                    // `metadata` is part of the event the content hash
                    // covers, so a re-record that omits it would leave a
                    // node whose hash no longer matches the mark the
                    // writer holds, and every compact reference to it
                    // would be refused.
                    if msg.metadata.is_none() && prev.metadata.is_some() {
                        msg.metadata = prev.metadata.clone();
                    }
                }
            }
        }

        // Idempotent re-ingest. If the field-merged node is exactly the node
        // already stored, this call carries no information: nothing to
        // persist, and nothing to announce. Writing it anyway would bump the
        // sequence and push a `NodeCreated` onto the broadcast ring, so a
        // client that re-sends a whole transcript after a restart would flood
        // the ring, and every subscriber that fell behind would pay a full
        // reload whose duration lets the flood overflow the ring again.
        //
        // Equality is content equality over every field the store persists
        // (`MessageNode: PartialEq`). `principal` is part of it on purpose: it
        // is stamped per request at the auth boundary and deliberately NOT
        // merged above, so a re-record by a different writer IS a change and
        // has to propagate.
        if let Some(&ni) = s.msg_idx.get(&id) {
            if let Some(NodeData::Message(prev)) = s.graph.node_weight(ni) {
                if *prev == msg {
                    return Ok(id);
                }
            }
        }

        self.rocks.batch_put_message(batch, &msg)?;

        // Broadcast the MERGED node, never the raw incoming event.
        // Subscribers (the policy engine among them) maintain their own
        // copy of the graph from this stream, so an update that omits a
        // field the merge above just preserved reads to them as "this
        // field is now empty" — and they drop it. Otherwise a re-record of
        // a tool message without `derived_from` would empty the engine's
        // `ToolResult` relation, taking every rule that joins on it.
        //
        // Rebuilt unconditionally so subscribers and storage always agree
        // on the canonical representation (an unrecognized role, say,
        // normalizes the same way on both sides). Reconstructing re-parses
        // `tools_json` / `derived_from_json`, but a write-path microbench
        // at 8 threads x 20k writes puts that inside the noise of the
        // RocksDB batch and graph insert around it (~35us/op either way).
        let merged_event = convert::message_to_event(&msg);

        let was_new = !s.msg_idx.contains_key(&id);
        let scope = msg.scope.clone();
        let node = NodeData::Message(msg);
        if let Some(&ni) = s.msg_idx.get(&id) {
            *s.graph.node_weight_mut(ni).unwrap() = node;
        } else {
            let ni = s.graph.add_node(node);
            s.msg_idx.insert(id.clone(), ni);
            s.msg_set.insert(id.clone());
        }
        if was_new {
            new_nodes.push(id.clone());
        }
        s.sequence += 1;
        self.global_sequence.fetch_add(1, Ordering::SeqCst);
        replay.push((s.sequence, ChangeKey::Node(id.clone())));

        updates.push(GraphUpdate::NodeCreated {
            id: id.clone(),
            event: merged_event,
            scope,
        });
        Ok(id)
    }

    /// Upsert dependency edges into a held shard.
    #[allow(clippy::too_many_arguments)] // inherent: shard + edges + tenant + principal + staging buffers + scope
    fn merge_dependencies_into_shard(
        &self,
        s: &mut SessionShard,
        edges: &[Edge],
        principal: Option<&str>,
        batch: &mut WriteBatch,
        updates: &mut Vec<GraphUpdate>,
        replay: &mut Vec<(i64, ChangeKey)>,
        scope: &SessionScope,
    ) -> Result<(), GraphError> {
        // Resolve every endpoint BEFORE applying any edge. Skipping the
        // unresolvable ones and returning success meant a caller was told its
        // provenance had been recorded when part of it had been discarded —
        // and an edge naming a node that does not exist is a client bug worth
        // surfacing, not a condition to absorb. All-or-nothing, so a rejected
        // batch leaves no half-applied graph.
        let mut unresolved: Vec<String> = Vec::new();
        for e in edges.iter() {
            // An empty endpoint was skipped with `Ok` — the told-success-while-
            // discarded shape this pre-pass exists to remove, one case over.
            if e.source.is_empty() || e.destination.is_empty() {
                unresolved.push(format!(
                    "{} -> {} (empty endpoint)",
                    e.source, e.destination
                ));
                continue;
            }
            if !s.msg_idx.contains_key(&e.source) || !s.msg_idx.contains_key(&e.destination) {
                unresolved.push(format!("{} -> {}", e.source, e.destination));
            }
        }
        if !unresolved.is_empty() {
            return Err(GraphError::InvalidEdge(format!(
                "dependency edge(s) name nodes that are not recorded in {scope}: {}. \
                 Record the endpoints first, or send them in the same \
                 events+dependencies call",
                unresolved.join(", ")
            )));
        }

        for e in edges {
            if e.source.is_empty() || e.destination.is_empty() {
                continue;
            }
            let (src_ni, dst_ni) = match (
                s.msg_idx.get(&e.source).copied(),
                s.msg_idx.get(&e.destination).copied(),
            ) {
                (Some(a), Some(b)) => (a, b),
                // Unreachable: the pre-pass above rejected these.
                _ => continue,
            };

            // Edges share the envelope's scope; per-edge inference
            // is gone.
            let mut data = EdgeData {
                kind: EdgeKindSer::DependsOn,
                message_index: e.message_index,
                proximal: e.proximal,
                scope: scope.clone(),
                principal: principal.map(|s| s.to_string()),
                entity: e.entity.clone(),
            };
            let from_id = node_id(&s.graph, dst_ni);
            let to_id = node_id(&s.graph, src_ni);
            let key = EdgeKey {
                source: to_id,
                destination: from_id,
                kind: data.kind,
            };
            // Idempotent re-ingest, the edge half — decided on CONTENT, the
            // same way the node half decides it, because `EdgeKey` is only
            // `(source, destination, kind)` and says nothing about
            // `message_index`, `proximal` or `entity`. Skipping on key
            // presence alone dropped a re-record that filled one of those in
            // (`common_policy.dl` documents `message_index = -1` for an edge
            // "recorded without an ordered history", i.e. a field a later
            // record legitimately supplies), and the new value then reached
            // neither RocksDB, nor the in-memory weight, nor any subscriber.
            //
            // Fields the incoming edge omits are carried over from the stored
            // one, as on the node path: a subscriber rebuilds its own copy
            // from the broadcast, so announcing a field as absent when the
            // store kept it is what severs provenance. `principal` is stamped
            // per request at the auth boundary, so the incoming value is
            // authoritative and is not merged.
            //
            // An unchanged re-ingest is what stays silent: there is nothing to
            // add, nothing to count and nothing to announce. Adding it anyway
            // created a PARALLEL edge in petgraph (the graph has no key),
            // inflating the in-memory edge count on every replay of a
            // transcript while RocksDB — keyed by exactly this triple — held
            // one row throughout. Skipping keeps the two in agreement and
            // keeps a restart's re-ingest off the broadcast ring.
            let existing = s.edge_idx.get(&key).copied();
            if let Some(ei) = existing {
                if let Some(prev) = s.graph.edge_weight(ei) {
                    if data.message_index.is_none() {
                        data.message_index = prev.message_index;
                    }
                    if data.proximal.is_none() {
                        data.proximal = prev.proximal;
                    }
                    if data.entity.is_none() {
                        data.entity = prev.entity.clone();
                    }
                    if *prev == data {
                        continue;
                    }
                }
            }
            self.rocks.batch_put_edge(batch, &key, &data)?;
            let announced = data.clone();
            match existing {
                // petgraph stores as (dst_ni -> src_ni); see `bfs`.
                None => {
                    let ei = s.graph.add_edge(dst_ni, src_ni, data);
                    s.edge_idx.insert(key.clone(), ei);
                    if !scope.is_global() {
                        s.edge_count += 1;
                    }
                }
                // Already counted, already indexed: only its data moves.
                Some(ei) => {
                    if let Some(w) = s.graph.edge_weight_mut(ei) {
                        *w = data;
                    }
                }
            }
            s.sequence += 1;
            self.global_sequence.fetch_add(1, Ordering::SeqCst);
            replay.push((s.sequence, ChangeKey::Edge(key)));

            // The MERGED data, never the raw incoming edge — see above.
            updates.push(GraphUpdate::EdgeCreated {
                source: e.source.clone(),
                destination: e.destination.clone(),
                kind: EdgeKindSer::DependsOn,
                message_index: announced.message_index,
                proximal: announced.proximal,
                scope: scope.clone(),
                principal: announced.principal,
                entity: announced.entity,
            });
        }
        Ok(())
    }

    /// Commit a batch + emit broadcasts (durably: commit before broadcast).
    /// ``new_nodes`` ids are folded into ``node_to_session`` *while
    /// the shard lock is still held* so a concurrent reader who sees
    /// the broadcast can always resolve the new node id back to its
    /// shard.
    #[allow(clippy::too_many_arguments)] // inherent: guard + batch + scope + three staging buffers
    fn finalize_shard_write(
        &self,
        mut s: parking_lot::RwLockWriteGuard<'_, SessionShard>,
        mut batch: WriteBatch,
        new_nodes: &[String],
        scope: &SessionScope,
        updates: Vec<GraphUpdate>,
        replay: Vec<(i64, ChangeKey)>,
        _seq_before: i64,
    ) -> Result<(), GraphError> {
        // A call that staged nothing changed nothing, so it says nothing.
        // Everything below writes or announces: the sequence meta-key would be
        // rewritten to the value it already holds (a WAL append and a memtable
        // entry per no-op call), and the two markers would take two slots on
        // the store-wide ring. A client that records one event per RPC and
        // replays its transcript therefore still overflowed the ring and still
        // forced every subscriber to reconcile — the exact flood idempotent
        // ingest exists to remove, surviving in proportion to the client's
        // batch size.
        //
        // Nothing waits on the suppressed markers. Both carry a sequence the
        // shard has already reached, so the write that reached it broadcast
        // that same value; a subscriber that missed it either still has it in
        // the ring, or lagged and reconciles (replay catch-up or a reload),
        // and a fence whose target the subscriber has already applied never
        // waits at all.
        if updates.is_empty() && replay.is_empty() {
            debug_assert!(
                new_nodes.is_empty(),
                "a new node must have staged an update"
            );
            drop(s);
            return Ok(());
        }
        let seq = self.global_sequence.load(Ordering::SeqCst);
        let session_seq = s.sequence;
        self.rocks.batch_set_sequence(&mut batch, seq as u64);
        if !new_nodes.is_empty() {
            let mut nts = self.node_to_session.write();
            let tenant = scope.tenant().to_string();
            for id in new_nodes {
                nts.insert((tenant.clone(), id.clone()), scope.clone());
            }
        }
        // Commit to disk BEFORE broadcasting, and broadcast WHILE STILL HOLDING the
        // shard write lock. Two invariants, both fail-open-closing:
        //
        //  1. Durability before visibility. `commit_batch` returns Err before any
        //     send, so a subscriber never incorporates a fact the write API then
        //     reports as failed — no event that changes authorization and then
        //     vanishes on restart.
        //  2. Per-shard send order == lock-serialized sequence order. If the sends
        //     happened after `drop(s)`, two concurrent same-shard writers (each
        //     having captured its `session_seq` under the lock) could race and emit
        //     `SessionSequence` markers out of order — `marker{6}` before
        //     `content{5}`+`marker{5}` — and a reader fenced at min_sequence=6 would
        //     consider itself caught up while missing the seq-5 write (a read-your-
        //     writes fail-open). `broadcast::Sender::send` never awaits, so holding
        //     the sync shard lock across these ring pushes is safe.
        //
        if let Err(e) = self.rocks.commit_batch(batch) {
            // The shard was mutated on the way here and nothing rolls that
            // back, so it is now ahead of the durable copy for good. Mark it,
            // so reads that feed enforcement refuse rather than serve state a
            // restart will erase. `s` is the write guard, still held.
            s.diverged = true;
            tracing::error!(
                scope = %scope, error = %e,
                "commit failed after the shard was updated; marking the session \
                 diverged — its reads now fail closed until a restart"
            );
            return Err(e);
        }
        // Retain the changes BEFORE announcing them, still under the shard
        // lock. Same critical section and same order as the sends below, so a
        // subscriber that reads `Lagged` off the ring and then asks
        // `updates_since` can never be told about a change that is not yet in
        // the log, and never sees the log's order disagree with the ring's.
        if !replay.is_empty() {
            s.replay.extend(replay);
            let overflow = s.replay.len().saturating_sub(REPLAY_LOG_CAP);
            for _ in 0..overflow {
                s.replay.pop_front();
            }
        }
        for upd in updates {
            let _ = self.tx.send(upd);
        }
        let _ = self.tx.send(GraphUpdate::SessionSequence {
            scope: scope.clone(),
            seq: session_seq,
        });
        let _ = self.tx.send(GraphUpdate::Sequence(seq));
        drop(s);
        Ok(())
    }

    fn open_at(path: &str) -> Result<Self, GraphError> {
        let rocks = RocksStore::open(path)?;
        let (tx, _) = broadcast::channel(CHANGE_CHANNEL_CAP);

        let mut shards: HashMap<SessionScope, Arc<RwLock<SessionShard>>> = HashMap::new();
        let mut node_to_session: HashMap<(String, String), SessionScope> = HashMap::new();
        let mut trace_to_sessions: HashMap<(String, String), HashSet<SessionScope>> =
            HashMap::new();

        for m in rocks.all_messages()? {
            let scope = m.scope.clone();
            let id = m.id.clone();
            let tenant = scope.tenant().to_string();
            let shard = shards
                .entry(scope.clone())
                .or_insert_with(|| Arc::new(RwLock::new(SessionShard::new())));
            let mut s = shard.write();
            let ni = s.graph.add_node(NodeData::Message(m));
            s.msg_idx.insert(id.clone(), ni);
            s.msg_set.insert(id.clone());
            drop(s);
            node_to_session.insert((tenant, id), scope);
        }
        for c in rocks.all_computations()? {
            let scope = c.scope.clone();
            let span_id = c.span_id.clone();
            let trace_id = c.trace_id.clone();
            let tenant = scope.tenant().to_string();
            let shard = shards
                .entry(scope.clone())
                .or_insert_with(|| Arc::new(RwLock::new(SessionShard::new())));
            let mut s = shard.write();
            let ni = s.graph.add_node(NodeData::Computation(c));
            s.comp_idx.insert(span_id.clone(), ni);
            drop(s);
            node_to_session.insert((tenant.clone(), span_id), scope.clone());
            trace_to_sessions
                .entry((tenant, trace_id))
                .or_default()
                .insert(scope);
        }
        for (key, data) in rocks.all_edges()? {
            // Route by the edge's OWN scope, which is authoritative: the live
            // write path puts an edge in the envelope scope's shard and stamps
            // that same scope here, and the storage key is built from it —
            // `all_edges` strips exactly that prefix to recover the key, so
            // trusting it for the shard too is not a new assumption.
            //
            // Resolving the shard from the endpoint ids instead — via
            // `node_to_session`, which is keyed `(tenant, id)` and is
            // last-writer-wins — meant a restart could drop a dependency edge
            // or file it under the wrong session, because clients choose their
            // own event ids and two sessions in one tenant can reuse one. A
            // dropped `DependsOn` edge is a broken taint chain: a rule that
            // denies on tainted provenance stops firing, silently, and only
            // after a restart.
            let scope = data.scope.clone();
            let Some(shard) = shards.get(&scope) else {
                if snapshots::is_immutable(&key.destination) && data.kind == EdgeKindSer::DependsOn
                {
                    return Err(GraphError::ImmutableViolation(
                        "immutable dependency has no persisted shard".into(),
                    ));
                }
                warn!(
                    scope = %scope, source = %key.source, destination = %key.destination,
                    "edge at load names a scope with no shard; dropping it"
                );
                continue;
            };
            let mut s = shard.write();
            let (Some(src_ni), Some(dst_ni)) =
                (s.resolve(&key.source), s.resolve(&key.destination))
            else {
                // Immutable snapshots must never reload with a smaller incoming
                // set than their durable transaction established.
                if snapshots::is_immutable(&key.destination) && data.kind == EdgeKindSer::DependsOn
                {
                    return Err(GraphError::ImmutableViolation(
                        "immutable dependency has a missing persisted endpoint".into(),
                    ));
                }
                // Legacy orphan rows retain their existing recovery behavior.
                warn!(
                    scope = %scope, source = %key.source, destination = %key.destination,
                    "edge at load has an endpoint missing from its shard; dropping it"
                );
                continue;
            };
            // `edge_count` is the number of MESSAGE-DEPENDENCY edges in the
            // scope — that is what every runtime bump and decrement counts,
            // and what an evaluator's bootstrap loads. Counting the
            // computation edges (CHILD_OF / PRODUCES / CONSUMES) here too made
            // the number mean one thing on a fresh shard and another on a
            // restarted one, and a catch-up that installs it as the
            // evaluator's edge total then reported more edges than the
            // evaluator holds.
            let counts_for_session = !data.scope.is_global() && data.kind == EdgeKindSer::DependsOn;
            let ei = s.graph.add_edge(dst_ni, src_ni, data);
            // Build the per-shard edge index as the shard loads, so a
            // re-ingest after a restart recognises an edge that was already
            // durable instead of adding a parallel copy of it.
            s.edge_idx.insert(key, ei);
            if counts_for_session {
                s.edge_count += 1;
            }
        }

        let sequence = rocks.get_sequence()? as i64;

        Ok(Self {
            shards: RwLock::new(shards),
            node_to_session: RwLock::new(node_to_session),
            trace_to_sessions: RwLock::new(trace_to_sessions),
            global_sequence: AtomicI64::new(sequence),
            rocks,
            tx,
            owner_claim_lock: parking_lot::Mutex::new(HashMap::new()),
        })
    }

    fn temp() -> Self {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().to_str().unwrap().to_string();
        let rocks = RocksStore::open(&path).expect("rocks open");
        let (tx, _) = broadcast::channel(CHANGE_CHANNEL_CAP);
        std::mem::forget(dir);
        Self {
            shards: RwLock::new(HashMap::new()),
            node_to_session: RwLock::new(HashMap::new()),
            trace_to_sessions: RwLock::new(HashMap::new()),
            global_sequence: AtomicI64::new(0),
            rocks,
            tx,
            owner_claim_lock: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    fn upsert_comp(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        c: &Computation,
    ) -> Result<String, GraphError> {
        if snapshots::is_immutable(&c.span_id) {
            return Err(GraphError::ImmutableViolation(
                "version IDs cannot name computations".into(),
            ));
        }
        let comp = convert::proto_to_comp(c, scope.clone(), principal);
        let span_id = comp.span_id.clone();
        let trace_id = comp.trace_id.clone();
        let scope = comp.scope.clone();

        let parent_span_id = comp.parent_span_id.clone();
        let output_msg_id = comp.output_message_id.clone();
        let input_msg_ids = comp.input_message_ids.clone();

        let shard = self.get_or_create_shard(&scope);
        let comp_changed;
        {
            let mut s = shard.write();
            let was_new = !s.comp_idx.contains_key(&span_id);
            let node = NodeData::Computation(comp);
            // A span the store already holds byte for byte is not news:
            // skip the RocksDB write and the ComputationCreated send, so a
            // transcript replay of the same spans costs the ring nothing.
            // The edges below are still evaluated — a link the first ingest
            // could not draw (its message had not arrived yet) is drawn now.
            let existing = s.comp_idx.get(&span_id).copied();
            // `changed` is decided against the STORED node, and the RocksDB
            // write happens BEFORE the shard is touched: durable before
            // visible, the same rule the message path and `delete_dependency`
            // keep. Deciding against the in-memory node after mutating it
            // would make a failed put unrepeatable — the retry would see
            // `changed == false` and never write, leaving the span in memory
            // only until a restart silently dropped it.
            let node_changed = match existing.and_then(|ni| s.graph.node_weight(ni)) {
                Some(w) => *w != node,
                None => true,
            };
            if node_changed {
                if let NodeData::Computation(ref stored) = node {
                    self.rocks.put_computation(stored)?;
                }
            }
            let ni = match existing {
                Some(ni) => {
                    if node_changed {
                        if let Some(w) = s.graph.node_weight_mut(ni) {
                            *w = node;
                        }
                    }
                    ni
                }
                None => {
                    let ni = s.graph.add_node(node);
                    s.comp_idx.insert(span_id.clone(), ni);
                    ni
                }
            };
            comp_changed = node_changed;
            // Populate reverse indices under the shard lock so a
            // reader who sees the broadcast can resolve span_id.
            if was_new {
                let tenant = scope.tenant().to_string();
                self.node_to_session
                    .write()
                    .insert((tenant.clone(), span_id.clone()), scope.clone());
                self.trace_to_sessions
                    .write()
                    .entry((tenant, trace_id.clone()))
                    .or_default()
                    .insert(scope.clone());
            }

            // CHILD_OF: add only if the parent lives in the same shard.
            if let Some(ref pid) = parent_span_id {
                if let Some(&pni) = s.comp_idx.get(pid) {
                    let changed = add_intra_shard_edge(
                        &mut s,
                        &self.rocks,
                        ni,
                        pni,
                        EdgeKind::ChildOf,
                        None,
                        None,
                        scope.clone(),
                        principal,
                    )?;
                    if changed {
                        let _ = self.tx.send(GraphUpdate::ComputationEdgeCreated {
                            parent_span_id: pid.clone(),
                            child_span_id: span_id.clone(),
                            scope: scope.clone(),
                        });
                    }
                }
            }
            // PRODUCES
            if let Some(ref mid) = output_msg_id {
                if let Some(&mni) = s.msg_idx.get(mid) {
                    let changed = add_intra_shard_edge(
                        &mut s,
                        &self.rocks,
                        ni,
                        mni,
                        EdgeKind::Produces,
                        None,
                        None,
                        scope.clone(),
                        principal,
                    )?;
                    if changed {
                        let _ = self.tx.send(GraphUpdate::ComputationMessageLinkCreated {
                            span_id: span_id.clone(),
                            message_id: mid.clone(),
                            is_produces: true,
                            scope: scope.clone(),
                        });
                    }
                }
            }
            // CONSUMES
            for (idx, mid) in input_msg_ids.iter().enumerate() {
                if let Some(&mni) = s.msg_idx.get(mid) {
                    let changed = add_intra_shard_edge(
                        &mut s,
                        &self.rocks,
                        ni,
                        mni,
                        EdgeKind::Consumes,
                        Some(idx as u32),
                        None,
                        scope.clone(),
                        principal,
                    )?;
                    if changed {
                        let _ = self.tx.send(GraphUpdate::ComputationMessageLinkCreated {
                            span_id: span_id.clone(),
                            message_id: mid.clone(),
                            is_produces: false,
                            scope: scope.clone(),
                        });
                    }
                }
            }
        }

        // No sequence increment for computations — they're not
        // policy-relevant and don't fire Sequence markers.
        if comp_changed {
            let _ = self.tx.send(GraphUpdate::ComputationCreated {
                computation: Computation {
                    principal: principal.map(|s| s.to_string()),
                    ..c.clone()
                },
                scope,
            });
        }
        Ok(span_id)
    }
}

// ── Free helpers ─────────────────────────────────────

/// The client-supplied id of the message/computation node at `ni`
/// (empty string if the node is missing).
fn node_id(graph: &StableDiGraph<NodeData, EdgeData>, ni: NodeIndex) -> String {
    graph
        .node_weight(ni)
        .map(|n| n.id().to_string())
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)] // inherent: edge endpoints + kind + index + proximal + scope + principal
fn add_intra_shard_edge(
    s: &mut SessionShard,
    rocks: &RocksStore,
    from: NodeIndex,
    to: NodeIndex,
    kind: EdgeKind,
    message_index: Option<u32>,
    proximal: Option<bool>,
    scope: SessionScope,
    principal: Option<&str>,
) -> Result<bool, GraphError> {
    let data = EdgeData {
        kind: EdgeKindSer::from(kind),
        message_index,
        proximal,
        scope,
        principal: principal.map(|s| s.to_string()),
        entity: None,
    };
    let from_id = node_id(&s.graph, from);
    let to_id = node_id(&s.graph, to);
    let key = EdgeKey {
        source: to_id,
        destination: from_id,
        kind: data.kind,
    };
    // Computation edges (ChildOf / Produces / Consumes) go through the same
    // key, so a re-recorded span updates the edge it already has instead of
    // stacking a parallel copy beside it. RocksDB was always an overwrite
    // here; only the in-memory graph inflated. The comparison sits above the
    // RocksDB write as well: an edge whose data is already what we would
    // store is not written and not announced, so a replayed transcript costs
    // neither a put nor a ring slot. Returns whether anything changed.
    //
    // `changed` is decided against the STORED weight and the put happens
    // BEFORE the shard is touched — durable before visible. Were the order
    // reversed, a failed put would leave the edge in memory only and the
    // retry would compare against the value it just wrote in memory, decide
    // nothing changed, and never reach disk.
    let existing = s.edge_idx.get(&key).copied();
    let changed = match existing.and_then(|ei| s.graph.edge_weight(ei)) {
        Some(w) => *w != data,
        None => true,
    };
    if !changed {
        return Ok(false);
    }
    rocks.put_edge(&key, &data)?;
    match existing.filter(|&ei| s.graph.edge_weight(ei).is_some()) {
        Some(ei) => {
            if let Some(w) = s.graph.edge_weight_mut(ei) {
                *w = data;
            }
        }
        // Absent, or indexed but absent from the graph: (re)create the entry.
        None => {
            let ei = s.graph.add_edge(from, to, data);
            s.edge_idx.insert(key, ei);
        }
    }
    Ok(true)
}

fn bfs(
    graph: &StableDiGraph<NodeData, EdgeData>,
    start: NodeIndex,
    direction: Direction,
    max_depth: Option<u32>,
) -> (Vec<NodeData>, Vec<(String, String, EdgeData)>) {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0u32));

    while let Some((ni, depth)) = queue.pop_front() {
        if let Some(nd) = graph.node_weight(ni) {
            nodes.push(nd.clone());
        }
        if max_depth.is_some_and(|m| depth >= m) {
            continue;
        }
        for edge in graph.edges_directed(ni, direction) {
            let neighbour = match direction {
                Direction::Outgoing => edge.target(),
                Direction::Incoming => edge.source(),
            };
            // DependsOn edges are stored in petgraph as
            // (destination → source); swap back to proto convention
            // (source = predecessor, destination = successor).
            let (pg_src, pg_dst) = graph.edge_endpoints(edge.id()).unwrap();
            edges.push((
                node_id(graph, pg_dst),
                node_id(graph, pg_src),
                edge.weight().clone(),
            ));
            if visited.insert(neighbour) {
                queue.push_back((neighbour, depth + 1));
            }
        }
    }
    (nodes, edges)
}

fn to_proto_graph(
    nodes: &[NodeData],
    edges: &[(String, String, EdgeData)],
) -> (Vec<Event>, Vec<Edge>) {
    let events: Vec<Event> = nodes.iter().filter_map(convert::node_to_event).collect();
    let proto_edges: Vec<Edge> = edges
        .iter()
        .filter_map(|(s, d, data)| convert::depends_on_to_proto(s, d, data))
        .collect();
    (events, proto_edges)
}

// ── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::observability::{Computation, Edge, Event, Role, Tool};

    const T: &str = "default";

    fn global_scope() -> SessionScope {
        SessionScope::global(T)
    }
    fn scope_for(sid: &str) -> SessionScope {
        if sid.is_empty() {
            SessionScope::global(T)
        } else {
            SessionScope::new(T, sid)
        }
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

    fn dep(src: &str, dst: &str) -> Edge {
        Edge {
            source: src.to_string(),
            destination: dst.to_string(),
            message_index: None,
            proximal: None,
            principal: None,
            entity: None,
        }
    }

    /// Measure concurrent merge_events throughput with a live broadcast
    /// subscriber, including durable commit and broadcast costs. Run explicitly:
    ///   cargo test -p sasy-graph --release write_path_microbench \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore]
    fn write_path_microbench() {
        use std::time::Instant;
        let threads: usize = std::env::var("BENCH_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let per: usize = std::env::var("BENCH_WRITES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20_000);

        let store = Arc::new(GraphStore::temp());
        // A live subscriber, so `tx.send` pushes to the ring (realistic) rather than
        // short-circuiting on "no receivers". Left un-drained: send stays O(1) either
        // way, and the point is the WRITE-side cost, not consumer throughput.
        let _rx = store.tx.subscribe();

        let run = |same_shard: bool| {
            let barrier = Arc::new(std::sync::Barrier::new(threads));
            let handles: Vec<_> = (0..threads)
                .map(|ti| {
                    let store = Arc::clone(&store);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        let t0 = Instant::now();
                        for i in 0..per {
                            let sess = if same_shard {
                                "shared".to_string()
                            } else {
                                format!("s{ti}")
                            };
                            let scope = SessionScope::new("bench", &sess);
                            let id = format!("{ti}-{i}");
                            store
                                .merge_events(&scope, None, vec![ev(&id, "x")])
                                .unwrap();
                        }
                        t0.elapsed()
                    })
                })
                .collect();
            let mut total_dt = std::time::Duration::ZERO;
            for h in handles {
                total_dt += h.join().unwrap();
            }
            let n = threads * per;
            // Wall-clock via the max per-thread time is noisy; use aggregate op time.
            let per_op_us = total_dt.as_micros() as f64 / n as f64;
            eprintln!(
                "[durbench] {} threads={} n={} mean_per_op={:.3}us throughput≈{:.0}/s",
                if same_shard {
                    "SAME-SHARD (contended)"
                } else {
                    "PER-SHARD (uncontended)"
                },
                threads,
                n,
                per_op_us,
                1_000_000.0 / per_op_us * threads as f64,
            );
        };
        run(false);
        run(true);
    }

    fn span(sid: &str, tid: &str) -> Computation {
        Computation {
            span_id: sid.to_string(),
            trace_id: tid.to_string(),
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
            output_message_id: None,
            linked_span_ids: vec![],
            principal: None,
            entity: None,
        }
    }

    #[test]
    fn session_metadata_append_get_delete() {
        use crate::persistence::PolicyMetadataFact;
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "conv-1");
        let fact = PolicyMetadataFact {
            rel: "detaint_denied".into(),
            a: "src-node".into(),
            b: "".into(),
        };
        store
            .append_session_metadata(&scope, std::slice::from_ref(&fact))
            .unwrap();
        assert_eq!(store.get_session_metadata(&scope).unwrap().len(), 1);
        // Exact-duplicate append is skipped (append-only set).
        store
            .append_session_metadata(&scope, std::slice::from_ref(&fact))
            .unwrap();
        assert_eq!(store.get_session_metadata(&scope).unwrap().len(), 1);
        // EndSession path drops it — a recycled id can't inherit the decision.
        store.delete_session_metadata(&scope).unwrap();
        assert!(store.get_session_metadata(&scope).unwrap().is_empty());
    }

    #[test]
    fn merge_and_get_full_state() {
        let store = GraphStore::new(None).unwrap();
        let ids = store
            .merge_events(
                &global_scope(),
                None,
                vec![ev("m1", "hello"), ev("m2", "world")],
            )
            .unwrap();
        assert_eq!(ids, vec!["m1", "m2"]);

        let (events, _, seq) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(seq, 2);
    }

    #[test]
    fn merge_idempotent() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("m1", "old")])
            .unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("m1", "new")])
            .unwrap();

        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text.as_deref(), Some("new"));
    }

    #[test]
    fn a_re_ingest_of_identical_events_emits_nothing() {
        // A client may re-send a whole transcript after a restart. If the
        // store treats each re-sent event as a mutation it floods the
        // broadcast ring, and every subscriber that falls behind pays a full
        // reload — the thrash this idempotency check exists to stop.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let events = vec![ev("m1", "hello"), ev("m2", "world")];
        store.merge_events(&scope, None, events.clone()).unwrap();
        let seq_after_first = store.session_sequence(&scope);

        let mut rx = store.subscribe();
        let ids = store.merge_events(&scope, None, events).unwrap();

        assert_eq!(ids, vec!["m1", "m2"], "the re-ingest still echoes the ids");
        assert_eq!(
            store.session_sequence(&scope),
            seq_after_first,
            "an identical re-ingest bumped the scope sequence"
        );
        let mut payloads = 0usize;
        let mut marker_seqs = vec![];
        while let Ok(upd) = rx.try_recv() {
            match upd {
                GraphUpdate::SessionSequence { seq, .. } => marker_seqs.push(seq),
                GraphUpdate::Sequence(_) => {}
                _ => payloads += 1,
            }
        }
        assert_eq!(payloads, 0, "an identical re-ingest broadcast a change");
        // Not even a marker: its sequence is one every subscriber was already
        // told about by the write that reached it, so re-announcing it only
        // spends ring slots — two per call, which is enough for a client that
        // records one event per RPC to overflow the ring on a replay all by
        // itself.
        assert!(
            marker_seqs.is_empty(),
            "an identical re-ingest spent ring slots on markers: {marker_seqs:?}"
        );
    }

    #[test]
    fn a_re_ingest_of_an_identical_span_emits_nothing() {
        // Spans took the one ingest path with no idempotency check: every
        // re-record wrote RocksDB and sent one ComputationCreated plus one
        // link per produces/consumes edge, so a replayed transcript overflowed
        // the ring on the span half alone even after the event half went
        // quiet.
        fn dir_bytes(p: &std::path::Path) -> u64 {
            std::fs::read_dir(p)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let store = GraphStore::new(Some(path.to_str().unwrap())).unwrap();
        let scope = scope_for("s1");
        store
            .merge_events(
                &scope,
                None,
                vec![ev("m0", "out"), ev("m1", "in1"), ev("m2", "in2")],
            )
            .unwrap();
        let mut root = span("root", "t1");
        let mut child = span("child", "t1");
        child.parent_span_id = Some("root".to_string());
        child.output_message_id = Some("m0".to_string());
        child.input_message_ids = vec!["m1".to_string(), "m2".to_string()];
        root.output_message_id = None;
        let spans = vec![root, child];
        store
            .merge_computations(&scope, None, spans.clone())
            .unwrap();
        let edges_after_first = store.get_trace(T, "t1", None, None, None).unwrap();

        let mut rx = store.subscribe();
        let before = dir_bytes(&path);
        for _ in 0..100 {
            let ids = store
                .merge_computations(&scope, None, spans.clone())
                .unwrap();
            assert_eq!(
                ids,
                vec!["root", "child"],
                "the re-ingest still echoes the ids"
            );
        }
        // ~840 bytes per re-ingest before the check; the slack absorbs the
        // housekeeping RocksDB does on its own.
        let grew = dir_bytes(&path).saturating_sub(before);
        assert!(
            grew < 10_000,
            "100 identical span re-ingests grew RocksDB by {grew} bytes"
        );

        let mut ring = 0usize;
        while rx.try_recv().is_ok() {
            ring += 1;
        }
        assert_eq!(
            ring, 0,
            "100 identical span re-ingests spent {ring} ring slots"
        );
        let edges_now = store.get_trace(T, "t1", None, None, None).unwrap();
        assert_eq!(
            edges_now.computation_edges.len(),
            edges_after_first.computation_edges.len(),
            "a span re-ingest stacked parallel edges"
        );
    }

    #[test]
    fn a_span_link_the_first_ingest_could_not_draw_is_drawn_on_the_second() {
        // The idempotency check is per node and per edge, not a whole-call
        // early return: a span recorded before its message arrives has no
        // PRODUCES edge, and the re-record that follows the message must
        // still draw it.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let mut sp = span("s1", "t1");
        sp.output_message_id = Some("m0".to_string());
        store
            .merge_computations(&scope, None, vec![sp.clone()])
            .unwrap();
        assert!(
            store
                .get_trace(T, "t1", None, None, None)
                .unwrap()
                .message_edges
                .is_empty(),
            "the link was drawn before the message existed"
        );

        store
            .merge_events(&scope, None, vec![ev("m0", "out")])
            .unwrap();
        let mut rx = store.subscribe();
        store.merge_computations(&scope, None, vec![sp]).unwrap();

        let mut links = 0usize;
        while let Ok(upd) = rx.try_recv() {
            if matches!(upd, GraphUpdate::ComputationMessageLinkCreated { .. }) {
                links += 1;
            }
        }
        assert_eq!(links, 1, "the newly drawable link was not announced");
        assert_eq!(
            store
                .get_trace(T, "t1", None, None, None)
                .unwrap()
                .message_edges
                .len(),
            1
        );
    }

    #[test]
    fn a_failed_span_write_leaves_the_shard_unchanged() {
        // Durable before visible on the computation path. With the put after
        // the in-memory insert, a failed put left the span in `s.graph` with
        // nothing on disk and nothing marking the shard diverged; worse, the
        // retry compared the incoming span against the copy the failed call
        // had already installed, decided nothing changed, and never wrote —
        // so the span stayed memory-only until a restart erased it.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let mut sp = span("s1", "t1");
        sp.output_message_id = Some("m0".to_string());
        store
            .merge_events(&scope, None, vec![ev("m0", "out")])
            .unwrap();
        let mut rx = store.subscribe();

        store.rocks.set_fail_writes_for_test(true);
        store
            .merge_computations(&scope, None, vec![sp.clone()])
            .expect_err("the RocksDB write failed, so the call must fail");

        let trace = store.get_trace(T, "t1", None, None, None).unwrap();
        assert!(
            trace.computations.is_empty(),
            "a failed put left the span visible in memory: {:?}",
            trace.computations
        );
        assert!(
            trace.message_edges.is_empty(),
            "a failed put left the PRODUCES edge visible in memory"
        );
        let announced = {
            let mut n = 0usize;
            while rx.try_recv().is_ok() {
                n += 1;
            }
            n
        };
        assert_eq!(announced, 0, "a failed put still announced something");

        // The retry must still see the span as a change and write it.
        store.rocks.set_fail_writes_for_test(false);
        store
            .merge_computations(&scope, None, vec![sp])
            .expect("the retry after the fault must write");
        let trace = store.get_trace(T, "t1", None, None, None).unwrap();
        assert_eq!(
            trace.computations.len(),
            1,
            "the retry did not store the span"
        );
        assert_eq!(
            trace.message_edges.len(),
            1,
            "the retry did not store the PRODUCES edge"
        );
        assert!(
            store.rocks.get_computation(&scope, "s1").unwrap().is_some(),
            "the retry left the span off disk"
        );
        let (nodes, links) = {
            let (mut nodes, mut links) = (0usize, 0usize);
            while let Ok(upd) = rx.try_recv() {
                match upd {
                    GraphUpdate::ComputationCreated { .. } => nodes += 1,
                    GraphUpdate::ComputationMessageLinkCreated { .. } => links += 1,
                    _ => {}
                }
            }
            (nodes, links)
        };
        assert_eq!(
            (nodes, links),
            (1, 1),
            "the retry announced the wrong count"
        );
    }

    #[test]
    fn a_failed_span_edge_write_leaves_the_shard_unchanged() {
        // The same rule for `add_intra_shard_edge`, exercised on its own: the
        // span node is already stored and unchanged, so only the edge write
        // runs — and only the edge write can fail.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let mut sp = span("s1", "t1");
        sp.output_message_id = Some("m0".to_string());
        store
            .merge_computations(&scope, None, vec![sp.clone()])
            .unwrap();
        store
            .merge_events(&scope, None, vec![ev("m0", "out")])
            .unwrap();
        let mut rx = store.subscribe();

        store.rocks.set_fail_writes_for_test(true);
        store
            .merge_computations(&scope, None, vec![sp.clone()])
            .expect_err("the edge write failed, so the call must fail");
        assert!(
            store
                .get_trace(T, "t1", None, None, None)
                .unwrap()
                .message_edges
                .is_empty(),
            "a failed edge put left the edge visible in memory"
        );

        store.rocks.set_fail_writes_for_test(false);
        store
            .merge_computations(&scope, None, vec![sp])
            .expect("the retry after the fault must write");
        assert_eq!(
            store
                .get_trace(T, "t1", None, None, None)
                .unwrap()
                .message_edges
                .len(),
            1,
            "the retry did not draw the edge — `changed` was decided against \
             the value the failed call installed in memory"
        );
        let links = {
            let mut n = 0usize;
            while let Ok(upd) = rx.try_recv() {
                if matches!(upd, GraphUpdate::ComputationMessageLinkCreated { .. }) {
                    n += 1;
                }
            }
            n
        };
        assert_eq!(links, 1, "the edge was announced {links} times, want once");
    }

    #[test]
    fn an_identical_re_ingest_writes_nothing_to_rocksdb() {
        // The sequence meta-key was staged and committed on every call,
        // whether or not anything changed — a WAL append plus a memtable entry
        // per no-op, on a path a replaying client walks once per RPC.
        fn dir_bytes(p: &std::path::Path) -> u64 {
            std::fs::read_dir(p)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let store = GraphStore::new(Some(path.to_str().unwrap())).unwrap();
        let scope = scope_for("s1");
        let events = vec![ev("m1", "hello")];
        store.merge_events(&scope, None, events.clone()).unwrap();

        let before = dir_bytes(&path);
        for _ in 0..20_000 {
            store.merge_events(&scope, None, events.clone()).unwrap();
        }
        let after = dir_bytes(&path);

        // Every committed batch grew the store by ~39 bytes, so 20k no-ops
        // cost ~780 KB. Zero is the expectation; the slack absorbs any
        // background housekeeping RocksDB does on its own.
        let grew = after.saturating_sub(before);
        assert!(
            grew < 100_000,
            "20k no-op re-ingests grew the store by {grew} bytes"
        );
    }

    #[test]
    fn a_re_record_that_fills_a_missing_field_emits_one_node_created() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let mut bare = ev("m1", "payload");
        bare.entity = None;
        store.merge_events(&scope, None, vec![bare]).unwrap();
        let seq_before = store.session_sequence(&scope);

        let mut rx = store.subscribe();
        let mut with_entity = ev("m1", "payload");
        with_entity.entity = Some("catalog-reader".to_string());
        store.merge_events(&scope, None, vec![with_entity]).unwrap();

        assert_eq!(
            store.session_sequence(&scope),
            seq_before + 1,
            "a real change must bump the sequence exactly once"
        );
        let created: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter(|u| matches!(u, GraphUpdate::NodeCreated { .. }))
            .collect();
        assert_eq!(created.len(), 1, "expected exactly one NodeCreated");
        let GraphUpdate::NodeCreated { event, .. } = &created[0] else {
            unreachable!()
        };
        assert_eq!(event.entity.as_deref(), Some("catalog-reader"));
    }

    #[test]
    fn merge_dependencies() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("m1", "a"), ev("m2", "b")])
            .unwrap();
        store
            .merge_dependencies(&global_scope(), None, vec![dep("m1", "m2")])
            .unwrap();

        let (_, edges, _) = store.get_full_state().unwrap();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn a_duplicate_dependency_never_creates_a_parallel_edge() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        store
            .merge_events(&scope, None, vec![ev("m1", "a"), ev("m2", "b")])
            .unwrap();
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();
        let (_, edges_after_first) = store.session_counts(&scope);
        let seq_after_first = store.session_sequence(&scope);

        let mut rx = store.subscribe();
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();

        assert_eq!(store.session_counts(&scope).1, edges_after_first);
        assert_eq!(
            store.session_sequence(&scope),
            seq_after_first,
            "a duplicate dependency bumped the scope sequence"
        );
        let payloads = std::iter::from_fn(|| rx.try_recv().ok())
            .filter(|u| {
                !matches!(
                    u,
                    GraphUpdate::SessionSequence { .. } | GraphUpdate::Sequence(_)
                )
            })
            .count();
        assert_eq!(payloads, 0, "a duplicate dependency broadcast an edge");

        let (_, proto_edges, _) = store.get_full_state().unwrap();
        assert_eq!(proto_edges.len(), 1, "a parallel edge was stored");
    }

    #[test]
    fn a_dependency_re_record_that_fills_a_missing_field_is_stored_and_announced() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        store
            .merge_events(&scope, None, vec![ev("m1", "a"), ev("m2", "b")])
            .unwrap();
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();
        let seq_after_first = store.session_sequence(&scope);
        let (_, edges_after_first) = store.session_counts(&scope);

        let mut rx = store.subscribe();
        let mut later = dep("m1", "m2");
        later.message_index = Some(7);
        later.entity = Some("late-entity".to_string());
        store.merge_dependencies(&scope, None, vec![later]).unwrap();

        assert_eq!(
            store.session_sequence(&scope),
            seq_after_first + 1,
            "an edge re-record carrying new data must bump the sequence once"
        );
        assert_eq!(
            store.session_counts(&scope).1,
            edges_after_first,
            "updating an edge's data must not add a second edge"
        );
        let created: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter(|u| matches!(u, GraphUpdate::EdgeCreated { .. }))
            .collect();
        assert_eq!(created.len(), 1, "expected exactly one EdgeCreated");
        let GraphUpdate::EdgeCreated {
            message_index,
            entity,
            ..
        } = &created[0]
        else {
            unreachable!()
        };
        assert_eq!(*message_index, Some(7));
        assert_eq!(entity.as_deref(), Some("late-entity"));

        // Stored, not just announced.
        let (_, proto_edges, _) = store.get_full_state().unwrap();
        assert_eq!(proto_edges.len(), 1, "a parallel edge was stored");

        // And a third record that omits the fields keeps them and stays silent:
        // announcing them as absent is what severs provenance downstream.
        let seq_after_update = store.session_sequence(&scope);
        while rx.try_recv().is_ok() {}
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();
        assert_eq!(
            store.session_sequence(&scope),
            seq_after_update,
            "an edge re-record that adds nothing bumped the sequence"
        );
        let payloads = std::iter::from_fn(|| rx.try_recv().ok())
            .filter(|u| {
                !matches!(
                    u,
                    GraphUpdate::SessionSequence { .. } | GraphUpdate::Sequence(_)
                )
            })
            .count();
        assert_eq!(payloads, 0, "an unchanged edge re-record broadcast");
    }

    #[test]
    fn deleting_an_absent_edge_emits_nothing() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        store
            .merge_events(&scope, None, vec![ev("m1", "a"), ev("m2", "b")])
            .unwrap();
        let seq_before = store.session_sequence(&scope);

        let mut rx = store.subscribe();
        assert!(
            !store.delete_dependency(&scope, "m1", "m2").unwrap(),
            "deleting an edge that was never recorded reported a removal"
        );
        assert_eq!(store.session_sequence(&scope), seq_before);
        assert!(rx.try_recv().is_err(), "an absent-edge delete broadcast");

        // And a real delete does emit, exactly once.
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();
        while rx.try_recv().is_ok() {}
        assert!(store.delete_dependency(&scope, "m1", "m2").unwrap());
        let deleted = std::iter::from_fn(|| rx.try_recv().ok())
            .filter(|u| matches!(u, GraphUpdate::EdgeDeleted { .. }))
            .count();
        assert_eq!(deleted, 1);
        assert_eq!(store.session_counts(&scope).1, 0);
        // Deleting it a second time is a no-op again.
        assert!(!store.delete_dependency(&scope, "m1", "m2").unwrap());
    }

    #[test]
    fn a_reopened_store_recognises_its_own_edges_instead_of_duplicating_them() {
        // The in-memory edge index is rebuilt from RocksDB at load. Without
        // it, the transcript replay that follows a restart re-added every
        // dependency as a parallel edge.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let scope = scope_for("s1");
        {
            let store = GraphStore::new(Some(path)).unwrap();
            store
                .merge_events(&scope, None, vec![ev("m1", "a"), ev("m2", "b")])
                .unwrap();
            store
                .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
                .unwrap();
        }
        let store = GraphStore::new(Some(path)).unwrap();
        assert_eq!(store.session_counts(&scope).1, 1);
        store
            .merge_dependencies(&scope, None, vec![dep("m1", "m2")])
            .unwrap();
        assert_eq!(
            store.session_counts(&scope).1,
            1,
            "the re-ingest after a reopen added a parallel edge"
        );
    }

    #[test]
    fn a_replay_hands_back_exactly_the_changes_after_a_sequence() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        store
            .merge_events(&scope, None, vec![ev("m1", "a"), ev("m2", "b")])
            .unwrap();
        let checkpoint = store.session_sequence(&scope);

        store
            .merge_events_with_dependencies(
                &scope,
                None,
                vec![ev("m3", "c")],
                vec![dep("m1", "m3")],
            )
            .unwrap();

        let replayed = store.updates_since(&scope, checkpoint).unwrap();
        assert_eq!(replayed.len(), 2, "expected the node and its edge, only");
        match &replayed[0] {
            GraphUpdate::NodeCreated { id, .. } => assert_eq!(id, "m3"),
            other => panic!("expected NodeCreated first, got {other:?}"),
        }
        match &replayed[1] {
            GraphUpdate::EdgeCreated {
                source,
                destination,
                ..
            } => {
                assert_eq!(source, "m1");
                assert_eq!(destination, "m3");
            }
            other => panic!("expected EdgeCreated second, got {other:?}"),
        }

        // Replaying everything reproduces the shard's whole state.
        let all = store.updates_since(&scope, 0).unwrap();
        let (events, edges, _) = store.get_session_state(&scope).unwrap();
        let mut replayed_ids: Vec<String> = all
            .iter()
            .filter_map(|u| match u {
                GraphUpdate::NodeCreated { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        replayed_ids.sort();
        let mut stored_ids: Vec<String> = events.iter().filter_map(|e| e.id.clone()).collect();
        stored_ids.sort();
        assert_eq!(replayed_ids, stored_ids);
        let replayed_edges = all
            .iter()
            .filter(|u| matches!(u, GraphUpdate::EdgeCreated { .. }))
            .count();
        assert_eq!(replayed_edges, edges.len());
    }

    #[test]
    fn a_subscriber_past_the_ring_still_catches_up_from_the_replay_log() {
        // More updates than the broadcast ring holds, so the subscriber is
        // guaranteed to see `Lagged`, which must be served from the replay log
        // rather than a full reload of the shard.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let mut rx = store.subscribe();
        let n = CHANGE_CHANNEL_CAP + 1_000;
        let events: Vec<Event> = (0..n).map(|i| ev(&format!("m{i}"), "x")).collect();
        store.merge_events(&scope, None, events).unwrap();

        let lagged = loop {
            match rx.try_recv() {
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => break true,
                Err(_) => break false,
            }
        };
        assert!(lagged, "the ring did not overflow, so this pins nothing");

        let replayed = store.updates_since(&scope, 0).unwrap();
        assert_eq!(replayed.len(), n, "replay lost updates the ring dropped");
        let (events, _, _) = store
            .get_full_state_for_tenant_scoped(scope.tenant())
            .unwrap();
        assert_eq!(
            replayed.len(),
            events.len(),
            "the caught-up state differs from the store's own"
        );
    }

    #[test]
    fn a_gap_older_than_the_retained_log_cannot_be_replayed() {
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let n = REPLAY_LOG_CAP + 1_000;
        let events: Vec<Event> = (0..n).map(|i| ev(&format!("m{i}"), "x")).collect();
        store.merge_events(&scope, None, events).unwrap();

        assert!(
            store.updates_since(&scope, 0).is_none(),
            "a gap the log no longer covers must send the caller to a reload"
        );
        // The oldest record still retained is reachable.
        let oldest = (n - REPLAY_LOG_CAP) as i64;
        let from_oldest = store.updates_since(&scope, oldest).unwrap();
        assert_eq!(from_oldest.len(), REPLAY_LOG_CAP);
    }

    #[test]
    fn the_documented_ring_and_log_sizes_match_the_types() {
        // The two capacity constants are documented in megabytes, and both
        // numbers come from a type's size: the broadcast ring allocates every
        // slot up front, and a replay entry is inline in the deque. A layout
        // change silently moves the baseline RSS of every binary that opens a
        // store, so pin the arithmetic rather than the prose.
        let slot = std::mem::size_of::<GraphUpdate>();
        let ring_mb = (slot * CHANGE_CHANNEL_CAP) as f64 / 1e6;
        assert!(
            (13.0..14.5).contains(&ring_mb),
            "the ring is {ring_mb:.1} MB ({slot} bytes x {CHANGE_CHANNEL_CAP}), not the ~13.6 MB documented"
        );
        let entry = std::mem::size_of::<(i64, ChangeKey)>();
        let log_mb = (entry * REPLAY_LOG_CAP) as f64 / 1e6;
        assert!(
            (4.0..6.0).contains(&log_mb),
            "a full replay log is {log_mb:.1} MB inline ({entry} bytes x {REPLAY_LOG_CAP}), \
             not the ~4.7 MB the ~7 MB estimate is built on"
        );
    }

    #[test]
    fn a_key_changed_many_times_replays_once() {
        // The replay materialises CURRENT state, so every record naming one
        // key produces the same update — and a session whose transcript is
        // re-recorded as it grows logs one record per turn per message. Left
        // undeduped the catch-up shipped one update per RECORD, which on a
        // small session with a long history is far more work than the reload
        // it is supposed to replace.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        for i in 0..50 {
            store
                .merge_events(&scope, None, vec![ev("m1", &format!("turn {i}"))])
                .unwrap();
            store
                .merge_events(&scope, None, vec![ev("m2", &format!("turn {i}"))])
                .unwrap();
        }
        assert_eq!(
            store.replay_sequences(&scope).len(),
            100,
            "each real change is retained"
        );

        let replayed = store.updates_since(&scope, 0).unwrap();
        assert_eq!(
            replayed.len(),
            2,
            "expected one update per distinct key, got {}",
            replayed.len()
        );
        // Deduping to the LAST record keeps the answer current.
        let texts: Vec<_> = replayed
            .iter()
            .map(|u| match u {
                GraphUpdate::NodeCreated { event, .. } => event.text.clone().unwrap(),
                other => panic!("unexpected update {other:?}"),
            })
            .collect();
        assert_eq!(texts, vec!["turn 49".to_string(), "turn 49".to_string()]);
        // And the state it lands on is the state a reload would have loaded.
        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn a_cursor_ahead_of_the_scope_cannot_be_served_incrementally() {
        // A cursor only ever moves forward, so answering "nothing changed" to
        // one that is ahead of the shard pins it there for good: every
        // sequence the shard hands out next is below it, and the subscriber
        // stays permanently short of state whose fences clear instantly.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let events: Vec<Event> = (0..5).map(|i| ev(&format!("m{i}"), "x")).collect();
        store.merge_events(&scope, None, events).unwrap();
        assert_eq!(store.session_sequence(&scope), 5);
        let ahead = 6;

        assert!(
            store.updates_since(&scope, ahead).is_none(),
            "a cursor ahead of the shard was told it was up to date"
        );
    }

    #[test]
    fn a_cursor_from_a_dropped_incarnation_of_a_scope_is_not_served() {
        // Dropping a scope must preserve its sequence as a tombstone and
        // consume a sequence number. A cursor from the old incarnation then
        // encounters a gap instead of silently consuming a replacement shard.
        // Test both lagging cursors and a cursor exactly at the pre-drop head.
        let store = GraphStore::new(None).unwrap();
        let scope = scope_for("s1");
        let events: Vec<Event> = (0..5).map(|i| ev(&format!("old{i}"), "x")).collect();
        store.merge_events(&scope, None, events).unwrap();
        let stale_cursor = 3;
        let level_cursor = store.session_sequence(&scope);

        store.drop_session(&scope).unwrap();
        for (what, cursor) in [("a mid-window", stale_cursor), ("a level", level_cursor)] {
            assert!(
                store.updates_since(&scope, cursor).is_none(),
                "{what} cursor from the dropped incarnation was served across the drop"
            );
        }

        let events: Vec<Event> = (0..10).map(|i| ev(&format!("new{i}"), "x")).collect();
        store.merge_events(&scope, None, events).unwrap();
        assert!(
            store.session_sequence(&scope) > 5,
            "the re-created shard re-used sequences the dropped one had spent"
        );
        for (what, cursor) in [("a mid-window", stale_cursor), ("a level", level_cursor)] {
            assert!(
                store.updates_since(&scope, cursor).is_none(),
                "{what} cursor from the dropped incarnation was served the new \
                 shard's tail"
            );
        }
    }

    #[test]
    fn concurrent_writers_to_one_scope_produce_a_gapless_replay_log() {
        use std::sync::Arc as StdArc;
        let store = StdArc::new(GraphStore::new(None).unwrap());
        let scope = scope_for("s1");
        let mut rx = store.subscribe();

        let mut handles = vec![];
        for w in 0..2 {
            let store = StdArc::clone(&store);
            let scope = scope.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..200 {
                    store
                        .merge_events(&scope, None, vec![ev(&format!("w{w}-m{i}"), "x")])
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let seqs = store.replay_sequences(&scope);
        assert_eq!(seqs.len(), 400);
        for (i, s) in seqs.iter().enumerate() {
            assert_eq!(
                *s,
                i as i64 + 1,
                "replay sequences are not strictly increasing without gaps"
            );
        }

        // The log's order is the broadcast's order: same ids, same positions.
        let broadcast_ids: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|u| match u {
                GraphUpdate::NodeCreated { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        let replay_ids: Vec<String> = store
            .updates_since(&scope, 0)
            .unwrap()
            .into_iter()
            .filter_map(|u| match u {
                GraphUpdate::NodeCreated { id, .. } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(broadcast_ids.len(), 400, "the ring dropped a send");
        assert_eq!(replay_ids, broadcast_ids);
    }

    #[test]
    fn backward_slice_proto() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(
                &global_scope(),
                None,
                vec![ev("a", "1"), ev("b", "2"), ev("c", "3")],
            )
            .unwrap();
        store
            .merge_dependencies(&global_scope(), None, vec![dep("a", "b"), dep("b", "c")])
            .unwrap();

        let (nodes, edges) = store.backward_slice(T, "c", None, None).unwrap();
        assert_eq!(nodes.len(), 3);
        let pairs: HashSet<(String, String)> = edges
            .iter()
            .map(|e| (e.source.clone(), e.destination.clone()))
            .collect();
        assert!(pairs.contains(&("a".to_string(), "b".to_string())));
        assert!(pairs.contains(&("b".to_string(), "c".to_string())));
    }

    #[test]
    fn backward_slice_depth() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(
                &global_scope(),
                None,
                vec![ev("a", "1"), ev("b", "2"), ev("c", "3")],
            )
            .unwrap();
        store
            .merge_dependencies(&global_scope(), None, vec![dep("a", "b"), dep("b", "c")])
            .unwrap();

        let (nodes, _) = store.backward_slice(T, "c", None, Some(1)).unwrap();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn forward_slice_proto() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("a", "1"), ev("b", "2")])
            .unwrap();
        store
            .merge_dependencies(&global_scope(), None, vec![dep("a", "b")])
            .unwrap();

        let (nodes, _) = store.forward_slice(T, "a", None, None).unwrap();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn computation_trace() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_computations(
                &global_scope(),
                None,
                vec![span("s1", "t1"), span("s2", "t1"), span("s3", "t2")],
            )
            .unwrap();

        let trace = store.get_trace(T, "t1", None, None, None).unwrap();
        assert_eq!(trace.computations.len(), 2);
    }

    #[test]
    fn get_span_by_id() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_computations(&global_scope(), None, vec![span("s1", "t1")])
            .unwrap();

        let c = store.get_span(T, "s1", None).unwrap();
        assert!(c.is_some());
        assert_eq!(c.unwrap().name, "test");
        assert!(store.get_span(T, "nope", None).unwrap().is_none());
    }

    #[test]
    fn computation_child_of() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_computations(&global_scope(), None, vec![span("parent", "t1")])
            .unwrap();

        let mut child = span("child", "t1");
        child.parent_span_id = Some("parent".to_string());
        store
            .merge_computations(&global_scope(), None, vec![child])
            .unwrap();

        let trace = store.get_trace(T, "t1", None, None, None).unwrap();
        assert_eq!(trace.computation_edges.len(), 1);
    }

    #[test]
    fn broadcast_updates() {
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();

        store
            .merge_events(&global_scope(), None, vec![ev("m1", "hi")])
            .unwrap();

        let upd = rx.try_recv().unwrap();
        assert!(matches!(upd, GraphUpdate::NodeCreated { .. }));
    }

    #[test]
    fn rerecord_broadcast_keeps_merged_provenance() {
        // A tool-result node is recorded once with `derived_from`, then
        // re-recorded without it (what an SDK does when it replays the
        // conversation history each turn). Both the stored node and the
        // broadcast update must still carry the provenance, or every
        // policy rule that joins on `ToolResult` goes silently empty.
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();

        let tool = Tool {
            name: Some("get_reservation_details".to_string()),
            arguments: Some("{\"reservation_id\":\"ABC123\"}".to_string()),
        };
        let mut with_provenance = ev("m1", "reservation payload");
        with_provenance.derived_from = Some(tool.clone());
        with_provenance.entity = Some("catalog-reader".to_string());
        store
            .merge_events(&global_scope(), None, vec![with_provenance])
            .unwrap();

        // Re-record the same id with no derived_from, role, or entity. The
        // text differs so the re-record is a real change — an identical
        // re-record is idempotent and broadcasts nothing at all, which is a
        // different behaviour, pinned by
        // `a_re_ingest_of_identical_events_emits_nothing`.
        let mut bare = ev("m1", "reservation payload (refreshed)");
        bare.role = None;
        bare.entity = None;
        store
            .merge_events(&global_scope(), None, vec![bare])
            .unwrap();

        let mut m1_events = vec![];
        while let Ok(upd) = rx.try_recv() {
            if let GraphUpdate::NodeCreated { id, event, .. } = upd {
                if id == "m1" {
                    m1_events.push(event);
                }
            }
        }
        // Assert on the re-record's own update, not on whatever the first
        // write broadcast — the first already carries everything, so
        // reading it would let this test pass even if the re-record
        // stopped broadcasting at all.
        assert_eq!(
            m1_events.len(),
            2,
            "expected one NodeCreated per write to m1"
        );
        let broadcast = m1_events.pop().unwrap();
        assert_eq!(
            broadcast.derived_from.as_ref(),
            Some(&tool),
            "re-record broadcast dropped or altered derived_from"
        );
        assert_eq!(
            broadcast.role,
            Some(Role::User as i32),
            "re-record broadcast dropped a field the store merged"
        );
        assert_eq!(
            broadcast.entity.as_deref(),
            Some("catalog-reader"),
            "re-record dropped entity, which policies match on"
        );

        let (events, _, _) = store.get_full_state().unwrap();
        let stored = events
            .iter()
            .find(|e| e.id.as_deref() == Some("m1"))
            .unwrap();
        assert!(stored.derived_from.is_some(), "store dropped derived_from");
    }

    #[test]
    fn concurrent_access() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let mut handles = vec![];

        for i in 0..10 {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let id = format!("m{i}");
                s.merge_events(&global_scope(), None, vec![ev(&id, "data")])
                    .unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 10);
    }

    #[test]
    fn skip_empty_events() {
        let store = GraphStore::new(None).unwrap();
        let ids = store
            .merge_events(
                &global_scope(),
                None,
                vec![Event {
                    text: None,
                    agent: None,
                    role: None,
                    id: Some("empty".to_string()),
                    tools: vec![],
                    derived_from: None,
                    principal: None,
                    entity: None,
                    metadata: None,
                }],
            )
            .unwrap();
        assert_eq!(ids[0], "");

        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn auto_generate_id() {
        let store = GraphStore::new(None).unwrap();
        let ids = store
            .merge_events(
                &global_scope(),
                None,
                vec![Event {
                    text: Some("no-id".to_string()),
                    agent: None,
                    role: None,
                    id: None,
                    tools: vec![],
                    derived_from: None,
                    principal: None,
                    entity: None,
                    metadata: None,
                }],
            )
            .unwrap();
        assert!(!ids[0].is_empty());

        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn edge_broadcast_includes_metadata() {
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();

        store
            .merge_events(
                &global_scope(),
                None,
                vec![ev("a", "first"), ev("b", "second")],
            )
            .unwrap();
        while let Ok(u) = rx.try_recv() {
            if matches!(u, GraphUpdate::EdgeCreated { .. }) {
                panic!("unexpected EdgeCreated before merge_dependencies");
            }
        }

        store
            .merge_dependencies(
                &global_scope(),
                None,
                vec![Edge {
                    source: "a".to_string(),
                    destination: "b".to_string(),
                    message_index: Some(7),
                    proximal: Some(true),
                    principal: None,
                    entity: None,
                }],
            )
            .unwrap();

        let mut upd = rx.try_recv().unwrap();
        while !matches!(upd, GraphUpdate::EdgeCreated { .. }) {
            upd = rx.try_recv().unwrap();
        }
        match upd {
            GraphUpdate::EdgeCreated {
                source,
                destination,
                kind,
                message_index,
                proximal,
                scope: _,
                principal: _,
                entity: _,
            } => {
                assert_eq!(source, "a");
                assert_eq!(destination, "b");
                assert_eq!(kind, crate::types::EdgeKindSer::DependsOn);
                assert_eq!(message_index, Some(7));
                assert_eq!(proximal, Some(true));
            }
            other => panic!("expected EdgeCreated, got {other:?}"),
        }
    }

    /// An edge's principal is the identity of the request that
    /// recorded it, and the store keeps one row per edge: the newest
    /// recording replaces the principal, and a client cannot choose
    /// it. Both the broadcast and the stored state say the same
    /// thing. This is the model the policy language documents —
    /// a policy that needs every principal that ever asserted an
    /// edge needs a relation of its own.
    #[test]
    fn an_edge_principal_is_the_recording_request_and_replaces_the_stored_one() {
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();

        store
            .merge_events(
                &global_scope(),
                None,
                vec![ev("a", "first"), ev("b", "second")],
            )
            .unwrap();

        // The request authenticated as "gateway"; the client sent
        // "admin" in the edge. The request wins.
        let record = |request_principal: Option<&str>, sent: Option<&str>| {
            store
                .merge_dependencies(
                    &global_scope(),
                    request_principal,
                    vec![Edge {
                        source: "a".to_string(),
                        destination: "b".to_string(),
                        message_index: Some(7),
                        proximal: Some(true),
                        principal: sent.map(str::to_string),
                        entity: None,
                    }],
                )
                .unwrap();
        };
        let announced_principal = |rx: &mut tokio::sync::broadcast::Receiver<GraphUpdate>| loop {
            match rx.try_recv().expect("an EdgeCreated must be announced") {
                GraphUpdate::EdgeCreated { principal, .. } => return principal,
                _ => continue,
            }
        };
        let stored_principal = || {
            let (_, edges, _) = store.get_full_state().unwrap();
            let edge = edges
                .iter()
                .find(|e| e.source == "a" && e.destination == "b")
                .expect("the edge is stored");
            edge.principal.clone()
        };

        record(Some("gateway"), Some("admin"));
        assert_eq!(
            announced_principal(&mut rx).as_deref(),
            Some("gateway"),
            "the broadcast names the recording request's principal, not the \
             one the client sent"
        );
        assert_eq!(
            stored_principal().as_deref(),
            Some("gateway"),
            "and so does the stored state"
        );

        // A later recording by a different caller replaces it.
        record(Some("intern"), None);
        assert_eq!(
            announced_principal(&mut rx).as_deref(),
            Some("intern"),
            "the newest recording's principal is the edge's principal"
        );
        assert_eq!(stored_principal().as_deref(), Some("intern"));

        // Including a recording with no auth-derived identity at all:
        // the edge is then attributed to nobody.
        record(None, Some("admin"));
        assert_eq!(
            announced_principal(&mut rx).as_deref(),
            None,
            "a recording request with no principal leaves the edge with none"
        );
        assert_eq!(stored_principal(), None);
    }

    #[test]
    fn drop_session_isolates_partitions() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(
                &scope_for("A"),
                None,
                vec![ev("a1", "from-A"), ev("a2", "from-A")],
            )
            .unwrap();
        store
            .merge_events(&scope_for("B"), None, vec![ev("b1", "from-B")])
            .unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("g1", "global")])
            .unwrap();

        let scope_a = SessionScope::new(T, "A");
        let scope_b = SessionScope::new(T, "B");
        let sessions: HashSet<_> = store.list_sessions().into_iter().collect();
        assert!(sessions.contains(&scope_a));
        assert!(sessions.contains(&scope_b));
        assert!(!sessions.contains(&SessionScope::global(T)));

        let a_msgs: HashSet<_> = store.events_for_session(&scope_a).into_iter().collect();
        assert_eq!(a_msgs.len(), 2);
        assert!(a_msgs.contains("a1"));
        assert!(a_msgs.contains("a2"));

        let mut rx = store.subscribe();

        store.drop_session(&scope_a).unwrap();

        let (events, _, _) = store.get_full_state().unwrap();
        let surviving_ids: HashSet<_> = events.iter().filter_map(|e| e.id.clone()).collect();
        assert!(!surviving_ids.contains("a1"));
        assert!(!surviving_ids.contains("a2"));
        assert!(surviving_ids.contains("b1"));
        assert!(surviving_ids.contains("g1"));

        assert!(!store.list_sessions().contains(&scope_a));
        assert!(store.events_for_session(&scope_a).is_empty());
        assert!(store.list_sessions().contains(&scope_b));
        assert_eq!(store.events_for_session(&scope_b).len(), 1);

        let mut saw_drop = false;
        while let Ok(u) = rx.try_recv() {
            if matches!(&u, GraphUpdate::DropSession(s) if *s == scope_a) {
                saw_drop = true;
            }
        }
        assert!(saw_drop, "expected DropSession({scope_a}) broadcast");
    }

    #[test]
    fn session_counts_track_nodes_and_edges() {
        let store = GraphStore::new(None).unwrap();
        let scope_a = scope_for("A");
        let scope_b = scope_for("B");
        let scope_unknown = SessionScope::new(T, "nope");

        store
            .merge_events(&scope_a, None, vec![ev("a1", "first"), ev("a2", "second")])
            .unwrap();
        store
            .merge_dependencies(&scope_a, None, vec![dep("a1", "a2")])
            .unwrap();

        let (nodes, edges) = store.session_counts(&scope_a);
        assert_eq!(nodes, 2);
        assert_eq!(edges, 1);

        store
            .merge_events(&scope_b, None, vec![ev("b1", "third")])
            .unwrap();

        let (nb, eb) = store.session_counts(&scope_b);
        assert_eq!(nb, 1);
        assert_eq!(eb, 0);

        assert_eq!(store.session_counts(&scope_unknown), (0, 0));

        store.drop_session(&scope_a).unwrap();
        assert_eq!(store.session_counts(&scope_a), (0, 0));
        assert_eq!(store.session_counts(&scope_b), (1, 0));
    }

    #[test]
    fn get_session_state_isolates_by_partition() {
        let store = GraphStore::new(None).unwrap();

        let scope_a = SessionScope::new(T, "A");
        let scope_b = SessionScope::new(T, "B");

        store
            .merge_events(&scope_a, None, vec![ev("a1", "first"), ev("a2", "second")])
            .unwrap();
        store
            .merge_events(&scope_b, None, vec![ev("b1", "other")])
            .unwrap();

        store
            .merge_dependencies(&scope_a, None, vec![dep("a1", "a2")])
            .unwrap();
        let (events, edges, seq) = store.get_session_state(&scope_a).unwrap();
        let event_ids: HashSet<_> = events.iter().filter_map(|e| e.id.clone()).collect();
        assert_eq!(events.len(), 2);
        assert!(event_ids.contains("a1"));
        assert!(event_ids.contains("a2"));
        assert_eq!(edges.len(), 1);
        // Preserve dependency direction through persistence and bootstrap,
        // matching backward_slice and the live EdgeCreated broadcast.
        assert_eq!(
            (edges[0].source.as_str(), edges[0].destination.as_str()),
            ("a1", "a2"),
        );
        // Session A's local seq: 2 events + 1 edge = 3
        assert_eq!(seq, 3);

        let (events, edges, seq_b) = store.get_session_state(&scope_b).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_deref(), Some("b1"));
        assert!(edges.is_empty());
        // Session B's local seq: 1 event = 1, independent of A.
        assert_eq!(seq_b, 1);

        // Unknown session: empty result, seq = 0.
        let scope_unknown = SessionScope::new(T, "nope");
        let (events, edges, seq_unknown) = store.get_session_state(&scope_unknown).unwrap();
        assert!(events.is_empty());
        assert!(edges.is_empty());
        assert_eq!(seq_unknown, 0);
    }

    #[test]
    fn durability_round_trip() {
        // Writes are durable and read back regardless of concurrency: the commit
        // always precedes the broadcast.
        let store = GraphStore::new(None).unwrap();

        let a = ev("a", "first");
        let b = ev("b", "second");
        let e_ab = dep("a", "b");

        let ids = store
            .merge_events_with_dependencies(&global_scope(), None, vec![a, b], vec![e_ab])
            .unwrap();
        assert_eq!(ids, vec!["a", "b"]);

        let (events, edges, seq) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(seq >= 3);
    }

    #[test]
    fn batched_writes_emit_one_sequence_marker() {
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();

        store
            .merge_events_with_dependencies(
                &global_scope(),
                None,
                vec![ev("a", "1"), ev("b", "2"), ev("c", "3")],
                vec![dep("a", "b"), dep("b", "c")],
            )
            .unwrap();

        let mut sequence_count = 0;
        for _ in 0..30 {
            match rx.try_recv() {
                Ok(GraphUpdate::Sequence(_)) => sequence_count += 1,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(
            sequence_count, 1,
            "batched merge should fire exactly one Sequence marker"
        );
    }

    #[test]
    fn drop_session_unknown_is_noop() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(&global_scope(), None, vec![ev("a", "one")])
            .unwrap();
        store
            .drop_session(&SessionScope::new(T, "nonexistent"))
            .unwrap();
        store.drop_session(&SessionScope::global(T)).unwrap();
        let (events, _, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn merge_events_with_dependencies_atomic() {
        let store = GraphStore::new(None).unwrap();
        let ids = store
            .merge_events_with_dependencies(
                &global_scope(),
                None,
                vec![ev("x", "first"), ev("y", "second"), ev("z", "third")],
                vec![dep("x", "y"), dep("y", "z")],
            )
            .unwrap();
        assert_eq!(ids, vec!["x", "y", "z"]);

        let (events, edges, seq) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(edges.len(), 2);
        assert!(seq >= 3);

        let (nodes, _) = store.backward_slice(T, "z", None, None).unwrap();
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn atomic_broadcast_nodes_and_edges() {
        use std::time::Duration;

        let store = Arc::new(GraphStore::new(None).unwrap());
        let mut rx = store.subscribe();

        let store2 = Arc::clone(&store);
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            store2
                .merge_events_with_dependencies(
                    &global_scope(),
                    None,
                    vec![ev("a", "one"), ev("b", "two")],
                    vec![dep("a", "b")],
                )
                .unwrap();
        });

        let mut updates = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Ok(u) => updates.push(u),
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
            if updates.len() >= 4 {
                break;
            }
        }
        writer.join().unwrap();

        let data_updates: Vec<_> = updates
            .iter()
            .filter(|u| {
                !matches!(
                    u,
                    GraphUpdate::Sequence(_) | GraphUpdate::SessionSequence { .. }
                )
            })
            .collect();
        assert_eq!(data_updates.len(), 3);
        let node_count = data_updates
            .iter()
            .take_while(|u| matches!(u, GraphUpdate::NodeCreated { .. }))
            .count();
        assert_eq!(node_count, 2, "expected 2 NodeCreated before EdgeCreated");
        assert!(matches!(data_updates[2], GraphUpdate::EdgeCreated { .. }));

        let (events, edges, _) = store.get_full_state().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(edges.len(), 1);
    }

    /// Two sessions writing concurrently must not serialize on a
    /// single global lock — each acquires its own shard's lock.
    /// Smoke check: spawn N threads each writing to a distinct
    /// session and verify all writes land.
    #[test]
    fn cross_session_writes_do_not_serialize() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let mut handles = vec![];
        for i in 0..16 {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let scope = SessionScope::new(T, format!("s{i}"));
                let e = ev(&format!("m{i}"), "data");
                s.merge_events(&scope, None, vec![e]).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let sessions = store.list_sessions();
        assert_eq!(sessions.len(), 16);
        for i in 0..16 {
            let sid = format!("s{i}");
            let (n, _) = store.session_counts(&SessionScope::new(T, sid.clone()));
            assert_eq!(n, 1, "session {sid} should have 1 message");
        }
    }

    /// A backward_slice for a node id must work even right after a
    /// concurrent merge_events finishes on a different thread —
    /// the reverse index update is folded into the shard write
    /// critical section so visibility is consistent.
    #[test]
    fn reverse_index_consistent_with_broadcast() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let mut rx = store.subscribe();

        let store2 = Arc::clone(&store);
        let writer = std::thread::spawn(move || {
            let e = ev("solo", "x");
            store2.merge_events(&global_scope(), None, vec![e]).unwrap();
        });
        writer.join().unwrap();

        // Drain to find the NodeCreated for "solo".
        let mut saw = false;
        while let Ok(u) = rx.try_recv() {
            if let GraphUpdate::NodeCreated { id, .. } = u {
                if id == "solo" {
                    saw = true;
                    break;
                }
            }
        }
        assert!(saw);
        // Now backward_slice must succeed — reverse index has been
        // populated before the broadcast was emitted.
        let (nodes, _) = store.backward_slice(T, "solo", None, None).unwrap();
        assert_eq!(nodes.len(), 1);
    }

    /// Per-tenant scopes are fully isolated: events, message lists,
    /// and drops are scoped to a single `(tenant, session)` and never
    /// leak across tenants. Even with the same session string, two
    /// tenants keep distinct shards.
    #[test]
    fn cross_tenant_isolation_does_not_leak() {
        let store = GraphStore::new(None).unwrap();

        let scope_acme = SessionScope::new("acme", "s1");
        let scope_orgb = SessionScope::new("orgb", "s1");

        store
            .merge_events(&scope_acme, None, vec![ev("a1", "from-acme")])
            .unwrap();
        store
            .merge_events(&scope_orgb, None, vec![ev("o1", "from-orgb")])
            .unwrap();

        // Each tenant sees only its own ids.
        let acme_msgs: HashSet<_> = store.events_for_session(&scope_acme).into_iter().collect();
        let orgb_msgs: HashSet<_> = store.events_for_session(&scope_orgb).into_iter().collect();
        assert_eq!(acme_msgs.len(), 1);
        assert!(acme_msgs.contains("a1"));
        assert!(!acme_msgs.contains("o1"));
        assert_eq!(orgb_msgs.len(), 1);
        assert!(orgb_msgs.contains("o1"));
        assert!(!orgb_msgs.contains("a1"));

        // Dropping acme/s1 leaves orgb/s1 untouched.
        store.drop_session(&scope_acme).unwrap();
        assert!(store.events_for_session(&scope_acme).is_empty());
        let orgb_after: HashSet<_> = store.events_for_session(&scope_orgb).into_iter().collect();
        assert_eq!(orgb_after.len(), 1);
        assert!(orgb_after.contains("o1"));

        // Cross-tenant list_sessions still surfaces orgb/s1 alone.
        let sessions: HashSet<_> = store.list_sessions().into_iter().collect();
        assert!(!sessions.contains(&scope_acme));
        assert!(sessions.contains(&scope_orgb));
    }

    /// `get_full_state_for_tenant_scoped` returns every session under
    /// the tenant, each event/edge paired with its OWN scope (not
    /// flattened), excludes other tenants, and reports the tenant's
    /// GLOBAL SHARD counter — deliberately not the store-wide
    /// `get_sequence()`, which also counts other tenants' and sessions'
    /// writes. This is what lets a global-scope evaluator cold-bootstrap
    /// the same tenant-wide view its live stream builds (c5), and the
    /// assertions below pin both halves of that.
    #[test]
    fn tenant_scoped_full_state_preserves_attribution() {
        let store = GraphStore::new(None).unwrap();
        let acme_s1 = SessionScope::new("acme", "s1");
        let acme_s2 = SessionScope::new("acme", "s2");
        let acme_global = SessionScope::global("acme");
        let orgb_s1 = SessionScope::new("orgb", "s1");

        store
            .merge_events(&acme_s1, None, vec![ev("a1", "one")])
            .unwrap();
        store
            .merge_events(&acme_s2, None, vec![ev("a2", "two")])
            .unwrap();
        store
            .merge_events(&acme_global, None, vec![ev("g1", "global")])
            .unwrap();
        store
            .merge_events(&orgb_s1, None, vec![ev("o1", "from-orgb")])
            .unwrap();

        let (events, _edges, seq) = store.get_full_state_for_tenant_scoped("acme").unwrap();

        // Exactly acme's three events, each under its own session;
        // orgb's event is excluded.
        let by_id: HashMap<String, String> = events
            .iter()
            .map(|(scope, ev)| {
                (
                    ev.id.clone().unwrap_or_default(),
                    scope.session().to_string(),
                )
            })
            .collect();
        assert_eq!(by_id.len(), 3, "expected acme's 3 events, got {by_id:?}");
        assert_eq!(by_id.get("a1").map(String::as_str), Some("s1"));
        assert_eq!(by_id.get("a2").map(String::as_str), Some("s2"));
        // The global session's id is the empty string.
        assert_eq!(by_id.get("g1").map(String::as_str), Some(""));
        assert!(!by_id.contains_key("o1"), "orgb event must not leak");

        // Sequence is the tenant's GLOBAL SHARD counter — the resync trigger a
        // global evaluator compares against. Deliberately NOT the store-wide
        // `get_sequence()`: that is bumped by every session of every tenant (orgb's
        // write above moves it), so a global evaluator gated on it would reload the
        // whole tenant on unrelated traffic — no cost bound on a path the refmon
        // takes per request.
        assert_eq!(seq, store.session_sequence(&acme_global));
        assert!(
            store.get_sequence() > seq,
            "store-wide counter also counts other sessions/tenants"
        );
    }

    /// Two tenants write nodes with the same client-supplied id.
    /// Keyed on id alone, `node_to_session` would let the later
    /// writer clobber the earlier, `scope_for_node` would return the
    /// wrong scope, and a `merge_dependencies` call could be routed
    /// into the wrong tenant's shard. With the `(tenant, id)` key
    /// both tenants see their own node and nothing leaks.
    #[test]
    fn colliding_node_ids_do_not_cross_tenants() {
        let store = GraphStore::new(None).unwrap();

        // Both tenants write a node with id "shared". Same session
        // string "s" in each tenant — distinct shards via the
        // tenant component of the scope key.
        let acme_s = SessionScope::new("acme", "s");
        let orgb_s = SessionScope::new("orgb", "s");
        store
            .merge_events(&acme_s, None, vec![ev("shared", "from-acme")])
            .unwrap();
        store
            .merge_events(&orgb_s, None, vec![ev("shared", "from-orgb")])
            .unwrap();

        // Each tenant resolves its own scope; neither sees the
        // other's "shared".
        let acme = store
            .scope_for_node("acme", "shared")
            .expect("acme owns shared");
        let orgb = store
            .scope_for_node("orgb", "shared")
            .expect("orgb owns shared");
        assert_eq!(acme.tenant(), "acme");
        assert_eq!(orgb.tenant(), "orgb");
        assert!(store.scope_for_node("acme", "absent").is_none());

        // An edge from one tenant must not be routed into the
        // other tenant's shard via the reverse index. The sibling
        // node and the edge land in `orgb/s` — the same shard as
        // the orgb "shared" endpoint.
        store
            .merge_events(&orgb_s, None, vec![ev("o-target", "x")])
            .unwrap();
        store
            .merge_dependencies(&orgb_s, None, vec![dep("shared", "o-target")])
            .unwrap();

        // The orgb edge lives in orgb's shard (resolved by id under
        // tenant "orgb"), regardless of acme also having a node
        // called "shared".
        let (_, edges) = store
            .backward_slice("orgb", "o-target", None, None)
            .unwrap();
        assert_eq!(edges.len(), 1);

        // The acme tenant's view is untouched.
        let acme_msgs: HashSet<_> = store.events_for_session(&acme_s).into_iter().collect();
        assert!(acme_msgs.contains("shared"));
    }

    /// `drop_session` must clear persisted edges for the dropped
    /// scope. Removing only messages and computations would leave
    /// edge rows on disk, inflating `all_edges()` scans on every
    /// restart.
    #[test]
    fn drop_session_clears_persisted_edges() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rocks").to_string_lossy().into_owned();
        let scope = SessionScope::new(T, "s1");

        // Run 1: write a session with two events + an edge, then
        // drop the session. At reopen the edge must be gone.
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            store
                .merge_events(&scope, None, vec![ev("a", "first"), ev("b", "second")])
                .unwrap();
            store
                .merge_dependencies(&scope, None, vec![dep("a", "b")])
                .unwrap();
            store.drop_session(&scope).unwrap();
        }
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            // Reopen reads the edges CF; orphaned edges from a
            // dropped session would show up here.
            let edges = store.rocks_for_test().all_edges().unwrap();
            let stale = edges
                .iter()
                .filter(|(k, _)| k.source == "a" || k.destination == "b")
                .count();
            assert_eq!(stale, 0, "drop_session left orphan edges in RocksDB");
        }
    }

    /// A shard that diverged from durable storage must not answer.
    ///
    /// This test exercises all read gates independently of the writer. The
    /// snapshot tests additionally inject a real batch-commit failure to
    /// verify that the write path sets the quarantine flag.
    #[test]
    fn a_diverged_shard_refuses_to_serve_its_graph() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");
        store
            .merge_events(&scope, None, vec![ev("a", "first"), ev("b", "second")])
            .unwrap();
        store
            .merge_dependencies(&scope, None, vec![dep("a", "b")])
            .unwrap();

        // Healthy: it answers.
        assert!(store.get_session_state(&scope).is_ok());
        assert!(store.backward_slice(T, "b", Some("s1"), Some(5)).is_ok());

        store.mark_diverged_for_test(&scope);

        let err = store
            .get_session_state(&scope)
            .expect_err("a diverged shard must not serve state");
        assert!(format!("{err}").contains("diverged"), "got: {err}");
        assert!(
            store.backward_slice(T, "b", Some("s1"), Some(5)).is_err(),
            "a diverged shard must not serve a slice either"
        );
        // The tenant-wide reads matter most: they are how a global-scope
        // evaluator bootstraps, so a gap here lets one diverged session reach
        // a decision through the tenant view.
        assert!(
            store.get_full_state_for_tenant(T).is_err(),
            "the tenant-wide read must refuse while one of its sessions is diverged"
        );
        assert!(
            store.get_full_state_for_tenant_scoped(T).is_err(),
            "the global evaluator's bootstrap read must refuse too"
        );
    }

    /// A diverged shard refuses WRITES too — on every entry point.
    ///
    /// The read guard quarantines what the session can influence; the write
    /// guard stops the quarantine from deepening. Computations were the gap:
    /// they take a separate path from events and edges, and a span written
    /// into a diverged shard links durably to a message that exists only in
    /// memory, so the loader drops the dangling edge at the next restart —
    /// a write reported as durable that does not survive one.
    #[test]
    fn a_diverged_shard_refuses_every_write_path() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");
        store
            .merge_events(&scope, None, vec![ev("a", "first")])
            .unwrap();

        // Healthy: computations land.
        store
            .merge_computations(&scope, None, vec![span("sp0", "tr0")])
            .expect("a healthy shard takes computations");

        store.mark_diverged_for_test(&scope);

        for (what, err) in [
            (
                "events",
                store
                    .merge_events(&scope, None, vec![ev("b", "second")])
                    .err(),
            ),
            (
                "dependencies",
                store
                    .merge_dependencies(&scope, None, vec![dep("a", "b")])
                    .err(),
            ),
            (
                "events with dependencies",
                store
                    .merge_events_with_dependencies(&scope, None, vec![ev("c", "third")], vec![])
                    .err(),
            ),
            (
                "computations",
                store
                    .merge_computations(&scope, None, vec![span("sp1", "tr1")])
                    .err(),
            ),
        ] {
            let err = err.unwrap_or_else(|| panic!("a diverged shard accepted {what}"));
            assert!(
                format!("{err}").contains("diverged"),
                "the {what} refusal must say why: {err}"
            );
        }
    }

    /// Dependency edges that cannot be attached are an error, not a silent
    /// drop reported as success.
    ///
    /// A `DependsOn` edge is a taint link, so a dropped one severs provenance
    /// and a rule that denies on tainted ancestry stops matching — while the
    /// caller was told the write succeeded and nothing retries.
    #[test]
    fn unattachable_dependency_edges_are_reported_not_dropped() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        // Nothing recorded at all: no edge in the batch can resolve.
        let err = store
            .merge_dependencies(&scope, None, vec![dep("a", "b")])
            .expect_err("a batch against an unknown scope must be reported");
        assert!(
            format!("{err}").contains("no recorded events"),
            "unexpected error: {err}"
        );

        // One endpoint recorded, the other not.
        store
            .merge_events(&scope, None, vec![ev("a", "first")])
            .unwrap();
        let err = store
            .merge_dependencies(&scope, None, vec![dep("a", "missing")])
            .expect_err("an edge naming an unrecorded node must be reported");
        assert!(
            format!("{err}").contains("a -> missing"),
            "the error must name the offending edge: {err}"
        );

        // And the rejection is all-or-nothing: nothing from that batch landed.
        assert_eq!(
            store.session_counts(&scope).1,
            0,
            "a rejected batch must leave no edges behind"
        );

        // With both endpoints present it succeeds.
        store
            .merge_events(&scope, None, vec![ev("b", "second")])
            .unwrap();
        store
            .merge_dependencies(&scope, None, vec![dep("a", "b")])
            .expect("a resolvable edge must be recorded");
        assert_eq!(store.session_counts(&scope).1, 1);
    }

    /// An edge with an empty endpoint must be rejected cleanly, and must not
    /// quarantine a shard nothing touched.
    ///
    /// The sibling of the skipped-event case: the combined call's pre-check
    /// filtered empty-endpoint edges OUT of its unresolved set while the inner
    /// pre-pass rejects them, so such an edge passed the first check, failed
    /// the second, and took the fallback — marking a pristine shard diverged
    /// and, through the tenant-wide reads, taking the tenant's session-less
    /// enforcement down with it. Two guards deciding the same question
    /// differently.
    #[test]
    fn an_empty_endpoint_edge_does_not_quarantine_an_untouched_shard() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        let err = store
            .merge_events_with_dependencies(&scope, None, vec![], vec![dep("a", "")])
            .expect_err("an edge with an empty endpoint must be rejected");
        assert!(format!("{err}").contains("empty endpoint"), "got: {err}");

        // With events in the batch, the pre-check has to catch it BEFORE they
        // are applied. This is what pins the pre-check specifically: the
        // fallback's mutation test would spare the session either way, but the
        // events would already be sitting in memory uncommitted.
        store
            .merge_events_with_dependencies(&scope, None, vec![ev("a", "hi")], vec![dep("a", "")])
            .expect_err("still rejected when the batch carries events");
        assert_eq!(
            store.session_counts(&scope),
            (0, 0),
            "the events must not have been applied before the edge was checked"
        );

        assert!(
            store.get_session_state(&scope).is_ok(),
            "a shard that was never mutated must not be quarantined"
        );
        assert!(
            store.get_full_state_for_tenant_scoped(T).is_ok(),
            "and the tenant's global bootstrap must survive it"
        );
    }

    /// An edge naming an event the batch will SKIP must be rejected cleanly,
    /// not quarantine the session.
    ///
    /// `upsert_event_into_shard` drops an event with nothing to store, so a
    /// pre-check that assumed every id in the batch would land let such an
    /// edge through; the other events were then applied, the inner pre-pass
    /// rejected, and the fallback marked the shard diverged. Because a
    /// diverged session makes the tenant-wide reads refuse, and those are how
    /// a global evaluator bootstraps, one empty event plus one edge from any
    /// writer took the whole tenant's session-less enforcement out of service
    /// until a restart.
    #[test]
    fn an_edge_to_a_skipped_event_does_not_quarantine_the_session() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        let empty = Event {
            text: None,
            agent: None,
            role: None,
            id: Some("e1".into()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: None,
            metadata: None,
        };
        let err = store
            .merge_events_with_dependencies(
                &scope,
                None,
                vec![empty, ev("a", "hi")],
                vec![dep("e1", "a")],
            )
            .expect_err("an edge to an event that will not be stored must be rejected");
        assert!(format!("{err}").contains("e1 -> a"), "got: {err}");

        assert!(
            store.get_session_state(&scope).is_ok(),
            "the session must not be quarantined by a malformed batch"
        );
        assert!(
            store.get_full_state_for_tenant_scoped(T).is_ok(),
            "and the tenant must not lose its global-evaluator bootstrap"
        );
        assert_eq!(
            store.session_counts(&scope),
            (0, 0),
            "nothing from the rejected batch may be applied"
        );
    }

    /// A bad edge in a combined events+dependencies call must reject BEFORE
    /// the events are applied — and must not take the session out of service.
    ///
    /// The edge pre-pass inside the dependency merge runs after the events
    /// have been added to the shard, so rejecting there unwinds without
    /// committing or broadcasting and leaves memory ahead of RocksDB with
    /// nobody told — the state `diverged` exists to quarantine, produced by a
    /// client sending one malformed edge. Validating against "already
    /// recorded, plus the ids in this batch" catches it while the shard is
    /// still untouched.
    #[test]
    fn a_bad_edge_rejects_before_its_events_are_applied() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        let err = store
            .merge_events_with_dependencies(
                &scope,
                None,
                vec![ev("a", "first")],
                vec![dep("a", "ghost")],
            )
            .expect_err("an edge naming a node in neither the shard nor the batch must fail");
        assert!(format!("{err}").contains("a -> ghost"), "got: {err}");

        assert_eq!(
            store.session_counts(&scope),
            (0, 0),
            "the events must not have been applied"
        );
        // And the session is still usable: a malformed edge is a client bug,
        // not grounds for quarantining the session.
        assert!(
            store.get_session_state(&scope).is_ok(),
            "a rejected batch must not mark the session diverged"
        );

        // An edge to a node carried in the SAME batch is fine.
        store
            .merge_events_with_dependencies(
                &scope,
                None,
                vec![ev("a", "first"), ev("b", "second")],
                vec![dep("a", "b")],
            )
            .expect("an intra-batch edge must be accepted");
        assert_eq!(store.session_counts(&scope), (2, 1));
    }

    /// A re-record that carries only a policy-relevant field must land.
    ///
    /// The field-merge below the empty gate exists so a client can set
    /// `entity` or `derived_from` alone and have the rest preserved. Gating
    /// "is this event empty" on text and tools alone discarded exactly those
    /// updates — silently, returning an empty id — so a rule keyed on
    /// `entity` under-matched and any edge naming that id was dropped too.
    #[test]
    fn an_event_carrying_only_entity_is_merged_not_dropped() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        store
            .merge_events(&scope, None, vec![ev("m1", "the original text")])
            .unwrap();

        // Second record sets only `entity`.
        let entity_only = Event {
            text: None,
            agent: None,
            role: None,
            id: Some("m1".into()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: Some("tool-runner".into()),
            metadata: None,
        };
        let ids = store.merge_events(&scope, None, vec![entity_only]).unwrap();
        assert_eq!(
            ids,
            vec!["m1".to_string()],
            "the id must come back, not empty"
        );

        let (events, _edges, _seq) = store.get_session_state(&scope).unwrap();
        let m = events
            .iter()
            .find(|m| m.id.as_deref() == Some("m1"))
            .expect("the message must still be there");
        assert_eq!(m.entity.as_deref(), Some("tool-runner"), "entity must land");
        assert_eq!(
            m.text.as_deref(),
            Some("the original text"),
            "and the prior text must be preserved by the merge"
        );
    }

    /// A re-record that omits `metadata` keeps the recorded one.
    ///
    /// The content hash is over the whole event, so a merge that dropped
    /// this field would leave a node that no longer answers to the mark
    /// its writer holds, and every compact reference to it would be
    /// refused.
    #[test]
    fn a_re_record_preserves_metadata() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new(T, "s1");

        let mut first = ev("m1", "the original text");
        first.metadata = Some("{\"parts\":[1]}".into());
        store.merge_events(&scope, None, vec![first]).unwrap();

        let mut without = ev("m1", "the original text");
        without.entity = Some("tool-runner".into());
        store.merge_events(&scope, None, vec![without]).unwrap();

        let (events, _edges, _seq) = store.get_session_state(&scope).unwrap();
        let m = events
            .iter()
            .find(|m| m.id.as_deref() == Some("m1"))
            .expect("the message must still be there");
        assert_eq!(m.metadata.as_deref(), Some("{\"parts\":[1]}"));
    }

    /// A restart must put every edge back in the session that owns it, even
    /// when another session in the same tenant reused its node ids.
    ///
    /// Clients choose their own event ids, so collisions within a tenant are
    /// expected. Resolving an edge's shard from those ids at load — through a
    /// `(tenant, id)` index that is last-writer-wins — filed this edge under
    /// the wrong session and took it away from the right one. A `DependsOn`
    /// edge is a taint link: losing it stops a provenance-gated denial from
    /// firing, and gaining one invents provenance the session never had.
    /// Neither shows up until a restart. The edge's own scope is persisted
    /// alongside it and settles the question.
    #[test]
    fn reopen_routes_edges_by_their_own_scope_not_by_colliding_ids() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rocks").to_string_lossy().into_owned();
        let owner = SessionScope::new(T, "s1");
        let other = SessionScope::new(T, "s2");

        {
            let store = GraphStore::new(Some(&path)).unwrap();
            // s1 owns the dependency.
            store
                .merge_events(&owner, None, vec![ev("x", "src"), ev("y", "dst")])
                .unwrap();
            store
                .merge_dependencies(&owner, None, vec![dep("x", "y")])
                .unwrap();
            // s2 reuses both ids and records no dependency of its own. Its
            // rows sort after s1's, so the reverse index resolves both ids to
            // s2, which must not capture s1's edge.
            store
                .merge_events(&other, None, vec![ev("x", "mine"), ev("y", "also mine")])
                .unwrap();
        }
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            assert_eq!(
                store.session_counts(&owner).1,
                1,
                "the session that recorded the edge must still have it after a restart"
            );
            assert_eq!(
                store.session_counts(&other).1,
                0,
                "a session that recorded no dependency must not inherit one"
            );
        }
    }

    /// Same-tenant sessions can pick colliding node ids; the
    /// reverse index resolves to the last writer. Dropping the
    /// earlier session must not erase the index entry, or the
    /// surviving session's reads break.
    #[test]
    fn drop_session_preserves_node_to_session_for_colliding_sibling() {
        let store = GraphStore::new(None).unwrap();
        let scope_a = SessionScope::new(T, "sa");
        let scope_b = SessionScope::new(T, "sb");

        // Both sessions mint a node with id "shared". The reverse
        // index ends up pointing at scope_b (last writer wins).
        store
            .merge_events(&scope_a, None, vec![ev("shared", "from a")])
            .unwrap();
        store
            .merge_events(&scope_b, None, vec![ev("shared", "from b")])
            .unwrap();
        assert_eq!(
            store.scope_for_node(T, "shared").as_ref(),
            Some(&scope_b),
            "reverse index is last-writer-wins"
        );

        // Drop scope_a — the older writer. The reverse index
        // pointer for "shared" belongs to scope_b and must
        // survive.
        store.drop_session(&scope_a).unwrap();
        assert_eq!(
            store.scope_for_node(T, "shared").as_ref(),
            Some(&scope_b),
            "drop_session must not erase a sibling session's index entry",
        );

        // Now dropping scope_b cleans the entry (no one left to
        // claim it).
        store.drop_session(&scope_b).unwrap();
        assert!(store.scope_for_node(T, "shared").is_none());
    }

    /// An explicit `session` hint makes an arbitrary-id read resolve
    /// to the caller's named shard, disambiguating same-tenant id
    /// collisions that the `(tenant, id)` reverse index can only
    /// resolve last-writer-wins — the fuller fix for that residual.
    #[test]
    fn backward_slice_prefers_explicit_session_over_reverse_index() {
        let store = GraphStore::new(None).unwrap();
        let s1 = SessionScope::new(T, "s1");
        let s2 = SessionScope::new(T, "s2");

        // Same dest id "shared" in both sessions, each with its own
        // dependency: a1->shared in s1, b1->shared in s2.
        store
            .merge_events_with_dependencies(
                &s1,
                None,
                vec![ev("a1", "a"), ev("shared", "sh")],
                vec![dep("a1", "shared")],
            )
            .unwrap();
        store
            .merge_events_with_dependencies(
                &s2,
                None,
                vec![ev("b1", "b"), ev("shared", "sh")],
                vec![dep("b1", "shared")],
            )
            .unwrap();

        // Reverse index is last-writer-wins → s2.
        assert_eq!(store.scope_for_node(T, "shared").as_ref(), Some(&s2));

        let ids = |nodes: Vec<Event>| -> HashSet<String> {
            nodes.into_iter().filter_map(|e| e.id).collect()
        };

        // No hint → resolves via the reverse index → s2's chain (b1).
        let (lww, _) = store.backward_slice(T, "shared", None, None).unwrap();
        let lww = ids(lww);
        assert!(
            lww.contains("b1") && !lww.contains("a1"),
            "no-hint slice follows last-writer-wins (s2); got {lww:?}"
        );

        // Explicit session hint reaches each session's own chain,
        // unambiguously: the reverse index does not hide the sibling.
        let s1_ids = ids(store
            .backward_slice(T, "shared", Some("s1"), None)
            .unwrap()
            .0);
        assert!(
            s1_ids.contains("a1") && !s1_ids.contains("b1"),
            "session=s1 hint resolves s1's shard; got {s1_ids:?}"
        );
        let s2_ids = ids(store
            .backward_slice(T, "shared", Some("s2"), None)
            .unwrap()
            .0);
        assert!(
            s2_ids.contains("b1") && !s2_ids.contains("a1"),
            "session=s2 hint resolves s2's shard; got {s2_ids:?}"
        );
    }

    // ── Policy persistence + session ownership ─────────────

    use crate::{PersistedPolicy, PolicyMetadataFact};

    #[test]
    fn policy_binding_round_trip() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new("acme", "s1");
        store.put_policy_binding(&scope, "abc123").unwrap();

        let all = store.all_policy_bindings().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, scope);
        assert_eq!(all[0].1, "abc123");

        store.delete_policy_binding(&scope).unwrap();
        assert!(store.all_policy_bindings().unwrap().is_empty());
    }

    #[test]
    fn tenant_default_round_trip() {
        let store = GraphStore::new(None).unwrap();
        store.put_tenant_default_policy("acme", "hash-a").unwrap();
        store.put_tenant_default_policy("orgb", "hash-b").unwrap();
        // Overwrite acme's default — second write wins.
        store.put_tenant_default_policy("acme", "hash-c").unwrap();

        let defaults: HashMap<String, String> =
            store.all_tenant_defaults().unwrap().into_iter().collect();
        assert_eq!(defaults.get("acme").map(String::as_str), Some("hash-c"));
        assert_eq!(defaults.get("orgb").map(String::as_str), Some("hash-b"));
    }

    #[test]
    fn policy_source_round_trip() {
        let store = GraphStore::new(None).unwrap();
        let bundle = PersistedPolicy {
            policy_source: "IsAuthorized(0).".into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            functor_admission: Default::default(),
        };
        store.put_policy_source("hash1", &bundle).unwrap();

        let got = store.get_policy_source("hash1").unwrap().unwrap();
        assert_eq!(got.policy_source, bundle.policy_source);
        assert_eq!(got.backend, bundle.backend);
        assert!(store.get_policy_source("missing").unwrap().is_none());
    }

    /// The two admission classes are independent records under one content
    /// hash, so an upload can never widen a stored record's class nor erase
    /// the other class's record.
    #[test]
    fn the_admission_class_is_part_of_the_policy_source_key() {
        use crate::persistence::FunctorAdmission;
        let store = GraphStore::new(None).unwrap();
        let bundle = |admission| PersistedPolicy {
            policy_source: "IsAuthorized(0).".into(),
            functor_source: "// C++".into(),
            backend: "souffle".into(),
            functor_admission: admission,
        };

        // A non-admin uploads it, then an admin uploads the same content.
        store
            .put_policy_source("hash-both", &bundle(FunctorAdmission::User))
            .unwrap();
        store
            .put_policy_source("hash-both", &bundle(FunctorAdmission::Admin))
            .unwrap();
        // The admin's write did not rewrite the user record's class...
        assert_eq!(
            store
                .get_policy_source_in_class("hash-both", FunctorAdmission::User)
                .unwrap()
                .map(|p| p.functor_admission),
            Some(FunctorAdmission::User),
            "an admin upload must not widen a record already stored as user-supplied"
        );
        // ...and the admin record is there too, on its own key.
        assert_eq!(
            store
                .get_policy_source_in_class("hash-both", FunctorAdmission::Admin)
                .unwrap()
                .map(|p| p.functor_admission),
            Some(FunctorAdmission::Admin)
        );
        assert_eq!(
            store
                .policy_sources_by_least_privilege("hash-both")
                .unwrap()
                .into_iter()
                .map(|(class, _)| class)
                .collect::<Vec<_>>(),
            vec![FunctorAdmission::User, FunctorAdmission::Admin],
            "records come back least-privileged first, each under the class its \
             key encodes"
        );

        // The other order: an admin uploads, then a non-admin uploads the same
        // content. The admin record survives.
        store
            .put_policy_source("hash-rev", &bundle(FunctorAdmission::Admin))
            .unwrap();
        store
            .put_policy_source("hash-rev", &bundle(FunctorAdmission::User))
            .unwrap();
        assert_eq!(
            store
                .get_policy_source_in_class("hash-rev", FunctorAdmission::Admin)
                .unwrap()
                .map(|p| p.functor_admission),
            Some(FunctorAdmission::Admin),
            "a non-admin upload must not take away an existing admin record"
        );

        // One class stored alone leaves the other class empty, not aliased.
        store
            .put_policy_source("hash-user-only", &bundle(FunctorAdmission::User))
            .unwrap();
        assert!(store
            .get_policy_source_in_class("hash-user-only", FunctorAdmission::Admin)
            .unwrap()
            .is_none());
    }

    /// A policy source is written once. A re-write of the same content is a
    /// no-op, and a write of DIFFERENT content under an occupied key is
    /// refused with the stored bytes left alone — the key is a content hash,
    /// so two contents under one key is a collision or a caller writing under
    /// a key it did not derive, and neither may replace a source that live
    /// bindings and boot replay read.
    #[test]
    fn a_policy_source_write_never_replaces_a_stored_one() {
        use crate::persistence::FunctorAdmission;
        let store = GraphStore::new(None).unwrap();
        let stored = PersistedPolicy {
            policy_source: "IsAuthorized(0).".into(),
            functor_source: "// C++".into(),
            backend: "souffle".into(),
            functor_admission: FunctorAdmission::User,
        };
        store.put_policy_source("hash-x", &stored).unwrap();

        // Same content again: accepted, nothing changes.
        store.put_policy_source("hash-x", &stored).unwrap();
        assert_eq!(
            store
                .get_policy_source("hash-x")
                .unwrap()
                .map(|p| p.policy_source),
            Some("IsAuthorized(0).".to_string()),
        );

        // Different content under the same key and class: refused.
        let planted = PersistedPolicy {
            policy_source: "IsAuthorized(idx) :- Actions(idx, _).".into(),
            ..stored.clone()
        };
        let err = store
            .put_policy_source("hash-x", &planted)
            .expect_err("a different source under an occupied key must be refused");
        assert!(
            matches!(err, crate::GraphError::ContentHashAmbiguity(_)),
            "expected a hash-ambiguity refusal, got: {err}"
        );
        // The refused write left the stored record untouched.
        let after = store.get_policy_source("hash-x").unwrap().unwrap();
        assert_eq!(after.policy_source, "IsAuthorized(0).");
        assert_eq!(after.functor_source, "// C++");

        // The explicit-class write is refused on the same terms, so no caller
        // reaches the store past this rule.
        let err = store
            .put_policy_source_in_class("hash-x", FunctorAdmission::User, &planted)
            .expect_err("the explicit-class write must be refused too");
        assert!(matches!(err, crate::GraphError::ContentHashAmbiguity(_)));
        assert_eq!(
            store
                .get_policy_source("hash-x")
                .unwrap()
                .unwrap()
                .policy_source,
            "IsAuthorized(0).",
        );

        // The OTHER class is a different key, so the same content still
        // writes there.
        store
            .put_policy_source_in_class("hash-x", FunctorAdmission::Admin, &stored)
            .expect("the admin key is free");
        assert!(store
            .get_policy_source_in_class("hash-x", FunctorAdmission::Admin)
            .unwrap()
            .is_some());
    }

    #[test]
    fn a_tenant_wide_delete_clears_idle_and_live_bindings_alike() {
        // What a force rollout relies on. Dropping the in-memory binding is
        // enough whenever the new default is different policy content: the
        // session re-binds to another hash and its stored row is never read
        // again. It is NOT enough when the same content is force-promoted —
        // the re-bound hash matches the stored row, metadata assembly finds it
        // and takes the "this session pinned itself" branch, and the
        // pre-rollout config is served from then on.
        let store = GraphStore::new(None).unwrap();
        let fact = |a: &str| crate::persistence::PolicyMetadataFact {
            rel: "RuleEnabled".into(),
            a: a.into(),
            b: String::new(),
        };
        let live = SessionScope::new("acme", "s-live");
        let idle = SessionScope::new("acme", "s-idle");
        let other = SessionScope::new("globex", "s-other");
        for (scope, who) in [(&live, "live"), (&idle, "idle"), (&other, "other")] {
            store
                .put_binding_metadata(scope, "hashSame", &[fact(who)])
                .unwrap();
        }

        store.clear_tenant_binding_metadata("acme").unwrap();

        // Both of the tenant's sessions are cleared — including the idle one,
        // which has a stored row and no live evaluator to evict.
        assert!(store
            .get_binding_metadata(&live, "hashSame")
            .unwrap()
            .is_none());
        assert!(store
            .get_binding_metadata(&idle, "hashSame")
            .unwrap()
            .is_none());
        // Another tenant on the same policy content keeps its own.
        assert!(store
            .get_binding_metadata(&other, "hashSame")
            .unwrap()
            .is_some());
    }

    #[test]
    fn policy_metadata_falls_back_to_a_pre_tenant_scoped_row() {
        // A database carried across the change that introduced tenant-scoped
        // keys still holds rows under the bare content hash. Reading those as
        // "no config" does not fail safe: a metadata-gated rule with no facts
        // matches nothing, so the policy silently stops enforcing.
        let store = GraphStore::new(None).unwrap();
        let facts = vec![crate::persistence::PolicyMetadataFact {
            rel: "TrustedDomain".into(),
            a: "registry.npmjs.org".into(),
            b: String::new(),
        }];
        store
            .put_legacy_policy_metadata("hashLegacy", &facts)
            .unwrap();

        let got = store.get_policy_metadata("acme", "hashLegacy").unwrap();
        assert_eq!(got.len(), 1, "a pre-tenant-scoped row must still be read");
        assert_eq!(got[0].a, "registry.npmjs.org");

        // The first tenant-scoped write supersedes it for that tenant, and
        // leaves other tenants on the legacy row until they write their own.
        let scoped = vec![crate::persistence::PolicyMetadataFact {
            rel: "TrustedDomain".into(),
            a: "internal.example".into(),
            b: String::new(),
        }];
        store
            .put_policy_metadata("acme", "hashLegacy", &scoped)
            .unwrap();
        assert_eq!(
            store.get_policy_metadata("acme", "hashLegacy").unwrap()[0].a,
            "internal.example"
        );
        assert_eq!(
            store.get_policy_metadata("globex", "hashLegacy").unwrap()[0].a,
            "registry.npmjs.org"
        );
    }

    #[test]
    fn policy_metadata_round_trip() {
        let store = GraphStore::new(None).unwrap();

        // Absent → empty (not an error): a policy with no config is
        // the common case.
        assert!(store
            .get_policy_metadata("acme", "missing")
            .unwrap()
            .is_empty());

        let facts = vec![
            PolicyMetadataFact {
                rel: "TrustedDomain".into(),
                a: "registry.npmjs.org".into(),
                b: String::new(),
            },
            PolicyMetadataFact {
                rel: "CooldownDays".into(),
                a: "7".into(),
                b: String::new(),
            },
        ];
        store.put_policy_metadata("acme", "hashM", &facts).unwrap();

        let got = store.get_policy_metadata("acme", "hashM").unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].rel, "TrustedDomain");
        assert_eq!(got[0].a, "registry.npmjs.org");
        assert_eq!(got[1].rel, "CooldownDays");
        assert_eq!(got[1].a, "7");

        // Another tenant binding the same source keeps its own config: a
        // policy source is not private to a tenant, so keying this row by the
        // hash alone let one tenant's configuration decide another's sessions.
        store
            .put_policy_metadata(
                "globex",
                "hashM",
                &[PolicyMetadataFact {
                    rel: "CooldownDays".into(),
                    a: "99".into(),
                    b: String::new(),
                }],
            )
            .unwrap();
        let mine = store.get_policy_metadata("acme", "hashM").unwrap();
        assert_eq!(mine.len(), 2, "another tenant's config replaced ours");
        assert_eq!(mine[1].a, "7");

        // Last writer wins within one tenant (the hash covers source, not config).
        store
            .put_policy_metadata(
                "acme",
                "hashM",
                &[PolicyMetadataFact {
                    rel: "CooldownDays".into(),
                    a: "14".into(),
                    b: String::new(),
                }],
            )
            .unwrap();
        let got = store.get_policy_metadata("acme", "hashM").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].a, "14");
    }

    /// Two sessions binding the *same* policy source with different config each
    /// keep their own. This is the whole point of keying by the binding: the
    /// content hash covers the policy source and not the config, so a
    /// hash-keyed entry is shared by every session that binds that source and
    /// the last writer wins for all of them. A metadata-gated policy running
    /// under a sibling's config matches nothing and therefore denies nothing.
    #[test]
    fn binding_metadata_is_per_session_not_per_source() {
        let store = GraphStore::new(None).unwrap();
        let alpha = SessionScope::new("acme", "alpha");
        let bravo = SessionScope::new("acme", "bravo");
        let hash = "sharedsourcehash";

        let blocked = |tool: &str| {
            vec![PolicyMetadataFact {
                rel: "blocked".into(),
                a: tool.into(),
                b: String::new(),
            }]
        };

        // Absent → empty, which is the signal to fall back to the policy-wide
        // entry rather than an error.
        assert!(store.get_binding_metadata(&alpha, hash).unwrap().is_none());

        store
            .put_binding_metadata(&alpha, hash, &blocked("tool_a"))
            .unwrap();
        store
            .put_binding_metadata(&bravo, hash, &blocked("tool_b"))
            .unwrap();

        let got_a = store.get_binding_metadata(&alpha, hash).unwrap().unwrap();
        let got_b = store.get_binding_metadata(&bravo, hash).unwrap().unwrap();
        assert_eq!(got_a.len(), 1);
        assert_eq!(got_b.len(), 1);
        assert_eq!(got_a[0].a, "tool_a", "alpha inherited bravo's config");
        assert_eq!(got_b[0].a, "tool_b", "bravo inherited alpha's config");

        // Replace, not append: a re-pin carrying a changed value must not leave
        // the old one behind, because a value fact is not a set member.
        store
            .put_binding_metadata(&alpha, hash, &blocked("tool_c"))
            .unwrap();
        let got_a = store.get_binding_metadata(&alpha, hash).unwrap().unwrap();
        assert_eq!(got_a.len(), 1);
        assert_eq!(got_a[0].a, "tool_c");

        // An empty set is a *statement*, distinct from never having pinned:
        // stored, read back as Some(empty), and so it suppresses the fallback
        // to the shared policy-wide key rather than inviting it.
        store.put_binding_metadata(&alpha, hash, &[]).unwrap();
        assert!(
            matches!(store.get_binding_metadata(&alpha, hash).unwrap(), Some(v) if v.is_empty()),
            "an explicit empty config was indistinguishable from no bind at all"
        );

        // One session may hold config for several policies; both are keyed
        // under its scope and both are dropped together at EndSession.
        store
            .put_binding_metadata(&alpha, "otherhash", &blocked("tool_d"))
            .unwrap();
        store.delete_binding_metadata(&alpha).unwrap();
        assert!(store.get_binding_metadata(&alpha, hash).unwrap().is_none());
        assert!(store
            .get_binding_metadata(&alpha, "otherhash")
            .unwrap()
            .is_none());
        // ...and only that scope's.
        assert_eq!(
            store
                .get_binding_metadata(&bravo, hash)
                .unwrap()
                .unwrap()
                .len(),
            1,
            "deleting alpha's config dropped bravo's too"
        );
    }

    /// A scope whose encoded bytes are a prefix of another must not have its
    /// deletion take the longer one with it. `to_storage_bytes` length-prefixes
    /// both components precisely so `(t, "s") ++ hash` cannot be confused with
    /// `(t, "sX") ++ hash`; this pins that the prefix-scan delete relies on it.
    #[test]
    fn binding_metadata_delete_does_not_bleed_across_similar_scopes() {
        let store = GraphStore::new(None).unwrap();
        let short = SessionScope::new("acme", "s");
        let long = SessionScope::new("acme", "sX");
        let fact = vec![PolicyMetadataFact {
            rel: "blocked".into(),
            a: "tool".into(),
            b: String::new(),
        }];

        store.put_binding_metadata(&short, "h", &fact).unwrap();
        store.put_binding_metadata(&long, "h", &fact).unwrap();

        store.delete_binding_metadata(&short).unwrap();
        assert!(store.get_binding_metadata(&short, "h").unwrap().is_none());
        assert_eq!(
            store
                .get_binding_metadata(&long, "h")
                .unwrap()
                .unwrap()
                .len(),
            1,
            "deleting scope \"s\" also dropped scope \"sX\""
        );
    }

    /// First write claims, second write by same principal is OK,
    /// second write by different principal returns the existing
    /// owner so the handler can deny.
    #[test]
    fn session_owner_claim_then_conflict() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new("acme", "s1");

        // First touch — claim.
        assert_eq!(
            store
                .check_or_claim_session_owner(&scope, Some("alice"))
                .unwrap(),
            None
        );
        // Second touch by alice — matches.
        assert_eq!(
            store
                .check_or_claim_session_owner(&scope, Some("alice"))
                .unwrap(),
            None
        );
        // Bob — conflict; handler must check and reject.
        assert_eq!(
            store
                .check_or_claim_session_owner(&scope, Some("bob"))
                .unwrap(),
            Some("alice".into())
        );
        // Anonymous on a claimed session — conflict.
        assert_eq!(
            store.check_or_claim_session_owner(&scope, None).unwrap(),
            Some("alice".into())
        );

        // EndSession path: delete the owner record. A new principal
        // can claim the recycled id.
        store.delete_session_owner(&scope).unwrap();
        assert_eq!(
            store
                .check_or_claim_session_owner(&scope, Some("bob"))
                .unwrap(),
            None
        );
    }

    /// Owner records are scoped on `(tenant, session_id)`. Two
    /// tenants picking the same human-friendly session id keep
    /// independent owners.
    #[test]
    fn session_owners_isolate_by_tenant() {
        let store = GraphStore::new(None).unwrap();
        let acme = SessionScope::new("acme", "conv-1");
        let orgb = SessionScope::new("orgb", "conv-1");

        assert!(store
            .check_or_claim_session_owner(&acme, Some("alice"))
            .unwrap()
            .is_none());
        // Same session_id, different tenant — fresh owner slot.
        assert!(store
            .check_or_claim_session_owner(&orgb, Some("bob"))
            .unwrap()
            .is_none());

        assert_eq!(
            store.get_session_owner(&acme).unwrap().as_deref(),
            Some("alice")
        );
        assert_eq!(
            store.get_session_owner(&orgb).unwrap().as_deref(),
            Some("bob")
        );
    }

    /// Concurrent first-touch claims on the same `(tenant,
    /// session)` must produce a deterministic winner. A plain
    /// read-then-write would be racy: both threads could observe
    /// `None` and both write, with last-writer-wins.
    #[test]
    fn session_owner_claim_is_atomic_under_contention() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new("acme", "race");

        let n_threads = 8;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let mut handles = Vec::with_capacity(n_threads);
        for i in 0..n_threads {
            let s = Arc::clone(&store);
            let b = Arc::clone(&barrier);
            let sc = scope.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let principal = format!("principal-{i}");
                s.check_or_claim_session_owner(&sc, Some(&principal))
                    .unwrap()
            }));
        }
        let outcomes: Vec<Option<String>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one thread sees Ok(None) (the winner that
        // claimed). The rest see Ok(Some(winner_principal)).
        let winners = outcomes.iter().filter(|o| o.is_none()).count();
        assert_eq!(winners, 1, "exactly one thread must win the claim");

        // All non-winners must report the same owner — the winner.
        let owner = store.get_session_owner(&scope).unwrap().unwrap();
        for o in outcomes.iter().flatten() {
            assert_eq!(o, &owner);
        }
    }

    #[test]
    fn completed_owner_claims_reclaim_bookkeeping_without_releasing_ownership() {
        let store = GraphStore::new(None).unwrap();
        let pending_scope = SessionScope::new("acme", "pending");
        let (_, pending) = store
            .claim_session_owner_with_token(&pending_scope, Some("alice"))
            .unwrap();
        for i in 0..1000 {
            let scope = SessionScope::new("acme", format!("completed-{i}"));
            let (_, token) = store
                .claim_session_owner_with_token(&scope, Some("alice"))
                .unwrap();
            let token = token.unwrap();
            store.commit_session_owner_claim(&token);
            assert!(!store.release_session_owner_claim(&token).unwrap());
            assert_eq!(
                store.get_session_owner(&scope).unwrap().as_deref(),
                Some("alice")
            );
            let wrapper_scope = SessionScope::new("acme", format!("wrapper-{i}"));
            assert_eq!(
                store
                    .claim_session_owner(&wrapper_scope, Some("alice"))
                    .unwrap(),
                OwnerClaim::Claimed
            );
            assert_eq!(store.owner_claim_lock.lock().len(), 1);
        }
        assert!(store
            .release_session_owner_claim(&pending.unwrap())
            .unwrap());
        assert!(store.owner_claim_lock.lock().is_empty());
    }

    #[test]
    fn stale_owner_claim_commit_cannot_disarm_a_recreated_claim() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new("acme", "aba-commit");
        let (_, old) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        store.delete_session_owner(&scope).unwrap();
        let (_, fresh) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        store.commit_session_owner_claim(&old.unwrap());
        assert!(store.release_session_owner_claim(&fresh.unwrap()).unwrap());
        assert!(store.owner_claim_lock.lock().is_empty());
    }

    #[test]
    fn owner_rollback_cannot_remove_an_in_flight_same_principal_claim() {
        let store = Arc::new(GraphStore::new(None).unwrap());
        let scope = SessionScope::new("acme", "race");
        let (_, first) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        let first = first.unwrap();
        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
        let (proceed_tx, proceed_rx) = std::sync::mpsc::channel();
        let writer = {
            let store = Arc::clone(&store);
            let scope = scope.clone();
            std::thread::spawn(move || {
                // The legacy wrapper must invalidate rollback tokens too.
                assert_eq!(
                    store.claim_session_owner(&scope, Some("alice")).unwrap(),
                    OwnerClaim::AlreadyHeld
                );
                claimed_tx.send(()).unwrap();
                proceed_rx.recv().unwrap();
                store
                    .merge_events(
                        &scope,
                        Some("alice"),
                        vec![Event {
                            id: Some("message".into()),
                            text: Some("committed".into()),
                            ..Default::default()
                        }],
                    )
                    .unwrap();
            })
        };
        claimed_rx.recv().unwrap();
        assert_eq!(store.session_counts(&scope), (0, 0));
        assert!(!store.release_session_owner_claim(&first).unwrap());
        proceed_tx.send(()).unwrap();
        writer.join().unwrap();
        assert_eq!(
            store.get_session_owner(&scope).unwrap().as_deref(),
            Some("alice")
        );
        assert_eq!(store.session_counts(&scope).0, 1);
        assert_eq!(
            store.claim_session_owner(&scope, Some("bob")).unwrap(),
            OwnerClaim::HeldBy("alice".into())
        );
    }

    #[test]
    fn owner_rollback_tokens_are_scope_bound_and_never_reused() {
        let store = GraphStore::new(None).unwrap();
        let scope = SessionScope::new("acme", "aba");
        let other = SessionScope::new("acme", "other");
        let (_, old) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        let (_, other_token) = store
            .claim_session_owner_with_token(&other, Some("alice"))
            .unwrap();
        store.delete_session_owner(&scope).unwrap();
        let (_, current) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        assert!(!store.release_session_owner_claim(&old.unwrap()).unwrap());
        assert!(store
            .release_session_owner_claim(&current.unwrap())
            .unwrap());
        assert!(store
            .release_session_owner_claim(&other_token.unwrap())
            .unwrap());
        assert!(store.owner_claim_lock.lock().is_empty());

        let (_, token) = store
            .claim_session_owner_with_token(&scope, Some("alice"))
            .unwrap();
        store.put_session_owner(&scope, "alice").unwrap();
        assert!(!store.release_session_owner_claim(&token.unwrap()).unwrap());
        assert_eq!(
            store.get_session_owner(&scope).unwrap().as_deref(),
            Some("alice")
        );
        assert!(store.owner_claim_lock.lock().is_empty());
    }

    /// `check_or_claim_session_owner` round-trips through reopen —
    /// the keyspace is persistent, not in-memory only.
    #[test]
    fn session_owner_survives_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rocks").to_string_lossy().into_owned();
        let scope = SessionScope::new("acme", "s1");
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            store
                .check_or_claim_session_owner(&scope, Some("alice"))
                .unwrap();
        }
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            // bob trying to claim sees alice already owns.
            assert_eq!(
                store
                    .check_or_claim_session_owner(&scope, Some("bob"))
                    .unwrap(),
                Some("alice".into())
            );
        }
    }

    /// Policy bindings + sources survive reopen, mirroring the
    /// session-owner case.
    #[test]
    fn policy_bindings_survive_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rocks").to_string_lossy().into_owned();
        let scope = SessionScope::new("acme", "s1");
        let bundle = PersistedPolicy {
            policy_source: "IsAuthorized(0).".into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            functor_admission: Default::default(),
        };
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            store.put_policy_binding(&scope, "h1").unwrap();
            store.put_tenant_default_policy("acme", "h1").unwrap();
            store.put_policy_source("h1", &bundle).unwrap();
        }
        {
            let store = GraphStore::new(Some(&path)).unwrap();
            let bindings = store.all_policy_bindings().unwrap();
            assert_eq!(bindings, vec![(scope.clone(), "h1".into())]);
            let defaults = store.all_tenant_defaults().unwrap();
            assert_eq!(defaults, vec![("acme".into(), "h1".into())]);
            let src = store.get_policy_source("h1").unwrap().unwrap();
            assert_eq!(src.policy_source, bundle.policy_source);
        }
    }
}
