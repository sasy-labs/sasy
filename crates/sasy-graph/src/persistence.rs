//! RocksDB persistence layer for the graph store.
//!
//! Column families — the graph itself:
//! - `messages`      — MessageNode keyed by `[scope_bytes][message_id]`
//! - `computations`  — ComputationNode keyed by `[scope_bytes][span_id]`
//! - `edges`         — EdgeData keyed by `[scope_bytes][EdgeKey JSON]`
//! - `meta`          — Scalar metadata (e.g. sequence counter)
//!
//! and the policy/session keyspaces that ride alongside it, each
//! documented at its `CF_*` constant below:
//! - `policy_bindings`  — `(tenant, session)` → content hash (a session pin)
//! - `policy_defaults`  — tenant → content hash (the tenant default)
//! - `policy_source`    — content hash → the source bytes to recompile from
//! - `policy_metadata`  — `(tenant, content hash)` → tenant-wide config
//! - `binding_metadata` — `(scope, content hash)` → that binding's config
//! - `session_metadata` — scope → the session's dynamic facts
//! - `session_owners`   — scope → the principal that claimed it
//!
//! All node/edge keys are prefixed with
//! [`SessionScope::to_storage_bytes`], which length-prefixes both
//! components: `[tenant_len: u32 BE][tenant][session_len: u32 BE][session]`.
//! Length-prefixing both is necessary so a user-controlled `id` (or
//! `session`) suffix can't be confused with another `(tenant,
//! session)` pair's prefix — e.g. `(t, "S") ++ "X"` vs `(t, "SX") ++
//! ""`. The encoding is a valid RocksDB scan prefix, so
//! per-`(tenant, session)` enumeration is just a `prefix_iterator_cf`
//! lookup.

use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use sasy_common::SessionScope;
use serde::{Deserialize, Serialize};

use crate::error::GraphError;
use crate::types::{ComputationNode, EdgeData, EdgeKey, MessageNode};

/// Who was allowed to supply this policy's custom C++ functor source.
///
/// Recorded when the policy is uploaded and read back every time the source
/// is recompiled, so the decision the gate made at upload time can be re-made
/// against the settings that are in force at load time.
///
/// The default is the least privileged value, [`FunctorAdmission::User`]. It
/// is also what a record that carries no admission field deserializes as:
/// nothing is known about who supplied such a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctorAdmission {
    /// Supplied by a caller the functor gate found to be an admin. Loads on
    /// any host, whatever the operator's opt-in says.
    Admin,
    /// Supplied by a caller who was not an admin, or by a caller nothing is
    /// known about. Loads only where the operator's opt-in admits it.
    #[default]
    User,
}

/// Snapshot of the inputs needed to recompile a policy from its
/// content hash. Persisted in [`CF_POLICY_SOURCE`] so a restart
/// can rebuild the in-memory registry without callers re-uploading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPolicy {
    pub policy_source: String,
    pub functor_source: String,
    pub backend: String,
    /// Who supplied [`Self::functor_source`]. Absent from records written
    /// before the field existed; those read back as
    /// [`FunctorAdmission::User`], the least privileged value. Meaningless —
    /// and never consulted — when `functor_source` is empty.
    ///
    /// **This field is part of the storage key, not just the value.** The
    /// mechanism is described on [`RocksStore::put_policy_source`]: the two
    /// classes are separate records under one content hash, so no upload can
    /// widen a stored record's class or erase the other class's record, and a
    /// load path asks within the class its settings admit.
    #[serde(default)]
    pub functor_admission: FunctorAdmission,
}

/// One static policy-metadata fact, persisted alongside the policy
/// (keyed by content hash, not part of it) so a per-session
/// evaluator can re-seed its `PolicyMetadata(rel, a, b)` EDB at
/// bootstrap without the caller re-sending config. Mirrors the IPC
/// fact shape; kept here so sasy-graph need not depend on the
/// evaluator types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetadataFact {
    pub rel: String,
    pub a: String,
    pub b: String,
}

/// Build a tenant-prefixed key from a scope + a content id. The
/// `scope` portion is length-prefixed (see
/// [`SessionScope::to_storage_bytes`]) so any byte content in `id`
/// is unambiguously separable from the tenant/session prefix.
fn scoped_key(scope: &SessionScope, id: &[u8]) -> Vec<u8> {
    let mut k = scope.to_storage_bytes();
    k.extend_from_slice(id);
    k
}

const CF_MESSAGES: &str = "messages";
const CF_COMPUTATIONS: &str = "computations";
const CF_EDGES: &str = "edges";
const CF_META: &str = "meta";
/// Key in [`CF_META`] holding the store's schema version.
const SCHEMA_VERSION_KEY: &[u8] = b"schema_version";
/// Serialized-format version of this store. Bump it whenever the on-disk
/// shape of a value changes — the node/edge JSON, a key encoding, the meaning
/// of a column family. A store stamped with anything else is refused at open
/// rather than read through the wrong shape: a value that deserializes
/// *partially* into a newer type loses fields silently, and losing a field of
/// a policy binding or a session owner is a change in who is allowed to do
/// what.
const SCHEMA_VERSION: u64 = 1;
/// `(tenant, session) → content_hash` for sessions explicitly
/// pinned via `SetPolicy(scope=Session)`. Absent entries mean the
/// session falls through to the tenant default.
const CF_POLICY_BINDINGS: &str = "policy_bindings";
/// `tenant → content_hash` for the tenant default policy.
const CF_POLICY_DEFAULTS: &str = "policy_defaults";
/// `content_hash → PersistedPolicy` (JSON-encoded source bytes +
/// functor source + backend) so a restart can recompile the same
/// policy from its hash without the caller re-uploading.
const CF_POLICY_SOURCE: &str = "policy_source";
/// `(tenant, content_hash) → Vec<PolicyMetadataFact>` (JSON) — static config
/// facts that travel with the policy but are *not* part of its
/// content hash, so a precompiled (restricted-build) evaluator
/// accepts config without recompiling. Read back at per-session
/// bootstrap and seeded into the `PolicyMetadata` EDB.
const CF_POLICY_METADATA: &str = "policy_metadata";
/// `scope → Vec<PolicyMetadataFact>` (JSON) — *dynamic*, session-scoped config
/// facts appended mid-session via `UpdatePolicyMetadata` (e.g. detaint
/// decisions). Append-only; read back at bootstrap and merged with the
/// policy's static `CF_POLICY_METADATA` into the `PolicyMetadata` EDB.
const CF_SESSION_METADATA: &str = "session_metadata";
/// `(scope ++ content_hash) → Vec<PolicyMetadataFact>` (JSON) — config supplied
/// on a *session-scoped* `SetPolicy`. Keyed by the **binding** rather than by
/// the policy source, because an identical source is one content hash however
/// many sessions bind it: writing such config to [`CF_POLICY_METADATA`] lets
/// concurrent sessions overwrite each other's configuration, and a
/// metadata-gated policy handed a sibling's config matches nothing and denies
/// nothing. Read back at bootstrap in preference to [`CF_POLICY_METADATA`],
/// which carries only tenant-wide (Default/Force) config.
const CF_BINDING_METADATA: &str = "binding_metadata";
/// `(tenant, session) → principal_id` recorded on first write to
/// a session. Subsequent SetPolicy / EndSession from a different
/// principal are rejected (unless the caller has the admin role).
const CF_SESSION_OWNERS: &str = "session_owners";

/// Thin wrapper around RocksDB with typed column
/// families.
pub struct RocksStore {
    db: DB,
    /// Serializes the session-metadata read-modify-write so two concurrent
    /// appends for the same scope can't both read the old value and clobber
    /// each other (which could drop a fact — most dangerously a detaint_denied).
    metadata_lock: parking_lot::Mutex<()>,
    /// Test-only write fault. When set, every node/edge put fails, which is
    /// the only way to exercise the "durable before visible" ordering on the
    /// computation path — RocksDB itself does not fail a memtable write on
    /// demand. Compiled out of production builds.
    #[cfg(test)]
    fail_writes: std::sync::atomic::AtomicBool,
}

/// One removed row, key and value exactly as RocksDB held them.
type ClearedRow = (Box<[u8]>, Box<[u8]>);

/// Binding-config rows taken out by a tenant-wide clear, held verbatim so the
/// exact rows can be put back if the rollout they were cleared for never
/// publishes. Opaque: the holder only carries it and hands it back.
#[derive(Debug, Default)]
pub struct ClearedBindingMetadata(Vec<ClearedRow>);

impl ClearedBindingMetadata {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl RocksStore {
    /// Open (or create) the database at `path`.
    pub fn open(path: &str) -> Result<Self, GraphError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_opts = Options::default();
        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_MESSAGES, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_COMPUTATIONS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_EDGES, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_META, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_POLICY_BINDINGS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_POLICY_DEFAULTS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_POLICY_SOURCE, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_POLICY_METADATA, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_SESSION_METADATA, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_BINDING_METADATA, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_SESSION_OWNERS, cf_opts),
        ];

        let db = DB::open_cf_descriptors(&opts, path, cfs)?;
        let store = Self {
            db,
            metadata_lock: parking_lot::Mutex::new(()),
            #[cfg(test)]
            fail_writes: std::sync::atomic::AtomicBool::new(false),
        };
        store.check_or_stamp_schema_version(path)?;
        Ok(store)
    }

    /// Agree with the store on the format before reading anything out of it.
    ///
    /// Absent: stamp the current version. Stores written before the stamp
    /// existed ARE version 1 — the format has not changed since — so this is a
    /// record of a fact, not a guess about one.
    ///
    /// Present and different: refuse. Continuing would read every value
    /// through the wrong shape, and the failure would surface later as missing
    /// bindings or a mis-parsed graph rather than as a startup error the
    /// operator can act on.
    fn check_or_stamp_schema_version(&self, path: &str) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_META).unwrap();
        let Some(raw) = self.db.get_cf(&cf, SCHEMA_VERSION_KEY)? else {
            self.db
                .put_cf(&cf, SCHEMA_VERSION_KEY, SCHEMA_VERSION.to_le_bytes())?;
            tracing::info!(
                path,
                version = SCHEMA_VERSION,
                "graph store carried no schema version; stamped the current one"
            );
            return Ok(());
        };
        // A value of any other width was not written by this code, so it names
        // no version we can compare against — treat it the same as a mismatch.
        let found = <[u8; 8]>::try_from(raw.as_slice()).map(u64::from_le_bytes);
        match found {
            Ok(v) if v == SCHEMA_VERSION => Ok(()),
            Ok(v) => Err(GraphError::SchemaVersion(format!(
                "graph store at {path} has schema version {v}, but this build reads \
                 version {SCHEMA_VERSION}; migrate the store or point --data-dir at a \
                 fresh directory"
            ))),
            Err(_) => Err(GraphError::SchemaVersion(format!(
                "graph store at {path} has an unreadable schema version ({} bytes, \
                 expected 8); this build reads version {SCHEMA_VERSION}. Migrate the \
                 store or point --data-dir at a fresh directory",
                raw.len()
            ))),
        }
    }

    // ── Messages ────────────────────────────────────

    pub fn put_message(&self, msg: &MessageNode) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_MESSAGES).unwrap();
        let val = serde_json::to_vec(msg)?;
        self.db
            .put_cf(&cf, scoped_key(&msg.scope, msg.id.as_bytes()), &val)?;
        Ok(())
    }

    pub fn get_message(
        &self,
        scope: &SessionScope,
        id: &str,
    ) -> Result<Option<MessageNode>, GraphError> {
        let cf = self.db.cf_handle(CF_MESSAGES).unwrap();
        match self.db.get_cf(&cf, scoped_key(scope, id.as_bytes()))? {
            Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
            None => Ok(None),
        }
    }

    pub fn delete_message(&self, scope: &SessionScope, id: &str) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_MESSAGES).unwrap();
        self.db.delete_cf(&cf, scoped_key(scope, id.as_bytes()))?;
        Ok(())
    }

    pub fn all_messages(&self) -> Result<Vec<MessageNode>, GraphError> {
        let cf = self.db.cf_handle(CF_MESSAGES).unwrap();
        let mut out = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, v) = item?;
            out.push(serde_json::from_slice(&v)?);
        }
        Ok(out)
    }

    // ── Computations ────────────────────────────────

    /// Make subsequent computation/edge writes and graph batch commits fail.
    /// Tests only; staging a batch remains possible to exercise commit failure.
    #[cfg(test)]
    pub fn set_fail_writes_for_test(&self, fail: bool) {
        self.fail_writes
            .store(fail, std::sync::atomic::Ordering::SeqCst);
    }

    /// `Err` while the test-only write fault is armed, `Ok` otherwise.
    #[cfg(test)]
    fn write_fault(&self) -> Result<(), GraphError> {
        if self.fail_writes.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(GraphError::InvalidEdge("injected write fault".into()));
        }
        Ok(())
    }

    #[cfg(not(test))]
    #[inline]
    fn write_fault(&self) -> Result<(), GraphError> {
        Ok(())
    }

    pub fn put_computation(&self, comp: &ComputationNode) -> Result<(), GraphError> {
        self.write_fault()?;
        let cf = self.db.cf_handle(CF_COMPUTATIONS).unwrap();
        let val = serde_json::to_vec(comp)?;
        self.db
            .put_cf(&cf, scoped_key(&comp.scope, comp.span_id.as_bytes()), &val)?;
        Ok(())
    }

    pub fn get_computation(
        &self,
        scope: &SessionScope,
        span_id: &str,
    ) -> Result<Option<ComputationNode>, GraphError> {
        let cf = self.db.cf_handle(CF_COMPUTATIONS).unwrap();
        match self.db.get_cf(&cf, scoped_key(scope, span_id.as_bytes()))? {
            Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
            None => Ok(None),
        }
    }

    pub fn delete_computation(
        &self,
        scope: &SessionScope,
        span_id: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_COMPUTATIONS).unwrap();
        self.db
            .delete_cf(&cf, scoped_key(scope, span_id.as_bytes()))?;
        Ok(())
    }

    pub fn all_computations(&self) -> Result<Vec<ComputationNode>, GraphError> {
        let cf = self.db.cf_handle(CF_COMPUTATIONS).unwrap();
        let mut out = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, v) = item?;
            out.push(serde_json::from_slice(&v)?);
        }
        Ok(out)
    }

    // ── Edges ───────────────────────────────────────

    pub fn put_edge(&self, key: &EdgeKey, data: &EdgeData) -> Result<(), GraphError> {
        self.write_fault()?;
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let k = scoped_key(&data.scope, &key.to_bytes());
        let v = serde_json::to_vec(data)?;
        self.db.put_cf(&cf, &k, &v)?;
        Ok(())
    }

    pub fn delete_edge(&self, scope: &SessionScope, key: &EdgeKey) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        self.db.delete_cf(&cf, scoped_key(scope, &key.to_bytes()))?;
        Ok(())
    }

    /// Delete every edge persisted under `scope`. Walks the edges
    /// CF with a prefix scan over the encoded scope bytes (the
    /// length-prefix encoding makes the scope a valid prefix). Used
    /// by `drop_session` so a teardown leaves no orphan edge rows
    /// on disk to bloat restart scans (`all_edges()` would still
    /// walk them otherwise).
    pub fn delete_edges_for_scope(&self, scope: &SessionScope) -> Result<usize, GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let prefix = scope.to_storage_bytes();
        let iter = self.db.prefix_iterator_cf(&cf, prefix.as_slice());
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for item in iter {
            let (k, _) = item?;
            // `prefix_iterator_cf` may return entries beyond the
            // strict prefix; double-check.
            if k.starts_with(prefix.as_slice()) {
                keys.push(k.to_vec());
            } else {
                break;
            }
        }
        let n = keys.len();
        for k in keys {
            self.db.delete_cf(&cf, &k)?;
        }
        Ok(n)
    }

    pub fn all_edges(&self) -> Result<Vec<(EdgeKey, EdgeData)>, GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let mut out = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (k, v) = item?;
            let data: EdgeData = serde_json::from_slice(&v)?;
            // Edge keys are `[scope_bytes][EdgeKey JSON]`. The scope
            // is persisted inside the value too, so we recompute its
            // encoded length and strip exactly that many leading
            // bytes to recover the inner EdgeKey.
            let scope_bytes = data.scope.to_storage_bytes();
            if k.len() < scope_bytes.len() {
                return Err(GraphError::InvalidEdge(
                    "edge key shorter than its scope prefix".to_string(),
                ));
            }
            let key = EdgeKey::from_bytes(&k[scope_bytes.len()..])?;
            out.push((key, data));
        }
        Ok(out)
    }

    // ── Meta ────────────────────────────────────────

    pub fn get_sequence(&self) -> Result<u64, GraphError> {
        let cf = self.db.cf_handle(CF_META).unwrap();
        match self.db.get_cf(&cf, b"sequence")? {
            Some(v) => {
                let arr: [u8; 8] = <[u8] as AsRef<[u8]>>::as_ref(&v)
                    .try_into()
                    .unwrap_or([0; 8]);
                Ok(u64::from_le_bytes(arr))
            }
            None => Ok(0),
        }
    }

    pub fn set_sequence(&self, seq: u64) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_META).unwrap();
        self.db.put_cf(&cf, b"sequence", seq.to_le_bytes())?;
        Ok(())
    }

    // ── Policy bindings (per-session pin → content hash) ────

    /// Record that `scope` is pinned to the policy identified by
    /// `content_hash`. Idempotent; overwrites any prior binding for
    /// the same scope.
    pub fn put_policy_binding(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_BINDINGS).unwrap();
        self.db
            .put_cf(&cf, scope.to_storage_bytes(), content_hash.as_bytes())?;
        Ok(())
    }

    /// Drop a session's policy binding. No-op if absent.
    pub fn delete_policy_binding(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_BINDINGS).unwrap();
        self.db.delete_cf(&cf, scope.to_storage_bytes())?;
        Ok(())
    }

    /// Read one scope's persisted binding, if it has one.
    ///
    /// A point lookup rather than a filtered [`Self::all_policy_bindings`]:
    /// the caller needs the prior value to restore if the bind it is about
    /// to publish fails, and a bind happens once per session — scanning
    /// every binding in the store each time is not affordable for a
    /// workload that opens a session per task.
    pub fn get_policy_binding(&self, scope: &SessionScope) -> Result<Option<String>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_BINDINGS).unwrap();
        let Some(v) = self.db.get_cf(&cf, scope.to_storage_bytes())? else {
            return Ok(None);
        };
        Ok(Some(
            std::str::from_utf8(&v)
                .map_err(|_| GraphError::InvalidEdge("policy_binding value not utf-8".into()))?
                .to_string(),
        ))
    }

    /// Read every persisted `(scope, content_hash)` binding. Used
    /// at boot to rebuild `session_to_policy` after a restart.
    pub fn all_policy_bindings(&self) -> Result<Vec<(SessionScope, String)>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_BINDINGS).unwrap();
        let mut out = Vec::new();
        for item in self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
            let (k, v) = item?;
            let scope = SessionScope::from_storage_bytes(&k).ok_or_else(|| {
                GraphError::InvalidEdge("policy_binding key not a valid SessionScope".into())
            })?;
            let hash = std::str::from_utf8(&v)
                .map_err(|_| GraphError::InvalidEdge("policy_binding value not utf-8".into()))?
                .to_string();
            out.push((scope, hash));
        }
        Ok(out)
    }

    // ── Tenant default policy (per-tenant fallback) ─────────

    /// Set the per-tenant default policy. Newly-spawned unpinned
    /// sessions resolve to this hash.
    pub fn put_tenant_default_policy(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_DEFAULTS).unwrap();
        self.db
            .put_cf(&cf, tenant.as_bytes(), content_hash.as_bytes())?;
        Ok(())
    }

    /// Read one tenant's persisted default, if it has one. Point lookup,
    /// for the same reason as [`Self::get_policy_binding`]: a rollout that
    /// fails after this row was overwritten has to put the old value back.
    pub fn get_tenant_default_policy(&self, tenant: &str) -> Result<Option<String>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_DEFAULTS).unwrap();
        let Some(v) = self.db.get_cf(&cf, tenant.as_bytes())? else {
            return Ok(None);
        };
        Ok(Some(
            std::str::from_utf8(&v)
                .map_err(|_| GraphError::InvalidEdge("tenant default value not utf-8".into()))?
                .to_string(),
        ))
    }

    /// Drop a tenant's persisted default. No-op if absent. Used to restore
    /// "no default was recorded" when a rollout is rolled back.
    pub fn delete_tenant_default_policy(&self, tenant: &str) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_DEFAULTS).unwrap();
        self.db.delete_cf(&cf, tenant.as_bytes())?;
        Ok(())
    }

    /// Read every persisted `(tenant, content_hash)` default.
    pub fn all_tenant_defaults(&self) -> Result<Vec<(String, String)>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_DEFAULTS).unwrap();
        let mut out = Vec::new();
        for item in self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
            let (k, v) = item?;
            let tenant = std::str::from_utf8(&k)
                .map_err(|_| GraphError::InvalidEdge("tenant key not utf-8".into()))?
                .to_string();
            let hash = std::str::from_utf8(&v)
                .map_err(|_| GraphError::InvalidEdge("tenant default value not utf-8".into()))?
                .to_string();
            out.push((tenant, hash));
        }
        Ok(out)
    }

    // ── Policy source store (content_hash → source bundle) ──

    /// Storage key for one persisted source record.
    ///
    /// The admission class is PART OF THE KEY, so the two classes are
    /// independent records rather than one row that the most recent uploader
    /// gets to rewrite. [`FunctorAdmission::User`] keeps the bare content hash
    /// — that is where records written before this field existed already live,
    /// and [`FunctorAdmission::User`] is exactly how they deserialize — while
    /// [`FunctorAdmission::Admin`] gets a suffixed key. A content hash is hex,
    /// so no hash can collide with a suffixed key.
    fn policy_source_key(content_hash: &str, admission: FunctorAdmission) -> Vec<u8> {
        let mut k = content_hash.as_bytes().to_vec();
        if admission == FunctorAdmission::Admin {
            k.extend_from_slice(b"\0admin");
        }
        k
    }

    /// Store the source bytes that hash to `content_hash`, under the class
    /// they were admitted in.
    ///
    /// Write-if-absent-or-identical within a class: a second write of the
    /// same `(content_hash, class)` carrying the same bytes is a no-op, and
    /// one carrying DIFFERENT bytes is refused (see
    /// [`Self::put_policy_source_in_class`]). ACROSS classes it writes a
    /// second, independent record; it never rewrites the other class's
    /// record. That is the whole mechanism behind the invariant the load
    /// paths rely on — nothing an uploader does can widen a record already
    /// stored as user-supplied into an admin-supplied one, and nothing can
    /// quietly erase an admin's record either.
    pub fn put_policy_source(
        &self,
        content_hash: &str,
        value: &PersistedPolicy,
    ) -> Result<(), GraphError> {
        self.put_policy_source_in_class(content_hash, value.functor_admission, value)
    }

    /// Store `value` under `(content_hash, admission)` explicitly.
    ///
    /// The KEY is what decides the class a record is read back in — see
    /// [`Self::policy_sources_by_least_privilege`] — so this is the write that
    /// says which class is meant. [`Self::put_policy_source`] is the ordinary
    /// caller: it takes the class from the value, which is where an upload
    /// path has it. The two disagree only for a record written by something
    /// other than this API, and the key wins.
    ///
    /// **Non-destructive.** The key is a content hash, so an occupied key
    /// already holds the bytes that hash to it: this writes when the key is
    /// free, does nothing when the stored record carries the same hashed
    /// content (source, functor source and backend), and REFUSES with
    /// [`GraphError::ContentHashAmbiguity`] when it carries different
    /// content, leaving the stored record untouched. Two contents under one
    /// hash means either a collision or a caller writing under a key it did
    /// not derive from these bytes, and neither is a reason to replace a
    /// record other policies may be bound to. The rule lives here, not only
    /// in the callers, so no future caller can overwrite a stored source —
    /// and because this key is NOT tenant-scoped, that is also what stops one
    /// tenant's upload from replacing another tenant's source.
    ///
    /// The record's own `functor_admission` field is not part of the hashed
    /// content and is not compared: a re-write that differs only there keeps
    /// the stored record, which the load paths do not read that field from
    /// anyway (the key decides the class).
    pub fn put_policy_source_in_class(
        &self,
        content_hash: &str,
        admission: FunctorAdmission,
        value: &PersistedPolicy,
    ) -> Result<(), GraphError> {
        if let Some(stored) = self.get_policy_source_in_class(content_hash, admission)? {
            if stored.policy_source == value.policy_source
                && stored.functor_source == value.functor_source
                && stored.backend == value.backend
            {
                return Ok(());
            }
            return Err(GraphError::ContentHashAmbiguity(format!(
                "policy source {content_hash} (class {admission:?}) is already stored with \
                 different content (stored: {} bytes of policy / {} of functor source, \
                 backend {}; offered: {} / {}, backend {}); refusing to overwrite it",
                stored.policy_source.len(),
                stored.functor_source.len(),
                stored.backend,
                value.policy_source.len(),
                value.functor_source.len(),
                value.backend,
            )));
        }
        let cf = self.db.cf_handle(CF_POLICY_SOURCE).unwrap();
        let v = serde_json::to_vec(value)?;
        self.db
            .put_cf(&cf, Self::policy_source_key(content_hash, admission), &v)?;
        Ok(())
    }

    /// Read the record stored for `content_hash` in exactly `admission`, if
    /// any. This is the lookup a load path uses once it knows which class it
    /// is willing to load.
    pub fn get_policy_source_in_class(
        &self,
        content_hash: &str,
        admission: FunctorAdmission,
    ) -> Result<Option<PersistedPolicy>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_SOURCE).unwrap();
        match self
            .db
            .get_cf(&cf, Self::policy_source_key(content_hash, admission))?
        {
            Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
            None => Ok(None),
        }
    }

    /// Every record stored for `content_hash`, least-privileged class first,
    /// each paired with THE CLASS ITS KEY ENCODES.
    ///
    /// The key is authoritative. A record's own `functor_admission` field is
    /// kept for compatibility and is what an ordinary write derives the key
    /// from, but a load path must gate on the class returned here: that is the
    /// class the record was actually found under, and the one the key split
    /// makes unforgeable by a later uploader. Gating on the value instead
    /// would leave the split constraining writes only.
    ///
    /// Callers that must pick one walk this in order and take the first their
    /// settings admit, so a source that exists in both classes loads as the
    /// user-supplied one while the opt-in still admits that, and the admin
    /// record is reached only when the user record is refused or absent.
    pub fn policy_sources_by_least_privilege(
        &self,
        content_hash: &str,
    ) -> Result<Vec<(FunctorAdmission, PersistedPolicy)>, GraphError> {
        let mut out = Vec::new();
        for class in [FunctorAdmission::User, FunctorAdmission::Admin] {
            if let Some(p) = self.get_policy_source_in_class(content_hash, class)? {
                out.push((class, p));
            }
        }
        Ok(out)
    }

    /// Read a source bundle for `content_hash`, if any is stored in either
    /// class, least-privileged first. An existence check; a load path that
    /// cares which class it gets asks
    /// [`Self::policy_sources_by_least_privilege`] instead.
    pub fn get_policy_source(
        &self,
        content_hash: &str,
    ) -> Result<Option<PersistedPolicy>, GraphError> {
        Ok(self
            .policy_sources_by_least_privilege(content_hash)?
            .into_iter()
            .next()
            .map(|(_, p)| p))
    }

    // ── Policy metadata store (content_hash → config facts) ──

    /// Store one tenant's config for `content_hash`.
    ///
    /// Keyed by tenant as well as hash: a policy source is not private to a
    /// tenant, so keying by the hash alone made this row database-wide and let
    /// one tenant's configuration decide another tenant's sessions.
    pub fn put_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
        facts: &[PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_METADATA).unwrap();
        let v = serde_json::to_vec(facts)?;
        self.db
            .put_cf(&cf, Self::tenant_metadata_key(tenant, content_hash), &v)?;
        Ok(())
    }

    /// Read one tenant's config for `content_hash`. Absent → empty (a
    /// policy with no config is the common case, not an error).
    ///
    /// Falls back to the pre-tenant-scoping key on a miss. Rows written before
    /// this store keyed by tenant are under the bare content hash, and a
    /// database carried across that change still holds them. Without the
    /// fallback every one of those rows reads as "no config", which does not
    /// fail safe: a metadata-gated rule with no facts matches nothing, so the
    /// policy silently stops enforcing.
    ///
    /// A row under the shared policy-wide key is not tenant-scoped, so serving
    /// it hands one tenant a row any tenant could have written. That is
    /// preferable to a silent loss of enforcement. Each such read is logged so
    /// an operator can see which policies still need re-uploading, and the
    /// first tenant-scoped write for that hash supersedes it for that tenant.
    pub fn get_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<Vec<PolicyMetadataFact>, GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_METADATA).unwrap();
        if let Some(v) = self
            .db
            .get_cf(&cf, Self::tenant_metadata_key(tenant, content_hash))?
        {
            return Ok(serde_json::from_slice(&v)?);
        }
        match self.db.get_cf(&cf, content_hash.as_bytes())? {
            Some(v) => {
                tracing::warn!(
                    tenant,
                    content_hash,
                    "serving policy config from a pre-tenant-scoped row; \
                     re-upload this policy to key it by tenant"
                );
                Ok(serde_json::from_slice(&v)?)
            }
            None => Ok(Vec::new()),
        }
    }

    // ── Session metadata store (scope → dynamic config facts) ──

    /// Append session-scoped dynamic facts (append-only; exact duplicates are
    /// skipped to keep the set small). The read-modify-write runs under
    /// `metadata_lock` so concurrent appends can't clobber each other.
    pub fn append_session_metadata(
        &self,
        scope: &SessionScope,
        facts: &[PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        if facts.is_empty() {
            return Ok(());
        }
        // Serialize the whole RMW: without it, two concurrent appends both read
        // the old value and the second put overwrites the first's fact.
        let _guard = self.metadata_lock.lock();
        let cf = self.db.cf_handle(CF_SESSION_METADATA).unwrap();
        let key = scope.to_storage_bytes();
        let mut existing: Vec<PolicyMetadataFact> = match self.db.get_cf(&cf, &key)? {
            Some(v) => serde_json::from_slice(&v)?,
            None => Vec::new(),
        };
        for f in facts {
            if !existing
                .iter()
                .any(|e| e.rel == f.rel && e.a == f.a && e.b == f.b)
            {
                existing.push(f.clone());
            }
        }
        self.db.put_cf(&cf, &key, &serde_json::to_vec(&existing)?)?;
        Ok(())
    }

    /// Drop a session's dynamic metadata facts. No-op if absent. Called from
    /// `EndSession` so a recycled session id doesn't inherit a prior principal's
    /// detaint decisions (a fail-open) and to bound disk growth.
    pub fn delete_session_metadata(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let _guard = self.metadata_lock.lock();
        let cf = self.db.cf_handle(CF_SESSION_METADATA).unwrap();
        self.db.delete_cf(&cf, scope.to_storage_bytes())?;
        Ok(())
    }

    /// Read the dynamic facts for `scope`. Absent → empty.
    pub fn get_session_metadata(
        &self,
        scope: &SessionScope,
    ) -> Result<Vec<PolicyMetadataFact>, GraphError> {
        let cf = self.db.cf_handle(CF_SESSION_METADATA).unwrap();
        match self.db.get_cf(&cf, scope.to_storage_bytes())? {
            Some(v) => Ok(serde_json::from_slice(&v)?),
            None => Ok(Vec::new()),
        }
    }

    // ── Policy-wide config ((tenant, content_hash) → config), then the
    //    per-binding store ((scope, content_hash) → config) ──

    /// Write a row under the PRE-tenant-scoping key, for tests that need to
    /// stand in for a database carried across that change.
    #[cfg(test)]
    pub(crate) fn put_legacy_policy_metadata(
        &self,
        content_hash: &str,
        facts: &[PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_METADATA).unwrap();
        let v = serde_json::to_vec(facts)?;
        self.db.put_cf(&cf, content_hash.as_bytes(), &v)?;
        Ok(())
    }

    /// Key a tenant's policy-wide config. Length-prefixing the tenant keeps
    /// `("a", "bc")` distinct from `("ab", "c")`.
    fn tenant_metadata_key(tenant: &str, content_hash: &str) -> Vec<u8> {
        let t = tenant.as_bytes();
        let mut key = Vec::with_capacity(4 + t.len() + content_hash.len());
        key.extend_from_slice(&(t.len() as u32).to_be_bytes());
        key.extend_from_slice(t);
        key.extend_from_slice(content_hash.as_bytes());
        key
    }

    /// Key a binding's config by scope *then* content hash.
    /// [`SessionScope::to_storage_bytes`] length-prefixes both of its
    /// components, so the encoded scope ends at a known offset and appending
    /// the hash is unambiguous. Scope-first is what makes
    /// [`Self::delete_binding_metadata`] a single prefix scan.
    fn binding_metadata_key(scope: &SessionScope, content_hash: &str) -> Vec<u8> {
        let mut key = scope.to_storage_bytes();
        key.extend_from_slice(content_hash.as_bytes());
        key
    }

    /// Store the config supplied on a session-scoped `SetPolicy`.
    ///
    /// Unlike [`Self::append_session_metadata`] this *replaces*: a re-pin of the
    /// same binding carrying a changed value must not leave the old one behind,
    /// because a value fact (`cooldown_days`) is not a set member.
    ///
    /// An empty fact list is stored, not skipped, and that is load-bearing. The
    /// record's *presence* is what says "this session pinned itself and its
    /// configuration is exactly this" — which is what suppresses the fallback to
    /// the shared policy-wide key. Without it a session that pinned with no
    /// config would inherit whatever that shared key happens to hold, including
    /// a relaxation left there by some other session.
    pub fn put_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
        facts: &[PolicyMetadataFact],
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        let v = serde_json::to_vec(facts)?;
        self.db
            .put_cf(&cf, Self::binding_metadata_key(scope, content_hash), &v)?;
        Ok(())
    }

    /// Drop one tenant's config for one policy. Used to undo a write made
    /// for a rollout that then failed to publish; a point delete, because the
    /// row for a *different* policy in the same tenant must survive.
    pub fn delete_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_POLICY_METADATA).unwrap();
        self.db
            .delete_cf(&cf, Self::tenant_metadata_key(tenant, content_hash))?;
        Ok(())
    }

    /// Read the config for one binding.
    ///
    /// `None` means this session never pinned itself to this policy, and is the
    /// signal to fall back to the policy-wide [`CF_POLICY_METADATA`] entry.
    /// `Some(vec![])` means it pinned itself and named no config — a different
    /// thing, and *not* an invitation to inherit another session's.
    pub fn get_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<Option<Vec<PolicyMetadataFact>>, GraphError> {
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        match self
            .db
            .get_cf(&cf, Self::binding_metadata_key(scope, content_hash))?
        {
            Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
            None => Ok(None),
        }
    }

    /// Drop the config for ONE binding. [`Self::delete_binding_metadata`]
    /// removes every policy's row for a scope, which is right for a teardown
    /// and wrong for undoing a single write: a session that is validly pinned
    /// to another policy must keep that policy's config.
    pub fn delete_binding_metadata_for_policy(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        self.db
            .delete_cf(&cf, Self::binding_metadata_key(scope, content_hash))?;
        Ok(())
    }

    /// Drop every binding's config for one tenant.
    ///
    /// Used by a force rollout. Dropping the in-memory binding is enough
    /// whenever the new default is different policy content — the session
    /// re-binds to another hash and its stored row is never read again — but
    /// NOT when the same content is force-promoted: the re-bound hash matches
    /// the stored row, the metadata assembly finds it and takes the "this
    /// session pinned itself" branch, and the pre-rollout config is served
    /// from then on.
    ///
    /// Keyed by tenant rather than by the scopes that happened to have a live
    /// evaluator, because an idle session has a stored row and no evaluator,
    /// and it would resume on the stale config. `SessionScope::to_storage_bytes`
    /// length-prefixes the tenant first, so the tenant is an unambiguous prefix
    /// of every scope key beneath it.
    /// Returns the rows it removed, so a rollout that clears them and then
    /// fails to publish can put them back exactly as they were. Nothing else
    /// can reconstruct them: the config is supplied by the client on each
    /// bind and is not derivable from the policy or the graph.
    pub fn clear_tenant_binding_metadata(
        &self,
        tenant: &str,
    ) -> Result<ClearedBindingMetadata, GraphError> {
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        let t = tenant.as_bytes();
        let mut prefix = Vec::with_capacity(4 + t.len());
        prefix.extend_from_slice(&(t.len() as u32).to_be_bytes());
        prefix.extend_from_slice(t);
        let mut batch = WriteBatch::default();
        let mut removed = Vec::new();
        let iter = self.db.prefix_iterator_cf(&cf, &prefix);
        for item in iter {
            let (key, value) = item?;
            // `prefix_iterator_cf` seeks, it does not filter, so it can hand
            // back keys past the prefix once it is exhausted.
            if !key.starts_with(&prefix) {
                break;
            }
            batch.delete_cf(&cf, &key);
            removed.push((key, value));
        }
        // One batch: a mid-scan failure deletes nothing, so `removed` either
        // describes exactly what went away or the call failed having changed
        // nothing.
        self.db.write(batch)?;
        Ok(ClearedBindingMetadata(removed))
    }

    /// Put back rows taken by [`Self::clear_tenant_binding_metadata`].
    pub fn restore_binding_metadata(
        &self,
        rows: &ClearedBindingMetadata,
    ) -> Result<(), GraphError> {
        if rows.is_empty() {
            return Ok(());
        }
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        let mut batch = WriteBatch::default();
        for (key, value) in &rows.0 {
            batch.put_cf(&cf, key, value);
        }
        self.db.write(batch)?;
        Ok(())
    }

    /// Drop a session's pin, its dynamic facts and every binding's config for
    /// it, in ONE batch.
    ///
    /// Atomic on purpose. Done as separate deletes, a failure partway leaves
    /// the session with some of its state gone and some retained — e.g. a pin
    /// removed while its accumulated detaint approvals survive, so a restart
    /// drops the session to the tenant default while keeping the relaxations
    /// that were granted under the stricter pinned policy.
    ///
    /// The session *owner* is deliberately not part of this: releasing the id
    /// is what lets another principal claim it, and that has to happen after
    /// the live evaluator is gone, not alongside the durable rows.
    pub fn delete_session_state(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let _guard = self.metadata_lock.lock();
        let bindings = self.db.cf_handle(CF_POLICY_BINDINGS).unwrap();
        let session_meta = self.db.cf_handle(CF_SESSION_METADATA).unwrap();
        let binding_meta = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        let key = scope.to_storage_bytes();
        let mut batch = WriteBatch::default();
        batch.delete_cf(&bindings, &key);
        batch.delete_cf(&session_meta, &key);
        let iter = self.db.prefix_iterator_cf(&binding_meta, &key);
        for item in iter {
            let (k, _) = item?;
            if !k.starts_with(&key[..]) {
                break;
            }
            batch.delete_cf(&binding_meta, k);
        }
        self.db.write(batch)?;
        Ok(())
    }

    /// Drop every binding's config for `scope`, whatever policy each was bound
    /// to. Called from `EndSession` alongside the other per-session state, so a
    /// recycled session id cannot inherit a prior principal's configuration and
    /// the CF does not grow without bound.
    pub fn delete_binding_metadata(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_BINDING_METADATA).unwrap();
        let prefix = scope.to_storage_bytes();
        let mut batch = WriteBatch::default();
        let iter = self.db.prefix_iterator_cf(&cf, &prefix);
        for item in iter {
            let (key, _) = item?;
            // `prefix_iterator_cf` may hand back keys past the prefix once the
            // prefix is exhausted (it seeks, it does not filter), so check.
            if !key.starts_with(&prefix) {
                break;
            }
            batch.delete_cf(&cf, key);
        }
        self.db.write(batch)?;
        Ok(())
    }

    // ── Session owners (tenant, session) → principal ────────

    /// Record `principal` as the owner of `scope`. Idempotent in the
    /// sense that re-writes overwrite; the caller is responsible for
    /// the first-write semantics (read existing, write only if
    /// absent or matches).
    pub fn put_session_owner(
        &self,
        scope: &SessionScope,
        principal: &str,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_SESSION_OWNERS).unwrap();
        self.db
            .put_cf(&cf, scope.to_storage_bytes(), principal.as_bytes())?;
        Ok(())
    }

    /// Look up `scope`'s recorded owner, if any.
    pub fn get_session_owner(&self, scope: &SessionScope) -> Result<Option<String>, GraphError> {
        let cf = self.db.cf_handle(CF_SESSION_OWNERS).unwrap();
        match self.db.get_cf(&cf, scope.to_storage_bytes())? {
            Some(v) => Ok(Some(
                std::str::from_utf8(&v)
                    .map_err(|_| GraphError::InvalidEdge("owner value not utf-8".into()))?
                    .to_string(),
            )),
            None => Ok(None),
        }
    }

    /// Drop a session's recorded owner. No-op if absent. Called from
    /// `EndSession` so a recycled session id can be claimed by the
    /// next principal that touches it.
    pub fn delete_session_owner(&self, scope: &SessionScope) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_SESSION_OWNERS).unwrap();
        self.db.delete_cf(&cf, scope.to_storage_bytes())?;
        Ok(())
    }

    // ── WriteBatch: atomic multi-key writes ─────────
    //
    // Building one WriteBatch per `merge_*` call collapses what
    // were 2N independent ``put_cf`` syscalls (per message + per
    // sequence-marker) into a single rocksdb commit. RocksDB's
    // WAL writes the batch atomically, so a crash either commits
    // every key in the batch or none — exactly what we want for
    // ``messages + edges + final-sequence`` arriving together.

    /// Start a new empty [`WriteBatch`]. Caller fills it via the
    /// ``batch_*`` helpers below and finalizes via [`Self::commit_batch`].
    pub fn new_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    pub fn batch_put_message(
        &self,
        batch: &mut WriteBatch,
        msg: &MessageNode,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_MESSAGES).unwrap();
        let val = serde_json::to_vec(msg)?;
        batch.put_cf(&cf, scoped_key(&msg.scope, msg.id.as_bytes()), &val);
        Ok(())
    }

    pub fn batch_put_edge(
        &self,
        batch: &mut WriteBatch,
        key: &EdgeKey,
        data: &EdgeData,
    ) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let k = scoped_key(&data.scope, &key.to_bytes());
        let v = serde_json::to_vec(data)?;
        batch.put_cf(&cf, &k, &v);
        Ok(())
    }

    pub fn batch_set_sequence(&self, batch: &mut WriteBatch, seq: u64) {
        let cf = self.db.cf_handle(CF_META).unwrap();
        batch.put_cf(&cf, b"sequence", seq.to_le_bytes());
    }

    /// Atomically commit a [`WriteBatch`].
    pub fn commit_batch(&self, batch: WriteBatch) -> Result<(), GraphError> {
        self.write_fault()?;
        self.db.write(batch)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A policy source bundle that carries no admission field reads back as
    /// user-admitted, the least privileged of the two values. Nothing is known
    /// about who supplied the C++ in such a record, and treating it as
    /// admin-admitted would let a store carry source past a gate that never
    /// saw it.
    #[test]
    fn a_persisted_policy_without_the_admission_field_reads_back_as_user() {
        let legacy = r#"{"policy_source":"IsAuthorized(0).",
                         "functor_source":"extern \"C\" int f() { return 0; }",
                         "backend":"souffle"}"#;
        let parsed: PersistedPolicy = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.functor_admission, FunctorAdmission::User);
        assert_eq!(FunctorAdmission::default(), FunctorAdmission::User);
    }

    /// The field survives a write/read round trip in both values, so a policy
    /// an admin uploaded is still recognisable as one after a restart.
    #[test]
    fn the_admission_field_round_trips() {
        for admission in [FunctorAdmission::Admin, FunctorAdmission::User] {
            let bundle = PersistedPolicy {
                policy_source: "IsAuthorized(0).".into(),
                functor_source: "extern \"C\" int f() { return 0; }".into(),
                backend: "souffle".into(),
                functor_admission: admission,
            };
            let encoded = serde_json::to_vec(&bundle).unwrap();
            let decoded: PersistedPolicy = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded.functor_admission, admission);
        }
    }

    fn version_in(store: &RocksStore) -> Option<u64> {
        let cf = store.db.cf_handle(CF_META).unwrap();
        store
            .db
            .get_cf(&cf, SCHEMA_VERSION_KEY)
            .unwrap()
            .map(|v| u64::from_le_bytes(<[u8; 8]>::try_from(v.as_slice()).unwrap()))
    }

    /// A store that carries no version — a fresh directory, or one written
    /// before the stamp existed — is stamped with the current one on open, so
    /// the next build to change the format has something to compare against.
    #[test]
    fn opening_a_store_without_a_version_stamps_the_current_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph");
        let path = path.to_str().unwrap();

        let store = RocksStore::open(path).expect("a fresh store opens");
        assert_eq!(version_in(&store), Some(SCHEMA_VERSION));

        // And re-opening an already-stamped store is a no-op, not a refusal.
        drop(store);
        let store = RocksStore::open(path).expect("a stamped store re-opens");
        assert_eq!(version_in(&store), Some(SCHEMA_VERSION));
    }

    /// A store written by a build with a different on-disk format is refused
    /// at open. Reading it would deserialize every value through the wrong
    /// shape, and the operator would see the damage as missing policy
    /// bindings rather than as a startup error.
    #[test]
    fn opening_a_store_from_another_schema_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph");
        let path = path.to_str().unwrap();

        let store = RocksStore::open(path).expect("a fresh store opens");
        let cf = store.db.cf_handle(CF_META).unwrap();
        store
            .db
            .put_cf(&cf, SCHEMA_VERSION_KEY, 2u64.to_le_bytes())
            .unwrap();
        drop(store); // release the RocksDB directory lock

        let msg = match RocksStore::open(path) {
            Ok(_) => panic!("a version this build cannot read must not open"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains(path), "the refusal must name the path: {msg}");
        assert!(
            msg.contains("version 2") && msg.contains("version 1"),
            "the refusal must name both versions: {msg}"
        );
    }
}
