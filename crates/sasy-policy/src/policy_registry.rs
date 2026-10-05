//! Per-tenant registry of policy variants.
//!
//! For each tenant, the registry holds:
//!
//! 1. A `(tenant, policy_id) → EvaluatorFactory` map of every
//!    uploaded policy that hasn't been pruned yet. Each factory
//!    spawns a fresh per-session evaluator subprocess against the
//!    same compiled program, so multiple sessions can share one
//!    policy variant without sharing state.
//! 2. A `tenant → PolicyId` "default" pointer naming the most
//!    recently installed *non-variant* policy for that tenant.
//!    New sessions that don't pin a `policy_id` adopt the default.
//!
//! The registry is intentionally per-tenant — a `policy_id`
//! installed by tenant `acme` is invisible to tenant `orgb`,
//! mirroring the (tenant, session) isolation invariant. Policy
//! ids are server-generated UUIDs so two tenants cannot collide
//! on the same id by accident or by design.
//!
//! Eviction: a TTL sweep drops policy entries that have had no
//! `last_referenced` activity for `SASY_POLICY_VARIANT_IDLE_TTL_SECS`
//! (default 1 hour). Default-pointed policies are never swept; this
//! ensures a tenant's "current" policy stays installed even if all
//! its sessions are idle.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

use crate::session_evaluator::EvaluatorFactory;

/// Server-assigned policy variant id. Opaque to clients —
/// generated as a UUIDv4 so two tenants can't collide on a
/// human-chosen string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PolicyId(String);

impl PolicyId {
    /// Generate a fresh policy id.
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Wrap an externally-supplied string. Used when looking up an
    /// existing policy by an id the client received earlier; the
    /// registry rejects unknown ids.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PolicyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One installed policy variant.
struct PolicyEntry {
    factory: EvaluatorFactory,
    backend_name: String,
    /// Updated whenever this policy is referenced (a session binds
    /// to it, the registry is asked for its factory, etc.). The
    /// orphan sweep uses this to prune unused variants.
    last_referenced: parking_lot::Mutex<Instant>,
    /// True iff this is the current per-tenant default. Defaults
    /// are exempt from the orphan sweep.
    is_default: bool,
}

/// Errors returned by [`PolicyRegistry`] operations.
#[derive(Debug, thiserror::Error)]
pub enum PolicyRegistryError {
    #[error("no policy registered for tenant '{0}'")]
    TenantUnknown(String),
    #[error("policy id '{policy_id}' not found in tenant '{tenant}'")]
    PolicyNotFound { tenant: String, policy_id: String },
}

/// Per-tenant policy variant registry.
pub struct PolicyRegistry {
    /// `(tenant, policy_id) → entry`. The tuple-key keeps tenants
    /// fully isolated by construction: a lookup with a foreign
    /// tenant string can never reach another tenant's policy even
    /// if both happen to mint UUID collisions.
    entries: RwLock<HashMap<(String, PolicyId), PolicyEntry>>,
    /// Per-tenant default policy id. Set by every non-variant
    /// `SetPolicy`; cleared if its target policy is removed.
    defaults: RwLock<HashMap<String, PolicyId>>,
    /// `(tenant, content_hash) → policy_id` index for upload dedup.
    /// Two uploads with byte-identical inputs (desugared source +
    /// functor source + word size + ...) collapse to the same
    /// `policy_id` so the registry doesn't bloat under repeated
    /// identical uploads across sessions.
    by_content: RwLock<HashMap<(String, String), PolicyId>>,
    /// `(tenant, profile_name) → policy_id` index for bind-by-name. Populated
    /// for baked curated profiles at startup (restricted build) via
    /// [`Self::register_name`]; lets a client bind an embedded policy by its
    /// human name without shipping the source or its content hash. Named
    /// profiles are also exempt from the orphan sweep (see [`Self::evict_orphans`]).
    by_name: RwLock<HashMap<(String, String), PolicyId>>,
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyRegistry {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            defaults: RwLock::new(HashMap::new()),
            by_content: RwLock::new(HashMap::new()),
            by_name: RwLock::new(HashMap::new()),
        }
    }

    /// Look up an existing variant by content hash. Returns `Some(id)`
    /// if the same `(tenant, content_hash)` pair has been installed
    /// before — even if the prior caller asked for a different
    /// `make_default` flag. Useful for upload-dedup callers that
    /// want to short-circuit before computing a fresh factory.
    pub fn lookup_by_content(&self, tenant: &str, content_hash: &str) -> Option<PolicyId> {
        self.by_content
            .read()
            .get(&(tenant.to_string(), content_hash.to_string()))
            .cloned()
    }

    /// Register a `(tenant, profile_name) → policy_id` mapping so a later
    /// `SetPolicy(bind_profile_name)` can resolve the name to its content hash
    /// and bind without source. Used for baked curated profiles at startup.
    pub fn register_name(&self, tenant: &str, name: &str, policy_id: &PolicyId) {
        self.by_name
            .write()
            .insert((tenant.to_string(), name.to_string()), policy_id.clone());
    }

    /// Resolve a baked profile name to its `policy_id` (== content hash).
    pub fn lookup_by_name(&self, tenant: &str, name: &str) -> Option<PolicyId> {
        self.by_name
            .read()
            .get(&(tenant.to_string(), name.to_string()))
            .cloned()
    }

    /// Install a freshly-compiled policy and register its content
    /// hash for dedup. If `(tenant, content_hash)` already exists,
    /// returns the existing `policy_id` and discards `factory`
    /// without re-inserting the entry: duplicate uploads share one slot.
    /// `make_default=true` is honored on a dedup hit too: it
    /// promotes the existing entry to the tenant default. The flag
    /// is only ever set for explicit Default/Force intent, so a bare
    /// `make_default=false` variant upload still cannot flip the
    /// default. This closes the TOCTOU race where a concurrent
    /// identical install lands the entry between a caller's
    /// dedup-check and its install call.
    pub fn install_dedup(
        &self,
        tenant: &str,
        content_hash: &str,
        factory: EvaluatorFactory,
        backend_name: String,
        make_default: bool,
    ) -> PolicyId {
        // Use the content hash as the PolicyId. This makes the id
        // stable across restarts: lazy boot rehydrates bindings as
        // `(scope → PolicyId::from_string(content_hash))`, and a
        // later live SetPolicy that recompiles the same source
        // dedups to the same id without inventing a fresh UUID
        // that would diverge from what the lazy path uses.
        let policy_id = PolicyId::from_string(content_hash);

        // Fast path: this content is already installed for the
        // tenant. Refresh recency; the supplied factory is discarded.
        let already_installed = {
            let entries = self.entries.read();
            if let Some(existing) = entries.get(&(tenant.to_string(), policy_id.clone())) {
                *existing.last_referenced.lock() = Instant::now();
                true
            } else {
                false
            }
        };
        if already_installed {
            if make_default {
                // Explicit Default/Force re-install of an
                // already-present hash: promote it now rather than
                // no-oping. `make_default` is only ever set for
                // explicit Default/Force intent (Session uploads pass
                // false), so this can't flip the default out from
                // under a bare variant upload. Closes the TOCTOU race
                // where a concurrent identical install lands the entry
                // between a caller's dedup-check and here, which would
                // otherwise drop the Default intent until restart. The
                // entry is present, so the only way promotion fails is
                // a concurrent sweep racing it away — that error is
                // benign, ignore it.
                let _ = self.promote_to_default(tenant, content_hash);
                debug!(
                    tenant,
                    policy_id = %policy_id,
                    "dedup hit: promoted existing entry to tenant default"
                );
            } else {
                debug!(
                    tenant,
                    policy_id = %policy_id,
                    "dedup hit: reusing existing policy_id"
                );
            }
            return policy_id;
        }

        let entry = PolicyEntry {
            factory,
            backend_name,
            last_referenced: parking_lot::Mutex::new(Instant::now()),
            is_default: make_default,
        };

        let mut entries = self.entries.write();

        if make_default {
            // Demote any prior default for this tenant. The old
            // policy stays addressable by id for in-flight sessions
            // that already bound to it, but loses the
            // default-immune flag and is sweep-eligible once its
            // last session ends.
            let mut defaults = self.defaults.write();
            if let Some(prev) = defaults.insert(tenant.to_string(), policy_id.clone()) {
                if prev != policy_id {
                    if let Some(prev_entry) = entries.get_mut(&(tenant.to_string(), prev.clone())) {
                        prev_entry.is_default = false;
                    }
                    debug!(
                        tenant,
                        new_default = %policy_id,
                        demoted = %prev,
                        "demoted prior default policy"
                    );
                }
            } else {
                debug!(tenant, default = %policy_id, "installed first default policy");
            }
        }

        entries.insert((tenant.to_string(), policy_id.clone()), entry);
        self.by_content.write().insert(
            (tenant.to_string(), content_hash.to_string()),
            policy_id.clone(),
        );
        info!(
            tenant,
            policy_id = %policy_id,
            make_default,
            "policy installed"
        );
        policy_id
    }

    /// Reserve `content_hash` as the tenant default *without*
    /// installing an entry. Used at boot to rehydrate the
    /// `(tenant → default_hash)` mapping from RocksDB before any
    /// session has triggered a lazy compile + install. The next
    /// dispatch under this default goes through
    /// [`SessionEvaluatorMap::ensure_installed`] which fetches the
    /// source bundle from the graph store and calls
    /// [`Self::install_dedup`].
    pub fn reserve_default(&self, tenant: &str, content_hash: &str) {
        let policy_id = PolicyId::from_string(content_hash);
        self.defaults.write().insert(tenant.to_string(), policy_id);
    }

    /// Promote an *already-installed* `content_hash` to the
    /// tenant default. Errors if the entry isn't present (the
    /// caller is responsible for ensuring an install — typically
    /// via the lazy `ensure_installed` path before this call).
    ///
    /// This is the explicit-intent promotion used by the SetPolicy
    /// dedup short-circuit, which returns before `install_dedup` is
    /// reached and so must apply the Default/Force side effect
    /// itself. (`install_dedup` with `make_default=true` now also
    /// promotes on a hit, but only the non-short-circuit install
    /// path reaches it.)
    pub fn promote_to_default(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<PolicyId, PolicyRegistryError> {
        let policy_id = PolicyId::from_string(content_hash);
        let mut entries = self.entries.write();
        let key = (tenant.to_string(), policy_id.clone());
        if !entries.contains_key(&key) {
            return Err(PolicyRegistryError::PolicyNotFound {
                tenant: tenant.to_string(),
                policy_id: policy_id.to_string(),
            });
        }
        // Mark the new entry as the default and demote the prior
        // one's `is_default` flag so the orphan sweep can prune it
        // when nothing references it.
        let mut defaults = self.defaults.write();
        if let Some(prev) = defaults.insert(tenant.to_string(), policy_id.clone()) {
            if prev != policy_id {
                if let Some(prev_entry) = entries.get_mut(&(tenant.to_string(), prev.clone())) {
                    prev_entry.is_default = false;
                }
            }
        }
        if let Some(new_entry) = entries.get_mut(&key) {
            new_entry.is_default = true;
            *new_entry.last_referenced.lock() = Instant::now();
        }
        Ok(policy_id)
    }

    /// Install a new policy for `tenant`. If `make_default` is
    /// true, the new policy becomes the tenant's default (replacing
    /// the previous default's privileged status — the old policy
    /// stays in the registry until orphan-swept). Returns the
    /// generated [`PolicyId`].
    ///
    /// Callers that want every existing session in `tenant` to
    /// re-bootstrap under the new policy must combine this with
    /// `SessionEvaluatorMap::evict_tenant` (or equivalent) — the
    /// registry doesn't reach into the per-session map directly.
    pub fn install(
        &self,
        tenant: &str,
        factory: EvaluatorFactory,
        backend_name: String,
        make_default: bool,
    ) -> PolicyId {
        let policy_id = PolicyId::generate();
        let entry = PolicyEntry {
            factory,
            backend_name,
            last_referenced: parking_lot::Mutex::new(Instant::now()),
            is_default: make_default,
        };

        let mut entries = self.entries.write();

        if make_default {
            // Demote any prior default for this tenant (it stays
            // accessible by id for in-flight sessions, but loses
            // the default-immune flag and is sweep-eligible once
            // its last session ends).
            let mut defaults = self.defaults.write();
            if let Some(prev) = defaults.insert(tenant.to_string(), policy_id.clone()) {
                if let Some(prev_entry) = entries.get_mut(&(tenant.to_string(), prev.clone())) {
                    prev_entry.is_default = false;
                }
                debug!(
                    tenant,
                    new_default = %policy_id,
                    demoted = %prev,
                    "demoted prior default policy"
                );
            } else {
                debug!(tenant, default = %policy_id, "installed first default policy");
            }
        }

        entries.insert((tenant.to_string(), policy_id.clone()), entry);
        info!(
            tenant,
            policy_id = %policy_id,
            make_default,
            "policy installed"
        );
        policy_id
    }

    /// Resolve `policy_id` (or the tenant default if `None`) to a
    /// factory + backend. Bumps the entry's `last_referenced`.
    pub fn resolve(
        &self,
        tenant: &str,
        policy_id: Option<&PolicyId>,
    ) -> Result<(EvaluatorFactory, String, PolicyId), PolicyRegistryError> {
        let resolved = match policy_id {
            Some(id) => id.clone(),
            None => self
                .defaults
                .read()
                .get(tenant)
                .cloned()
                .ok_or_else(|| PolicyRegistryError::TenantUnknown(tenant.to_string()))?,
        };

        let entries = self.entries.read();
        let entry = entries
            .get(&(tenant.to_string(), resolved.clone()))
            .ok_or_else(|| PolicyRegistryError::PolicyNotFound {
                tenant: tenant.to_string(),
                policy_id: resolved.to_string(),
            })?;
        *entry.last_referenced.lock() = Instant::now();
        Ok((
            Arc::clone(&entry.factory),
            entry.backend_name.clone(),
            resolved,
        ))
    }

    /// True iff `(tenant, policy_id)` exists in the registry.
    pub fn contains(&self, tenant: &str, policy_id: &PolicyId) -> bool {
        self.entries
            .read()
            .contains_key(&(tenant.to_string(), policy_id.clone()))
    }

    /// Current default policy id for `tenant`, if any.
    pub fn default_for(&self, tenant: &str) -> Option<PolicyId> {
        self.defaults.read().get(tenant).cloned()
    }

    /// Set `is_default=true` on `(tenant, policy_id)`'s entry
    /// without changing the `defaults` map. Used by the lazy-replay
    /// path: boot rehydration writes `defaults[tenant]` ahead of
    /// any entry, then the first dispatch lands here via
    /// [`Self::install_dedup`] with `make_default=false` — that
    /// keeps the existing default pointer intact but leaves the
    /// freshly-inserted entry with `is_default=false`. Without
    /// this fix-up, the entry is sweep-eligible by
    /// [`Self::evict_orphans`] even though it's the tenant default.
    /// No-op if the entry isn't present.
    pub fn mark_entry_default(&self, tenant: &str, policy_id: &PolicyId) {
        if let Some(entry) = self
            .entries
            .write()
            .get_mut(&(tenant.to_string(), policy_id.clone()))
        {
            entry.is_default = true;
        }
    }

    /// Drop policies whose `last_referenced` is older than
    /// `idle_ttl`. Defaults are exempt. Returns the number of
    /// entries removed. Intended to be called by a periodic sweep.
    ///
    /// `by_content` is purged in the same pass so a later
    /// `lookup_by_content` doesn't return a stale `PolicyId` that
    /// resolves to a removed entry — that would cause the
    /// `SetPolicy` dedup short-circuit to skip recompilation and
    /// then fail to bind the session ("policy not found").
    pub fn evict_orphans(&self, idle_ttl: Duration) -> usize {
        let now = Instant::now();
        let mut entries = self.entries.write();
        // Named (curated/baked) profiles are pinned: they are a fixed set the
        // client may bind by name at any time, so they must never be swept (that
        // would dangle their by_name pointer and make them unbindable mid-run).
        let named: std::collections::HashSet<PolicyId> =
            self.by_name.read().values().cloned().collect();
        let stale: Vec<(String, PolicyId)> = entries
            .iter()
            .filter(|(k, e)| {
                !e.is_default
                    && !named.contains(&k.1)
                    && now.duration_since(*e.last_referenced.lock()) >= idle_ttl
            })
            .map(|(k, _)| k.clone())
            .collect();
        let n = stale.len();
        let stale_set: std::collections::HashSet<(String, PolicyId)> =
            stale.iter().cloned().collect();
        for k in &stale {
            entries.remove(k);
            debug!(tenant = %k.0, policy_id = %k.1, "orphan policy evicted");
        }
        drop(entries);

        // Purge `by_content` of pointers to the removed entries.
        // by_content keys on (tenant, content_hash) but the value
        // is the same `PolicyId` we just dropped — walk it once
        // and retain only entries whose target still exists.
        if !stale_set.is_empty() {
            self.by_content
                .write()
                .retain(|(tenant, _), pid| !stale_set.contains(&(tenant.clone(), pid.clone())));
            // Defence-in-depth: named profiles are sweep-exempt above, so this
            // is normally a no-op, but keep by_name consistent with entries.
            self.by_name
                .write()
                .retain(|(tenant, _), pid| !stale_set.contains(&(tenant.clone(), pid.clone())));
        }
        n
    }

    /// Snapshot of `(tenant, policy_id)` pairs currently held.
    /// Test/observability helper.
    pub fn list(&self) -> Vec<(String, PolicyId)> {
        self.entries.read().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GraphUpdate;
    use crate::evaluator::types::{EvalAuthRequest, EvalAuthResponse};
    use crate::evaluator::{Evaluator, EvaluatorError};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Trivial mock evaluator — only used to make a factory
    /// closure typecheck; we don't actually drive it in registry tests.
    struct NoopEvaluator;
    #[tonic::async_trait]
    impl Evaluator for NoopEvaluator {
        async fn update(&self, _: Vec<GraphUpdate>) -> Result<(), EvaluatorError> {
            Ok(())
        }
        async fn query(&self, _: EvalAuthRequest) -> Result<EvalAuthResponse, EvaluatorError> {
            Ok(EvalAuthResponse { results: vec![] })
        }
        async fn reset(&self) -> Result<(), EvaluatorError> {
            Ok(())
        }
        fn backend_name(&self) -> &str {
            "noop"
        }
    }

    fn factory_with_counter(counter: Arc<AtomicUsize>) -> EvaluatorFactory {
        Arc::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>)
        })
    }

    #[test]
    fn install_assigns_unique_ids() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let id1 = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let id2 = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            false,
        );
        assert_ne!(id1, id2, "every install gets a fresh id");
    }

    #[test]
    fn resolve_default_when_policy_id_omitted() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let default = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let _variant = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            false,
        );

        let (_factory, backend, resolved) = r.resolve("acme", None).unwrap();
        assert_eq!(resolved, default);
        assert_eq!(backend, "souffle");
    }

    #[test]
    fn resolve_explicit_id_picks_variant() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let _default = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let variant = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            false,
        );

        let (_, _, resolved) = r.resolve("acme", Some(&variant)).unwrap();
        assert_eq!(resolved, variant);
    }

    #[test]
    fn cross_tenant_id_is_invisible() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let acme_id = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let _ = r.install(
            "orgb",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );

        // orgb cannot resolve acme's policy by id even if it knows it.
        match r.resolve("orgb", Some(&acme_id)) {
            Ok(_) => panic!("orgb should not resolve acme's policy id"),
            Err(PolicyRegistryError::PolicyNotFound { tenant, .. }) => {
                assert_eq!(tenant, "orgb");
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn install_default_demotes_prior_default() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let v1 = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let v2 = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );

        assert_eq!(r.default_for("acme"), Some(v2.clone()));

        // v1 is still resolvable by id (in-flight sessions); only its
        // default-immune flag is dropped, which the orphan sweep test
        // covers.
        let (_, _, resolved) = r.resolve("acme", Some(&v1)).unwrap();
        assert_eq!(resolved, v1);
    }

    #[test]
    fn orphan_sweep_drops_non_default_idle_variants() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let default = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let variant = r.install(
            "acme",
            factory_with_counter(Arc::clone(&c)),
            "souffle".into(),
            false,
        );

        // Sweep with a 0 TTL: every non-default entry is "stale".
        let n = r.evict_orphans(Duration::ZERO);
        assert_eq!(n, 1, "exactly one orphan should be swept");
        assert!(r.contains("acme", &default), "default survives sweep");
        assert!(!r.contains("acme", &variant), "variant evicted");
    }

    #[test]
    fn unknown_tenant_returns_tenant_unknown() {
        let r = PolicyRegistry::new();
        match r.resolve("ghost", None) {
            Ok(_) => panic!("unknown tenant should error"),
            Err(PolicyRegistryError::TenantUnknown(t)) => assert_eq!(t, "ghost"),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    /// Repeated `install_dedup` with the same content hash must
    /// return the same id — this is what keeps the registry from
    /// growing when sessions repeatedly pin the same source.
    #[test]
    fn install_dedup_collapses_identical_content() {
        let r = PolicyRegistry::new();
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let make_factory = |c: Arc<AtomicUsize>| -> EvaluatorFactory {
            Arc::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>)
            })
        };

        let id1 = r.install_dedup(
            "acme",
            "hashA",
            make_factory(Arc::clone(&factory_calls)),
            "souffle".into(),
            true,
        );
        let id2 = r.install_dedup(
            "acme",
            "hashA",
            make_factory(Arc::clone(&factory_calls)),
            "souffle".into(),
            false,
        );
        let id3 = r.install_dedup(
            "acme",
            "hashB",
            make_factory(Arc::clone(&factory_calls)),
            "souffle".into(),
            false,
        );

        assert_eq!(id1, id2, "identical content_hash → same policy_id");
        assert_ne!(id1, id3, "different content_hash → distinct policy_id");
        assert_eq!(r.list().len(), 2, "registry holds two entries, not three");
    }

    /// Dedup MUST be tenant-scoped: the same content hash uploaded
    /// by tenant A and tenant B yields *different* ids. Otherwise
    /// tenant B could grab tenant A's factory by guessing its
    /// content hash, breaking the isolation invariant.
    #[test]
    fn install_dedup_is_per_tenant() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let make_factory = |c: Arc<AtomicUsize>| -> EvaluatorFactory {
            Arc::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>)
            })
        };

        let acme_id = r.install_dedup(
            "acme",
            "shared-hash",
            make_factory(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        let orgb_id = r.install_dedup(
            "orgb",
            "shared-hash",
            make_factory(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        // Both tenants now derive their PolicyId from the same
        // content hash, so the ids are equal — but the registry
        // entries themselves are keyed on `(tenant, PolicyId)` so
        // they stay disjoint, which is what the isolation invariant
        // actually requires. Resolving each tenant's id returns its
        // own factory.
        assert_eq!(acme_id, orgb_id);
        assert!(r.lookup_by_content("acme", "shared-hash").is_some());
        assert!(r.lookup_by_content("orgb", "shared-hash").is_some());
        let (_, _, acme_resolved) = r.resolve("acme", Some(&acme_id)).unwrap();
        let (_, _, orgb_resolved) = r.resolve("orgb", Some(&orgb_id)).unwrap();
        assert_eq!(acme_resolved, orgb_resolved);
        // The registry holds two distinct entries keyed on
        // `(tenant, PolicyId)`.
        assert_eq!(r.list().len(), 2);
    }

    /// A bare `make_default=false` dedup hit must not promote or
    /// demote — a Session-scoped re-upload of an existing source
    /// can't flip the tenant default under another session's feet.
    /// But `make_default=true` (only ever set for explicit
    /// Default/Force intent) *is* honored on a hit: it promotes the
    /// existing entry, closing the TOCTOU race where a concurrent
    /// identical install lands the entry between the SetPolicy
    /// dedup-check and the install call.
    #[test]
    fn install_dedup_hit_honors_make_default() {
        let r = PolicyRegistry::new();
        let c = Arc::new(AtomicUsize::new(0));
        let make_factory = |c: Arc<AtomicUsize>| -> EvaluatorFactory {
            Arc::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>)
            })
        };

        // First install: NOT a default.
        let v_id = r.install_dedup(
            "acme",
            "h",
            make_factory(Arc::clone(&c)),
            "souffle".into(),
            false,
        );
        assert_eq!(r.default_for("acme"), None);

        // A second make_default=false hit must NOT promote.
        let v_id_b = r.install_dedup(
            "acme",
            "h",
            make_factory(Arc::clone(&c)),
            "souffle".into(),
            false,
        );
        assert_eq!(v_id, v_id_b);
        assert_eq!(
            r.default_for("acme"),
            None,
            "make_default=false dedup hit must not promote a variant",
        );

        // make_default=true on a hit DOES promote (explicit intent).
        let v_id2 = r.install_dedup(
            "acme",
            "h",
            make_factory(Arc::clone(&c)),
            "souffle".into(),
            true,
        );
        assert_eq!(v_id, v_id2);
        assert_eq!(
            r.default_for("acme").as_ref(),
            Some(&v_id),
            "make_default=true dedup hit must promote to tenant default",
        );
    }

    /// `promote_to_default` is the *explicit-intent* path:
    /// `SetPolicy(scope=Default)` on a previously-uploaded source
    /// must actually flip the tenant default even though install
    /// is a no-op. The SetPolicy dedup short-circuit must not skip
    /// this step, or the default lags until restart.
    #[test]
    fn promote_to_default_flips_existing_entry() {
        let r = PolicyRegistry::new();
        let make_factory = || -> EvaluatorFactory {
            Arc::new(move || Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>))
        };
        let v = r.install_dedup("acme", "hashV", make_factory(), "souffle".into(), false);
        assert_eq!(r.default_for("acme"), None);
        let promoted = r.promote_to_default("acme", "hashV").unwrap();
        assert_eq!(promoted, v);
        assert_eq!(r.default_for("acme").as_ref(), Some(&v));

        // Same hash promoted again is a no-op (idempotent).
        r.promote_to_default("acme", "hashV").unwrap();
        assert_eq!(r.default_for("acme").as_ref(), Some(&v));

        // Promoting an unknown hash errors.
        match r.promote_to_default("acme", "ghost") {
            Err(PolicyRegistryError::PolicyNotFound { .. }) => {}
            other => panic!("expected PolicyNotFound; got {other:?}"),
        }
    }

    /// `evict_orphans` must purge `by_content` along with `entries`,
    /// otherwise `lookup_by_content` returns a `PolicyId` that no
    /// longer resolves, and the SetPolicy dedup short-circuit
    /// reuses the stale id → `set_session_policy` fails with
    /// "policy not found".
    #[test]
    fn evict_orphans_clears_by_content_index() {
        let r = PolicyRegistry::new();
        let make_factory = || -> EvaluatorFactory {
            Arc::new(move || Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>))
        };
        // Non-default variant → eligible for orphan sweep.
        r.install_dedup("acme", "hashX", make_factory(), "souffle".into(), false);
        assert!(r.lookup_by_content("acme", "hashX").is_some());

        // Zero idle TTL → sweep eligible immediately.
        let n = r.evict_orphans(Duration::ZERO);
        assert_eq!(n, 1);

        // After eviction, by_content must NOT return a dangling id.
        assert_eq!(
            r.lookup_by_content("acme", "hashX"),
            None,
            "by_content must be purged in lockstep with entries",
        );
    }

    /// `register_name` makes a baked profile bindable by its human name; the
    /// name resolves to the same `PolicyId` (== content hash) used by dedup.
    #[test]
    fn register_name_resolves_to_policy_id() {
        let r = PolicyRegistry::new();
        let make_factory = || -> EvaluatorFactory {
            Arc::new(move || Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>))
        };
        let pid = r.install_dedup("acme", "hashSec", make_factory(), "souffle".into(), true);
        r.register_name("acme", "security", &pid);

        assert_eq!(r.lookup_by_name("acme", "security"), Some(pid.clone()));
        // Unknown name and foreign tenant both miss (tenant isolation).
        assert_eq!(r.lookup_by_name("acme", "nope"), None);
        assert_eq!(r.lookup_by_name("other", "security"), None);
    }

    /// Named (curated/baked) profiles are pinned: even a non-default variant
    /// that would otherwise be orphan-swept must survive so a client can bind
    /// it by name for the life of the process.
    #[test]
    fn evict_orphans_exempts_named_profiles() {
        let r = PolicyRegistry::new();
        let make_factory = || -> EvaluatorFactory {
            Arc::new(move || Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>))
        };
        // A non-default variant — normally sweep-eligible (cf.
        // evict_orphans_clears_by_content_index) — but registered by name.
        let pid = r.install_dedup("acme", "hashDeny", make_factory(), "souffle".into(), false);
        r.register_name("acme", "deny-all", &pid);

        let n = r.evict_orphans(Duration::ZERO);
        assert_eq!(
            n, 0,
            "named (curated) profiles must be exempt from the orphan sweep"
        );
        assert!(r.lookup_by_content("acme", "hashDeny").is_some());
        assert_eq!(r.lookup_by_name("acme", "deny-all"), Some(pid));
    }

    /// Lazy-replay path: `reserve_default` lands the pointer in
    /// `defaults` ahead of any entry, then `install_dedup` with
    /// `make_default=false` materialises the entry. Without the
    /// `mark_entry_default` fix-up, the entry is `is_default=false`
    /// even though the tenant default points at it, so the orphan
    /// sweep would silently evict the tenant default on first
    /// idle pass.
    #[test]
    fn mark_entry_default_protects_lazy_replay_from_sweep() {
        let r = PolicyRegistry::new();
        let make_factory = || -> EvaluatorFactory {
            Arc::new(move || Ok(Arc::new(NoopEvaluator) as Arc<dyn Evaluator>))
        };

        // Boot replay: pointer lands in `defaults` before the
        // entry exists.
        r.reserve_default("acme", "hashX");

        // First dispatch: lazy install materialises the entry with
        // `make_default=false` so the existing default pointer
        // isn't clobbered.
        r.install_dedup("acme", "hashX", make_factory(), "souffle".into(), false);

        // Apply the fix-up that the production caller
        // (`SessionEvaluatorMap::ensure_installed`) runs.
        let policy_id = PolicyId::from_string("hashX");
        r.mark_entry_default("acme", &policy_id);

        // Sweep with zero idle TTL would otherwise evict this — but
        // `is_default=true` makes it exempt.
        let n = r.evict_orphans(Duration::ZERO);
        assert_eq!(n, 0, "tenant default must not be swept");
        assert!(
            r.lookup_by_content("acme", "hashX").is_some(),
            "entry must survive — it's the tenant default"
        );
        assert_eq!(r.default_for("acme"), Some(policy_id));
    }
}
