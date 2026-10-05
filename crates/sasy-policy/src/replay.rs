//! Boot-time replay + on-demand compile for persisted policies.
//!
//! On binary startup the in-memory `PolicyRegistry` and
//! `session_to_policy` map are empty. The graph store, however,
//! holds three things from the previous run:
//!
//! 1. `(tenant, session_id) → content_hash` for sessions that were
//!    pinned via `SetPolicy(scope=Session)`.
//! 2. `tenant → content_hash` for tenant defaults set via
//!    `SetPolicy(scope=Default | Force)`.
//! 3. `content_hash → source bytes + functor source + backend` so
//!    we can recompile each unique policy without the caller
//!    re-uploading.
//!
//! Replay is **lazy by default**:
//!
//! * [`rehydrate_persisted_bindings`] walks (1) and (2) and installs
//!   the bindings into the engine's in-memory maps. No compiles
//!   happen here — the registry is left without the actual
//!   `PolicyEntry` for each restored hash. This makes startup O(1)
//!   regardless of how many sessions were persisted.
//! * On the first dispatch for a restored binding,
//!   `SessionEvaluatorMap::ensure_installed` reads the source bundle
//!   from the graph store, compiles it via
//!   [`compile_factory_from_source`], and installs the entry. The
//!   souffle build cache makes the per-session hit ≈150 ms warm.
//!
//! Eager replay (compile every persisted policy at boot) is still
//! available behind `SASY_REPLAY_EAGER=1` for deployments where
//! first-request latency matters more than startup time.
//! [`replay_persisted_policies_eager`] is the entry point.

use std::collections::HashSet;

use sasy_graph::GraphStore;
use tracing::{info, warn};

use crate::engine::{Engine, InstallMode};

/// Outcome counts for the boot replay pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReplayStats {
    /// Per-session bindings rehydrated into `session_to_policy`.
    pub bindings_restored: usize,
    /// Tenant defaults rehydrated into `PolicyRegistry::defaults`.
    pub defaults_restored: usize,
    /// Distinct content hashes referenced by bindings or defaults
    /// that have no source bundle on disk. Logged on boot so the
    /// operator sees the gap; the next live `SetPolicy` for that
    /// hash repairs it. Doesn't fail the replay.
    pub orphans_no_source: usize,
    /// (Eager mode only) Unique `(tenant, content_hash)` policies
    /// compiled + installed up front.
    pub policies_installed_eager: usize,
    /// (Eager mode only) Persisted entries whose compile failed
    /// during pre-install. Stays as a lazy candidate; first
    /// dispatch retries.
    pub skipped_compile_failed_eager: usize,
    /// Persisted entries carrying user-admitted functor source that the
    /// functor gate refuses under the settings this process was started
    /// with. Not compiled, and not retried into success: a session bound to
    /// one fails closed on dispatch with the same refusal.
    pub skipped_functor_gate: usize,
}

/// Lazy boot replay: rehydrate `session_to_policy` and the tenant
/// defaults map without compiling anything. The first dispatch for
/// each restored binding triggers
/// [`super::session_evaluator::SessionEvaluatorMap::ensure_installed`]
/// which fetches the source from the graph store and runs
/// [`compile_factory_from_source`].
pub fn rehydrate_persisted_bindings(
    graph_store: &GraphStore,
    engine: &dyn Engine,
) -> Result<ReplayStats, anyhow::Error> {
    let bindings = graph_store
        .all_policy_bindings()
        .map_err(|e| anyhow::anyhow!("read policy bindings: {e}"))?;
    let defaults = graph_store
        .all_tenant_defaults()
        .map_err(|e| anyhow::anyhow!("read tenant defaults: {e}"))?;

    if bindings.is_empty() && defaults.is_empty() {
        return Ok(ReplayStats::default());
    }

    // Pre-flight: report any binding/default whose source bundle
    // is missing. Doesn't fail the rehydration — those hashes are
    // still bound, dispatch just errors with a clear "missing
    // source" message until someone re-uploads.
    let mut hashes: HashSet<String> = HashSet::new();
    for (_, h) in &bindings {
        hashes.insert(h.clone());
    }
    for (_, h) in &defaults {
        hashes.insert(h.clone());
    }
    let mut orphans = 0usize;
    for h in &hashes {
        match graph_store.get_policy_source(h) {
            Ok(Some(_)) => {}
            Ok(None) => {
                warn!(content_hash = %h, "persisted binding has no source bundle on disk; first dispatch will fail until re-uploaded");
                orphans += 1;
            }
            Err(e) => {
                warn!(content_hash = %h, error = %e, "could not read policy source; treating as missing");
                orphans += 1;
            }
        }
    }

    for (scope, hash) in &bindings {
        engine.lazy_bind_session(scope.clone(), hash);
    }
    for (tenant, hash) in &defaults {
        engine.lazy_set_default(tenant, hash);
    }

    let stats = ReplayStats {
        bindings_restored: bindings.len(),
        defaults_restored: defaults.len(),
        orphans_no_source: orphans,
        ..Default::default()
    };
    info!(
        bindings = stats.bindings_restored,
        defaults = stats.defaults_restored,
        orphans_no_source = stats.orphans_no_source,
        "policy bindings rehydrated (lazy: compile happens on first dispatch)"
    );
    Ok(stats)
}

/// Eager boot replay: compile + install every persisted policy at
/// startup, then attach the bindings. Higher startup cost but no
/// first-dispatch latency for restored sessions.
///
/// Triggered when `SASY_REPLAY_EAGER=1`. Failed compiles fall
/// through to the lazy path — the binding is still rehydrated, and
/// the first dispatch retries the compile (which lets the operator
/// recover by uploading a working source via SetPolicy).
pub fn replay_persisted_policies_eager(
    graph_store: &GraphStore,
    engine: &dyn Engine,
    config: &crate::service::PolicyServiceConfig,
) -> Result<ReplayStats, anyhow::Error> {
    let mut stats = rehydrate_persisted_bindings(graph_store, engine)?;
    if stats.bindings_restored == 0 && stats.defaults_restored == 0 {
        return Ok(stats);
    }

    let bindings = graph_store
        .all_policy_bindings()
        .map_err(|e| anyhow::anyhow!("read policy bindings: {e}"))?;
    let defaults = graph_store
        .all_tenant_defaults()
        .map_err(|e| anyhow::anyhow!("read tenant defaults: {e}"))?;

    // Compile each distinct (tenant, content_hash) once.
    let mut want: HashSet<(String, String)> = HashSet::new();
    for (scope, hash) in &bindings {
        want.insert((scope.tenant().to_string(), hash.clone()));
    }
    for (tenant, hash) in &defaults {
        want.insert((tenant.clone(), hash.clone()));
    }

    for (tenant, hash) in want {
        // Ask within the admission class in force: this returns the record
        // this process may compile, or the refusal when every stored record
        // for the hash is one it may not.
        let (admission, source) = match admitted_policy_source(graph_store, &hash, config) {
            Ok(Some(found)) => found,
            Ok(None) => continue, // already counted in orphans_no_source
            Err(e) if e.starts_with(crate::service::FUNCTOR_REFUSED_AT_LOAD) => {
                warn!(tenant, hash, "skipping a persisted policy at boot: {e}");
                stats.skipped_functor_gate += 1;
                continue;
            }
            Err(e) => {
                warn!(tenant, hash, error = %e, "could not read the persisted policy source; will retry on first dispatch");
                stats.skipped_compile_failed_eager += 1;
                continue;
            }
        };
        let mode = if defaults.iter().any(|(t, h)| t == &tenant && h == &hash) {
            InstallMode::Default
        } else {
            InstallMode::Variant
        };
        // `admission` is the class the record was FOUND under — the class its
        // key encodes — which is what the gate below re-checks.
        match install_from_source(engine, &tenant, &hash, admission, &source, mode, config) {
            Ok(_) => stats.policies_installed_eager += 1,
            Err(e) if e.starts_with(crate::service::FUNCTOR_REFUSED_AT_LOAD) => {
                warn!(
                    tenant,
                    hash,
                    admission = ?admission,
                    "skipping a persisted policy at boot: {e}"
                );
                stats.skipped_functor_gate += 1;
            }
            Err(e) => {
                warn!(tenant, hash, error = %e, "eager replay compile failed; will retry on first dispatch");
                stats.skipped_compile_failed_eager += 1;
            }
        }
    }
    Ok(stats)
}

/// True iff the operator opted into eager replay via the
/// `SASY_REPLAY_EAGER` env variable.
pub fn eager_replay_requested() -> bool {
    sasy_common::env_flag("SASY_REPLAY_EAGER")
}

/// Pick the persisted record for `content_hash` that THIS process is allowed
/// to compile, asking within the admission class in force.
///
/// The store keeps the two admission classes as separate records (see
/// [`sasy_graph::persistence::RocksStore::put_policy_source`]), so one content
/// hash can have a user-supplied record, an admin-supplied record, or both.
/// This walks them least-privileged first and returns the first one `config`
/// admits, paired with the class it was found under.
///
/// The class returned — and the class gated on — is THE ONE ITS STORAGE KEY
/// ENCODES, not the `functor_admission` field inside the record. The key is
/// what a later uploader cannot rewrite; the field is kept for compatibility
/// and is only what an ordinary write derives the key from. Gating on the
/// field would leave the key split constraining writes and nothing else.
///
/// The consequences are the ones the class split exists for:
///
/// * A later admin upload of content already stored as user-supplied does not
///   turn the user record into an admin one. While the opt-in admits
///   user-supplied functors, that content still loads as user-supplied; once
///   the opt-in is switched off, it loads only because an admin really did
///   upload those exact bytes under the admin class.
/// * A later non-admin upload of content an admin uploaded does not take the
///   admin record away — it is still there when the user record is refused.
///
/// `Ok(None)` means nothing is stored under that hash in either class. `Err`
/// is a store read failure, or — carrying
/// [`crate::service::FUNCTOR_REFUSED_AT_LOAD`] — the refusal from the last
/// record tried, when every stored record is refused.
pub(crate) fn admitted_policy_source(
    graph_store: &GraphStore,
    content_hash: &str,
    config: &crate::service::PolicyServiceConfig,
) -> Result<Option<(sasy_graph::FunctorAdmission, sasy_graph::PersistedPolicy)>, String> {
    let records = graph_store
        .policy_sources_by_least_privilege(content_hash)
        .map_err(|e| format!("read persisted policy source: {e}"))?;
    let mut refusal = None;
    for (admission, source) in records {
        // A bundle with no functor source has nothing to admit.
        if source.functor_source.is_empty() {
            return Ok(Some((admission, source)));
        }
        // `admission` is the class the record's KEY encodes, which is the
        // class it was stored under; `source.functor_admission` is a field a
        // record could carry any value in.
        match config.admits_functor_source(admission) {
            Ok(()) => return Ok(Some((admission, source))),
            Err(reason) => refusal = Some(reason),
        }
    }
    match refusal {
        Some(reason) => Err(reason),
        None => Ok(None),
    }
}

/// Helper: compile + install a single `(tenant, content_hash)` via
/// the engine's normal install path. Used by the eager replay
/// codepath.
///
/// Runs the functor gate first, against `config` — the settings in force in
/// THIS process, not the ones that were in force when the source was
/// uploaded. A policy whose functor source was admitted as user-supplied
/// under an opt-in that has since been turned off does not come back at boot.
/// A bundle with no functor source has nothing to admit and is never refused
/// here.
///
/// `admission` is the class the record was FOUND under, i.e. the class its
/// storage key encodes. That is the authoritative one: the key is the half a
/// later uploader cannot rewrite, while the record's own
/// `functor_admission` field is only what an ordinary write derives the key
/// from and may say anything (see `PersistedPolicy::functor_admission`).
/// Gating on the field here would make a key/value mismatch mean one thing on
/// this path and another on the lazy one, which asks
/// [`admitted_policy_source`] and gets the key's class.
pub(crate) fn install_from_source(
    engine: &dyn Engine,
    tenant: &str,
    content_hash: &str,
    admission: sasy_graph::FunctorAdmission,
    source: &sasy_graph::PersistedPolicy,
    mode: InstallMode,
    config: &crate::service::PolicyServiceConfig,
) -> Result<String, String> {
    if !source.functor_source.is_empty() {
        config.admits_functor_source(admission)?;
    }
    let (factory, factory_backend) = compile_factory_from_source(
        &source.policy_source,
        &source.functor_source,
        &source.backend,
    )?;
    let id = engine
        .install_policy(tenant, content_hash, factory, factory_backend, mode)
        .map_err(|e| format!("install: {e}"))?;
    Ok(id)
}

/// Per-work-directory locks for [`compile_factory_from_source`]. See its body
/// for why the lazy path needs them. Absent from the restricted build, whose
/// `compile_factory_from_source` is a fail-closed stub with no work directory.
#[cfg(feature = "compiler")]
fn lazy_compile_lock(key: &str) -> std::sync::Arc<parking_lot::Mutex<()>> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    static LOCKS: OnceLock<parking_lot::Mutex<HashMap<String, Arc<parking_lot::Mutex<()>>>>> =
        OnceLock::new();
    let mut map = LOCKS.get_or_init(Default::default).lock();
    // Drop entries nobody holds once the map grows; a long-lived process
    // otherwise keeps one per policy it has ever lazily compiled.
    if map.len() > 256 {
        map.retain(|_, v| Arc::strong_count(v) > 1);
    }
    Arc::clone(map.entry(key.to_string()).or_default())
}

/// Compile `(policy_source, functor_source)` for `backend` and return the
/// spawn factory plus the canonical backend name. Mirrors the inner branches
/// of `service::set_policy`'s build closure without doing the install step,
/// so the SetPolicy path and the lazy-compile path share one compile
/// pipeline.
#[cfg(feature = "compiler")]
pub(crate) fn compile_factory_from_source(
    policy_source: &str,
    functor_source: &str,
    backend: &str,
) -> Result<(crate::session_evaluator::EvaluatorFactory, String), String> {
    let upload_id = format!(
        "lazy-{}-{}",
        std::process::id(),
        // Trim the hash to keep the dir name reasonable; prefix
        // collisions are harmless because we mkdir-recursive.
        &short_id_for_workdir(policy_source, functor_source),
    );
    let work_root = std::path::Path::new("/tmp/sasy-uploads").join(&upload_id);
    // Single-flight per work directory. The directory is derived from the
    // policy content, so two callers compiling the same not-yet-materialised
    // policy — the ordinary post-restart wake-up burst — write `policy.dl`,
    // `desugared.dl` and the linked binary over each other in place: one
    // compile reads a file the other is truncating, and the content-addressed
    // cache can be handed a binary that is still being linked.
    //
    // Keyed on the directory, so different policies still compile in
    // parallel; the same policy compiles once and the second caller gets the
    // finished artifact from the cache rather than rebuilding it.
    let build_guard = lazy_compile_lock(&upload_id);
    let _build_guard = build_guard.lock();
    std::fs::create_dir_all(&work_root).map_err(|e| format!("Create workspace: {e}"))?;

    let policy_path = work_root.join("policy.dl");
    std::fs::write(&policy_path, policy_source).map_err(|e| format!("Write policy: {e}"))?;

    let custom_functor_path = if !functor_source.is_empty() {
        let path = work_root.join("functors.cpp");
        std::fs::write(&path, functor_source).map_err(|e| format!("Write functors: {e}"))?;
        Some(path)
    } else {
        None
    };

    let Some(be) = sasy_common::Backend::from_wire(backend) else {
        return Err(format!(
            "Backend '{backend}' not supported via lazy compile"
        ));
    };
    // Exhaustive over the backend set: a new variant is a compile error
    // here until handled. The non-compiled backends (flowlog/stub) have
    // no source-compile pipeline — the lazy/eager replay path can't
    // reconstruct them, and they can't be installed via SetPolicy either
    // (see `service.rs`), so a persisted policy in those backends
    // shouldn't exist; reject with an actionable error rather than
    // letting them drift.
    match be {
        sasy_common::Backend::SouffleInterpreted | sasy_common::Backend::Souffle => {
            crate::evaluator::factory::build_souffle_factory(
                be,
                &work_root,
                &policy_path,
                custom_functor_path.as_deref(),
            )
        }
        sasy_common::Backend::Flowlog => {
            Err("the flowlog backend is experimental: it has static policy \
             assets and no compile pipeline, so it is not reconstructed \
             from persisted source via lazy/eager replay."
                .to_string())
        }
        sasy_common::Backend::Stub => Err(
            "the stub backend has no compiled policy to replay (the stub \
             evaluator ignores policy source)."
                .to_string(),
        ),
    }
}

/// Restricted build: the policy compiler is not linked, so recompiling from
/// persisted source is unavailable. Identical signature to the real fn; the
/// lazy/eager replay callers get a fail-closed `Err`. A restricted build binds
/// its precompiled policies by content hash instead (service.rs).
#[cfg(not(feature = "compiler"))]
pub(crate) fn compile_factory_from_source(
    policy_source: &str,
    functor_source: &str,
    backend: &str,
) -> Result<(crate::session_evaluator::EvaluatorFactory, String), String> {
    let _ = (policy_source, functor_source, backend);
    Err("policy compiler not built into this (restricted) binary: \
         persisted-source recompile is unavailable; bind a pre-installed \
         curated profile by content hash instead"
        .to_string())
}

#[cfg(feature = "compiler")]
fn short_id_for_workdir(policy: &str, functors: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(policy.as_bytes());
    h.update(b"\0");
    h.update(functors.as_bytes());
    crate::hash::hex(&h.finalize()[..6])
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;
