//! gRPC `PolicyEngine` service implementation.
//!
//! API surface (admin, except `SetPolicy(Session)` — see its handler):
//! - `SetPolicy`: compiles a policy, registers it, and binds it to the
//!   requested scope (Session / Default / Force). Install is
//!   synchronous — the call returns once the new policy is compiled
//!   and installed. It does not spawn evaluators: those spawn lazily at
//!   first dispatch, and a `Force` rollout drops the ones already running.
//! - `GetEvaluatorStatus`: queries the live evaluator backend name
//! - `Health`: includes backend name in status message

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Per-process monotonic counter for upload workspace naming. Combined
/// with the pid (set on first read) gives a globally unique ID without
/// pulling in `uuid`.
fn next_upload_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}", std::process::id(), n)
}

use sasy_common::policy_engine::{
    policy_engine_server::PolicyEngine as PolicyEngineTrait,
    policy_scope::Target as PolicyScopeTarget, AuthorizationRequest, AuthorizationResponse,
    EndSessionRequest, EndSessionResponse, EvaluatorStatusRequest, EvaluatorStatusResponse,
    HealthRequest, HealthResponse, SetPolicyRequest, SetPolicyResponse, SyncStatusRequest,
    SyncStatusResponse, UpdatePolicyMetadataRequest, UpdatePolicyMetadataResponse,
    ValidatePolicyRequest, ValidatePolicyResponse,
};
use sasy_common::SessionScope;
use sasy_graph::FunctorAdmission;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::engine::Engine;

/// A durable write SetPolicy made *before* publishing the change it belongs
/// to, together with what the row held beforehand.
///
/// Nothing spans the in-memory registry and RocksDB transactionally, so the
/// two can only be kept consistent by ordering: every fallible durable write
/// happens first, and publication — which is what makes a policy live and,
/// under `Force`, evicts the tenant — happens last. When publication still
/// fails, these are replayed backwards to leave the store untouched.
///
/// Every durable write SetPolicy makes has a variant here, with one deliberate
/// exception: the policy source, which is content-addressed, so re-writing it
/// is idempotent and an orphaned copy is unreachable — nothing reads it unless
/// a binding or default names that hash. Any other write without a variant is
/// a write that a declined request leaves behind, to take effect at the next
/// restart or the next evaluator respawn.
enum DurableWrite {
    TenantDefault {
        tenant: String,
        prior: Option<String>,
    },
    Binding {
        scope: SessionScope,
        prior: Option<String>,
    },
    /// Config supplied on a session-scoped bind. `None` prior means the row
    /// was absent — which is not the same as present-and-empty, since an
    /// empty row suppresses the fallback to the tenant-wide config.
    BindingMetadata {
        scope: SessionScope,
        content_hash: String,
        prior: Option<Vec<sasy_graph::PolicyMetadataFact>>,
    },
    /// Config supplied on a tenant-wide (Default/Force) bind.
    PolicyMetadata {
        tenant: String,
        content_hash: String,
        prior: Vec<sasy_graph::PolicyMetadataFact>,
    },
    /// The per-session pins a `Force` rollout removed, in full, so they can
    /// be re-recorded if the rollout does not go live.
    ClearedPins(Vec<(SessionScope, String)>),
    /// The per-session config a `Force` rollout removed, likewise. Carries
    /// the tenant so the undo can evict the evaluators that may have seeded
    /// from the cleared state.
    ClearedBindingConfig {
        tenant: String,
        rows: sasy_graph::ClearedBindingMetadata,
    },
    /// Rollback permission for the ownership row this publication created.
    SessionOwner(sasy_graph::OwnerClaimToken),
}

/// The durable writes staged for one publication.
///
/// Every fallible step between the first durable write and the publication
/// goes through [`Self::step`], and [`Drop`] puts back anything still staged,
/// so there is no path that returns while leaving a durable write behind an
/// operation that did not complete. Keeping the writes is the case that has
/// to be stated explicitly, via [`Self::published`].
///
/// The alternative — a rollback written out at each call site — is easy to
/// defeat by accident: a `?` inside the argument of the call that records the
/// undo runs before the record is made, so the rollback it was meant to
/// enable never sees it.
struct Staged<'a, E: Engine> {
    svc: &'a PolicyService<E>,
    written: Vec<DurableWrite>,
    /// False once a publication has been started that this task may not see
    /// the result of. See [`Self::publishing`].
    undo_on_drop: bool,
}

impl<'a, E: Engine + 'static> Staged<'a, E> {
    fn new(svc: &'a PolicyService<E>) -> Self {
        Self {
            svc,
            written: Vec::new(),
            undo_on_drop: true,
        }
    }

    /// A publication is about to start that cannot be cancelled and whose
    /// outcome this task may never observe.
    ///
    /// `spawn_blocking` detaches when its handle is dropped, so if the client
    /// disconnects or its deadline expires while we await the install, the
    /// install still runs and still publishes — while this future is dropped.
    /// Rolling back on that drop would be a decision made without knowing
    /// whether the thing being rolled back went live, and gets it wrong in the
    /// common case: the policy is live, the store says otherwise, and the next
    /// restart silently reverts a rollout that is currently in force.
    ///
    /// Leaving the writes instead means a cancelled call whose install then
    /// fails has recorded a rollout that never went live, which a restart
    /// would apply. That is the lesser of the two, because it converges with
    /// what the operator asked for rather than against it.
    ///
    /// Under `Force` that path also loses data: the pins and per-session
    /// config the rollout cleared are held only in this future's staged set,
    /// so they die with it and nothing can reconstruct them. The rollout is
    /// the one the operator asked for, so converging on it is still the right
    /// direction — but a cancelled `Force` is not a no-op, and re-running it
    /// will not bring those rows back.
    fn publishing(&mut self) {
        self.undo_on_drop = false;
    }

    /// The publication went through. Everything staged for it is now backed
    /// by a live policy and must be KEPT, so it leaves the staged set — while
    /// `Drop` is armed again for whatever is staged next.
    ///
    /// Both halves matter. Anything recorded after this point (a session's
    /// ownership claim, its config, its pin) is written with no await in
    /// front of it, so cancellation cannot reach that stretch and only a
    /// panic can; leaving `Drop` disarmed across it would let a panic leave
    /// exactly the state the staging exists to prevent. But re-arming without
    /// clearing would point `Drop` at writes whose publication already
    /// happened, and undo them.
    fn publication_succeeded(&mut self) {
        self.commit_owner_claims();
        self.written.clear();
        self.undo_on_drop = true;
    }

    /// The publication did not happen. The staged writes have to come back
    /// out; the caller does that through `step`, and `Drop` covers a panic on
    /// the way there.
    fn publication_failed(&mut self) {
        self.undo_on_drop = true;
    }

    /// Run one fallible step; on failure, put back everything staged so far
    /// and return the error to hand the caller.
    #[allow(clippy::result_large_err)]
    fn step<T>(&mut self, r: Result<T, Status>) -> Result<T, Status> {
        match r {
            Ok(v) => Ok(v),
            Err(e) => Err(self.svc.rollback_then(std::mem::take(&mut self.written), e)),
        }
    }

    fn record(&mut self, w: DurableWrite) {
        self.written.push(w);
    }

    /// The publication succeeded: the staged writes are what the store should
    /// keep.
    fn published(mut self) {
        self.commit_owner_claims();
        self.written.clear();
    }
}

impl<E: Engine> Staged<'_, E> {
    fn commit_owner_claims(&self) {
        for step in &self.written {
            if let DurableWrite::SessionOwner(token) = step {
                self.svc.commit_owner_claim(token);
            }
        }
    }
}

impl<E: Engine> Drop for Staged<'_, E> {
    /// Rollback is the default; keeping the writes is the thing that has to
    /// be said explicitly. Anything still staged here left by a path that did
    /// not go through `step` — a `?` on a call routed around it, a panic —
    /// and would otherwise be a durable change behind a request that failed.
    fn drop(&mut self) {
        if self.written.is_empty() || !self.undo_on_drop {
            self.commit_owner_claims();
            return;
        }
        tracing::error!(
            staged = self.written.len(),
            "SetPolicy returned with durable writes neither published nor rolled back; \
             rolling them back"
        );
        for step in std::mem::take(&mut self.written).iter().rev() {
            if let Err(e) = self.svc.undo_durable(step) {
                tracing::error!(error = %e, "rollback on drop failed");
            }
        }
    }
}

/// Holds whichever kind of tenant guard the operation needs, so both can be
/// kept alive by one binding.
enum RolloutGuard {
    // Held for the scope of the operation, never read.
    Shared(#[allow(dead_code)] tokio::sync::OwnedRwLockReadGuard<()>),
    Exclusive(#[allow(dead_code)] tokio::sync::OwnedRwLockWriteGuard<()>),
}

/// Mutual exclusion for the operations that read shared state, write it, and
/// then publish.
///
/// `SetPolicy` and `EndSession` are read-modify-write sequences spanning
/// RocksDB and the in-memory registry. Run concurrently over the same rows
/// they interleave destructively, in ways no amount of per-step ordering
/// fixes: one call reads a prior, a second call writes and publishes
/// successfully, then the first fails and "restores" the prior over the
/// second's result. A `Force` rollout's tenant-wide clear cuts across an
/// in-flight bind, leaving a session pinned to a policy whose config was
/// deleted. An `EndSession` completes between a bind's ownership claim and
/// its pin write, so the pin lands on a released id.
///
/// Two levels, always acquired outermost-first, so there is no cycle and no
/// deadlock:
/// - **per tenant** — a rollout (`Default`/`Force`) takes it exclusively,
///   because it touches every scope in the tenant. Everything else takes it
///   shared, so ordinary session traffic does not serialize against itself.
/// - **per scope** — any operation on one session takes it exclusively.
///
/// `CheckAuthorization` takes neither: it publishes nothing, and putting a
/// lock on the enforcement path to protect the control path would be the
/// wrong trade.
#[derive(Default)]
struct ScopeLocks {
    tenants: parking_lot::Mutex<HashMap<String, Arc<tokio::sync::RwLock<()>>>>,
    scopes: parking_lot::Mutex<HashMap<SessionScope, Arc<tokio::sync::Mutex<()>>>>,
}

/// Above this many idle entries, drop the ones nobody holds. Keyed locks
/// otherwise grow one entry per session id ever seen.
const LOCK_MAP_PRUNE_AT: usize = 512;

impl ScopeLocks {
    fn tenant(&self, tenant: &str) -> Arc<tokio::sync::RwLock<()>> {
        let mut map = self.tenants.lock();
        if map.len() > LOCK_MAP_PRUNE_AT {
            map.retain(|_, v| Arc::strong_count(v) > 1);
        }
        Arc::clone(map.entry(tenant.to_string()).or_default())
    }

    fn scope(&self, scope: &SessionScope) -> Arc<tokio::sync::Mutex<()>> {
        let mut map = self.scopes.lock();
        if map.len() > LOCK_MAP_PRUNE_AT {
            map.retain(|_, v| Arc::strong_count(v) > 1);
        }
        Arc::clone(map.entry(scope.clone()).or_default())
    }
}

/// gRPC service wrapping an [`Engine`].
///
/// Holds an [`Arc<GraphStore>`] alongside the engine so the
/// `SetPolicy` / `EndSession` handlers can persist policy bindings,
/// tenant defaults, source bytes, and session ownership directly
/// to RocksDB. The store is the single source of truth for these
/// keyspaces; the engine's in-memory `PolicyRegistry` and
/// `session_to_policy` map are rebuilt from the store at boot.
pub struct PolicyService<E: Engine> {
    engine: Arc<E>,
    /// See [`ScopeLocks`]. Internal serialization, not configuration, so it
    /// is absent from every constructor's signature.
    locks: ScopeLocks,
    graph_store: Option<Arc<sasy_graph::GraphStore>>,
    /// Restricted (public) build: only pre-installed curated policies are
    /// accepted. A SetPolicy whose content hash isn't already installed is
    /// rejected instead of compiled — there is no toolchain to compile with.
    restricted: bool,
    /// Policy lock: when `Some(set)`, SetPolicy may ONLY (re-)bind a policy
    /// whose content hash is IN `set` — the curated profiles baked at startup.
    /// Any other source (incl. a baked permissive profile like allow-all, which
    /// would otherwise dedup-hit and bind) is refused. The set is the build-time
    /// curated pack (allow-all is deliberately excluded), so a co-located caller
    /// that reaches the loopback engine (the policed agent runs as the same OS
    /// user; per-RPC auth can't keep it out) may switch among curated embedded
    /// profiles by name (e.g. `security` ↔ `deny-all`) but cannot weaken
    /// enforcement below the curated floor. Re-binding the SAME profile is
    /// allowed so a client can (re-)pin it and refresh per-session metadata
    /// (rule_off/rule_on, cooldowns) — those travel as policy_metadata and are
    /// NOT part of the content hash.
    ///
    /// NOTE: this is an in-process mitigation, not a full boundary. A same-user
    /// attacker can still SIGKILL the engine, swap its binary, or rebind its
    /// socket. The robust deployment runs `sasy serve` as a SEPARATE OS user so
    /// the agent cannot kill/replace/impersonate it; see the restricted-binary
    /// docs.
    locked_policy_hashes: Option<std::collections::HashSet<String>>,
    /// Operator decisions the service reads at request time. See
    /// [`PolicyServiceConfig`].
    config: PolicyServiceConfig,
}

/// How far the operator has opened up custom C++ functor source to callers
/// who are not admins. Set by `--allow-user-functors` /
/// `SASY_ALLOW_USER_FUNCTORS`; see the functor gate in `set_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UserFunctors {
    /// The default. Only an admin may supply functor source; everyone else is
    /// refused whatever the host looks like.
    #[default]
    Refuse,
    /// A non-admin may supply functor source on a host where the bubblewrap
    /// sandbox is available, and only there — so the C++ that gets compiled
    /// and loaded runs confined.
    Sandboxed,
    /// A non-admin may supply functor source on any host, including one with
    /// no sandbox, where it compiles and runs unconfined as the user running
    /// the binary. For a development host; the binary says so at startup.
    Unsandboxed,
}

impl UserFunctors {
    /// The value an operator writes: what `--allow-user-functors=<value>` and
    /// `SASY_ALLOW_USER_FUNCTORS=<value>` accept for this variant.
    pub fn as_flag_value(self) -> &'static str {
        match self {
            UserFunctors::Refuse => "off",
            UserFunctors::Sandboxed => "sandboxed",
            UserFunctors::Unsandboxed => "unsandboxed",
        }
    }
}

/// Deployment decisions the policy service consults per request.
///
/// Passed in at construction rather than read from the environment at the use
/// site, so a test can build the service that a given host would produce
/// instead of arranging for that host. The same value is handed to the two
/// paths that LOAD persisted functor source (boot replay and lazy install),
/// so all three ask one question of one configuration.
#[derive(Debug, Clone)]
pub struct PolicyServiceConfig {
    /// What non-admin callers may do with functor source. See
    /// [`UserFunctors`].
    pub user_functors: UserFunctors,
    /// Whether caller-supplied C++ actually runs confined on this host: no
    /// on macOS, on any Linux host without bubblewrap, and whenever
    /// `DISABLE_BWRAP=1` is set.
    ///
    /// `None` — what the binary passes — means ASK
    /// [`crate::sandbox::sandbox_available`] at the moment the question is
    /// asked, rather than storing an answer here at startup. That matters
    /// because the compile chain calls the same function when it decides
    /// whether to wrap, so a separately stored answer could say "confined"
    /// where the compile then runs unconfined.
    ///
    /// Be exact about what asking-when-asked buys. Two of the three inputs
    /// really are re-read on every upload: the operator's switches
    /// (`DISABLE_BWRAP`, `SASY_EVALUATOR_BWRAP`) and whether a `bwrap` binary
    /// is on `PATH`. The third is not. Whether the `bwrap` that is there can
    /// actually create the namespaces it needs — the kernel feature bubblewrap
    /// isolates with — is settled by a probe that runs ONCE per process, at
    /// the first call, and cached for the life of the process. A `bwrap`
    /// upgraded, downgraded or otherwise replaced under a running binary keeps
    /// its old verdict until the binary restarts.
    ///
    /// What IS decided per request, by this service rather than by the host,
    /// is the pair the gate turns on: the role the calling identity carries,
    /// and the [`UserFunctors`] setting in force at that moment.
    ///
    /// `Some(_)` fixes the host answer, which is for tests: it lets one build
    /// the service a given host would produce instead of arranging for that
    /// host.
    pub sandbox_available: Option<bool>,
}

impl Default for PolicyServiceConfig {
    /// The conservative pair: no user functors, and no assumption that a
    /// sandbox exists. A service built this way admits functor source from
    /// admins only, which is what a caller who never configured the question
    /// should get.
    fn default() -> Self {
        Self {
            user_functors: UserFunctors::Refuse,
            sandbox_available: Some(false),
        }
    }
}

impl PolicyServiceConfig {
    /// Does a working bubblewrap sandbox exist right now? Probes the host
    /// unless a fixed answer was injected (tests).
    pub fn sandbox_available_now(&self) -> bool {
        self.sandbox_available
            .unwrap_or_else(crate::sandbox::sandbox_available)
    }

    /// May functor source admitted as `admission` be compiled and loaded
    /// under these settings? `Ok(())` means compile it; `Err(reason)` is a
    /// sentence naming the condition that stopped it and the flag value that
    /// would admit it.
    ///
    /// Asked at upload time by `set_policy`, and again by every path that
    /// loads persisted functor source, because the settings in force now are
    /// not necessarily the ones that were in force when the source was
    /// uploaded.
    pub fn admits_functor_source(&self, admission: FunctorAdmission) -> Result<(), String> {
        if admission == FunctorAdmission::Admin {
            return Ok(());
        }
        match self.user_functors {
            UserFunctors::Refuse => Err(format!(
                "{FUNCTOR_REFUSED_AT_LOAD}: it was admitted as user-supplied, and this \
                 binary was started without --allow-user-functors. Start it with \
                 --allow-user-functors=sandboxed (a host with bubblewrap) or \
                 --allow-user-functors=unsandboxed, or re-upload the policy as an admin."
            )),
            UserFunctors::Sandboxed if !self.sandbox_available_now() => Err(format!(
                "{FUNCTOR_REFUSED_AT_LOAD}: it was admitted as user-supplied and this \
                 binary was started with --allow-user-functors=sandboxed, but this host \
                 has no working bubblewrap sandbox right now, so the C++ would compile \
                 and run unconfined. Install bubblewrap (and unset DISABLE_BWRAP), \
                 start the binary with --allow-user-functors=unsandboxed, or re-upload \
                 the policy as an admin."
            )),
            UserFunctors::Sandboxed | UserFunctors::Unsandboxed => Ok(()),
        }
    }
}

/// Opening words of every refusal [`PolicyServiceConfig::admits_functor_source`]
/// produces. The load paths return the refusal as a plain error string through
/// layers that carry no error type, so this marker is how `check_authorization`
/// recognises one and answers `FAILED_PRECONDITION` rather than `INTERNAL`.
///
/// Every reader matches it as a PREFIX, and every path that wraps a refusal
/// keeps it leading, adding its own detail after. A compile error quotes the
/// policy source it choked on, and the source is written by the caller, so a
/// match anywhere in the message would let a caller pick the status code its
/// unrelated failure comes back as.
pub const FUNCTOR_REFUSED_AT_LOAD: &str =
    "the persisted policy's custom functor source is refused at load";

impl<E: Engine> PolicyService<E> {
    /// Construct a service without a graph store. Suitable for
    /// tests and stub engines that don't persist anything; the
    /// runtime binary uses [`Self::with_persistence`] so policy
    /// bindings survive restarts.
    pub fn new(engine: Arc<E>) -> Self {
        Self {
            engine,
            locks: ScopeLocks::default(),
            graph_store: None,
            restricted: false,
            locked_policy_hashes: None,
            config: PolicyServiceConfig::default(),
        }
    }

    /// Apply the operator's deployment decisions. See
    /// [`PolicyServiceConfig`]; the default is the conservative one.
    pub fn with_config(mut self, config: PolicyServiceConfig) -> Self {
        self.config = config;
        self
    }

    /// Mark this service as a restricted (public) build: reject SetPolicy
    /// uploads whose content isn't a pre-installed curated profile.
    pub fn with_restricted(mut self, restricted: bool) -> Self {
        self.restricted = restricted;
        self
    }

    /// Lock the bindable policies to the curated set of content hashes baked at
    /// startup. `None` leaves the service unlocked (the full engine, used for
    /// policy authoring). See [`Self::locked_policy_hashes`].
    pub fn with_policy_lock(
        mut self,
        locked_policy_hashes: Option<std::collections::HashSet<String>>,
    ) -> Self {
        self.locked_policy_hashes = locked_policy_hashes;
        self
    }

    /// Construct a service wired to a real graph store.
    ///
    /// SetPolicy persists the source bytes keyed on `content_hash`, and
    /// whichever pointers the request implies: `(scope → content_hash)` for a
    /// session bind, `(tenant → content_hash)` for a rollout, plus the config
    /// that travels with either. EndSession drops a session's pin, its dynamic
    /// facts, its per-binding config and its owner.
    pub fn with_persistence(engine: Arc<E>, graph_store: Arc<sasy_graph::GraphStore>) -> Self {
        Self {
            engine,
            locks: ScopeLocks::default(),
            graph_store: Some(graph_store),
            restricted: false,
            locked_policy_hashes: None,
            config: PolicyServiceConfig::default(),
        }
    }

    /// Read the binding currently recorded for `scope`, so it can be put
    /// back if the bind about to be published fails.
    #[allow(clippy::result_large_err)]
    fn read_persisted_binding(&self, scope: &SessionScope) -> Result<Option<String>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(None);
        };
        store.get_policy_binding(scope).map_err(|e| {
            tracing::error!(scope = %scope, error = %e, "read persisted binding failed");
            Status::internal(format!(
                "could not read the current binding for {scope}: {e}"
            ))
        })
    }

    /// Read the default currently recorded for `tenant`, for the same reason.
    #[allow(clippy::result_large_err)]
    fn read_persisted_tenant_default(&self, tenant: &str) -> Result<Option<String>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(None);
        };
        store.get_tenant_default_policy(tenant).map_err(|e| {
            tracing::error!(tenant, error = %e, "read persisted tenant default failed");
            Status::internal(format!(
                "could not read the current default for {tenant}: {e}"
            ))
        })
    }

    /// Read the config recorded for one binding, so it can be put back if the
    /// bind it belongs to does not publish.
    #[allow(clippy::result_large_err)]
    fn read_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
    ) -> Result<Option<Vec<sasy_graph::PolicyMetadataFact>>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(None);
        };
        store
            .get_binding_metadata(scope, content_hash)
            .map_err(|e| {
                tracing::error!(scope = %scope, error = %e, "read binding metadata failed");
                Status::internal(format!(
                    "could not read the current config for {scope}: {e}"
                ))
            })
    }

    /// Same, for the tenant-wide config that travels with a policy.
    #[allow(clippy::result_large_err)]
    fn read_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
    ) -> Result<Vec<sasy_graph::PolicyMetadataFact>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(Vec::new());
        };
        store
            .get_policy_metadata(tenant, content_hash)
            .map_err(|e| {
                tracing::error!(tenant, error = %e, "read policy metadata failed");
                Status::internal(format!(
                    "could not read the current config for {tenant}: {e}"
                ))
            })
    }

    /// Take out every binding's config in `tenant`, returning the rows so a
    /// rollout that then fails to publish can put them back.
    ///
    /// Why a force rollout needs this at all: `evict_tenant` clears the
    /// in-memory evaluator and binding, which is enough whenever the new
    /// default differs from what a session pinned — it re-binds to a different
    /// content hash and its stored row, keyed by (scope, old hash), is never
    /// read again. It is NOT enough when the same policy content is
    /// force-promoted: the re-bound hash matches the stored row,
    /// `assemble_metadata` finds it, takes the "this session pinned itself"
    /// branch, and serves the pre-rollout config from then on. Tenant-wide
    /// rather than per evicted scope, because an idle session has a stored row
    /// and no live evaluator to evict, and would resume on the stale config.
    ///
    /// It runs before the eviction, and the service owns it rather than the
    /// engine because nothing can reconstruct these rows afterwards — the
    /// config comes from the client on each bind and is not derivable from the
    /// policy or the graph — so the side that can undo the rollout has to be
    /// the side that takes them out.
    #[allow(clippy::result_large_err)]
    fn clear_binding_config_for_tenant(
        &self,
        tenant: &str,
    ) -> Result<sasy_graph::ClearedBindingMetadata, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(Default::default());
        };
        store.clear_tenant_binding_metadata(tenant).map_err(|e| {
            tracing::error!(tenant, error = %e, "clearing per-session config failed");
            Status::internal(format!(
                "could not clear per-session config for {tenant}: {e}"
            ))
        })
    }

    /// Undo one durable write made ahead of a publication that then failed.
    ///
    /// Every write SetPolicy makes before publishing is recorded as one of
    /// these, holding the value that was there before. A rollout that fails
    /// to publish therefore leaves the store as it found it, rather than a
    /// declined request quietly taking effect at the next restart.
    fn undo_durable(&self, step: &DurableWrite) -> Result<(), String> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        match step {
            DurableWrite::TenantDefault { tenant, prior } => match prior {
                Some(hash) => store.put_tenant_default_policy(tenant, hash),
                None => store.delete_tenant_default_policy(tenant),
            }
            .map_err(|e| format!("tenant default for {tenant}: {e}")),
            DurableWrite::Binding { scope, prior } => match prior {
                Some(hash) => store.put_policy_binding(scope, hash),
                None => store.delete_policy_binding(scope),
            }
            .map_err(|e| format!("binding for {scope}: {e}")),
            DurableWrite::BindingMetadata {
                scope,
                content_hash,
                prior,
            } => {
                let restored = match prior {
                    Some(facts) => store.put_binding_metadata(scope, content_hash, facts),
                    // Not `delete_binding_metadata`, which drops every
                    // policy's row for the scope: a session validly pinned to
                    // a different policy must keep that policy's config.
                    None => store.delete_binding_metadata_for_policy(scope, content_hash),
                };
                // Authorization takes no lock, by design, so it can have
                // spawned an evaluator that seeded from the row we just took
                // back out — and a seed lasts the evaluator's lifetime. Drop
                // it so the next call reads the restored configuration. The
                // binding stays; only the evaluator goes.
                self.engine.evict_session_evaluator(scope);
                restored.map_err(|e| format!("binding config for {scope}: {e}"))
            }
            DurableWrite::PolicyMetadata {
                tenant,
                content_hash,
                prior,
            } => {
                let restored = if prior.is_empty() {
                    store.delete_policy_metadata(tenant, content_hash)
                } else {
                    store.put_policy_metadata(tenant, content_hash, prior)
                };
                // Tenant-wide counterpart of the eviction below: an evaluator
                // that spawned while the rolled-back config was in place is
                // seeded from it for its lifetime, and the on-success evictor
                // never ran because the rollout never published.
                self.engine.evict_tenant_evaluators(tenant);
                restored.map_err(|e| format!("tenant config for {tenant}: {e}"))
            }
            DurableWrite::ClearedPins(pins) => {
                // Every pin is attempted. Stopping at the first failure would
                // leave the tenant *more* split than not rolling back at all.
                let failed: Vec<String> = pins
                    .iter()
                    .filter_map(|(scope, hash)| {
                        store
                            .put_policy_binding(scope, hash)
                            .err()
                            .map(|e| format!("pin on {scope}: {e}"))
                    })
                    .collect();
                if failed.is_empty() {
                    Ok(())
                } else {
                    Err(failed.join("; "))
                }
            }
            DurableWrite::ClearedBindingConfig { tenant, rows } => {
                let restored = store.restore_binding_metadata(rows);
                // Same hazard, tenant-wide: between the clear and this restore
                // an idle session could have respawned, found its binding row
                // gone, and seeded from the tenant-wide fallback instead.
                self.engine.evict_tenant_evaluators(tenant);
                restored.map_err(|e| format!("per-session config: {e}"))
            }
            DurableWrite::SessionOwner(token) => self.release_created_owner(token),
        }
    }

    /// Roll the durable writes back and return the error to give the caller.
    ///
    /// When the rollback itself fails there is nothing further to try, so
    /// say so in the error: the operator needs to know that the change they
    /// were told did not happen may still appear after a restart.
    #[allow(clippy::result_large_err)]
    fn rollback_then(&self, steps: Vec<DurableWrite>, err: Status) -> Status {
        let failures: Vec<String> = steps
            .iter()
            .rev()
            .filter_map(|s| self.undo_durable(s).err())
            .collect();
        if failures.is_empty() {
            return err;
        }
        tracing::error!(failures = ?failures, "rollback after a failed publication failed");
        Status::internal(format!(
            "{}; the stored state could not be restored ({}), so this change may take \
             effect after a restart despite this error",
            err.message(),
            failures.join("; ")
        ))
    }

    /// Persist `(scope) → content_hash` so a restart resumes this bind.
    ///
    /// Returns an error. Publishing the in-memory bind and then failing to
    /// record it reports success for a binding that disappears at the next
    /// restart, dropping the session back to the tenant default — a silent
    /// reversal of what the caller was told happened.
    #[allow(clippy::result_large_err)]
    fn persist_binding(&self, scope: &SessionScope, content_hash: &str) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        store.put_policy_binding(scope, content_hash).map_err(|e| {
            tracing::error!(error = %e, "persist_binding failed");
            Status::internal(format!("could not persist the binding for {scope}: {e}"))
        })?;
        Ok(())
    }

    /// Persist `tenant → content_hash` as the tenant default.
    ///
    /// Returns an error, for the same reason as [`Self::persist_binding`]: a
    /// default that is live but unrecorded reverts on restart while the
    /// rollout is reported as successful.
    #[allow(clippy::result_large_err)]
    fn persist_tenant_default(&self, tenant: &str, content_hash: &str) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        store
            .put_tenant_default_policy(tenant, content_hash)
            .map_err(|e| {
                tracing::error!(error = %e, "persist_tenant_default failed");
                Status::internal(format!(
                    "could not persist the tenant default for {tenant}: {e}"
                ))
            })?;
        Ok(())
    }

    /// Drop every persisted session pin in `tenant`, for a `Force` rollout.
    /// [`SessionEvaluatorMap::evict_tenant`] clears the in-memory bindings,
    /// but without this the persisted ones survive and boot replay
    /// resurrects the pre-Force pins after a restart, silently reverting the
    /// rollout for exactly the sessions it was meant to move.
    ///
    /// Errors propagate: a force update that cannot clear the old pins has
    /// not rolled anything out. Returns the pins it removed, so a rollout
    /// that then fails to publish can put them back.
    #[allow(clippy::result_large_err)]
    fn clear_persisted_bindings_for_tenant(
        &self,
        tenant: &str,
    ) -> Result<Vec<(SessionScope, String)>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(Vec::new());
        };
        let bindings = store.all_policy_bindings().map_err(|e| {
            tracing::error!(tenant, error = %e, "clear persisted bindings: list failed");
            Status::internal(format!("could not list policy bindings for {tenant}: {e}"))
        })?;
        let mut removed: Vec<(SessionScope, String)> = Vec::new();
        for (scope, hash) in bindings.into_iter().filter(|(s, _)| s.tenant() == tenant) {
            if let Err(e) = store.delete_policy_binding(&scope) {
                tracing::error!(scope = %scope, error = %e, "clear persisted binding failed");
                // Put back what this call already removed. Otherwise a
                // rollout reported as failed has still un-pinned some of the
                // tenant's sessions and left the rest pinned — a split that
                // only shows up at the next restart.
                for (s, h) in &removed {
                    if let Err(e) = store.put_policy_binding(s, h) {
                        tracing::error!(scope = %s, error = %e, "restoring a cleared pin failed");
                    }
                }
                return Err(Status::internal(format!(
                    "could not clear the pin on {scope}: {e}"
                )));
            }
            removed.push((scope, hash));
        }
        Ok(removed)
    }

    /// Persist the source bundle keyed on `content_hash`.
    ///
    /// Returns an error. Boot replay recompiles from this, so a policy that
    /// is live but whose source was not stored becomes a binding pointing at
    /// nothing after a restart.
    ///
    /// The write is non-destructive: the key is a content hash, so an
    /// occupied key already holds the bytes that hash to it. An absent key is
    /// written, a key holding the same content is left as it is, and a key
    /// holding DIFFERENT content is a hash ambiguity — two contents claiming
    /// one hash, i.e. a collision or a caller writing under a key it did not
    /// derive from these bytes. That refuses the request rather than
    /// replacing a source other sessions may be bound to and boot replay
    /// recompiles from. The store enforces the same rule
    /// ([`sasy_graph::persistence::RocksStore::put_policy_source_in_class`]),
    /// so a caller that skipped this check cannot overwrite either.
    #[allow(clippy::result_large_err)]
    fn persist_policy_source(
        &self,
        content_hash: &str,
        value: sasy_graph::PersistedPolicy,
    ) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        // The class the record is keyed under: the value's, which is what
        // `put_policy_source` derives the key from.
        let class = value.functor_admission;
        let stored = store
            .get_policy_source_in_class(content_hash, class)
            .map_err(|e| {
                tracing::error!(error = %e, "persist_policy_source: reading the stored record failed");
                Status::internal(format!(
                    "could not read the stored policy source for {content_hash}: {e}"
                ))
            })?;
        if let Some(stored) = stored {
            if stored.policy_source == value.policy_source
                && stored.functor_source == value.functor_source
                && stored.backend == value.backend
            {
                // Same content under the same key: nothing to write.
                return Ok(());
            }
            tracing::error!(
                content_hash = %content_hash,
                key_class = ?class,
                stored_class = ?stored.functor_admission,
                offered_class = ?value.functor_admission,
                "content hash ambiguity: the key already holds DIFFERENT policy source \
                 bytes; refusing the upload rather than overwriting the stored record"
            );
            return Err(Status::internal(format!(
                "policy source {content_hash} is already stored with different content in \
                 class {class:?}; refusing to overwrite it"
            )));
        }
        store.put_policy_source(content_hash, &value).map_err(|e| {
            tracing::error!(error = %e, "persist_policy_source failed");
            Status::internal(format!(
                "could not persist the policy source for {content_hash}: {e}"
            ))
        })?;
        Ok(())
    }

    /// Persist the *tenant-wide* config facts that travel with `content_hash`,
    /// i.e. those supplied on a Default/Force bind. Read back at per-session
    /// bootstrap when the session's own binding carries none, and seeded into
    /// the `PolicyMetadata` EDB. Empty is treated as "no change", not a wipe:
    /// a client re-sends the full set on every pin, and a stray config-less
    /// upload of the same source must not clear an existing config.
    ///
    /// Config supplied on a *session*-scoped bind does not belong here — an
    /// identical source is one content hash however many sessions bind it, so
    /// this key is shared and last-writer-wins. Use
    /// [`Self::persist_binding_metadata`] for that. No-op without a graph store.
    ///
    /// Returns an error rather than logging one, for the same reason
    /// [`Self::persist_binding_metadata`] does: a policy can derive its whole
    /// sink/source taxonomy from these facts, and one published without them
    /// then has no sinks and denies nothing. Reporting success for that is
    /// worse than refusing. (A policy in the shape of the shipped guard
    /// profile errs the other way — its rule groups are on unless a `rule_off`
    /// names them, so missing config is over-strict. Either way the seed has
    /// to match the binding.)
    #[allow(clippy::result_large_err)]
    fn persist_policy_metadata(
        &self,
        tenant: &str,
        content_hash: &str,
        facts: &[sasy_common::policy_engine::PolicyMetadataFact],
    ) -> Result<(), Status> {
        if facts.is_empty() {
            return Ok(());
        }
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        let persisted: Vec<sasy_graph::PolicyMetadataFact> = facts
            .iter()
            .map(|f| sasy_graph::PolicyMetadataFact {
                rel: f.rel.clone(),
                a: f.a.clone(),
                b: f.b.clone(),
            })
            .collect();
        if let Err(e) = store.put_policy_metadata(tenant, content_hash, &persisted) {
            tracing::error!(content_hash, error = %e, "persist policy metadata failed");
            return Err(Status::internal(format!(
                "could not store enforcement configuration for tenant {tenant}: {e}"
            )));
        }
        Ok(())
    }

    /// Persist config supplied on a *session-scoped* bind, keyed by the binding
    /// `(scope, content_hash)` rather than by the policy source.
    ///
    /// This is the difference that matters: `content_hash` covers the policy
    /// *source*, not the config, so four sessions binding one identical source
    /// with four different configs all address one key. Concurrently they
    /// overwrite each other, and a policy whose rules are gated on metadata
    /// evaluates under a sibling's configuration — where its own tools are not
    /// sinks, its own rules are not enabled, and so nothing matches and nothing
    /// is denied. That failure is silent, permissive, and appears only under
    /// concurrency. Keying by the binding removes the sharing.
    ///
    /// An empty set is *written*, not skipped: a session-scoped bind states this
    /// binding's configuration in full, and "none" is a statement. Recording it
    /// is what stops the session falling back to the shared policy-wide key and
    /// picking up configuration it never asked for. No-op without a graph store.
    ///
    /// Returns an error rather than logging one: a policy can derive its whole
    /// sink/source taxonomy from these facts, so a bind published without them
    /// can yield an evaluator that matches no denial rule while `SetPolicy`
    /// reports success.
    #[allow(clippy::result_large_err)]
    fn persist_binding_metadata(
        &self,
        scope: &SessionScope,
        content_hash: &str,
        facts: &[sasy_common::policy_engine::PolicyMetadataFact],
    ) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        let persisted: Vec<sasy_graph::PolicyMetadataFact> = facts
            .iter()
            .map(|f| sasy_graph::PolicyMetadataFact {
                rel: f.rel.clone(),
                a: f.a.clone(),
                b: f.b.clone(),
            })
            .collect();
        if let Err(e) = store.put_binding_metadata(scope, content_hash, &persisted) {
            tracing::error!(scope = %scope, content_hash, error = %e,
                "persist binding metadata failed");
            return Err(Status::internal(format!(
                "could not store enforcement configuration for {scope}: {e}"
            )));
        }
        Ok(())
    }

    /// Does this request carry an admin identity strong enough to waive the
    /// session-ownership check below?
    ///
    /// Ownership is recorded and compared in the *effective* terms: a
    /// `service-proxy` relay forwards the end user's `x-tenant` /
    /// `x-principal`, and the scope and principal every ownership call site
    /// builds already honour them. The waiver has to be asked in the same
    /// terms. Reading the connection's roles alone would let anyone behind
    /// such a relay reach every session in the tenant they name: a relay
    /// that holds `service-proxy` and `admin` together is admin on the
    /// connection no matter whose call it is relaying.
    ///
    /// The forwarded roles alone are not enough either — `x-roles` is written
    /// by the caller, so a `service-proxy` credential could claim `admin` for
    /// an invented end user. So both identities must be admin: an admin RELAY
    /// forwarding an admin USER. A direct admin with no delegation headers
    /// satisfies both, since without delegation the effective roles ARE the
    /// connection's. Same conjunction as the Default / Force scope gate and
    /// the functor-source gate.
    fn ownership_bypass_is_admin<T>(request: &Request<T>) -> bool {
        sasy_auth::request_has_role(request, sasy_common::roles::ADMIN)
            && sasy_auth::request_effective_has_role(request, sasy_common::roles::ADMIN)
    }

    /// Verify or claim ownership of `scope` for `principal`. On
    /// conflict (an existing principal differs from the caller),
    /// returns `PERMISSION_DENIED` unless `is_admin` is true. With
    /// no graph store wired (test path), this is a no-op.
    ///
    /// Returns a token only for a newly created row. Rollback checks that no
    /// other operation has relied on the claim in the meantime.
    #[allow(clippy::result_large_err)]
    fn enforce_session_ownership(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        is_admin: bool,
    ) -> Result<Option<sasy_graph::OwnerClaimToken>, Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(None);
        };
        match store.claim_session_owner_with_token(scope, principal) {
            Ok((sasy_graph::OwnerClaim::Claimed, token)) => Ok(token),
            Ok((sasy_graph::OwnerClaim::AlreadyHeld, _)) => Ok(None),
            Ok((sasy_graph::OwnerClaim::HeldBy(existing), _)) => {
                if is_admin {
                    Ok(None)
                } else {
                    Err(Status::permission_denied(format!(
                        "session is owned by another principal: {existing}"
                    )))
                }
            }
            Err(e) => Err(Status::internal(format!("ownership check: {e}"))),
        }
    }

    /// Read-only ownership check: deny if another non-admin
    /// principal already owns `scope`, but do **not** claim it.
    /// Use to gate expensive work (compile/install); pair with
    /// [`Self::enforce_session_ownership`] after the work succeeds
    /// so a build failure can't lock a session against its
    /// rightful future owner.
    #[allow(clippy::result_large_err)]
    fn peek_session_ownership_or_deny(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        is_admin: bool,
    ) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        let existing = store
            .get_session_owner(scope)
            .map_err(|e| Status::internal(format!("ownership check: {e}")))?;
        match (existing, principal) {
            (None, _) => Ok(()),
            (Some(e), Some(p)) if e == p => Ok(()),
            (Some(e), _) => {
                if is_admin {
                    Ok(())
                } else {
                    Err(Status::permission_denied(format!(
                        "session is owned by another principal: {e}"
                    )))
                }
            }
        }
    }

    /// Drop the durable state that belongs to `scope`: its pin, its dynamic
    /// facts, and the config supplied on its binds — in one batch.
    ///
    /// Not the owner. Releasing the id is what lets another principal claim
    /// it, and that has to happen after the live evaluator is gone as well as
    /// after these rows are, so it is a separate step at the call site.
    #[allow(clippy::result_large_err)]
    fn purge_session_state(&self, scope: &SessionScope) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        store.delete_session_state(scope).map_err(|e| {
            tracing::error!(scope = %scope, error = %e, "session teardown failed");
            Status::internal(format!("could not tear down {scope}: {e}"))
        })
    }

    /// Undo an untouched claim; another service may already rely on it even
    /// before that service commits any durable state.
    fn release_created_owner(&self, token: &sasy_graph::OwnerClaimToken) -> Result<(), String> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        let result = store.release_session_owner_claim(token);
        store.commit_session_owner_claim(token);
        result
            .map(|_| ())
            .map_err(|e| format!("owner claim rollback: {e}"))
    }

    fn commit_owner_claim(&self, token: &sasy_graph::OwnerClaimToken) {
        if let Some(store) = self.graph_store.as_ref() {
            store.commit_session_owner_claim(token);
        }
    }

    fn release_claim_if_unused(
        &self,
        scope: &SessionScope,
        token: Option<sasy_graph::OwnerClaimToken>,
    ) {
        let Some(token) = token else {
            return;
        };
        if let Err(e) = self.release_created_owner(&token) {
            tracing::error!(scope = %scope, error = %e,
                "could not release unused owner claim; ownership retained");
        }
    }

    /// Release ownership of `scope`, making the id claimable again.
    #[allow(clippy::result_large_err)]
    fn release_session_owner(&self, scope: &SessionScope) -> Result<(), Status> {
        let Some(store) = self.graph_store.as_ref() else {
            return Ok(());
        };
        store.delete_session_owner(scope).map_err(|e| {
            tracing::error!(scope = %scope, error = %e, "drop persisted owner failed");
            Status::internal(format!("could not release ownership of {scope}: {e}"))
        })
    }
}

#[tonic::async_trait]
impl<E: Engine + 'static> PolicyEngineTrait for PolicyService<E> {
    async fn check_authorization(
        &self,
        request: Request<AuthorizationRequest>,
    ) -> Result<Response<AuthorizationResponse>, Status> {
        // Tenant and principal come from auth — wire
        // `tenant_id` and `principal` are ignored. `entity` stays
        // user-supplied (free-form actor).
        // Use the *effective* tenant/principal so split deployments
        // (refmon → engine over gRPC) correctly route per-end-user
        // calls. When the caller is a `service-proxy`-trusted
        // peer it forwards the end-user's tenant + principal via
        // `x-tenant` / `x-principal`; without that role the
        // metadata is ignored and we fall back to the connection's
        // own auth context.
        let tenant = sasy_auth::request_effective_tenant(&request, "default");
        let principal = sasy_auth::request_effective_principal(&request);
        // Roles used in the policy decision must come from the caller's
        // authenticated identity — NOT the wire `roles` field, which any
        // caller controls. Only a trusted `service-proxy` relay may forward
        // the end-user's roles (mirroring the tenant/principal delegation
        // above); every other caller has `req.roles` ignored and the roles
        // resolved server-side from their auth context. Without this, any
        // authenticated low-privilege caller could set roles=[...] and
        // self-elevate for the duration of the check.
        let is_service_proxy = sasy_auth::get_auth_result(&request)
            .map(|a| a.has_role(sasy_common::roles::SERVICE_PROXY))
            .unwrap_or(false);
        let auth_roles: Vec<String> = sasy_auth::get_auth_result(&request)
            .map(|a| a.roles.clone())
            .unwrap_or_default();
        // Captured before `into_inner` for the session-ownership gate below.
        let is_admin = Self::ownership_bypass_is_admin(&request);
        let req = request.into_inner();
        let effective_roles: Vec<String> = if is_service_proxy {
            req.roles.clone()
        } else {
            auth_roles
        };
        let session_id = req.session_id.unwrap_or_default();
        let scope = SessionScope::new(&tenant, session_id.clone());
        // Session-ownership gate: a non-global session bound to one principal
        // must not have its graph state / denial traces read by another
        // same-tenant caller who supplies its id. Mirrors the guard on
        // SetPolicy(SessionTarget) and the observability read path
        // (enforce_session_read). The global ("") session is tenant-shared and
        // unowned, so it is exempt. Admin bypasses; a `service-proxy` relay
        // forwards the end user's principal via `x-principal`, so
        // owner == principal holds for the real owner. `peek_*` is read-only —
        // it must NOT claim ownership on a read — and is a no-op when no graph
        // store is configured.
        if !session_id.is_empty() {
            self.peek_session_ownership_or_deny(&scope, principal.as_deref(), is_admin)?;
        }
        // Sessions are bound to a policy explicitly via SetPolicy.
        // CheckAuthorization takes no inline policy_id — `None` goes
        // to the engine, which looks up the session's binding (or
        // falls through to the tenant default).
        let resp = self
            .engine
            .check_authorization(
                &req.current_node_ids,
                &req.actions,
                req.entity.as_deref(),
                &effective_roles,
                &scope,
                principal.as_deref(),
                None,
            )
            .map_err(|e| {
                // A dispatch can be the moment a persisted policy is first
                // compiled in this process (the lazy install). When the
                // functor gate refuses that compile, the session has no
                // policy installed — a precondition the caller can act on
                // (re-upload as an admin, or restart the binary with the
                // opt-in), not an internal fault. Every other failure keeps
                // the status it had.
                //
                // Matched as a PREFIX, like every other reader of this marker:
                // the compiler echoes the offending policy source into its
                // error, so a source that quotes the marker text and then
                // fails to compile would otherwise be reported as a functor
                // refusal. Every path that wraps the refusal keeps it leading.
                let message = e.to_string();
                if message.starts_with(FUNCTOR_REFUSED_AT_LOAD) {
                    Status::failed_precondition(message)
                } else {
                    Status::internal(message)
                }
            })?;
        Ok(Response::new(resp))
    }

    async fn get_sync_status(
        &self,
        _request: Request<SyncStatusRequest>,
    ) -> Result<Response<SyncStatusResponse>, Status> {
        let status = self.engine.get_sync_status();
        Ok(Response::new(SyncStatusResponse {
            current_sequence: status.current_sequence,
            node_count: status.node_count as i64,
            edge_count: status.edge_count as i64,
            connected: status.connected,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let status = self.engine.get_sync_status();
        let backend = self.engine.backend_name();

        let message = if status.connected {
            format!(
                "backend={} nodes={} edges={} synced",
                backend, status.node_count, status.edge_count
            )
        } else {
            format!("backend={} (graph sync not connected)", backend)
        };

        Ok(Response::new(HealthResponse {
            healthy: true,
            message,
        }))
    }

    async fn set_policy(
        &self,
        request: Request<SetPolicyRequest>,
    ) -> Result<Response<SetPolicyResponse>, Status> {
        // Tenant comes from auth. Sessions can only ever
        // bind policies within their own tenant — there's no wire
        // field for cross-tenant operation.
        //
        // Admin role is required for tenant-wide ops (Default /
        // Force); SessionTarget binds the caller's own session and
        // doesn't need admin. We dispatch the role check based on
        // the scope below.
        // Use the *effective* tenant/principal so split deployments
        // (refmon → engine over gRPC) correctly route per-end-user
        // calls. When the caller is a `service-proxy`-trusted
        // peer it forwards the end-user's tenant + principal via
        // `x-tenant` / `x-principal`; without that role the
        // metadata is ignored and we fall back to the connection's
        // own auth context.
        let tenant = sasy_auth::request_effective_tenant(&request, "default");
        let principal = sasy_auth::request_effective_principal(&request);
        let is_admin = Self::ownership_bypass_is_admin(&request);
        // Look at the scope before consuming `request` so we can
        // dispatch role checks off it without moving it.
        let scope_peek = request
            .get_ref()
            .scope
            .as_ref()
            .and_then(|s| s.target.as_ref());
        let needs_admin = matches!(
            scope_peek,
            Some(PolicyScopeTarget::Default(_)) | Some(PolicyScopeTarget::Force(_))
        );
        // The per-tenant **global** session (`session_id == ""`) is
        // shared by every observer in the tenant — binding a policy
        // to it changes what every session in the tenant sees. Gate
        // this behind admin role, matching tenant-wide ops.
        let needs_admin_for_global_session = matches!(
            scope_peek,
            Some(PolicyScopeTarget::Session(s)) if s.session_id.is_empty()
        );
        if needs_admin || needs_admin_for_global_session {
            // Both identities on the request must be admin, for the same
            // reason the functor gate below asks for both.
            //
            // The tenant this call writes to is the EFFECTIVE tenant: a
            // `service-proxy` relay forwards the end user's `x-tenant`, and
            // `tenant` above already honours it. The relay's own roles
            // therefore answer the wrong question. A relay can hold
            // `service-proxy` and `admin` together, so checking the connection
            // alone would let any user behind such a relay replace their
            // tenant's default policy —
            // an allow-all default turns enforcement off for that tenant —
            // or Force-roll it and evict every live session in it.
            //
            // The delegated roles alone are not enough either: `x-roles` is
            // written by the caller, so any holder of a `service-proxy`
            // credential could claim `admin` for an invented end user. So we
            // require an admin RELAY forwarding an admin USER. A direct admin
            // with no delegation headers satisfies both, since without
            // delegation the effective roles are the connection's.
            sasy_auth::check_request_role(&request, sasy_common::roles::ADMIN)?;
            if !sasy_auth::request_effective_has_role(&request, sasy_common::roles::ADMIN) {
                return Err(Status::permission_denied(
                    "tenant-wide policy scopes (Default / Force) and the per-tenant global \
                     session require the 'admin' role for the identity the request speaks \
                     for, not only for the connection. A 'service-proxy' relay forwarding \
                     an end user must forward that user's roles in 'x-roles', and they must \
                     include 'admin'.",
                ));
            }
        } else if matches!(scope_peek, Some(PolicyScopeTarget::Session(_))) {
            // Non-admin SessionTarget: still requires the
            // `reference-monitor-user` role so unprivileged
            // entities (e.g. an observability-only key) can't
            // mint policies for sessions in their tenant. Admin
            // would have already been accepted above; this is
            // the floor for everyone else.
            // `check_request_any_role`, not a hand-rolled
            // `if let Some(auth) = …`: the latter waves through a
            // request that carries no auth context at all, so a
            // service an embedder wired up without the auth
            // interceptor would hand session policy binding to a
            // caller nobody authenticated.
            sasy_auth::check_request_any_role(
                &request,
                &[
                    sasy_common::roles::ADMIN,
                    sasy_common::roles::REFERENCE_MONITOR_USER,
                ],
            )?;
        }

        // Custom C++ functors are a separate privilege from uploading a
        // policy, and a much larger one: `functor_source` is written to
        // `functors.cpp`, compiled by g++, and dynamically loaded into the
        // evaluator, so it is arbitrary native code running as whoever runs
        // `sasy serve`. A session-scoped SetPolicy needs only
        // `reference-monitor-user`, which is the floor for any agent the
        // reference monitor fronts — far below "may run code on this host".
        //
        // The sandbox does not close the gap on its own. Compile and
        // evaluator run under `bwrap --unshare-all --clearenv` only on Linux
        // with a working bubblewrap; where bwrap is absent (macOS, a Linux
        // host without the package) or `DISABLE_BWRAP=1` is set, the caller's
        // C++ runs unconfined as the service user. So the default is
        // admin-only everywhere, and the operator opt-in
        // (`--allow-user-functors`) still refuses a non-admin on a host with
        // no sandbox — there, "confined" would be a claim with nothing behind
        // it.
        //
        // `ValidatePolicy` does not compile functor source, so this gate
        // belongs here and only here.
        //
        // Whose `admin` this asks about matters, and the answer here is BOTH
        // identities on the request.
        //
        // The connection's own roles are not enough. A `service-proxy` relay —
        // a relay can hold both `service-proxy` and `admin` —
        // forwards the end user via `x-tenant` / `x-principal`, and the top of
        // this handler already routes by that delegated identity; testing only
        // the relay's roles would let any user behind such a relay load
        // native code into the engine.
        //
        // The delegated roles are not enough either, and are the weaker half:
        // `x-roles` is metadata the CALLER writes. Any principal holding a
        // `service-proxy` credential — a compromised relay, a second gateway,
        // the shipped `reference-monitor` entity, which has `service-proxy`
        // but not `admin` — can put `admin` in that header and speak for a
        // made-up end user. Trusting it alone would make `service-proxy`
        // silently equivalent to `admin` for running native code.
        //
        // So a functor upload needs an admin RELAY forwarding an admin USER:
        // the connection is authenticated as admin by the auth provider, and
        // the identity it says it speaks for claims admin too. A direct admin
        // (no delegation headers) satisfies both, since without delegation the
        // effective roles ARE the connection's.
        let functor_caller_is_admin =
            sasy_auth::request_has_role(&request, sasy_common::roles::ADMIN)
                && sasy_auth::request_effective_has_role(&request, sasy_common::roles::ADMIN);
        //
        // What the gate decides here is also written down. The class this
        // upload is admitted under — admin, or user — is persisted with the
        // source, and every later load of that source asks the same question
        // again against the settings in force then. An operator who turns the
        // opt-in back off is not left with yesterday's uploads still
        // compiling on every restart.
        //
        // "Persisted with the source" means every accepted upload that carries
        // functor bytes, including one whose content THIS PROCESS has already
        // compiled and which therefore takes the dedup short-circuit below.
        // (Dedup is per process: it asks the registry what is installed here
        // and now, not what is on disk. A source the load-time gate refused
        // was not compiled here, so an admin re-upload of it does not dedup —
        // it compiles, like any other upload of content this process has not
        // seen.) The class is part of the storage key, so an admin re-upload
        // of content a user uploaded first writes a second, independent admin
        // record instead of touching the user's — that is what makes "or
        // re-upload the policy as an admin" an actual remediation for a
        // source refused at load.
        //
        // An upload with NO functor source has nothing to admit, and the load
        // paths never gate one (`replay::admitted_policy_source` and
        // `replay::install_from_source` both skip the gate on empty source).
        // Such an upload keeps the plain class, whoever sent it: the class is
        // part of the storage key, so recording an admin one would store a
        // second identical copy of the same bytes under a second key for no
        // effect.
        let functor_admission = if request.get_ref().functor_source.is_empty() {
            FunctorAdmission::User
        } else if functor_caller_is_admin {
            FunctorAdmission::Admin
        } else {
            FunctorAdmission::User
        };
        let mut admitted_unsandboxed_user_functors = false;
        if !request.get_ref().functor_source.is_empty() && !functor_caller_is_admin {
            match self.config.user_functors {
                UserFunctors::Refuse => {
                    return Err(Status::permission_denied(
                        "custom functor source requires the 'admin' role: it is compiled \
                         and loaded as native code in the policy engine's own process. An \
                         operator can allow non-admin callers to supply functors by \
                         starting the binary with --allow-user-functors (or \
                         SASY_ALLOW_USER_FUNCTORS=1), which is only advisable on a host \
                         where the bubblewrap sandbox is available. The Python SDK \
                         attaches a companion <policy>_functors.cpp / functors.cpp file \
                         automatically when it binds a policy from a path, which is why \
                         a plain session bind can carry functor source.",
                    ));
                }
                // Asked here rather than read off a startup snapshot: this
                // decision is only as good as the confinement that exists when
                // the compile runs, a few lines below.
                UserFunctors::Sandboxed if !self.config.sandbox_available_now() => {
                    return Err(Status::permission_denied(
                        "custom functor source from a non-admin caller is refused because \
                         this host has no working sandbox: bubblewrap is absent or \
                         disabled (DISABLE_BWRAP=1), so the supplied C++ would compile and \
                         run unconfined as the user running sasy serve. Install bubblewrap \
                         and leave it enabled, start the binary with \
                         --allow-user-functors=unsandboxed if this is a development host \
                         where that is acceptable, or upload functors with the 'admin' \
                         role.",
                    ));
                }
                UserFunctors::Sandboxed => {}
                // Admitted, and loud about it: the source is about to be
                // compiled and loaded as native code with no confinement
                // around it. The hash identifies the upload without putting
                // the caller's C++ in the log.
                UserFunctors::Unsandboxed => admitted_unsandboxed_user_functors = true,
            }
        }

        let req = request.into_inner();
        let scope_target = req
            .scope
            .and_then(|s| s.target)
            .ok_or_else(|| Status::invalid_argument("scope is required"))?;

        // SessionTarget: read-only ownership pre-check. We don't
        // claim ownership yet — a failed compile shouldn't lock
        // the session against a future legitimate owner. The
        // atomic claim happens post-install, just before the bind
        // (see below).
        if let PolicyScopeTarget::Session(ref s) = scope_target {
            let session_scope = SessionScope::new(&tenant, s.session_id.clone());
            self.peek_session_ownership_or_deny(&session_scope, principal.as_deref(), is_admin)?;
        }

        let current_backend = self.engine.backend_name();
        let mut target_backend = if req.backend.is_empty() {
            current_backend.clone()
        } else {
            req.backend.clone()
        };
        // An empty `backend` means "use the engine's current backend" (the SDK
        // sends this by default). A fresh tenant with no policy installed yet
        // reports `"uninitialized"`, which isn't a real backend — fall back to
        // compiled souffle (the standard backend, and the deny-all bootstrap's)
        // rather than rejecting the upload with "Backend 'uninitialized' …".
        if target_backend.is_empty() || target_backend == "uninitialized" {
            target_backend = sasy_common::Backend::Souffle.as_str().to_string();
        }

        info!(
            "SetPolicy: {} bytes, backend={} (current={}, scope={:?}, tenant={})",
            req.policy_source.len(),
            target_backend,
            current_backend,
            scope_target,
            tenant,
        );

        // A process ID and monotonic counter isolate concurrent SetPolicy
        // workspaces, preventing policy/functor files from being overwritten.
        let upload_id = next_upload_id();
        let work_root = std::path::Path::new("/tmp/sasy-uploads").join(&upload_id);
        std::fs::create_dir_all(&work_root)
            .map_err(|e| Status::internal(format!("Create workspace: {}", e)))?;
        let policy_path = work_root.join("policy.dl").to_string_lossy().into_owned();
        std::fs::write(&policy_path, &req.policy_source)
            .map_err(|e| Status::internal(format!("Write policy: {}", e)))?;

        let custom_functor_path = if !req.functor_source.is_empty() {
            let path = work_root.join("functors.cpp");
            std::fs::write(&path, &req.functor_source)
                .map_err(|e| Status::internal(format!("Write functors: {}", e)))?;
            Some(path.to_string_lossy().into_owned())
        } else {
            None
        };

        let response_backend = target_backend.clone();
        let tenant_for_install = tenant.clone();
        // Map proto scope to engine InstallMode. SessionTarget
        // installs as a variant — the per-session bind is a
        // follow-up call below after the install returns the id.
        let install_mode = match &scope_target {
            PolicyScopeTarget::Session(_) => crate::engine::InstallMode::Variant,
            PolicyScopeTarget::Default(_) => crate::engine::InstallMode::Default,
            PolicyScopeTarget::Force(_) => crate::engine::InstallMode::Force,
        };
        // Pull session_id out before scope_target is moved into
        // the closure (it's needed for the post-install bind step).
        let session_target_id: Option<String> = match &scope_target {
            PolicyScopeTarget::Session(s) => Some(s.session_id.clone()),
            _ => None,
        };
        // Source bytes the persistence layer needs after the build
        // closure has consumed `req`. Cheap clone — only happens
        // once per non-deduped upload.
        let policy_source_for_persist = req.policy_source.clone();
        let functor_source_for_persist = req.functor_source.clone();
        let backend_for_persist = target_backend.clone();
        // Static config facts travel with the policy (keyed by the
        // content hash below, but NOT part of it). Cloned here so both
        // the dedup-hit short-circuit and the fresh-compile path can
        // persist them after `req` is moved into the build closure.
        let policy_metadata_for_persist = req.policy_metadata.clone();

        // Content hash for upload dedup. Same source +
        // functors + backend + magic-set env upload from the same
        // tenant collapses to one registry entry. Different from
        // the binary cache key (which uses *desugared* source) —
        // this one uses *raw* source so we can short-circuit the
        // sugar.py + souffle/g++ pipeline entirely on a registry
        // hit. Sugar.py is deterministic for a given input, so
        // raw-source hash equality implies desugared-source
        // equality, which implies binary-cache equality.
        let magic_set = std::env::var("SASY_SOUFFLE_MAGIC_SET").unwrap_or_default();
        // Bind-by-hash mode: the caller supplies the content hash of an
        // already-installed (baked) profile instead of its source, so the
        // engine binds without recompiling and the source is never shipped.
        // The supplied hash flows through the SAME lock check + dedup-bind path
        // below, so it is safe-by-construction — a non-locked hash is rejected
        // by the policy lock, and a hash with no installed policy is rejected
        // (see the bind-miss guard after the dedup block).
        let content_hash = if !req.bind_content_hash.is_empty() {
            req.bind_content_hash.clone()
        } else if !req.bind_profile_name.is_empty() {
            // Bind-by-name: resolve the baked profile name to its content hash
            // via the registry name index, then flow through the SAME policy
            // lock + dedup-bind path as bind-by-hash. A name with no installed
            // profile is rejected here (no compile), mirroring the hash miss.
            match self
                .engine
                .lookup_policy_by_name(&tenant, &req.bind_profile_name)
            {
                Some(h) => h,
                None => {
                    return Err(Status::failed_precondition(format!(
                        "no installed policy for bind_profile_name '{}' (tenant={})",
                        req.bind_profile_name, tenant
                    )));
                }
            }
        } else {
            crate::hash::upload_content_hash(
                &target_backend,
                &magic_set,
                &req.policy_source,
                &req.functor_source,
            )
        };

        // One line per admitted unsandboxed upload. Emitted here rather than
        // at the gate because the content hash — the only way to recognise
        // this upload later, in the graph store and in the load-time refusals
        // — is computed here. The source itself is never logged.
        if admitted_unsandboxed_user_functors {
            tracing::warn!(
                tenant = %tenant,
                content_hash = %content_hash,
                "--allow-user-functors=unsandboxed: admitting custom C++ functor \
                 source from a caller who is not an admin; it will be compiled and \
                 run with no sandbox around it, as the user running sasy serve"
            );
        }

        // Policy lock (restricted build): the only bindable policy is the
        // curated profile selected at startup. Reject any other source BEFORE
        // the dedup short-circuit below — otherwise a baked permissive profile
        // (e.g. allow-all) would dedup-hit its pre-installed entry and bind,
        // letting a co-located caller disable enforcement for its own session.
        // Re-binding the locked profile itself is allowed (same hash), so the
        // client can still (re-)pin it and refresh per-session metadata
        // (rule_off/rule_on, cooldowns), which is keyed by — but not part of —
        // the content hash.
        if let Some(allowed) = &self.locked_policy_hashes {
            if !allowed.contains(&content_hash) {
                return Err(Status::permission_denied(
                    "policy is locked: this build enforces the curated profiles baked \
                     at startup and does not allow binding any other policy",
                ));
            }
        }

        // Short-circuit: if the same content has been uploaded
        // already in this tenant, return the existing policy_id
        // without recompiling. Saves the ~150ms sugar.py + the
        // (cheap, but nonzero) binary-cache hardlink + the new
        // registry entry.
        if let Some(existing_id) = self.engine.lookup_policy_by_content(&tenant, &content_hash) {
            // Serialize against any other rollout or bind touching the same
            // state — see `ScopeLocks`. Taken here, after the dedup lookup,
            // so the lookup itself is not serialized; nothing has been read
            // for a rollback or written yet.
            let tenant_lock = self.locks.tenant(&tenant);
            let _rollout = match &scope_target {
                PolicyScopeTarget::Session(_) => {
                    RolloutGuard::Shared(Arc::clone(&tenant_lock).read_owned().await)
                }
                _ => RolloutGuard::Exclusive(Arc::clone(&tenant_lock).write_owned().await),
            };
            let scope_lock = session_target_id
                .as_ref()
                .map(|sid| self.locks.scope(&SessionScope::new(&tenant, sid.clone())));
            let _scope_guard = match &scope_lock {
                Some(l) => Some(l.lock().await),
                None => None,
            };
            info!(
                tenant = %tenant,
                policy_id = %existing_id,
                "set_policy dedup hit: skipping recompile"
            );
            // Record this upload's admission class even though nothing is
            // compiled on this path. The class is part of the storage key, so
            // this is a write of the record for THIS upload's class: a no-op
            // when that class already holds this content, and a new admin
            // record when a user uploaded the same content first. It never
            // replaces a stored record — different content under an occupied
            // key is a hash ambiguity and fails the request. Without it, an
            // admin's "re-upload the policy as an admin" would record nothing
            // whenever this process happens to have the content installed
            // already, and the next boot with the opt-in off would find only
            // the user record and refuse the policy again.
            //
            // Skipped for bind-by-hash and bind-by-name: those requests name a
            // hash instead of shipping bytes, so there is no source to store
            // and writing one would put an empty record under that key.
            if req.bind_content_hash.is_empty() && req.bind_profile_name.is_empty() {
                self.persist_policy_source(
                    &content_hash,
                    sasy_graph::PersistedPolicy {
                        policy_source: policy_source_for_persist.clone(),
                        functor_source: functor_source_for_persist.clone(),
                        backend: backend_for_persist.clone(),
                        functor_admission,
                    },
                )?;
            }
            // Even on dedup hit we may still need to bind the
            // session — install_dedup is a no-op for the registry,
            // but the (tenant, session) → policy mapping has to be
            // established for SessionTarget regardless.
            if let Some(sid) = &session_target_id {
                let scope = SessionScope::new(&tenant, sid.clone());
                // Atomic claim under the store's owner_claim_lock.
                // If a different principal raced and won between
                // our peek and here, this denies; the registry
                // dedup is harmless leftover work.
                //
                // Staged, because claiming an unowned session writes the
                // ownership row: a bind that fails after this point would
                // otherwise leave the id locked to a caller that never bound
                // anything. Recorded only when the claim was ours to make —
                // the row has a writer outside these locks, so "was it absent
                // a moment ago" is not the same question as "did I set it".
                let mut staged = Staged::new(self);
                let claimed = staged.step(self.enforce_session_ownership(
                    &scope,
                    principal.as_deref(),
                    is_admin,
                ))?;
                if let Some(token) = claimed {
                    staged.record(DurableWrite::SessionOwner(token));
                }
                // Config and pin BEFORE the bind. `set_session_policy`
                // publishes the binding and evicts the live evaluator, so a
                // concurrent check_authorization can respawn in that gap; if
                // the config is not down yet, the new evaluator seeds from
                // the tenant-wide fallback and keeps that seed until it is
                // next evicted. A client that opens a session per task hits
                // this window constantly at high concurrency.
                // Publishing first and recording afterwards has the mirror
                // problem: a write that fails returns an error for a binding
                // that is already authorizing traffic and disappears at the
                // next restart.
                let prior_config =
                    staged.step(self.read_binding_metadata(&scope, &content_hash))?;
                staged.record(DurableWrite::BindingMetadata {
                    scope: scope.clone(),
                    content_hash: content_hash.clone(),
                    prior: prior_config,
                });
                staged.step(self.persist_binding_metadata(
                    &scope,
                    &content_hash,
                    &policy_metadata_for_persist,
                ))?;
                let prior = staged.step(self.read_persisted_binding(&scope))?;
                staged.record(DurableWrite::Binding {
                    scope: scope.clone(),
                    prior,
                });
                staged.step(self.persist_binding(&scope, &content_hash))?;
                staged.step(
                    self.engine
                        .set_session_policy(&scope, &existing_id)
                        .map_err(|e| Status::failed_precondition(e.to_string())),
                )?;
                staged.published();
            }
            // Apply Default/Force in-memory side effects on dedup
            // hit. This short-circuit returns before `install_dedup`
            // is reached, so the explicit-intent default change is
            // applied here; without this call a Default/Force
            // re-upload of an already-installed source would take
            // effect only after a restart.
            // Persist config BEFORE publishing. `promote_to_default` makes
            // the policy live — and Force additionally evicts the tenant's
            // evaluators, so the next authorization rebuilds immediately. An
            // authorization landing between publication and this write seeds
            // an evaluator with the config that was there before, and that
            // seed lasts the evaluator's lifetime.
            //
            // Stale or missing config can either loosen or tighten a policy.
            // For example, a subtractive policy may keep a rule disabled by a
            // stale rule_off fact, while a policy deriving sinks from config
            // may miss all sinks when facts are absent. Always seed config
            // matching the published binding, as on the session-target path.
            if session_target_id.is_none() {
                let mut staged = Staged::new(self);
                if matches!(
                    install_mode,
                    crate::engine::InstallMode::Default | crate::engine::InstallMode::Force
                ) {
                    let force = matches!(install_mode, crate::engine::InstallMode::Force);
                    // Every durable write goes down before
                    // `promote_to_default` publishes — and, under Force,
                    // before it evicts the tenant. Publishing first left an
                    // error to the caller with the rollout live: a pin-clear
                    // failure kept the old pins for replay to restore, and a
                    // default-write failure meant a restart resumed under the
                    // old default. Force must also drop the persisted pins
                    // and the per-session config, or replay restores the old
                    // bindings in preference to the forced default and idle
                    // sessions resume on their pre-rollout configuration.
                    let prior_config =
                        staged.step(self.read_policy_metadata(&tenant, &content_hash))?;
                    staged.record(DurableWrite::PolicyMetadata {
                        tenant: tenant.clone(),
                        content_hash: content_hash.clone(),
                        prior: prior_config,
                    });
                    staged.step(self.persist_policy_metadata(
                        &tenant,
                        &content_hash,
                        &policy_metadata_for_persist,
                    ))?;
                    let prior = staged.step(self.read_persisted_tenant_default(&tenant))?;
                    staged.record(DurableWrite::TenantDefault {
                        tenant: tenant.clone(),
                        prior,
                    });
                    staged.step(self.persist_tenant_default(&tenant, &content_hash))?;
                    if force {
                        let cleared_config =
                            staged.step(self.clear_binding_config_for_tenant(&tenant))?;
                        staged.record(DurableWrite::ClearedBindingConfig {
                            tenant: tenant.clone(),
                            rows: cleared_config,
                        });
                        let cleared_pins =
                            staged.step(self.clear_persisted_bindings_for_tenant(&tenant))?;
                        staged.record(DurableWrite::ClearedPins(cleared_pins));
                    }
                    staged.step(
                        self.engine
                            .promote_to_default(&tenant, &content_hash, force)
                            .map_err(|e| Status::internal(format!("promote_to_default: {e}"))),
                    )?;
                } else {
                    staged.step(self.persist_policy_metadata(
                        &tenant,
                        &content_hash,
                        &policy_metadata_for_persist,
                    ))?;
                }
                staged.published();
            }
            return Ok(Response::new(SetPolicyResponse {
                accepted: true,
                error_output: String::new(),
                message: format!(
                    "Policy deduped to existing entry (tenant={}, backend={})",
                    tenant, response_backend
                ),
                policy_id: existing_id,
            }));
        }

        // Bind-by-hash with no matching installed policy: the caller asked to
        // bind a profile that isn't installed here. Reject explicitly rather
        // than fall through to the restricted reject / compile path (which
        // would try to compile the empty source). Covers both the full and
        // restricted builds.
        if !req.bind_content_hash.is_empty() {
            return Err(Status::failed_precondition(format!(
                "no installed policy for bind_content_hash {} (tenant={})",
                req.bind_content_hash, tenant
            )));
        }

        // Restricted build: the content hash didn't match a pre-installed
        // curated profile, and there is no toolchain to compile a new one.
        // Reject rather than attempt (and fail) to compile.
        if self.restricted {
            return Err(Status::permission_denied(
                "this build accepts only its curated policies; custom policy authoring \
                 requires the full SASY engine",
            ));
        }

        let content_hash_for_persist = content_hash.clone();
        // `policy_path` is moved into the compile closure; the install step
        // re-reads the same file for rule metadata.
        let policy_path_for_metadata = policy_path.clone();
        #[cfg(feature = "compiler")]
        let build_policy =
            move || -> Result<(crate::session_evaluator::EvaluatorFactory, String), String> {
                let be = sasy_common::Backend::from_wire(&target_backend).ok_or_else(|| {
                    format!("Backend '{}' not supported via upload API", target_backend)
                })?;
                match be {
                    sasy_common::Backend::Flowlog => {
                        // flowlog is an experimental external-process evaluator with
                        // its own (static) policy assets; it does not go through the
                        // sugar.py/souffle/g++ compile pipeline that SetPolicy drives.
                        // On a flowlog deployment, omitting `backend` keeps the
                        // current backend, which resolves to "flowlog" here — so give
                        // an actionable error instead of the generic "not supported".
                        return Err("the flowlog backend is experimental and disabled: policy \
                         upload via SetPolicy is not supported for it. Omitting \
                         `backend` on a flowlog deployment keeps the current \
                         (flowlog) backend — pass an explicit `souffle` backend, \
                         or run a Soufflé-backed engine, to use SetPolicy."
                            .to_string());
                    }
                    sasy_common::Backend::Stub => {
                        return Err("the stub backend has no policy to compile or install via \
                         SetPolicy (the stub evaluator ignores policy source)."
                            .to_string());
                    }
                    sasy_common::Backend::Souffle | sasy_common::Backend::SouffleInterpreted => {}
                }
                // Compile only. The factory this returns is inert — nothing is
                // live until `install_policy` puts it in the registry — which is
                // what lets the durable writes below happen first and still be
                // abandoned cleanly if one of them fails.
                crate::evaluator::factory::build_souffle_factory(
                    be,
                    &work_root,
                    std::path::Path::new(&policy_path),
                    custom_functor_path.as_deref().map(std::path::Path::new),
                )
            };
        // Restricted build: the compiler is gated out. This stub never runs at
        // runtime — the restricted reject above returns first — but must
        // compile so spawn_blocking(build_policy) type-checks.
        #[cfg(not(feature = "compiler"))]
        let build_policy =
            move || -> Result<(crate::session_evaluator::EvaluatorFactory, String), String> {
                Err("policy compiler not built into this (restricted) binary".to_string())
            };

        // Synchronous compile. With dedup + binary cache, identical
        // re-uploads are µs; only genuinely-new sources pay the
        // souffle/g++ cost (typically 5–7 s cold, ~150 ms when the binary
        // cache is warm). The hot-reload background path is gone — its
        // swallowed-error semantics caused more confusion than it saved.
        let result = tokio::task::spawn_blocking(build_policy)
            .await
            .map_err(|e| Status::internal(format!("Build task: {}", e)))?;

        let (factory, factory_backend) = match result {
            Ok(pair) => pair,
            Err(e) => {
                // A policy the compiler rejected is not an internal failure,
                // so this is an `Ok` with `accepted: false`. Nothing has been
                // written yet — the config write moved below the build for
                // exactly this reason, and so that the tenant lock is not
                // held across a cold compile.
                return Ok(Response::new(SetPolicyResponse {
                    accepted: false,
                    message: "Policy build failed".to_string(),
                    error_output: e,
                    policy_id: String::new(),
                }));
            }
        };

        // Serialize from here to the publication — see `ScopeLocks`. Taken
        // after the build so a cold compile does not hold up the tenant.
        let tenant_lock = self.locks.tenant(&tenant);
        // Owned, and kept in an `Option`, so a rollout can hand it to the
        // install task below — see the comment there.
        let mut rollout = Some(match install_mode {
            crate::engine::InstallMode::Default | crate::engine::InstallMode::Force => {
                RolloutGuard::Exclusive(Arc::clone(&tenant_lock).write_owned().await)
            }
            crate::engine::InstallMode::Variant => {
                RolloutGuard::Shared(Arc::clone(&tenant_lock).read_owned().await)
            }
        });
        let scope_lock = session_target_id
            .as_ref()
            .map(|sid| self.locks.scope(&SessionScope::new(&tenant, sid.clone())));
        let _scope_guard = match &scope_lock {
            Some(l) => Some(l.lock().await),
            None => None,
        };

        // Everything this call writes is staged from here on — bar the
        // content-addressed source, see `DurableWrite` — so any failure
        // between the first write and the publication puts the store back.
        let mut staged = Staged::new(self);

        // Tenant-wide config is persisted BEFORE the install below, which
        // makes the policy the tenant default — and under Force also evicts
        // the tenant's sessions. An authorization arriving between
        // publication and a later write would seed an evaluator with no
        // config at all and keep that seed for its lifetime, so the rollout's
        // configuration silently would not apply to it. See the dedup path
        // above for which direction that errs in — it depends on the policy.
        // Writing early is safe: the row is keyed by content hash, so nothing
        // reads it unless a policy with that hash goes live.
        if session_target_id.is_none() {
            let prior_config =
                staged.step(self.read_policy_metadata(&tenant, &content_hash_for_persist))?;
            staged.record(DurableWrite::PolicyMetadata {
                tenant: tenant.clone(),
                content_hash: content_hash_for_persist.clone(),
                prior: prior_config,
            });
            staged.step(self.persist_policy_metadata(
                &tenant,
                &content_hash_for_persist,
                &policy_metadata_for_persist,
            ))?;
        }

        // Persistence order: **source first**, then the pointers
        // that reference it. If the server crashes between writing
        // a binding/default and writing the source, replay sees a
        // dangling pointer ("no persisted source for content_hash
        // X") and the affected session/default can't recover
        // until someone re-uploads. Writing the source first
        // narrows the crash window to "source on disk but no
        // binding/default" — which is harmless (the bytes are
        // addressable on next SetPolicy with the same content).
        // Non-destructive — content_hash determines the bytes, so a re-write
        // of the same content is a no-op and a write of DIFFERENT content
        // under an occupied key is refused, which fails this request.
        // No undo is recorded for the source: nothing stored is replaced, and
        // an orphaned entry is unreachable — nothing reads it unless a
        // binding or default names that hash.
        staged.step(self.persist_policy_source(
            &content_hash_for_persist,
            sasy_graph::PersistedPolicy {
                policy_source: policy_source_for_persist,
                functor_source: functor_source_for_persist,
                backend: backend_for_persist,
                // The class the gate above admitted this upload under. Read
                // back by boot replay and by the lazy install, which re-run
                // the gate against the settings in force at that moment.
                functor_admission,
            },
        ))?;
        // Config travels with the *binding*, not with the source, whenever the
        // request names a session: one source is one content hash however many
        // sessions bind it, so a shared key would let concurrent sessions
        // overwrite each other's config. A Default/Force bind is tenant-wide
        // and keeps the (tenant, content-hash) key. Either way it is persisted
        // so a restart or any later bootstrap re-seeds it — and it is written
        // *before* the bind below is published, so a concurrent
        // check_authorization cannot spawn an evaluator that misses it.
        // The session-scoped write happens after the ownership claim
        // below; the tenant-wide one already happened before the build.
        // Under Default/Force the install below publishes: it makes this the
        // tenant default, and Force additionally evicts every live session so
        // the next call rebinds. Every row therefore goes down first. `Force`
        // must also drop the persisted per-session pins and config, or boot
        // replay restores the pre-Force pins after a restart and idle
        // sessions resume on their pre-rollout configuration — silently
        // reverting the rollout for exactly the sessions it was meant to
        // move. `Default` (gradual) deliberately leaves both intact.
        if matches!(
            install_mode,
            crate::engine::InstallMode::Default | crate::engine::InstallMode::Force
        ) {
            let prior = staged.step(self.read_persisted_tenant_default(&tenant))?;
            staged.record(DurableWrite::TenantDefault {
                tenant: tenant.clone(),
                prior,
            });
            staged.step(self.persist_tenant_default(&tenant, &content_hash_for_persist))?;
        }
        if matches!(install_mode, crate::engine::InstallMode::Force) {
            let cleared_config = staged.step(self.clear_binding_config_for_tenant(&tenant))?;
            staged.record(DurableWrite::ClearedBindingConfig {
                tenant: tenant.clone(),
                rows: cleared_config,
            });
            let cleared_pins = staged.step(self.clear_persisted_bindings_for_tenant(&tenant))?;
            staged.record(DurableWrite::ClearedPins(cleared_pins));
        }

        // Publish. `mode` controls whether this is a pinnable variant only,
        // sets the tenant default, or sets default + evicts. `content_hash`
        // deduplicates identical re-uploads — same source twice is one
        // registry entry. Everything fallible that this policy's durability
        // depends on has already succeeded, so a failure here means nothing
        // went live and the writes above are put back.
        let engine_for_install = Arc::clone(&self.engine);
        let install_hash = content_hash_for_persist.clone();
        let install_tenant = tenant_for_install.clone();
        let metadata_path = policy_path_for_metadata.clone();
        // Past this point the publication is out of our hands if the caller
        // goes away; see `Staged::publishing`.
        staged.publishing();
        // A rollout hands its tenant guard to the install task. `spawn_blocking`
        // detaches when its handle is dropped, so on cancellation the install
        // still runs and still sets the tenant default — and if the lock went
        // with this future, a second rollout could acquire it, publish, be
        // told it succeeded, and then be silently overwritten when the first
        // install finishes. Holding the lock inside the task keeps the last
        // publication and the last durable write the same one.
        //
        // A `Variant` install publishes no default, so its guard stays here:
        // the session bind below still needs it, and a detached variant
        // install is harmless (a content-deduped registry entry nobody names).
        let handed_guard = match install_mode {
            crate::engine::InstallMode::Default | crate::engine::InstallMode::Force => {
                rollout.take()
            }
            crate::engine::InstallMode::Variant => None,
        };
        let joined = tokio::task::spawn_blocking(move || {
            // Returned, not just held: the guard has to outlive the
            // rollback too. Dropping it when this closure ends would free
            // the tenant the instant an install *fails*, letting another
            // rollout publish before the rollback below restores the
            // prior state over it.
            let held = handed_guard;
            // Caught, so a panic returns the guard with the result instead
            // of dropping it as the stack unwinds. A guard dropped there
            // frees the tenant for the whole unwind, letting a queued
            // rollout acquire it and publish — and the rollback below
            // would then restore this call's stale priors over that
            // rollout's committed rows.
            //
            // Catching treats a panic as "did not publish", which holds
            // only because everything after the publication inside this
            // closure is infallible: the rule-metadata load runs first,
            // `install_dedup` and `promote_to_default` are map writes
            // under parking_lot (which does not poison), and the tail is
            // logging. A panic AFTER publication would be rolled back in
            // the store while staying live in the registry. Anything
            // added below the install here has to preserve that.
            let installed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Rule metadata FIRST. It is keyed by content hash and
                // does not touch the registry, so it can fail here
                // without anything having been published — which is the
                // whole point: run after the install, its failure would
                // have left the policy live as the tenant default (and
                // the tenant already evicted under Force) while the
                // rollback put the stored state back, so the caller is
                // told the rollout failed and a restart silently reverts
                // live traffic.
                engine_for_install
                    .load_rule_metadata_for_policy(
                        &install_hash,
                        std::path::Path::new(&metadata_path),
                    )
                    .map_err(|e| format!("Load metadata: {}", e))?;
                engine_for_install
                    .install_policy(
                        &install_tenant,
                        &install_hash,
                        factory,
                        factory_backend,
                        install_mode,
                    )
                    .map_err(|e| format!("Install: {}", e))
            }))
            .unwrap_or_else(|panic| {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".to_string());
                Err(format!("Install panicked: {what}"))
            });
            (installed, held)
        })
        .await;

        // Take the guard back before anything below can roll back. On the
        // cancellation path this future is already gone and the guard went
        // with the task, which is the point.
        let installed = match joined {
            Ok((installed, held)) => {
                // Only a rollout handed its guard over. Assigning
                // unconditionally would overwrite the `Variant` path's guard
                // — still held here, because the session bind below needs it
                // — with the `None` that path put in.
                if held.is_some() {
                    rollout = held;
                }
                if installed.is_ok() {
                    staged.publication_succeeded();
                } else {
                    staged.publication_failed();
                }
                installed
            }
            // A panic in the install task is an error like any other, and has
            // to roll back like one. The guard died with the task, so retake
            // the tenant before restoring: the rollback must not race a
            // rollout that slipped in while the lock was briefly free.
            Err(e) => {
                let _retaken = match install_mode {
                    crate::engine::InstallMode::Default | crate::engine::InstallMode::Force => {
                        Some(RolloutGuard::Exclusive(
                            Arc::clone(&tenant_lock).write_owned().await,
                        ))
                    }
                    crate::engine::InstallMode::Variant => None,
                };
                staged.publication_failed();
                staged.step(Err(Status::internal(format!("Install task: {}", e))))?;
                unreachable!("step on Err always returns Err");
            }
        };

        let policy_id = staged.step(installed.map_err(Status::internal))?;
        let message = format!(
            "Policy applied (backend={}, tenant={}, mode={:?})",
            response_backend, tenant_for_install, install_mode
        );

        // SessionTarget: bind the freshly-installed policy to the
        // requested session. install_policy(Variant) gave us the id;
        // set_session_policy creates / updates the (tenant, session)
        // → policy mapping. `session_id == ""` is allowed and binds
        // the per-tenant global session.
        if let Some(sid) = &session_target_id {
            let scope = SessionScope::new(&tenant, sid.clone());
            // Atomic ownership claim. Pre-compile we only peeked,
            // so a parallel SetPolicy on the same session may have
            // raced; the claim under owner_claim_lock either wins
            // (we proceed to bind) or denies (the just-completed
            // registry install is harmless, since the policy
            // dedupes by content for future callers).
            let claimed = staged.step(self.enforce_session_ownership(
                &scope,
                principal.as_deref(),
                is_admin,
            ))?;
            if let Some(token) = claimed {
                staged.record(DurableWrite::SessionOwner(token));
            }
            // Only now, with ownership established. Writing before the
            // claim lets a caller whose SetPolicy is about to be denied
            // leave its configuration behind for the rightful owner's
            // next evaluator to seed from. Still ahead of the bind, so a
            // concurrent check_authorization cannot miss it.
            let prior_config =
                staged.step(self.read_binding_metadata(&scope, &content_hash_for_persist))?;
            staged.record(DurableWrite::BindingMetadata {
                scope: scope.clone(),
                content_hash: content_hash_for_persist.clone(),
                prior: prior_config,
            });
            staged.step(self.persist_binding_metadata(
                &scope,
                &content_hash_for_persist,
                &policy_metadata_for_persist,
            ))?;
            // The pin, too, before the bind is published — and put back if
            // the bind fails, so a session the caller was told did not bind
            // is not bound by replay after the next restart.
            let prior = staged.step(self.read_persisted_binding(&scope))?;
            staged.record(DurableWrite::Binding {
                scope: scope.clone(),
                prior,
            });
            staged.step(self.persist_binding(&scope, &content_hash_for_persist))?;
            staged.step(
                self.engine
                    .set_session_policy(&scope, &policy_id)
                    .map_err(|e| Status::failed_precondition(e.to_string())),
            )?;
        }

        // Explicit, so the tenant guard is visibly held until every write and
        // every rollback above is done rather than until an arbitrary end of
        // scope — and so the compiler does not read the reassignments made
        // around the install as dead.
        drop(rollout);
        staged.published();
        Ok(Response::new(SetPolicyResponse {
            accepted: true,
            message,
            error_output: String::new(),
            policy_id,
        }))
    }

    async fn get_evaluator_status(
        &self,
        request: Request<EvaluatorStatusRequest>,
    ) -> Result<Response<EvaluatorStatusResponse>, Status> {
        sasy_auth::check_request_role(&request, sasy_common::roles::ADMIN)?;
        Ok(Response::new(EvaluatorStatusResponse {
            backend: self.engine.backend_name(),
            alive: true,
            policy_path: String::new(),
        }))
    }

    async fn end_session(
        &self,
        request: Request<EndSessionRequest>,
    ) -> Result<Response<EndSessionResponse>, Status> {
        // Use the *effective* tenant/principal so split deployments
        // (refmon → engine over gRPC) correctly route per-end-user
        // calls. When the caller is a `service-proxy`-trusted
        // peer it forwards the end-user's tenant + principal via
        // `x-tenant` / `x-principal`; without that role the
        // metadata is ignored and we fall back to the connection's
        // own auth context.
        let tenant = sasy_auth::request_effective_tenant(&request, "default");
        let principal = sasy_auth::request_effective_principal(&request);
        let is_admin = Self::ownership_bypass_is_admin(&request);
        let req = request.into_inner();
        if req.session_id.is_empty() {
            return Err(Status::invalid_argument("session_id is required"));
        }
        let scope = SessionScope::new(&tenant, req.session_id);

        // Serialize against a bind or a rollout touching this session — see
        // `ScopeLocks`. Without it a SetPolicy can complete its ownership
        // claim, have this teardown run to completion underneath it, and then
        // write its pin and publish its binding onto an id that now has no
        // owner: the next principal to claim that id inherits the previous
        // one's policy.
        let tenant_lock = self.locks.tenant(&tenant);
        let _rollout = tenant_lock.read().await;
        let scope_lock = self.locks.scope(&scope);
        let _scope_guard = scope_lock.lock().await;

        // Ownership check, UNDER the lock. EndSession is destructive — only
        // the owner (or an admin) may tear a session down — and checking
        // before taking the lock authorizes against state that can change
        // while this task waits at either await: another teardown can release
        // the id and a different principal can claim and bind it, and this
        // call would then destroy that session on the strength of a
        // permission it no longer has. Both SetPolicy paths claim under the
        // lock for the same reason.
        // The check itself claims an unowned id (that is what makes first
        // touch the owner), so a teardown that then fails would leave the id
        // locked to a caller that tore nothing down. Release it on the way
        // out if this call is what created it.
        let claimed = self.enforce_session_ownership(&scope, principal.as_deref(), is_admin)?;

        // Three steps, in this order, because a session id is reusable after
        // EndSession — the next principal that touches it claims ownership
        // fresh, and `peek_session_ownership_or_deny` lets anyone through an
        // ownerless scope.
        //
        // 1. The durable state, atomically. A failure here leaves the session
        //    entirely intact, live and durable, with ownership as it was
        //    before the call — which means released again if this call is
        //    what claimed it.
        if let Err(e) = self.purge_session_state(&scope) {
            self.release_claim_if_unused(&scope, claimed);
            return Err(e);
        }
        // 2. The live evaluator. Releasing the owner first would leave a
        //    window where the scope is ownerless and its evaluator is still
        //    up, so a concurrent CheckAuthorization from any other principal
        //    in the tenant passes the ownership check and is decided by the
        //    departed session's evaluator — under the seed it was built with,
        //    including any `rule_off` relaxations it was granted.
        if let Some(token) = claimed {
            self.commit_owner_claim(&token);
        }
        let was_active = self.engine.end_session(&scope);
        // 3. Only now is the id safe to hand out. If this fails the session
        //    has been torn down but stays owned, and the caller is told so —
        //    the safe direction.
        self.release_session_owner(&scope)?;
        if was_active {
            info!(scope = %scope, "ended session (evaluator dropped)");
        }
        Ok(Response::new(EndSessionResponse { was_active }))
    }

    async fn update_policy_metadata(
        &self,
        request: Request<UpdatePolicyMetadataRequest>,
    ) -> Result<Response<UpdatePolicyMetadataResponse>, Status> {
        // Same effective-tenant routing as EndSession so a split
        // refmon→engine deployment records the detaint against the
        // end-user's scope, not the proxy's connection context.
        let tenant = sasy_auth::request_effective_tenant(&request, "default");
        let principal = sasy_auth::request_effective_principal(&request);
        let is_admin = Self::ownership_bypass_is_admin(&request);
        let req = request.into_inner();
        if req.session_id.is_empty() {
            return Err(Status::invalid_argument("session_id is required"));
        }
        let scope = SessionScope::new(&tenant, req.session_id);
        // Same serialization as EndSession: an append that lands after a
        // teardown leaves a dynamic fact — a detaint approval — attached to a
        // released id, for whoever claims it next to inherit.
        let tenant_lock = self.locks.tenant(&tenant);
        let _rollout = tenant_lock.read().await;
        let scope_lock = self.locks.scope(&scope);
        let _scope_guard = scope_lock.lock().await;
        // Mutating a session's metadata (recording a detaint
        // approval/denial) is owner-or-admin only, like EndSession — and, like
        // EndSession, the check claims an unowned id, so a failure below has
        // to give it back rather than leave it locked.
        let claimed = self.enforce_session_ownership(&scope, principal.as_deref(), is_admin)?;

        let facts: Vec<crate::evaluator::types::PolicyMetadataFact> = req
            .facts
            .into_iter()
            .map(|f| crate::evaluator::types::PolicyMetadataFact {
                rel: f.rel,
                a: f.a,
                b: f.b,
            })
            .collect();
        if facts.is_empty() {
            // A no-op writes nothing, so it must not keep the ownership claim
            // the check above just made: this handler has no role floor, so
            // any same-tenant principal could otherwise stake a claim on a
            // victim's session id with an empty fact list and lock the
            // rightful principal out of every later write, bind, and check.
            // `RegisterEvents` refuses the same trick for the same reason.
            self.release_claim_if_unused(&scope, claimed);
            return Ok(Response::new(UpdatePolicyMetadataResponse {
                accepted: true,
            }));
        }
        let n = facts.len();
        if let Err(e) = self.engine.update_session_metadata(&scope, facts) {
            self.release_claim_if_unused(&scope, claimed);
            return Err(Status::internal(format!("update session metadata: {e}")));
        }
        if let Some(token) = claimed {
            self.commit_owner_claim(&token);
        }
        info!(scope = %scope, facts = n, "recorded session metadata");
        Ok(Response::new(UpdatePolicyMetadataResponse {
            accepted: true,
        }))
    }

    async fn validate_policy(
        &self,
        request: Request<ValidatePolicyRequest>,
    ) -> Result<Response<ValidatePolicyResponse>, Status> {
        // Restricted (public) builds ship NO compiler toolchain at runtime.
        // ValidatePolicy compiles caller-supplied source (sugar.py → souffle →
        // g++), so it must be rejected here too — not only in SetPolicy — or it
        // defeats the "no souffle/g++/python at runtime" invariant.
        if self.restricted {
            return Err(Status::permission_denied(
                "this build accepts only its curated policies; policy validation \
                 requires the full SASY engine",
            ));
        }
        // Compiling an arbitrary policy drives the sandboxed sugar.py +
        // souffle toolchain on untrusted input. Require at least the
        // SetPolicy(Session) role floor so an unprivileged / observability-
        // only caller can't use it as a free compile-pipeline DoS or recon
        // surface.
        // A request with no auth context is refused rather than skipping the
        // floor — see the same check in `set_policy`.
        sasy_auth::check_request_any_role(
            &request,
            &[
                sasy_common::roles::ADMIN,
                sasy_common::roles::REFERENCE_MONITOR_USER,
            ],
        )?;

        // RAII cleanup for the per-request /tmp workspace — removed on every
        // exit path of the blocking task (including early error returns).
        struct WorkspaceCleanup(std::path::PathBuf);
        impl Drop for WorkspaceCleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let req = request.into_inner();
        // Bound attacker-supplied policy source so one ValidatePolicy call
        // can't drive an unbounded compile + static-analysis pass.
        const MAX_POLICY_SOURCE_BYTES: usize = 1024 * 1024;
        if req.policy_source.len() > MAX_POLICY_SOURCE_BYTES {
            return Err(Status::invalid_argument(format!(
                "policy source too large: {} bytes (max {MAX_POLICY_SOURCE_BYTES})",
                req.policy_source.len()
            )));
        }
        let run_analyses = req.run_analyses;
        let extra_reach: Vec<String> = req.extra_reach_targets.clone();

        #[cfg(feature = "compiler")]
        let result = tokio::task::spawn_blocking(move || {
            let python = crate::compiler::selected_tool("python3", "python3")
                .map_err(|e| format!("Select Python tool: {e}"))?;
            let souffle = crate::compiler::selected_tool("souffle", "souffle")
                .map_err(|e| format!("Select Soufflé tool: {e}"))?;
            let assets = crate::compiler::SouffleAssets::discover()
                .map_err(|e| format!("Discover assets: {}", e))?;

            // Prepend common_policy.dl and write combined source
            let full_source = crate::compiler::prepend_common_policy(&req.policy_source, &assets)
                .map_err(|e| format!("Prepend common policy: {}", e))?;

            // Per-request workspace so concurrent ValidatePolicy calls
            // don't clobber each other's intermediate files.
            let upload_id = next_upload_id();
            let work_root = std::path::Path::new("/tmp/sasy-validates").join(&upload_id);
            std::fs::create_dir_all(&work_root).map_err(|e| format!("Create workspace: {}", e))?;
            let _cleanup = WorkspaceCleanup(work_root.clone());
            let policy_path = work_root.join("validate_policy.dl");
            std::fs::write(&policy_path, &full_source).map_err(|e| format!("Write: {}", e))?;

            // Step 1: Run sugar.py preprocessor — sandboxed against a
            // malicious .dl driving sugar.py to read host state via
            // an injected `#include`.
            let preprocess = crate::sandbox::sandboxed_command(
                python
                    .to_str()
                    .ok_or("authenticated Python path is not UTF-8")?,
                &[
                    assets.sugar_py.to_str().unwrap(),
                    "--resolve-includes",
                    policy_path.to_str().unwrap(),
                ],
                &work_root,
                true,
                &[assets.sugar_py.as_path()],
            )?
            .output()
            .map_err(|e| format!("Preprocess: {}", e))?;
            if !preprocess.status.success() {
                return Ok((
                    false,
                    String::from_utf8_lossy(&preprocess.stderr).to_string(),
                    String::new(),
                    full_source,
                ));
            }
            let desugared = crate::compiler::apply_policy_inference(&String::from_utf8_lossy(
                &preprocess.stdout,
            ))?;
            let desugared_path = work_root.join("validate_policy_desugared.dl");
            std::fs::write(&desugared_path, &desugared)
                .map_err(|e| format!("Write desugared: {}", e))?;

            // Step 2: Run souffle parse+type check (no codegen) — sandboxed.
            let check = crate::sandbox::sandboxed_command(
                souffle
                    .to_str()
                    .ok_or("authenticated Soufflé path is not UTF-8")?,
                &[
                    "--show=transformed-datalog",
                    desugared_path.to_str().unwrap(),
                ],
                &work_root,
                true,
                &[assets.include_dir.as_path()],
            )?
            .output()
            .map_err(|e| format!("souffle: {}", e))?;

            if check.status.success() {
                Ok((true, String::new(), desugared, full_source))
            } else {
                let err = format!(
                    "{}{}",
                    String::from_utf8_lossy(&check.stderr),
                    String::from_utf8_lossy(&check.stdout),
                );
                Ok((false, err, desugared, full_source))
            }
        })
        .await
        .map_err(|e| Status::internal(format!("Validate task: {}", e)))?;
        // Restricted build: compiler gated out — validate-by-compile is
        // unavailable. The restricted reject above returns first, so this is
        // unreachable at runtime; it only makes the match below type-check.
        #[cfg(not(feature = "compiler"))]
        let result: Result<(bool, String, String, String), String> =
            Err("policy compiler not built into this (restricted) binary".to_string());

        match result {
            Ok((valid, error_output, desugared, full_source)) => {
                // Analyses run on the AST-level pipeline
                // (`analyze_raw`), which preserves source
                // spans on the user's combined source —
                // no desugared-line-mapping indirection.
                // The Soufflé validation step above
                // ensures the source parses; analyze_raw
                // re-parses with our parser and resolves
                // dot-notation/annotations at AST level.
                let analyses = if valid && run_analyses {
                    let extra_refs: Vec<&str> = extra_reach.iter().map(|s| s.as_str()).collect();
                    run_static_analyses(&full_source, &extra_refs)
                } else {
                    None
                };
                Ok(Response::new(ValidatePolicyResponse {
                    valid,
                    error_output,
                    desugared_source: desugared,
                    analyses,
                }))
            }
            Err(e) => Ok(Response::new(ValidatePolicyResponse {
                valid: false,
                error_output: e,
                desugared_source: String::new(),
                analyses: None,
            })),
        }
    }
}

/// Run the four static analyses on a (sugar-bearing)
/// `.dl` source via the AST-native pipeline. Returns
/// `None` if the parser or resolve_dots rejects the
/// source — Soufflé already passed it, so this only
/// fires on a parser bug we should learn about (logged).
fn run_static_analyses(
    raw_source: &str,
    extra_reach_targets: &[&str],
) -> Option<sasy_common::policy_engine::AnalysisReport> {
    use sasy_common::policy_engine as pb;
    match crate::analysis::pipeline::analyze_raw(raw_source, "<validate>", extra_reach_targets) {
        Ok(report) => Some(pb::AnalysisReport {
            contradictions: report
                .contradictions
                .into_iter()
                .map(|c| pb::ContradictionFinding {
                    category: c.category,
                    allow_location: c.allow_location,
                    deny_location: c.deny_location,
                    message: c.message,
                    allow_body: c.allow_body,
                    deny_body: c.deny_body,
                })
                .collect(),
            redundancies: report
                .redundancies
                .into_iter()
                .map(|r| pb::RedundancyFinding {
                    head_relation: r.head_relation,
                    redundant_location: r.redundant_location,
                    covered_by_location: r.covered_by_location,
                    redundant_body: r.redundant_body,
                    covered_by_body: r.covered_by_body,
                })
                .collect(),
            reachability: report
                .reachability
                .into_iter()
                .map(|r| pb::ReachabilityFinding {
                    target: r.target,
                    disjuncts: r.disjuncts,
                    opaque: r.opaque,
                    pruned: r.pruned as u32,
                })
                .collect(),
            broad_rules: report.broad_rules,
        }),
        Err(e) => {
            tracing::warn!("analysis pipeline failed on Soufflé-validated source: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::StubEngine;

    fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        source
            .split_once(start)
            .unwrap_or_else(|| panic!("missing source marker: {start}"))
            .1
            .split_once(end)
            .unwrap_or_else(|| panic!("missing source marker: {end}"))
            .0
    }

    #[test]
    fn interpreted_set_policy_uses_selected_python_and_souffle() {
        // The interpreted SetPolicy route builds its evaluator in the factory, so
        // the scan reads that file: markers naming this one match the test itself.
        let source = include_str!("evaluator/factory.rs");
        let route = source_between(
            source,
            "fn build_interpreted_factory(",
            "fn build_compiled_factory(",
        );
        assert!(route.contains("selected_tool(\"python3\", \"python3\")"));
        assert!(route.contains("selected_tool(\"souffle\", \"souffle\")"));
        assert!(!route.contains("\"souffle\".to_string()"));
    }

    #[test]
    fn validate_policy_uses_selected_python_and_souffle() {
        let source = include_str!("service.rs");
        let route = source_between(
            source,
            "async fn validate_policy(",
            "/// Run the four static analyses",
        );
        assert!(route.contains("selected_tool(\"python3\", \"python3\")"));
        assert!(route.contains("selected_tool(\"souffle\", \"souffle\")"));
    }

    #[tokio::test]
    async fn health_reports_backend() {
        let engine = StubEngine::new();
        engine.set_connected(true);
        let svc = PolicyService::new(engine);

        let resp = svc
            .health(Request::new(HealthRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert!(resp.healthy);
        assert!(resp.message.contains("backend="));
    }

    #[tokio::test]
    async fn sync_status() {
        let engine = StubEngine::new();
        engine.set_sequence(42);
        let svc = PolicyService::new(engine);

        let resp = svc
            .get_sync_status(Request::new(SyncStatusRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.current_sequence, 42);
    }

    #[tokio::test]
    async fn check_authorization_stub() {
        let engine = StubEngine::new();
        let svc = PolicyService::new(engine);

        let resp = svc
            .check_authorization(Request::new(AuthorizationRequest {
                current_node_ids: vec!["n1".into()],
                actions: vec![],
                entity: Some("user".into()),
                roles: vec!["admin".into()],
                session_id: Some(String::new()),
                principal: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(resp.results.is_empty());
    }

    /// Server-stamped principal: a [`PolicyService`] passes the
    /// auth-derived principal into the engine's
    /// `check_authorization`, ignoring any `principal` value the
    /// client sent on the wire.
    #[tokio::test]
    async fn check_authorization_uses_server_stamped_principal() {
        use crate::engine::{Engine, GraphUpdate, SyncStatus};
        use parking_lot::Mutex;
        use sasy_auth::AuthResult;
        use std::sync::Arc;

        type CapturedIdentity = (Option<String>, Option<String>, String);

        struct CapturingEngine {
            seen: Arc<Mutex<Vec<CapturedIdentity>>>,
        }

        impl Engine for CapturingEngine {
            fn apply_graph_updates(&self, _: Vec<GraphUpdate>) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn check_authorization(
                &self,
                _current_node_ids: &[String],
                _actions: &[sasy_common::policy_engine::Action],
                entity: Option<&str>,
                _roles: &[String],
                scope: &SessionScope,
                principal: Option<&str>,
                _policy_id: Option<&str>,
            ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error>
            {
                self.seen.lock().push((
                    entity.map(String::from),
                    principal.map(String::from),
                    scope.tenant().to_string(),
                ));
                Ok(sasy_common::policy_engine::AuthorizationResponse {
                    results: vec![],
                    timing: None,
                })
            }
            fn reset(&self) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn get_sync_status(&self) -> SyncStatus {
                SyncStatus {
                    current_sequence: 0,
                    node_count: 0,
                    edge_count: 0,
                    connected: true,
                }
            }
            fn set_connected(&self, _: bool) {}
            fn set_sequence(&self, _: i64) {}
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let engine = Arc::new(CapturingEngine {
            seen: Arc::clone(&seen),
        });
        let svc = PolicyService::new(engine);

        // Alice authenticates as the acme principal but tries to
        // claim a different identity in the wire `principal` field.
        let mut alice = Request::new(AuthorizationRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: Some("user-supplied-actor".into()),
            roles: vec![],
            session_id: None,
            principal: Some("evil".into()),
        });
        alice
            .extensions_mut()
            .insert(AuthResult::success("alice", vec![], "test").with_tenant("acme"));
        svc.check_authorization(alice).await.unwrap();

        // Bob authenticates anonymously (no extension) — principal
        // should fall back to None.
        let bob = Request::new(AuthorizationRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: None,
            roles: vec![],
            session_id: None,
            principal: Some("evil2".into()),
        });
        svc.check_authorization(bob).await.unwrap();

        let captured = seen.lock();
        assert_eq!(captured.len(), 2);
        // entity is left user-supplied; principal is server-stamped.
        assert_eq!(captured[0].0.as_deref(), Some("user-supplied-actor"));
        assert_eq!(
            captured[0].1.as_deref(),
            Some("alice"),
            "principal must come from auth, not wire 'evil'"
        );
        assert_eq!(captured[0].2, "acme");
        assert_eq!(captured[1].1, None, "anonymous request → no principal");
    }

    // ── EndSession + ownership ─────────────────────────────

    use sasy_auth::AuthResult;
    use sasy_graph::GraphStore;

    fn end_session_request(
        session_id: &str,
        auth: Option<AuthResult>,
    ) -> Request<EndSessionRequest> {
        let mut req = Request::new(EndSessionRequest {
            session_id: session_id.into(),
        });
        if let Some(auth) = auth {
            req.extensions_mut().insert(auth);
        }
        req
    }

    /// First principal to touch a session owns it; a second
    /// principal hits `PERMISSION_DENIED` on `EndSession`.
    #[tokio::test]
    async fn end_session_rejects_cross_principal() {
        let engine = StubEngine::new();
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(engine, Arc::clone(&graph));

        // Alice claims session "conv-1" in tenant "acme".
        graph
            .check_or_claim_session_owner(&SessionScope::new("acme", "conv-1"), Some("alice"))
            .unwrap();

        // Bob (also in acme, not admin) tries to end alice's session.
        let bob_auth = AuthResult::success("bob", vec![], "test").with_tenant("acme");
        let err = svc
            .end_session(end_session_request("conv-1", Some(bob_auth)))
            .await
            .expect_err("cross-principal end_session should be denied");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("alice"),
            "denial should mention the existing owner: {}",
            err.message()
        );
    }

    /// A non-admin, same-tenant caller cannot run CheckAuthorization against a
    /// session owned by another principal — that would leak the victim's
    /// per-session graph state + denial traces. Owner and admin are allowed.
    #[tokio::test]
    async fn check_authorization_rejects_cross_principal_session() {
        let engine = StubEngine::new();
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(engine, Arc::clone(&graph));

        // Alice claims session "conv-1" in tenant "acme".
        graph
            .check_or_claim_session_owner(&SessionScope::new("acme", "conv-1"), Some("alice"))
            .unwrap();

        let req_as = |principal: &str, roles: Vec<String>| {
            let mut req = Request::new(AuthorizationRequest {
                current_node_ids: vec![],
                actions: vec![],
                entity: None,
                roles: vec![],
                session_id: Some("conv-1".into()),
                principal: None,
            });
            req.extensions_mut()
                .insert(AuthResult::success(principal, roles, "test").with_tenant("acme"));
            req
        };

        // Bob (acme, non-admin) is denied — it's alice's session.
        let err = svc
            .check_authorization(req_as("bob", vec![]))
            .await
            .expect_err("cross-principal check_authorization should be denied");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("alice"),
            "denial should name the owner"
        );

        // The owner is allowed.
        svc.check_authorization(req_as("alice", vec![]))
            .await
            .expect("owner should be allowed");

        // Admin bypasses ownership.
        svc.check_authorization(req_as("ops", vec!["admin".into()]))
            .await
            .expect("admin should bypass ownership");
    }

    /// Admin role bypasses ownership: admin can end any session.
    #[tokio::test]
    async fn end_session_admin_bypass() {
        let engine = StubEngine::new();
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(engine, Arc::clone(&graph));

        graph
            .check_or_claim_session_owner(&SessionScope::new("acme", "conv-1"), Some("alice"))
            .unwrap();

        let admin_auth =
            AuthResult::success("ops", vec!["admin".into()], "test").with_tenant("acme");
        svc.end_session(end_session_request("conv-1", Some(admin_auth)))
            .await
            .expect("admin should bypass ownership");

        // EndSession scrubs the owner record so the next principal
        // can claim.
        assert!(graph
            .get_session_owner(&SessionScope::new("acme", "conv-1"))
            .unwrap()
            .is_none());
    }

    /// The session owner can end their own session.
    #[tokio::test]
    async fn end_session_owner_can_end() {
        let engine = StubEngine::new();
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(engine, Arc::clone(&graph));

        graph
            .check_or_claim_session_owner(&SessionScope::new("acme", "conv-1"), Some("alice"))
            .unwrap();

        let alice_auth = AuthResult::success("alice", vec![], "test").with_tenant("acme");
        svc.end_session(end_session_request("conv-1", Some(alice_auth)))
            .await
            .expect("owner can end their own session");
    }

    /// A failed SetPolicy(Session) must NOT claim ownership.
    /// Otherwise a malformed-policy upload could permanently lock a
    /// `(tenant, session_id)` against a future legitimate caller.
    /// We trigger a guaranteed build failure by passing an
    /// unsupported `backend` so the build_policy closure rejects
    /// without touching real Soufflé tooling.
    #[tokio::test]
    async fn set_policy_build_failure_leaves_session_unclaimed() {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};

        let engine = StubEngine::new();
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(engine, Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv-broken");

        // Alice tries to bind an invalid policy to "conv-broken".
        let alice_auth =
            AuthResult::success("alice", vec!["reference-monitor-user".into()], "test")
                .with_tenant("acme");
        let mut req = Request::new(SetPolicyRequest {
            policy_source: "// anything — backend below rejects before we even look".into(),
            functor_source: String::new(),
            backend: "not-a-real-backend".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Session(SessionTarget {
                    session_id: "conv-broken".into(),
                })),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut().insert(alice_auth);

        let resp = svc
            .set_policy(req)
            .await
            .expect("RPC returns Ok with accepted=false on build fail")
            .into_inner();
        assert!(!resp.accepted, "build must fail for an unsupported backend");

        // The session must not have an owner — the build failure
        // discarded the claim. Bob can still bind a valid policy
        // (or end the session) on the next call.
        assert_eq!(
            graph.get_session_owner(&scope).unwrap(),
            None,
            "failed SetPolicy must not lock ownership on the session"
        );
    }

    /// A request that never passed the auth interceptor carries no auth
    /// context. The session-bind floor refuses it instead of falling through
    /// the check: an embedder that registers the service without the
    /// interceptor must not be handing policy binding — which decides every
    /// later authorization for that session — to a caller nobody
    /// authenticated.
    #[tokio::test]
    async fn a_session_bind_with_no_auth_context_is_refused() {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let req = Request::new(SetPolicyRequest {
            policy_source: "// anyone at all".into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Session(SessionTarget {
                    session_id: "conv".into(),
                })),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        let err = svc
            .set_policy(req)
            .await
            .expect_err("an unauthenticated caller must not bind a session policy");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
        assert!(
            err.message().contains("no authentication context"),
            "the refusal should name the missing context, got: {}",
            err.message()
        );
    }

    /// The same floor guards `ValidatePolicy`, which drives the sugar.py →
    /// souffle → g++ chain on caller-supplied source.
    #[tokio::test]
    async fn validating_a_policy_with_no_auth_context_is_refused() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let err = svc
            .validate_policy(Request::new(ValidatePolicyRequest {
                policy_source: ".decl Foo(x: symbol)".into(),
                run_analyses: false,
                extra_reach_targets: vec![],
            }))
            .await
            .expect_err("an unauthenticated caller must not drive the compile chain");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    // ── Custom C++ functors ──────────────────────────────────────────
    //
    // The engine is `RefusesToPublish`, which reports every source as already
    // installed: SetPolicy takes the dedup short-circuit and never reaches
    // g++, so these tests exercise the gate and nothing behind it. A request
    // that gets past the gate therefore fails at the bind instead — which is
    // what "allowed" looks like here.

    /// A session-scoped SetPolicy carrying functor source, from a caller with
    /// `roles`.
    fn functor_request(roles: &[&str]) -> Request<SetPolicyRequest> {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};
        let mut req = Request::new(SetPolicyRequest {
            policy_source: "// already installed".into(),
            functor_source: "extern \"C\" const char* my_functor() { return \"\"; }".into(),
            backend: "souffle".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Session(SessionTarget {
                    session_id: "conv".into(),
                })),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut().insert(
            AuthResult::success(
                "agent",
                roles.iter().map(|r| r.to_string()).collect(),
                "test",
            )
            .with_tenant("acme"),
        );
        req
    }

    fn functor_service(config: PolicyServiceConfig) -> PolicyService<RefusesToPublish> {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        PolicyService::with_persistence(Arc::new(RefusesToPublish), graph).with_config(config)
    }

    fn assert_passed_the_functor_gate(err: tonic::Status) {
        assert_ne!(
            err.code(),
            tonic::Code::PermissionDenied,
            "the functor gate must not be what stopped this: {}",
            err.message()
        );
        assert!(
            !err.message().contains("functor"),
            "the functor gate must not be what stopped this: {}",
            err.message()
        );
    }

    /// The content hash `functor_request` uploads under. The bytes are the
    /// same whatever roles the caller holds, so two callers with different
    /// roles collide in the dedup short-circuit.
    fn functor_request_content_hash() -> String {
        let req = functor_request(&[]);
        crate::hash::upload_content_hash(
            "souffle",
            "", // SASY_SOUFFLE_MAGIC_SET (unset in tests)
            &req.get_ref().policy_source,
            &req.get_ref().functor_source,
        )
    }

    /// An accepted upload records its admission class on the dedup path too —
    /// the path taken when this process has already compiled that content.
    ///
    /// This is what makes "or re-upload the policy as an admin" — the
    /// remediation the load-time refusal advertises — work in every case. An
    /// admin re-upload of content this process has NOT compiled goes down the
    /// ordinary path and is recorded there; one of content it HAS compiled
    /// (a user uploaded it earlier in this process's life, as here) takes the
    /// dedup short-circuit, and if that short-circuit returned before
    /// recording the class, the next boot with the opt-in off would find only
    /// the user record and refuse the policy again.
    #[tokio::test]
    async fn an_admin_upload_of_deduped_content_still_records_the_admin_class() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(AcceptsTheBind), Arc::clone(&graph))
            .with_config(PolicyServiceConfig {
                user_functors: UserFunctors::Unsandboxed,
                sandbox_available: Some(false),
            });
        let hash = functor_request_content_hash();

        // With the opt-in on, a non-admin uploads the source. It is admitted,
        // as user-supplied.
        svc.set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect("the opt-in admits a non-admin's functor source");
        assert_eq!(
            graph
                .get_policy_source_in_class(&hash, sasy_graph::FunctorAdmission::User)
                .unwrap()
                .map(|p| p.functor_admission),
            Some(sasy_graph::FunctorAdmission::User),
            "the non-admin upload is recorded as user-supplied"
        );

        // An admin re-uploads the identical source. This process installed it
        // a moment ago, so the upload dedups and nothing is compiled again —
        // and the admin class is still recorded.
        svc.set_policy(functor_request(&["admin"]))
            .await
            .expect("an admin may always upload functor source");

        // With the opt-in switched back off, the user record is refused and
        // the admin record is what loads.
        let (class, _record) = crate::replay::admitted_policy_source(
            &graph,
            &hash,
            &PolicyServiceConfig {
                user_functors: UserFunctors::Refuse,
                sandbox_available: Some(false),
            },
        )
        .expect("the admin re-upload left a record the opt-in-off gate admits")
        .expect("a record is stored under this hash");
        assert_eq!(
            class,
            sasy_graph::FunctorAdmission::Admin,
            "the admin's re-upload must be recorded even though it deduped"
        );
    }

    /// A policy with no functor source is stored once, not once per uploader.
    ///
    /// The admission class is part of the storage key, and it exists to keep
    /// an admin's functor bytes separate from a user's. A policy carrying no
    /// functor source has nothing for the class to decide — the load paths do
    /// not gate one — so an admin and a non-admin uploading the same bytes
    /// must share one record instead of storing two identical copies.
    #[tokio::test]
    async fn a_functor_less_policy_is_stored_under_one_key_for_every_uploader() {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};

        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(AcceptsTheBind), Arc::clone(&graph));

        let upload = |roles: &[&str]| {
            let mut req = Request::new(SetPolicyRequest {
                policy_source: "// no functors here".into(),
                functor_source: String::new(),
                backend: "souffle".into(),
                scope: Some(PolicyScope {
                    target: Some(PolicyScopeTarget::Session(SessionTarget {
                        session_id: "conv".into(),
                    })),
                }),
                policy_metadata: vec![],
                bind_content_hash: String::new(),
                bind_profile_name: String::new(),
            });
            req.extensions_mut().insert(
                AuthResult::success(
                    "agent",
                    roles.iter().map(|r| r.to_string()).collect(),
                    "test",
                )
                .with_tenant("acme"),
            );
            req
        };
        let hash = crate::hash::upload_content_hash(
            "souffle",
            "", // SASY_SOUFFLE_MAGIC_SET (unset in tests)
            "// no functors here",
            "",
        );

        svc.set_policy(upload(&["reference-monitor-user"]))
            .await
            .expect("a functor-less policy needs no functor privilege");
        svc.set_policy(upload(&["admin"]))
            .await
            .expect("an admin may upload the same policy");

        assert!(
            graph
                .get_policy_source_in_class(&hash, sasy_graph::FunctorAdmission::User)
                .unwrap()
                .is_some(),
            "the policy must be stored"
        );
        assert!(
            graph
                .get_policy_source_in_class(&hash, sasy_graph::FunctorAdmission::Admin)
                .unwrap()
                .is_none(),
            "the admin upload stored a second copy of identical bytes under a second key"
        );
    }

    /// SetPolicy refuses rather than overwriting a stored source.
    ///
    /// The store key is a content hash and is not tenant-scoped, so before
    /// this rule any caller whose upload hashed to an occupied key replaced
    /// the bytes there — the bytes live sessions are bound to and boot replay
    /// recompiles from. A key already holding DIFFERENT content is a hash
    /// ambiguity; the upload fails and the stored record is left alone.
    #[tokio::test]
    async fn an_upload_that_would_overwrite_a_stored_source_is_refused() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(AcceptsTheBind), Arc::clone(&graph))
            .with_config(PolicyServiceConfig {
                user_functors: UserFunctors::Unsandboxed,
                sandbox_available: Some(false),
            });
        let hash = functor_request_content_hash();

        // Someone else's record is already under the key this upload derives.
        let planted = sasy_graph::PersistedPolicy {
            policy_source: "IsAuthorized(idx) :- Actions(idx, _).".into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            functor_admission: sasy_graph::FunctorAdmission::User,
        };
        graph.put_policy_source(&hash, &planted).unwrap();

        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("the upload must not overwrite the stored source");
        assert!(
            err.message().contains("different content"),
            "the refusal must say the key already holds other bytes, got: {}",
            err.message()
        );

        let after = graph
            .get_policy_source_in_class(&hash, sasy_graph::FunctorAdmission::User)
            .unwrap()
            .expect("the planted record is still there");
        assert_eq!(
            after.policy_source, "IsAuthorized(idx) :- Actions(idx, _).",
            "the refused upload must leave the stored bytes alone"
        );
        assert_eq!(after.functor_source, "");
    }

    /// Functor source is native code compiled into the evaluator's process, so
    /// by default only an admin may supply it — `reference-monitor-user`, the
    /// floor for binding a session policy, is not enough.
    #[tokio::test]
    async fn functor_source_from_a_non_admin_is_refused_by_default() {
        let svc = functor_service(PolicyServiceConfig::default());
        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("a non-admin must not be able to load native code");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("admin") && err.message().contains("--allow-user-functors"),
            "the refusal must name the role it wants and the operator's opt-in, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("Python SDK"),
            "the refusal must explain how an ordinary session bind came to carry functor \
             source at all, got: {}",
            err.message()
        );
    }

    /// An admin may supply functor source with no flag set, on any host: the
    /// role already carries the authority to change what the engine runs.
    #[tokio::test]
    async fn functor_source_from_an_admin_is_allowed_by_default() {
        let svc = functor_service(PolicyServiceConfig::default());
        let err = svc
            .set_policy(functor_request(&["admin"]))
            .await
            .expect_err("this engine refuses every bind");
        assert_passed_the_functor_gate(err);
    }

    /// With the operator's opt-in AND a real sandbox, a non-admin may supply
    /// functors — the C++ then runs confined rather than as the service user.
    #[tokio::test]
    async fn functor_source_from_a_non_admin_is_allowed_with_the_flag_and_a_sandbox() {
        let svc = functor_service(PolicyServiceConfig {
            user_functors: UserFunctors::Sandboxed,
            sandbox_available: Some(true),
        });
        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("this engine refuses every bind");
        assert_passed_the_functor_gate(err);
    }

    /// The opt-in is not enough on a host with no sandbox. Without bubblewrap
    /// the supplied C++ compiles and runs unconfined as the user running the
    /// binary, so the flag would be granting more than the operator asked for.
    #[tokio::test]
    async fn the_flag_does_not_admit_a_non_admin_where_there_is_no_sandbox() {
        let svc = functor_service(PolicyServiceConfig {
            user_functors: UserFunctors::Sandboxed,
            sandbox_available: Some(false),
        });
        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("no sandbox means no non-admin functors");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("sandbox"),
            "the refusal must say which condition applied, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("--allow-user-functors=unsandboxed"),
            "the refusal must name the escape hatch for a development host, got: {}",
            err.message()
        );
    }

    /// The unsandboxed opt-in admits the same upload the sandboxed one just
    /// refused: the operator said, in as many words, that this host runs
    /// caller-supplied C++ unconfined.
    #[tokio::test]
    async fn the_unsandboxed_opt_in_admits_a_non_admin_where_there_is_no_sandbox() {
        let svc = functor_service(PolicyServiceConfig {
            user_functors: UserFunctors::Unsandboxed,
            sandbox_available: Some(false),
        });
        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("this engine refuses every bind");
        assert_passed_the_functor_gate(err);
    }

    /// `None` is not a third answer: it means the gate asks the host at
    /// request time, which is what the binary passes so a `bwrap` that
    /// vanishes under a long-lived process cannot leave the gate admitting
    /// uploads as confined. Whichever host this runs on, the gate's verdict
    /// matches what the sandbox probe says right now.
    #[tokio::test]
    async fn an_unconfigured_sandbox_answer_comes_from_the_host() {
        let svc = functor_service(PolicyServiceConfig {
            user_functors: UserFunctors::Sandboxed,
            sandbox_available: None,
        });
        let err = svc
            .set_policy(functor_request(&["reference-monitor-user"]))
            .await
            .expect_err("this engine refuses every bind, sandbox or not");
        if crate::sandbox::sandbox_available() {
            assert_passed_the_functor_gate(err);
        } else {
            assert_eq!(err.code(), tonic::Code::PermissionDenied);
            assert!(
                err.message().contains("sandbox"),
                "the refusal must say which condition applied, got: {}",
                err.message()
            );
        }
    }

    /// A relay that holds `admin` for its own administrative calls does not
    /// pass that admin to the end users it fronts. A relay can hold
    /// `service-proxy` + `admin` together while forwarding each end user
    /// via `x-tenant` / `x-principal`. Reading the connection's roles here
    /// would let any of its users load native code into the engine.
    #[tokio::test]
    async fn a_relay_with_admin_cannot_load_functors_for_the_user_it_speaks_for() {
        let svc = functor_service(PolicyServiceConfig::default());
        let mut req = functor_request(&["service-proxy", "admin"]);
        req.metadata_mut()
            .insert("x-principal", "alice".parse().unwrap());
        req.metadata_mut()
            .insert("x-tenant", "acme".parse().unwrap());
        let err = svc
            .set_policy(req)
            .await
            .expect_err("a delegated end user is not the relay's admin");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("admin"),
            "the refusal must name the role it wants, got: {}",
            err.message()
        );
    }

    /// `x-roles` is written by the caller, so a `service-proxy` principal that
    /// is NOT an admin cannot become one by claiming `admin` for the user it
    /// says it speaks for. The shipped `reference-monitor` entity holds
    /// exactly this credential; if the header alone decided the gate,
    /// `service-proxy` would be a second way to run native code in the engine.
    #[tokio::test]
    async fn a_relay_without_admin_cannot_claim_it_in_the_forwarded_roles() {
        let svc = functor_service(PolicyServiceConfig::default());
        let mut req = functor_request(&["service-proxy", "reference-monitor-user"]);
        req.metadata_mut()
            .insert("x-principal", "alice".parse().unwrap());
        req.metadata_mut()
            .insert("x-roles", "admin".parse().unwrap());
        let err = svc
            .set_policy(req)
            .await
            .expect_err("a forwarded role claim is not an authenticated admin");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("admin"),
            "the refusal must name the role it wants, got: {}",
            err.message()
        );
    }

    /// The same relay acting for ITSELF — no delegation headers — is still an
    /// admin, so operator-driven functor uploads through it keep working.
    #[tokio::test]
    async fn a_relay_acting_for_itself_keeps_its_own_admin() {
        let svc = functor_service(PolicyServiceConfig::default());
        let err = svc
            .set_policy(functor_request(&["service-proxy", "admin"]))
            .await
            .expect_err("this engine refuses every bind");
        assert_passed_the_functor_gate(err);
    }

    /// The gate is satisfied when both halves are admin: the relay is
    /// authenticated as admin AND the user it forwards claims admin. That is
    /// how an operator administering the engine through a relay uploads
    /// functors.
    #[tokio::test]
    async fn an_admin_relay_forwarding_an_admin_user_may_load_functors() {
        let svc = functor_service(PolicyServiceConfig::default());
        let mut req = functor_request(&["service-proxy", "admin"]);
        req.metadata_mut()
            .insert("x-principal", "root".parse().unwrap());
        req.metadata_mut()
            .insert("x-roles", "admin".parse().unwrap());
        let err = svc
            .set_policy(req)
            .await
            .expect_err("this engine refuses every bind");
        assert_passed_the_functor_gate(err);
    }

    // ── Tenant-wide scopes (Default / Force) ─────────────────────────
    //
    // Same two-identity rule as the functor gate, and for the same reason:
    // the tenant these scopes write to is the EFFECTIVE tenant, so the
    // relay's own roles answer the wrong question.

    /// Give `req` a `service-proxy` + `admin` connection that forwards an end
    /// user in tenant `acme`, with `roles` as the forwarded `x-roles` (none
    /// forwarded when empty).
    fn delegated_by_an_admin_relay<T>(req: &mut Request<T>, roles: &str) {
        req.extensions_mut().insert(
            AuthResult::success(
                "cloud-api",
                vec!["service-proxy".into(), "admin".into()],
                "test",
            )
            .with_tenant("gateway"),
        );
        req.metadata_mut()
            .insert("x-tenant", "acme".parse().unwrap());
        req.metadata_mut()
            .insert("x-principal", "alice".parse().unwrap());
        if !roles.is_empty() {
            req.metadata_mut().insert("x-roles", roles.parse().unwrap());
        }
    }

    /// A relay can hold `service-proxy` and `admin` together.
    /// Reading the connection's roles alone would let any end user it fronts
    /// replace their own tenant's default policy — an allow-all default turns
    /// enforcement off for that tenant.
    #[tokio::test]
    async fn a_relay_with_admin_cannot_set_a_tenant_default_for_the_user_it_speaks_for() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let mut req = default_request("// already installed");
        delegated_by_an_admin_relay(&mut req, "");
        let err = svc
            .set_policy(req)
            .await
            .expect_err("a delegated end user is not the relay's admin");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("admin"),
            "the refusal must name the role it wants, got: {}",
            err.message()
        );
    }

    /// `Force` additionally evicts every live session in the tenant, so it
    /// gets the same treatment.
    #[tokio::test]
    async fn a_relay_with_admin_cannot_force_a_rollout_for_the_user_it_speaks_for() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let mut req = force_request("// already installed");
        delegated_by_an_admin_relay(&mut req, "");
        let err = svc
            .set_policy(req)
            .await
            .expect_err("a delegated end user is not the relay's admin");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
    }

    /// An admin relay forwarding an admin user still administers the tenant:
    /// that is how an operator drives a rollout through a relay. The bind
    /// fails afterwards because this engine publishes nothing, which is what
    /// "past the role gate" looks like here.
    #[tokio::test]
    async fn an_admin_relay_forwarding_an_admin_user_may_set_the_tenant_default() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let mut req = default_request("// already installed");
        delegated_by_an_admin_relay(&mut req, "admin");
        let err = svc
            .set_policy(req)
            .await
            .expect_err("this engine refuses every bind");
        assert_ne!(
            err.code(),
            tonic::Code::PermissionDenied,
            "the role gate must not be what stopped this: {}",
            err.message()
        );
    }

    /// A direct admin — no delegation headers at all — is unaffected: without
    /// delegation the effective roles are the connection's own.
    #[tokio::test]
    async fn a_direct_admin_may_still_set_the_tenant_default() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), graph);
        let err = svc
            .set_policy(default_request("// already installed"))
            .await
            .expect_err("this engine refuses every bind");
        assert_ne!(
            err.code(),
            tonic::Code::PermissionDenied,
            "the role gate must not be what stopped this: {}",
            err.message()
        );
    }

    // ── Session ownership under delegation ───────────────────────────
    //
    // The waiver that lets an admin reach somebody else's session is the same
    // two-identity question, for the same reason: the session is looked up in
    // the EFFECTIVE tenant under the EFFECTIVE principal, so the relay's own
    // roles answer the wrong question. Bob owns `conv` in `acme`; the relay
    // forwards alice, who is not an admin and does not own it.

    /// Bob's session in the tenant the relay forwards.
    fn a_session_owned_by_bob(graph: &GraphStore) {
        graph
            .check_or_claim_session_owner(&SessionScope::new("acme", "conv"), Some("bob"))
            .unwrap();
    }

    /// A SetPolicy targeting session `conv`, relayed for alice with `roles`
    /// forwarded (none when empty).
    fn delegated_session_bind(roles: &str) -> Request<SetPolicyRequest> {
        let mut req = locked_test_request("// already installed");
        delegated_by_an_admin_relay(&mut req, roles);
        req
    }

    #[tokio::test]
    async fn a_relay_with_admin_cannot_bind_a_policy_to_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));
        let err = svc
            .set_policy(delegated_session_bind(""))
            .await
            .expect_err("alice does not own bob's session and is not an admin");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("bob"),
            "the refusal must name the owner, got: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn a_relay_with_admin_cannot_end_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let mut req = end_session_request("conv", None);
        delegated_by_an_admin_relay(&mut req, "");
        let err = svc
            .end_session(req)
            .await
            .expect_err("alice may not tear down bob's session");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert_eq!(
            graph
                .get_session_owner(&SessionScope::new("acme", "conv"))
                .unwrap()
                .as_deref(),
            Some("bob"),
            "a refused teardown must leave the owner in place"
        );
    }

    #[tokio::test]
    async fn a_relay_with_admin_cannot_annotate_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let mut req = Request::new(UpdatePolicyMetadataRequest {
            session_id: "conv".into(),
            facts: vec![sasy_common::policy_engine::PolicyMetadataFact {
                rel: "detaint_approved".into(),
                a: "node-1".into(),
                b: String::new(),
            }],
        });
        delegated_by_an_admin_relay(&mut req, "");
        let err = svc
            .update_policy_metadata(req)
            .await
            .expect_err("alice may not record a detaint approval on bob's session");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
    }

    #[tokio::test]
    async fn a_relay_with_admin_cannot_read_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let mut req = Request::new(AuthorizationRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: None,
            roles: vec![],
            session_id: Some("conv".into()),
            principal: None,
        });
        delegated_by_an_admin_relay(&mut req, "");
        let err = svc
            .check_authorization(req)
            .await
            .expect_err("alice may not evaluate against bob's session state");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
    }

    /// An admin relay forwarding an admin user still administers other
    /// people's sessions — that is how an operator drives support traffic
    /// through a relay.
    #[tokio::test]
    async fn an_admin_relay_forwarding_an_admin_user_may_end_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let mut req = end_session_request("conv", None);
        delegated_by_an_admin_relay(&mut req, "admin");
        svc.end_session(req)
            .await
            .expect("an admin user behind an admin relay may end the session");
    }

    /// A direct admin — no delegation headers at all — is unaffected: without
    /// delegation the effective roles are the connection's own.
    #[tokio::test]
    async fn a_direct_admin_may_still_end_another_principals_session() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        a_session_owned_by_bob(&graph);
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let admin = AuthResult::success("root", vec!["admin".into()], "test").with_tenant("acme");
        svc.end_session(end_session_request("conv", Some(admin)))
            .await
            .expect("a direct admin may end the session");
    }

    /// An engine whose every dispatch fails with `message`. Stands in for the
    /// lazy install refusing to compile a persisted policy's functor source.
    struct DispatchFailsWith(&'static str);

    impl Engine for DispatchFailsWith {
        fn apply_graph_updates(
            &self,
            _: Vec<crate::engine::GraphUpdate>,
        ) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn check_authorization(
            &self,
            _: &[String],
            _: &[sasy_common::policy_engine::Action],
            _: Option<&str>,
            _: &[String],
            _: &SessionScope,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
            Err(anyhow::anyhow!("{}", self.0))
        }
        fn reset(&self) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn get_sync_status(&self) -> crate::engine::SyncStatus {
            crate::engine::SyncStatus {
                current_sequence: 0,
                node_count: 0,
                edge_count: 0,
                connected: true,
            }
        }
        fn set_connected(&self, _: bool) {}
        fn set_sequence(&self, _: i64) {}
    }

    async fn authorize_against(engine: DispatchFailsWith) -> Status {
        let svc = PolicyService::new(Arc::new(engine));
        svc.check_authorization(Request::new(AuthorizationRequest {
            current_node_ids: vec![],
            actions: vec![],
            entity: None,
            roles: vec![],
            session_id: Some("conv".into()),
            principal: None,
        }))
        .await
        .expect_err("this engine fails every dispatch")
    }

    /// A dispatch can be the moment a persisted policy is first compiled in
    /// this process. When the functor gate refuses that compile, the session
    /// has no policy installed — a precondition the caller can do something
    /// about, so it comes back as `FAILED_PRECONDITION` carrying the reason,
    /// not as an opaque internal error.
    #[tokio::test]
    async fn a_dispatch_refused_by_the_functor_gate_is_a_failed_precondition() {
        let message: &'static str = Box::leak(
            format!("{FUNCTOR_REFUSED_AT_LOAD}: start with --allow-user-functors").into_boxed_str(),
        );
        let err = authorize_against(DispatchFailsWith(message)).await;
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(
            err.message().contains("--allow-user-functors"),
            "the reason must reach the caller, got: {}",
            err.message()
        );
    }

    /// A failure that merely quotes the marker is not one.
    ///
    /// A compile error carries the policy source that caused it, and that
    /// source is written by whoever uploaded the policy. If a match anywhere
    /// in the message were enough, a caller could choose the status code an
    /// unrelated failure comes back as by putting the marker text in a
    /// comment. The refusal always leads the message; a quotation does not.
    #[tokio::test]
    async fn a_failure_that_only_quotes_the_marker_is_still_internal() {
        let message: &'static str = Box::leak(
            format!("g++ error in policy source: // {FUNCTOR_REFUSED_AT_LOAD}").into_boxed_str(),
        );
        let err = authorize_against(DispatchFailsWith(message)).await;
        assert_eq!(err.code(), tonic::Code::Internal);
    }

    /// Every other dispatch failure keeps the status it had. The mapping above
    /// is for the functor gate's refusal only.
    #[tokio::test]
    async fn any_other_dispatch_failure_is_still_internal() {
        let err = authorize_against(DispatchFailsWith("the evaluator subprocess died")).await;
        assert_eq!(err.code(), tonic::Code::Internal);
    }

    /// A policy with no functor source is untouched by any of this — the gate
    /// reads `functor_source`, not the caller's role.
    #[tokio::test]
    async fn a_policy_without_functor_source_is_unaffected() {
        for config in [
            PolicyServiceConfig::default(),
            PolicyServiceConfig {
                user_functors: UserFunctors::Sandboxed,
                sandbox_available: Some(false),
            },
            PolicyServiceConfig {
                user_functors: UserFunctors::Unsandboxed,
                sandbox_available: Some(false),
            },
            PolicyServiceConfig {
                user_functors: UserFunctors::Refuse,
                sandbox_available: Some(true),
            },
        ] {
            let svc = functor_service(config.clone());
            let err = svc
                .set_policy(locked_test_request("// already installed"))
                .await
                .expect_err("this engine refuses every bind");
            assert_passed_the_functor_gate(err);
        }
    }

    // Build a SetPolicy(session) request for `source` with a reference-monitor
    // role attached (the floor for binding a session policy).
    fn locked_test_request(source: &str) -> Request<SetPolicyRequest> {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};
        let mut req = Request::new(SetPolicyRequest {
            policy_source: source.into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Session(SessionTarget {
                    session_id: "conv".into(),
                })),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut().insert(
            AuthResult::success("agent", vec!["reference-monitor-user".into()], "test")
                .with_tenant("acme"),
        );
        req
    }

    // Build a SetPolicy(session) BIND-BY-HASH request (no source) with the
    // reference-monitor role attached — the source-free bind path.
    fn bind_test_request(content_hash: &str) -> Request<SetPolicyRequest> {
        use sasy_common::policy_engine::{PolicyScope, SessionTarget};
        let mut req = Request::new(SetPolicyRequest {
            policy_source: String::new(),
            functor_source: String::new(),
            backend: String::new(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Session(SessionTarget {
                    session_id: "conv".into(),
                })),
            }),
            policy_metadata: vec![],
            bind_content_hash: content_hash.into(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut().insert(
            AuthResult::success("agent", vec!["reference-monitor-user".into()], "test")
                .with_tenant("acme"),
        );
        req
    }

    // The set_policy content hash for a souffle upload with an empty
    // magic-set and no functor source, so a test can pin the locked hash.
    //
    // This calls the same function production calls, so it cannot drift from
    // it; `hash::tests::upload_content_hash_pins_the_v2_formula` is what
    // fails if the formula itself changes.
    fn content_hash_souffle(source: &str) -> String {
        crate::hash::upload_content_hash(
            "souffle", "", // SASY_SOUFFLE_MAGIC_SET (unset in tests)
            source, "", // functor_source
        )
    }

    #[tokio::test]
    async fn policy_lock_rejects_a_non_default_source() {
        // A restricted build locked to one policy must refuse a DIFFERENT
        // source before the dedup short-circuit — otherwise a baked allow-all
        // would dedup-bind and disable enforcement for the caller's session.
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), graph)
            .with_restricted(true)
            .with_policy_lock(Some(std::collections::HashSet::from(["0".repeat(64)]))); // a hash no real source produces
        let err = svc
            .set_policy(locked_test_request("IsAuthorized(idx) :- Actions(idx, _)."))
            .await
            .expect_err("locked build must reject a non-default source");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(
            err.message().contains("locked"),
            "expected the policy-lock message, got: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn policy_lock_lets_the_locked_source_through() {
        // The matching (locked) source must pass the lock so a client can
        // still (re-)pin its profile and refresh metadata. With restricted=true
        // and nothing pre-installed it then hits the curated-only reject — the
        // point is only that the LOCK isn't what stopped it.
        let source = "IsAuthorized(idx) :- Actions(idx, _).";
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), graph)
            .with_restricted(true)
            .with_policy_lock(Some(std::collections::HashSet::from([
                content_hash_souffle(source),
            ])));
        if let Err(e) = svc.set_policy(locked_test_request(source)).await {
            assert!(
                !e.message().contains("locked"),
                "the locked source must pass the lock, got: {}",
                e.message()
            );
        }
    }

    #[tokio::test]
    async fn bind_by_hash_rejected_by_lock_for_non_default() {
        // A restricted build must refuse a bind_content_hash that isn't the
        // locked profile — otherwise a co-located caller could bind a DIFFERENT
        // baked profile (e.g. deny-all) and change what its session enforces.
        // The lock compares exactly the supplied hash (service.rs:712-719).
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), graph)
            .with_restricted(true)
            .with_policy_lock(Some(std::collections::HashSet::from(["a".repeat(64)])));
        let err = svc
            .set_policy(bind_test_request(&"b".repeat(64)))
            .await
            .expect_err("a non-locked bind hash must be rejected by the lock");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert!(err.message().contains("locked"), "got: {}", err.message());
    }

    #[tokio::test]
    async fn bind_by_hash_locked_passes_lock_then_misses() {
        // The locked hash passes the lock (so a client CAN bind the curated
        // profile); with nothing pre-installed in the stub the bind then misses
        // and returns FailedPrecondition — NOT the lock message, NOT a compile.
        let locked = "a".repeat(64);
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), graph)
            .with_restricted(true)
            .with_policy_lock(Some(std::collections::HashSet::from([locked.clone()])));
        let err = svc
            .set_policy(bind_test_request(&locked))
            .await
            .expect_err("nothing pre-installed in the stub, so the bind misses");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(
            !err.message().contains("locked"),
            "the locked hash must pass the lock, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("no installed policy"),
            "got: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn bind_by_hash_miss_is_failed_precondition_not_compile() {
        // Full build (no lock): a bind for an un-installed hash must return
        // FailedPrecondition BEFORE the compile path — never try to compile the
        // (empty) source. Guards the requiredFix from the security verdict.
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), graph);
        let err = svc
            .set_policy(bind_test_request(&"c".repeat(64)))
            .await
            .expect_err("an un-installed bind hash must be rejected, not compiled");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(
            err.message().contains("no installed policy"),
            "got: {}",
            err.message()
        );
    }
    // ── Durability before publication ──────────────────────

    /// An engine that reports every source as already installed — so
    /// SetPolicy takes the dedup short-circuit and never reaches the
    /// compiler — and refuses to publish anything.
    struct RefusesToPublish;

    impl Engine for RefusesToPublish {
        fn apply_graph_updates(
            &self,
            _: Vec<crate::engine::GraphUpdate>,
        ) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn check_authorization(
            &self,
            _: &[String],
            _: &[sasy_common::policy_engine::Action],
            _: Option<&str>,
            _: &[String],
            _: &SessionScope,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
            Ok(sasy_common::policy_engine::AuthorizationResponse {
                results: vec![],
                timing: None,
            })
        }
        fn reset(&self) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn get_sync_status(&self) -> crate::engine::SyncStatus {
            crate::engine::SyncStatus {
                current_sequence: 0,
                node_count: 0,
                edge_count: 0,
                connected: true,
            }
        }
        fn set_connected(&self, _: bool) {}
        fn set_sequence(&self, _: i64) {}
        fn lookup_policy_by_content(&self, _: &str, _: &str) -> Option<String> {
            Some("pid-existing".into())
        }
        fn promote_to_default(&self, _: &str, _: &str, _: bool) -> Result<(), anyhow::Error> {
            Err(anyhow::anyhow!("registry refused the promotion"))
        }
        fn set_session_policy(&self, _: &SessionScope, _: &str) -> Result<bool, anyhow::Error> {
            Err(anyhow::anyhow!("registry refused the bind"))
        }
    }

    /// An engine that accepts every bind and reports every source as already
    /// installed, so a SetPolicy carrying source takes the dedup
    /// short-circuit and returns success. Used by the tests that pin what a
    /// SUCCESSFUL bind writes.
    struct AcceptsTheBind;
    impl Engine for AcceptsTheBind {
        fn apply_graph_updates(
            &self,
            _: Vec<crate::engine::GraphUpdate>,
        ) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn check_authorization(
            &self,
            _: &[String],
            _: &[sasy_common::policy_engine::Action],
            _: Option<&str>,
            _: &[String],
            _: &SessionScope,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
            Ok(sasy_common::policy_engine::AuthorizationResponse {
                results: vec![],
                timing: None,
            })
        }
        fn reset(&self) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn get_sync_status(&self) -> crate::engine::SyncStatus {
            crate::engine::SyncStatus {
                current_sequence: 0,
                node_count: 0,
                edge_count: 0,
                connected: true,
            }
        }
        fn set_connected(&self, _: bool) {}
        fn set_sequence(&self, _: i64) {}
        fn lookup_policy_by_content(&self, _: &str, _: &str) -> Option<String> {
            Some("pid-existing".into())
        }
        fn set_session_policy(&self, _: &SessionScope, _: &str) -> Result<bool, anyhow::Error> {
            Ok(true)
        }
    }

    fn force_request(source: &str) -> Request<SetPolicyRequest> {
        use sasy_common::policy_engine::{ForceTarget, PolicyScope};
        let mut req = Request::new(SetPolicyRequest {
            policy_source: source.into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Force(ForceTarget {})),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut()
            .insert(AuthResult::success("admin", vec!["admin".into()], "test").with_tenant("acme"));
        req
    }

    fn default_request(source: &str) -> Request<SetPolicyRequest> {
        use sasy_common::policy_engine::{DefaultTarget, PolicyScope};
        let mut req = Request::new(SetPolicyRequest {
            policy_source: source.into(),
            functor_source: String::new(),
            backend: "souffle".into(),
            scope: Some(PolicyScope {
                target: Some(PolicyScopeTarget::Default(DefaultTarget {})),
            }),
            policy_metadata: vec![],
            bind_content_hash: String::new(),
            bind_profile_name: String::new(),
        });
        req.extensions_mut()
            .insert(AuthResult::success("admin", vec!["admin".into()], "test").with_tenant("acme"));
        req
    }

    /// A `Force` rollout that cannot publish must leave the store as it found
    /// it. The default pointer is overwritten and the per-session pins are
    /// cleared *before* the promotion, so that a write which fails cannot
    /// return an error with the rollout already live. That ordering is only
    /// safe if a failed promotion puts both back — otherwise a restart
    /// applies the rollout the caller was told had failed.
    #[tokio::test]
    async fn failed_force_rollout_restores_the_store() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        graph.put_tenant_default_policy("acme", "old-hash").unwrap();
        let pinned = SessionScope::new("acme", "conv-pinned");
        graph.put_policy_binding(&pinned, "pinned-hash").unwrap();

        let err = svc
            .set_policy(force_request("// already installed"))
            .await
            .expect_err("a promotion the engine refuses must fail the RPC");
        assert!(
            !err.message().contains("could not be restored"),
            "the rollback itself must succeed here, got: {}",
            err.message()
        );

        assert_eq!(
            graph.get_tenant_default_policy("acme").unwrap().as_deref(),
            Some("old-hash"),
            "a rollout that never went live must not change the stored default"
        );
        assert_eq!(
            graph.get_policy_binding(&pinned).unwrap().as_deref(),
            Some("pinned-hash"),
            "pins cleared for a rollout that never went live must be put back"
        );
    }

    /// Same invariant for a session pin: it is recorded before the bind is
    /// published, so a bind the registry refuses has to take the record back
    /// out rather than leave a pin for replay to apply after a restart.
    #[tokio::test]
    async fn failed_session_bind_restores_the_pin() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv");
        graph.put_policy_binding(&scope, "prior-hash").unwrap();

        svc.set_policy(locked_test_request("// already installed"))
            .await
            .expect_err("a bind the engine refuses must fail the RPC");

        assert_eq!(
            graph.get_policy_binding(&scope).unwrap().as_deref(),
            Some("prior-hash"),
            "a bind that never went live must leave the previous pin in place"
        );
    }

    /// Ending a session must remove the config that travels with it, not only
    /// its owner. `assemble_metadata` prefers a session-scoped row over the
    /// tenant one, so a binding-metadata row left behind on a released id is
    /// seeded into the next principal's evaluator — and that config can carry
    /// a `rule_off`, i.e. a metadata-gated denial switched off.
    #[tokio::test]
    async fn end_session_removes_the_config_with_the_owner() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv-1");
        graph
            .check_or_claim_session_owner(&scope, Some("alice"))
            .unwrap();
        graph.put_policy_binding(&scope, "some-hash").unwrap();
        graph
            .put_binding_metadata(
                &scope,
                "some-hash",
                &[sasy_graph::PolicyMetadataFact {
                    rel: "rule_off".into(),
                    a: "cancellation_requires_reason".into(),
                    b: String::new(),
                }],
            )
            .unwrap();

        let auth = AuthResult::success("alice", vec![], "test").with_tenant("acme");
        svc.end_session(end_session_request("conv-1", Some(auth)))
            .await
            .expect("alice owns the session");

        assert_eq!(graph.get_session_owner(&scope).unwrap(), None);
        assert_eq!(graph.get_policy_binding(&scope).unwrap(), None);
        assert!(
            graph
                .get_binding_metadata(&scope, "some-hash")
                .unwrap()
                .is_none(),
            "the released id must not carry the previous principal's config — and \
             absent, not present-and-empty: an empty row still suppresses the \
             fallback to tenant config for whoever next runs on this id"
        );
    }
    /// The tenant-wide config a rollout writes must come back out when the
    /// rollout does not publish. Left behind, it is seeded into every
    /// evaluator that later respawns on that policy — so a refused
    /// configuration takes effect session by session, silently.
    #[tokio::test]
    async fn failed_rollout_restores_the_tenant_config() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        graph
            .put_policy_metadata(
                "acme",
                &content_hash_souffle("// already installed"),
                &[sasy_graph::PolicyMetadataFact {
                    rel: "rule_on".into(),
                    a: "exfil".into(),
                    b: String::new(),
                }],
            )
            .unwrap();

        let mut req = force_request("// already installed");
        req.get_mut().policy_metadata = vec![sasy_common::policy_engine::PolicyMetadataFact {
            rel: "rule_off".into(),
            a: "exfil".into(),
            b: String::new(),
        }];
        svc.set_policy(req)
            .await
            .expect_err("a promotion the engine refuses must fail the RPC");

        let kept = graph
            .get_policy_metadata("acme", &content_hash_souffle("// already installed"))
            .unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].rel, "rule_on", "the refused config must not stand");
    }

    /// Same for the per-session config a Force rollout clears tenant-wide.
    /// Nothing can reconstruct it — it comes from the client on each bind —
    /// so a rollout that restores the pins and not the config leaves every
    /// pinned session running on configuration it never asked for.
    #[tokio::test]
    async fn failed_force_rollout_restores_the_per_session_config() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        let pinned = SessionScope::new("acme", "conv-pinned");
        graph
            .put_binding_metadata(
                &pinned,
                "pinned-hash",
                &[sasy_graph::PolicyMetadataFact {
                    rel: "rule_off".into(),
                    a: "review_gate".into(),
                    b: String::new(),
                }],
            )
            .unwrap();

        svc.set_policy(force_request("// already installed"))
            .await
            .expect_err("a promotion the engine refuses must fail the RPC");

        let kept = graph
            .get_binding_metadata(&pinned, "pinned-hash")
            .unwrap()
            .expect("the cleared per-session config must be restored");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].a, "review_gate");
    }

    /// A bind that does not publish must not leave its config row behind.
    /// An empty row is not nothing: `assemble_metadata` treats the presence
    /// of a binding row as "this session pinned itself, its config is exactly
    /// this", so a leftover row from a failed bind permanently shadows the
    /// tenant-wide config for that session.
    #[tokio::test]
    async fn failed_session_bind_leaves_no_config_row() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv");
        let hash = content_hash_souffle("// already installed");

        svc.set_policy(locked_test_request("// already installed"))
            .await
            .expect_err("a bind the engine refuses must fail the RPC");

        assert!(
            graph.get_binding_metadata(&scope, &hash).unwrap().is_none(),
            "a bind that never published must not leave a config row to shadow the tenant's"
        );
    }

    /// EndSession must not release the session id while its evaluator is
    /// still live. In that window the scope is ownerless, so any other
    /// principal in the tenant passes the ownership check and is decided by
    /// the departed session's evaluator — under the seed it was built with.
    #[tokio::test]
    async fn end_session_releases_the_id_only_after_the_evaluator_is_gone() {
        use parking_lot::Mutex as PlMutex;

        struct OwnerAtEviction {
            graph: Arc<GraphStore>,
            owned_when_evicted: Arc<PlMutex<Option<bool>>>,
        }
        impl Engine for OwnerAtEviction {
            fn apply_graph_updates(
                &self,
                _: Vec<crate::engine::GraphUpdate>,
            ) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn check_authorization(
                &self,
                _: &[String],
                _: &[sasy_common::policy_engine::Action],
                _: Option<&str>,
                _: &[String],
                _: &SessionScope,
                _: Option<&str>,
                _: Option<&str>,
            ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error>
            {
                Ok(sasy_common::policy_engine::AuthorizationResponse {
                    results: vec![],
                    timing: None,
                })
            }
            fn reset(&self) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
                Ok(())
            }
            fn get_sync_status(&self) -> crate::engine::SyncStatus {
                crate::engine::SyncStatus {
                    current_sequence: 0,
                    node_count: 0,
                    edge_count: 0,
                    connected: true,
                }
            }
            fn set_connected(&self, _: bool) {}
            fn set_sequence(&self, _: i64) {}
            fn end_session(&self, scope: &SessionScope) -> bool {
                *self.owned_when_evicted.lock() =
                    Some(self.graph.get_session_owner(scope).unwrap().is_some());
                true
            }
        }

        let graph = Arc::new(GraphStore::new(None).unwrap());
        let owned_when_evicted = Arc::new(PlMutex::new(None));
        let svc = PolicyService::with_persistence(
            Arc::new(OwnerAtEviction {
                graph: Arc::clone(&graph),
                owned_when_evicted: Arc::clone(&owned_when_evicted),
            }),
            Arc::clone(&graph),
        );

        let scope = SessionScope::new("acme", "conv-1");
        graph
            .check_or_claim_session_owner(&scope, Some("alice"))
            .unwrap();

        let auth = AuthResult::success("alice", vec![], "test").with_tenant("acme");
        svc.end_session(end_session_request("conv-1", Some(auth)))
            .await
            .expect("alice owns the session");

        assert_eq!(
            *owned_when_evicted.lock(),
            Some(true),
            "the id was released while the evaluator was still live"
        );
        assert_eq!(graph.get_session_owner(&scope).unwrap(), None);
    }
    /// Claiming an unowned session is a durable write, so a bind that fails
    /// after the claim has to give the id back. Otherwise the session is
    /// locked to a principal that never bound anything: nobody else can bind
    /// it, end it, or authorize against it until an admin intervenes.
    #[tokio::test]
    async fn failed_session_bind_releases_a_fresh_ownership_claim() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv");
        assert_eq!(graph.get_session_owner(&scope).unwrap(), None);

        svc.set_policy(locked_test_request("// already installed"))
            .await
            .expect_err("a bind the engine refuses must fail the RPC");

        assert_eq!(
            graph.get_session_owner(&scope).unwrap(),
            None,
            "a bind that never published must not leave the session id claimed"
        );
    }

    /// `enforce_session_ownership` reports whether IT created the ownership
    /// row, and says no when the row was already someone else's.
    ///
    /// This is why the claim reports its own outcome instead of the caller
    /// reading the row first: the row is also written by observability
    /// ingest, which holds none of this service's locks, so a claim can land
    /// between a read and the claim that follows it. A rollback keyed on the
    /// earlier read would then delete a live owner's claim and leave their
    /// session unowned for anyone to take. Tested at this level because the
    /// pre-compile peek denies a non-owner before `set_policy` ever reaches
    /// the claim — the interleaving that gets here needs the owner to change
    /// underneath an in-flight call.
    #[tokio::test]
    async fn ownership_claim_reports_only_rows_it_created() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let scope = SessionScope::new("acme", "conv");

        // Absent row: this call creates it, and may undo it.
        assert!(
            svc.enforce_session_ownership(&scope, Some("agent"), false)
                .unwrap()
                .is_some(),
            "claiming an unowned session must report that it claimed"
        );
        // Already ours: nothing was created, so nothing may be undone.
        assert!(
            svc.enforce_session_ownership(&scope, Some("agent"), false)
                .unwrap()
                .is_none(),
            "re-touching a session we already own must not report a claim"
        );

        // Someone else's: denied, nothing created, and the row is untouched.
        let other = SessionScope::new("acme", "theirs");
        graph
            .check_or_claim_session_owner(&other, Some("someone-else"))
            .unwrap();
        let err = svc
            .enforce_session_ownership(&other, Some("agent"), false)
            .expect_err("another principal's session must be refused");
        assert_eq!(err.code(), tonic::Code::PermissionDenied);
        assert_eq!(
            graph.get_session_owner(&other).unwrap().as_deref(),
            Some("someone-else"),
        );

        // Admin passes the check but still claims nothing.
        assert!(
            svc.enforce_session_ownership(&other, Some("root"), true)
                .unwrap()
                .is_none(),
            "an admin override must not report a claim it did not make"
        );
        assert_eq!(
            graph.get_session_owner(&other).unwrap().as_deref(),
            Some("someone-else"),
        );
    }

    /// A bind that SUCCEEDS writes what the failure tests assert is absent.
    ///
    /// Without this, those tests are satisfied by a bind that never writes a
    /// pin, a config row or an ownership claim at all — "absent after a
    /// failure" and "never written" look identical from the outside. Every
    /// other engine fake in this file refuses to publish, so this is the only
    /// place the positive half is pinned.
    #[tokio::test]
    async fn a_successful_bind_writes_the_pin_the_config_and_the_claim() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(AcceptsTheBind), Arc::clone(&graph));
        let scope = SessionScope::new("acme", "conv");
        let hash = content_hash_souffle("// already installed");

        let mut req = locked_test_request("// already installed");
        req.get_mut().policy_metadata = vec![sasy_common::policy_engine::PolicyMetadataFact {
            rel: "rule_off".into(),
            a: "exfil".into(),
            b: String::new(),
        }];
        let resp = svc
            .set_policy(req)
            .await
            .expect("the bind succeeds")
            .into_inner();
        assert!(resp.accepted);

        assert_eq!(
            graph.get_policy_binding(&scope).unwrap().as_deref(),
            Some(hash.as_str()),
            "a successful bind records the pin"
        );
        let config = graph
            .get_binding_metadata(&scope, &hash)
            .unwrap()
            .expect("a successful bind records its config row");
        assert_eq!(config.len(), 1);
        assert_eq!(config[0].a, "exfil");
        assert_eq!(
            graph.get_session_owner(&scope).unwrap().as_deref(),
            Some("agent"),
            "a successful bind keeps the ownership claim it made"
        );
    }

    #[tokio::test]
    async fn kept_policy_publications_disarm_owner_rollback_tokens() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        for (mode, session) in ["published", "publication_succeeded", "detached"]
            .iter()
            .enumerate()
        {
            let scope = SessionScope::new("acme", *session);
            let token = svc
                .enforce_session_ownership(&scope, Some("agent"), false)
                .unwrap()
                .unwrap();
            let retained = token.clone();
            let mut staged = Staged::new(&svc);
            staged.record(DurableWrite::SessionOwner(token));
            match mode {
                0 => staged.published(),
                1 => {
                    staged.publication_succeeded();
                    drop(staged);
                }
                _ => {
                    staged.publishing();
                    drop(staged);
                }
            }
            assert!(!graph.release_session_owner_claim(&retained).unwrap());
            assert_eq!(
                graph.get_session_owner(&scope).unwrap().as_deref(),
                Some("agent")
            );
        }
    }

    #[tokio::test]
    async fn unused_policy_claim_is_released_but_intervening_ingest_preserves_it() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let scope = SessionScope::new("acme", "conv");
        let unused = svc
            .enforce_session_ownership(&scope, Some("agent"), false)
            .unwrap();
        assert!(unused.is_some());
        svc.release_claim_if_unused(&scope, unused);
        assert_eq!(graph.get_session_owner(&scope).unwrap(), None);

        let policy_claim = svc
            .enforce_session_ownership(&scope, Some("agent"), false)
            .unwrap();
        // Observability uses this same store API outside the policy locks.
        let (result, token) = graph
            .claim_session_owner_with_token(&scope, Some("agent"))
            .unwrap();
        assert_eq!(result, sasy_graph::OwnerClaim::AlreadyHeld);
        assert!(token.is_none());
        graph
            .merge_events(
                &scope,
                Some("agent"),
                vec![sasy_common::observability::Event {
                    id: Some("message".into()),
                    text: Some("ingest succeeded".into()),
                    ..Default::default()
                }],
            )
            .unwrap();
        svc.release_claim_if_unused(&scope, policy_claim);
        assert_eq!(
            graph.get_session_owner(&scope).unwrap().as_deref(),
            Some("agent")
        );
        assert_eq!(graph.session_counts(&scope).0, 1);
    }

    #[tokio::test]
    async fn policy_claim_invalidates_observability_rollback() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(StubEngine::new(), Arc::clone(&graph));
        let scope = SessionScope::new("acme", "conv");
        let (_, ingest_claim) = graph
            .claim_session_owner_with_token(&scope, Some("agent"))
            .unwrap();
        assert!(svc
            .enforce_session_ownership(&scope, Some("agent"), false)
            .unwrap()
            .is_none());
        graph.put_policy_binding(&scope, "policy-hash").unwrap();
        assert!(!graph
            .release_session_owner_claim(&ingest_claim.unwrap())
            .unwrap());
        assert_eq!(
            graph.get_session_owner(&scope).unwrap().as_deref(),
            Some("agent")
        );
        assert_eq!(
            graph.get_policy_binding(&scope).unwrap().as_deref(),
            Some("policy-hash")
        );
    }

    /// The mirror case: a session someone already owns must keep its owner
    /// when a later bind by that same principal fails. The rollback restores
    /// the prior value rather than deleting the row.
    #[tokio::test]
    async fn failed_session_bind_keeps_an_existing_owner() {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        let svc = PolicyService::with_persistence(Arc::new(RefusesToPublish), Arc::clone(&graph));

        let scope = SessionScope::new("acme", "conv");
        graph
            .check_or_claim_session_owner(&scope, Some("agent"))
            .unwrap();

        svc.set_policy(locked_test_request("// already installed"))
            .await
            .expect_err("a bind the engine refuses must fail the RPC");

        assert_eq!(
            graph.get_session_owner(&scope).unwrap().as_deref(),
            Some("agent"),
            "an existing owner must survive a failed re-bind"
        );
    }
    /// Rolling a configuration write back has to drop the evaluators that
    /// could have seeded from it. Authorization takes no lock by design, so
    /// one can spawn between the write and the rollback, and a seed lasts the
    /// evaluator's lifetime — leaving a session policed under configuration
    /// from a rollout the caller was told had failed.
    ///
    /// Counted, not merely observed: a `Force` rollback undoes two separate
    /// configuration writes (the tenant-wide row and the per-session rows it
    /// cleared) and each has to evict. Asserting only "something evicted"
    /// passes with either one removed.
    struct RecordsEvictions {
        tenants: Arc<parking_lot::Mutex<Vec<String>>>,
    }

    impl Engine for RecordsEvictions {
        fn apply_graph_updates(
            &self,
            _: Vec<crate::engine::GraphUpdate>,
        ) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn check_authorization(
            &self,
            _: &[String],
            _: &[sasy_common::policy_engine::Action],
            _: Option<&str>,
            _: &[String],
            _: &SessionScope,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
            Ok(sasy_common::policy_engine::AuthorizationResponse {
                results: vec![],
                timing: None,
            })
        }
        fn reset(&self) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
            Ok(())
        }
        fn get_sync_status(&self) -> crate::engine::SyncStatus {
            crate::engine::SyncStatus {
                current_sequence: 0,
                node_count: 0,
                edge_count: 0,
                connected: true,
            }
        }
        fn set_connected(&self, _: bool) {}
        fn set_sequence(&self, _: i64) {}
        fn lookup_policy_by_content(&self, _: &str, _: &str) -> Option<String> {
            Some("pid-existing".into())
        }
        fn promote_to_default(&self, _: &str, _: &str, _: bool) -> Result<(), anyhow::Error> {
            Err(anyhow::anyhow!("registry refused the promotion"))
        }
        fn evict_tenant_evaluators(&self, tenant: &str) -> usize {
            self.tenants.lock().push(tenant.to_string());
            0
        }
    }

    async fn evictions_from(req: Request<SetPolicyRequest>) -> Vec<String> {
        let graph = Arc::new(GraphStore::new(None).unwrap());
        // Both "configuration writes" have to write something, or the test
        // counts evictions for rollbacks that restored nothing and would also
        // fail a correct future change that skipped the evict when there was
        // nothing to restore.
        graph
            .put_policy_metadata(
                "acme",
                &content_hash_souffle("// already installed"),
                &[sasy_graph::PolicyMetadataFact {
                    rel: "rule_on".into(),
                    a: "exfil".into(),
                    b: String::new(),
                }],
            )
            .unwrap();
        graph
            .put_binding_metadata(
                &SessionScope::new("acme", "pinned"),
                "pinned-hash",
                &[sasy_graph::PolicyMetadataFact {
                    rel: "rule_off".into(),
                    a: "review_gate".into(),
                    b: String::new(),
                }],
            )
            .unwrap();
        let tenants = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let svc = PolicyService::with_persistence(
            Arc::new(RecordsEvictions {
                tenants: Arc::clone(&tenants),
            }),
            Arc::clone(&graph),
        );
        svc.set_policy(req)
            .await
            .expect_err("a promotion the engine refuses must fail the RPC");
        let out = tenants.lock().clone();
        out
    }

    /// Default rollback undoes one configuration write — the tenant-wide row.
    #[tokio::test]
    async fn failed_default_rollout_evicts_for_the_tenant_config() {
        let mut req = default_request("// already installed");
        req.get_mut().policy_metadata = vec![sasy_common::policy_engine::PolicyMetadataFact {
            rel: "rule_off".into(),
            a: "exfil".into(),
            b: String::new(),
        }];
        assert_eq!(
            evictions_from(req).await,
            vec!["acme".to_string()],
            "the rolled-back tenant config must evict the tenant's evaluators"
        );
    }

    /// Force rollback undoes two — the tenant-wide row and the per-session
    /// rows it cleared — so it must evict for both.
    #[tokio::test]
    async fn failed_force_rollout_evicts_for_both_config_writes() {
        assert_eq!(
            evictions_from(force_request("// already installed")).await,
            vec!["acme".to_string(), "acme".to_string()],
            "both the cleared per-session config and the tenant config must evict"
        );
    }
}
