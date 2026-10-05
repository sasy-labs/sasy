//! gRPC Observability service implementation.

use std::collections::HashSet;
use std::sync::Arc;

use sasy_auth::{is_rbac_enabled, request_has_role, request_principal, request_tenant};
use sasy_common::observability::{
    observability_server::Observability, Computation, Computations, Dependencies, Edge, Event,
    EventSnapshots, Events, EventsWithDependencies, Graph, IDs, Response, SliceRequest,
    SpanRequest, TraceGraph, TraceRequest,
};
use sasy_common::SessionScope;
use sasy_graph::GraphStore;
use tonic::{Request, Response as TonicResponse, Status};

use crate::rbac::check_role;

/// Per-request auth context, derived once at the top of every handler: the
/// RBAC role gate, then the tenant / principal / admin / enforce flags that
/// the downstream write- and read-ownership gates consume.
///
/// All three identity fields are the CONNECTION's own: `request_tenant`,
/// `request_principal` and `request_has_role`. This service never reads the
/// delegation headers (`x-tenant` / `x-principal` / `x-roles`) that a
/// `service-proxy` relay attaches, so there is no end user here whose rows
/// could be reached with somebody else's admin. That is why `is_admin` is a
/// single role check, and not the two-identity conjunction the policy engine
/// uses for its scope, functor-source and session-ownership waivers: those
/// gates route by the DELEGATED identity, so they have to ask about the relay
/// and the user it forwards. If any handler here ever starts routing by
/// `request_effective_tenant` or `request_effective_principal`, `is_admin`
/// has to become that conjunction at the same time.
struct RequestAuth {
    tenant: String,
    principal: Option<String>,
    is_admin: bool,
    enforce: bool,
}

/// Run the RBAC role check and capture the request's auth context. Borrows
/// the request (metadata only), so the caller can still take `into_inner()`.
#[allow(clippy::result_large_err)]
fn request_auth<T>(request: &Request<T>, required_role: &str) -> Result<RequestAuth, Status> {
    check_role(request, required_role)?;
    Ok(RequestAuth {
        tenant: request_tenant(request, "default"),
        principal: request_principal(request),
        is_admin: request_has_role(request, sasy_common::roles::ADMIN),
        enforce: is_rbac_enabled(),
    })
}

/// Wraps a [`GraphStore`] and implements the
/// `Observability` gRPC service trait.
pub struct ObservabilityService {
    store: Arc<GraphStore>,
}

impl ObservabilityService {
    pub fn new(store: Arc<GraphStore>) -> Self {
        Self { store }
    }

    /// Check + claim ownership of the batch's `(tenant,
    /// session_id)` scope. The session is an envelope field, so
    /// the whole batch shares one scope — one ownership claim
    /// covers the entire request.
    ///
    /// Empty `session_id` targets the per-tenant **global**
    /// session (visible to every observer in the tenant);
    /// non-admin writes there are rejected.
    /// Returns rollback permission only when this call creates the owner.
    /// A subsequent ownership check invalidates it, including by the same
    /// principal. See [`Self::release_claim_if_unused`].
    #[allow(clippy::result_large_err)]
    fn enforce_writes(
        &self,
        tenant: &str,
        principal: Option<&str>,
        is_admin: bool,
        session_id: &str,
    ) -> Result<Option<sasy_graph::OwnerClaimToken>, Status> {
        if session_id.is_empty() {
            if !is_admin {
                return Err(Status::permission_denied(
                    "writes to the per-tenant global session require admin",
                ));
            }
            // Admin writes to global: no ownership to claim.
            return Ok(None);
        }
        let scope = SessionScope::new(tenant, session_id.to_string());
        match self.store.claim_session_owner_with_token(&scope, principal) {
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

    /// Give back an unused claim only if no other caller has relied on it.
    fn release_claim_if_unused(
        &self,
        scope: &SessionScope,
        token: Option<sasy_graph::OwnerClaimToken>,
    ) {
        let Some(token) = token else {
            return;
        };
        if let Err(e) = self.store.release_session_owner_claim(&token) {
            tracing::error!(scope = %scope, error = %e,
                "could not release unused owner claim; ownership retained");
        }
        self.store.commit_session_owner_claim(&token);
    }

    fn commit_claim(&self, token: Option<sasy_graph::OwnerClaimToken>) {
        if let Some(token) = token {
            self.store.commit_session_owner_claim(&token);
        }
    }

    #[allow(clippy::result_large_err)]
    fn resolve_snapshot_request(
        &self,
        request: Request<EventSnapshots>,
        allow_references: bool,
    ) -> Result<TonicResponse<IDs>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_WRITER)?;
        let inner = request.into_inner();
        if !allow_references && inner.snapshots.iter().any(|s| s.content_hash.is_some()) {
            return Err(Status::invalid_argument(
                "compact references require ResolveSnapshots",
            ));
        }
        if inner.snapshots.is_empty() {
            return Ok(TonicResponse::new(IDs { ids: Vec::new() }));
        }
        let session_id = inner.session_id.unwrap_or_default();
        check_session_id_transportable(&session_id)?;
        for snapshot in &inner.snapshots {
            let event = snapshot
                .event
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("event is required"))?;
            check_events_transportable(std::slice::from_ref(event))?;
            check_edges_transportable(&snapshot.dependencies)?;
            reject_embedded_nul(&[snapshot.base_id.as_deref()])?;
        }
        let scope = SessionScope::new(&tenant, session_id.clone());
        let claimed = if enforce {
            self.enforce_writes(&tenant, principal.as_deref(), is_admin, &session_id)?
        } else {
            None
        };
        match self
            .store
            .resolve_events(&scope, principal.as_deref(), inner.snapshots)
        {
            Ok(ids) => {
                self.commit_claim(claimed);
                Ok(TonicResponse::new(IDs { ids }))
            }
            Err(error) => {
                self.release_claim_if_unused(&scope, claimed);
                Err(graph_write_status(error))
            }
        }
    }

    /// Read-side counterpart to [`Self::enforce_writes`]. The
    /// `(tenant, id) → scope` reverse index dedups node ids
    /// across tenants but not across sessions within a tenant,
    /// so a same-tenant principal could otherwise resolve another
    /// principal's session via an arbitrary-id read
    /// (`backward_slice`, `forward_slice`, `get_span`, `get_trace`).
    /// Verify the resolved scope's owner matches the caller, or
    /// return `not_found` so cross-session existence isn't leaked
    /// (matches the cross-tenant guard above the same call).
    ///
    /// Unowned scopes (no claim on file) are readable by any
    /// caller in the tenant: a session with no stored owner is not
    /// claimed by anyone. Once any write hits the
    /// session, [`enforce_writes`] claims it and this check kicks
    /// in.
    #[allow(clippy::result_large_err)]
    fn enforce_session_read(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        is_admin: bool,
        not_found_kind: &str,
        not_found_id: &str,
    ) -> Result<(), Status> {
        if is_admin {
            return Ok(());
        }
        let owner = self
            .store
            .get_session_owner(scope)
            .map_err(|e| Status::internal(format!("ownership check: {e}")))?;
        match (owner.as_deref(), principal) {
            (None, _) => Ok(()),
            (Some(o), Some(p)) if o == p => Ok(()),
            (Some(_), _) => Err(Status::not_found(format!(
                "{not_found_kind} not found: {not_found_id}"
            ))),
        }
    }

    /// Resolve the owning scope for an id-based read and authorize it.
    /// `session` is the caller's optional session hint (reads its own shard
    /// directly; otherwise the id is resolved via the tenant-wide reverse
    /// index). `kind` labels the entity ("event", "span") in not-found /
    /// ownership errors. Enforces the global-session admin gate and, for
    /// non-global scopes, per-session read ownership. Shared by
    /// backward_slice / forward_slice / get_span.
    // Mirrors the per-field auth context (RequestAuth) the handlers already
    // hold as locals; passing them individually keeps the read handlers'
    // preamble identical to the write handlers' rather than splitting on
    // struct-vs-fields. The arg count is inherent to the inputs.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn authorize_id_read(
        &self,
        tenant: &str,
        principal: Option<&str>,
        is_admin: bool,
        enforce: bool,
        session: Option<&str>,
        kind: &str,
        id: &str,
    ) -> Result<(), Status> {
        let scope = match session {
            Some(sid) => SessionScope::new(tenant, sid),
            None => self
                .store
                .scope_for_node(tenant, id)
                .ok_or_else(|| Status::not_found(format!("{kind} not found: {id}")))?,
        };
        // Nodes in the per-tenant **global** session are written by admins
        // only; reading them is similarly admin-only so a non-admin can't
        // observe admin-written cross-session state. (Test bypass: with no
        // auth interceptor wired, `enforce` is false and this is skipped.)
        if enforce && scope.is_global() && !is_admin {
            return Err(Status::permission_denied(
                "reading the per-tenant global session requires admin",
            ));
        }
        if enforce && !scope.is_global() {
            self.enforce_session_read(&scope, principal, is_admin, kind, id)?;
        }
        Ok(())
    }
}

/// Reject any record carrying an embedded NUL before it is stored.
///
/// The evaluator IPC layer refuses a request containing one
/// (`sasy-policy`'s `validate_transportable_request`), and it refuses the
/// WHOLE request — a session's bootstrap is one message, so a single poisoned
/// field takes that session's entire graph down with it. Accepting the record
/// here writes the poison to RocksDB, where it survives the process: every
/// respawn re-reads it, fails the same way, and the session can never be
/// evaluated again. There is no self-healing path, and no way to remove it
/// short of dropping the session's data.
///
/// So it is refused at the door, where the client still learns about it and
/// can send something else. Other control characters are left alone: they
/// travel through the IPC layer intact and are the caller's business.
#[allow(
    clippy::result_large_err,
    reason = "Return tonic::Status directly for propagation into the gRPC response"
)]
fn reject_embedded_nul(fields: &[Option<&str>]) -> Result<(), Status> {
    if fields.iter().flatten().any(|f| f.contains('\0')) {
        return Err(Status::invalid_argument(
            "record contains an embedded NUL character, which the policy evaluator \
             cannot accept; strip it before recording",
        ));
    }
    Ok(())
}

/// The session id travels into every `NodeCreated`/`EdgeCreated` the
/// bootstrap emits, so a NUL in it poisons the partition exactly as a NUL in
/// a message would. The writer picks its own id, so this is self-inflicted
/// rather than an attack on somebody else's session — but the failure is the
/// same permanent one, and refusing costs nothing.
#[allow(
    clippy::result_large_err,
    reason = "Return tonic::Status directly for propagation into the gRPC response"
)]
fn check_session_id_transportable(session_id: &str) -> Result<(), Status> {
    reject_embedded_nul(&[Some(session_id)])
}

#[allow(
    clippy::result_large_err,
    reason = "Return tonic::Status directly for propagation into the gRPC response"
)]
fn check_events_transportable(events: &[Event]) -> Result<(), Status> {
    for e in events {
        reject_embedded_nul(&[
            e.text.as_deref(),
            e.agent.as_deref(),
            e.id.as_deref(),
            e.entity.as_deref(),
            e.metadata.as_deref(),
        ])?;
        for t in e.tools.iter().chain(e.derived_from.iter()) {
            reject_embedded_nul(&[t.name.as_deref(), t.arguments.as_deref()])?;
        }
    }
    Ok(())
}

#[allow(
    clippy::result_large_err,
    reason = "Return tonic::Status directly for propagation into the gRPC response"
)]
fn check_edges_transportable(edges: &[Edge]) -> Result<(), Status> {
    for e in edges {
        reject_embedded_nul(&[
            Some(e.source.as_str()),
            Some(e.destination.as_str()),
            e.entity.as_deref(),
        ])?;
    }
    Ok(())
}

#[allow(
    clippy::result_large_err,
    reason = "Return tonic::Status directly for propagation into the gRPC response"
)]
fn check_computations_transportable(comps: &[Computation]) -> Result<(), Status> {
    for c in comps {
        reject_embedded_nul(&[
            Some(c.trace_id.as_str()),
            Some(c.span_id.as_str()),
            c.parent_span_id.as_deref(),
            Some(c.name.as_str()),
            c.status_message.as_deref(),
            Some(c.attributes_json.as_str()),
            Some(c.events_json.as_str()),
            c.service_name.as_deref(),
            c.service_version.as_deref(),
            c.output_message_id.as_deref(),
        ])?;
        for id in c.input_message_ids.iter().chain(c.linked_span_ids.iter()) {
            reject_embedded_nul(&[Some(id.as_str())])?;
        }
    }
    Ok(())
}

fn graph_write_status(error: sasy_graph::GraphError) -> Status {
    match error {
        sasy_graph::GraphError::InvalidSnapshot(message) => Status::invalid_argument(message),
        sasy_graph::GraphError::ImmutableViolation(message)
        | sasy_graph::GraphError::ContentHashAmbiguity(message) => {
            Status::failed_precondition(message)
        }
        other => Status::internal(other.to_string()),
    }
}

#[tonic::async_trait]
impl Observability for ObservabilityService {
    async fn register_events(
        &self,
        request: Request<Events>,
    ) -> Result<TonicResponse<IDs>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_WRITER)?;
        let inner = request.into_inner();
        let session_id = inner.session_id.clone().unwrap_or_default();
        let scope = SessionScope::new(&tenant, session_id.clone());
        tracing::info!("RegisterEvents count={}", inner.events.len());
        // A no-op batch writes no graph state, so it must not claim
        // ownership — otherwise any same-tenant observability-writer could
        // stake a claim on a victim's session_id with `events=[]` and lock
        // the rightful principal out of later writes / SetPolicy / checks.
        // "No-op" has to mean what the STORE means by it, not just an empty
        // vector: `upsert_event_into_shard` skips an event with nothing to
        // store, so a batch of one content-free event is non-empty here and
        // writes nothing there — and would otherwise claim a victim's session
        // id on the way past. Asking the store's own predicate keeps the two
        // in step.
        if inner.events.iter().all(sasy_graph::event_carries_nothing) {
            return Ok(TonicResponse::new(IDs { ids: Vec::new() }));
        }
        check_session_id_transportable(&session_id)?;
        check_events_transportable(&inner.events)?;
        // One envelope = one scope = one ownership claim per
        // request. The HashSet-dedup logic from the per-event
        // model is gone with the per-event field.
        let claimed = if enforce {
            self.enforce_writes(&tenant, principal.as_deref(), is_admin, &session_id)?
        } else {
            None
        };
        let ids = match self
            .store
            .merge_events(&scope, principal.as_deref(), inner.events)
        {
            Ok(ids) => ids,
            Err(e) => {
                self.release_claim_if_unused(&scope, claimed);
                return Err(graph_write_status(e));
            }
        };
        self.commit_claim(claimed);
        tracing::info!("RegisterEvents ok ids={}", ids.len());
        Ok(TonicResponse::new(IDs { ids }))
    }

    async fn register_dependencies(
        &self,
        request: Request<Dependencies>,
    ) -> Result<TonicResponse<Response>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_WRITER)?;
        let inner = request.into_inner();
        let session_id = inner.session_id.clone().unwrap_or_default();
        let scope = SessionScope::new(&tenant, session_id.clone());
        tracing::info!("RegisterDependencies count={}", inner.edges.len());
        // No edges → nothing to persist; don't claim ownership for a no-op
        // (see register_events).
        if inner.edges.is_empty() {
            return Ok(TonicResponse::new(Response {
                response: Some("ok".to_string()),
            }));
        }
        check_session_id_transportable(&session_id)?;
        check_edges_transportable(&inner.edges)?;
        let claimed = if enforce {
            self.enforce_writes(&tenant, principal.as_deref(), is_admin, &session_id)?
        } else {
            None
        };
        if let Err(e) = self
            .store
            .merge_dependencies(&scope, principal.as_deref(), inner.edges)
        {
            self.release_claim_if_unused(&scope, claimed);
            return Err(graph_write_status(e));
        }
        self.commit_claim(claimed);
        tracing::info!("RegisterDependencies ok");
        Ok(TonicResponse::new(Response {
            response: Some("ok".to_string()),
        }))
    }

    async fn register_events_with_dependencies(
        &self,
        request: Request<EventsWithDependencies>,
    ) -> Result<TonicResponse<IDs>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_WRITER)?;
        let inner = request.into_inner();
        let session_id = inner.session_id.clone().unwrap_or_default();
        let scope = SessionScope::new(&tenant, session_id.clone());
        // No events and no edges → nothing to persist; don't let a no-op
        // batch claim ownership of the session (see register_events).
        // Same rule as `register_events`: a batch of content-free events
        // stores nothing, so it must not claim the session either.
        if inner.events.iter().all(sasy_graph::event_carries_nothing) && inner.edges.is_empty() {
            return Ok(TonicResponse::new(IDs { ids: Vec::new() }));
        }
        check_session_id_transportable(&session_id)?;
        check_events_transportable(&inner.events)?;
        check_edges_transportable(&inner.edges)?;
        let claimed = if enforce {
            self.enforce_writes(&tenant, principal.as_deref(), is_admin, &session_id)?
        } else {
            None
        };
        let ids = match self.store.merge_events_with_dependencies(
            &scope,
            principal.as_deref(),
            inner.events,
            inner.edges,
        ) {
            Ok(ids) => ids,
            Err(e) => {
                self.release_claim_if_unused(&scope, claimed);
                return Err(graph_write_status(e));
            }
        };
        self.commit_claim(claimed);
        Ok(TonicResponse::new(IDs { ids }))
    }

    async fn resolve_events(
        &self,
        request: Request<EventSnapshots>,
    ) -> Result<TonicResponse<IDs>, Status> {
        self.resolve_snapshot_request(request, false)
    }

    async fn resolve_snapshots(
        &self,
        request: Request<EventSnapshots>,
    ) -> Result<TonicResponse<IDs>, Status> {
        self.resolve_snapshot_request(request, true)
    }

    async fn backward_slice(
        &self,
        request: Request<SliceRequest>,
    ) -> Result<TonicResponse<Graph>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_READER)?;
        let req = request.into_inner();
        let event_id = req
            .event_id
            .ok_or_else(|| Status::invalid_argument("event_id is required"))?;
        // Tenant guard: don't reveal the existence of cross-tenant
        // nodes — a NotFound for an unknown id and a NotFound for
        // a wrong-tenant id are indistinguishable to the caller.
        // The reverse index is keyed on `(tenant, id)` so a colliding
        // id in a different tenant returns None.
        // Prefer the caller-named session (reads its own shard
        // directly, removing the intra-tenant id->scope ambiguity);
        // fall back to the tenant-wide reverse index for id-only
        // probes. The tenant always comes from auth, so a supplied
        // session can never reach another tenant's shard.
        let session = req.session_id.as_deref().filter(|s| !s.is_empty());
        self.authorize_id_read(
            &tenant,
            principal.as_deref(),
            is_admin,
            enforce,
            session,
            "event",
            &event_id,
        )?;
        let (nodes, edges) = self
            .store
            .backward_slice(&tenant, &event_id, session, req.max_depth)
            .map_err(|e| Status::from(crate::error::ServerError::Graph(e)))?;
        Ok(TonicResponse::new(Graph { nodes, edges }))
    }

    async fn forward_slice(
        &self,
        request: Request<SliceRequest>,
    ) -> Result<TonicResponse<Graph>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_READER)?;
        let req = request.into_inner();
        let event_id = req
            .event_id
            .ok_or_else(|| Status::invalid_argument("event_id is required"))?;
        // Prefer the caller-named session (reads its own shard
        // directly, removing the intra-tenant id->scope ambiguity);
        // fall back to the tenant-wide reverse index for id-only
        // probes. The tenant always comes from auth, so a supplied
        // session can never reach another tenant's shard.
        let session = req.session_id.as_deref().filter(|s| !s.is_empty());
        self.authorize_id_read(
            &tenant,
            principal.as_deref(),
            is_admin,
            enforce,
            session,
            "event",
            &event_id,
        )?;
        let (nodes, edges) = self
            .store
            .forward_slice(&tenant, &event_id, session, req.max_depth)
            .map_err(|e| Status::from(crate::error::ServerError::Graph(e)))?;
        Ok(TonicResponse::new(Graph { nodes, edges }))
    }

    async fn register_computations(
        &self,
        request: Request<Computations>,
    ) -> Result<TonicResponse<IDs>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_WRITER)?;
        let inner = request.into_inner();
        let session_id = inner.session_id.clone().unwrap_or_default();
        let scope = SessionScope::new(&tenant, session_id.clone());
        // A no-op batch writes no state, so it must not claim ownership —
        // the same guard the three sibling handlers carry. Without it any
        // same-tenant observability-writer stakes a claim on a victim's
        // session_id with `computations=[]` and locks the rightful principal
        // out of every later write, SetPolicy and EndSession.
        if inner.computations.is_empty() {
            return Ok(TonicResponse::new(IDs { ids: Vec::new() }));
        }
        check_session_id_transportable(&session_id)?;
        check_computations_transportable(&inner.computations)?;
        let claimed = if enforce {
            self.enforce_writes(&tenant, principal.as_deref(), is_admin, &session_id)?
        } else {
            None
        };
        let ids =
            match self
                .store
                .merge_computations(&scope, principal.as_deref(), inner.computations)
            {
                Ok(ids) => ids,
                Err(e) => {
                    self.release_claim_if_unused(&scope, claimed);
                    return Err(graph_write_status(e));
                }
            };
        self.commit_claim(claimed);
        Ok(TonicResponse::new(IDs { ids }))
    }

    async fn get_trace(
        &self,
        request: Request<TraceRequest>,
    ) -> Result<TonicResponse<TraceGraph>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_READER)?;
        let req = request.into_inner();
        if !self.store.trace_has_tenant(&tenant, &req.trace_id) {
            return Err(Status::not_found(format!(
                "trace not found: {}",
                req.trace_id
            )));
        }
        // A trace can span multiple sessions in a tenant. Filter
        // the walk to scopes the caller owns so cross-principal
        // spans aren't included in the response (admin bypasses).
        // No allowed-scope filter for admin keeps the result
        // identical to the legacy callsite.
        let allowed: Option<HashSet<SessionScope>> = if enforce && !is_admin {
            let scopes = self.store.scopes_for_trace(&tenant, &req.trace_id);
            let mut keep = HashSet::with_capacity(scopes.len());
            for scope in scopes {
                if scope.is_global() {
                    continue;
                }
                if self
                    .enforce_session_read(
                        &scope,
                        principal.as_deref(),
                        is_admin,
                        "trace",
                        &req.trace_id,
                    )
                    .is_ok()
                {
                    keep.insert(scope);
                }
            }
            if keep.is_empty() {
                return Err(Status::not_found(format!(
                    "trace not found: {}",
                    req.trace_id
                )));
            }
            Some(keep)
        } else {
            None
        };
        let trace = self
            .store
            .get_trace(
                &tenant,
                &req.trace_id,
                req.start_time_ns,
                req.end_time_ns,
                allowed.as_ref(),
            )
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(TonicResponse::new(trace))
    }

    async fn get_span(
        &self,
        request: Request<SpanRequest>,
    ) -> Result<TonicResponse<Computation>, Status> {
        let RequestAuth {
            tenant,
            principal,
            is_admin,
            enforce,
        } = request_auth(&request, sasy_common::roles::OBSERVABILITY_READER)?;
        let req = request.into_inner();
        let session = req.session_id.as_deref().filter(|s| !s.is_empty());
        self.authorize_id_read(
            &tenant,
            principal.as_deref(),
            is_admin,
            enforce,
            session,
            "span",
            &req.span_id,
        )?;
        let comp = self
            .store
            .get_span(&tenant, &req.span_id, session)
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found(format!("span not found: {}", req.span_id,)))?;
        Ok(TonicResponse::new(comp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::observability::{Edge, Event, Role};

    fn make_store() -> Arc<GraphStore> {
        Arc::new(GraphStore::new(None).unwrap())
    }

    /// Helper for tests: wrap `value` in a Request whose
    /// extensions carry an admin AuthResult, so the handler's
    /// admin-gates / ownership checks see a real auth context.
    /// Tests that want to exercise non-admin paths construct
    /// their own AuthResult inline.
    fn admin_request<T>(value: T) -> Request<T> {
        let mut req = Request::new(value);
        req.extensions_mut().insert(
            sasy_auth::AuthResult::success(
                "test-admin",
                vec![
                    "admin".into(),
                    "observability-writer".into(),
                    "observability-reader".into(),
                ],
                "test",
            )
            .with_tenant("default"),
        );
        req
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

    /// Build a non-admin `observability-writer` request for `principal`.
    fn writer_request<T>(value: T, principal: &str) -> Request<T> {
        let mut req = Request::new(value);
        req.extensions_mut().insert(
            sasy_auth::AuthResult::success(principal, vec!["observability-writer".into()], "test")
                .with_tenant("default"),
        );
        req
    }

    fn edge(source: &str, destination: &str) -> Edge {
        Edge {
            source: source.into(),
            destination: destination.into(),
            message_index: None,
            proximal: None,
            principal: None,
            entity: None,
        }
    }

    /// A write that the store refuses must not leave the session id staked
    /// out by the principal whose write never landed.
    ///
    /// Ownership is claimed by the permission check, which runs before the
    /// store is touched. The store can REFUSE a batch — an unattachable
    /// dependency edge, a quarantined shard — and a refused call must give the
    /// claim back, or the session's rightful owner is locked out of its own
    /// first event, its policy bind and its teardown by a session that does
    /// not exist.
    #[tokio::test]
    async fn a_refused_write_gives_back_the_ownership_it_claimed() {
        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));
        let scope = SessionScope::new("default", "s-contested");

        // `ghost` is not in the batch and not in the store, so the edge
        // cannot attach and the whole batch is refused.
        let doomed = EventsWithDependencies {
            session_id: Some("s-contested".into()),
            events: vec![ev("a", "first")],
            edges: vec![edge("ghost", "a")],
        };
        let err = svc
            .register_events_with_dependencies(writer_request(doomed, "squatter"))
            .await
            .expect_err("an unattachable edge must fail the batch");
        assert_eq!(err.code(), tonic::Code::Internal, "got: {err}");

        assert_eq!(
            store.get_session_owner(&scope).unwrap(),
            None,
            "a write that stored nothing must not own the session"
        );

        // And the consequence that matters: somebody else can still have it.
        let good = EventsWithDependencies {
            session_id: Some("s-contested".into()),
            events: vec![ev("a", "first"), ev("b", "second")],
            edges: vec![edge("a", "b")],
        };
        svc.register_events_with_dependencies(writer_request(good, "rightful"))
            .await
            .expect("the rightful owner must not be locked out by a failed write");
        assert_eq!(
            store.get_session_owner(&scope).unwrap().as_deref(),
            Some("rightful")
        );

        // The release is confined to claims the failing call itself made: a
        // later failure by the owner must not hand its session away.
        let doomed_again = EventsWithDependencies {
            session_id: Some("s-contested".into()),
            events: vec![ev("c", "third")],
            edges: vec![edge("ghost", "c")],
        };
        assert!(svc
            .register_events_with_dependencies(writer_request(doomed_again, "rightful"))
            .await
            .is_err());
        assert_eq!(
            store.get_session_owner(&scope).unwrap().as_deref(),
            Some("rightful"),
            "an established owner keeps its session across a failed write"
        );
    }

    /// A NUL in any recorded string is refused at the door.
    ///
    /// Accepting it writes poison to RocksDB: the evaluator IPC layer rejects
    /// the whole request containing one, and a session's bootstrap is one
    /// request, so that session can never be evaluated again — across
    /// restarts, with no self-healing path.
    #[tokio::test]
    async fn an_event_with_an_embedded_nul_is_refused() {
        let svc = ObservabilityService::new(make_store());
        let err = svc
            .register_events(admin_request(Events {
                session_id: Some("s1".into()),
                events: vec![ev("x", "hello\0world")],
            }))
            .await
            .expect_err("a NUL-bearing event must be refused");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// `metadata` is one of those strings: it travels to the evaluator like
    /// the rest of the event, so a NUL in it wedges the session the same way.
    #[tokio::test]
    async fn an_event_with_an_embedded_nul_in_its_metadata_is_refused() {
        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));
        let mut event = ev("x", "hello");
        event.metadata = Some("{\"s:a\":\"b\0c\"}".into());

        let err = svc
            .register_events(admin_request(Events {
                session_id: Some("s1".into()),
                events: vec![event],
            }))
            .await
            .expect_err("a NUL in metadata must be refused");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert_eq!(
            store.session_counts(&SessionScope::new("default", "s1")),
            (0, 0),
            "nothing may be stored for a refused batch"
        );
    }

    /// Refused before anything is stored, so a retry with clean text works.
    #[tokio::test]
    async fn a_refused_nul_event_leaves_the_session_empty() {
        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));
        let scope = SessionScope::new("default", "s1");

        let _ = svc
            .register_events(admin_request(Events {
                session_id: Some("s1".into()),
                events: vec![ev("x", "bad\0")],
            }))
            .await;
        assert_eq!(
            store.session_counts(&scope),
            (0, 0),
            "nothing may be stored for a refused batch"
        );

        svc.register_events(admin_request(Events {
            session_id: Some("s1".into()),
            events: vec![ev("x", "clean")],
        }))
        .await
        .expect("a clean retry must succeed");
        assert_eq!(store.session_counts(&scope).0, 1);
    }

    /// A batch of content-free events stores nothing, so it must not claim
    /// the session id either.
    ///
    /// `events.is_empty()` is the wrong predicate: the store decides per
    /// event whether there is anything to write, so one event with every field
    /// unset is "non-empty" here and "nothing" there, and would stake a claim
    /// on a victim's session while persisting no graph state. The rightful
    /// principal would then be locked out of writes, SetPolicy and EndSession.
    #[tokio::test]
    async fn a_batch_of_content_free_events_does_not_claim_the_session() {
        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));
        let scope = SessionScope::new("default", "victim");

        let hollow = Event {
            text: None,
            agent: None,
            role: None,
            id: Some("x".into()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: None,
            metadata: None,
        };
        svc.register_events(admin_request(Events {
            session_id: Some("victim".into()),
            events: vec![hollow],
        }))
        .await
        .expect("a content-free batch is accepted");

        assert_eq!(
            store.get_session_owner(&scope).unwrap(),
            None,
            "a batch that stored nothing must leave the id claimable"
        );
        assert_eq!(store.session_counts(&scope), (0, 0));
    }

    /// A no-op `RegisterComputations` must not claim the session id, matching
    /// the three sibling handlers. Otherwise any same-tenant writer squats a
    /// victim's session with an empty batch and locks them out.
    #[tokio::test]
    async fn an_empty_computation_batch_does_not_claim_the_session() {
        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));
        let scope = SessionScope::new("default", "victim");

        svc.register_computations(admin_request(Computations {
            session_id: Some("victim".into()),
            computations: vec![],
        }))
        .await
        .expect("an empty batch is accepted");

        assert_eq!(
            store.get_session_owner(&scope).unwrap(),
            None,
            "a batch that stored nothing must leave the id claimable"
        );
    }

    #[tokio::test]
    async fn register_and_slice() {
        let store = make_store();
        let svc = ObservabilityService::new(store);

        // Events without an explicit session_id route to the global
        // (per-tenant) session — that path is admin-only now, so
        // exercise the test surface under the admin AuthResult.
        let ids = svc
            .register_events(admin_request(Events {
                events: vec![ev("a", "first"), ev("b", "second"), ev("c", "third")],
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(ids.ids, vec!["a", "b", "c"]);

        svc.register_dependencies(admin_request(Dependencies {
            edges: vec![
                Edge {
                    source: "a".into(),
                    destination: "b".into(),
                    message_index: None,
                    proximal: None,
                    principal: None,
                    entity: None,
                },
                Edge {
                    source: "b".into(),
                    destination: "c".into(),
                    message_index: None,
                    proximal: None,
                    principal: None,
                    entity: None,
                },
            ],
            session_id: None,
        }))
        .await
        .unwrap();

        let graph = svc
            .backward_slice(admin_request(SliceRequest {
                event_id: Some("c".into()),
                max_depth: None,
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(graph.nodes.len(), 3);

        let graph = svc
            .forward_slice(admin_request(SliceRequest {
                event_id: Some("a".into()),
                max_depth: None,
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(graph.nodes.len(), 3);
    }

    #[tokio::test]
    async fn register_computations_and_trace() {
        let store = make_store();
        let svc = ObservabilityService::new(store);

        let comp = |sid: &str| Computation {
            span_id: sid.to_string(),
            trace_id: "t1".to_string(),
            parent_span_id: None,
            name: "op".to_string(),
            start_time_ns: 0,
            end_time_ns: 100,
            duration_ns: 100,
            status_code: 1,
            status_message: None,
            attributes_json: "{}".into(),
            events_json: "[]".into(),
            service_name: None,
            service_version: None,
            input_message_ids: vec![],
            output_message_id: None,
            linked_span_ids: vec![],
            principal: None,
            entity: None,
        };

        svc.register_computations(admin_request(Computations {
            computations: vec![comp("s1"), comp("s2")],
            session_id: None,
        }))
        .await
        .unwrap();

        let trace = svc
            .get_trace(admin_request(TraceRequest {
                trace_id: "t1".into(),
                start_time_ns: None,
                end_time_ns: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(trace.computations.len(), 2);

        let span = svc
            .get_span(admin_request(SpanRequest {
                span_id: "s1".into(),
                include_children: None,
                include_messages: None,
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(span.name, "op");
    }

    #[tokio::test]
    async fn register_events_with_dependencies() {
        use sasy_common::observability::EventsWithDependencies;
        let store = make_store();
        let svc = ObservabilityService::new(store);

        let ids = svc
            .register_events_with_dependencies(admin_request(EventsWithDependencies {
                events: vec![ev("a", "first"), ev("b", "second"), ev("c", "third")],
                edges: vec![
                    Edge {
                        source: "a".into(),
                        destination: "b".into(),
                        message_index: None,
                        proximal: None,
                        principal: None,
                        entity: None,
                    },
                    Edge {
                        source: "b".into(),
                        destination: "c".into(),
                        message_index: None,
                        proximal: None,
                        principal: None,
                        entity: None,
                    },
                ],
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(ids.ids, vec!["a", "b", "c"]);

        let graph = svc
            .backward_slice(admin_request(SliceRequest {
                event_id: Some("c".into()),
                max_depth: None,
                session_id: None,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }

    #[tokio::test]
    async fn get_span_not_found() {
        let store = make_store();
        let svc = ObservabilityService::new(store);
        let err = svc
            .get_span(admin_request(SpanRequest {
                span_id: "nope".into(),
                include_children: None,
                include_messages: None,
                session_id: None,
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn missing_event_id() {
        let store = make_store();
        let svc = ObservabilityService::new(store);
        let err = svc
            .backward_slice(admin_request(SliceRequest {
                event_id: None,
                max_depth: None,
                session_id: None,
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// `principal` on stored events comes from auth, not the wire.
    /// A client that lies about its principal in the proto field is
    /// silently overridden by `request_principal`.
    #[tokio::test]
    async fn principal_is_server_stamped_not_wire_supplied() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Alice writes; she also tries to spoof "evil" on the wire.
        // Give her a non-empty session_id — non-admins can't write
        // to the per-tenant global session anymore.
        let mut alice_ev = ev("a1", "from alice");
        alice_ev.principal = Some("evil".into());
        let mut alice_req = Request::new(Events {
            events: vec![alice_ev],
            session_id: Some("s1".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(alice_req).await.unwrap();

        // Bob writes under a different tenant.
        let bob_ev = ev("b1", "from bob");
        let mut bob_req = Request::new(Events {
            events: vec![bob_ev],
            session_id: Some("s1".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("orgb"),
        );
        svc.register_events(bob_req).await.unwrap();

        // Walk the store directly and assert principals are
        // server-stamped, not wire-supplied.
        let (events, _, _) = store.get_full_state_for_tenant("acme").unwrap();
        let a1 = events
            .iter()
            .find(|e| e.id.as_deref() == Some("a1"))
            .unwrap();
        assert_eq!(
            a1.principal.as_deref(),
            Some("alice"),
            "alice's principal must be server-stamped from auth (not 'evil')",
        );

        let (events, _, _) = store.get_full_state_for_tenant("orgb").unwrap();
        let b1 = events
            .iter()
            .find(|e| e.id.as_deref() == Some("b1"))
            .unwrap();
        assert_eq!(b1.principal.as_deref(), Some("bob"));
    }

    /// `backward_slice` with a node from a different tenant returns
    /// NotFound regardless of role — cross-tenant existence must
    /// not be revealed.
    #[tokio::test]
    async fn backward_slice_rejects_cross_tenant_event_id() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Bob writes under orgb. Use a non-empty session id —
        // non-admins can't write to the per-tenant global session.
        let bob_ev = ev("b1", "secret");
        let mut bob_req = Request::new(Events {
            events: vec![bob_ev],
            session_id: Some("s1".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("orgb"),
        );
        svc.register_events(bob_req).await.unwrap();

        // Alice tries to backward_slice on b1 from acme.
        let mut alice_slice = Request::new(SliceRequest {
            event_id: Some("b1".into()),
            max_depth: None,
            session_id: None,
        });
        alice_slice.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.backward_slice(alice_slice).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    /// Non-admin write with `session_id = None` (or empty) targets
    /// the per-tenant global session — denied. Same caller with a
    /// non-empty session id is allowed.
    #[tokio::test]
    async fn non_admin_cannot_write_to_global_session() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        let mut req = Request::new(Events {
            events: vec![ev("a", "global write")],
            session_id: None,
        });
        req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.register_events(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::PermissionDenied);

        // Same caller, non-empty session: allowed.
        let ok_event = ev("a", "session write");
        let mut ok_req = Request::new(Events {
            events: vec![ok_event],
            session_id: Some("s1".into()),
        });
        ok_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(ok_req).await.unwrap();
    }

    /// Once alice claims `(acme, s-alice)`, bob writing to the same
    /// `(tenant, session)` is denied. Bob writing to *his own*
    /// session in the same tenant is allowed.
    #[tokio::test]
    async fn cross_principal_session_writes_are_denied() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Alice writes — claims s1 in acme.
        let alice_ev = ev("a", "first");
        let mut alice_req = Request::new(Events {
            events: vec![alice_ev],
            session_id: Some("s1".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(alice_req).await.unwrap();

        // Bob (also acme) tries to write to alice's session.
        let bob_ev = ev("b", "intrusion");
        let mut bob_req = Request::new(Events {
            events: vec![bob_ev],
            session_id: Some("s1".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.register_events(bob_req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::PermissionDenied);

        // Bob to his own session — allowed.
        let bob_own = ev("b2", "own");
        let mut bob_own_req = Request::new(Events {
            events: vec![bob_own],
            session_id: Some("s2".into()),
        });
        bob_own_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(bob_own_req).await.unwrap();
    }

    /// Admin role bypasses the global-session write gate AND
    /// per-session ownership.
    #[tokio::test]
    async fn admin_bypasses_global_and_ownership() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Alice claims acme/s1.
        let alice_ev = ev("a", "first");
        let mut alice_req = Request::new(Events {
            events: vec![alice_ev],
            session_id: Some("s1".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(alice_req).await.unwrap();

        // Admin in acme: bypasses both per-session ownership AND
        // the global-session write gate. The session is an
        // envelope field, so these are two separate writes.
        let admin_ev = ev("admin-1", "admin override");
        let mut admin_req = Request::new(Events {
            events: vec![admin_ev],
            session_id: Some("s1".into()),
        });
        admin_req.extensions_mut().insert(
            AuthResult::success(
                "ops",
                vec!["observability-writer".into(), "admin".into()],
                "test",
            )
            .with_tenant("acme"),
        );
        svc.register_events(admin_req).await.unwrap();

        let global_ev = ev("admin-2", "broadcast");
        let mut global_req = Request::new(Events {
            events: vec![global_ev],
            session_id: None,
        });
        global_req.extensions_mut().insert(
            AuthResult::success(
                "ops",
                vec!["observability-writer".into(), "admin".into()],
                "test",
            )
            .with_tenant("acme"),
        );
        svc.register_events(global_req).await.unwrap();
    }

    /// Two same-tenant principals reuse the same node id across
    /// distinct sessions. The reverse index keys on `(tenant, id)`
    /// so the later writer wins routing; without the read-side
    /// ownership check, alice probing for bob's id would resolve
    /// to bob's scope and leak his slice. This asserts the
    /// not-found denial — same as the cross-tenant guard above it
    /// in the handler, indistinguishable from a genuinely missing
    /// id.
    #[tokio::test]
    async fn slice_and_span_reject_cross_principal_same_tenant() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Alice writes "shared" to acme/s1.
        let mut alice_req = Request::new(Events {
            events: vec![ev("shared", "from alice")],
            session_id: Some("s1".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(alice_req).await.unwrap();

        // Bob writes "shared" to acme/s2 — reverse index now
        // points to s2 (last writer wins).
        let mut bob_req = Request::new(Events {
            events: vec![ev("shared", "from bob")],
            session_id: Some("s2".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_events(bob_req).await.unwrap();

        // Alice probes "shared" — the index resolves to bob's s2,
        // owned by bob. Read-side ownership check converts this
        // to NotFound rather than returning bob's slice.
        let mut alice_back = Request::new(SliceRequest {
            event_id: Some("shared".into()),
            max_depth: None,
            session_id: None,
        });
        alice_back.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.backward_slice(alice_back).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);

        let mut alice_fwd = Request::new(SliceRequest {
            event_id: Some("shared".into()),
            max_depth: None,
            session_id: None,
        });
        alice_fwd.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.forward_slice(alice_fwd).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);

        // Bob can still read his own — sanity check.
        let mut bob_back = Request::new(SliceRequest {
            event_id: Some("shared".into()),
            max_depth: None,
            session_id: None,
        });
        bob_back.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let g = svc.backward_slice(bob_back).await.unwrap().into_inner();
        assert_eq!(g.nodes.len(), 1);

        // Admin in acme sees bob's slice via alice-ish probe.
        let g = svc
            .backward_slice(admin_request_for_tenant(
                SliceRequest {
                    event_id: Some("shared".into()),
                    max_depth: None,
                    session_id: None,
                },
                "acme",
            ))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(g.nodes.len(), 1);
    }

    /// `get_span` honours the same ownership rule as the slice
    /// handlers.
    #[tokio::test]
    async fn get_span_rejects_cross_principal_same_tenant() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        // Alice records span "s-shared" in acme/sess-a.
        let comp = |sid: &str, tid: &str| Computation {
            span_id: sid.to_string(),
            trace_id: tid.to_string(),
            parent_span_id: None,
            name: "op".to_string(),
            start_time_ns: 0,
            end_time_ns: 100,
            duration_ns: 100,
            status_code: 1,
            status_message: None,
            attributes_json: "{}".into(),
            events_json: "[]".into(),
            service_name: None,
            service_version: None,
            input_message_ids: vec![],
            output_message_id: None,
            linked_span_ids: vec![],
            principal: None,
            entity: None,
        };

        let mut alice_req = Request::new(Computations {
            computations: vec![comp("s-shared", "t-alice")],
            session_id: Some("sess-a".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_computations(alice_req).await.unwrap();

        // Bob writes a *different* span id but in his own session
        // — wouldn't trigger the issue. To exercise the read
        // guard, bob re-uses "s-shared" in his own session.
        let mut bob_req = Request::new(Computations {
            computations: vec![comp("s-shared", "t-bob")],
            session_id: Some("sess-b".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_computations(bob_req).await.unwrap();

        // Alice probes get_span — resolves to bob's scope, denied.
        let mut alice_get = Request::new(SpanRequest {
            span_id: "s-shared".into(),
            include_children: None,
            include_messages: None,
            session_id: None,
        });
        alice_get.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let err = svc.get_span(alice_get).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    /// `get_trace` filters spans by per-scope ownership: a trace
    /// that spans two principals' sessions returns only the
    /// caller's spans. Admin sees everything.
    #[tokio::test]
    async fn get_trace_filters_by_session_ownership() {
        use sasy_auth::AuthResult;

        let store = make_store();
        let svc = ObservabilityService::new(Arc::clone(&store));

        let comp = |sid: &str| Computation {
            span_id: sid.to_string(),
            trace_id: "T".to_string(),
            parent_span_id: None,
            name: "op".to_string(),
            start_time_ns: 0,
            end_time_ns: 100,
            duration_ns: 100,
            status_code: 1,
            status_message: None,
            attributes_json: "{}".into(),
            events_json: "[]".into(),
            service_name: None,
            service_version: None,
            input_message_ids: vec![],
            output_message_id: None,
            linked_span_ids: vec![],
            principal: None,
            entity: None,
        };

        // Alice and bob both contribute spans under trace T.
        let mut alice_req = Request::new(Computations {
            computations: vec![comp("a1"), comp("a2")],
            session_id: Some("sa".into()),
        });
        alice_req.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_computations(alice_req).await.unwrap();

        let mut bob_req = Request::new(Computations {
            computations: vec![comp("b1")],
            session_id: Some("sb".into()),
        });
        bob_req.extensions_mut().insert(
            AuthResult::success("bob", vec!["observability-writer".into()], "test")
                .with_tenant("acme"),
        );
        svc.register_computations(bob_req).await.unwrap();

        // Alice sees only her two spans.
        let mut alice_trace = Request::new(TraceRequest {
            trace_id: "T".into(),
            start_time_ns: None,
            end_time_ns: None,
        });
        alice_trace.extensions_mut().insert(
            AuthResult::success("alice", vec!["observability-reader".into()], "test")
                .with_tenant("acme"),
        );
        let trace = svc.get_trace(alice_trace).await.unwrap().into_inner();
        let mut sids: Vec<_> = trace
            .computations
            .iter()
            .map(|c| c.span_id.clone())
            .collect();
        sids.sort();
        assert_eq!(sids, vec!["a1", "a2"]);

        // Admin sees all three.
        let trace = svc
            .get_trace(admin_request_for_tenant(
                TraceRequest {
                    trace_id: "T".into(),
                    start_time_ns: None,
                    end_time_ns: None,
                },
                "acme",
            ))
            .await
            .unwrap()
            .into_inner();
        let mut sids: Vec<_> = trace
            .computations
            .iter()
            .map(|c| c.span_id.clone())
            .collect();
        sids.sort();
        assert_eq!(sids, vec!["a1", "a2", "b1"]);
    }

    /// Helper: admin auth pinned to a specific tenant. The default
    /// `admin_request` uses tenant "default"; cross-principal
    /// tests need acme.
    fn admin_request_for_tenant<T>(value: T, tenant: &str) -> Request<T> {
        let mut req = Request::new(value);
        req.extensions_mut().insert(
            sasy_auth::AuthResult::success(
                "ops",
                vec![
                    "admin".into(),
                    "observability-writer".into(),
                    "observability-reader".into(),
                ],
                "test",
            )
            .with_tenant(tenant),
        );
        req
    }
    fn event_snapshot(id: &str, text: &str) -> sasy_common::observability::EventSnapshot {
        sasy_common::observability::EventSnapshot {
            event: Some(ev(id, text)),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn compact_rpc_is_explicit_atomic_and_quiet() {
        use sasy_common::observability::EventSnapshot;
        let store = make_store();
        let service = ObservabilityService::new(Arc::clone(&store));
        let original = EventSnapshot {
            event: Some(Event {
                id: Some("origin".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = EventSnapshots {
            snapshots: vec![original.clone()],
            session_id: Some("compact".into()),
        };
        let id = service
            .resolve_snapshots(writer_request(request, "owner"))
            .await
            .unwrap()
            .into_inner()
            .ids[0]
            .clone();
        let hex = "7caf6b69c70cfa1e7aa15a8720643b70a7199a0b96ee26eb938dac1dc0ba3116";
        let hash = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let reference = EventSnapshot {
            base_id: Some(id.clone()),
            reuse_dependencies: true,
            content_hash: Some(hash),
            ..original
        };
        let scope = SessionScope::new("default", "compact");
        let seq = store.session_sequence(&scope);
        let mut rx = store.subscribe();
        let request = EventSnapshots {
            snapshots: vec![reference.clone()],
            session_id: Some("compact".into()),
        };
        assert_eq!(
            service
                .resolve_snapshots(writer_request(request.clone(), "owner"))
                .await
                .unwrap()
                .into_inner()
                .ids,
            vec![id.clone()]
        );
        assert_eq!(store.session_sequence(&scope), seq);
        assert!(rx.try_recv().is_err());
        assert_eq!(
            service
                .resolve_events(writer_request(request.clone(), "owner"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            service
                .resolve_snapshots(writer_request(request, "other"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::PermissionDenied
        );
        let mut bad = reference.clone();
        bad.content_hash.as_mut().unwrap()[0] ^= 1;
        let invalid = EventSnapshots {
            snapshots: vec![event_snapshot("new", "must not land"), bad],
            session_id: Some("compact".into()),
        };
        assert_eq!(
            service
                .resolve_snapshots(writer_request(invalid, "owner"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(store.session_counts(&scope), (1, 0));
        assert_eq!(store.session_sequence(&scope), seq);
        assert!(rx.try_recv().is_err());
        for legacy in [false, true] {
            let fresh = if legacy {
                "old-endpoint"
            } else {
                "missing-scope"
            };
            let request = writer_request(
                EventSnapshots {
                    snapshots: vec![reference.clone()],
                    session_id: Some(fresh.into()),
                },
                "owner",
            );
            let error = if legacy {
                service.resolve_events(request).await
            } else {
                service.resolve_snapshots(request).await
            }
            .unwrap_err();
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert!(store
                .get_session_owner(&SessionScope::new("default", fresh))
                .unwrap()
                .is_none());
        }
        let mut output = event_snapshot("output", "answer");
        output.dependencies.push(edge("origin", "output"));
        let ids = service
            .resolve_snapshots(writer_request(
                EventSnapshots {
                    snapshots: vec![output, reference],
                    session_id: Some("compact".into()),
                },
                "owner",
            ))
            .await
            .unwrap()
            .into_inner()
            .ids;
        assert_eq!(ids[1], id);
        assert_eq!(store.session_counts(&scope), (2, 1));
    }

    #[tokio::test]
    async fn resolve_snapshots_auth_stamps_and_empty_replay_is_quiet() {
        let store = make_store();
        let service = ObservabilityService::new(Arc::clone(&store));
        let mut snapshot = event_snapshot("origin", "value");
        snapshot.event.as_mut().unwrap().principal = Some("forged".into());
        let request = EventSnapshots {
            snapshots: vec![snapshot.clone()],
            session_id: Some("immutable".into()),
        };
        let first = service
            .resolve_events(writer_request(request.clone(), "owner"))
            .await
            .unwrap()
            .into_inner()
            .ids;
        let scope = SessionScope::new("default", "immutable");
        let seq = store.session_sequence(&scope);
        let mut rx = store.subscribe();
        let again = service
            .resolve_events(writer_request(request, "owner"))
            .await
            .unwrap()
            .into_inner()
            .ids;
        assert_eq!(again, first);
        assert_eq!(store.session_sequence(&scope), seq);
        assert!(rx.try_recv().is_err());
        let (nodes, _, _) = store.get_full_state().unwrap();
        assert_eq!(nodes[0].principal.as_deref(), Some("owner"));
        let stolen = EventSnapshots {
            snapshots: vec![snapshot],
            session_id: Some("immutable".into()),
        };
        assert_eq!(
            service
                .resolve_events(writer_request(stolen, "intruder"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::PermissionDenied
        );
        service
            .resolve_events(writer_request(
                EventSnapshots {
                    snapshots: vec![],
                    session_id: Some("empty".into()),
                },
                "intruder",
            ))
            .await
            .unwrap();
        assert!(store
            .get_session_owner(&SessionScope::new("default", "empty"))
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn invalid_snapshot_batch_does_not_claim_or_publish_partial_state() {
        let store = make_store();
        let service = ObservabilityService::new(Arc::clone(&store));
        let mut bad = event_snapshot("bad", "bad");
        bad.dependencies.push(edge("missing", "bad"));
        let request = EventSnapshots {
            snapshots: vec![event_snapshot("good", "good"), bad],
            session_id: Some("failed".into()),
        };
        let mut rx = store.subscribe();
        assert_eq!(
            service
                .resolve_events(writer_request(request, "writer"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        let scope = SessionScope::new("default", "failed");
        assert!(store.get_session_owner(&scope).unwrap().is_none());
        assert_eq!(store.session_counts(&scope), (0, 0));
        assert!(rx.try_recv().is_err());
        let mut bad_base = event_snapshot("origin", "value");
        bad_base.base_id = Some("sasy:mv1:missing".into());
        let invalid = EventSnapshots {
            snapshots: vec![bad_base],
            session_id: Some("base-failed".into()),
        };
        assert_eq!(
            service
                .resolve_events(writer_request(invalid, "writer"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert!(store
            .get_session_owner(&SessionScope::new("default", "base-failed"))
            .unwrap()
            .is_none());
        let request = EventSnapshots {
            snapshots: vec![event_snapshot("good", "good")],
            session_id: Some("failed".into()),
        };
        service
            .resolve_events(writer_request(request, "rightful"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn legacy_mutation_of_resolved_snapshot_is_failed_precondition() {
        let store = make_store();
        let service = ObservabilityService::new(Arc::clone(&store));
        let request = EventSnapshots {
            snapshots: vec![event_snapshot("origin", "initial")],
            session_id: Some("versioned".into()),
        };
        let id = service
            .resolve_events(writer_request(request, "writer"))
            .await
            .unwrap()
            .into_inner()
            .ids
            .remove(0);
        let legacy = Events {
            events: vec![ev("ordinary", "must not land"), ev(&id, "mutated")],
            session_id: Some("versioned".into()),
        };
        assert_eq!(
            service
                .register_events(writer_request(legacy, "writer"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
        assert_eq!(
            store.session_counts(&SessionScope::new("default", "versioned")),
            (1, 0)
        );
    }

    #[tokio::test]
    async fn snapshot_transport_validation_precedes_ownership() {
        let store = make_store();
        let service = ObservabilityService::new(Arc::clone(&store));
        let request = EventSnapshots {
            snapshots: vec![event_snapshot("origin", "bad\0text")],
            session_id: Some("transport".into()),
        };
        assert_eq!(
            service
                .resolve_events(writer_request(request, "writer"))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert!(store
            .get_session_owner(&SessionScope::new("default", "transport"))
            .unwrap()
            .is_none());
    }
}
