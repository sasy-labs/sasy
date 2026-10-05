//! PolicyChecker trait for authorization decisions.
//!
//! This trait abstracts over in-process and remote policy
//! engine implementations, so the reference monitor can
//! work with either.

use std::future::Future;

use sasy_common::policy_engine::{Action, AuthorizationResponse};
use sasy_common::SessionScope;

use crate::error::RefmonError;

/// Trait for checking authorization via the policy engine.
///
/// Implementors may call a remote gRPC service or execute
/// the policy engine in-process.
///
/// `scope` is the `(tenant, session)` partition. Refmon callers
/// derive the tenant from their own auth context and combine
/// with the user-supplied `session_id` to build the scope. Remote
/// gRPC-backed checkers ignore the tenant field — the destination
/// engine's auth interceptor derives it server-side from the
/// connection's principal — but in-process checkers use it directly.
///
/// `policy_id` pins the session→policy binding when set;
/// `None` (or empty string) means the server falls back to the
/// caller's tenant default policy.
///
/// `entity` is the *user-supplied* per-request actor (preserved
/// verbatim from the wire), surfaced to policies as `Entity(id)`.
/// `principal` is the *auth-derived* identity stamped by the server
/// from the connection's mTLS subject / JWT sub / API key entity,
/// surfaced as `Principal(id)` and the basis for `HasPrincipal()` /
/// `HasRole(...)`. Authoritative authorization rules should key on
/// `principal`; free-form actor tracking on `entity`.
pub trait PolicyChecker: Send + Sync {
    #[allow(clippy::too_many_arguments)] // inherent: full authorization context (entity, principal, roles, scope, ...)
    fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: Vec<Action>,
        entity: Option<&str>,
        roles: &[String],
        scope: SessionScope,
        principal: Option<&str>,
        policy_id: Option<String>,
    ) -> impl Future<Output = Result<AuthorizationResponse, RefmonError>> + Send;
}

// ── gRPC implementation (distributed mode) ────────────────────────

use parking_lot::Mutex;
use sasy_common::policy_engine::{policy_engine_client::PolicyEngineClient, AuthorizationRequest};
use tonic::transport::{Channel, ClientTlsConfig};

/// True if the gRPC `endpoint` targets a loopback host — a same-host
/// deployment where an unauthenticated engine channel is acceptable.
/// Remote endpoints require client credentials (see [`GrpcPolicyChecker`]):
/// without an authenticated identity the engine can't bind the refmon
/// to `service-proxy`, so its per-end-user tenant/principal delegation
/// is either rejected or — on a misconfigured trust-all engine —
/// blindly accepted, letting any reachable client assert arbitrary
/// tenants.
pub fn endpoint_is_loopback(endpoint: &str) -> bool {
    let rest = endpoint.split("://").nth(1).unwrap_or(endpoint);
    let authority = rest.split('/').next().unwrap_or(rest);

    // Userinfo is refused rather than parsed past. `127.0.0.1:10089@evil.com`
    // has an authority whose host is `evil.com`, but reading the host as
    // "everything before the last colon" answers `127.0.0.1` — so a remote
    // endpoint passed as loopback and the caller's guard let an
    // unauthenticated connection out to it. Nothing legitimate puts
    // credentials in a gRPC endpoint, so the safe reading of any `@` here is
    // "not loopback".
    if authority.contains('@') {
        return false;
    }

    let host = if let Some(stripped) = authority.strip_prefix('[') {
        // IPv6 literal: [::1]:port
        match stripped.split_once(']') {
            Some((inside, _)) => inside,
            // An unterminated bracket is malformed; do not guess.
            None => return false,
        }
    } else {
        match authority.rsplit_once(':') {
            Some((h, _)) => h,
            None => authority,
        }
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// PolicyChecker backed by a remote gRPC PolicyEngine service.
pub struct GrpcPolicyChecker {
    client: Mutex<Option<PolicyEngineClient<Channel>>>,
    endpoint: String,
    /// Client TLS for the engine connection (server-auth CA and/or an
    /// mTLS client identity). A client identity is what lets the
    /// engine recognize this refmon as `service-proxy` and honor the
    /// tenant/principal delegation in [`Self::check_authorization`].
    tls: Option<ClientTlsConfig>,
    /// API key presented to the engine as `x-api-key` on every call —
    /// an alternative to mTLS for establishing the refmon's
    /// service-proxy identity at the engine.
    api_key: Option<String>,
}

impl GrpcPolicyChecker {
    /// Unauthenticated checker. Only safe when the engine is reached
    /// over a trusted loopback / same-host channel; prefer
    /// [`Self::with_auth`] for split deployments.
    pub fn new(endpoint: &str) -> Self {
        Self {
            client: Mutex::new(None),
            endpoint: endpoint.to_string(),
            tls: None,
            api_key: None,
        }
    }

    /// Checker that authenticates to the engine via client TLS and/or
    /// an API key, so the engine can bind this refmon to a
    /// `service-proxy` identity and honor its delegation metadata.
    pub fn with_auth(
        endpoint: &str,
        tls: Option<ClientTlsConfig>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: Mutex::new(None),
            endpoint: endpoint.to_string(),
            tls,
            api_key,
        }
    }

    async fn get_client(&self) -> Result<PolicyEngineClient<Channel>, RefmonError> {
        {
            let guard = self.client.lock();
            if let Some(c) = guard.as_ref() {
                return Ok(c.clone());
            }
        }
        let mut builder = Channel::from_shared(self.endpoint.clone())
            .map_err(|e| RefmonError::PolicyCheck(e.to_string()))?;
        if let Some(tls) = &self.tls {
            builder = builder
                .tls_config(tls.clone())
                .map_err(|e| RefmonError::PolicyCheck(format!("engine client TLS: {}", e)))?;
        }
        let channel = builder
            .connect()
            .await
            .map_err(|e| RefmonError::PolicyCheck(format!("connect to policy engine: {}", e)))?;
        let client = PolicyEngineClient::new(channel);
        *self.client.lock() = Some(client.clone());
        Ok(client)
    }
}

impl PolicyChecker for GrpcPolicyChecker {
    async fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: Vec<Action>,
        entity: Option<&str>,
        roles: &[String],
        scope: SessionScope,
        principal: Option<&str>,
        policy_id: Option<String>,
    ) -> Result<AuthorizationResponse, RefmonError> {
        // Forward the resolved (end-user) tenant + principal as
        // delegation metadata. The engine's auth helpers
        // (`request_effective_tenant` / `request_effective_principal`)
        // accept these only when the caller's connection has the
        // `service-proxy` role — the reference monitor is the
        // trusted relay in the API-gateway pattern. Without delegation
        // every per-end-user check would land in the engine as
        // "from refmon", collapsing multi-tenant deployments to
        // the refmon's own tenant.
        let session = scope.session();
        let tenant = scope.tenant().to_string();
        let mut client = self.get_client().await?;
        let _ = policy_id; // sessions bound via SetPolicy now

        let mut request = tonic::Request::new(AuthorizationRequest {
            current_node_ids: current_node_ids.to_vec(),
            actions,
            entity: entity.map(|s| s.to_string()),
            roles: roles.to_vec(),
            session_id: if session.is_empty() {
                None
            } else {
                Some(session.to_string())
            },
            // Wire `principal` field stays for backward compat
            // but is ignored by engines that use the effective-
            // identity helpers; metadata `x-principal` is the
            // authoritative channel.
            principal: principal.map(|s| s.to_string()),
        });
        let md = request.metadata_mut();
        // Authenticate this refmon to the engine so its delegation is
        // honored (mTLS client identity is applied on the channel in
        // get_client; the API key, if configured, rides per-request).
        if let Some(key) = &self.api_key {
            if let Ok(v) = key.parse() {
                md.insert(sasy_common::headers::API_KEY, v);
            }
        }
        if let Ok(t) = tenant.parse() {
            md.insert(sasy_common::headers::TENANT, t);
        }
        if let Some(p) = principal {
            if let Ok(p) = p.parse() {
                md.insert(sasy_common::headers::PRINCIPAL, p);
            }
        }

        let resp = client
            .check_authorization(request)
            .await
            .map_err(|e| RefmonError::PolicyCheck(format!("policy engine call: {}", e)))?;
        Ok(resp.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::endpoint_is_loopback;

    /// An endpoint carrying userinfo must not read as loopback.
    ///
    /// The authority of `http://127.0.0.1:50053@evil.com/` has host
    /// `evil.com`; taking "everything before the last colon" answered
    /// `127.0.0.1`. The caller uses this to decide whether it may connect
    /// UNAUTHENTICATED, so a false positive means trusting a remote policy
    /// engine with no client credentials — and believing whatever it says
    /// about authorization.
    #[test]
    fn userinfo_endpoints_are_not_loopback() {
        for ep in [
            "http://127.0.0.1:50053@evil.com/",
            "http://127.0.0.1@evil.com",
            "127.0.0.1:50053@evil.com",
            "http://localhost@evil.com:50053",
            "http://[::1]:50053@evil.com/",
        ] {
            assert!(
                !endpoint_is_loopback(ep),
                "{ep} carries userinfo and must not be treated as loopback"
            );
        }
    }

    /// A malformed IPv6 authority is not guessed at either.
    #[test]
    fn unterminated_ipv6_bracket_is_not_loopback() {
        assert!(!endpoint_is_loopback("http://[::1:50053"));
    }

    #[test]
    fn loopback_endpoints_are_recognized() {
        for ep in [
            "http://localhost:50053",
            "https://localhost:50053",
            "http://127.0.0.1:50053",
            "127.0.0.1:50053",
            "http://[::1]:50053",
            "http://127.0.0.5/path",
            "localhost",
        ] {
            assert!(endpoint_is_loopback(ep), "{ep} should be loopback");
        }
    }

    #[test]
    fn remote_endpoints_are_not_loopback() {
        for ep in [
            "http://engine.internal:50053",
            "https://10.0.0.4:50053",
            "http://engine.example.com:443",
            "http://192.168.1.10:50053",
            "[2001:db8::1]:50053",
        ] {
            assert!(!endpoint_is_loopback(ep), "{ep} should NOT be loopback");
        }
    }
}
