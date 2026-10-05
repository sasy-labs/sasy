//! Tonic interceptor and RBAC helpers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tonic::service::Interceptor;

use crate::provider::{AuthProvider, AuthResult};

// ── Global RBAC toggle ─────────────────────────────────

static RBAC_ENABLED: AtomicBool = AtomicBool::new(true);

/// Disable RBAC checks globally.
///
/// Test-only, and deliberately unreachable from a shipped binary: this is a
/// process-global that turns every `require_role` in every service into a
/// pass, so a single stray call — a debug flag, an embedder's convenience
/// helper — disarms authorization for the whole process with nothing at the
/// call sites to show for it. Nothing outside tests calls it. Embedders whose
/// own test suites need it enable the `test-util` feature.
#[cfg(any(test, feature = "test-util"))]
pub fn disable_rbac() {
    RBAC_ENABLED.store(false, Ordering::SeqCst);
    tracing::warn!("RBAC disabled — authorization checks skipped");
}

/// Re-enable RBAC checks. Test-only, for the same reason as
/// [`disable_rbac`] — a test that disables RBAC restores it with this.
#[cfg(any(test, feature = "test-util"))]
pub fn enable_rbac() {
    RBAC_ENABLED.store(true, Ordering::SeqCst);
    tracing::info!("RBAC enabled");
}

/// Check whether RBAC is currently enabled.
pub fn is_rbac_enabled() -> bool {
    RBAC_ENABLED.load(Ordering::SeqCst)
}

// ── Role checking helpers ──────────────────────────────

/// Require a specific role, respecting the RBAC toggle.
#[allow(clippy::result_large_err)]
pub fn require_role(auth: &AuthResult, role: &str) -> Result<(), tonic::Status> {
    if !is_rbac_enabled() {
        return Ok(());
    }
    if auth.has_role(role) {
        Ok(())
    } else {
        Err(tonic::Status::permission_denied(format!(
            "role '{}' required, have: {:?}",
            role, auth.roles
        )))
    }
}

/// Require any of the given roles, respecting the RBAC
/// toggle.
#[allow(clippy::result_large_err)]
pub fn require_any_role(auth: &AuthResult, roles: &[&str]) -> Result<(), tonic::Status> {
    if !is_rbac_enabled() {
        return Ok(());
    }
    if auth.has_any_role(roles) {
        Ok(())
    } else {
        Err(tonic::Status::permission_denied(format!(
            "one of {:?} required, have: {:?}",
            roles, auth.roles
        )))
    }
}

// ── Tonic interceptor ──────────────────────────────────

/// Tonic interceptor that authenticates requests and
/// stores the [`AuthResult`] in request extensions.
#[derive(Clone)]
pub struct AuthInterceptor {
    provider: Arc<dyn AuthProvider>,
}

impl AuthInterceptor {
    pub fn new(provider: Arc<dyn AuthProvider>) -> Self {
        Self { provider }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        // A TLS-derived client identity comes ONLY from a verified mTLS peer
        // certificate, never from client input. Strip any inbound value
        // first, then set it from the cert below. Without this, a caller on a
        // non-mTLS listener (TLS without a client CA, or a plaintext internal
        // listener) could spoof an identity — including admin.
        //
        // The default key is stripped unconditionally, and so is any other
        // key the configured provider says it reads: the mTLS provider's
        // header name is configurable, and stripping only the hardcoded one
        // left a custom name taken straight from client metadata.
        request
            .metadata_mut()
            .remove(sasy_common::headers::CLIENT_CN);
        for header in self.provider.tls_identity_headers() {
            if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(header.as_bytes()) {
                request.metadata_mut().remove(key);
            }
        }

        // Extract CN from TLS peer certificate and inject as metadata
        // so the mTLS auth provider can read it. The restricted guard enables
        // this certificate-only feature too. Stripping above remains
        // unconditional so a non-mTLS caller can never spoof a CN.
        #[cfg(feature = "mtls")]
        if let Some(certs) = request.peer_certs() {
            if let Some(cn) = extract_cn_from_certs(&certs) {
                if let Ok(val) = cn.parse::<tonic::metadata::MetadataValue<_>>() {
                    request
                        .metadata_mut()
                        .insert(sasy_common::headers::CLIENT_CN, val.clone());
                    // Also under whatever key the provider actually reads, so
                    // a configured custom name still gets the CERT-derived CN
                    // rather than nothing (and, before the strip above, rather
                    // than whatever the client sent).
                    for header in self.provider.tls_identity_headers() {
                        if header.eq_ignore_ascii_case(sasy_common::headers::CLIENT_CN) {
                            continue;
                        }
                        if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(header.as_bytes())
                        {
                            request.metadata_mut().insert(key, val.clone());
                        }
                    }
                }
            }
        }

        let metadata = request.metadata();
        let result = self
            .provider
            .authenticate(metadata)
            .map_err(|e| -> tonic::Status { e.into() })?;

        request.extensions_mut().insert(result);
        Ok(request)
    }
}

/// Extract the Common Name (CN) from DER-encoded peer certificates.
#[cfg(feature = "mtls")]
fn extract_cn_from_certs(certs: &[rustls_pki_types::CertificateDer<'_>]) -> Option<String> {
    use x509_parser::prelude::*;
    let cert_der = certs.first()?;
    let (_, cert) = X509Certificate::from_der(cert_der.as_ref()).ok()?;
    let cn = cert
        .subject()
        .iter_common_name()
        .next()
        .and_then(|cn| cn.as_str().ok())
        .map(String::from);
    cn
}

/// Extract [`AuthResult`] from request extensions.
///
/// Use in service handlers after the [`AuthInterceptor`]
/// has run.
pub fn get_auth_result<T>(request: &tonic::Request<T>) -> Option<&AuthResult> {
    request.extensions().get::<AuthResult>()
}

/// Resolve the tenant for a request from the auth context, falling
/// back to `default_if_missing` (typically `"default"`) when the
/// request was anonymous or the auth provider didn't set a tenant.
/// This is the *only* way server code should obtain the
/// per-request tenant — wire `tenant_id` fields are ignored.
pub fn request_tenant<T>(request: &tonic::Request<T>, default_if_missing: &str) -> String {
    get_auth_result(request)
        .and_then(|a| a.tenant.clone())
        .unwrap_or_else(|| default_if_missing.to_string())
}

/// Auth-derived principal (entity from `AuthResult`). `None` if
/// the request was anonymous. Server code uses this to stamp
/// the immutable `principal` field on writes — never read the
/// wire-supplied value.
pub fn request_principal<T>(request: &tonic::Request<T>) -> Option<String> {
    get_auth_result(request).and_then(|a| a.entity.clone())
}

/// Check a required role on a request.
///
/// Extracts the auth result from extensions and calls [`require_role`].
///
/// A request carrying no [`AuthResult`] is REFUSED. [`AuthInterceptor`] stamps
/// one on every request it lets through — even the passthrough provider — so a
/// handler that sees none was reached without the interceptor in front of it:
/// a service wired up bare by an embedder, or a direct call. Passing silently
/// there is the wrong default, because it hands every role-gated RPC (set the
/// tenant default policy, read a credential) to a caller nobody authenticated,
/// and it fails open exactly in the deployment that got the wiring wrong.
#[allow(clippy::result_large_err)]
pub fn check_request_role<T>(request: &tonic::Request<T>, role: &str) -> Result<(), tonic::Status> {
    let Some(auth) = get_auth_result(request) else {
        return Err(tonic::Status::unauthenticated(
            "no authentication context on the request",
        ));
    };
    require_role(auth, role)
}

/// Check that a request carries at least one of `roles`.
///
/// Like [`check_request_role`], rejects a missing [`AuthResult`]. The role check
/// must fail closed even when a request has bypassed the auth interceptor.
#[allow(clippy::result_large_err)]
pub fn check_request_any_role<T>(
    request: &tonic::Request<T>,
    roles: &[&str],
) -> Result<(), tonic::Status> {
    let Some(auth) = get_auth_result(request) else {
        return Err(tonic::Status::unauthenticated(
            "no authentication context on the request",
        ));
    };
    require_any_role(auth, roles)
}

/// True iff the request's auth context carries `role`. Used for
/// admin-bypass branches where the caller wants to enforce
/// ownership *unless* the caller is admin. Returns `false` on
/// anonymous requests (no interceptor) — admin bypass has to be
/// explicit.
pub fn request_has_role<T>(request: &tonic::Request<T>, role: &str) -> bool {
    get_auth_result(request)
        .map(|a| a.has_role(role))
        .unwrap_or(false)
}

/// Effective tenant for a request, accounting for `service-proxy`
/// delegation. When the caller's auth context carries the
/// `service-proxy` role and the request metadata includes an
/// `x-tenant` header, that value is trusted as the end-user's
/// tenant. Otherwise falls back to [`request_tenant`].
///
/// This is what split-deployment server handlers should use
/// instead of `request_tenant` directly: in a refmon→engine
/// connection the engine sees the *refmon's* tenant from auth,
/// but the actual end-user's tenant lives in the metadata the
/// refmon attached after resolving its own delegation.
pub fn request_effective_tenant<T>(
    request: &tonic::Request<T>,
    default_if_missing: &str,
) -> String {
    if let Some(auth) = get_auth_result(request) {
        if auth.has_role(sasy_common::roles::SERVICE_PROXY) {
            if let Some(t) = request
                .metadata()
                .get(sasy_common::headers::TENANT)
                .and_then(|v| v.to_str().ok())
            {
                if !t.is_empty() {
                    return t.to_string();
                }
            }
        }
    }
    request_tenant(request, default_if_missing)
}

/// Effective role set for a request, accounting for `service-proxy`
/// delegation.
///
/// The role half of [`request_effective_tenant`] / [`request_effective_principal`],
/// and it exists for the same reason: once a trusted relay speaks for an end
/// user, "what may this caller do" has to be answered about the END USER, not
/// about the relay. A gateway that holds `admin` so it can administer the
/// engine would otherwise lend that admin to every user it fronts.
///
/// A relay is speaking for somebody else as soon as it forwards any of
/// `x-tenant` / `x-principal` / `x-roles`. The effective roles for such a
/// request are exactly the ones listed in `x-roles` (comma-separated, as the
/// reference monitor writes them) — an EMPTY set if it forwarded none, since
/// falling back to the relay's own roles there is precisely the confusion this
/// resolves. A request with no delegation headers is the relay acting for
/// itself, and resolves to its own authenticated roles.
///
/// Callers that ask about the CONNECTION itself — "is this a trusted relay?",
/// as the credential server's `is_relay` check does — still use
/// [`request_has_role`]. Callers that WAIVE a restriction for a privileged
/// caller ask both: the session-ownership bypass in the policy service, the
/// Default / Force scope gate and the functor-source gate each require
/// `request_has_role` and `request_effective_has_role` together, so an admin
/// relay cannot lend its admin to the user it forwards.
pub fn request_effective_roles<T>(request: &tonic::Request<T>) -> Vec<String> {
    let Some(auth) = get_auth_result(request) else {
        return Vec::new();
    };
    if auth.has_role(sasy_common::roles::SERVICE_PROXY) {
        let header = |key: &str| {
            request
                .metadata()
                .get(key)
                .and_then(|v| v.to_str().ok())
                .filter(|v| !v.is_empty())
        };
        let delegating = header(sasy_common::headers::TENANT).is_some()
            || header(sasy_common::headers::PRINCIPAL).is_some()
            || header(sasy_common::headers::ROLES).is_some();
        if delegating {
            return header(sasy_common::headers::ROLES)
                .map(|roles| {
                    roles
                        .split(',')
                        .map(|r| r.trim().to_string())
                        .filter(|r| !r.is_empty())
                        .collect()
                })
                .unwrap_or_default();
        }
    }
    auth.roles.clone()
}

/// True iff the request's *effective* roles (see [`request_effective_roles`])
/// include `role`.
pub fn request_effective_has_role<T>(request: &tonic::Request<T>, role: &str) -> bool {
    request_effective_roles(request).iter().any(|r| r == role)
}

/// Effective principal for a request, accounting for
/// `service-proxy` delegation. Same shape as
/// [`request_effective_tenant`]: when the caller has
/// `service-proxy`, prefer `x-principal` from metadata; else fall
/// back to the auth-derived entity.
pub fn request_effective_principal<T>(request: &tonic::Request<T>) -> Option<String> {
    if let Some(auth) = get_auth_result(request) {
        if auth.has_role(sasy_common::roles::SERVICE_PROXY) {
            if let Some(p) = request
                .metadata()
                .get(sasy_common::headers::PRINCIPAL)
                .and_then(|v| v.to_str().ok())
            {
                if !p.is_empty() {
                    return Some(p.to_string());
                }
            }
        }
    }
    request_principal(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_auth() -> AuthResult {
        AuthResult::success("alice", vec!["admin".into(), "user".into()], "test")
    }

    #[test]
    fn test_require_role_present() {
        let auth = sample_auth();
        assert!(require_role(&auth, "admin").is_ok());
    }

    #[test]
    fn test_require_role_missing() {
        let auth = sample_auth();
        assert!(require_role(&auth, "superadmin").is_err());
    }

    #[test]
    fn test_require_any_role() {
        let auth = sample_auth();
        assert!(require_any_role(&auth, &["superadmin", "admin"]).is_ok());
        assert!(require_any_role(&auth, &["x", "y"]).is_err());
    }

    #[test]
    fn test_rbac_toggle() {
        // Ensure clean state
        enable_rbac();
        assert!(is_rbac_enabled());

        let auth = AuthResult::anonymous("test");

        // With RBAC enabled, missing role fails
        assert!(require_role(&auth, "admin").is_err());

        // Disable RBAC
        disable_rbac();
        assert!(!is_rbac_enabled());
        assert!(require_role(&auth, "admin").is_ok());

        // Re-enable
        enable_rbac();
        assert!(is_rbac_enabled());
        assert!(require_role(&auth, "admin").is_err());
    }

    /// A request that never passed through the interceptor carries no
    /// `AuthResult`. The role check refuses it instead of waving it through —
    /// otherwise a service wired up without the interceptor serves its
    /// admin-only RPCs to anyone who can reach the socket.
    #[test]
    fn a_request_with_no_auth_context_is_refused() {
        let req = tonic::Request::new(());
        let err = check_request_role(&req, "admin")
            .expect_err("a request with no auth context must not pass a role check");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
        assert!(
            err.message().contains("no authentication context"),
            "the refusal should name the missing context, got: {}",
            err.message()
        );
    }

    /// The multi-role check refuses an unauthenticated request too: a handler
    /// whose floor is "any of these roles" must not become no floor at all
    /// when the interceptor is missing.
    #[test]
    fn a_request_with_no_auth_context_is_refused_by_the_any_role_check() {
        let req = tonic::Request::new(());
        let err = check_request_any_role(&req, &["admin", "reference-monitor-user"])
            .expect_err("a request with no auth context must not pass a role check");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
        assert!(
            err.message().contains("no authentication context"),
            "the refusal should name the missing context, got: {}",
            err.message()
        );
    }

    fn make_request_with_auth_and_meta(
        auth: AuthResult,
        meta: &[(&'static str, &str)],
    ) -> tonic::Request<()> {
        let mut req = tonic::Request::new(());
        req.extensions_mut().insert(auth);
        for (k, v) in meta {
            req.metadata_mut().insert(*k, v.parse().unwrap());
        }
        req
    }

    /// A relay's own roles are not the end user's. Once the relay forwards
    /// delegation headers, the effective role set is what it says the user
    /// has — nothing, if it says nothing — so a gateway that holds `admin`
    /// for its own administrative calls does not hand that admin to every
    /// user it fronts.
    #[test]
    fn a_relay_speaking_for_a_user_does_not_lend_that_user_its_own_roles() {
        let relay = || {
            AuthResult::success(
                "cloud-api",
                vec!["service-proxy".into(), "admin".into()],
                "test",
            )
            .with_tenant("default")
        };

        // Delegating, with no roles claimed for the user: the user has none.
        let req = make_request_with_auth_and_meta(relay(), &[("x-principal", "alice")]);
        assert!(request_effective_roles(&req).is_empty());
        assert!(!request_effective_has_role(&req, "admin"));

        // Delegating, with the user's roles: those, and only those.
        let req = make_request_with_auth_and_meta(
            relay(),
            &[
                ("x-principal", "alice"),
                ("x-roles", "reference-monitor-user, policy-client"),
            ],
        );
        assert_eq!(
            request_effective_roles(&req),
            vec![
                "reference-monitor-user".to_string(),
                "policy-client".to_string()
            ]
        );
        assert!(!request_effective_has_role(&req, "admin"));

        // Not delegating: the relay is acting for itself and keeps its roles.
        let req = make_request_with_auth_and_meta(relay(), &[]);
        assert!(request_effective_has_role(&req, "admin"));

        // No `service-proxy`: forwarded roles are ignored outright.
        let plain = AuthResult::success("agent", vec!["reference-monitor-user".into()], "test");
        let req = make_request_with_auth_and_meta(plain, &[("x-roles", "admin")]);
        assert!(!request_effective_has_role(&req, "admin"));
        assert!(request_effective_has_role(&req, "reference-monitor-user"));
    }

    /// Caller without `service-proxy`: x-tenant / x-principal are
    /// ignored; the effective values come from the auth context.
    /// Caller with `service-proxy`: the metadata wins.
    #[test]
    fn effective_identity_requires_service_proxy_role() {
        // Plain caller (developer-key, acme tenant). x-tenant /
        // x-principal in metadata must NOT shift the resolved
        // identity — that would let any peer impersonate any
        // tenant.
        let plain =
            AuthResult::success("developer", vec!["user".into()], "test").with_tenant("acme");
        let req = make_request_with_auth_and_meta(
            plain,
            &[("x-tenant", "evil-tenant"), ("x-principal", "ghost")],
        );
        assert_eq!(request_effective_tenant(&req, "default"), "acme");
        assert_eq!(
            request_effective_principal(&req).as_deref(),
            Some("developer")
        );

        // Trusted relay (`service-proxy`): metadata is honored.
        let proxy = AuthResult::success(
            "refmon",
            vec!["service-proxy".into(), "policy-client".into()],
            "test",
        )
        .with_tenant("default");
        let req = make_request_with_auth_and_meta(
            proxy,
            &[("x-tenant", "acme"), ("x-principal", "alice")],
        );
        assert_eq!(request_effective_tenant(&req, "default"), "acme");
        assert_eq!(request_effective_principal(&req).as_deref(), Some("alice"));

        // Trusted relay without metadata: falls back to its own
        // auth-derived identity.
        let proxy = AuthResult::success("refmon", vec!["service-proxy".into()], "test")
            .with_tenant("relay-tenant");
        let req = make_request_with_auth_and_meta(proxy, &[]);
        assert_eq!(request_effective_tenant(&req, "default"), "relay-tenant");
        assert_eq!(request_effective_principal(&req).as_deref(), Some("refmon"));
    }
}
