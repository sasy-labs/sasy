//! AuthProvider trait and AuthResult type.

use crate::error::AuthError;

/// Result of an authentication attempt.
///
/// `tenant` is the isolation bucket the authenticated entity belongs
/// to (`acme`, `globex`, `default`, ...). It is populated by the auth
/// provider from `auth_config.yaml` at authentication time and is the
/// sole source of truth server-side — clients never send tenant ids
/// over the wire. `None` means the request was anonymous
/// (no entity matched), in which case downstream code should fall
/// back to a deployment-level default tenant.
#[derive(Debug, Clone)]
pub struct AuthResult {
    pub entity: Option<String>,
    pub roles: Vec<String>,
    pub tenant: Option<String>,
    pub auth_method: String,
}

impl AuthResult {
    /// Successful authentication for `entity` with `roles`. The tenant
    /// is left unset; providers populate it via [`Self::with_tenant`]
    /// once they look the entity up in the auth config.
    pub fn success(entity: impl Into<String>, roles: Vec<String>, method: &str) -> Self {
        Self {
            entity: Some(entity.into()),
            roles,
            tenant: None,
            auth_method: method.to_string(),
        }
    }

    pub fn anonymous(method: &str) -> Self {
        Self {
            entity: None,
            roles: vec![],
            tenant: None,
            auth_method: method.to_string(),
        }
    }

    /// Bind the authenticated identity to a tenant. Providers call this
    /// once they've looked the entity up in the auth config.
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    pub fn has_any_role(&self, roles: &[&str]) -> bool {
        roles.iter().any(|r| self.has_role(r))
    }
}

/// Trait for authentication providers.
///
/// Each provider examines gRPC request metadata and returns
/// an [`AuthResult`] on success or an [`AuthError`] on failure.
pub trait AuthProvider: Send + Sync {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError>;

    fn name(&self) -> &str;

    /// Metadata keys this provider reads a TLS-derived client identity from.
    ///
    /// The interceptor strips these on the way in and re-injects them from
    /// the verified peer certificate, so the value can never come from the
    /// wire. A provider that reads such a header under a name the interceptor
    /// does not know about would take it straight from client metadata —
    /// which is an identity, including `admin`, that the caller chose.
    ///
    /// Default: none, for providers that derive identity from a token or key
    /// rather than from the transport.
    fn tls_identity_headers(&self) -> Vec<String> {
        Vec::new()
    }
}
