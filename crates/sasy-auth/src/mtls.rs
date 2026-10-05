//! mTLS authentication provider.
//!
//! Extracts entity identity from the client certificate's
//! Common Name (CN) and looks up roles from the auth config.

use crate::config::AuthConfig;
use crate::error::AuthError;
use crate::provider::{AuthProvider, AuthResult};

/// mTLS authentication provider.
///
/// In tonic, the peer certificate CN is typically passed
/// via a custom metadata header (e.g. `x-client-cn`) set
/// by a TLS termination layer or extracted from the TLS
/// connection info. This provider reads from that header.
///
/// **Security note**: The provider consumes a trusted internal header.
/// SASY services MUST wrap it with `AuthInterceptor`, which removes inbound
/// identity headers and replaces them with the verified TLS peer CN.
/// Calling this provider directly on untrusted metadata allows identity
/// spoofing. Loadable from the JSON auth provider config via
/// `"type": "mtls"` (see `load_provider_from_file`).
pub struct MtlsAuthProvider {
    auth_config: AuthConfig,
    /// Metadata key containing the client CN
    /// (default: `x-client-cn`).
    cn_header: String,
}

impl MtlsAuthProvider {
    pub fn new(auth_config: AuthConfig) -> Self {
        Self {
            auth_config,
            cn_header: sasy_common::headers::CLIENT_CN.to_string(),
        }
    }

    pub fn with_cn_header(mut self, header: impl Into<String>) -> Self {
        self.cn_header = header.into();
        self
    }
}

impl AuthProvider for MtlsAuthProvider {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        let cn = metadata
            .get(&self.cn_header)
            .ok_or_else(|| AuthError::Unauthenticated("no client certificate CN found".into()))?
            .to_str()
            .map_err(|_| AuthError::InvalidToken("non-ASCII CN value".into()))?;

        let entity = cn.to_string();
        let roles = self.auth_config.get_roles(&entity);
        let tenant = self.auth_config.get_tenant(&entity);

        Ok(AuthResult::success(entity, roles, "mtls").with_tenant(tenant))
    }

    fn name(&self) -> &str {
        "mtls"
    }

    fn tls_identity_headers(&self) -> Vec<String> {
        vec![self.cn_header.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EntityConfig;

    /// The provider must declare whatever header it reads, so the interceptor
    /// can strip it. Reporting only the default left a configured custom name
    /// unstripped, i.e. read straight from client metadata — an identity the
    /// caller chose, `admin` included.
    #[test]
    fn a_custom_cn_header_is_declared_for_stripping() {
        let default = MtlsAuthProvider::new(test_config());
        assert_eq!(
            default.tls_identity_headers(),
            vec![sasy_common::headers::CLIENT_CN.to_string()]
        );

        let custom = MtlsAuthProvider::new(test_config()).with_cn_header("x-forwarded-client-cn");
        assert_eq!(
            custom.tls_identity_headers(),
            vec!["x-forwarded-client-cn".to_string()],
            "a configured header must be declared, or nothing strips it"
        );
    }

    fn test_config() -> AuthConfig {
        let mut cfg = AuthConfig::default();
        cfg.entities.insert(
            "policy-engine".into(),
            EntityConfig {
                description: Some("Policy engine service".into()),
                roles: vec!["observability-reader".into()],
                tenant: None,
            },
        );
        cfg
    }

    #[test]
    fn test_mtls_valid_cn() {
        let provider = MtlsAuthProvider::new(test_config());
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-client-cn", "policy-engine".parse().unwrap());

        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("policy-engine"));
        assert_eq!(result.roles, vec!["observability-reader"]);
        assert_eq!(result.auth_method, "mtls");
    }

    #[test]
    fn test_mtls_unknown_entity() {
        let provider = MtlsAuthProvider::new(test_config());
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-client-cn", "unknown-service".parse().unwrap());

        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("unknown-service"));
        assert!(result.roles.is_empty());
    }

    #[test]
    fn test_mtls_missing_header() {
        let provider = MtlsAuthProvider::new(test_config());
        let meta = tonic::metadata::MetadataMap::new();
        assert!(provider.authenticate(&meta).is_err());
    }
}
