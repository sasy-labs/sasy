//! Passthrough (no-auth) provider.

use crate::config::AuthConfig;
use crate::error::AuthError;
use crate::provider::{AuthProvider, AuthResult};

/// Passthrough authentication — always succeeds.
///
/// Optionally extracts entity from `x-entity` metadata. Used when
/// authentication is disabled (the explicit `SASY_ALLOW_NO_AUTH` local-dev
/// opt-in). `default_roles` is granted when the caller's entity resolves to no
/// roles (unknown/absent entity) — so a co-located single-user deployment (the
/// locally spawned engine) "just works" with NO auth_config / entity /
/// certs. The trust boundary is loopback + the same OS user.
pub struct PassthroughAuthProvider {
    auth_config: Option<AuthConfig>,
    entity_header: String,
    default_roles: Vec<String>,
}

impl PassthroughAuthProvider {
    pub fn new(auth_config: Option<AuthConfig>) -> Self {
        Self::with_default_roles(auth_config, Vec::new())
    }

    /// Passthrough with a fallback role set for entities that map to nothing
    /// (or no entity at all) — the friction-free local-trust default.
    pub fn with_default_roles(auth_config: Option<AuthConfig>, default_roles: Vec<String>) -> Self {
        Self {
            auth_config,
            entity_header: sasy_common::headers::ENTITY.to_string(),
            default_roles,
        }
    }
}

impl AuthProvider for PassthroughAuthProvider {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        let entity = metadata
            .get(&self.entity_header)
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        let (mut roles, tenant) = match (&entity, &self.auth_config) {
            (Some(e), Some(cfg)) => (cfg.get_roles(e), Some(cfg.get_tenant(e))),
            _ => (vec![], None),
        };
        // Entity unknown/absent → fall back to the local-trust default role set
        // (empty unless configured), so a stale/missing entity doesn't strand
        // the local client with no roles.
        if roles.is_empty() {
            roles = self.default_roles.clone();
        }

        Ok(AuthResult {
            entity,
            roles,
            tenant,
            auth_method: "passthrough".to_string(),
        })
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_no_entity() {
        let provider = PassthroughAuthProvider::new(None);
        let meta = tonic::metadata::MetadataMap::new();
        let result = provider.authenticate(&meta).unwrap();
        assert!(result.entity.is_none());
        assert!(result.roles.is_empty());
        assert_eq!(result.auth_method, "passthrough");
    }

    #[test]
    fn test_passthrough_with_entity() {
        let provider = PassthroughAuthProvider::new(None);
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-entity", "test-user".parse().unwrap());
        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("test-user"));
    }

    #[test]
    fn test_passthrough_default_roles_fallback() {
        // An entity absent from auth_config
        // falls back to the local-trust default role set instead of stranding
        // the caller with no roles.
        let provider = PassthroughAuthProvider::with_default_roles(
            None,
            vec!["reference-monitor-user".to_string()],
        );
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-entity", "unknown-agent".parse().unwrap());
        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.roles, vec!["reference-monitor-user".to_string()]);
        // And with no entity at all.
        let none = provider
            .authenticate(&tonic::metadata::MetadataMap::new())
            .unwrap();
        assert_eq!(none.roles, vec!["reference-monitor-user".to_string()]);
    }
}
