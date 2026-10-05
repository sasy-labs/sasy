//! Auth config loader — parses `auth_config.yaml` for
//! entity-to-role mappings.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::AuthError;

/// Per-entity configuration.
///
/// `tenant` is the isolation bucket the entity belongs to (e.g.
/// `acme`, `globex`, `default`). The server populates the
/// per-request tenant from this mapping at authentication time —
/// clients never send tenant ids on the wire. `roles` stay
/// scoped to the entity within its tenant.
#[derive(Debug, Clone, Deserialize)]
pub struct EntityConfig {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub tenant: Option<String>,
}

/// Top-level authorization config matching
/// `config/auth_config.yaml`.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    #[serde(default = "default_trust_domain")]
    pub trust_domain: Option<String>,
    /// Tenant assigned to entities that don't declare one explicitly.
    /// Defaults to `"default"`, so a deployment that never mentions a
    /// tenant has exactly one.
    #[serde(default = "default_tenant")]
    pub default_tenant: String,
    #[serde(default)]
    pub entities: HashMap<String, EntityConfig>,
}

fn default_trust_domain() -> Option<String> {
    Some("observability.local".to_string())
}

fn default_tenant() -> String {
    "default".to_string()
}

impl AuthConfig {
    /// Load config from a YAML file.
    pub fn load(path: &str) -> Result<Self, AuthError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AuthError::Config(format!("failed to read {path}: {e}")))?;
        let config: AuthConfig = serde_yaml::from_str(&content)
            .map_err(|e| AuthError::Config(format!("failed to parse {path}: {e}")))?;
        Ok(config)
    }

    /// Get roles for a given entity name.
    pub fn get_roles(&self, entity: &str) -> Vec<String> {
        self.entities
            .get(entity)
            .map(|e| e.roles.clone())
            .unwrap_or_default()
    }

    /// Tenant the entity is bound to. Falls back to `default_tenant`
    /// for entities that don't declare one (or unknown entities).
    pub fn get_tenant(&self, entity: &str) -> String {
        self.entities
            .get(entity)
            .and_then(|e| e.tenant.clone())
            .unwrap_or_else(|| self.default_tenant.clone())
    }

    /// Check whether an entity is defined.
    pub fn has_entity(&self, entity: &str) -> bool {
        self.entities.contains_key(entity)
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            trust_domain: default_trust_domain(),
            default_tenant: default_tenant(),
            entities: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_yaml() {
        let yaml = r#"
trust_domain: observability.local

entities:
  policy-engine:
    description: Policy engine
    roles:
      - observability-reader
  client:
    description: Standard client
    roles:
      - user
      - writer
"#;
        let cfg: AuthConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.trust_domain.as_deref(), Some("observability.local"));
        assert_eq!(cfg.entities.len(), 2);
        assert_eq!(cfg.get_roles("policy-engine"), vec!["observability-reader"]);
        assert_eq!(cfg.get_roles("client").len(), 2);
        assert!(cfg.get_roles("unknown").is_empty());
        assert!(cfg.has_entity("client"));
        assert!(!cfg.has_entity("missing"));
    }

    #[test]
    fn tenant_lookup_falls_back_to_default() {
        let yaml = r#"
default_tenant: shared
entities:
  alice:
    tenant: acme
    roles: [user]
  bob:
    roles: [user]
"#;
        let cfg: AuthConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.get_tenant("alice"), "acme");
        assert_eq!(
            cfg.get_tenant("bob"),
            "shared",
            "no entity tenant ⇒ default_tenant"
        );
        assert_eq!(
            cfg.get_tenant("unknown"),
            "shared",
            "unknown entity ⇒ default_tenant"
        );
    }

    #[test]
    fn default_tenant_is_default_when_unspecified() {
        let yaml = r#"
entities:
  bob:
    roles: [user]
"#;
        let cfg: AuthConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.default_tenant, "default");
        assert_eq!(cfg.get_tenant("bob"), "default");
    }

    #[test]
    fn test_load_real_config() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/auth_config.yaml");
        if std::path::Path::new(path).exists() {
            let cfg = AuthConfig::load(path).unwrap();
            assert!(cfg.has_entity("policy-engine"));
            assert!(cfg.has_entity("admin"));
            assert!(!cfg.get_roles("admin").is_empty());
        }
    }
}
