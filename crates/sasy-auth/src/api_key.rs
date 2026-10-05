//! API key authentication provider.

use std::collections::HashMap;

use crate::config::AuthConfig;
use crate::error::AuthError;
use crate::provider::{AuthProvider, AuthResult};

/// Environment variable holding the per-deployment API key suffix.
pub const API_KEY_SUFFIX_ENV: &str = "SASY_API_KEY_SUFFIX";

/// Environment variable acknowledging a run on the shipped key names.
pub const ALLOW_DEFAULT_KEYS_ENV: &str = "SASY_ALLOW_DEFAULT_KEYS";

/// The API key names this repository ships in its example auth configs
/// (`config/auth/apikey.example.json`, `config/auth/all.example.json`).
///
/// They are public knowledge — anyone who has read the repository knows that
/// `admin-test-key` maps to the `admin` entity — so a deployment that accepts
/// them verbatim has no admin credential worth the name. The provider refuses
/// to load a config containing any of them unless the operator either sets a
/// suffix (making the effective keys unguessable) or explicitly acknowledges an
/// insecure local run.
pub const SHIPPED_API_KEY_NAMES: [&str; 2] = ["admin-test-key", "demo-key"];

/// What the suffix rule did to a config's `static_keys`.
///
/// Returned instead of a bare map so the caller logs the insecure-run warning
/// exactly once, at the point the config is loaded, rather than the rule
/// reaching for a logger of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticKeyResolution {
    /// A suffix was set: every configured key is now `<name>-<suffix>`.
    Suffixed(HashMap<String, String>),
    /// No suffix, and no shipped key name appears in the config — the operator
    /// chose their own key values, so there is nothing predictable to refuse.
    Unchanged(HashMap<String, String>),
    /// No suffix, shipped key names present, and the operator set
    /// `SASY_ALLOW_DEFAULT_KEYS=1`. The keys are used verbatim; the caller
    /// warns.
    DefaultsAcknowledged(HashMap<String, String>),
}

impl StaticKeyResolution {
    /// The keys the provider should match against.
    pub fn into_keys(self) -> HashMap<String, String> {
        match self {
            Self::Suffixed(k) | Self::Unchanged(k) | Self::DefaultsAcknowledged(k) => k,
        }
    }
}

/// Apply the API key suffix rule to a config's `static_keys`.
///
/// The config file keeps the plain names; the running provider knows only the
/// values this returns. `suffix` is the value of `SASY_API_KEY_SUFFIX` (already
/// trimmed; `None` when unset or empty) and `allow_default_keys` the value of
/// `SASY_ALLOW_DEFAULT_KEYS`.
///
/// Pure and env-free so the rule can be tested without mutating process
/// environment — env reading lives in [`resolve_static_keys_from_env`].
pub fn resolve_static_keys(
    keys: HashMap<String, String>,
    suffix: Option<&str>,
    allow_default_keys: bool,
) -> Result<StaticKeyResolution, AuthError> {
    if let Some(suffix) = suffix {
        let suffixed = keys
            .into_iter()
            .map(|(name, entity)| (format!("{name}-{suffix}"), entity))
            .collect();
        return Ok(StaticKeyResolution::Suffixed(suffixed));
    }

    let mut shipped: Vec<&str> = SHIPPED_API_KEY_NAMES
        .iter()
        .copied()
        .filter(|name| keys.contains_key(*name))
        .collect();
    if shipped.is_empty() {
        // Every key is a value the operator chose. Nothing predictable to
        // refuse, suffix or not.
        return Ok(StaticKeyResolution::Unchanged(keys));
    }
    shipped.sort_unstable();

    if allow_default_keys {
        return Ok(StaticKeyResolution::DefaultsAcknowledged(keys));
    }

    Err(AuthError::Config(format!(
        "refusing to load API keys that this repository ships verbatim: {}. \
         Their names and the entities they map to are public, so they are not \
         credentials. Rule: with {API_KEY_SUFFIX_ENV} set, every static key in \
         the config is matched as <name>-<suffix>, and the file keeps the plain \
         names. Either set {API_KEY_SUFFIX_ENV} to a random value (`make serve` \
         writes one into .env for you), or set {ALLOW_DEFAULT_KEYS_ENV}=1 to \
         acknowledge an insecure local run.",
        shipped.join(", ")
    )))
}

/// [`resolve_static_keys`] driven by the process environment, logging the
/// insecure-run warning when the operator has acknowledged the shipped keys.
pub fn resolve_static_keys_from_env(
    keys: HashMap<String, String>,
) -> Result<HashMap<String, String>, AuthError> {
    let raw = std::env::var(API_KEY_SUFFIX_ENV).unwrap_or_default();
    let suffix = match raw.trim() {
        "" => None,
        s => Some(s.to_string()),
    };
    let allow_default_keys = sasy_common::env_flag(ALLOW_DEFAULT_KEYS_ENV);

    let resolution = resolve_static_keys(keys, suffix.as_deref(), allow_default_keys)?;
    match &resolution {
        StaticKeyResolution::Suffixed(k) => {
            tracing::info!(
                keys = k.len(),
                "applied {API_KEY_SUFFIX_ENV} to static API keys (<name>-<suffix>)"
            );
        }
        StaticKeyResolution::DefaultsAcknowledged(_) => {
            tracing::warn!(
                "{ALLOW_DEFAULT_KEYS_ENV} is set: accepting the API keys this \
                 repository ships verbatim (e.g. admin-test-key -> admin). Their \
                 values are public. INSECURE; local development only."
            );
        }
        StaticKeyResolution::Unchanged(_) => {}
    }
    Ok(resolution.into_keys())
}

/// API key authentication provider.
///
/// Validates a static API key from request metadata and
/// maps it to an entity + roles.
pub struct ApiKeyAuthProvider {
    /// Map of API key → entity name.
    keys: HashMap<String, String>,
    auth_config: AuthConfig,
    /// Metadata key for the API key
    /// (default: `x-api-key`).
    metadata_key: String,
}

impl ApiKeyAuthProvider {
    pub fn new(keys: HashMap<String, String>, auth_config: AuthConfig) -> Self {
        Self {
            keys,
            auth_config,
            metadata_key: sasy_common::headers::API_KEY.to_string(),
        }
    }

    pub fn with_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.metadata_key = key.into();
        self
    }
}

impl AuthProvider for ApiKeyAuthProvider {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        let api_key = metadata
            .get(&self.metadata_key)
            .ok_or_else(|| AuthError::Unauthenticated(format!("missing '{}'", self.metadata_key)))?
            .to_str()
            .map_err(|_| AuthError::InvalidToken("non-ASCII API key".into()))?;

        // Only configured keys authenticate. An empty map denies all callers;
        // never interpret an unknown key as an entity name.
        let entity = self
            .keys
            .get(api_key)
            .cloned()
            .ok_or_else(|| AuthError::Unauthenticated("invalid API key".into()))?;

        let roles = self.auth_config.get_roles(&entity);
        let tenant = self.auth_config.get_tenant(&entity);
        Ok(AuthResult::success(entity, roles, "api_key").with_tenant(tenant))
    }

    fn name(&self) -> &str {
        "api_key"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EntityConfig;

    fn test_config() -> AuthConfig {
        let mut cfg = AuthConfig::default();
        cfg.entities.insert(
            "client".into(),
            EntityConfig {
                description: None,
                roles: vec!["user".into(), "writer".into()],
                tenant: None,
            },
        );
        cfg
    }

    #[test]
    fn test_api_key_valid() {
        let mut keys = HashMap::new();
        keys.insert("secret-123".into(), "client".into());
        let provider = ApiKeyAuthProvider::new(keys, test_config());

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-api-key", "secret-123".parse().unwrap());

        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("client"));
        assert_eq!(result.roles, vec!["user", "writer"]);
        assert_eq!(result.auth_method, "api_key");
    }

    #[test]
    fn test_api_key_invalid() {
        let mut keys = HashMap::new();
        keys.insert("secret-123".into(), "client".into());
        let provider = ApiKeyAuthProvider::new(keys, test_config());

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-api-key", "wrong-key".parse().unwrap());

        assert!(provider.authenticate(&meta).is_err());
    }

    #[test]
    fn test_api_key_missing() {
        let keys = HashMap::new();
        let provider = ApiKeyAuthProvider::new(keys, test_config());
        let meta = tonic::metadata::MetadataMap::new();
        assert!(provider.authenticate(&meta).is_err());
    }

    // ── The suffix rule ───────────────────────────────────────────────
    //
    // Exercised through the pure entry point, so no test here depends on the
    // process environment. The env-driven wrapper and the loader path are
    // covered in the crate-root tests.

    fn keys(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn a_suffix_rewrites_every_key_and_keeps_its_entity() {
        let resolved = resolve_static_keys(
            keys(&[("admin-test-key", "admin"), ("demo-key", "demo")]),
            Some("abc123"),
            false,
        )
        .expect("a suffix is always accepted");
        let out = resolved.into_keys();
        assert_eq!(
            out.get("admin-test-key-abc123").map(String::as_str),
            Some("admin")
        );
        assert_eq!(out.get("demo-key-abc123").map(String::as_str), Some("demo"));
        assert!(
            !out.contains_key("admin-test-key"),
            "the plain name must not survive"
        );
    }

    /// The acknowledgement does not silently win over a suffix: with both set,
    /// the keys are still suffixed.
    #[test]
    fn a_suffix_wins_over_the_acknowledgement() {
        let resolved =
            resolve_static_keys(keys(&[("admin-test-key", "admin")]), Some("abc123"), true)
                .expect("a suffix is always accepted");
        assert!(matches!(resolved, StaticKeyResolution::Suffixed(_)));
        assert!(resolved.into_keys().contains_key("admin-test-key-abc123"));
    }

    #[test]
    fn a_shipped_key_without_a_suffix_is_refused() {
        let err = resolve_static_keys(
            keys(&[("admin-test-key", "admin"), ("custom-value", "demo")]),
            None,
            false,
        )
        .expect_err("a shipped key name must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("admin-test-key"), "got: {msg}");
        assert!(
            !msg.contains("custom-value"),
            "only the shipped names are named: {msg}"
        );
    }

    /// Every one of the nine shipped names triggers the refusal — a name added
    /// to the config files but not to the constant would be a silent hole.
    #[test]
    fn every_shipped_name_is_refused_on_its_own() {
        for name in SHIPPED_API_KEY_NAMES {
            let result = resolve_static_keys(keys(&[(name, "someone")]), None, false);
            assert!(result.is_err(), "{name} was not refused");
        }
    }

    /// Every key name in the tracked provider configs is listed in the
    /// constant. A key added to a config file without being added here would
    /// be accepted verbatim by the provider.
    #[test]
    fn the_constant_covers_the_tracked_config_files() {
        // The crate sits at a different depth in different checkouts, and a
        // checkout carries either the working configs or their examples. A
        // crate unpacked on its own has no repository config directory at all.
        let base = env!("CARGO_MANIFEST_DIR");
        if !["/../../config/auth", "/../../config/auth"]
            .iter()
            .any(|dir| std::path::Path::new(&format!("{base}{dir}")).is_dir())
        {
            return;
        }
        let mut checked = 0;
        for rel in [
            "/../../config/auth/apikey.json",
            "/../../config/auth/all.json",
            "/../../config/auth/apikey.example.json",
            "/../../config/auth/all.example.json",
            "/../../config/auth/apikey.example.json",
            "/../../config/auth/all.example.json",
        ] {
            let path = format!("{}{rel}", env!("CARGO_MANIFEST_DIR"));
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
            let mut stack = vec![doc];
            while let Some(node) = stack.pop() {
                if let Some(map) = node.get("static_keys").and_then(|v| v.as_object()) {
                    for name in map.keys() {
                        checked += 1;
                        assert!(
                            SHIPPED_API_KEY_NAMES.contains(&name.as_str()),
                            "{path} ships '{name}', which SHIPPED_API_KEY_NAMES does not list, \
                             so the provider would accept it verbatim"
                        );
                    }
                }
                if let Some(providers) = node.get("providers").and_then(|v| v.as_array()) {
                    stack.extend(providers.iter().cloned());
                }
            }
        }
        assert!(checked > 0, "no API key name was found to check");
    }

    #[test]
    fn a_shipped_key_is_accepted_verbatim_once_acknowledged() {
        let resolved = resolve_static_keys(keys(&[("admin-test-key", "admin")]), None, true)
            .expect("the acknowledgement allows the shipped keys");
        assert!(matches!(
            resolved,
            StaticKeyResolution::DefaultsAcknowledged(_)
        ));
        assert!(resolved.into_keys().contains_key("admin-test-key"));
    }

    #[test]
    fn custom_keys_need_no_suffix_and_no_acknowledgement() {
        let resolved =
            resolve_static_keys(keys(&[("9f2c-not-a-shipped-name", "demo")]), None, false)
                .expect("custom key values are the operator's business");
        assert!(matches!(resolved, StaticKeyResolution::Unchanged(_)));
        assert!(resolved.into_keys().contains_key("9f2c-not-a-shipped-name"));
    }

    /// An empty or whitespace-only `SASY_API_KEY_SUFFIX` is not a suffix: it
    /// would rewrite `admin-test-key` to `admin-test-key-`, a value just as
    /// guessable. `resolve_static_keys_from_env` maps it to `None`, so the
    /// refusal applies.
    #[test]
    fn an_empty_suffix_is_treated_as_unset_by_the_env_wrapper() {
        let _env = crate::test_env::EnvGuard::set(&[
            (API_KEY_SUFFIX_ENV, Some("   ")),
            (ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        let err = resolve_static_keys_from_env(keys(&[("admin-test-key", "admin")]))
            .expect_err("a blank suffix must not count as a suffix");
        assert!(format!("{err}").contains("admin-test-key"));
    }

    /// The env wrapper trims, so a suffix pasted with a trailing newline (the
    /// usual shape of `openssl rand -hex 16` piped through a file) still
    /// produces the same key values as the shell's own `<name>-<suffix>`.
    #[test]
    fn the_env_wrapper_trims_the_suffix() {
        let _env = crate::test_env::EnvGuard::set(&[
            (API_KEY_SUFFIX_ENV, Some(" abc123\n")),
            (ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        let out = resolve_static_keys_from_env(keys(&[("admin-test-key", "admin")])).unwrap();
        assert!(out.contains_key("admin-test-key-abc123"), "got {out:?}");
    }
}
