//! Authentication providers and role-based access
//! control interceptors for the SASY engine.
//!
//! Provides:
//! - Multiple auth providers (JWT, mTLS, API key,
//!   passthrough)
//! - Auth chain for fallback authentication
//! - Tonic interceptor for gRPC request auth
//! - RBAC role checking helpers
//! - TLS configuration utilities

#[cfg(feature = "api-key")]
pub mod api_key;
#[cfg(feature = "chain")]
pub mod chain;
pub mod config;
pub mod error;
pub mod interceptor;
#[cfg(feature = "jwt")]
pub mod jwt;
#[cfg(feature = "mtls")]
pub mod mtls;
pub mod passthrough;
pub mod provider;
#[cfg(test)]
mod test_env;
pub mod tls;

// Re-export key types at crate root.
pub use config::AuthConfig;
pub use error::AuthError;
pub use interceptor::{
    check_request_any_role, check_request_role, get_auth_result, is_rbac_enabled,
    request_effective_has_role, request_effective_principal, request_effective_roles,
    request_effective_tenant, request_has_role, request_principal, request_tenant,
    require_any_role, require_role, AuthInterceptor,
};
/// The RBAC off switch is test-only — see [`interceptor::disable_rbac`].
#[cfg(any(test, feature = "test-util"))]
pub use interceptor::{disable_rbac, enable_rbac};
pub use passthrough::PassthroughAuthProvider;
pub use provider::{AuthProvider, AuthResult};
pub use tls::TlsConfig;

use std::path::Path;
use std::sync::Arc;

/// Auth provider config JSON schema.
///
/// Matches files like `config/auth/apikey.json`:
/// ```json
/// {
///   "type": "api_key",
///   "metadata_key": "x-api-key",
///   "static_keys": { "key": "entity" }
/// }
/// ```
// Most fields are read only by the gated provider builders; when full-auth is
// off (passthrough-only restricted build) they're parsed but unused.
#[cfg_attr(not(feature = "full-auth"), allow(dead_code))]
#[derive(Debug, serde::Deserialize)]
struct AuthProviderConfig {
    #[serde(rename = "type")]
    provider_type: String,
    #[serde(default = "default_metadata_key")]
    metadata_key: String,
    #[serde(default)]
    static_keys: std::collections::HashMap<String, String>,
    // JWT fields
    #[serde(default)]
    hmac_secret: Option<String>,
    #[serde(default)]
    rsa_pem_file: Option<String>,
    #[serde(default)]
    jwks_url: Option<String>,
    #[serde(default)]
    issuer: Option<String>,
    #[serde(default)]
    audience: Option<String>,
    /// Seconds to cache a fetched JWKS. Defaults to [`jwt::JwtConfig`]'s value.
    /// This interval also bounds how long a removed signing key remains cached.
    #[serde(default)]
    jwks_cache_seconds: Option<u64>,
    #[serde(default = "default_entity_claim")]
    entity_claim: String,
    #[serde(default = "default_roles_claim")]
    roles_claim: String,
    #[serde(default)]
    use_central_roles: bool,
    // mTLS fields
    #[serde(default)]
    cn_header: Option<String>,
    // Chain fields
    #[serde(default)]
    providers: Vec<AuthProviderConfig>,
}

fn default_metadata_key() -> String {
    sasy_common::headers::API_KEY.to_string()
}
fn default_entity_claim() -> String {
    "sub".to_string()
}
fn default_roles_claim() -> String {
    "roles".to_string()
}

/// Load an auth provider from a JSON config file.
///
/// Supported provider types:
///
/// - `"api_key"` — static API key authentication
/// - `"passthrough"` — no authentication (development)
///
/// The `auth_config` provides entity→role mappings
/// (from `auth_config.yaml`). Pass `None` if no role
/// mappings are needed.
pub fn load_provider_from_file(
    path: &Path,
    auth_config: Option<AuthConfig>,
) -> Result<Arc<dyn AuthProvider>, AuthError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        AuthError::Config(format!(
            "failed to read auth provider config {}: {}",
            path.display(),
            e
        ))
    })?;
    let cfg: AuthProviderConfig = serde_json::from_str(&content).map_err(|e| {
        AuthError::Config(format!(
            "failed to parse auth provider config {}: {}",
            path.display(),
            e
        ))
    })?;

    let ac = auth_config.unwrap_or_default();
    build_provider(cfg, &ac)
}

fn build_provider(
    cfg: AuthProviderConfig,
    ac: &AuthConfig,
) -> Result<Arc<dyn AuthProvider>, AuthError> {
    match cfg.provider_type.as_str() {
        #[cfg(feature = "api-key")]
        "api_key" => {
            // The config file carries plain key names; what the provider
            // matches is decided here, by the suffix rule. Reached from every
            // caller that builds an api_key provider, the `chain` arm below
            // included, so a composite config (all.json) gets the same rule
            // applied to its API-key member.
            let keys = api_key::resolve_static_keys_from_env(cfg.static_keys)?;
            let mut provider = api_key::ApiKeyAuthProvider::new(keys, ac.clone());
            if cfg.metadata_key != sasy_common::headers::API_KEY {
                provider = provider.with_metadata_key(cfg.metadata_key);
            }
            Ok(Arc::new(provider))
        }
        #[cfg(feature = "jwt")]
        "jwt" => {
            let rsa_pem = if let Some(ref pem_path) = cfg.rsa_pem_file {
                Some(std::fs::read(pem_path).map_err(|e| {
                    AuthError::Config(format!("failed to read RSA PEM {}: {}", pem_path, e))
                })?)
            } else {
                None
            };
            let jwt_config = jwt::JwtConfig {
                hmac_secret: cfg.hmac_secret,
                rsa_pem,
                jwks_url: cfg.jwks_url,
                issuer: cfg.issuer,
                audience: cfg.audience,
                entity_claim: cfg.entity_claim,
                roles_claim: cfg.roles_claim,
                metadata_key: cfg.metadata_key,
                use_central_roles: cfg.use_central_roles,
                jwks_cache_ttl: cfg
                    .jwks_cache_seconds
                    .map(std::time::Duration::from_secs)
                    .unwrap_or_else(|| jwt::JwtConfig::default().jwks_cache_ttl),
                ..Default::default()
            };
            Ok(Arc::new(jwt::JwtAuthProvider::new(
                jwt_config,
                Some(ac.clone()),
            )))
        }
        #[cfg(feature = "mtls")]
        "mtls" => {
            let mut provider = mtls::MtlsAuthProvider::new(ac.clone());
            if let Some(header) = cfg.cn_header {
                provider = provider.with_cn_header(header);
            }
            Ok(Arc::new(provider))
        }
        #[cfg(feature = "chain")]
        "chain" => {
            let providers: Vec<Box<dyn AuthProvider>> = cfg
                .providers
                .into_iter()
                .map(|p| {
                    build_provider(p, ac)
                        .map(|arc| Box::new(ArcProvider(arc)) as Box<dyn AuthProvider>)
                })
                .collect::<Result<_, _>>()?;
            Ok(Arc::new(chain::AuthChain::new(providers)))
        }
        // Fail-closed stubs for providers compiled out of this build (the
        // restricted build keeps only passthrough). A misconfigured
        // --auth-provider gets a clear error, never a silent downgrade to
        // no-auth.
        #[cfg(not(feature = "api-key"))]
        "api_key" => Err(AuthError::Config(
            "api_key auth provider not compiled into this build".into(),
        )),
        #[cfg(not(feature = "jwt"))]
        "jwt" => Err(AuthError::Config(
            "jwt auth provider not compiled into this build".into(),
        )),
        #[cfg(not(feature = "mtls"))]
        "mtls" => Err(AuthError::Config(
            "mtls auth provider not compiled into this build".into(),
        )),
        #[cfg(not(feature = "chain"))]
        "chain" => Err(AuthError::Config(
            "chain auth provider not compiled into this build".into(),
        )),
        "passthrough" | "none" => Ok(Arc::new(PassthroughAuthProvider::new(Some(ac.clone())))),
        other => Err(AuthError::Config(format!(
            "unknown auth provider type: '{other}'"
        ))),
    }
}

/// Wrapper to convert `Arc<dyn AuthProvider>` to `Box<dyn AuthProvider>`.
#[cfg(feature = "chain")]
struct ArcProvider(Arc<dyn AuthProvider>);

#[cfg(feature = "chain")]
impl AuthProvider for ArcProvider {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        self.0.authenticate(metadata)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    /// Forwarded, not defaulted. Every chain member is wrapped in this type,
    /// so inheriting the trait's empty default made `AuthChain`'s union always
    /// empty — and the interceptor then stripped nothing beyond the hardcoded
    /// key, leaving a chain member's custom CN header readable straight from
    /// client metadata. A chain ending in mTLS is the normal deployment shape.
    fn tls_identity_headers(&self) -> Vec<String> {
        self.0.tls_identity_headers()
    }
}

#[cfg(all(test, feature = "full-auth"))]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// A chain built from config must report its members' TLS identity
    /// headers, so the interceptor strips them.
    ///
    /// Exercised through `build_provider`, not by constructing an `AuthChain`
    /// directly: the builder wraps every member in `ArcProvider`, and that
    /// wrapper inheriting the trait's empty default is the whole defect. A
    /// test that builds the chain by hand passes either way.
    #[cfg(all(feature = "chain", feature = "mtls"))]
    #[test]
    fn a_config_built_chain_reports_a_members_custom_cn_header() {
        let cfg: AuthProviderConfig = serde_json::from_str(
            r#"{
                "type": "chain",
                "providers": [
                    {"type": "api_key"},
                    {"type": "mtls", "cn_header": "x-forwarded-client-cn"}
                ]
            }"#,
        )
        .expect("config parses");
        let _env = no_suffix_no_shipped_keys();
        let provider = build_provider(cfg, &sample_auth_config()).expect("provider builds");
        assert!(
            provider
                .tls_identity_headers()
                .contains(&"x-forwarded-client-cn".to_string()),
            "a chain member's header must reach the interceptor, or nothing strips it; got {:?}",
            provider.tls_identity_headers()
        );
    }

    fn sample_auth_config() -> AuthConfig {
        let yaml = r#"
entities:
  service-client:
    roles: [reader, writer]
  developer:
    roles: [reader]
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    /// No suffix, and the config under test carries no shipped key name, so
    /// the suffix rule leaves it alone. Pinned rather than inherited from the
    /// ambient environment: a developer with `SASY_API_KEY_SUFFIX` exported
    /// would otherwise see different key values than the test asserts.
    fn no_suffix_no_shipped_keys() -> test_env::EnvGuard {
        test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, None),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, None),
        ])
    }

    /// No suffix, shipped key names accepted verbatim — the acknowledged
    /// insecure-local-run path. Fixtures below match on the plain names
    /// because they exercise the provider, not the suffix rule.
    fn shipped_keys_acknowledged() -> test_env::EnvGuard {
        test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, None),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, Some("1")),
        ])
    }

    #[test]
    fn load_api_key_provider_from_json() {
        let json = r#"{
            "type": "api_key",
            "metadata_key": "x-api-key",
            "static_keys": {
                "globex-service-key": "service-client",
                "globex-dev-key": "developer"
            }
        }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let _env = shipped_keys_acknowledged();
        let provider = load_provider_from_file(f.path(), Some(sample_auth_config())).unwrap();

        assert_eq!(provider.name(), "api_key");

        // Valid key authenticates and gets roles
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-api-key", "globex-service-key".parse().unwrap());
        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("service-client"));
        assert!(result.roles.contains(&"reader".into()));

        // Invalid key is rejected
        let mut meta2 = tonic::metadata::MetadataMap::new();
        meta2.insert("x-api-key", "bad-key".parse().unwrap());
        assert!(provider.authenticate(&meta2).is_err());
    }

    #[test]
    fn load_passthrough_provider_from_json() {
        let json = r#"{ "type": "passthrough" }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let provider = load_provider_from_file(f.path(), None).unwrap();
        assert_eq!(provider.name(), "passthrough");

        // Passthrough always succeeds
        let meta = tonic::metadata::MetadataMap::new();
        assert!(provider.authenticate(&meta).is_ok());
    }

    #[test]
    fn load_unknown_provider_type_errors() {
        let json = r#"{ "type": "magic" }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let result = load_provider_from_file(f.path(), None);
        let err = result.err().expect("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("unknown auth provider type"), "got: {msg}");
    }

    #[test]
    fn load_missing_file_errors() {
        let result = load_provider_from_file(Path::new("/nonexistent/auth.json"), None);
        let err = result.err().expect("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("failed to read"), "got: {msg}");
    }

    #[test]
    fn load_bad_json_errors() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"not json").unwrap();

        let result = load_provider_from_file(f.path(), None);
        let err = result.err().expect("should fail");
        let msg = format!("{err}");
        assert!(msg.contains("failed to parse"), "got: {msg}");
    }

    #[test]
    fn load_real_apikey_config() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth/apikey.example.json"
        );
        if Path::new(path).exists() {
            let auth_cfg_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../config/auth_config.example.yaml"
            );
            let auth_config = if Path::new(auth_cfg_path).exists() {
                Some(AuthConfig::load(auth_cfg_path).unwrap())
            } else {
                None
            };
            let _env = shipped_keys_acknowledged();
            let provider = load_provider_from_file(Path::new(path), auth_config).unwrap();
            assert_eq!(provider.name(), "api_key");
        }
    }

    /// The public example resolves client/admin roles through the central map.
    #[test]
    fn public_example_keys_resolve_entities_and_roles() {
        let auth_config = AuthConfig::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth_config.example.yaml"
        ))
        .unwrap();
        let _env = shipped_keys_acknowledged();
        let provider =
            load_provider_from_file(real_apikey_config_path(), Some(auth_config)).unwrap();
        for (key, entity, admin) in [
            ("demo-key", "client", false),
            ("admin-test-key", "admin", true),
        ] {
            let mut meta = tonic::metadata::MetadataMap::new();
            meta.insert("x-api-key", key.parse().unwrap());
            let result = provider.authenticate(&meta).unwrap();
            assert_eq!(result.entity.as_deref(), Some(entity));
            assert!(result.roles.contains(&"reference-monitor-user".into()));
            assert_eq!(result.roles.contains(&"admin".into()), admin);
        }
    }

    /// Public example identities use the configured default tenant.
    #[test]
    fn public_example_keys_resolve_the_configured_tenant() {
        let auth_config = AuthConfig::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth_config.example.yaml"
        ))
        .unwrap();
        let _env = shipped_keys_acknowledged();
        let provider =
            load_provider_from_file(real_apikey_config_path(), Some(auth_config)).unwrap();
        for key in ["demo-key", "admin-test-key"] {
            let mut meta = tonic::metadata::MetadataMap::new();
            meta.insert("x-api-key", key.parse().unwrap());
            let result = provider.authenticate(&meta).unwrap();
            assert_eq!(result.tenant.as_deref(), Some("default"));
        }
    }
    #[test]
    fn load_chain_provider_from_json() {
        let json = r#"{
            "type": "chain",
            "providers": [
                {
                    "type": "api_key",
                    "metadata_key": "x-api-key",
                    "static_keys": { "test-key": "test-entity" }
                },
                {
                    "type": "mtls",
                    "entity_source": "cn"
                }
            ]
        }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let _env = no_suffix_no_shipped_keys();
        let provider = load_provider_from_file(f.path(), Some(sample_auth_config())).unwrap();
        assert_eq!(provider.name(), "chain");

        // API key succeeds (first in chain)
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-api-key", "test-key".parse().unwrap());
        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("test-entity"));

        // No credentials fails (all providers fail)
        let empty = tonic::metadata::MetadataMap::new();
        assert!(provider.authenticate(&empty).is_err());
    }

    #[test]
    fn load_real_all_json_config() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth/all.example.json"
        );
        if Path::new(path).exists() {
            let auth_cfg_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../config/auth_config.example.yaml"
            );
            let auth_config = if Path::new(auth_cfg_path).exists() {
                Some(AuthConfig::load(auth_cfg_path).unwrap())
            } else {
                None
            };
            let _env = shipped_keys_acknowledged();
            let provider = load_provider_from_file(Path::new(path), auth_config).unwrap();
            assert_eq!(provider.name(), "chain");

            // API key auth works through the chain
            let mut meta = tonic::metadata::MetadataMap::new();
            meta.insert("x-api-key", "demo-key".parse().unwrap());
            let result = provider.authenticate(&meta).unwrap();
            assert_eq!(result.entity.as_deref(), Some("client"));
        }
    }

    // ── The API key suffix rule, through the config loader ────────────
    //
    // The rule itself is unit-tested in `api_key::tests` without touching
    // process environment. These exercise the path `sasy serve` takes: the
    // tracked config files, loaded through `load_provider_from_file`.

    fn real_apikey_config_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth/apikey.example.json"
        ))
    }

    fn real_all_json_path() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/auth/all.example.json"
        ))
    }

    /// Without a suffix and without the acknowledgement, loading the tracked
    /// `apikey.json` fails — and the message names the rule and both remedies,
    /// because an operator who hits this at startup has nothing else to go on.
    #[test]
    fn shipped_keys_are_refused_when_no_suffix_is_set() {
        let _env = test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, None),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        // `let Err(..) else` rather than `expect_err`: the Ok type is
        // `Arc<dyn AuthProvider>`, which is not `Debug`.
        let Err(err) = load_provider_from_file(real_apikey_config_path(), None) else {
            panic!("the shipped keys must not load unguarded");
        };
        let msg = format!("{err}");
        assert!(msg.contains("admin-test-key"), "got: {msg}");
        assert!(msg.contains("<name>-<suffix>"), "rule not stated: {msg}");
        assert!(
            msg.contains("SASY_API_KEY_SUFFIX"),
            "remedy 1 missing: {msg}"
        );
        assert!(
            msg.contains("SASY_ALLOW_DEFAULT_KEYS"),
            "remedy 2 missing: {msg}"
        );
    }

    /// The composite provider applies the same rule to its API-key member.
    #[cfg(all(feature = "chain", feature = "mtls"))]
    #[test]
    fn the_chain_configs_api_key_member_is_refused_too() {
        let _env = test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, None),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        let Err(err) = load_provider_from_file(real_all_json_path(), None) else {
            panic!("all.json carries the shipped keys in its api_key member");
        };
        assert!(format!("{err}").contains("admin-test-key"), "got: {err}");
    }

    /// With a suffix set, the file's plain names stop working and the suffixed
    /// values take over. This is the whole point of the rule: the tracked
    /// config is unchanged, the running provider knows different values.
    #[test]
    fn a_suffix_replaces_every_shipped_key_with_its_suffixed_form() {
        let _env = test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, Some("f00dcafe")),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        let provider = load_provider_from_file(real_apikey_config_path(), None)
            .expect("a suffix makes the shipped config loadable");

        let mut bare = tonic::metadata::MetadataMap::new();
        bare.insert("x-api-key", "admin-test-key".parse().unwrap());
        assert!(
            provider.authenticate(&bare).is_err(),
            "the plain shipped key must no longer authenticate"
        );

        let mut suffixed = tonic::metadata::MetadataMap::new();
        suffixed.insert("x-api-key", "admin-test-key-f00dcafe".parse().unwrap());
        let result = provider
            .authenticate(&suffixed)
            .expect("the suffixed key must authenticate");
        assert_eq!(result.entity.as_deref(), Some("admin"));
    }

    /// A config whose static keys are all operator-chosen values is not
    /// refused, suffix or no suffix — there is nothing predictable in it.
    #[test]
    fn a_config_of_custom_keys_loads_without_a_suffix() {
        let json = r#"{
            "type": "api_key",
            "static_keys": { "9f2c-not-a-shipped-name": "developer" }
        }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let _env = test_env::EnvGuard::set(&[
            (api_key::API_KEY_SUFFIX_ENV, None),
            (api_key::ALLOW_DEFAULT_KEYS_ENV, None),
        ]);
        let provider = load_provider_from_file(f.path(), Some(sample_auth_config()))
            .expect("custom keys are the operator's business");

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("x-api-key", "9f2c-not-a-shipped-name".parse().unwrap());
        let result = provider.authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("developer"));
    }
}

#[cfg(all(test, feature = "mtls"))]
mod mtls_feature_tests {
    use super::*;
    use tonic::service::Interceptor;

    #[test]
    fn minimal_mtls_provider_loads_and_rejects_an_unverified_identity_header() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"{"type":"mtls"}"#).unwrap();
        let provider = load_provider_from_file(file.path(), None).unwrap();
        assert_eq!(provider.name(), "mtls");
        let mut interceptor = AuthInterceptor::new(provider);
        let mut request = tonic::Request::new(());
        request
            .metadata_mut()
            .insert("x-client-cn", "admin".parse().unwrap());
        assert_eq!(
            interceptor.call(request).unwrap_err().code(),
            tonic::Code::Unauthenticated
        );
    }
}
