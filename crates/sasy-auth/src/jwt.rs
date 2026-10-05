//! JWT authentication provider with JWKS support.
//!
//! Supports static keys (HMAC/RSA PEM) and dynamic JWKS
//! endpoint fetching with TTL-based caching.

use std::sync::Arc;
use std::time::{Duration, Instant};

use jsonwebtoken::{decode, decode_header, jwk, Algorithm, DecodingKey, Validation};
use parking_lot::RwLock;
use serde_json::Value;
use tracing::debug;

use crate::config::AuthConfig;
use crate::error::AuthError;
use crate::provider::{AuthProvider, AuthResult};

/// Configuration for JWT authentication.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// HMAC secret (for HS256/HS384/HS512).
    pub hmac_secret: Option<String>,
    /// RSA public key PEM (for RS256/RS384/RS512).
    pub rsa_pem: Option<Vec<u8>>,
    /// JWKS endpoint URL for dynamic key fetching.
    /// When set, keys are fetched from this URL and
    /// cached with `jwks_cache_ttl`.
    pub jwks_url: Option<String>,
    /// JWKS cache TTL (default: 1 hour).
    pub jwks_cache_ttl: Duration,
    /// Expected issuer (`iss` claim).
    pub issuer: Option<String>,
    /// Expected audience (`aud` claim).
    pub audience: Option<String>,
    /// Claim containing the entity (default: `sub`).
    pub entity_claim: String,
    /// Claim containing roles (default: `roles`).
    /// Supports dot-notation for nested claims.
    pub roles_claim: String,
    /// Metadata key for the token (default:
    /// `authorization`).
    pub metadata_key: String,
    /// Token prefix (default: `Bearer `).
    pub token_prefix: String,
    /// If true, look up roles from auth config instead
    /// of the token.
    pub use_central_roles: bool,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            hmac_secret: None,
            rsa_pem: None,
            jwks_url: None,
            jwks_cache_ttl: Duration::from_secs(3600),
            issuer: None,
            audience: None,
            entity_claim: "sub".into(),
            roles_claim: "roles".into(),
            metadata_key: "authorization".into(),
            token_prefix: "Bearer ".into(),
            use_central_roles: false,
        }
    }
}

/// Cached JWKS key set.
struct JwksCache {
    keys: jwk::JwkSet,
    fetched_at: Instant,
}

/// JWT authentication provider with optional JWKS support.
pub struct JwtAuthProvider {
    config: JwtConfig,
    auth_config: Option<AuthConfig>,
    jwks_cache: Arc<RwLock<Option<JwksCache>>>,
}

impl JwtAuthProvider {
    pub fn new(config: JwtConfig, auth_config: Option<AuthConfig>) -> Self {
        Self {
            config,
            auth_config,
            jwks_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Get a decoding key for the given token.
    ///
    /// Priority: static HMAC → static RSA PEM → JWKS (by KID).
    fn get_decoding_key(&self, token: &str) -> Result<(DecodingKey, Algorithm), AuthError> {
        // 1. Static HMAC
        if let Some(ref secret) = self.config.hmac_secret {
            return Ok((
                DecodingKey::from_secret(secret.as_bytes()),
                Algorithm::HS256,
            ));
        }

        // 2. Static RSA PEM
        if let Some(ref pem) = self.config.rsa_pem {
            let key = DecodingKey::from_rsa_pem(pem)
                .map_err(|e| AuthError::Config(format!("invalid RSA PEM: {e}")))?;
            return Ok((key, Algorithm::RS256));
        }

        // 3. JWKS endpoint
        if self.config.jwks_url.is_some() {
            return self.get_jwks_key(token);
        }

        Err(AuthError::Config(
            "no HMAC secret, RSA key, or JWKS URL configured".into(),
        ))
    }

    /// Fetch or use cached JWKS keys, then find the key matching the token's KID.
    fn get_jwks_key(&self, token: &str) -> Result<(DecodingKey, Algorithm), AuthError> {
        let header = decode_header(token)
            .map_err(|e| AuthError::InvalidToken(format!("invalid JWT header: {e}")))?;
        let kid = header.kid.as_deref().ok_or_else(|| {
            AuthError::InvalidToken("JWT has no 'kid' header for JWKS lookup".into())
        })?;

        // Check cache
        let needs_refresh = {
            let cache = self.jwks_cache.read();
            !matches!(cache.as_ref(), Some(c) if c.fetched_at.elapsed() < self.config.jwks_cache_ttl)
        };

        if needs_refresh {
            self.refresh_jwks()?;
        }

        // Look up key by KID
        let cache = self.jwks_cache.read();
        let jwks = cache
            .as_ref()
            .ok_or_else(|| AuthError::Config("JWKS not loaded".into()))?;

        let jwk = jwks
            .keys
            .keys
            .iter()
            .find(|k| k.common.key_id.as_deref() == Some(kid))
            .ok_or_else(|| AuthError::InvalidToken(format!("no JWKS key with kid '{kid}'")))?;

        let alg = jwk
            .common
            .key_algorithm
            .map(|a| match a {
                jwk::KeyAlgorithm::RS256 => Algorithm::RS256,
                jwk::KeyAlgorithm::RS384 => Algorithm::RS384,
                jwk::KeyAlgorithm::RS512 => Algorithm::RS512,
                jwk::KeyAlgorithm::ES256 => Algorithm::ES256,
                jwk::KeyAlgorithm::ES384 => Algorithm::ES384,
                _ => Algorithm::RS256,
            })
            .unwrap_or(Algorithm::RS256);

        let key = DecodingKey::from_jwk(jwk)
            .map_err(|e| AuthError::Config(format!("invalid JWK: {e}")))?;

        Ok((key, alg))
    }

    /// Fetch JWKS from the endpoint and update cache.
    fn refresh_jwks(&self) -> Result<(), AuthError> {
        let url = self
            .config
            .jwks_url
            .as_ref()
            .ok_or_else(|| AuthError::Config("no JWKS URL".into()))?;

        debug!(url = %url, "fetching JWKS");

        // Blocking HTTP fetch — use block_in_place since we may be
        // called from within a tokio runtime (via tonic interceptor).
        let resp = tokio::task::block_in_place(|| reqwest::blocking::get(url))
            .map_err(|e| AuthError::Config(format!("JWKS fetch failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(AuthError::Config(format!(
                "JWKS fetch returned {}",
                resp.status()
            )));
        }

        let jwks: jwk::JwkSet = resp
            .json()
            .map_err(|e| AuthError::Config(format!("JWKS parse failed: {e}")))?;

        debug!(keys = jwks.keys.len(), "JWKS loaded");

        *self.jwks_cache.write() = Some(JwksCache {
            keys: jwks,
            fetched_at: Instant::now(),
        });

        Ok(())
    }

    /// Extract a possibly nested claim using dot notation.
    fn extract_claim<'a>(payload: &'a Value, path: &str) -> Option<&'a Value> {
        let mut current = payload;
        for part in path.split('.') {
            current = current.get(part)?;
        }
        Some(current)
    }

    /// Parse roles from a JSON value.
    fn parse_roles(value: &Value) -> Vec<String> {
        match value {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            Value::String(s) => vec![s.clone()],
            _ => vec![],
        }
    }
}

impl AuthProvider for JwtAuthProvider {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        let header_val = metadata
            .get(&self.config.metadata_key)
            .ok_or_else(|| {
                AuthError::Unauthenticated(format!("missing '{}' header", self.config.metadata_key))
            })?
            .to_str()
            .map_err(|_| AuthError::InvalidToken("non-ASCII header value".into()))?;

        let token = if !self.config.token_prefix.is_empty() {
            header_val
                .strip_prefix(&self.config.token_prefix)
                .ok_or_else(|| {
                    AuthError::InvalidToken(format!(
                        "token must start with '{}'",
                        self.config.token_prefix
                    ))
                })?
        } else {
            header_val
        };

        // Get decoding key (static or JWKS)
        let (key, algorithm) = self.get_decoding_key(token)?;

        // Build validation
        let mut validation = Validation::new(algorithm);

        // `nbf` is off by default in jsonwebtoken, so a token that is not yet
        // valid was accepted as long as `exp` was still in the future. The
        // check only applies when the claim is present — a token without one
        // is unaffected — and it honours `leeway`, so clock skew is covered.
        // It also rejects an `nbf` that fails to parse, rather than ignoring
        // it.
        validation.validate_nbf = true;

        // Only validate iss/aud when configured. set_issuer/set_audience
        // alone is not enough: jsonwebtoken skips the check when the claim
        // is simply ABSENT from the token, so a token with no `iss`/`aud`
        // would pass despite the operator configuring one. Mark each
        // configured claim as required (in addition to the default `exp`)
        // so a missing claim is a hard rejection.
        if let Some(ref iss) = self.config.issuer {
            validation.set_issuer(&[iss.as_str()]);
            validation.required_spec_claims.insert("iss".to_string());
        }

        if let Some(ref aud) = self.config.audience {
            validation.set_audience(&[aud.as_str()]);
            validation.required_spec_claims.insert("aud".to_string());
        }

        // Decode
        let token_data = decode::<Value>(token, &key, &validation).map_err(|e| match e.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
            _ => AuthError::InvalidToken(e.to_string()),
        })?;

        let payload = &token_data.claims;

        // Extract entity
        let entity = payload
            .get(&self.config.entity_claim)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AuthError::InvalidToken(format!("missing '{}' claim", self.config.entity_claim))
            })?
            .to_string();

        // Extract roles
        let roles = if self.config.use_central_roles {
            self.auth_config
                .as_ref()
                .map(|c| c.get_roles(&entity))
                .unwrap_or_default()
        } else {
            Self::extract_claim(payload, &self.config.roles_claim)
                .map(Self::parse_roles)
                .unwrap_or_default()
        };

        // Tenant always comes from auth_config — JWT claims aren't
        // trusted to set the tenant since that would let an IdP
        // (or a stolen token) impersonate cross-tenant.
        let result = AuthResult::success(entity.clone(), roles, "jwt");
        Ok(match self.auth_config.as_ref() {
            Some(cfg) => result.with_tenant(cfg.get_tenant(&entity)),
            None => result,
        })
    }

    fn name(&self) -> &str {
        "jwt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};
    use serde_json::json;

    fn make_token(claims: &Value, secret: &str) -> String {
        encode(
            &Header::default(),
            claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    }

    fn provider(secret: &str) -> JwtAuthProvider {
        JwtAuthProvider::new(
            JwtConfig {
                hmac_secret: Some(secret.into()),
                ..Default::default()
            },
            None,
        )
    }

    #[test]
    fn test_jwt_valid_token() {
        let secret = "test-secret-key-1234567890123456";
        let claims = json!({
            "sub": "alice",
            "roles": ["admin", "user"],
            "exp": 9999999999u64,
        });
        let token = make_token(&claims, secret);

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());

        let result = provider(secret).authenticate(&meta).unwrap();
        assert_eq!(result.entity.as_deref(), Some("alice"));
        assert_eq!(result.roles, vec!["admin", "user"]);
        assert_eq!(result.auth_method, "jwt");
    }

    /// Reject future-dated tokens by enabling `nbf` validation explicitly;
    /// jsonwebtoken does not enable it by default.
    #[test]
    fn jwt_not_yet_valid_token_is_rejected() {
        let secret = "test-secret-key-1234567890123456";
        let far_future = 9_000_000_000u64;
        let claims = json!({
            "sub": "alice",
            "roles": ["admin"],
            "exp": 9_999_999_999u64,
            "nbf": far_future,
        });
        let token = make_token(&claims, secret);

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());

        assert!(
            provider(secret).authenticate(&meta).is_err(),
            "a token whose nbf is in the future must not authenticate"
        );
    }

    /// And a token with an `nbf` already past still works — the check applies
    /// only when the claim is present and reached, so enabling it must not
    /// break ordinary tokens.
    #[test]
    fn jwt_token_whose_nbf_has_passed_is_accepted() {
        let secret = "test-secret-key-1234567890123456";
        let claims = json!({
            "sub": "alice",
            "roles": ["admin"],
            "exp": 9_999_999_999u64,
            "nbf": 1u64,
        });
        let token = make_token(&claims, secret);

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());

        assert_eq!(
            provider(secret)
                .authenticate(&meta)
                .expect("an already-valid token must authenticate")
                .entity
                .as_deref(),
            Some("alice")
        );
    }

    #[test]
    fn test_jwt_missing_header() {
        let secret = "test-secret-key-1234567890123456";
        let meta = tonic::metadata::MetadataMap::new();
        let err = provider(secret).authenticate(&meta);
        assert!(err.is_err());
    }

    #[test]
    fn test_jwt_invalid_token() {
        let secret = "test-secret-key-1234567890123456";
        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert(
            "authorization",
            "Bearer invalid.token.here".parse().unwrap(),
        );
        let err = provider(secret).authenticate(&meta);
        assert!(err.is_err());
    }

    #[test]
    fn test_jwt_missing_sub() {
        let secret = "test-secret-key-1234567890123456";
        let claims = json!({
            "name": "alice",
            "exp": 9999999999u64,
        });
        let token = make_token(&claims, secret);

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());
        let err = provider(secret).authenticate(&meta);
        assert!(err.is_err());
    }

    #[test]
    fn test_jwt_nested_roles_claim() {
        let secret = "test-secret-key-1234567890123456";
        let claims = json!({
            "sub": "bob",
            "realm_access": {
                "roles": ["viewer"]
            },
            "exp": 9999999999u64,
        });
        let token = make_token(&claims, secret);

        let p = JwtAuthProvider::new(
            JwtConfig {
                hmac_secret: Some(secret.into()),
                roles_claim: "realm_access.roles".into(),
                ..Default::default()
            },
            None,
        );

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());
        let result = p.authenticate(&meta).unwrap();
        assert_eq!(result.roles, vec!["viewer"]);
    }

    #[test]
    fn test_jwt_central_roles() {
        let secret = "test-secret-key-1234567890123456";
        let claims = json!({
            "sub": "client",
            "exp": 9999999999u64,
        });
        let token = make_token(&claims, secret);

        let mut auth_cfg = AuthConfig::default();
        auth_cfg.entities.insert(
            "client".into(),
            crate::config::EntityConfig {
                description: None,
                roles: vec!["user".into(), "writer".into()],
                tenant: None,
            },
        );

        let p = JwtAuthProvider::new(
            JwtConfig {
                hmac_secret: Some(secret.into()),
                use_central_roles: true,
                ..Default::default()
            },
            Some(auth_cfg),
        );

        let mut meta = tonic::metadata::MetadataMap::new();
        meta.insert("authorization", format!("Bearer {token}").parse().unwrap());
        let result = p.authenticate(&meta).unwrap();
        assert_eq!(result.roles, vec!["user", "writer"]);
    }
}
