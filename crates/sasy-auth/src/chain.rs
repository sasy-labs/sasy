//! Auth chain — tries multiple providers in order.

use tracing::debug;

use crate::error::AuthError;
use crate::provider::{AuthProvider, AuthResult};

/// Tries multiple authentication providers in order.
///
/// Returns the result from the first provider that
/// succeeds.  If all fail, returns the error from the
/// last provider.
pub struct AuthChain {
    providers: Vec<Box<dyn AuthProvider>>,
}

impl AuthChain {
    pub fn new(providers: Vec<Box<dyn AuthProvider>>) -> Self {
        assert!(
            !providers.is_empty(),
            "AuthChain requires at least one provider"
        );
        Self { providers }
    }
}

impl AuthProvider for AuthChain {
    fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<AuthResult, AuthError> {
        let mut errors: Vec<String> = Vec::new();

        for provider in &self.providers {
            match provider.authenticate(metadata) {
                Ok(result) => {
                    debug!(
                        provider = provider.name(),
                        entity = ?result.entity,
                        "auth chain: succeeded"
                    );
                    return Ok(result);
                }
                Err(e) => {
                    debug!(
                        provider = provider.name(),
                        error = %e,
                        "auth chain: failed, trying next"
                    );
                    errors.push(format!("{}: {}", provider.name(), e));
                }
            }
        }

        Err(AuthError::Unauthenticated(format!(
            "all auth providers failed: {}",
            errors.join("; ")
        )))
    }

    fn name(&self) -> &str {
        "chain"
    }

    /// The union over the chain: any member that reads a TLS-derived header
    /// needs it stripped, whichever member ends up authenticating.
    fn tls_identity_headers(&self) -> Vec<String> {
        let mut headers: Vec<String> = self
            .providers
            .iter()
            .flat_map(|p| p.tls_identity_headers())
            .collect();
        headers.sort();
        headers.dedup();
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passthrough::PassthroughAuthProvider;

    /// A provider that always fails.
    struct FailProvider;
    impl AuthProvider for FailProvider {
        fn authenticate(
            &self,
            _metadata: &tonic::metadata::MetadataMap,
        ) -> Result<AuthResult, AuthError> {
            Err(AuthError::Unauthenticated("always fails".into()))
        }
        fn name(&self) -> &str {
            "fail"
        }
    }

    #[test]
    fn test_chain_first_succeeds() {
        let chain = AuthChain::new(vec![
            Box::new(PassthroughAuthProvider::new(None)),
            Box::new(FailProvider),
        ]);
        let meta = tonic::metadata::MetadataMap::new();
        let result = chain.authenticate(&meta).unwrap();
        assert_eq!(result.auth_method, "passthrough");
    }

    #[test]
    fn test_chain_fallback() {
        let chain = AuthChain::new(vec![
            Box::new(FailProvider),
            Box::new(PassthroughAuthProvider::new(None)),
        ]);
        let meta = tonic::metadata::MetadataMap::new();
        let result = chain.authenticate(&meta).unwrap();
        assert_eq!(result.auth_method, "passthrough");
    }

    #[test]
    fn test_chain_all_fail() {
        let chain = AuthChain::new(vec![Box::new(FailProvider), Box::new(FailProvider)]);
        let meta = tonic::metadata::MetadataMap::new();
        assert!(chain.authenticate(&meta).is_err());
    }
}
