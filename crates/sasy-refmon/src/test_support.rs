//! Shared stand-ins for the tests of both credential-injection call sites.
//!
//! The gRPC proxy arm and the HTTP forward proxy have to fail the same way
//! when a credential cannot be fetched, so they are tested against the same
//! backend and the same policy answer.

use std::future::Future;

use sasy_common::policy_engine::{Action, ActionResult, AuthorizationResponse};
use sasy_common::SessionScope;
use sasy_credential::{
    CredentialError, CredentialSource, CredentialView, Freshness, ResolvedCredentials,
};

use crate::policy::PolicyChecker;
use crate::RefmonError;

/// Authorizes everything and asks for one transform, so the authorized arm
/// actually runs a credential lookup.
pub(crate) struct AllowAllWithTransform;

impl PolicyChecker for AllowAllWithTransform {
    fn check_authorization(
        &self,
        _current_node_ids: &[String],
        actions: Vec<Action>,
        _entity: Option<&str>,
        _roles: &[String],
        _scope: SessionScope,
        _principal: Option<&str>,
        _policy_id: Option<String>,
    ) -> impl Future<Output = Result<AuthorizationResponse, RefmonError>> + Send {
        let results = actions
            .iter()
            .enumerate()
            .map(|(i, _)| ActionResult {
                index: i as u32,
                authorized: true,
                trace: None,
                transform_ids: vec!["inject_key".to_string()],
                deny_if_unauthorized: true,
            })
            .collect();
        std::future::ready(Ok(AuthorizationResponse {
            results,
            timing: None,
        }))
    }
}

/// The kind of message a real secret manager produces on a failed read: it
/// names the manager's address, the mount, and the path under it.
pub(crate) const DISCLOSING_BACKEND_MESSAGE: &str =
    "openbao at https://vault.internal:8200: reading kv/data/sasy/default/alice/openai: \
permission denied";

/// A credential backend whose error text carries deployment detail — the
/// address, mount and path from [`DISCLOSING_BACKEND_MESSAGE`]. Used to check
/// that none of it reaches the caller of the proxy.
pub(crate) struct DisclosingSource;

#[tonic::async_trait]
impl CredentialSource for DisclosingSource {
    fn backend(&self) -> &'static str {
        "test"
    }
    fn location(&self) -> String {
        "https://vault.internal:8200".to_string()
    }
    async fn resolve(
        &self,
        _tenant: &str,
        _entity: &str,
        _service: &str,
        _view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError> {
        Err(CredentialError::Backend(
            DISCLOSING_BACKEND_MESSAGE.to_string(),
        ))
    }
    fn is_current(&self, _freshness: &Freshness) -> bool {
        false
    }
    async fn set_credentials(
        &self,
        _tenant: &str,
        _entity: &str,
        _service: &str,
        _credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        unreachable!("these tests never write")
    }
}

/// A credential backend that is unavailable — a secret manager that cannot be
/// reached, or a token that has been revoked.
pub(crate) struct UnreachableSource;

#[tonic::async_trait]
impl CredentialSource for UnreachableSource {
    fn backend(&self) -> &'static str {
        "test"
    }
    fn location(&self) -> String {
        "nowhere".to_string()
    }
    async fn resolve(
        &self,
        _tenant: &str,
        _entity: &str,
        _service: &str,
        _view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError> {
        Err(CredentialError::Backend(
            "the credential backend is unreachable".to_string(),
        ))
    }
    fn is_current(&self, _freshness: &Freshness) -> bool {
        false
    }
    async fn set_credentials(
        &self,
        _tenant: &str,
        _entity: &str,
        _service: &str,
        _credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        unreachable!("these tests never write")
    }
}

/// A transform config with one header injection, keyed `inject_key` to match
/// what [`AllowAllWithTransform`] asks for.
pub(crate) fn injecting_transform_config() -> crate::transforms::TransformConfig {
    let mut config = crate::transforms::TransformConfig::default();
    config.transforms.insert(
        "inject_key".into(),
        crate::transforms::Transform::AddHeader {
            key: "Authorization".into(),
            format_str: "Bearer {api_key}".into(),
            service: "openai".into(),
        },
    );
    config
}
