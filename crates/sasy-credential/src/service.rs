//! gRPC service wrapper around a [`CredentialSource`].

use std::sync::Arc;

use tonic::{Request, Response, Status};

use sasy_common::credential_server::{
    credential_server_server, Credential, CredentialResponse, Credentials, CredentialsRequest,
    SetCredentialsRequest,
};

use crate::error::CredentialError;
use crate::source::{CredentialSource, CredentialView};
use crate::store::WILDCARD_ENTITY;

/// tonic gRPC service that delegates to the configured credential backend.
pub struct CredentialService {
    source: Arc<dyn CredentialSource>,
}

impl CredentialService {
    pub fn new(source: Arc<dyn CredentialSource>) -> Self {
        Self { source }
    }
}

/// Turn a backend error from a write into a status.
///
/// A read-only backend is not a failure of the server: the operator asked for
/// a write this deployment does not do, so it gets `FAILED_PRECONDITION` and
/// a message naming the backend, rather than an opaque internal error that
/// invites a retry.
fn write_status(e: CredentialError) -> Status {
    match e {
        CredentialError::ReadOnly(msg) => Status::failed_precondition(msg),
        other => Status::internal(format!("credential store failed: {other}")),
    }
}

#[tonic::async_trait]
impl credential_server_server::CredentialServer for CredentialService {
    async fn get_credentials(
        &self,
        request: Request<CredentialsRequest>,
    ) -> Result<Response<Credentials>, Status> {
        sasy_auth::check_request_role(&request, sasy_common::roles::CREDENTIAL_READER)?;
        // Tenant is auth-derived. Any wire `tenant_id` is
        // ignored — letting clients pick the tenant would let one
        // credential reader fish credentials from another tenant.
        let tenant_id = sasy_auth::request_tenant(&request, "default");
        // A trusted `service-proxy` relay (e.g. the reference monitor doing
        // credential injection on behalf of end-users) may read any entity's
        // credentials within its tenant, including the wildcard "*". Any other
        // `credential-reader` is confined to its OWN authenticated principal —
        // otherwise one reader could fish another entity's plaintext secrets.
        let is_relay = sasy_auth::request_has_role(&request, sasy_common::roles::SERVICE_PROXY);
        let principal = sasy_auth::request_principal(&request);
        let req = request.into_inner();

        let entity = req.entity.unwrap_or_default();
        let service = req.service.unwrap_or_default();

        if entity.is_empty() || service.is_empty() {
            return Ok(Response::new(Credentials {
                credentials: vec![],
            }));
        }

        // The wildcard is checked outright, not left to the principal
        // comparison below. `*` is the key the tenant-wide platform secrets
        // live under, and a principal that happens to BE the string `*` —
        // an entity named that, or a token whose subject claim is that —
        // satisfied `principal == entity` and collected them. The message
        // promised a rule the code never applied.
        if !is_relay && entity == WILDCARD_ENTITY {
            return Err(Status::permission_denied(
                "reading the wildcard entity '*' requires the service-proxy role",
            ));
        }
        if !is_relay && principal.as_deref() != Some(entity.as_str()) {
            return Err(Status::permission_denied(
                "credential-reader may only read its own principal's credentials; \
                 reading another entity (or the wildcard '*') requires the service-proxy role",
            ));
        }

        // A relay reads on somebody's behalf and needs the hierarchical view —
        // the tenant-wide defaults with the entity's own entries overlaid,
        // which is what gets injected. Anyone else gets only what belongs to
        // the principal it just proved it is. Merging the defaults in for
        // them made the wildcard check above decorative: asking for your own
        // credentials returned the tenant's shared platform secrets
        // regardless.
        let view = if is_relay {
            CredentialView::Relay
        } else {
            CredentialView::Own
        };
        let resolved = self
            .source
            .resolve(&tenant_id, &entity, &service, view)
            .await
            .map_err(|e| Status::internal(format!("credential lookup failed: {e}")))?;

        let credentials = resolved
            .values
            .into_iter()
            .map(|(k, v)| Credential {
                key: Some(k),
                value: Some(v),
            })
            .collect();

        Ok(Response::new(Credentials { credentials }))
    }

    async fn set_credentials(
        &self,
        request: Request<SetCredentialsRequest>,
    ) -> Result<Response<CredentialResponse>, Status> {
        sasy_auth::check_request_role(&request, sasy_common::roles::CREDENTIAL_WRITER)?;
        let tenant_id = sasy_auth::request_tenant(&request, "default");
        let req = request.into_inner();

        let entity = req.entity.unwrap_or_default();
        let service = req.service.unwrap_or_default();

        if entity.is_empty() || service.is_empty() {
            return Ok(Response::new(CredentialResponse {
                response: Some("Missing fields.".to_string()),
            }));
        }

        let pairs: Vec<(String, String)> = req
            .credentials
            .into_iter()
            .map(|c| (c.key.unwrap_or_default(), c.value.unwrap_or_default()))
            .collect();

        self.source
            .set_credentials(&tenant_id, &entity, &service, pairs)
            .await
            .map_err(write_status)?;

        Ok(Response::new(CredentialResponse {
            response: Some("Successfully saved credentials.".to_string()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::credential_server::credential_server_server::CredentialServer as _;

    use crate::memory::MemorySource;
    use crate::source::{Freshness, ResolvedCredentials};

    #[tokio::test]
    async fn authenticated_tenant_writes_keep_other_tenants_seed_fallback() {
        let source = Arc::new(MemorySource::new().unwrap());
        source
            .seed([("OPENAI_API_KEY".into(), "deployment-default".into())])
            .unwrap();
        let svc = CredentialService::new(source);
        let mut write = Request::new(SetCredentialsRequest {
            entity: Some(WILDCARD_ENTITY.into()),
            service: Some("openai".into()),
            credentials: vec![Credential {
                key: Some("api_key".into()),
                value: Some("tenant-default".into()),
            }],
        });
        write.extensions_mut().insert(
            sasy_auth::AuthResult::success(
                "writer",
                vec![sasy_common::roles::CREDENTIAL_WRITER.into()],
                "test",
            )
            .with_tenant("default"),
        );
        svc.set_credentials(write).await.unwrap();
        for (tenant, expected) in [
            ("default", "tenant-default"),
            ("acme", "deployment-default"),
        ] {
            let mut read = Request::new(CredentialsRequest {
                entity: Some(WILDCARD_ENTITY.into()),
                service: Some("openai".into()),
            });
            read.extensions_mut().insert(
                sasy_auth::AuthResult::success(
                    "relay",
                    vec![
                        sasy_common::roles::CREDENTIAL_READER.into(),
                        sasy_common::roles::SERVICE_PROXY.into(),
                    ],
                    "test",
                )
                .with_tenant(tenant),
            );
            let credentials = svc
                .get_credentials(read)
                .await
                .unwrap()
                .into_inner()
                .credentials;
            assert_eq!(credentials.len(), 1);
            assert_eq!(credentials[0].value.as_deref(), Some(expected));
        }
    }

    /// A principal literally named `*` must not collect the tenant's shared
    /// secrets.
    ///
    /// The wildcard is the key the tenant-wide platform credentials live
    /// under, and the only rule confining a plain `credential-reader` was
    /// `principal == entity`. An entity named `*` — or a token whose subject
    /// claim is that string — satisfies it, so the reader asked for `*`,
    /// matched itself, and was handed every tenant-wide secret. The role gate
    /// the error message describes has to be applied, not merely described.
    #[tokio::test]
    async fn a_principal_named_star_cannot_read_the_wildcard_entity() {
        let source = Arc::new(MemorySource::new().unwrap());
        source
            .set_credentials(
                "acme",
                WILDCARD_ENTITY,
                "openai",
                vec![("platform_key".into(), "sk-shared".into())],
            )
            .await
            .unwrap();
        let svc = CredentialService::new(source);

        let request = |roles: Vec<String>| {
            let mut req = Request::new(CredentialsRequest {
                entity: Some(WILDCARD_ENTITY.to_string()),
                service: Some("openai".to_string()),
            });
            req.extensions_mut().insert(
                sasy_auth::AuthResult::success(WILDCARD_ENTITY, roles, "test").with_tenant("acme"),
            );
            req
        };

        let err = svc
            .get_credentials(request(vec![
                sasy_common::roles::CREDENTIAL_READER.to_string()
            ]))
            .await
            .expect_err("a reader named '*' must not read the wildcard entity");
        assert_eq!(err.code(), tonic::Code::PermissionDenied, "got: {err}");

        // The relay still can — that is the injection path, and it is what
        // the role exists to gate.
        let ok = svc
            .get_credentials(request(vec![
                sasy_common::roles::CREDENTIAL_READER.to_string(),
                sasy_common::roles::SERVICE_PROXY.to_string(),
            ]))
            .await
            .expect("a service-proxy relay reads the wildcard");
        assert_eq!(ok.into_inner().credentials.len(), 1);
    }

    /// A backend that does not accept writes, standing in for OpenBao without
    /// needing a server to talk to.
    struct ReadOnlySource;

    #[tonic::async_trait]
    impl CredentialSource for ReadOnlySource {
        fn backend(&self) -> &'static str {
            "openbao"
        }
        fn location(&self) -> String {
            "https://bao.example.com:8200".to_string()
        }
        async fn resolve(
            &self,
            _tenant: &str,
            _entity: &str,
            _service: &str,
            _view: CredentialView,
        ) -> Result<ResolvedCredentials, CredentialError> {
            Ok(ResolvedCredentials {
                values: Default::default(),
                freshness: Freshness::Until(std::time::Instant::now()),
            })
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
            Err(CredentialError::ReadOnly(format!(
                "the {} credential backend at {} is read-only",
                self.backend(),
                self.location()
            )))
        }
    }

    /// Writing to a read-only backend is a refused precondition that names
    /// the backend, not an internal error.
    ///
    /// The distinction is what an operator needs: the write did not fail, it
    /// was refused, and the message says where the credential actually
    /// belongs.
    #[tokio::test]
    async fn setting_a_credential_on_a_read_only_backend_is_refused_by_name() {
        let svc = CredentialService::new(Arc::new(ReadOnlySource));
        let mut req = Request::new(SetCredentialsRequest {
            entity: Some("agent".to_string()),
            service: Some("openai".to_string()),
            credentials: vec![Credential {
                key: Some("api_key".to_string()),
                value: Some("sk-nope".to_string()),
            }],
        });
        req.extensions_mut().insert(
            sasy_auth::AuthResult::success(
                "admin",
                vec![sasy_common::roles::CREDENTIAL_WRITER.to_string()],
                "test",
            )
            .with_tenant("acme"),
        );

        let err = svc
            .set_credentials(req)
            .await
            .expect_err("a read-only backend must refuse the write");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition, "got: {err}");
        assert!(
            err.message().contains("openbao") && err.message().contains("bao.example.com"),
            "the refusal must name the backend and where it reads from: {}",
            err.message()
        );
    }
}
