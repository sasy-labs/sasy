//! gRPC RMProxy service implementation.

use std::sync::Arc;

#[cfg(feature = "proxy")]
use reqwest::Client;
use sasy_auth::{get_auth_result, AuthConfig};
use sasy_common::policy_engine;
use sasy_common::services::{
    rm_proxy_server::RmProxy, HttpResponse, ToolCallRequest, ToolCallResponse,
};
#[cfg(feature = "proxy")]
use sasy_common::services::{BaseRequest, HttpHeader, HttpRequest, Message};
#[cfg(feature = "proxy")]
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;
#[cfg(feature = "proxy")]
use tracing::warn;

#[cfg(feature = "proxy")]
use crate::net_guard::{redact_url, resolve_public_target};
use crate::policy::PolicyChecker;
#[cfg(feature = "proxy")]
use crate::transforms::TransformExecutor;

/// Create a proto `HttpHeader` from a key and value bytes.
#[cfg(feature = "proxy")]
fn make_header(key: &str, value: &[u8]) -> HttpHeader {
    HttpHeader {
        key: Some(key.as_bytes().to_vec()),
        value: Some(value.to_vec()),
    }
}

/// gRPC reference monitor service.
pub struct RefmonService<P: PolicyChecker + 'static> {
    policy_checker: Arc<P>,
    #[cfg(feature = "proxy")]
    transform_executor: Arc<TransformExecutor>,
    auth_config: Option<AuthConfig>,
    #[cfg(feature = "proxy")]
    http_client: Client,
    /// When true, accept `x-entity`/`x-roles` from any caller.
    /// Use only when the RM is behind a trusted gateway (e.g., same container).
    trust_proxy: bool,
}

impl<P: PolicyChecker + 'static> RefmonService<P> {
    #[cfg(feature = "proxy")]
    pub fn new(policy_checker: Arc<P>, transform_executor: Arc<TransformExecutor>) -> Self {
        Self {
            policy_checker,
            transform_executor,
            auth_config: None,
            // Redirects disabled (see do_proxy): this is the fallback client
            // used when the per-request pinned client fails to build, so it
            // must be just as conservative about not following 3xx.
            http_client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(std::time::Duration::from_secs(10))
                .read_timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("static reqwest client config is valid"),
            trust_proxy: false,
        }
    }

    /// Restricted build: the reference monitor serves only `CheckToolCall`
    /// (the enforcement entry point for tool calls). The credential-injecting proxy
    /// is compiled out, so there is no transform executor or HTTP client.
    #[cfg(not(feature = "proxy"))]
    pub fn new(policy_checker: Arc<P>) -> Self {
        Self {
            policy_checker,
            auth_config: None,
            trust_proxy: false,
        }
    }

    pub fn with_auth_config(mut self, config: AuthConfig) -> Self {
        self.auth_config = Some(config);
        self
    }

    /// Trust `x-entity`/`x-roles` metadata from any caller, regardless
    /// of their role. Use ONLY when the RM is not externally exposed
    /// (e.g., same-container deployment behind a trusted gateway).
    pub fn trust_proxy_headers(mut self) -> Self {
        self.trust_proxy = true;
        self
    }

    /// Extract the auth-derived principal and its roles from the request.
    ///
    /// The returned principal is server-stamped from the connection's
    /// authentication chain (mTLS subject / JWT sub / API key →
    /// auth_config.yaml mapping); it is the value the policy engine
    /// surfaces as `Principal(id)` and the basis for `HasPrincipal()` /
    /// `HasRole(...)`.
    ///
    /// Supports **delegated identity**: if the authenticated caller has the
    /// `service-proxy` role, it may pass the end user's identity/roles via
    /// `x-entity` and `x-roles` gRPC metadata. This is the trusted-proxy
    /// pattern used by an API gateway — the delegated value
    /// becomes the principal for downstream policy evaluation, since the
    /// proxy is vouching for it.
    ///
    /// Without `service-proxy`, the caller's own identity is used (normal path).
    /// Resolve `(principal, roles, tenant)` for this request.
    ///
    /// Tenant follows the *resolved* identity: under
    /// `service-proxy` delegation the tenant comes from the
    /// delegated end-user's `auth_config` mapping, not from the
    /// proxy's own tenant. Without that, delegated users would run
    /// in the proxy's tenant (typically `default`) and policy /
    /// graph state would land in the wrong shard.
    fn extract_auth<T>(
        &self,
        request: &tonic::Request<T>,
    ) -> (Option<String>, Vec<String>, String) {
        let fallback_tenant = sasy_auth::request_tenant(request, "default");
        if let Some(auth) = get_auth_result(request) {
            // Check for delegated identity (trusted proxy).
            // Allowed if: caller has service-proxy role, OR trust_proxy is set.
            if self.trust_proxy
                || auth
                    .roles
                    .iter()
                    .any(|r| r == sasy_common::roles::SERVICE_PROXY)
            {
                let meta = request.metadata();
                let delegated_entity = meta
                    .get(sasy_common::headers::ENTITY)
                    .and_then(|v| v.to_str().ok())
                    .map(String::from);
                let delegated_roles = meta
                    .get(sasy_common::headers::ROLES)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| {
                        s.split(',')
                            .map(|r| r.trim().to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                if let Some(ref entity) = delegated_entity {
                    // Use delegated identity, enriched with central role config
                    let mut roles = delegated_roles;
                    let mut tenant = fallback_tenant.clone();
                    if let Some(ref config) = self.auth_config {
                        let central = config.get_roles(entity);
                        if !central.is_empty() {
                            roles = central;
                        }
                        // Map the delegated entity to its declared
                        // tenant. Without this, every delegated
                        // user inherits the proxy's tenant — which
                        // collapses multi-tenant deployments to a
                        // single shard at the proxy boundary.
                        tenant = config.get_tenant(entity);
                    }
                    info!(
                        proxy = auth.entity.as_deref().unwrap_or("?"),
                        delegated_entity = entity,
                        delegated_tenant = %tenant,
                        "Delegated identity from service-proxy"
                    );
                    return (Some(entity.clone()), roles, tenant);
                }
            }

            // Normal path: use the caller's own identity
            let entity = auth.entity.clone();
            let mut roles = auth.roles.clone();
            if let (Some(ref entity), Some(ref config)) = (&entity, &self.auth_config) {
                let central_roles = config.get_roles(entity);
                if !central_roles.is_empty() {
                    roles = central_roles;
                }
            }
            (entity, roles, fallback_tenant)
        } else {
            (None, vec![], fallback_tenant)
        }
    }

    #[cfg(feature = "proxy")]
    fn request_to_action(req: &BaseRequest) -> policy_engine::Action {
        // Canonicalize the URL so the policy decision is made on the SAME
        // host the proxy will actually connect to. The engine extracts the
        // host from this string (`@url_host`/`QueriesHost`) while the fetch
        // path parses it with the `url` crate; feeding the normalized form
        // to the policy closes the parser-divergence (TOCTOU) gap where e.g.
        // backslashes or userinfo (`http://allowed\@evil/`) make a naive host
        // extractor and the real client disagree. If it doesn't parse, leave
        // it raw — the fetch path rejects unparseable URLs anyway.
        let raw_url = req.url.clone().unwrap_or_default();
        let canonical_url = reqwest::Url::parse(&raw_url)
            .map(|u| u.to_string())
            .unwrap_or(raw_url);
        policy_engine::Action {
            action_type: Some(policy_engine::action::ActionType::HttpRequest(
                policy_engine::HttpRequestAction {
                    url: canonical_url,
                    body: req
                        .message
                        .as_ref()
                        .and_then(|m| m.content.as_ref())
                        .map(|c| String::from_utf8_lossy(c).to_string())
                        .unwrap_or_default(),
                    headers: req
                        .message
                        .as_ref()
                        .map(|m| {
                            m.headers
                                .iter()
                                .filter_map(|h| match (&h.key, &h.value) {
                                    (Some(k), Some(v)) => Some(policy_engine::Header {
                                        key: String::from_utf8_lossy(k).to_string(),
                                        value: String::from_utf8_lossy(v).to_string(),
                                    }),
                                    _ => None,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                },
            )),
            ..Default::default()
        }
    }

    fn tool_call_to_action(req: &ToolCallRequest) -> policy_engine::Action {
        policy_engine::Action {
            action_type: Some(policy_engine::action::ActionType::ToolCall(
                policy_engine::ToolCallAction {
                    fn_name: req.fn_name.clone(),
                    args: req.args.clone(),
                },
            )),
            // Forward caller-resolved per-action metadata verbatim → the
            // policy engine projects it into `ActionMetadata(idx, rel, a, b)`.
            metadata: req.metadata.clone(),
        }
    }

    #[cfg(feature = "proxy")]
    fn denial_response(result: &policy_engine::ActionResult) -> HttpResponse {
        let message = if let Some(ref trace) = result.trace {
            let mut msg = trace.action_description.clone();
            if !trace.reasons.is_empty() {
                let parts: Vec<&str> = trace
                    .reasons
                    .iter()
                    .map(|r| {
                        if r.details.is_empty() {
                            "denied"
                        } else {
                            r.details.as_str()
                        }
                    })
                    .collect();
                msg.push_str(": ");
                msg.push_str(&parts.join("; "));
            }
            for fix in &trace.suggested_fixes {
                msg.push_str("\nSuggestion: ");
                msg.push_str(fix);
            }
            msg
        } else {
            "Authorization denied".to_string()
        };
        HttpResponse {
            status: Some(403),
            message: Some(Message {
                headers: vec![],
                content: Some(message.into_bytes()),
            }),
        }
    }

    /// The response for a request whose credential transform could not be
    /// applied.
    ///
    /// Same shape as the other refusals on this path — a 403 and a reason —
    /// because the outcome is the same: nothing is sent upstream. The reason
    /// is fixed text: the backend's own message names the secret manager's
    /// address, its mount and the path it tried to read, so it goes to the
    /// server log at `warn` level and never to the caller.
    #[cfg(feature = "proxy")]
    fn transform_failure_response() -> HttpResponse {
        HttpResponse {
            status: Some(403),
            message: Some(Message {
                headers: vec![],
                content: Some(
                    crate::transforms::TRANSFORM_FAILURE_REASON
                        .as_bytes()
                        .to_vec(),
                ),
            }),
        }
    }

    /// Core proxy logic shared between ProxyHTTP (unary) and ProxyHTTPBidi (streaming).
    ///
    /// - `first_msg`: the initial HTTPRequest with URL + method + headers + optional first body chunk
    /// - `body_stream`: optional stream of additional body chunks (None for unary requests)
    /// - `principal`/`roles`: auth-derived (from the gRPC interceptor); stamped onto the policy request
    /// - `tx`: channel to send response messages
    ///
    /// User-supplied `entity` is read off the wire from `first_msg.entity`
    /// and forwarded verbatim — it is *not* trusted for authorization.
    #[cfg(feature = "proxy")]
    async fn proxy_core(
        &self,
        first_msg: HttpRequest,
        body_stream: Option<tonic::Streaming<HttpRequest>>,
        principal: Option<String>,
        roles: Vec<String>,
        tenant: String,
        tx: mpsc::Sender<Result<HttpResponse, tonic::Status>>,
    ) {
        let base_req = match first_msg.request {
            Some(r) => r,
            None => {
                let _ = tx
                    .send(Ok(HttpResponse {
                        status: Some(400),
                        message: Some(Message {
                            headers: vec![],
                            content: Some(b"missing request".to_vec()),
                        }),
                    }))
                    .await;
                return;
            }
        };
        let input_node_ids = first_msg.input_node_ids;
        let wire_entity = first_msg.entity.clone();

        // The caller supplies this URL verbatim, so it can carry a key in its
        // query, its fragment or its userinfo before any transform runs. Log
        // the redacted form — scheme, host, port and path is what the line is
        // for.
        let url_display = base_req
            .url
            .as_deref()
            .map(redact_url)
            .unwrap_or_else(|| "(none)".to_string());
        info!(
            url = %url_display,
            principal = principal.as_deref().unwrap_or("(anon)"),
            entity = wire_entity.as_deref().unwrap_or("(none)"),
            "ProxyHTTP"
        );

        // Policy check
        let action = Self::request_to_action(&base_req);
        // session_id is user-supplied; tenant is the auth-derived
        // value passed in by the RPC handler. Together they
        // form the typed `(tenant, session)` partition the policy
        // engine evaluates against. An empty session_id maps to the
        // tenant's global partition.
        let scope = sasy_common::SessionScope::new(
            &tenant,
            first_msg.session_id.clone().unwrap_or_default(),
        );
        // Sessions are bound explicitly via SetPolicy now —
        // no inline policy_id on the auth request.
        let auth_resp = match self
            .policy_checker
            .check_authorization(
                &input_node_ids,
                vec![action],
                wire_entity.as_deref(),
                &roles,
                scope,
                principal.as_deref(),
                None,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(tonic::Status::from(e))).await;
                return;
            }
        };

        let action_result = auth_resp.results.into_iter().next();
        let transform_executor = Arc::clone(&self.transform_executor);
        let http_client = self.http_client.clone();
        // Credentials are keyed on the auth-derived principal (the
        // entity in `auth_config.yaml`'s entity → roles/credentials
        // map), not on the user-supplied wire entity.
        let principal_for_transforms = principal.clone();

        match action_result {
            None => {
                info!("DENY (no policy result)");
                let resp = Self::denial_response(&policy_engine::ActionResult {
                    index: 0,
                    authorized: false,
                    trace: None,
                    transform_ids: vec![],
                    deny_if_unauthorized: true,
                });
                let _ = tx.send(Ok(resp)).await;
            }
            Some(ar) if !ar.authorized => {
                if ar.deny_if_unauthorized {
                    info!("DENY (unauthorized)");
                    let _ = tx.send(Ok(Self::denial_response(&ar))).await;
                } else {
                    info!("PASSTHROUGH");
                    Self::do_proxy(http_client, base_req, vec![], body_stream, tx).await;
                }
            }
            Some(ar) => {
                info!(transforms = ?ar.transform_ids, "AUTHORIZED");
                match Self::transformed_request(
                    &transform_executor,
                    &base_req,
                    &ar.transform_ids,
                    &tenant,
                    principal_for_transforms.as_deref(),
                )
                .await
                {
                    Ok((req_out, extra_headers)) => {
                        Self::do_proxy(http_client, req_out, extra_headers, body_stream, tx).await;
                    }
                    Err(refusal) => {
                        let _ = tx.send(Ok(refusal)).await;
                    }
                }
            }
        }
    }

    /// Inject the credentials the policy attached and return the request to
    /// proxy: the post-transform URL, and the headers to add to it.
    ///
    /// Split from [`Self::proxy_core`] so a test can see exactly what would go
    /// upstream — the URL the credential is written onto included — without a
    /// network round trip. An `Err` is the refusal the caller gets, and the
    /// request never reaches [`Self::do_proxy`].
    #[cfg(feature = "proxy")]
    async fn transformed_request(
        transform_executor: &TransformExecutor,
        base_req: &BaseRequest,
        transform_ids: &[String],
        tenant: &str,
        principal: Option<&str>,
    ) -> Result<(BaseRequest, Vec<(String, String)>), HttpResponse> {
        let mut extra_headers = vec![];
        let mut url = base_req.url.clone().unwrap_or_default();

        if let Err(e) = transform_executor
            .execute(
                &mut extra_headers,
                &mut url,
                transform_ids,
                tenant,
                principal,
            )
            .await
        {
            // Fail closed. A transform that could not be applied means the
            // upstream credential is missing — or that the leg it would ride
            // on is not TLS — and proxying anyway sends the request
            // unauthenticated, under whatever identity the caller supplied
            // instead, or in the clear. Nothing goes upstream.
            warn!(error = %e, "credential transform failed; denying");
            return Err(Self::transform_failure_response());
        }

        let mut req_out = base_req.clone();
        req_out.url = Some(url);
        Ok((req_out, extra_headers))
    }

    /// Build headers for the same authority the policy and network guard vetted.
    #[cfg(feature = "proxy")]
    fn forward_headers(
        mut builder: reqwest::RequestBuilder,
        message: Option<&Message>,
        extra_headers: &[(String, String)],
    ) -> reqwest::RequestBuilder {
        let override_keys: std::collections::HashSet<String> = extra_headers
            .iter()
            .map(|(k, _)| k.to_ascii_lowercase())
            .collect();
        if let Some(msg) = message {
            for h in &msg.headers {
                if let (Some(k), Some(v)) = (&h.key, &h.value) {
                    let k_str = String::from_utf8_lossy(k);
                    if !crate::net_guard::forward_request_header(&k_str)
                        || override_keys.contains(&k_str.to_ascii_lowercase())
                    {
                        continue;
                    }
                    builder = builder.header(k_str.as_ref(), String::from_utf8_lossy(v).as_ref());
                }
            }
        }
        for (k, v) in extra_headers {
            if crate::net_guard::forward_request_header(k) {
                builder = builder.header(k.as_str(), v.as_str());
            }
        }

        builder
    }

    /// Execute the upstream HTTP request and stream the response.
    ///
    /// If `body_stream` is Some, additional body chunks from the client
    /// are piped to the upstream (for bidirectional streaming).
    #[cfg(feature = "proxy")]
    async fn do_proxy(
        http_client: Client,
        base_req: BaseRequest,
        extra_headers: Vec<(String, String)>,
        body_stream: Option<tonic::Streaming<HttpRequest>>,
        tx: mpsc::Sender<Result<HttpResponse, tonic::Status>>,
    ) {
        use futures_util::StreamExt;

        let url = base_req.url.as_deref().unwrap_or_default();
        let method_str = base_req.method.as_deref().unwrap_or("GET").to_uppercase();
        let method: reqwest::Method = match method_str.parse() {
            Ok(m) => m,
            Err(_) => {
                let _ = tx
                    .send(Ok(HttpResponse {
                        status: Some(400),
                        message: Some(Message {
                            headers: vec![],
                            content: Some(b"bad method".to_vec()),
                        }),
                    }))
                    .await;
                return;
            }
        };

        // SSRF / DNS-rebinding guard: resolve the target once, reject internal
        // addresses, and pin the validated IP for the actual connection.
        let (pin_host, pin_addr) = match resolve_public_target(url).await {
            Ok(v) => v,
            Err(reason) => {
                warn!(url = %redact_url(url), reason = %reason, "proxy target rejected (SSRF guard)");
                let _ = tx
                    .send(Ok(HttpResponse {
                        status: Some(403),
                        message: Some(Message {
                            headers: vec![],
                            content: Some(format!("blocked: {reason}").into_bytes()),
                        }),
                    }))
                    .await;
                return;
            }
        };

        // Pinned to the validated IP via the shared SSRF-hardened builder (see
        // [`crate::net_guard::pinned_client_builder`]); on the rare build
        // failure fall back to the conservative base client.
        let raw_client = crate::net_guard::pinned_client_builder(&pin_host, pin_addr)
            .build()
            .unwrap_or(http_client);

        let mut builder = raw_client.request(method, url);

        builder = Self::forward_headers(builder, base_req.message.as_ref(), &extra_headers);

        // Request body: first chunk from the initial message
        let first_body = base_req
            .message
            .as_ref()
            .and_then(|m| m.content.as_ref())
            .cloned()
            .unwrap_or_default();

        if let Some(body_stream_inner) = body_stream {
            // Bidirectional: pipe first chunk + subsequent chunks via a streaming body
            let (body_tx, body_rx) =
                tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);

            // Spawn task to pipe client body chunks
            tokio::spawn(async move {
                // Send first chunk
                if !first_body.is_empty()
                    && body_tx
                        .send(Ok(bytes::Bytes::from(first_body)))
                        .await
                        .is_err()
                {
                    return;
                }
                // Pipe subsequent chunks
                let mut stream = body_stream_inner;
                while let Some(Ok(msg)) = stream.next().await {
                    if let Some(ref req) = msg.request {
                        if let Some(ref m) = req.message {
                            if let Some(ref content) = m.content {
                                if !content.is_empty()
                                    && body_tx
                                        .send(Ok(bytes::Bytes::from(content.clone())))
                                        .await
                                        .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            });

            let body =
                reqwest::Body::wrap_stream(tokio_stream::wrappers::ReceiverStream::new(body_rx));
            builder = builder.body(body);
        } else if !first_body.is_empty() {
            // Unary: single body chunk
            builder = builder.body(first_body);
        }

        // Send request
        let resp = match builder.send().await {
            Ok(r) => r,
            Err(e) => {
                // `without_url()` drops the (post-transform, possibly
                // credential-laden) URL from the reqwest error before logging;
                // the caller gets a generic body, not the raw error string.
                warn!(error = %e.without_url(), "upstream error");
                let _ = tx
                    .send(Ok(HttpResponse {
                        status: Some(502),
                        message: Some(Message {
                            headers: vec![],
                            content: Some(b"upstream request failed".to_vec()),
                        }),
                    }))
                    .await;
                return;
            }
        };

        let status = resp.status().as_u16() as u32;
        let resp_headers: Vec<_> = resp
            .headers()
            .iter()
            .map(|(k, v)| make_header(k.as_str(), v.as_bytes()))
            .collect();

        info!(url = %redact_url(url), status, "streaming response");

        // First response: status + headers, no body (client uses Stream for body)
        if tx
            .send(Ok(HttpResponse {
                status: Some(status),
                message: Some(Message {
                    headers: resp_headers,
                    content: None,
                }),
            }))
            .await
            .is_err()
        {
            return;
        }

        // Stream response body
        let mut stream = resp.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if chunk.is_empty() {
                        continue;
                    }
                    if tx
                        .send(Ok(HttpResponse {
                            status: None,
                            message: Some(Message {
                                headers: vec![],
                                content: Some(chunk.to_vec()),
                            }),
                        }))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(e) => {
                    warn!(error = %e.without_url(), "stream error");
                    break;
                }
            }
        }
    }
}

#[tonic::async_trait]
impl<P: PolicyChecker + 'static> RmProxy for RefmonService<P> {
    type ProxyHTTPStream = ReceiverStream<Result<HttpResponse, tonic::Status>>;

    #[cfg(feature = "proxy")]
    async fn proxy_http(
        &self,
        request: tonic::Request<tonic::Streaming<HttpRequest>>,
    ) -> Result<tonic::Response<Self::ProxyHTTPStream>, tonic::Status> {
        use tokio_stream::StreamExt;

        sasy_auth::check_request_role(&request, sasy_common::roles::REFERENCE_MONITOR_USER)?;
        let (principal, roles, tenant) = self.extract_auth(&request);
        let mut client_stream = request.into_inner();

        // First message: URL + method + headers + optional first body chunk
        let first_msg = client_stream
            .next()
            .await
            .ok_or_else(|| tonic::Status::invalid_argument("empty stream"))?
            .map_err(|e| tonic::Status::internal(format!("stream error: {}", e)))?;

        let (tx, rx) = mpsc::channel(16);
        let svc = RefmonService {
            policy_checker: self.policy_checker.clone(),
            transform_executor: self.transform_executor.clone(),
            auth_config: self.auth_config.clone(),
            http_client: self.http_client.clone(),
            trust_proxy: self.trust_proxy,
        };

        tokio::spawn(async move {
            // Pass remaining stream for potential body chunks
            svc.proxy_core(first_msg, Some(client_stream), principal, roles, tenant, tx)
                .await;
        });

        Ok(tonic::Response::new(ReceiverStream::new(rx)))
    }

    /// Without the proxy feature, keep RmProxy registered for CheckToolCall
    /// while returning Unimplemented for HTTP forwarding.
    #[cfg(not(feature = "proxy"))]
    async fn proxy_http(
        &self,
        _request: tonic::Request<tonic::Streaming<sasy_common::services::HttpRequest>>,
    ) -> Result<tonic::Response<Self::ProxyHTTPStream>, tonic::Status> {
        Err(tonic::Status::unimplemented(
            "forward proxy is not available in this build",
        ))
    }

    async fn check_tool_call(
        &self,
        request: tonic::Request<ToolCallRequest>,
    ) -> Result<tonic::Response<ToolCallResponse>, tonic::Status> {
        sasy_auth::check_request_role(&request, sasy_common::roles::REFERENCE_MONITOR_USER)?;
        let (principal, roles, tenant) = self.extract_auth(&request);
        let tool_req = request.into_inner();
        let input_node_ids = tool_req.input_node_ids.clone();
        // User-supplied per-request actor, preserved verbatim from the
        // wire. Distinct from the auth-derived principal extracted above.
        let wire_entity = tool_req.entity.clone();

        info!(
            fn_name = tool_req.fn_name,
            principal = principal.as_deref().unwrap_or("(anon)"),
            entity = wire_entity.as_deref().unwrap_or("(none)"),
            "CheckToolCall"
        );

        let action = Self::tool_call_to_action(&tool_req);
        let scope = sasy_common::SessionScope::new(tenant, tool_req.session_id.unwrap_or_default());
        let auth_resp = self
            .policy_checker
            .check_authorization(
                &input_node_ids,
                vec![action],
                wire_entity.as_deref(),
                &roles,
                scope,
                principal.as_deref(),
                None,
            )
            .await
            .map_err(|e| -> tonic::Status { e.into() })?;

        let action_result = auth_resp.results.into_iter().next();
        let (authorized, denial_trace, transform_ids) = match action_result {
            None => {
                info!("DENY {} (no result)", tool_req.fn_name);
                (false, None, Vec::new())
            }
            Some(ar) => {
                if ar.authorized {
                    info!(transforms = ?ar.transform_ids, "AUTHORIZED {}", tool_req.fn_name);
                } else {
                    info!("DENY {}", tool_req.fn_name);
                }
                // Carry the policy's ApplyTransform ids through so the caller
                // (a hook or SDK) can run credential injection.
                (ar.authorized, ar.trace, ar.transform_ids)
            }
        };

        Ok(tonic::Response::new(ToolCallResponse {
            authorized,
            denial_trace,
            transform_ids,
        }))
    }
}

#[cfg(all(test, feature = "proxy"))]
mod tests {
    use super::*;
    use sasy_common::policy_engine::{ActionResult, AuthorizationResponse};
    use sasy_credential::{CredentialSource, MemorySource};

    use crate::test_support::{
        injecting_transform_config, AllowAllWithTransform, DisclosingSource, UnreachableSource,
        DISCLOSING_BACKEND_MESSAGE,
    };
    use crate::transforms::{TransformConfig, TransformExecutor};
    use crate::RefmonError;

    struct AllowAll;

    impl PolicyChecker for AllowAll {
        async fn check_authorization(
            &self,
            _current_node_ids: &[String],
            actions: Vec<policy_engine::Action>,
            _entity: Option<&str>,
            _roles: &[String],
            _scope: sasy_common::SessionScope,
            _principal: Option<&str>,
            _policy_id: Option<String>,
        ) -> Result<AuthorizationResponse, RefmonError> {
            let results = actions
                .iter()
                .enumerate()
                .map(|(i, _)| ActionResult {
                    index: i as u32,
                    authorized: true,
                    trace: None,
                    transform_ids: vec![],
                    deny_if_unauthorized: false,
                })
                .collect();
            Ok(AuthorizationResponse {
                results,
                timing: None,
            })
        }
    }

    fn make_service() -> RefmonService<AllowAll> {
        let source = Arc::new(MemorySource::new().unwrap());
        let config = TransformConfig::default();
        let executor = Arc::new(TransformExecutor::new(config, source));
        RefmonService::new(Arc::new(AllowAll), executor)
    }

    #[test]
    fn grpc_proxy_uses_url_authority_and_preserves_transformed_headers() {
        let message = Message {
            headers: vec![
                make_header("Host", b"original.example.com"),
                make_header("Proxy-Connection", b"keep-alive"),
                make_header("Authorization", b"original"),
                make_header("Accept", b"application/json"),
            ],
            content: None,
        };
        let request = RefmonService::<AllowAll>::forward_headers(
            Client::new().get("https://api.example.com/path"),
            Some(&message),
            &[("Authorization".into(), "transformed".into())],
        )
        .build()
        .unwrap();
        assert_eq!(request.url().host_str(), Some("api.example.com"));
        assert!(!request.headers().contains_key("host"));
        assert!(!request.headers().contains_key("proxy-connection"));
        assert_eq!(request.headers()["authorization"], "transformed");
        assert_eq!(request.headers()["accept"], "application/json");
        assert_eq!(request.headers().get_all("authorization").iter().count(), 1);
    }

    /// Deny a request whose credential cannot be fetched. The documentation-only
    /// address 192.0.2.1 ensures an accidental network attempt cannot succeed.
    #[tokio::test]
    async fn a_credential_that_cannot_be_fetched_denies_the_request() {
        let executor = Arc::new(TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(UnreachableSource) as Arc<dyn CredentialSource>,
        ));
        let svc = RefmonService::new(Arc::new(AllowAllWithTransform), executor);

        let (tx, mut rx) = mpsc::channel(4);
        svc.proxy_core(
            sasy_common::services::HttpRequest {
                request: Some(BaseRequest {
                    url: Some("https://192.0.2.1/v1/chat".to_string()),
                    method: Some("GET".to_string()),
                    message: None,
                }),
                input_node_ids: vec![],
                session_id: None,
                entity: None,
            },
            None,
            Some("alice".to_string()),
            vec![],
            "default".to_string(),
            tx,
        )
        .await;

        let resp = rx
            .recv()
            .await
            .expect("the caller must get a response")
            .expect("a denial is a response, not a stream error");
        assert_eq!(resp.status, Some(403));
        let body = String::from_utf8(
            resp.message
                .and_then(|m| m.content)
                .expect("the denial carries a reason"),
        )
        .unwrap();
        assert!(
            body == crate::transforms::TRANSFORM_FAILURE_REASON,
            "the reason must say the credential is what failed: {body}"
        );
        assert!(
            rx.recv().await.is_none(),
            "nothing follows the denial — the request was never proxied"
        );
    }

    /// The ProxyHTTP refusal carries no deployment detail either.
    ///
    /// Same property as the forward proxy: the backend's message names the
    /// secret manager's address, mount and path, and the caller gets the
    /// fixed reason instead.
    #[tokio::test]
    async fn the_grpc_refusal_does_not_name_the_secret_manager() {
        let executor = Arc::new(TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(DisclosingSource) as Arc<dyn CredentialSource>,
        ));
        let svc = RefmonService::new(Arc::new(AllowAllWithTransform), executor);

        let (tx, mut rx) = mpsc::channel(4);
        svc.proxy_core(
            sasy_common::services::HttpRequest {
                request: Some(BaseRequest {
                    url: Some("https://192.0.2.1/v1/chat".to_string()),
                    method: Some("GET".to_string()),
                    message: None,
                }),
                input_node_ids: vec![],
                session_id: None,
                entity: None,
            },
            None,
            Some("alice".to_string()),
            vec![],
            "default".to_string(),
            tx,
        )
        .await;

        let resp = rx
            .recv()
            .await
            .expect("the caller must get a response")
            .expect("a denial is a response, not a stream error");
        assert_eq!(resp.status, Some(403));
        let body = String::from_utf8(
            resp.message
                .and_then(|m| m.content)
                .expect("the denial carries a reason"),
        )
        .unwrap();
        assert_eq!(body, crate::transforms::TRANSFORM_FAILURE_REASON);
        for detail in ["vault.internal", "8200", "kv/data", "openbao"] {
            assert!(
                !body.contains(detail),
                "the refusal leaks {detail} from the backend message: {body}"
            );
        }
        assert!(
            !body.contains(DISCLOSING_BACKEND_MESSAGE),
            "the backend message must not be forwarded verbatim: {body}"
        );
    }

    /// Deny missing secrets or fields instead of forwarding the caller's
    /// original Authorization header when the required transform cannot run.
    #[tokio::test]
    async fn a_credential_that_resolves_to_nothing_denies_the_request() {
        let executor = Arc::new(TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(MemorySource::new().unwrap()) as Arc<dyn CredentialSource>,
        ));
        let svc = RefmonService::new(Arc::new(AllowAllWithTransform), executor);

        let (tx, mut rx) = mpsc::channel(4);
        svc.proxy_core(
            sasy_common::services::HttpRequest {
                request: Some(BaseRequest {
                    url: Some("https://192.0.2.1/v1/chat".to_string()),
                    method: Some("GET".to_string()),
                    message: Some(Message {
                        headers: vec![make_header("Authorization", b"Bearer caller-supplied")],
                        content: None,
                    }),
                }),
                input_node_ids: vec![],
                session_id: None,
                entity: None,
            },
            None,
            Some("alice".to_string()),
            vec![],
            "default".to_string(),
            tx,
        )
        .await;

        let resp = rx
            .recv()
            .await
            .expect("the caller must get a response")
            .expect("a denial is a response, not a stream error");
        assert_eq!(resp.status, Some(403));
        let body = String::from_utf8(
            resp.message
                .and_then(|m| m.content)
                .expect("the denial carries a reason"),
        )
        .unwrap();
        assert!(
            body == crate::transforms::TRANSFORM_FAILURE_REASON,
            "the reason must say the credential is what failed: {body}"
        );
        assert!(
            rx.recv().await.is_none(),
            "nothing follows the denial — the request was never proxied"
        );
    }

    /// An executor holding a working platform credential for `alice`.
    fn executor_with_credential() -> Arc<TransformExecutor> {
        let source = MemorySource::new().unwrap();
        source
            .store()
            .set_credentials(
                "alice",
                "openai",
                vec![("api_key".into(), "sk-test".into())],
            )
            .unwrap();
        Arc::new(TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(source) as Arc<dyn CredentialSource>,
        ))
    }

    /// The request that carries the credential is made over TLS, even when the
    /// caller asked for plain HTTP.
    ///
    /// A `ProxyHTTP` caller supplies the URL verbatim, and the policy
    /// authorizes on the host alone — so `http://api.openai.com/…` was
    /// authorized like the `https://` form and had the platform key written
    /// onto a plaintext connection. The assertion is on the URL that would be
    /// fetched, not on a network round trip.
    #[tokio::test]
    async fn a_credentialed_plain_http_request_goes_upstream_over_tls() {
        let executor = executor_with_credential();

        let (req_out, extra_headers) = RefmonService::<AllowAll>::transformed_request(
            &executor,
            &BaseRequest {
                url: Some("http://192.0.2.1/v1/chat".to_string()),
                method: Some("GET".to_string()),
                message: None,
            },
            &["inject_key".to_string()],
            "default",
            Some("alice"),
        )
        .await
        .expect("the request must be prepared, not refused");

        assert_eq!(
            req_out.url.as_deref(),
            Some("https://192.0.2.1/v1/chat"),
            "the leg that carries the credential is TLS"
        );
        assert!(
            extra_headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"),
            "the credential was injected"
        );
    }

    /// A request with no transform attached is proxied exactly as it came in,
    /// plain HTTP included — there is no credential to protect.
    #[tokio::test]
    async fn a_request_without_transforms_keeps_plain_http() {
        let executor = executor_with_credential();

        let (req_out, extra_headers) = RefmonService::<AllowAll>::transformed_request(
            &executor,
            &BaseRequest {
                url: Some("http://192.0.2.1/health".to_string()),
                method: Some("GET".to_string()),
                message: None,
            },
            &[],
            "default",
            Some("alice"),
        )
        .await
        .expect("an untransformed request must not be refused");

        assert_eq!(req_out.url.as_deref(), Some("http://192.0.2.1/health"));
        assert!(extra_headers.is_empty());
    }

    /// Plain HTTP on a port that is not the HTTP default is denied rather than
    /// upgraded: whether that port speaks TLS is a guess, and the alternative
    /// is putting the platform credential on the wire in the clear. The request
    /// never reaches `do_proxy`.
    #[tokio::test]
    async fn a_credentialed_plain_http_request_to_another_port_is_denied() {
        let svc = RefmonService::new(Arc::new(AllowAllWithTransform), executor_with_credential());

        let (tx, mut rx) = mpsc::channel(4);
        svc.proxy_core(
            sasy_common::services::HttpRequest {
                request: Some(BaseRequest {
                    url: Some("http://192.0.2.1:8080/v1/chat".to_string()),
                    method: Some("GET".to_string()),
                    message: None,
                }),
                input_node_ids: vec![],
                session_id: None,
                entity: None,
            },
            None,
            Some("alice".to_string()),
            vec![],
            "default".to_string(),
            tx,
        )
        .await;

        let resp = rx
            .recv()
            .await
            .expect("the caller must get a response")
            .expect("a denial is a response, not a stream error");
        assert_eq!(resp.status, Some(403));
        let body = String::from_utf8(
            resp.message
                .and_then(|m| m.content)
                .expect("the denial carries a reason"),
        )
        .unwrap();
        assert_eq!(
            body,
            crate::transforms::TRANSFORM_FAILURE_REASON,
            "the refusal is fixed text; the port it refused to write to is an internal detail that belongs in the server log"
        );
        assert!(
            !body.contains("8080"),
            "the refusal must not describe the upstream leg: {body}"
        );
        assert!(
            !body.contains("sk-test"),
            "no credential value belongs in the refusal: {body}"
        );
        assert!(
            rx.recv().await.is_none(),
            "nothing follows the denial — the request was never proxied"
        );
    }

    #[tokio::test]
    async fn check_tool_call_allows() {
        let svc = make_service();
        // The handler's role check refuses a request that carries no
        // `AuthResult`, so the test attaches the one `AuthInterceptor` would
        // have stamped rather than relying on an absent context passing.
        let mut request = tonic::Request::new(ToolCallRequest {
            fn_name: "test".into(),
            args: "{}".into(),
            input_node_ids: vec![],
            session_id: None,
            entity: None,
            metadata: vec![],
        });
        request
            .extensions_mut()
            .insert(sasy_auth::AuthResult::success(
                "agent",
                vec![sasy_common::roles::REFERENCE_MONITOR_USER.into()],
                "test",
            ));
        let resp = svc.check_tool_call(request).await.unwrap().into_inner();
        assert!(resp.authorized);
    }
}
