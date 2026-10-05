//! HTTP forward proxy with policy enforcement.
//!
//! Accepts standard HTTP proxy requests (absolute-form URLs for plain
//! HTTP, CONNECT for HTTPS tunnels) and runs them through the policy
//! engine. Denied requests get a 403. Allowed plain HTTP requests are
//! forwarded via reqwest. Allowed CONNECT requests establish a
//! bidirectional TCP tunnel.
//!
//! This is a stateless domain-level check — `input_node_ids` is always
//! empty. Useful for integrations where raw `fetch()` calls
//! bypass the gRPC instrumentation path.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

use sasy_common::policy_engine;
use sasy_common::SessionScope;

use crate::policy::PolicyChecker;
use crate::transforms::TransformExecutor;

/// Global entity used for proxy-mode credential injection.
/// The credential server stores platform-wide keys under entity="*".
const PROXY_ENTITY: &str = "*";

/// Hard cap on a single proxied request/response body (16 MiB). Prevents an
/// unbounded in-memory buffer (memory-exhaustion DoS); combined with disabled
/// auto-decompression it also caps decompression bombs.
const MAX_PROXY_BODY: usize = 16 * 1024 * 1024;

/// CONNECT tunnel limits. A raw bidirectional copy is otherwise unbounded in
/// both time and volume — a stalled or high-volume tunnel ties up resources
/// indefinitely. Drop a tunnel that goes idle for `TUNNEL_IDLE_TIMEOUT`, and
/// cap total bytes per direction.
const TUNNEL_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const TUNNEL_MAX_BYTES_PER_DIR: u64 = 512 * 1024 * 1024;

/// Copy one direction of a CONNECT tunnel with an idle timeout + byte budget.
/// Returns when the reader hits EOF/error, goes idle past
/// `TUNNEL_IDLE_TIMEOUT`, or the byte budget is exhausted; then shuts the
/// writer down so the peer sees a clean half-close.
async fn pump<R, W>(mut reader: R, mut writer: W) -> std::io::Result<u64>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = [0u8; 16 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = match tokio::time::timeout(TUNNEL_IDLE_TIMEOUT, reader.read(&mut buf)).await {
            Ok(Ok(0)) | Err(_) => break, // EOF or idle timeout
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
        };
        total += n as u64;
        if total > TUNNEL_MAX_BYTES_PER_DIR {
            warn!(
                "CONNECT tunnel exceeded {} bytes per direction; dropping",
                TUNNEL_MAX_BYTES_PER_DIR
            );
            break;
        }
        writer.write_all(&buf[..n]).await?;
    }
    let _ = writer.shutdown().await;
    Ok(total)
}

/// Configuration for the forward proxy.
pub struct ForwardProxyConfig {
    pub listen_addr: SocketAddr,
    /// Tenant the listener is pinned to. The forward proxy is
    /// auth-less today (raw HTTP, no client identity available),
    /// so the tenant has to be a deployment property of the
    /// listener. Multi-tenant operators run one proxy instance
    /// per tenant; the default value preserves single-tenant
    /// behavior.
    pub tenant: String,
}

impl Default for ForwardProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: ([0, 0, 0, 0], 0).into(),
            tenant: "default".to_string(),
        }
    }
}

/// Start the HTTP forward proxy.
///
/// Runs until the server is shut down. Call from a spawned task.
pub async fn run_forward_proxy<P: PolicyChecker + 'static>(
    config: ForwardProxyConfig,
    policy_checker: Arc<P>,
    transform_executor: Arc<TransformExecutor>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(config.listen_addr).await?;
    let tenant = Arc::new(config.tenant);
    info!(addr = %config.listen_addr, tenant = %tenant, "forward proxy listening");

    loop {
        let (stream, peer) = listener.accept().await?;
        let pc = Arc::clone(&policy_checker);
        let te = Arc::clone(&transform_executor);
        let t = Arc::clone(&tenant);

        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req: Request<Incoming>| {
                let pc = Arc::clone(&pc);
                let te = Arc::clone(&te);
                let t = Arc::clone(&t);
                async move { handle_request(req, &pc, &te, &t, peer).await }
            });
            if let Err(e) = http1::Builder::new()
                .preserve_header_case(true)
                .serve_connection(io, svc)
                .with_upgrades()
                .await
            {
                if !e.to_string().contains("connection closed") {
                    warn!("proxy connection error: {}", e);
                }
            }
        });
    }
}

async fn handle_request<P: PolicyChecker>(
    req: Request<Incoming>,
    policy_checker: &Arc<P>,
    transform_executor: &Arc<TransformExecutor>,
    tenant: &str,
    _peer: SocketAddr,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.method() == Method::CONNECT {
        Ok(handle_connect(req, policy_checker.as_ref(), tenant).await)
    } else {
        Ok(handle_proxy(
            req,
            policy_checker.as_ref(),
            transform_executor.as_ref(),
            tenant,
        )
        .await)
    }
}

/// Handle CONNECT requests (HTTPS tunnels).
///
/// Checks the target domain against the policy engine.
/// If allowed, establishes a bidirectional TCP tunnel.
/// Policy is domain-level only (encrypted payload is not inspected).
async fn handle_connect<P: PolicyChecker>(
    req: Request<Incoming>,
    policy_checker: &P,
    tenant: &str,
) -> Response<Full<Bytes>> {
    let host_port = match connect_target(req.uri()) {
        Ok(target) => target,
        Err(reason) => {
            // Never log or echo a rejected authority: it is caller data and
            // may contain credentials. All subsequent uses are canonical.
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Full::new(Bytes::from(reason)))
                .unwrap();
        }
    };

    // Extract host for policy check
    let url = format!("https://{}", host_port);

    info!(target = %host_port, "CONNECT request");

    // Policy check with stateless context
    let action = policy_engine::Action {
        action_type: Some(policy_engine::action::ActionType::HttpRequest(
            policy_engine::HttpRequestAction {
                url: url.clone(),
                body: String::new(),
                headers: vec![],
            },
        )),
        ..Default::default()
    };

    let auth_resp = match policy_checker
        .check_authorization(
            &[],
            vec![action],
            // No user-supplied wire entity on the HTTP CONNECT path.
            None,
            &[],
            // The forward proxy is auth-less per request, but it's
            // pinned to a tenant at deployment time so policy
            // evaluation lands in the right shard. Multi-tenant
            // operators run one proxy listener per tenant.
            SessionScope::global(tenant),
            // The proxy itself is the principal — there's no caller
            // identity here, just the forward-proxy daemon. Treat
            // PROXY_ENTITY as the principal so policies can rule on
            // `Principal("forward-proxy")` if they care.
            Some(PROXY_ENTITY),
            // Forward proxy doesn't have a per-session policy
            // binding either; Multi-policy bindings are wired
            // through the gRPC ToolCallRequest path only.
            None,
        )
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            warn!("CONNECT policy check error: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from(format!("Policy error: {}", e))))
                .unwrap();
        }
    };

    let result = auth_resp.results.first();
    let authorized = result.map(|r| r.authorized).unwrap_or(false);

    // A CONNECT tunnel is opaque: the proxy relays bytes and never sees the
    // request inside it, so a transform the policy attached cannot be applied
    // here. Skipping it is tolerable for the same reason an unknown transform
    // id is: what the skip loses is the platform credential. Whatever the
    // client put in the tunneled request — the header this transform would
    // have replaced included — travels as the client wrote it, so the call is
    // made under the client's own identity, or uncredentialed and rejected
    // upstream. The platform credential is neither exposed nor spent, and the
    // policy's host-level authorization already applied. It is still a
    // policy/deployment mismatch worth naming, so say which transforms went
    // unapplied and to where. Transform ids only; a credential never reaches a
    // log line.
    if authorized {
        if let Some(ids) = result
            .map(|r| r.transform_ids.as_slice())
            .filter(|i| !i.is_empty())
        {
            warn!(
                target = %host_port,
                transform_ids = %ids.join(","),
                "CONNECT tunnel cannot carry a credential transform; skipping"
            );
        }
    }

    if !authorized {
        info!(target = %host_port, "CONNECT DENIED");
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("X-Sasy-Policy-Denied", "true")
            .body(Full::new(Bytes::from(format!(
                "CONNECT to {} denied by policy",
                host_port
            ))))
            .unwrap();
    }

    // Restrict CONNECT to standard HTTPS (port 443). Tunnelling to arbitrary
    // ports would let an allowed host be reached on non-HTTPS services
    // (SSH/SMTP/Redis/…) and smuggle arbitrary TCP through the opaque tunnel —
    // the policy only authorized the *host*, not the port/protocol.
    let connect_port = host_port
        .rsplit_once(':')
        .and_then(|(_, p)| p.parse::<u16>().ok());
    if connect_port != Some(443) {
        warn!(target = %host_port, "CONNECT to non-443 port rejected");
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("X-Sasy-Policy-Denied", "true")
            .body(Full::new(Bytes::from(
                "CONNECT permitted only to port 443 (HTTPS)".to_string(),
            )))
            .unwrap();
    }

    // SSRF / DNS-rebinding guard: resolve the tunnel target, reject internal
    // addresses, and connect to the vetted IP rather than re-resolving the
    // name. Policy authorization alone does not stop a name that resolves to
    // an internal address (cloud metadata, RFC1918, loopback).
    let pinned = match crate::net_guard::resolve_host_port_public(&host_port).await {
        Ok(addr) => addr,
        Err(reason) => {
            warn!(target = %host_port, reason = %reason, "CONNECT target rejected (SSRF guard)");
            return Response::builder()
                .status(StatusCode::FORBIDDEN)
                .header("X-Sasy-Policy-Denied", "true")
                .body(Full::new(Bytes::from(format!(
                    "CONNECT to {host_port} blocked: {reason}"
                ))))
                .unwrap();
        }
    };

    info!(target = %host_port, "CONNECT ALLOWED");

    // Establish TCP tunnel to the vetted IP. Each direction is pumped with an
    // idle timeout + per-direction byte cap (see `pump`) so a stalled or
    // high-volume tunnel can't tie up resources indefinitely.
    tokio::task::spawn(async move {
        match hyper::upgrade::on(req).await {
            Ok(upgraded) => match TcpStream::connect(pinned).await {
                Ok(target_stream) => {
                    let (client_read, client_write) = tokio::io::split(TokioIo::new(upgraded));
                    let (target_read, target_write) = tokio::io::split(target_stream);
                    let c2t = pump(client_read, target_write);
                    let t2c = pump(target_read, client_write);
                    let _ = tokio::join!(c2t, t2c);
                }
                Err(e) => {
                    warn!("CONNECT tunnel to {} failed: {}", host_port, e);
                }
            },
            Err(e) => {
                warn!("CONNECT upgrade failed: {}", e);
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

/// Handle plain HTTP proxy requests.
///
/// Checks the URL against the policy engine.
/// If allowed, applies transforms (credential injection) and forwards.
async fn handle_proxy<P: PolicyChecker>(
    req: Request<Incoming>,
    policy_checker: &P,
    transform_executor: &TransformExecutor,
    tenant: &str,
) -> Response<Full<Bytes>> {
    let url = req.uri().to_string();
    let method = req.method().to_string();

    // Collect headers
    let headers: Vec<(String, String)> = req
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();

    // Read body with a hard size cap so a client can't exhaust memory.
    let body_bytes = match http_body_util::Limited::new(req.into_body(), MAX_PROXY_BODY)
        .collect()
        .await
    {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(Full::new(Bytes::from("request body too large")))
                .unwrap();
        }
    };
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    proxy_request(
        url,
        method,
        headers,
        body,
        policy_checker,
        transform_executor,
        tenant,
    )
    .await
}

/// Decide on, transform and forward one request, once it has been read off
/// the wire.
///
/// Split from [`handle_proxy`] so the decision path can be driven directly by
/// a test: a hyper `Incoming` body only exists on a live connection, so a
/// test cannot build the request that `handle_proxy` takes.
#[allow(clippy::too_many_arguments)]
async fn proxy_request<P: PolicyChecker>(
    url: String,
    method: String,
    headers: Vec<(String, String)>,
    body: String,
    policy_checker: &P,
    transform_executor: &TransformExecutor,
    tenant: &str,
) -> Response<Full<Bytes>> {
    // The URL is the client's, before any transform: it can already carry a
    // key the client put in the query, in a fragment, or as userinfo. Only the
    // redacted form — scheme, host, port and path, which is what these lines
    // are for — is ever logged or echoed back.
    let logged_url = crate::net_guard::redact_url(&url);
    info!(url = %logged_url, method = %method, "proxy request");

    // Policy check
    let action = policy_engine::Action {
        action_type: Some(policy_engine::action::ActionType::HttpRequest(
            policy_engine::HttpRequestAction {
                url: url.clone(),
                body: body.clone(),
                headers: headers
                    .iter()
                    .map(|(k, v)| policy_engine::Header {
                        key: k.clone(),
                        value: v.clone(),
                    })
                    .collect(),
            },
        )),
        ..Default::default()
    };

    let auth_resp = match policy_checker
        .check_authorization(
            &[],
            vec![action],
            // No user-supplied wire entity on the HTTP CONNECT path.
            None,
            &[],
            // Tenant is pinned at deployment time (see
            // `ForwardProxyConfig::tenant`). Multi-tenant
            // operators run one proxy listener per tenant.
            SessionScope::global(tenant),
            // The proxy itself is the principal — there's no caller
            // identity here, just the forward-proxy daemon. Treat
            // PROXY_ENTITY as the principal so policies can rule on
            // `Principal("forward-proxy")` if they care.
            Some(PROXY_ENTITY),
            // Forward proxy doesn't have a per-session policy
            // binding either; Multi-policy bindings are wired
            // through the gRPC ToolCallRequest path only.
            None,
        )
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            warn!("proxy policy check error: {}", e);
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from(format!("Policy error: {}", e))))
                .unwrap();
        }
    };

    let action_result = auth_resp.results.into_iter().next();
    let (authorized, deny, transform_ids) = match &action_result {
        Some(ar) => (
            ar.authorized,
            ar.deny_if_unauthorized,
            ar.transform_ids.clone(),
        ),
        None => (false, true, vec![]),
    };

    if !authorized && deny {
        info!(url = %logged_url, "proxy DENIED");
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("X-Sasy-Policy-Denied", "true")
            .body(Full::new(Bytes::from(format!(
                "Request to {} denied by policy",
                logged_url
            ))))
            .unwrap();
    }

    // Apply transforms (credential injection) and vet the fetch target. Use
    // the same listener-pinned tenant as the policy check above so the
    // credential lookup hits the matching shard. Transforms run only on the
    // authorized arm; a passthrough carries no credential.
    let upstream = match prepare_upstream(
        &url,
        headers,
        if authorized { &transform_ids } else { &[] },
        transform_executor,
        tenant,
    )
    .await
    {
        Ok(u) => u,
        Err(refusal) => return refusal,
    };
    let final_url = upstream.url;
    let final_headers = upstream.headers;

    // Forward request through the shared SSRF-pinned client builder (see
    // [`crate::net_guard::pinned_client_builder`] for the security rationale).
    let client = crate::net_guard::pinned_client_builder(&upstream.pin_host, upstream.pin_addr)
        .build()
        .expect("static reqwest client config is valid");
    let req_method: reqwest::Method = method.parse().unwrap_or(reqwest::Method::GET);
    let mut builder = client.request(req_method, &final_url);

    for (k, v) in &final_headers {
        if crate::net_guard::forward_request_header(k) {
            builder = builder.header(k.as_str(), v.as_str());
        }
    }
    if !body.is_empty() {
        builder = builder.body(body);
    }

    match builder.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            // Cap the response body: stream and abort if it exceeds the limit
            // so a malicious upstream can't OOM us (decompression already off).
            use futures_util::StreamExt;
            let mut stream = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(c) => {
                        if buf.len() + c.len() > MAX_PROXY_BODY {
                            warn!("upstream response exceeded {} bytes", MAX_PROXY_BODY);
                            return Response::builder()
                                .status(StatusCode::BAD_GATEWAY)
                                .body(Full::new(Bytes::from("upstream response too large")))
                                .unwrap();
                        }
                        buf.extend_from_slice(&c);
                    }
                    Err(e) => {
                        warn!(error = %e.without_url(), "proxy response stream error");
                        return Response::builder()
                            .status(StatusCode::BAD_GATEWAY)
                            .body(Full::new(Bytes::from("upstream error")))
                            .unwrap();
                    }
                }
            }
            Response::builder()
                .status(status)
                .body(Full::new(Bytes::from(buf)))
                .unwrap()
        }
        Err(e) => {
            // Generic body to the caller; redacted URL + URL-stripped error
            // server-side (the post-transform error may carry credentials).
            warn!(
                url = %logged_url,
                error = %e.without_url(),
                "proxy forward error"
            );
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from("upstream error")))
                .unwrap()
        }
    }
}

/// Accept only CONNECT's authority form and return a canonical host:port.
/// The error contains no caller text and is safe to display before logging.
fn connect_target(uri: &hyper::Uri) -> Result<String, &'static str> {
    const INVALID: &str = "invalid CONNECT target: expected host:port without userinfo";
    if uri.scheme().is_some() || uri.path_and_query().is_some() {
        return Err(INVALID);
    }
    let authority = uri.authority().ok_or(INVALID)?;
    let port = authority.port_u16().filter(|p| *p != 0).ok_or(INVALID)?;
    if authority.as_str().contains('@') {
        return Err(INVALID);
    }
    let parsed = url::Url::parse(&format!("https://{authority}")).map_err(|_| INVALID)?;
    if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(INVALID);
    }
    let host = parsed.host_str().ok_or(INVALID)?;
    Ok(format!("{host}:{port}"))
}

/// What the proxy is about to send: the post-transform URL and headers, and
/// the SSRF-vetted address the connection is pinned to.
struct Upstream {
    url: String,
    headers: Vec<(String, String)>,
    pin_host: String,
    pin_addr: SocketAddr,
}

/// Inject the credentials the policy attached and vet the fetch target.
///
/// Split from [`proxy_request`] so a test can see exactly what would go
/// upstream — the URL the credential is written onto included — without a
/// network round trip. Either arm of the `Err` is the response the caller
/// gets; nothing is sent upstream in that case.
async fn prepare_upstream(
    url: &str,
    headers: Vec<(String, String)>,
    transform_ids: &[String],
    transform_executor: &TransformExecutor,
    tenant: &str,
) -> Result<Upstream, Response<Full<Bytes>>> {
    let mut final_headers = headers;
    let mut final_url = url.to_string();

    if !transform_ids.is_empty() {
        if let Err(e) = transform_executor
            .execute(
                &mut final_headers,
                &mut final_url,
                transform_ids,
                tenant,
                Some(PROXY_ENTITY),
            )
            .await
        {
            // Fail closed. A transform that could not be applied means the
            // upstream credential is missing — or that the leg it would ride on
            // is not TLS — and forwarding anyway sends the request
            // unauthenticated, under whatever identity the caller supplied
            // instead, or in the clear. Nothing goes upstream.
            warn!(
                url = %crate::net_guard::redact_url(url),
                error = %e,
                "credential transform failed; denying"
            );
            // The backend's message (`e`) went to the log above and stops
            // there: it names the secret manager's address, mount and path.
            return Err(Response::builder()
                .status(StatusCode::FORBIDDEN)
                .header("X-Sasy-Policy-Denied", "true")
                .body(Full::new(Bytes::from(
                    crate::transforms::TRANSFORM_FAILURE_REASON,
                )))
                .unwrap());
        }
    }

    // SSRF / DNS-rebinding guard on the (post-transform) fetch target:
    // resolve, reject internal addresses, and pin the vetted IP.
    match crate::net_guard::resolve_public_target(&final_url).await {
        Ok((pin_host, pin_addr)) => Ok(Upstream {
            url: final_url,
            headers: final_headers,
            pin_host,
            pin_addr,
        }),
        Err(reason) => {
            warn!(
                url = %crate::net_guard::redact_url(&final_url),
                reason = %reason,
                "proxy target rejected (SSRF guard)"
            );
            Err(Response::builder()
                .status(StatusCode::FORBIDDEN)
                .header("X-Sasy-Policy-Denied", "true")
                .body(Full::new(Bytes::from(format!("blocked: {reason}"))))
                .unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_targets_are_canonical_authorities_before_display() {
        for (input, expected) in [
            ("API.EXAMPLE.COM:443", "api.example.com:443"),
            ("[2001:db8::1]:443", "[2001:db8::1]:443"),
        ] {
            assert_eq!(connect_target(&input.parse().unwrap()).unwrap(), expected);
        }
        for input in [
            "/path",
            "https://api.example.com:443/",
            "api.example.com",
            "user@api.example.com:443",
        ] {
            let uri = input.parse().unwrap();
            assert_eq!(
                connect_target(&uri).unwrap_err(),
                "invalid CONNECT target: expected host:port without userinfo"
            );
        }
    }

    use sasy_credential::CredentialSource;

    use crate::test_support::{
        injecting_transform_config, AllowAllWithTransform, DisclosingSource, UnreachableSource,
        DISCLOSING_BACKEND_MESSAGE,
    };
    use crate::transforms::TransformExecutor;

    /// An executor holding a working platform credential for the entity the
    /// forward proxy injects under.
    fn executor_with_credential() -> TransformExecutor {
        let source = sasy_credential::MemorySource::new().unwrap();
        source
            .store()
            .set_credentials(
                PROXY_ENTITY,
                "openai",
                vec![("api_key".into(), "sk-test".into())],
            )
            .unwrap();
        TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(source) as Arc<dyn CredentialSource>,
        )
    }

    /// The credential backend's own words stop at the server.
    ///
    /// A failed read names the secret manager's address, its mount and the
    /// path under it — which is a map of the deployment, handed to the very
    /// caller the reference monitor exists to police. The caller gets the
    /// fixed reason and nothing else.
    #[tokio::test]
    async fn the_refusal_does_not_name_the_secret_manager() {
        let executor = TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(DisclosingSource) as Arc<dyn CredentialSource>,
        );

        let resp = prepare_upstream(
            "https://192.0.2.1/v1/chat",
            vec![],
            &["inject_key".to_string()],
            &executor,
            "default",
        )
        .await
        .err()
        .expect("a credential that cannot be read must refuse the request");

        let body = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
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
            !DISCLOSING_BACKEND_MESSAGE.is_empty() && !body.contains(DISCLOSING_BACKEND_MESSAGE),
            "the backend message must not be forwarded verbatim: {body}"
        );
    }

    /// The request that carries the credential is made over TLS, even though
    /// the client asked the proxy for plain HTTP.
    ///
    /// Clients send absolute-form `http://…` on purpose — a `CONNECT` tunnel is
    /// opaque, so there would be nothing to inject into — and the policy
    /// authorizes on the host alone. Without the upgrade the platform key was
    /// written onto a plaintext connection to the upstream.
    ///
    /// The assertion is on the URL the proxy would fetch, not on a network
    /// round trip: `192.0.2.1` is reserved for documentation and routed
    /// nowhere.
    #[tokio::test]
    async fn a_credentialed_plain_http_request_goes_upstream_over_tls() {
        let executor = executor_with_credential();

        let upstream = prepare_upstream(
            "http://192.0.2.1/v1/chat",
            vec![],
            &["inject_key".to_string()],
            &executor,
            "default",
        )
        .await
        .map_err(|_| "the request must be prepared, not refused")
        .unwrap();

        assert_eq!(
            upstream.url, "https://192.0.2.1/v1/chat",
            "the leg that carries the credential is TLS"
        );
        assert_eq!(
            upstream.pin_addr.port(),
            443,
            "and it connects on the HTTPS port, not 80"
        );
        assert!(
            upstream
                .headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"),
            "the credential was injected: {:?}",
            upstream.headers.iter().map(|(k, _)| k).collect::<Vec<_>>()
        );
    }

    /// A request with no transform attached is forwarded exactly as it came in,
    /// plain HTTP included — there is no credential to protect.
    #[tokio::test]
    async fn a_request_without_transforms_keeps_plain_http() {
        let executor = executor_with_credential();

        let upstream =
            prepare_upstream("http://192.0.2.1/health", vec![], &[], &executor, "default")
                .await
                .map_err(|_| "an untransformed request must not be refused")
                .unwrap();

        assert_eq!(upstream.url, "http://192.0.2.1/health");
        assert_eq!(upstream.pin_addr.port(), 80);
    }

    /// Plain HTTP on a port that is not the HTTP default is denied rather than
    /// upgraded: whether that port speaks TLS is a guess, and the alternative
    /// is putting the platform credential on the wire in the clear. The request
    /// is never forwarded.
    #[tokio::test]
    async fn a_credentialed_plain_http_request_to_another_port_is_denied() {
        let executor = executor_with_credential();

        let resp = proxy_request(
            "http://192.0.2.1:8080/v1/chat".to_string(),
            "GET".to_string(),
            vec![],
            String::new(),
            &AllowAllWithTransform,
            &executor,
            "default",
        )
        .await;

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            resp.headers()
                .get("X-Sasy-Policy-Denied")
                .and_then(|v| v.to_str().ok()),
            Some("true"),
        );
        let body = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
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
    }

    /// Deny a request whose credential cannot be fetched. The documentation-only
    /// address 192.0.2.1 ensures an accidental network attempt cannot succeed.
    #[tokio::test]
    async fn a_credential_that_cannot_be_fetched_denies_the_request() {
        let executor = TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(UnreachableSource) as Arc<dyn CredentialSource>,
        );

        let resp = proxy_request(
            "http://192.0.2.1/v1/chat".to_string(),
            "GET".to_string(),
            vec![],
            String::new(),
            &AllowAllWithTransform,
            &executor,
            "default",
        )
        .await;

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            resp.headers()
                .get("X-Sasy-Policy-Denied")
                .and_then(|v| v.to_str().ok()),
            Some("true"),
            "the refusal is reported the same way as the proxy's other refusals"
        );
        let body = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(
            body == crate::transforms::TRANSFORM_FAILURE_REASON,
            "the reason must say the credential is what failed: {body}"
        );
    }

    /// Deny missing secrets or fields instead of forwarding the caller's
    /// original Authorization header when the required transform cannot run.
    #[tokio::test]
    async fn a_credential_that_resolves_to_nothing_denies_the_request() {
        let executor = TransformExecutor::new(
            injecting_transform_config(),
            Arc::new(sasy_credential::MemorySource::new().unwrap()) as Arc<dyn CredentialSource>,
        );

        let resp = proxy_request(
            "http://192.0.2.1/v1/chat".to_string(),
            "GET".to_string(),
            vec![(
                "Authorization".to_string(),
                "Bearer caller-supplied".to_string(),
            )],
            String::new(),
            &AllowAllWithTransform,
            &executor,
            "default",
        )
        .await;

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(
            body == crate::transforms::TRANSFORM_FAILURE_REASON,
            "the reason must say the credential is what failed: {body}"
        );
    }
}
