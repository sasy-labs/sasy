//! Read-only credential backend for OpenBao (and API-compatible Vault)
//! KV version 2 secret engines.
//!
//! "KV version 2" is OpenBao's versioned key/value store: a secret lives at a
//! path, holds a flat map of names to values, and is read over HTTP with a
//! token in the `X-Vault-Token` header. This backend reads two paths per
//! lookup and merges them:
//!
//! ```text
//! <prefix>/<tenant>/_default/<service>   tenant-wide defaults
//! <prefix>/<tenant>/<entity>/<service>   the entity's own, laid over the top
//! ```
//!
//! Nothing here writes: a credential is created and rotated in OpenBao, by
//! whoever owns it, and this process only ever reads. The `SetCredentials`
//! RPC is refused for the same reason.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use reqwest::{Client, StatusCode};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::error::CredentialError;
use crate::source::{CredentialSource, CredentialView, Freshness, ResolvedCredentials};
use crate::store::WILDCARD_ENTITY;

/// Default KV v2 mount to read from.
pub const DEFAULT_MOUNT: &str = "secret";
/// Default first path segment, so one OpenBao can serve more than this
/// platform.
pub const DEFAULT_PATH_PREFIX: &str = "sasy";
/// Default lifetime of a resolved credential before it is read again.
pub const DEFAULT_CACHE_TTL_SECS: u64 = 60;
/// Default per-request HTTP timeout.
pub const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// The path segment the tenant-wide defaults live under. Reserved: an entity
/// may not be called this, or it could serve its own credentials as the
/// tenant's shared ones.
const DEFAULTS_SEGMENT: &str = "_default";

/// Renew a lease once two thirds of it have passed, leaving a third of it as
/// margin for a slow or briefly unreachable server.
const RENEW_AFTER: f64 = 2.0 / 3.0;

/// Maximum renewal delay. Lease lengths are untrusted remote input; capping
/// them prevents overflow in Duration/Instant arithmetic. A longer lease is
/// simply renewed earlier.
const MAX_RENEW_AFTER: Duration = Duration::from_secs(24 * 60 * 60);

/// How long to wait before trying again after a failed renewal, so a server
/// that is refusing renewals is not asked on every single request.
const RENEW_RETRY_AFTER: Duration = Duration::from_secs(30);

/// How this process proves who it is to OpenBao.
#[derive(Clone)]
pub enum OpenBaoAuth {
    /// A file holding a token. Read once at startup.
    TokenFile(String),
    /// AppRole: a role id (an identifier) plus a secret id read from a file.
    /// Logging in exchanges them for a token with a lease.
    AppRole {
        role_id: String,
        secret_id_file: String,
    },
}

impl std::fmt::Debug for OpenBaoAuth {
    /// Prints paths and the mode, never the material read from them.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenFile(path) => write!(f, "TokenFile({path})"),
            Self::AppRole { secret_id_file, .. } => {
                write!(f, "AppRole {{ secret_id_file: {secret_id_file} }}")
            }
        }
    }
}

impl OpenBaoAuth {
    fn is_approle(&self) -> bool {
        matches!(self, Self::AppRole { .. })
    }
}

/// Everything the backend needs to reach one OpenBao server.
#[derive(Clone, Debug)]
pub struct OpenBaoConfig {
    /// Base address, e.g. `https://bao.internal:8200`.
    pub addr: String,
    /// KV v2 mount, e.g. `secret`.
    pub mount: String,
    /// Path segments prepended to every lookup.
    pub path_prefix: String,
    /// Optional namespace, sent as `X-Vault-Namespace`.
    pub namespace: Option<String>,
    pub auth: OpenBaoAuth,
    /// PEM file holding the certificate authority that signed the server's
    /// certificate, for a private CA.
    pub ca_cert: Option<String>,
    /// How long a resolved credential may be reused before it is read again.
    pub cache_ttl: Duration,
    /// Per-request HTTP timeout.
    pub timeout: Duration,
}

impl OpenBaoConfig {
    /// A configuration with the documented defaults, for `addr` and `auth`.
    pub fn new(addr: impl Into<String>, auth: OpenBaoAuth) -> Self {
        Self {
            addr: addr.into(),
            mount: DEFAULT_MOUNT.to_string(),
            path_prefix: DEFAULT_PATH_PREFIX.to_string(),
            namespace: None,
            auth,
            ca_cert: None,
            cache_ttl: Duration::from_secs(DEFAULT_CACHE_TTL_SECS),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }
}

/// The token in use, and when it needs attention.
struct TokenState {
    token: String,
    refresh_at: Option<Instant>,
    expires_at: Option<Instant>,
    renewable: bool,
}

impl TokenState {
    fn from_lease(token: String, lease_secs: u64, renewable: bool) -> Self {
        // A finite lease needs attention even when it cannot be renewed:
        // AppRole must log in again instead. A zero lease does not expire.
        let now = Instant::now();
        let expires_at = (lease_secs > 0).then(|| {
            now.checked_add(Duration::from_secs(lease_secs))
                .unwrap_or(now + MAX_RENEW_AFTER)
        });
        let refresh_at = if lease_secs > 0 {
            // Integer arithmetic, capped: see MAX_RENEW_AFTER for why the
            // server's number is not trusted to be small.
            let seconds = ((lease_secs as f64 * RENEW_AFTER) as u64).max(1);
            let after = Duration::from_secs(seconds).min(MAX_RENEW_AFTER);
            // The cap above is what makes this addition safe: `after` is at
            // most 24 hours, which every supported clock can represent. The
            // `checked_add` is belt-and-braces for a platform whose clock
            // cannot — it schedules the retry interval instead of taking the
            // process down — so no test reaches this fallback.
            Some(
                Instant::now()
                    .checked_add(after)
                    .unwrap_or_else(|| Instant::now() + RENEW_RETRY_AFTER),
            )
        } else {
            None
        };
        Self {
            token,
            refresh_at,
            expires_at,
            renewable,
        }
    }

    fn due_for_refresh(&self) -> bool {
        self.refresh_at.is_some_and(|at| Instant::now() >= at)
    }

    fn usable_token(&self) -> Result<String, CredentialError> {
        if self.expires_at.is_some_and(|at| Instant::now() >= at) {
            return Err(CredentialError::Backend(
                "the OpenBao token has expired; authentication must be refreshed".to_string(),
            ));
        }
        Ok(self.token.clone())
    }

    fn defer_retry(&mut self) {
        self.refresh_at = Some(Instant::now() + RENEW_RETRY_AFTER);
    }
}

/// A read-only credential source backed by an OpenBao KV v2 mount.
pub struct OpenBaoSource {
    config: OpenBaoConfig,
    /// `addr` with any trailing slash removed, so path joins are predictable.
    addr: String,
    client: Client,
    token: RwLock<TokenState>,
}

impl OpenBaoSource {
    /// Connect and establish authentication.
    ///
    /// Fails — and so refuses to let the process start — when the address is
    /// unusable, the token or secret-id file cannot be read, or the server
    /// rejects the credentials. Starting anyway would produce a server that
    /// looks healthy and denies every authorized request that needs a
    /// credential, which is a worse way to learn the same thing.
    pub async fn connect(config: OpenBaoConfig) -> Result<Self, CredentialError> {
        vet_address(&config.addr)?;
        for segment in config.path_prefix.split('/') {
            check_segment("path prefix segment", segment)?;
        }
        check_segment("mount", &config.mount)?;

        let mut builder = Client::builder()
            .timeout(config.timeout)
            // Never follow a redirect: it would carry the token to whatever
            // host the response names.
            .redirect(reqwest::redirect::Policy::none());
        if let Some(ref ca_path) = config.ca_cert {
            let pem = std::fs::read(ca_path).map_err(|e| {
                CredentialError::Config(format!(
                    "reading the OpenBao CA certificate {ca_path}: {e}"
                ))
            })?;
            let certs = reqwest::Certificate::from_pem_bundle(&pem).map_err(|e| {
                CredentialError::Config(format!(
                    "parsing the OpenBao CA certificate {ca_path}: {e}"
                ))
            })?;
            for cert in certs {
                builder = builder.add_root_certificate(cert);
            }
        }
        let client = builder.build().map_err(|e| {
            CredentialError::Config(format!("building the OpenBao HTTP client: {e}"))
        })?;

        let addr = config.addr.trim_end_matches('/').to_string();
        let source = Self {
            addr,
            client,
            // Placeholder; replaced below once authentication is established.
            token: RwLock::new(TokenState::from_lease(String::new(), 0, false)),
            config,
        };

        let state = match source.config.auth {
            OpenBaoAuth::TokenFile(ref path) => {
                let token = read_secret_file("OpenBao token", path)?;
                source.lookup_self(&token).await?
            }
            OpenBaoAuth::AppRole { .. } => source.login().await?,
        };
        *source.token.write().await = state;
        info!(
            addr = %source.addr,
            mount = %source.config.mount,
            prefix = %source.config.path_prefix,
            "authenticated to OpenBao"
        );
        Ok(source)
    }

    /// The address this backend reads from, for operator-facing messages.
    pub fn address(&self) -> &str {
        &self.addr
    }

    fn request(&self, method: reqwest::Method, url: &str, token: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .request(method, url)
            .header("X-Vault-Token", token);
        if let Some(ref ns) = self.config.namespace {
            req = req.header("X-Vault-Namespace", ns);
        }
        req
    }

    /// Confirm a token from a file is one the server accepts, and learn its
    /// remaining lease so it can be renewed before it lapses.
    async fn lookup_self(&self, token: &str) -> Result<TokenState, CredentialError> {
        let url = format!("{}/v1/auth/token/lookup-self", self.addr);
        let resp = self
            .request(reqwest::Method::GET, &url, token)
            .send()
            .await
            .map_err(|e| {
                CredentialError::Backend(format!(
                    "OpenBao at {} could not be reached: {}",
                    self.addr,
                    transport_reason(&e)
                ))
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(CredentialError::Config(format!(
                "OpenBao at {} rejected the configured token ({status})",
                self.addr
            )));
        }
        let body: Value = resp.json().await.map_err(|e| {
            CredentialError::Backend(format!("OpenBao returned an unreadable token lookup: {e}"))
        })?;
        let data = body.get("data");
        let ttl = data
            .and_then(|d| d.get("ttl"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let renewable = data
            .and_then(|d| d.get("renewable"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(TokenState::from_lease(token.to_string(), ttl, renewable))
    }

    /// Exchange the AppRole role id and secret id for a token.
    async fn login(&self) -> Result<TokenState, CredentialError> {
        let OpenBaoAuth::AppRole {
            ref role_id,
            ref secret_id_file,
        } = self.config.auth
        else {
            return Err(CredentialError::Config(
                "OpenBao AppRole login attempted without an AppRole configuration".to_string(),
            ));
        };
        let secret_id = read_secret_file("OpenBao AppRole secret id", secret_id_file)?;
        let url = format!("{}/v1/auth/approle/login", self.addr);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "role_id": role_id,
            "secret_id": secret_id,
        }));
        if let Some(ref ns) = self.config.namespace {
            req = req.header("X-Vault-Namespace", ns);
        }
        let resp = req.send().await.map_err(|e| {
            CredentialError::Backend(format!(
                "OpenBao at {} could not be reached for AppRole login: {}",
                self.addr,
                transport_reason(&e)
            ))
        })?;
        let status = resp.status();
        if !status.is_success() {
            // The body of a rejected login can echo back what was sent, so
            // only the status is reported.
            return Err(CredentialError::Config(format!(
                "OpenBao at {} rejected the AppRole login ({status})",
                self.addr
            )));
        }
        let body: Value = resp.json().await.map_err(|e| {
            CredentialError::Backend(format!(
                "OpenBao returned an unreadable login response: {e}"
            ))
        })?;
        let auth = body.get("auth").ok_or_else(|| {
            CredentialError::Backend("OpenBao login response carried no auth block".to_string())
        })?;
        let token = auth
            .get("client_token")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                CredentialError::Backend(
                    "OpenBao login response carried no client token".to_string(),
                )
            })?;
        let lease = auth
            .get("lease_duration")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let renewable = auth
            .get("renewable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        debug!(
            lease_duration = lease,
            renewable, "OpenBao AppRole login succeeded"
        );
        Ok(TokenState::from_lease(token.to_string(), lease, renewable))
    }

    /// Ask the server to extend the current token's lease.
    ///
    /// `Ok(None)` means the server refused (403) — the token is gone, not
    /// merely un-extended.
    async fn renew_self(&self, token: &str) -> Result<Option<TokenState>, CredentialError> {
        let url = format!("{}/v1/auth/token/renew-self", self.addr);
        let resp = self
            .request(reqwest::Method::POST, &url, token)
            .send()
            .await
            .map_err(|e| {
                CredentialError::Backend(format!(
                    "renewing the OpenBao token failed: {}",
                    transport_reason(&e)
                ))
            })?;
        let status = resp.status();
        if status == StatusCode::FORBIDDEN {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(CredentialError::Backend(format!(
                "OpenBao returned {status} when renewing the token"
            )));
        }
        let body: Value = resp.json().await.map_err(|e| {
            CredentialError::Backend(format!("OpenBao returned an unreadable renewal: {e}"))
        })?;
        let auth = body.get("auth");
        // A renewal keeps the same token; only its lease moves.
        let renewed = auth
            .and_then(|a| a.get("client_token"))
            .and_then(Value::as_str)
            .unwrap_or(token);
        let lease = auth
            .and_then(|a| a.get("lease_duration"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let renewable = auth
            .and_then(|a| a.get("renewable"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(Some(TokenState::from_lease(
            renewed.to_string(),
            lease,
            renewable,
        )))
    }

    /// The token to use for the next request, renewing or logging in again
    /// first if the lease is running out.
    async fn current_token(&self) -> Result<String, CredentialError> {
        {
            let state = self.token.read().await;
            if !state.due_for_refresh() {
                return state.usable_token();
            }
        }
        let mut state = self.token.write().await;
        // Another task may have renewed it while this one waited.
        if !state.due_for_refresh() {
            return state.usable_token();
        }
        if !state.renewable {
            if self.config.auth.is_approle() {
                // Schedule retry before awaiting login, so a refused login
                // cannot be retried by every request waiting for this lock.
                state.defer_retry();
                *state = self.login().await?;
            } else {
                // Token files are read once at startup. A nonrenewable token
                // cannot be extended; retain it only through its actual lease.
                state.refresh_at = None;
            }
            return state.usable_token();
        }
        match self.renew_self(&state.token).await {
            Ok(Some(renewed)) => {
                *state = renewed;
                return state.usable_token();
            }
            Ok(None) if self.config.auth.is_approle() => {
                // The token is gone. AppRole can get another one; a token from
                // a file cannot, so only this branch logs in again.
                warn!("OpenBao refused to renew the token; logging in again");
                state.defer_retry();
                *state = self.login().await?;
                return state.usable_token();
            }
            Ok(None) => {
                warn!(
                    "OpenBao refused to renew the configured token and no AppRole is \
                     configured; credential lookups will start failing when it expires"
                );
                state.defer_retry();
            }
            Err(e) => {
                warn!(error = %e, "renewing the OpenBao token failed; will retry");
                state.defer_retry();
            }
        }
        state.usable_token()
    }

    /// Read one KV v2 secret. A path that does not exist is an empty map, not
    /// an error — an entity with no credentials of its own is ordinary.
    async fn read_secret(&self, path: &str) -> Result<HashMap<String, String>, CredentialError> {
        let token = self.current_token().await?;
        let url = format!("{}/v1/{}/data/{}", self.addr, self.config.mount, path);
        let resp = self
            .request(reqwest::Method::GET, &url, &token)
            .send()
            .await
            .map_err(|e| {
                CredentialError::Backend(format!(
                    "reading {path} from OpenBao at {} failed: {}",
                    self.addr,
                    transport_reason(&e)
                ))
            })?;
        let status = resp.status();
        if status == StatusCode::NOT_FOUND {
            return Ok(HashMap::new());
        }
        if status == StatusCode::FORBIDDEN {
            return Err(CredentialError::Backend(format!(
                "OpenBao denied this process access to {}/{path}; the token's policy does \
                 not grant read on that path",
                self.config.mount
            )));
        }
        if !status.is_success() {
            return Err(CredentialError::Backend(format!(
                "OpenBao returned {status} for {}/{path}",
                self.config.mount
            )));
        }
        let body: Value = resp.json().await.map_err(|e| {
            CredentialError::Backend(format!(
                "OpenBao returned an unreadable secret at {path}: {e}"
            ))
        })?;
        match body.get("data").and_then(|d| d.get("data")) {
            Some(Value::Object(fields)) => {
                let mut values = HashMap::new();
                for (key, value) in fields {
                    match value.as_str() {
                        Some(s) => {
                            values.insert(key.clone(), s.to_string());
                        }
                        // Only strings can be interpolated into a header or a
                        // query parameter; anything else is a mis-typed
                        // secret, worth saying out loud by name.
                        None => warn!(
                            path,
                            key = %key, "OpenBao secret field is not a string; ignoring it"
                        ),
                    }
                }
                Ok(values)
            }
            // A deleted version reads as `data: null`.
            Some(Value::Null) | None => Ok(HashMap::new()),
            Some(_) => Err(CredentialError::Backend(format!(
                "OpenBao secret at {path} is not a map of names to values"
            ))),
        }
    }
}

#[tonic::async_trait]
impl CredentialSource for OpenBaoSource {
    fn backend(&self) -> &'static str {
        "openbao"
    }

    fn location(&self) -> String {
        self.addr.clone()
    }

    async fn resolve(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError> {
        check_segment("tenant", tenant)?;
        check_entity(entity, view)?;
        check_segment("service", service)?;

        let prefix = self.config.path_prefix.trim_matches('/');
        let mut values = HashMap::new();
        let relay = view == CredentialView::Relay;
        if relay {
            values.extend(
                self.read_secret(&format!("{prefix}/{tenant}/{DEFAULTS_SEGMENT}/{service}"))
                    .await?,
            );
        }
        // The HTTP forward proxy has no entity of its own: it relays on
        // behalf of the tenant and asks under `*`, the name the file-backed
        // store gives the tenant-wide defaults. Those have just been read,
        // and `*` names no directory in OpenBao, so there is nothing left to
        // overlay — asking for it would only be a request certain to 404.
        if relay && entity == WILDCARD_ENTITY {
            return Ok(ResolvedCredentials {
                values,
                freshness: Freshness::Until(Instant::now() + self.config.cache_ttl),
            });
        }
        // The entity's own entries are read second, so they win.
        values.extend(
            self.read_secret(&format!("{prefix}/{tenant}/{entity}/{service}"))
                .await?,
        );

        Ok(ResolvedCredentials {
            values,
            freshness: Freshness::Until(Instant::now() + self.config.cache_ttl),
        })
    }

    fn is_current(&self, freshness: &Freshness) -> bool {
        match freshness {
            Freshness::Until(deadline) => Instant::now() < *deadline,
            // A generation belongs to a store this process can watch. This
            // one cannot be watched, so nothing stamped that way is current.
            Freshness::Generation(_) | Freshness::LayeredGeneration { .. } => false,
        }
    }

    async fn set_credentials(
        &self,
        _tenant: &str,
        _entity: &str,
        _service: &str,
        _credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        Err(CredentialError::ReadOnly(format!(
            "the openbao credential backend at {} is read-only; write the secret to \
             OpenBao itself and this process will read it",
            self.addr
        )))
    }
}

/// Read a file that holds authentication material.
///
/// The content never appears in an error: only the path and the reason the
/// read failed.
fn read_secret_file(what: &str, path: &str) -> Result<String, CredentialError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| CredentialError::Config(format!("reading the {what} from {path}: {e}")))?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        return Err(CredentialError::Config(format!(
            "the {what} file {path} is empty"
        )));
    }
    Ok(trimmed)
}

/// Reject an address that would send a token in the clear.
///
/// Plain HTTP is allowed only to a loopback address, where the request never
/// leaves the host — the shape a `bao server -dev` on a developer's machine
/// takes.
fn vet_address(addr: &str) -> Result<(), CredentialError> {
    let url = url::Url::parse(addr)
        .map_err(|e| CredentialError::Config(format!("--openbao-addr {addr} is not a URL: {e}")))?;
    match url.scheme() {
        "https" => Ok(()),
        "http" if is_loopback(&url) => Ok(()),
        "http" => Err(CredentialError::Config(format!(
            "--openbao-addr {addr} uses plain HTTP to a non-loopback host; the token \
             would travel in the clear. Use https, or a loopback address for a local \
             development server."
        ))),
        other => Err(CredentialError::Config(format!(
            "--openbao-addr {addr} uses the unsupported scheme {other}"
        ))),
    }
}

fn is_loopback(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(d)) => d == "localhost" || d.ends_with(".localhost"),
        None => false,
    }
}

/// Vet one path segment before it is spliced into a URL.
///
/// Tenant, entity and service names arrive from configuration and from
/// authenticated identities, not from a request body — but they end up in a
/// path, and a separator or a `..` in one of them would read a secret
/// belonging to someone else. Refuse rather than encode: a name that needs
/// escaping to be a path segment is a name nobody should be using.
fn check_segment(kind: &str, value: &str) -> Result<(), CredentialError> {
    if value.is_empty() {
        return Err(CredentialError::Config(format!("the {kind} is empty")));
    }
    if value == "." || value == ".." {
        return Err(CredentialError::Config(format!(
            "the {kind} {value:?} is a path traversal"
        )));
    }
    if value
        .chars()
        .any(|c| c.is_control() || matches!(c, '/' | '\\' | '?' | '#' | '%' | ' '))
    {
        return Err(CredentialError::Config(format!(
            "the {kind} {value:?} contains a character that is not allowed in an \
             OpenBao path segment"
        )));
    }
    Ok(())
}

/// An entity may not occupy a reserved segment.
///
/// `_default` is where the tenant-wide credentials live, and `*` is the name
/// the file-backed store gives them. An entity called either one would be
/// asking for the tenant's shared secrets by pretending to be them — in the
/// OWN view, which is the view a credential-reader gets and the one the
/// reservation protects.
///
/// The relay view is different: it merges the tenant-wide defaults in
/// already, on purpose, because that merge is what gets injected. `*` there
/// asks for exactly what the view hands out anyway, so it grants nothing
/// extra — and it is the entity the HTTP forward proxy always relays under,
/// so refusing it would leave that whole injection path unable to resolve a
/// credential against this backend.
fn check_entity(entity: &str, view: CredentialView) -> Result<(), CredentialError> {
    if entity == WILDCARD_ENTITY && view == CredentialView::Relay {
        return Ok(());
    }
    if entity == DEFAULTS_SEGMENT || entity == WILDCARD_ENTITY {
        return Err(CredentialError::Config(format!(
            "{entity:?} is reserved for the tenant-wide defaults and cannot be an \
             entity name in the openbao backend"
        )));
    }
    check_segment("entity", entity)
}

/// The reason a request failed, without the URL reqwest would otherwise
/// attach — that keeps a namespace or path out of a log line that may be
/// shipped elsewhere.
fn transport_reason(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "timed out".to_string()
    } else if e.is_connect() {
        "connection refused or unreachable".to_string()
    } else {
        "transport error".to_string()
    }
}

/// Build the source without contacting the server. Tests only: `connect` is
/// the supported entry point, and it authenticates.
#[cfg(test)]
fn source_for_test(config: OpenBaoConfig, token: &str) -> OpenBaoSource {
    let addr = config.addr.trim_end_matches('/').to_string();
    OpenBaoSource {
        addr,
        client: Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("static test client config is valid"),
        token: RwLock::new(TokenState {
            token: token.to_string(),
            refresh_at: None,
            expires_at: None,
            renewable: false,
        }),
        config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use http_body_util::{BodyExt, Full};
    use hyper::body::{Bytes, Incoming};
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    /// Extreme server-supplied lease lengths schedule renewal within the cap
    /// without overflowing Duration or Instant arithmetic.
    #[test]
    fn an_out_of_range_lease_is_capped_not_fatal() {
        for lease in [13835058055281994751_u64, u64::MAX] {
            let state = TokenState::from_lease("t".to_string(), lease, true);
            let at = state
                .refresh_at
                .expect("a renewable lease schedules a renewal");
            let wait = at.saturating_duration_since(Instant::now());
            assert!(
                wait <= MAX_RENEW_AFTER,
                "lease {lease} scheduled a renewal {wait:?} out, past the cap"
            );
            assert!(
                !state.due_for_refresh(),
                "lease {lease} scheduled a renewal in the past"
            );
        }
    }

    /// An ordinary lease still renews after two thirds of it.
    #[test]
    fn an_ordinary_lease_renews_after_two_thirds_of_it() {
        let state = TokenState::from_lease("t".to_string(), 900, true);
        let wait = state
            .refresh_at
            .expect("a renewable lease schedules a renewal")
            .saturating_duration_since(Instant::now());
        assert!(
            wait > Duration::from_secs(595) && wait <= Duration::from_secs(600),
            "expected ~600s, got {wait:?}"
        );
    }

    #[tokio::test]
    async fn finite_nonrenewable_approle_tokens_refresh_once_for_concurrent_readers() {
        let logins = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&logins);
        let addr = spawn_fake(Arc::new(move |_method, path, _body| {
            assert_eq!(path, "/v1/auth/approle/login");
            let n = count.fetch_add(1, Ordering::SeqCst) + 1;
            (200, serde_json::json!({"auth": {
                    "client_token": format!("token-{n}"), "lease_duration": 1, "renewable": false
            }}).to_string())
        }))
        .await;
        let dir = tempfile::TempDir::new().unwrap();
        let secret = dir.path().join("secret-id");
        std::fs::write(&secret, "test-secret-id").unwrap();
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::AppRole {
            role_id: "test-role".into(),
            secret_id_file: secret.to_string_lossy().into_owned(),
        };
        let source = OpenBaoSource::connect(config).await.unwrap();
        {
            let state = source.token.read().await;
            assert!(state.refresh_at.is_some());
            assert!(state.expires_at.is_some());
        }
        // Let the real finite lease expire. The next pair of callers must
        // refresh it once, rather than reuse the expired token.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let (a, b) = tokio::join!(source.current_token(), source.current_token());
        assert_eq!(a.unwrap(), "token-2");
        assert_eq!(b.unwrap(), "token-2");
        assert_eq!(logins.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn failed_approle_refresh_backs_off_and_does_not_return_expired_tokens() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&attempts);
        let addr = spawn_fake(Arc::new(move |_method, path, _body| {
            assert_eq!(path, "/v1/auth/approle/login");
            count.fetch_add(1, Ordering::SeqCst);
            (503, "{}".into())
        }))
        .await;
        let dir = tempfile::TempDir::new().unwrap();
        let secret = dir.path().join("secret-id");
        std::fs::write(&secret, "test-secret-id").unwrap();
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::AppRole {
            role_id: "test-role".into(),
            secret_id_file: secret.to_string_lossy().into_owned(),
        };
        let source = source_for_test(config, "initial-token");
        {
            let mut state = source.token.write().await;
            *state = TokenState::from_lease("initial-token".into(), 60, false);
            state.refresh_at = Some(Instant::now());
            state.expires_at = Some(Instant::now());
        }
        assert!(source.current_token().await.is_err());
        assert!(source.current_token().await.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        source.token.write().await.refresh_at = Some(Instant::now());
        assert!(source.current_token().await.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    /// What a fake OpenBao answers: `(method, path, body) -> (status, body)`.
    type Handler = Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>;

    /// A minimal in-process HTTP server standing in for OpenBao, so the tests
    /// exercise the real client — headers, statuses, JSON shapes — rather
    /// than a mock of it.
    async fn spawn_fake(handler: Handler) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |req: Request<Incoming>| {
                        let handler = Arc::clone(&handler);
                        async move {
                            let method = req.method().to_string();
                            let path = req.uri().path().to_string();
                            let token = req
                                .headers()
                                .get("x-vault-token")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("")
                                .to_string();
                            let body = req
                                .into_body()
                                .collect()
                                .await
                                .map(|b| b.to_bytes())
                                .unwrap_or_default();
                            let body = String::from_utf8_lossy(&body).to_string();
                            // The token travels in a header; pass it to the
                            // handler alongside the body so a test can assert
                            // on it.
                            let combined = format!("{token}\n{body}");
                            let (status, out) = handler(&method, &path, &combined);
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(status)
                                    .body(Full::new(Bytes::from(out)))
                                    .unwrap(),
                            )
                        }
                    });
                    let _ = http1::Builder::new().serve_connection(io, service).await;
                });
            }
        });
        addr
    }

    fn kv_body(fields: &[(&str, &str)]) -> String {
        let data: serde_json::Map<String, Value> = fields
            .iter()
            .map(|(k, v)| ((*k).to_string(), Value::String((*v).to_string())))
            .collect();
        serde_json::json!({ "data": { "data": data, "metadata": { "version": 1 } } }).to_string()
    }

    fn config_for(addr: SocketAddr) -> OpenBaoConfig {
        OpenBaoConfig::new(
            format!("http://{addr}"),
            OpenBaoAuth::TokenFile("unused-in-this-test".to_string()),
        )
    }

    /// The entity's own values are laid over the tenant-wide defaults, key by
    /// key — that overlay is what gets injected.
    #[tokio::test]
    async fn an_entitys_own_secret_overrides_the_tenant_default() {
        let addr = spawn_fake(Arc::new(
            |_method: &str, path: &str, _body: &str| match path {
                "/v1/secret/data/sasy/acme/_default/openai" => (
                    200,
                    kv_body(&[("api_key", "shared"), ("org_id", "org-acme")]),
                ),
                "/v1/secret/data/sasy/acme/agent/openai" => (200, kv_body(&[("api_key", "mine")])),
                _ => (404, "{}".to_string()),
            },
        ))
        .await;

        let source = source_for_test(config_for(addr), "test-token");
        let got = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(got.values.get("api_key").map(String::as_str), Some("mine"));
        assert_eq!(
            got.values.get("org_id").map(String::as_str),
            Some("org-acme"),
            "a default the entity does not override still applies"
        );

        // The own view never reads the defaults path at all.
        let own = source
            .resolve("acme", "agent", "openai", CredentialView::Own)
            .await
            .unwrap();
        assert_eq!(own.values.get("api_key").map(String::as_str), Some("mine"));
        assert!(!own.values.contains_key("org_id"));
    }

    /// A path that holds nothing is an entity with no credentials, not a
    /// failure.
    #[tokio::test]
    async fn a_missing_path_resolves_to_no_credentials() {
        let addr = spawn_fake(Arc::new(|_m: &str, _p: &str, _b: &str| {
            (404, r#"{"errors":[]}"#.to_string())
        }))
        .await;
        let source = source_for_test(config_for(addr), "test-token");
        let got = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(got.values.is_empty());
    }

    /// A path this process is not allowed to read is an error, so injection
    /// fails closed rather than silently proceeding without the credential.
    #[tokio::test]
    async fn a_denied_path_is_an_error() {
        let addr = spawn_fake(Arc::new(|_m: &str, _p: &str, _b: &str| {
            (403, r#"{"errors":["permission denied"]}"#.to_string())
        }))
        .await;
        let source = source_for_test(config_for(addr), "test-token");
        let err = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .expect_err("a 403 must not read as 'no credentials'");
        let msg = format!("{err}");
        assert!(msg.contains("denied"), "got: {msg}");
    }

    /// The token from the file is what the requests carry, and a server that
    /// rejects it stops the process from starting.
    #[tokio::test]
    async fn a_token_file_authenticates_every_request() {
        let seen = Arc::new(AtomicUsize::new(0));
        let seen_in_handler = Arc::clone(&seen);
        let addr = spawn_fake(Arc::new(move |_m: &str, path: &str, body: &str| {
            let token = body.lines().next().unwrap_or("");
            if token != "file-token" {
                return (403, r#"{"errors":["permission denied"]}"#.to_string());
            }
            if path == "/v1/auth/token/lookup-self" {
                seen_in_handler.fetch_add(1, Ordering::SeqCst);
                return (
                    200,
                    serde_json::json!({ "data": { "ttl": 0, "renewable": false } }).to_string(),
                );
            }
            (200, kv_body(&[("api_key", "from-bao")]))
        }))
        .await;

        let dir = tempfile::TempDir::new().unwrap();
        let token_path = dir.path().join("token");
        std::fs::write(&token_path, "file-token\n").unwrap();
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::TokenFile(token_path.to_string_lossy().into_owned());

        let source = OpenBaoSource::connect(config.clone()).await.unwrap();
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "the token is checked once, at startup"
        );
        let got = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("from-bao")
        );

        // A token the server does not accept is a refusal to start.
        std::fs::write(&token_path, "wrong-token\n").unwrap();
        let err = OpenBaoSource::connect(config)
            .await
            .map(|_| ())
            .expect_err("a rejected token must stop the process from starting");
        assert!(format!("{err}").contains("rejected"), "got: {err}");
    }

    /// Authentication material that cannot be read is a refusal to start, and
    /// the reason names the file rather than its contents.
    ///
    /// An empty file is the shape a mis-mounted secret volume
    /// takes: present, so a check for existence passes, and useless. Reading
    /// it as a token would produce a process that starts healthy and then
    /// fails every credential lookup at request time, which is the outcome
    /// starting-time authentication exists to prevent.
    #[tokio::test]
    async fn a_token_file_that_cannot_be_read_stops_the_process_from_starting() {
        let touched = Arc::new(AtomicUsize::new(0));
        let touched_in_handler = Arc::clone(&touched);
        let addr = spawn_fake(Arc::new(move |_m: &str, _p: &str, _b: &str| {
            touched_in_handler.fetch_add(1, Ordering::SeqCst);
            (
                200,
                serde_json::json!({ "data": { "ttl": 0, "renewable": false } }).to_string(),
            )
        }))
        .await;

        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("not-mounted");
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::TokenFile(missing.to_string_lossy().into_owned());
        let err = OpenBaoSource::connect(config.clone())
            .await
            .map(|_| ())
            .expect_err("a token file that is not there must stop the process");
        let msg = format!("{err}");
        assert!(
            msg.contains(&missing.to_string_lossy().into_owned()),
            "the error names the file that could not be read: {msg}"
        );

        let blank = dir.path().join("mounted-but-empty");
        std::fs::write(&blank, "  \n").unwrap();
        config.auth = OpenBaoAuth::TokenFile(blank.to_string_lossy().into_owned());
        let err = OpenBaoSource::connect(config)
            .await
            .map(|_| ())
            .expect_err("an empty token file must stop the process, not read as a token");
        assert!(format!("{err}").contains("empty"), "got: {err}");

        assert_eq!(
            touched.load(Ordering::SeqCst),
            0,
            "neither case ever reached the server"
        );
    }

    /// A secret id the server will not accept is a refusal to start, and the
    /// refusal carries nothing of what was sent.
    #[tokio::test]
    async fn an_approle_login_the_server_rejects_stops_the_process_from_starting() {
        let addr = spawn_fake(Arc::new(|_m: &str, path: &str, body: &str| {
            if path != "/v1/auth/approle/login" {
                return (404, "{}".to_string());
            }
            let payload: Value =
                serde_json::from_str(body.split_once('\n').map(|(_, b)| b).unwrap_or("{}"))
                    .unwrap_or(Value::Null);
            if payload.get("secret_id").and_then(Value::as_str) != Some("right-secret") {
                // What a server sends back can echo the request.
                return (
                    400,
                    r#"{"errors":["invalid secret id: stale-secret-id"]}"#.to_string(),
                );
            }
            (
                200,
                serde_json::json!({
                    "auth": { "client_token": "t", "lease_duration": 0, "renewable": false }
                })
                .to_string(),
            )
        }))
        .await;

        let dir = tempfile::TempDir::new().unwrap();
        let secret_path = dir.path().join("secret-id");
        std::fs::write(&secret_path, "stale-secret-id\n").unwrap();
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::AppRole {
            role_id: "role-1".to_string(),
            secret_id_file: secret_path.to_string_lossy().into_owned(),
        };

        let err = OpenBaoSource::connect(config.clone())
            .await
            .map(|_| ())
            .expect_err("a rejected login must stop the process from starting");
        let msg = format!("{err}");
        assert!(
            msg.contains("rejected the AppRole login"),
            "the error says what was refused: {msg}"
        );
        assert!(
            !msg.contains("stale-secret-id"),
            "no authentication material travels in the error: {msg}"
        );

        // And the same configuration with the secret the server accepts
        // starts, so the test above fails for the reason it claims.
        std::fs::write(&secret_path, "right-secret\n").unwrap();
        OpenBaoSource::connect(config)
            .await
            .map(|_| ())
            .expect("the accepted secret id must authenticate");
    }

    /// AppRole exchanges its role id and secret id for a token at startup,
    /// and gets another one when the server refuses to renew the one it has.
    #[tokio::test]
    async fn approle_logs_in_again_when_a_renewal_is_refused() {
        let logins = Arc::new(AtomicUsize::new(0));
        let logins_in_handler = Arc::clone(&logins);
        let addr = spawn_fake(Arc::new(move |_m: &str, path: &str, body: &str| {
            match path {
                "/v1/auth/approle/login" => {
                    let payload: Value =
                        serde_json::from_str(body.split_once('\n').map(|(_, b)| b).unwrap_or("{}"))
                            .unwrap_or(Value::Null);
                    if payload.get("role_id").and_then(Value::as_str) != Some("role-1")
                        || payload.get("secret_id").and_then(Value::as_str) != Some("secret-1")
                    {
                        return (
                            400,
                            r#"{"errors":["invalid role or secret id"]}"#.to_string(),
                        );
                    }
                    let n = logins_in_handler.fetch_add(1, Ordering::SeqCst) + 1;
                    (
                        200,
                        serde_json::json!({
                            "auth": {
                                "client_token": format!("token-{n}"),
                                // One second, so the renewal is due almost
                                // immediately and the test does not have to
                                // wait out a realistic lease.
                                "lease_duration": 1,
                                "renewable": true,
                            }
                        })
                        .to_string(),
                    )
                }
                // The token cannot be extended: it is gone.
                "/v1/auth/token/renew-self" => {
                    (403, r#"{"errors":["permission denied"]}"#.to_string())
                }
                _ => (200, kv_body(&[("api_key", "after-relogin")])),
            }
        }))
        .await;

        let dir = tempfile::TempDir::new().unwrap();
        let secret_path = dir.path().join("secret-id");
        std::fs::write(&secret_path, "secret-1\n").unwrap();
        let mut config = config_for(addr);
        config.auth = OpenBaoAuth::AppRole {
            role_id: "role-1".to_string(),
            secret_id_file: secret_path.to_string_lossy().into_owned(),
        };

        let source = OpenBaoSource::connect(config).await.unwrap();
        assert_eq!(logins.load(Ordering::SeqCst), 1, "startup logs in once");
        assert_eq!(source.token.read().await.token, "token-1");

        // Wait out the lease (renewal is scheduled at two thirds of it, or
        // one second, whichever is longer), then read: the renewal is refused
        // and a fresh login takes its place.
        tokio::time::sleep(Duration::from_millis(1300)).await;
        let got = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("after-relogin")
        );
        assert_eq!(
            logins.load(Ordering::SeqCst),
            2,
            "a refused renewal is followed by exactly one new login"
        );
        assert_eq!(source.token.read().await.token, "token-2");
    }

    /// The deadline is what makes a rotated secret reach the next request.
    ///
    /// A reader keeps a resolved credential while `is_current` says so and
    /// reads again when it does not. This walks that loop against a server
    /// that counts requests and changes its answer, so the test fails both
    /// when an expired entry is reused and when the re-read never happens.
    #[tokio::test]
    async fn an_expired_credential_is_read_again_and_picks_up_the_new_value() {
        let reads = Arc::new(AtomicUsize::new(0));
        let reads_in_handler = Arc::clone(&reads);
        let addr = spawn_fake(Arc::new(move |_m: &str, path: &str, _b: &str| {
            if !path.ends_with("/agent/openai") {
                // The tenant-wide defaults path: empty, and not counted, so
                // the count is exactly the entity reads.
                return (404, "{}".to_string());
            }
            let n = reads_in_handler.fetch_add(1, Ordering::SeqCst) + 1;
            // The secret is rotated between the first read and the second.
            (
                200,
                kv_body(&[("api_key", if n == 1 { "old" } else { "new" })]),
            )
        }))
        .await;

        // Stands in for what the reference monitor's cache does: reuse while
        // the source says the entry is current, otherwise read again.
        async fn read_through(
            source: &OpenBaoSource,
            held: &mut Option<ResolvedCredentials>,
        ) -> String {
            if let Some(entry) = held {
                if source.is_current(&entry.freshness) {
                    return entry.values["api_key"].clone();
                }
            }
            let fresh = source
                .resolve("acme", "agent", "openai", CredentialView::Relay)
                .await
                .unwrap();
            let value = fresh.values["api_key"].clone();
            *held = Some(fresh);
            value
        }

        let mut config = config_for(addr);
        config.cache_ttl = Duration::from_secs(60);
        let long_lived = source_for_test(config, "test-token");
        let mut held = None;
        assert_eq!(read_through(&long_lived, &mut held).await, "old");
        assert_eq!(read_through(&long_lived, &mut held).await, "old");
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "inside the lifetime the credential is reused, not re-read"
        );

        let mut config = config_for(addr);
        config.cache_ttl = Duration::ZERO;
        let expiring = source_for_test(config, "test-token");
        let mut held = None;
        assert_eq!(read_through(&expiring, &mut held).await, "new");
        assert_eq!(
            read_through(&expiring, &mut held).await,
            "new",
            "a spent credential is read again rather than reused"
        );
        assert_eq!(
            reads.load(Ordering::SeqCst),
            3,
            "each read past the deadline reaches the server"
        );
    }

    /// Resolved credentials carry a deadline, and past it they are no longer
    /// current — which is what makes the reference monitor read again.
    #[tokio::test]
    async fn a_resolved_credential_stops_being_current_when_its_ttl_runs_out() {
        let addr = spawn_fake(Arc::new(|_m: &str, _p: &str, _b: &str| {
            (200, kv_body(&[("api_key", "k")]))
        }))
        .await;

        let mut config = config_for(addr);
        config.cache_ttl = Duration::from_secs(60);
        let source = source_for_test(config, "test-token");
        let fresh = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(source.is_current(&fresh.freshness));

        let mut config = config_for(addr);
        config.cache_ttl = Duration::ZERO;
        let source = source_for_test(config, "test-token");
        let spent = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(
            !source.is_current(&spent.freshness),
            "a zero lifetime means every lookup reads the server again"
        );
    }

    /// This backend never writes, and says so in a way an operator can act
    /// on: the refusal names the address the credential should be written to
    /// instead.
    ///
    /// The kind of error matters as much as the fact of it — the gRPC layer
    /// turns `ReadOnly` into `FAILED_PRECONDITION` (a refused write) and
    /// everything else into an internal error (a failed one), and the
    /// difference is what tells an operator whether to look for a bug.
    #[tokio::test]
    async fn writing_to_this_backend_is_refused_and_says_where_to_write_instead() {
        let addr = spawn_fake(Arc::new(|_m: &str, _p: &str, _b: &str| {
            (200, kv_body(&[("api_key", "unused")]))
        }))
        .await;
        let source = source_for_test(config_for(addr), "test-token");

        let err = source
            .set_credentials(
                "acme",
                "agent",
                "openai",
                vec![("api_key".into(), "sk-nope".into())],
            )
            .await
            .expect_err("a read-only backend must refuse the write");
        assert!(
            matches!(err, CredentialError::ReadOnly(_)),
            "the write was refused, not attempted and failed: {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains(&format!("{addr}")) && msg.contains("read-only"),
            "the refusal must name the server the secret belongs in: {msg}"
        );
        assert!(
            !msg.contains("sk-nope"),
            "the value must not travel back in the error: {msg}"
        );
    }

    /// An entity cannot name itself into the reserved segment and collect the
    /// tenant's shared credentials, and no name may carry a path separator.
    #[tokio::test]
    async fn a_reserved_or_traversing_name_is_refused_before_any_request() {
        // Nothing may reach the server: bind a fake that fails the test if it
        // is contacted.
        let touched = Arc::new(AtomicUsize::new(0));
        let touched_in_handler = Arc::clone(&touched);
        let addr = spawn_fake(Arc::new(move |_m: &str, _p: &str, _b: &str| {
            touched_in_handler.fetch_add(1, Ordering::SeqCst);
            (200, kv_body(&[("api_key", "leaked")]))
        }))
        .await;
        let source = source_for_test(config_for(addr), "test-token");

        for entity in ["_default", "*"] {
            let err = source
                .resolve("acme", entity, "openai", CredentialView::Own)
                .await
                .unwrap_err();
            assert!(
                format!("{err}").contains("reserved"),
                "entity {entity:?}: {err}"
            );
        }
        for entity in ["..", "a/b", "a%2fb"] {
            source
                .resolve("acme", entity, "openai", CredentialView::Own)
                .await
                .expect_err("a name that is not a single path segment must be refused");
        }
        source
            .resolve("acme/../other", "agent", "openai", CredentialView::Own)
            .await
            .expect_err("a tenant that traverses must be refused");
        assert_eq!(
            touched.load(Ordering::SeqCst),
            0,
            "a refused name must never reach the server"
        );
    }

    /// The entity the HTTP forward proxy relays under resolves the tenant's
    /// defaults instead of being refused.
    ///
    /// `forward_proxy.rs` has no authenticated principal of its own and asks
    /// for every injection as the entity `*`. Refusing that name in the relay
    /// view — where the defaults are merged in anyway, so it asks for nothing
    /// it would not already be given — made every forward-proxy request that
    /// carried a transform fail with a reserved-name error, which is the
    /// whole path dead against this backend.
    #[tokio::test]
    async fn the_forward_proxys_wildcard_entity_reads_the_tenant_defaults() {
        let wildcard_paths = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&wildcard_paths);
        let addr = spawn_fake(Arc::new(move |_m: &str, path: &str, _b: &str| {
            if path.contains("/*/") {
                seen.fetch_add(1, Ordering::SeqCst);
                return (404, "{}".to_string());
            }
            match path {
                "/v1/secret/data/sasy/acme/_default/openai" => {
                    (200, kv_body(&[("api_key", "tenant-wide")]))
                }
                _ => (404, "{}".to_string()),
            }
        }))
        .await;
        let source = source_for_test(config_for(addr), "test-token");

        let got = source
            .resolve("acme", WILDCARD_ENTITY, "openai", CredentialView::Relay)
            .await
            .expect("the relay's own entity must resolve, not be refused");
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("tenant-wide")
        );
        assert_eq!(
            wildcard_paths.load(Ordering::SeqCst),
            0,
            "`*` is not a directory in OpenBao; no path is built out of it"
        );

        // The own view is what the reservation protects, and it still
        // refuses: a credential-reader may not collect the tenant's shared
        // secrets by calling itself `*`.
        let err = source
            .resolve("acme", WILDCARD_ENTITY, "openai", CredentialView::Own)
            .await
            .expect_err("the own view must still refuse the reserved name");
        assert!(format!("{err}").contains("reserved"), "got: {err}");
    }

    /// A token must not travel in the clear to another host.
    #[test]
    fn plain_http_is_only_allowed_to_loopback() {
        vet_address("https://bao.example.com:8200").expect("https anywhere");
        vet_address("http://127.0.0.1:8200").expect("loopback development server");
        vet_address("http://localhost:8200").expect("loopback by name");
        vet_address("http://[::1]:8200").expect("loopback over IPv6");

        let err = vet_address("http://bao.example.com:8200")
            .expect_err("plain HTTP off the host must be refused");
        assert!(format!("{err}").contains("clear"), "got: {err}");
        vet_address("ftp://bao.example.com").expect_err("only http(s) is understood");
        vet_address("not a url").expect_err("an unparseable address is refused");
    }

    /// The same path layout, against a real server.
    ///
    /// Run one with:
    ///
    /// ```text
    /// docker run --rm -p 8200:8200 -e BAO_DEV_ROOT_TOKEN_ID=dev-root \
    ///     openbao/openbao server -dev -dev-listen-address=0.0.0.0:8200
    /// ```
    ///
    /// then seed a secret and run this test:
    ///
    /// ```text
    /// docker exec -e BAO_ADDR=http://127.0.0.1:8200 -e BAO_TOKEN=dev-root <id> \
    ///     bao kv put secret/sasy/acme/_default/openai api_key=from-real-bao
    /// OPENBAO_ADDR=http://127.0.0.1:8200 OPENBAO_TOKEN=dev-root \
    ///     cargo test -p sasy-credential -- --ignored openbao
    /// ```
    #[tokio::test]
    #[ignore = "needs a real OpenBao; set OPENBAO_ADDR and OPENBAO_TOKEN"]
    async fn it_reads_a_secret_from_a_real_openbao() {
        let (Ok(addr), Ok(token)) = (
            std::env::var("OPENBAO_ADDR"),
            std::env::var("OPENBAO_TOKEN"),
        ) else {
            panic!("set OPENBAO_ADDR and OPENBAO_TOKEN to run this test");
        };
        let dir = tempfile::TempDir::new().unwrap();
        let token_path = dir.path().join("token");
        std::fs::write(&token_path, token).unwrap();

        let config = OpenBaoConfig::new(
            addr,
            OpenBaoAuth::TokenFile(token_path.to_string_lossy().into_owned()),
        );
        let source = OpenBaoSource::connect(config).await.unwrap();
        let got = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("from-real-bao"),
            "seed the secret first (see the comment on this test)"
        );
    }
}
