//! Transform executor — applies credential-injecting
//! transforms to HTTP requests based on policy engine
//! output.

use std::collections::HashMap;
use std::sync::Arc;

use sasy_credential::{CredentialSource, CredentialView, Freshness};
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::error::RefmonError;

/// The whole reason a caller is told when a credential transform fails.
///
/// The backend's own message names the secret manager's address, the mount
/// and the path it tried to read, plus the tenant/entity/service mapping that
/// produced them. The caller of the proxy is the agent the reference monitor
/// exists to police, so it gets this fixed sentence and nothing else; the
/// backend's message goes to the server log at `warn` level, where the
/// operator can read it.
pub const TRANSFORM_FAILURE_REASON: &str = "credential transform failed";

// ── Transform types ────────────────────────────────────

/// A transform that modifies an outgoing HTTP request.
#[derive(Debug, Clone)]
pub enum Transform {
    /// Add (or overwrite) an HTTP header.
    AddHeader {
        key: String,
        format_str: String,
        service: String,
    },
    /// Append a URL query parameter.
    AddUrlParam {
        parameter: String,
        format_str: String,
        service: String,
    },
}

// ── Transform config (loaded from JSON) ────────────────

/// Raw JSON schema for a single transform entry.
#[derive(Debug, Deserialize)]
struct RawTransform {
    #[serde(rename = "type")]
    kind: String,
    key: String,
    format: String,
    #[serde(default)]
    service: String,
}

#[derive(Debug, Deserialize)]
struct RawTransformFile {
    #[serde(default)]
    transforms: HashMap<String, RawTransform>,
}

/// Loaded transform configuration keyed by transform ID.
#[derive(Debug, Clone, Default)]
pub struct TransformConfig {
    pub transforms: HashMap<String, Transform>,
}

impl TransformConfig {
    /// Load transforms from a JSON file.
    pub fn load(path: &str) -> Result<Self, RefmonError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| RefmonError::Config(format!("read {path}: {e}")))?;
        let raw: RawTransformFile = serde_json::from_str(&content)
            .map_err(|e| RefmonError::Config(format!("parse {path}: {e}")))?;

        let mut transforms = HashMap::new();
        for (id, rt) in raw.transforms {
            let t = match rt.kind.as_str() {
                "add_header" => Transform::AddHeader {
                    key: rt.key,
                    format_str: rt.format,
                    service: rt.service,
                },
                "add_url_param" => Transform::AddUrlParam {
                    parameter: rt.key,
                    format_str: rt.format,
                    service: rt.service,
                },
                other => {
                    return Err(RefmonError::Config(format!(
                        "unknown transform type: \
                             {other}"
                    )));
                }
            };
            transforms.insert(id, t);
        }

        Ok(Self { transforms })
    }
}

// ── Transform executor ─────────────────────────────────

/// Maximum number of (tenant, entity, service) tuples cached.
/// When exceeded the cache is cleared (simple eviction).
const MAX_CACHE_SIZE: usize = 1024;

/// Executes transforms on HTTP requests, fetching
/// credentials from the configured [`CredentialSource`] as needed.
///
/// Cache and backend lookups are keyed on the auth-derived tenant
/// so two tenants that happen to register the same
/// `entity` name see disjoint credentials.
pub struct TransformExecutor {
    config: TransformConfig,
    credential_source: Arc<dyn CredentialSource>,
    #[allow(clippy::type_complexity)]
    // inherent: cache keyed by (tenant, entity, service) → field map, so two
    // transforms that read the same service share one lookup
    /// Resolved credentials, with the freshness token they were read with.
    /// The backend decides whether that token still describes what it holds —
    /// see `get_credentials`.
    cache: RwLock<HashMap<(String, String, String), (Freshness, HashMap<String, String>)>>,
}

impl TransformExecutor {
    pub fn new(config: TransformConfig, credential_source: Arc<dyn CredentialSource>) -> Self {
        Self {
            config,
            credential_source,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Apply the listed transforms (by ID) to the
    /// mutable request parts.
    ///
    /// `headers` and `url` are modified in place. Every listed transform the
    /// configuration defines either produces its value or the whole call
    /// fails, so a caller that denies on `Err` never proxies a request the
    /// policy authorized only on condition of the injection.
    /// If `entity` is `None` there is nothing to look a credential up under,
    /// which is an error unless no transforms were attached. `tenant` is the
    /// auth-derived tenant for this request and partitions the credential
    /// store.
    ///
    /// Once anything has been injected, `url` also goes through
    /// [`require_tls_leg`], so the connection that carries the credential is
    /// TLS. A request that had nothing injected is left exactly as it came in.
    pub async fn execute(
        &self,
        headers: &mut Vec<(String, String)>,
        url: &mut String,
        transform_ids: &[String],
        tenant: &str,
        entity: Option<&str>,
    ) -> Result<(), RefmonError> {
        let entity = match entity {
            Some(e) => e,
            // No entity means no credential lookup is possible. With nothing
            // to inject there is nothing to skip; with transforms attached,
            // the policy authorized this call ON CONDITION that they were
            // applied, so proceeding would send it upstream uncredentialed —
            // or under whatever identity the caller put in the header the
            // transform was meant to replace.
            None if transform_ids.is_empty() => return Ok(()),
            None => {
                return Err(RefmonError::Transform(format!(
                    "{} transform(s) attached but the request has no entity to \
                     resolve credentials for",
                    transform_ids.len()
                )));
            }
        };

        let mut applied = 0usize;
        for tid in transform_ids {
            let transform = match self.config.transforms.get(tid) {
                Some(t) => t,
                None => {
                    // Skipping withholds the platform credential, and that is
                    // all it does. Both types here replace what the caller put
                    // under the same name before writing their own value, so a
                    // skipped one leaves the caller's header or parameter in
                    // place: the request goes upstream under the caller's own
                    // identity, or — if it supplied none — uncredentialed, and
                    // the upstream rejects it. Tolerable because the platform
                    // credential is neither exposed nor spent, and the policy's
                    // host-level authorization already applied. A sanitizing
                    // type (redaction, header stripping) added later must DENY
                    // on a missing definition — skipping it forwards exactly
                    // what it was to remove.
                    warn!(
                        transform_id = %tid,
                        "unknown transform ID, skipping"
                    );
                    continue;
                }
            };
            self.apply(transform, headers, url, tenant, entity).await?;
            applied += 1;
        }
        if applied > 0 {
            require_tls_leg(url)?;
        }
        Ok(())
    }

    async fn apply(
        &self,
        transform: &Transform,
        headers: &mut Vec<(String, String)>,
        url: &mut String,
        tenant: &str,
        entity: &str,
    ) -> Result<(), RefmonError> {
        match transform {
            Transform::AddHeader {
                key,
                format_str,
                service,
            } => {
                let creds = self.get_credentials(tenant, entity, service).await?;
                // Missing credentials must fail closed. Continuing would leave
                // a caller-supplied header in place of the required credential.
                let value = interpolate(format_str, &creds).map_err(|missing| {
                    RefmonError::Transform(format!(
                        "the {service} credential has no {missing:?}, which the \
                         {key} header needs"
                    ))
                })?;
                let key_lower = key.to_lowercase();
                headers.retain(|(k, _)| k.to_lowercase() != key_lower);
                headers.push((key.clone(), value));
            }
            Transform::AddUrlParam {
                parameter,
                format_str,
                service,
            } => {
                let creds = self.get_credentials(tenant, entity, service).await?;
                let value = interpolate(format_str, &creds).map_err(|missing| {
                    RefmonError::Transform(format!(
                        "the {service} credential has no {missing:?}, which the \
                         {parameter} query parameter needs"
                    ))
                })?;
                append_query_param(url, parameter, &value)?;
            }
        }
        Ok(())
    }

    async fn get_credentials(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
    ) -> Result<HashMap<String, String>, RefmonError> {
        let cache_key = (tenant.to_string(), entity.to_string(), service.to_string());

        // Ask the backend whether what we hold is still current. Without this
        // the cache had no expiry and no invalidation hook, so a key rotated
        // or REVOKED kept being injected into authorized upstream calls until
        // 1024 distinct tuples accumulated or the process restarted.
        // Revocation that does not take effect is the failure that matters
        // about a rotation.
        //
        // What "current" means is the backend's to decide: the file-backed
        // store compares a version anchored in the database, so it also
        // catches writes from another process (`init-credentials`, the
        // documented way to load credentials, is one); a remote secret
        // manager, which cannot be watched, gives a deadline instead.
        {
            let cache = self.cache.read().await;
            if let Some((freshness, creds)) = cache.get(&cache_key) {
                if self.credential_source.is_current(freshness) {
                    return Ok(creds.clone());
                }
            }
        }

        // Read through the backend: the tenant-wide defaults with the
        // entity's own entries overlaid, which is what gets injected.
        let resolved = self
            .credential_source
            .resolve(tenant, entity, service, CredentialView::Relay)
            .await?;

        // Cache under write lock; evict if at capacity.
        {
            let mut cache = self.cache.write().await;
            if cache.len() >= MAX_CACHE_SIZE {
                debug!(
                    "credential cache full ({MAX_CACHE_SIZE}), \
                     clearing"
                );
                cache.clear();
            }
            cache.insert(cache_key, (resolved.freshness, resolved.values.clone()));
        }

        Ok(resolved.values)
    }
}

// ── Helpers ────────────────────────────────────────────

/// Simple `{key}` interpolation against a credential map.
///
/// Returns the name of the first placeholder the credential does not carry —
/// absent, or present and empty, which are the same thing to whatever is
/// being authenticated — as `Err`, so the caller can say which field is
/// missing. The name is a credential KEY (`api_key`), never a value.
fn interpolate(format_str: &str, creds: &HashMap<String, String>) -> Result<String, String> {
    let mut result = format_str.to_string();
    // Scan forward from the end of each substitution rather than restarting
    // at the beginning. Restarting re-read the value just written, so a
    // credential whose value contains braces was interpreted as a
    // placeholder: `{api_key}` stored as the value of `api_key` looped
    // forever, hanging the request thread, and a value like `{org_id}`
    // silently spliced in a second credential. A substituted value is data,
    // not template.
    let mut from = 0usize;
    while let Some(rel_start) = result[from..].find('{') {
        let start = from + rel_start;
        let end = match result[start..].find('}') {
            Some(i) => start + i,
            None => break,
        };
        let key = &result[start + 1..end];
        // An empty value counts as no value. A blank string interpolates
        // without complaint, and the request then goes upstream carrying
        // `Authorization: Bearer ` or `?api_key=` — the caller's own header
        // having been stripped — which an API that treats a blank key as
        // anonymous will serve. That is the same "uncredentialed request the
        // policy authorized as a credentialed one" this path exists to
        // prevent, so it is the same denial. The backends can each produce
        // one: a `Credential` message whose value field is unset, an OpenBao
        // field holding `""`.
        let value = creds
            .get(key)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| key.to_string())?;
        let resume = start + value.len();
        result = format!("{}{}{}", &result[..start], value, &result[end + 1..]);
        from = resume;
    }
    Ok(result)
}

/// Require TLS on the outbound leg that carries an injected credential.
///
/// A client may send absolute-form HTTP URLs over the loopback proxy connection
/// so the proxy can inspect and transform them; CONNECT tunnels are opaque.
/// Upgrade targets with no port or port 80 to HTTPS/443. Reject explicit
/// non-default HTTP ports rather than guessing whether they support TLS.
///
/// HTTPS targets stay unchanged. Invalid URLs fail here; other schemes are
/// rejected by `net_guard::resolve_public_target`. Both outbound clients disable
/// redirects to prevent a later downgrade.
fn require_tls_leg(url: &mut String) -> Result<(), RefmonError> {
    let mut parsed = url::Url::parse(url).map_err(|e| {
        RefmonError::Transform(format!(
            "the request URL cannot be parsed, so it cannot be confirmed to \
             carry the credential over TLS: {e}"
        ))
    })?;
    if parsed.scheme() != "http" {
        return Ok(());
    }
    match parsed.port() {
        None | Some(80) => {
            parsed.set_scheme("https").map_err(|_| {
                RefmonError::Transform(
                    "the request URL cannot be upgraded to HTTPS, so the \
                     credential would go out in the clear"
                        .to_string(),
                )
            })?;
            // An explicit `:80` survives the scheme change and would keep the
            // request on the plaintext port; the HTTPS default is 443.
            if parsed.port() == Some(80) {
                parsed.set_port(None).map_err(|_| {
                    RefmonError::Transform(
                        "the request URL keeps port 80 after the HTTPS \
                         upgrade, so the credential would go out in the clear"
                            .to_string(),
                    )
                })?;
            }
            *url = parsed.to_string();
            Ok(())
        }
        Some(port) => Err(RefmonError::Transform(format!(
            "the credential cannot be sent over plain HTTP to port {port}; \
             use https for this target"
        ))),
    }
}

/// Put a query parameter into a URL, replacing any parameter the caller
/// already supplied under the same name.
///
/// The URL is parsed rather than edited as a string. Editing the string
/// appended to the very end, which is the wrong place whenever the caller's
/// URL carries a fragment (the `#…` tail): `https://host/p?a=1#` became
/// `https://host/p?a=1#&api_key=…`, where the parameter sits inside the
/// fragment. A fragment is never sent to the server, so the request went
/// upstream with no credential while the transform reported success.
///
/// An existing parameter of the same name is dropped before the injected one
/// is appended, the way [`Transform::AddHeader`] strips the caller's header.
/// Left in place, a caller-supplied `api_key` would be the *first* occurrence,
/// and upstreams that read the first occurrence would bill the call against
/// the caller's key rather than the platform credential the policy attached.
/// The match is ASCII case-insensitive, again like the header strip: an
/// upstream that reads `API_KEY` would otherwise still see the caller's value,
/// sitting beside the injected `api_key`.
///
/// The caller's other parameters are moved across as raw text, byte for byte.
/// Reading them out as decoded pairs and writing them back re-encoded rewrote
/// the caller's own query even when nothing collided: a valueless `?verbose`
/// came out as `verbose=`, a `%20` as `+`, and a `%FF` — a byte that is not
/// text — as the replacement character, changing what the upstream was asked
/// for. Only the injected pair is encoded (with
/// `application/x-www-form-urlencoded` rules, so `&`, `=` or a space in a
/// credential cannot corrupt the URL structure). The fragment is left alone.
///
/// Raw semicolons in the existing query are refused: some upstream parsers
/// treat them as separators, so an ampersand-only replacement cannot establish
/// which parameters will reach those parsers. Encoded value bytes are retained.
/// A URL that does not parse is an error, which the call sites turn into a
/// denial: there is no safe place to put the credential in a URL we cannot
/// read.
fn append_query_param(url: &mut String, key: &str, value: &str) -> Result<(), RefmonError> {
    let mut parsed = url::Url::parse(url).map_err(|e| {
        RefmonError::Transform(format!(
            "the request URL cannot be parsed, so the {key} query parameter \
             cannot be placed in it: {e}"
        ))
    })?;

    // Keep the caller's segments verbatim, minus any whose name is the one
    // being injected. A segment's name is compared decoded — `api%5Fkey` is
    // `api_key` to the upstream — but kept as it was written.
    let mut segments = Vec::new();
    if let Some(existing) = parsed.query() {
        if existing.contains(';') {
            return Err(RefmonError::Transform(
                "query parameter injection requires an unambiguous ampersand-separated query"
                    .to_string(),
            ));
        }
        for segment in existing.split('&') {
            let name = url::form_urlencoded::parse(segment.as_bytes())
                .next()
                .map(|(name, _)| name.into_owned())
                .unwrap_or_default();
            if name.eq_ignore_ascii_case(key) {
                continue;
            }
            segments.push(segment);
        }
    }
    let mut query = segments.join("&");
    if !segments.is_empty() {
        query.push('&');
    }
    query.extend(url::form_urlencoded::byte_serialize(key.as_bytes()));
    query.push('=');
    query.extend(url::form_urlencoded::byte_serialize(value.as_bytes()));

    parsed.set_query(Some(&query));
    *url = parsed.to_string();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::injecting_transform_config;
    use sasy_credential::MemorySource;

    /// A substituted value is data, not template.
    ///
    /// Rescanning from the start re-read what had just been written, so a
    /// credential value containing braces was interpreted again: a
    /// self-referential one looped forever and hung the request thread, and
    /// one naming another key silently spliced that credential in.
    /// Run `interpolate` with a deadline, so the regression this guards shows
    /// up as a failure rather than a hung test run: the defect is an infinite
    /// loop, and a test that hangs is indistinguishable from one that never
    /// built.
    fn interpolate_bounded(
        format_str: &'static str,
        creds: HashMap<String, String>,
    ) -> Result<String, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(interpolate(format_str, &creds));
        });
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("interpolate did not terminate — it re-read its own substitution")
    }

    /// The cache is keyed by service, not by transform.
    ///
    /// Two transforms that read the same service resolve one credential
    /// between them: the second is served from the cache entry the first
    /// wrote. Keying by transform would look up the same secret twice.
    #[tokio::test]
    async fn two_transforms_on_one_service_share_a_cache_entry() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingSource {
            resolves: AtomicUsize,
        }

        #[tonic::async_trait]
        impl CredentialSource for CountingSource {
            fn backend(&self) -> &'static str {
                "test"
            }
            fn location(&self) -> String {
                "counting".to_string()
            }
            async fn resolve(
                &self,
                _tenant: &str,
                _entity: &str,
                _service: &str,
                _view: sasy_credential::CredentialView,
            ) -> Result<sasy_credential::ResolvedCredentials, sasy_credential::CredentialError>
            {
                self.resolves.fetch_add(1, Ordering::SeqCst);
                Ok(sasy_credential::ResolvedCredentials {
                    values: HashMap::from([("api_key".to_string(), "sk-test".to_string())]),
                    freshness: Freshness::Until(
                        std::time::Instant::now() + std::time::Duration::from_secs(60),
                    ),
                })
            }
            fn is_current(&self, _freshness: &Freshness) -> bool {
                true
            }
            async fn set_credentials(
                &self,
                _tenant: &str,
                _entity: &str,
                _service: &str,
                _credentials: Vec<(String, String)>,
            ) -> Result<(), sasy_credential::CredentialError> {
                unreachable!("this test never writes")
            }
        }

        let source = Arc::new(CountingSource {
            resolves: AtomicUsize::new(0),
        });
        let mut config = TransformConfig::default();
        config.transforms.insert(
            "header_from_openai".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );
        config.transforms.insert(
            "param_from_openai".into(),
            Transform::AddUrlParam {
                parameter: "key".into(),
                format_str: "{api_key}".into(),
                service: "openai".into(),
            },
        );
        let executor = TransformExecutor::new(config, source.clone() as Arc<dyn CredentialSource>);

        let mut headers = vec![];
        let mut url = "https://192.0.2.1/v1/chat".to_string();
        executor
            .execute(
                &mut headers,
                &mut url,
                &[
                    "header_from_openai".to_string(),
                    "param_from_openai".to_string(),
                ],
                "default",
                Some("alice"),
            )
            .await
            .expect("both transforms apply");

        assert_eq!(
            source.resolves.load(Ordering::SeqCst),
            1,
            "the second transform must reuse the entry keyed by the service"
        );
    }

    #[test]
    fn a_credential_value_containing_braces_is_not_re_interpolated() {
        let mut creds = HashMap::new();
        creds.insert("api_key".into(), "{api_key}".into());
        assert_eq!(
            interpolate_bounded("Bearer {api_key}", creds),
            Ok("Bearer {api_key}".into()),
            "a self-referential value must be emitted literally, not re-expanded"
        );

        let mut creds = HashMap::new();
        creds.insert("api_key".into(), "{org_id}".into());
        creds.insert("org_id".into(), "secret-org".into());
        assert_eq!(
            interpolate_bounded("Bearer {api_key}", creds),
            Ok("Bearer {org_id}".into()),
            "a value naming another key must not pull that credential in"
        );
    }

    #[test]
    fn test_interpolate() {
        let mut creds = HashMap::new();
        creds.insert("api_key".into(), "sk-123".into());
        assert_eq!(
            interpolate("Bearer {api_key}", &creds),
            Ok("Bearer sk-123".into())
        );
        assert_eq!(
            interpolate("{missing}", &creds),
            Err("missing".into()),
            "the missing placeholder is named, so the denial can say which field it was"
        );
        assert_eq!(
            interpolate("no-placeholders", &creds),
            Ok("no-placeholders".into())
        );
    }

    #[test]
    fn test_append_query_param() {
        let mut url = "https://api.example.com/v1".into();
        append_query_param(&mut url, "key", "val").unwrap();
        assert_eq!(url, "https://api.example.com/v1?key=val");

        append_query_param(&mut url, "k2", "v2").unwrap();
        assert_eq!(url, "https://api.example.com/v1?key=val&k2=v2");
    }

    #[test]
    fn query_injection_preserves_empty_segments_and_encoded_value_bytes() {
        for (original, expected) in [
            ("?&&verbose&", "?&&verbose&&api_key=K"),
            ("?&api_key=old&&q=a%3Bb&", "?&&q=a%3Bb&&api_key=K"),
            ("?", "?&api_key=K"),
        ] {
            let mut url = format!("https://api.example.com/{original}");
            append_query_param(&mut url, "api_key", "K").unwrap();
            assert_eq!(url, format!("https://api.example.com/{expected}"));
        }
    }

    #[test]
    fn query_injection_requires_unambiguous_separators() {
        let original = "https://api.example.com/?page=1;size=20";
        let mut url = original.to_string();
        assert!(append_query_param(&mut url, "api_key", "K").is_err());
        assert_eq!(url, original, "a refused injection leaves the URL intact");
    }

    #[test]
    fn tls_upgrade_preserves_https_ports_and_ipv6_authorities() {
        for (original, expected) in [
            (
                "https://api.example.com:8443/p?q=1",
                "https://api.example.com:8443/p?q=1",
            ),
            (
                "http://[2001:db8::1]:80/p?q=1",
                "https://[2001:db8::1]/p?q=1",
            ),
            ("HTTP://api.example.com:80/p", "https://api.example.com/p"),
        ] {
            let mut url = original.to_string();
            require_tls_leg(&mut url).unwrap();
            assert_eq!(url, expected);
            require_tls_leg(&mut url).unwrap();
            assert_eq!(
                url, expected,
                "applying the TLS requirement twice is stable"
            );
        }
    }

    /// The injected parameter goes in the query, never after the fragment.
    ///
    /// A gRPC `ProxyHTTP` caller supplies the URL verbatim, so it can carry a
    /// `#`. Appending to the raw string put the credential inside the
    /// fragment, which is never sent to the server: the request reached the
    /// upstream with no key, and nothing reported a problem.
    #[test]
    fn a_fragment_does_not_swallow_the_injected_parameter() {
        let mut url = "https://api.fda.gov/drug/event.json?search=x#".to_string();
        append_query_param(&mut url, "api_key", "K").unwrap();
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json?search=x&api_key=K#",
            "the parameter belongs in the query, ahead of the fragment"
        );

        let mut url = "https://api.fda.gov/drug/event.json#section".to_string();
        append_query_param(&mut url, "api_key", "K").unwrap();
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json?api_key=K#section",
            "the fragment itself is preserved"
        );
    }

    /// A URL the parser cannot read is an error, so the call site denies
    /// rather than proxying a request the credential never made it into.
    #[test]
    fn a_url_that_does_not_parse_is_an_error() {
        let mut url = "not a url".to_string();
        let err = append_query_param(&mut url, "api_key", "K")
            .expect_err("an unparsable URL must not read as a successful injection");
        let msg = format!("{err}");
        assert!(
            msg.contains("api_key"),
            "the failing parameter is named: {msg}"
        );
        assert!(
            !msg.contains('K'),
            "no credential value belongs in the error: {msg}"
        );
    }

    /// A caller-supplied parameter of the injected name is removed, the way
    /// `AddHeader` strips the caller's header.
    ///
    /// Left in place it would be the first occurrence, and an upstream that
    /// reads the first occurrence would bill the call against the caller's own
    /// key instead of the platform credential the policy attached.
    #[test]
    fn a_caller_supplied_parameter_of_the_same_name_is_replaced() {
        let mut url =
            "https://api.fda.gov/drug/event.json?search=x&api_key=attacker&limit=5".to_string();
        append_query_param(&mut url, "api_key", "platform").unwrap();
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json?search=x&limit=5&api_key=platform",
            "the caller's value is gone and the other parameters keep their order"
        );
        assert!(
            !url.contains("attacker"),
            "no second api_key may survive: {url}"
        );
    }

    /// The caller's parameter is replaced whatever case it was written in.
    ///
    /// `AddHeader` strips the caller's header case-insensitively, because HTTP
    /// header names are case-insensitive. Query parameter names are not, but
    /// plenty of upstreams read them that way — so an exact-case comparison
    /// left `?API_KEY=attacker` sitting beside the injected `api_key`, and an
    /// upstream that folds the name billed the call against the caller's key.
    #[test]
    fn the_caller_parameter_is_replaced_whatever_its_case() {
        let mut url = "https://api.fda.gov/drug/event.json?API_KEY=attacker&limit=5".to_string();
        append_query_param(&mut url, "api_key", "platform").unwrap();
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json?limit=5&api_key=platform",
            "the caller's differently-cased parameter is gone"
        );
        assert!(
            !url.contains("attacker"),
            "no second key of any case may survive: {url}"
        );
    }

    /// Everything the caller wrote in the query survives byte for byte.
    ///
    /// Reading the query out as decoded pairs and writing it back re-encoded
    /// rewrote parameters that had nothing to do with the injection: a
    /// valueless `?verbose` gained an `=`, a `%20` became a `+`, and `%FF` — a
    /// byte that is not text — became the replacement character. The upstream
    /// was then asked for something the caller had not asked for.
    #[test]
    fn the_caller_query_is_carried_across_byte_for_byte() {
        let mut url =
            "https://api.fda.gov/drug/event.json?verbose&search=a%20b&x=%FF#frag".to_string();
        append_query_param(&mut url, "api_key", "K").unwrap();
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json?verbose&search=a%20b&x=%FF&api_key=K#frag",
            "only the injected pair is added; the fragment stays too"
        );
    }

    /// Upgrade an authorized plain-HTTP target before sending credentials.
    #[test]
    fn a_credentialed_plain_http_target_is_upgraded_to_tls() {
        let mut url = "http://api.openai.com/v1/chat/completions".to_string();
        require_tls_leg(&mut url).unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");

        let mut url = "http://api.openai.com:80/v1/chat".to_string();
        require_tls_leg(&mut url).unwrap();
        assert_eq!(
            url, "https://api.openai.com/v1/chat",
            "an explicit port 80 must not survive the upgrade — it is the \
             plaintext port"
        );

        let mut url = "https://api.openai.com/v1/chat?a=1#f".to_string();
        require_tls_leg(&mut url).unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat?a=1#f", "already TLS");
    }

    /// A plain-HTTP target on a port that is not the HTTP default is refused:
    /// whether that port speaks TLS is a guess, and the fallback is not to send
    /// the credential in the clear.
    #[test]
    fn a_credentialed_plain_http_target_on_another_port_is_refused() {
        let mut url = "http://api.openai.com:8080/v1/chat".to_string();
        let err = require_tls_leg(&mut url).expect_err("plain HTTP on :8080 must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("8080") && msg.contains("plain HTTP"),
            "the refusal must name the port and the reason: {msg}"
        );
        assert_eq!(
            url, "http://api.openai.com:8080/v1/chat",
            "the URL is left as it was; the call site denies"
        );
    }

    /// The upgrade is part of injecting, so a request that had nothing injected
    /// is left exactly as it came in — including its scheme.
    #[tokio::test]
    async fn a_request_with_no_transform_applied_keeps_plain_http() {
        let store = Arc::new(MemorySource::new().unwrap());
        let executor = TransformExecutor::new(TransformConfig::default(), store);

        let mut headers = vec![];
        let mut url = "http://example.com/p".to_string();
        executor
            .execute(&mut headers, &mut url, &[], "default", Some("alice"))
            .await
            .unwrap();
        assert_eq!(url, "http://example.com/p", "no transforms, no change");

        // An unknown transform id is warned about and skipped, so nothing was
        // injected and there is no credential to protect.
        let mut url = "http://example.com/p".to_string();
        executor
            .execute(
                &mut headers,
                &mut url,
                &["not_configured".into()],
                "default",
                Some("alice"),
            )
            .await
            .unwrap();
        assert_eq!(
            url, "http://example.com/p",
            "a skipped transform injects nothing"
        );
    }

    /// Applying a header transform upgrades the URL too: the credential is in a
    /// header, but it is the connection that has to be encrypted.
    #[tokio::test]
    async fn injecting_a_header_upgrades_the_request_url() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials(
                "alice",
                "openai",
                vec![("api_key".into(), "sk-test".into())],
            )
            .unwrap();
        let executor = TransformExecutor::new(injecting_transform_config(), store);

        let mut headers = vec![];
        let mut url = "http://api.openai.com/v1/chat".to_string();
        executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_key".into()],
                "default",
                Some("alice"),
            )
            .await
            .unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat");
        assert_eq!(headers[0].1, "Bearer sk-test");
    }

    #[test]
    fn test_transform_config_load() {
        let json = r#"{
            "transforms": {
                "inject_openai_key": {
                    "type": "add_header",
                    "key": "Authorization",
                    "format": "Bearer {api_key}",
                    "service": "openai"
                },
                "inject_fda_key": {
                    "type": "add_url_param",
                    "key": "api_key",
                    "format": "{api_key}",
                    "service": "fda"
                }
            }
        }"#;

        let tmp = std::env::temp_dir().join("obs_test_transforms.json");
        std::fs::write(&tmp, json).unwrap();

        let cfg = TransformConfig::load(tmp.to_str().unwrap()).unwrap();
        assert_eq!(cfg.transforms.len(), 2);
        assert!(cfg.transforms.contains_key("inject_openai_key"));
        assert!(cfg.transforms.contains_key("inject_fda_key"));

        std::fs::remove_file(&tmp).ok();
    }

    #[tokio::test]
    async fn test_add_header_replaces_existing() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials("alice", "openai", vec![("api_key".into(), "sk-new".into())])
            .unwrap();

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_key".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );

        let executor = TransformExecutor::new(cfg, store);
        let mut headers = vec![("authorization".into(), "Bearer sk-old".into())];
        let mut url = "https://api.openai.com".into();
        executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_key".into()],
                "default",
                Some("alice"),
            )
            .await
            .unwrap();

        // Should replace, not append
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert_eq!(headers[0].1, "Bearer sk-new");
    }

    /// With no transforms attached there is nothing to inject, so a request
    /// without an entity is ordinary and passes.
    #[tokio::test]
    async fn no_entity_and_no_transforms_is_not_a_failure() {
        let store = Arc::new(MemorySource::new().unwrap());
        let executor = TransformExecutor::new(TransformConfig::default(), store);
        let mut headers = vec![];
        let mut url = "https://example.com".to_string();
        executor
            .execute(&mut headers, &mut url, &[], "default", None)
            .await
            .unwrap();
        assert!(headers.is_empty());
    }

    /// A transform the policy attached cannot be applied without an entity to
    /// resolve credentials for, and the authorization was conditional on it
    /// being applied — so the caller is refused rather than proxied.
    #[tokio::test]
    async fn a_transform_without_an_entity_is_refused() {
        let store = Arc::new(MemorySource::new().unwrap());
        let executor = TransformExecutor::new(TransformConfig::default(), store);
        let mut headers = vec![("authorization".to_string(), "Bearer caller".to_string())];
        let mut url = "https://example.com".to_string();
        let err = executor
            .execute(
                &mut headers,
                &mut url,
                &["some_transform".into()],
                "default",
                None,
            )
            .await
            .expect_err("a transform that cannot run must not be skipped");
        assert!(format!("{err}").contains("no entity"), "got: {err}");
    }

    /// A backend that answers but holds nothing is the ordinary shape of a
    /// misconfiguration: a secret that was never created, a secret variable
    /// left unset, a field missing from the secret. The request must be
    /// refused, and the caller's own header must still be there for the
    /// caller, never forwarded upstream in place of the credential.
    #[tokio::test]
    async fn a_credential_that_resolves_to_nothing_is_refused() {
        let empty = Arc::new(MemorySource::new().unwrap());

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_key".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );

        let executor = TransformExecutor::new(cfg, empty);
        let mut headers = vec![(
            "Authorization".to_string(),
            "Bearer caller-supplied".to_string(),
        )];
        let mut url = "https://api.openai.com/v1/chat".to_string();
        let err = executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_key".into()],
                "default",
                Some("alice"),
            )
            .await
            .expect_err("an empty credential must not read as a successful injection");
        let msg = format!("{err}");
        assert!(
            msg.contains("api_key") && msg.contains("openai"),
            "the refusal must name the missing field and its service: {msg}"
        );
        assert!(
            !msg.contains("caller-supplied"),
            "no request material belongs in the error: {msg}"
        );
    }

    /// The same for a query-parameter transform: the URL is not sent on
    /// without the key.
    #[tokio::test]
    async fn a_url_parameter_with_no_credential_is_refused() {
        let empty = Arc::new(MemorySource::new().unwrap());

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_param".into(),
            Transform::AddUrlParam {
                parameter: "api_key".into(),
                format_str: "{api_key}".into(),
                service: "fda".into(),
            },
        );

        let executor = TransformExecutor::new(cfg, empty);
        let mut headers = vec![];
        let mut url = "https://api.fda.gov/drug/event.json".to_string();
        let err = executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_param".into()],
                "default",
                Some("alice"),
            )
            .await
            .expect_err("an empty credential must not read as a successful injection");
        assert!(format!("{err}").contains("fda"), "got: {err}");
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json",
            "the URL is left as it was; nothing goes upstream"
        );
    }

    /// A credential that is there but blank is refused exactly like a missing
    /// one.
    ///
    /// A blank value is what a `Credential` message with no value set, or an
    /// OpenBao field holding `""`, resolves to. It interpolates without
    /// complaint into `Bearer ` or `?api_key=`, and the caller's own header
    /// has already been stripped, so the request would reach the upstream
    /// carrying no credential at all — which an API that treats a blank key
    /// as anonymous access will happily serve.
    #[tokio::test]
    async fn a_blank_credential_value_is_refused_like_a_missing_one() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials("alice", "openai", vec![("api_key".into(), String::new())])
            .unwrap();
        store
            .store()
            .set_credentials("alice", "fda", vec![("api_key".into(), String::new())])
            .unwrap();

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_key".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );
        cfg.transforms.insert(
            "inject_param".into(),
            Transform::AddUrlParam {
                parameter: "api_key".into(),
                format_str: "{api_key}".into(),
                service: "fda".into(),
            },
        );
        let executor = TransformExecutor::new(cfg, store);

        let mut headers = vec![(
            "Authorization".to_string(),
            "Bearer caller-supplied".to_string(),
        )];
        let mut url = "https://api.openai.com/v1/chat".to_string();
        let err = executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_key".into()],
                "default",
                Some("alice"),
            )
            .await
            .expect_err("a blank credential must not read as a successful injection");
        assert!(
            format!("{err}").contains("api_key"),
            "the refusal must name the field: {err}"
        );
        assert_eq!(
            headers,
            vec![(
                "Authorization".to_string(),
                "Bearer caller-supplied".to_string()
            )],
            "nothing was rewritten, so nothing can go upstream half-credentialed"
        );

        let mut headers = vec![];
        let mut url = "https://api.fda.gov/drug/event.json".to_string();
        executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_param".into()],
                "default",
                Some("alice"),
            )
            .await
            .expect_err("a blank credential must not read as a successful injection");
        assert_eq!(
            url, "https://api.fda.gov/drug/event.json",
            "the URL keeps no empty api_key parameter"
        );
    }

    /// A credential rotated behind the executor's back is picked up: the
    /// freshness token the backend handed out with the first read stops being
    /// current the moment the value changes, so the cached copy is not reused.
    ///
    /// This is what makes a revocation take effect. Without the check the
    /// executor would keep injecting the old value until 1024 distinct tuples
    /// accumulated or the process restarted.
    #[tokio::test]
    async fn a_rotated_credential_replaces_the_cached_one() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials("alice", "openai", vec![("api_key".into(), "sk-old".into())])
            .unwrap();

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_key".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );
        let executor = TransformExecutor::new(cfg, Arc::clone(&store) as Arc<dyn CredentialSource>);

        async fn inject(executor: &TransformExecutor) -> String {
            let mut headers = vec![];
            let mut url = "https://api.openai.com".to_string();
            executor
                .execute(
                    &mut headers,
                    &mut url,
                    &["inject_key".into()],
                    "default",
                    Some("alice"),
                )
                .await
                .unwrap();
            headers[0].1.clone()
        }

        assert_eq!(inject(&executor).await, "Bearer sk-old");
        // Cached: a second call must not read the store again, but must still
        // produce the same header.
        assert_eq!(inject(&executor).await, "Bearer sk-old");

        store
            .store()
            .set_credentials("alice", "openai", vec![("api_key".into(), "sk-new".into())])
            .unwrap();

        assert_eq!(
            inject(&executor).await,
            "Bearer sk-new",
            "a rotation must reach the next request, not wait out the cache"
        );
    }

    #[tokio::test]
    async fn test_executor_add_header() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials(
                "alice",
                "openai",
                vec![("api_key".into(), "sk-test".into())],
            )
            .unwrap();

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject_key".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );

        let executor = TransformExecutor::new(cfg, store);
        let mut headers = vec![];
        let mut url = "https://api.openai.com".into();
        executor
            .execute(
                &mut headers,
                &mut url,
                &["inject_key".into()],
                "default",
                Some("alice"),
            )
            .await
            .unwrap();

        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert_eq!(headers[0].1, "Bearer sk-test");
    }

    #[test]
    fn test_append_query_param_encodes_special_chars() {
        let mut url = "https://api.example.com/v1".into();
        append_query_param(&mut url, "q", "a&b=c d+e").unwrap();
        assert_eq!(url, "https://api.example.com/v1?q=a%26b%3Dc+d%2Be");

        let mut url2 = "https://x.com?a=1".into();
        append_query_param(&mut url2, "key with space", "100%").unwrap();
        assert!(url2.contains("key+with+space=100%25"));
    }

    /// Two tenants register credentials under the same entity name
    /// and service; each tenant must see only its own value.
    #[tokio::test]
    async fn test_tenant_isolation() {
        let store = Arc::new(MemorySource::new().unwrap());
        store
            .store()
            .set_credentials_for_tenant(
                "tenant-a",
                "alice",
                "openai",
                vec![("api_key".into(), "sk-A".into())],
            )
            .unwrap();
        store
            .store()
            .set_credentials_for_tenant(
                "tenant-b",
                "alice",
                "openai",
                vec![("api_key".into(), "sk-B".into())],
            )
            .unwrap();

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "inject".into(),
            Transform::AddHeader {
                key: "Authorization".into(),
                format_str: "Bearer {api_key}".into(),
                service: "openai".into(),
            },
        );

        let exec = TransformExecutor::new(cfg, store);

        let mut h_a = vec![];
        let mut u_a = "https://x.com".into();
        exec.execute(
            &mut h_a,
            &mut u_a,
            &["inject".into()],
            "tenant-a",
            Some("alice"),
        )
        .await
        .unwrap();
        assert_eq!(h_a[0].1, "Bearer sk-A");

        let mut h_b = vec![];
        let mut u_b = "https://x.com".into();
        exec.execute(
            &mut h_b,
            &mut u_b,
            &["inject".into()],
            "tenant-b",
            Some("alice"),
        )
        .await
        .unwrap();
        assert_eq!(h_b[0].1, "Bearer sk-B");
    }

    #[tokio::test]
    async fn test_cache_bounded() {
        let store = Arc::new(MemorySource::new().unwrap());

        // Pre-populate credentials for MAX_CACHE_SIZE + 1
        // distinct entities.
        for i in 0..=MAX_CACHE_SIZE {
            let entity = format!("entity_{i}");
            store
                .store()
                .set_credentials(&entity, "svc", vec![("k".into(), format!("v{i}"))])
                .unwrap();
        }

        let mut cfg = TransformConfig::default();
        cfg.transforms.insert(
            "t".into(),
            Transform::AddHeader {
                key: "X-Key".into(),
                format_str: "{k}".into(),
                service: "svc".into(),
            },
        );

        let exec = TransformExecutor::new(cfg, store);

        for i in 0..=MAX_CACHE_SIZE {
            let entity = format!("entity_{i}");
            let mut headers = vec![];
            let mut url = "https://x.com".into();
            exec.execute(
                &mut headers,
                &mut url,
                &["t".into()],
                "default",
                Some(&entity),
            )
            .await
            .unwrap();
            assert_eq!(headers[0].1, format!("v{i}"), "entity_{i} lookup failed");
        }

        // After eviction + re-insert the cache must be
        // within bounds.
        let cache = exec.cache.read().await;
        assert!(
            cache.len() <= MAX_CACHE_SIZE,
            "cache size {} exceeds limit {MAX_CACHE_SIZE}",
            cache.len()
        );
    }
}
