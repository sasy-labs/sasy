//! Shared SSRF / DNS-rebinding guards and log hygiene for the reference
//! monitor's outbound HTTP paths (the gRPC proxy in [`crate::service`] and
//! the forward proxy in [`crate::forward_proxy`]).
//!
//! Both paths must reject targets that resolve to internal addresses and
//! connect to the *vetted* IP rather than re-resolving the name at fetch
//! time (which would reopen the DNS-rebinding window).

use std::net::{IpAddr, SocketAddr};

/// Both outbound proxies derive Host from the vetted URL. A proxy-only
/// connection header also has no meaning on the upstream leg.
pub(crate) fn forward_request_header(name: &str) -> bool {
    !name.eq_ignore_ascii_case("host") && !name.eq_ignore_ascii_case("proxy-connection")
}

/// True if `ip` is one the proxy must never connect to: IPv4 loopback,
/// RFC1918 private, link-local (169.254), unspecified, broadcast, `0.0.0.0/8`,
/// and CGNAT `100.64.0.0/10`; plus the IPv6 equivalents (loopback,
/// unique-local `fc00::/7`, link-local `fe80::/10`, v4-mapped). Defends the
/// proxy against SSRF / DNS-rebinding to internal hosts even when the policy
/// authorized the (textual) host.
pub(crate) fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.octets()[0] == 0
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(v4));
            }
            let seg0 = v6.segments()[0];
            (seg0 & 0xfe00) == 0xfc00 || (seg0 & 0xffc0) == 0xfe80
        }
    }
}

/// Resolve `url`'s host, reject internal targets, and return the host string
/// plus a pinned [`SocketAddr`] to connect to. Pinning the validated IP closes
/// the DNS-rebinding window: the caller connects to the address we vetted
/// rather than re-resolving the name to a different (internal) IP at fetch
/// time.
pub(crate) async fn resolve_public_target(url: &str) -> Result<(String, SocketAddr), String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("bad url: {e}"))?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(format!("unsupported scheme: {scheme}"));
    }
    let host = parsed.host_str().ok_or("missing host")?.to_string();
    let port = parsed.port_or_known_default().ok_or("missing port")?;
    let resolved: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|e| format!("dns resolution failed: {e}"))?
        .collect();
    let addr = resolved
        .into_iter()
        .find(|a| !is_blocked_ip(a.ip()))
        .ok_or("host resolves only to disallowed (internal) addresses")?;
    Ok((host, addr))
}

/// Resolve a `host:port` authority (a CONNECT tunnel target), reject internal
/// targets, and return the pinned [`SocketAddr`] to connect to. Accepts both
/// `host:port` and bracketed-IPv6 `[::1]:port` forms (via `lookup_host`).
pub(crate) async fn resolve_host_port_public(host_port: &str) -> Result<SocketAddr, String> {
    let resolved: Vec<SocketAddr> = tokio::net::lookup_host(host_port)
        .await
        .map_err(|e| format!("dns resolution failed: {e}"))?
        .collect();
    resolved
        .into_iter()
        .find(|a| !is_blocked_ip(a.ip()))
        .ok_or_else(|| "host resolves only to disallowed (internal) addresses".to_string())
}

/// Remove the three parts of a URL that a credential is usually written into,
/// before the URL is logged or echoed back to a caller.
///
/// Exactly these go:
///
/// - the **query** (`?api_key=…`), which is where a credential-injecting
///   transform puts one;
/// - the **fragment** (`#…`), the tail after a `#`. It is never sent to the
///   server, but it is part of the string we were handed, and a caller that
///   put a key there — or a transform that appended into one — would have had
///   it written to the log;
/// - the **userinfo** (`https://user:pass@host/…`), the credentials some
///   clients place before the host.
///
/// The scheme, host, port and **path** are kept, which is what makes the line
/// worth logging: it says what was contacted. The path is not scrubbed, so a
/// URL that carries its token as a path segment — a Telegram bot API call,
/// `/bot<id>:<secret>/sendMessage`, is the usual example — still shows it. That
/// is a deliberate limit, not an oversight: redacting the path would leave a
/// line that no longer identifies the request.
///
/// A URL that does not parse, and one that cannot have a host (`mailto:`,
/// `data:` — the parser calls these *cannot-be-a-base*, and refuses to set
/// userinfo on them), both fall back to a substring cut: everything from the
/// first `?` or `#` onward goes, and so does any `user:pass@` in the authority.
pub(crate) fn redact_url(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_query(None);
            parsed.set_fragment(None);
            // Both refuse on a URL that cannot have a host. Refusing means
            // nothing was removed, so cut the string instead rather than
            // return a URL whose userinfo was never touched.
            if parsed.set_username("").is_err() || parsed.set_password(None).is_err() {
                return redact_by_cut(url);
            }
            parsed.to_string()
        }
        Err(_) => redact_by_cut(url),
    }
}

/// The fallback for [`redact_url`]: cut at the first `?` or `#`, then drop any
/// `user:pass@` from the authority — no parser involved.
fn redact_by_cut(url: &str) -> String {
    let cut = match url.find(['?', '#']) {
        Some(i) => &url[..i],
        None => url,
    };
    // The authority — `host[:port]`, optionally preceded by `user:pass@` —
    // starts after `scheme://`, or after a leading `//`. With neither marker
    // there is no authority, so an `@` in what is left is part of a path, not
    // userinfo.
    let authority_start = match cut.find("://") {
        Some(i) => Some(i + 3),
        None => cut.starts_with("//").then_some(2),
    };
    match authority_start {
        Some(start) => {
            let rest = &cut[start..];
            let authority_end = rest.find('/').unwrap_or(rest.len());
            match rest[..authority_end].rfind('@') {
                Some(at) => format!("{}{}", &cut[..start], &rest[at + 1..]),
                None => cut.to_string(),
            }
        }
        None => cut.to_string(),
    }
}

/// A reqwest client builder hardened for the proxy's outbound fetch leg,
/// shared by the gRPC proxy ([`crate::service`]) and the forward proxy
/// ([`crate::forward_proxy`]) so the security-relevant settings can't drift:
///
/// - **DNS pinned** to the SSRF-vetted `(pin_host, pin_addr)` — connect to the
///   IP already validated by [`resolve_public_target`] rather than re-resolving
///   at fetch time (which would reopen the DNS-rebinding window).
/// - **Redirects disabled** — a 3xx is surfaced to the caller, not followed.
///   Following one would re-target a host that never passed the SSRF guard /
///   policy and replay transform-injected credentials to it.
/// - **Auto-decompression off** (`no_gzip`/`no_brotli`/`no_deflate`) — a
///   decompression-bomb guard.
/// - **connect/read timeouts** bound a stalled or slow-loris upstream without
///   imposing a total deadline that would truncate long-lived SSE streams.
///
/// Callers `.build()` and apply their own fallback for the (config-static,
/// effectively-infallible) build step.
pub(crate) fn pinned_client_builder(
    pin_host: &str,
    pin_addr: SocketAddr,
) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(60))
        .resolve(pin_host, pin_addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A log line says what was contacted, never with what.
    ///
    /// Stripping only the query left two other places a credential lives: the
    /// fragment (`#…`) and the userinfo (`user:pass@`). Post-transform URLs
    /// go through here on the streaming-response line and at both
    /// SSRF-rejection sites, so anything left in reaches the log sink.
    #[test]
    fn a_logged_url_keeps_the_target_and_drops_every_secret() {
        assert_eq!(
            redact_url("https://user:pass@api.fda.gov/drug/event.json?api_key=K#tok=T"),
            "https://api.fda.gov/drug/event.json",
            "query, fragment and userinfo all go; scheme, host and path stay"
        );
        assert_eq!(
            redact_url("https://api.fda.gov:8443/drug/event.json#api_key=K"),
            "https://api.fda.gov:8443/drug/event.json",
            "a non-default port is part of what was contacted, so it stays"
        );
        assert_eq!(
            redact_url("https://api.fda.gov/drug/event.json"),
            "https://api.fda.gov/drug/event.json",
            "a URL with nothing to redact is unchanged"
        );
    }

    /// Redact userinfo even when Url::set_username rejects an opaque URL such
    /// as mailto: or data:; use the conservative string fallback.
    #[test]
    fn a_url_that_cannot_hold_userinfo_still_gets_cut() {
        assert_eq!(
            redact_url("mailto:user:pass@example.com?subject=K"),
            "mailto:user:pass@example.com",
            "the query goes; there is no authority here, so the `@` is not userinfo"
        );
    }

    /// The path is kept, secret or not — that is the documented limit.
    ///
    /// A token carried as a path segment (`/bot<id>:<secret>/…`) survives
    /// redaction. Cutting the path would leave a line that no longer says what
    /// was contacted, which is the only reason the line exists.
    #[test]
    fn the_path_is_kept_even_when_it_carries_a_token() {
        assert_eq!(
            redact_url("https://api.telegram.org/bot123:SECRET/sendMessage?chat_id=1"),
            "https://api.telegram.org/bot123:SECRET/sendMessage"
        );
    }

    /// The same holds when the URL does not parse and the fallback cut runs.
    #[test]
    fn the_fallback_cut_also_drops_the_fragment_and_the_userinfo() {
        // No scheme, so `Url::parse` refuses it as a relative URL and the
        // fallback runs.
        assert!(reqwest::Url::parse("//user:pass@api.fda.gov/p?api_key=K").is_err());
        assert_eq!(
            redact_url("//user:pass@api.fda.gov/p?api_key=K"),
            "//api.fda.gov/p",
            "a scheme-relative URL still loses its userinfo and query"
        );
        assert_eq!(
            redact_url("ht tp://user:pass@api.fda.gov/p#api_key=K"),
            "ht tp://api.fda.gov/p",
            "the fragment is cut and the userinfo removed without a parse"
        );
        assert_eq!(
            redact_url("ht tp://api.fda.gov/p?api_key=K"),
            "ht tp://api.fda.gov/p",
            "the query is cut at the first `?`"
        );
    }
}
