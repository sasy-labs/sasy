//! Protobuf stubs and shared domain types for the SASY engine.
//!
//! This crate re-exports the generated gRPC service definitions and message
//! types of every proto file, plus shared domain enums used across crates.

/// Generated types for the Observability service
/// (from `observability.proto`, package `observability`).
pub mod observability {
    tonic::include_proto!("observability");
}

/// Generated types for the PolicyEngine service
/// (from `policy_engine.proto`, package `policy_engine`).
pub mod policy_engine {
    tonic::include_proto!("policy_engine");
}

/// Generated types for the policy plugin C ABI transport
/// (from `policy_plugin.proto`, package `policy_plugin`).
pub mod policy_plugin {
    tonic::include_proto!("policy_plugin");
}

/// Generated types for the CredentialServer and RMProxy services.
///
/// Both `credential_server.proto` and `reference_monitor.proto` have no
/// `package` declaration, so prost places them in the default (empty)
/// package. They share a single generated file (`_.rs`).
pub mod services {
    // reference_monitor.proto imports policy_engine.proto, so the
    // generated code references `policy_engine::DenialTrace` etc.
    use super::policy_engine;
    tonic::include_proto!("_");
}

// Re-export the credential server and reference monitor types at
// convenient module paths for downstream crates.

/// Credential server types (re-exported from `services`).
pub mod credential_server {
    pub use super::services::{
        credential_server_client, credential_server_server, Credential, CredentialResponse,
        Credentials, CredentialsRequest, SetCredentialsRequest,
    };
}

/// Reference monitor types (re-exported from `services`).
pub mod reference_monitor {
    pub use super::services::{
        rm_proxy_client, rm_proxy_server, BaseRequest, HttpHeader, HttpRequest, HttpResponse,
        Message, ToolCallRequest, ToolCallResponse,
    };
}

// ── Shared cross-crate constants ─────────────────────────────────

pub mod headers;
pub mod roles;

/// Evaluator backend selector. The string forms are wire + persisted
/// values: `SetPolicyRequest.backend`, `PolicyEngineStatus.backend`, the
/// persisted `PersistedPolicy.backend`, and the `--evaluator` CLI flag.
/// [`Self::as_str`] / [`Self::from_wire`] therefore emit/accept exactly
/// these spellings and must stay byte-stable.
///
/// Dispatching on this enum (rather than the raw string) makes the
/// backend set exhaustive: adding a variant is a compile error at every
/// dispatch site until handled, instead of silently falling through a
/// catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Compiled Soufflé (per-policy g++ ELF). The default.
    Souffle,
    /// Interpreted Soufflé (shared `libfunctors.so`, no per-policy compile).
    SouffleInterpreted,
    /// Experimental external FlowLog evaluator (static policy assets).
    Flowlog,
    /// No-op stub evaluator (tests / dev).
    Stub,
}

impl Backend {
    /// The canonical wire/persisted string for this backend.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Souffle => "souffle",
            Self::SouffleInterpreted => "souffle-interpreted",
            Self::Flowlog => "flowlog",
            Self::Stub => "stub",
        }
    }

    /// Parse a wire/persisted/CLI backend string; `None` for an
    /// unrecognized value (the caller decides how to reject it).
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "souffle" => Some(Self::Souffle),
            "souffle-interpreted" => Some(Self::SouffleInterpreted),
            "flowlog" => Some(Self::Flowlog),
            "stub" => Some(Self::Stub),
            _ => None,
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Read an opt-in boolean environment flag with one canonical,
/// case-insensitive truthy set: `1`, `true`, `yes`, `on`. Anything else
/// — including unset or an unrecognized value — is `false`.
///
/// Use this for default-off flags so the accepted spellings don't drift
/// between call sites, so the same value cannot enable a flag at one site
/// and not another. Default-*on* opt-out toggles (e.g. a sandbox enabled unless
/// explicitly `0`) intentionally do not use this.
pub fn env_flag(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

// ── Shared domain types ──────────────────────────────────────────

/// Role of a message in the conversation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MessageRole {
    System,
    User,
    Llm,
    Agent,
}

impl MessageRole {
    pub fn from_proto(val: i32) -> Self {
        match observability::Role::try_from(val) {
            Ok(observability::Role::System) => Self::System,
            Ok(observability::Role::User) => Self::User,
            Ok(observability::Role::Llm) => Self::Llm,
            Ok(observability::Role::Agent) => Self::Agent,
            _ => Self::User,
        }
    }

    pub fn to_proto(self) -> i32 {
        match self {
            Self::System => observability::Role::System as i32,
            Self::User => observability::Role::User as i32,
            Self::Llm => observability::Role::Llm as i32,
            Self::Agent => observability::Role::Agent as i32,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Llm => "llm",
            Self::Agent => "agent",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Self::System),
            "user" => Some(Self::User),
            "llm" => Some(Self::Llm),
            "agent" => Some(Self::Agent),
            _ => None,
        }
    }

    /// Map a proto `Role` tag to a [`MessageRole`], returning `None`
    /// for an out-of-range value. Unlike [`Self::from_proto`] (which
    /// defaults unknown tags to `User`), this preserves "no/invalid
    /// role" so callers converting `Option<i32>` role tags to a
    /// canonical string don't synthesize a spurious `user`.
    pub fn from_proto_opt(val: i32) -> Option<Self> {
        observability::Role::try_from(val).ok().map(|r| match r {
            observability::Role::System => Self::System,
            observability::Role::User => Self::User,
            observability::Role::Llm => Self::Llm,
            observability::Role::Agent => Self::Agent,
        })
    }
}

/// Types of edges in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    DependsOn,
    ChildOf,
    Produces,
    Consumes,
}

/// (`tenant`, `session`) partition key for the graph store and the
/// per-session evaluator map.
///
/// Constructed at the gRPC service boundary from the
/// authentication-derived tenant and the user-supplied
/// `session_id`. Carried by-value through every storage layer call
/// so the partition is enforced at the type level — an empty
/// tenant is rejected at construction, and you cannot accidentally
/// pass a raw user-supplied session string where a scope is
/// required.
///
/// An empty `session` denotes the per-tenant **global partition**.
/// Storage writes always target a concrete `(tenant, session)`;
/// evaluator subscriptions for a global scope match every update
/// whose tenant equals theirs (regardless of session) — see
/// [`Self::matches`].
///
/// Encoding-free: the type is a pair of `String`s, so no separator
/// or escaping rules apply. A tenant `"a"` with session `"cme-x"`
/// and a tenant `"acme"` with session `"-x"` are distinct keys at
/// every level (HashMap, broadcast filter, log message), regardless
/// of how the strings happen to look concatenated.
#[derive(Debug, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SessionScope {
    tenant: String,
    session: String,
}

impl SessionScope {
    /// Construct a scope. Panics if `tenant` is empty (server bug —
    /// auth always supplies a non-empty tenant; an empty value would
    /// silently merge into the deployment-default partition and
    /// invalidate isolation invariants).
    pub fn new(tenant: impl Into<String>, session: impl Into<String>) -> Self {
        let tenant = tenant.into();
        assert!(
            !tenant.is_empty(),
            "SessionScope tenant must be non-empty (server-derived from auth)",
        );
        Self {
            tenant,
            session: session.into(),
        }
    }

    /// The per-tenant global scope. Storage writes against this
    /// scope land in the `(tenant, "")` shard; subscriptions with
    /// this scope match every update under `tenant`.
    pub fn global(tenant: impl Into<String>) -> Self {
        Self::new(tenant, "")
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    pub fn session(&self) -> &str {
        &self.session
    }

    /// True iff this scope is the per-tenant global partition.
    pub fn is_global(&self) -> bool {
        self.session.is_empty()
    }

    /// Broadcast filter: does an update emitted under `producer`
    /// belong to a subscriber with `*self` as its scope?
    ///
    /// Cross-tenant updates are always rejected. Within a tenant,
    /// a global subscriber (`session.is_empty()`) matches every
    /// session; a session subscriber matches only its exact session.
    pub fn matches(&self, producer: &SessionScope) -> bool {
        self.tenant == producer.tenant && (self.is_global() || self.session == producer.session)
    }

    /// Encode the scope as a byte string for use in any
    /// serialization layer that demands a single key (RocksDB
    /// column-family keys, on-wire forwarded events when those
    /// don't carry both fields, etc.). Length-prefix *both*
    /// components:
    ///
    /// ```text
    /// [tenant_len: u32 BE][tenant bytes][session_len: u32 BE][session bytes]
    /// ```
    ///
    /// Callers that append a third component (e.g. a node id) for
    /// composite keys can do so directly — the encoded scope ends
    /// at a known offset, so a suffix is unambiguously separable.
    /// Prefixing only the tenant would let `(t, "S") ++ "X"` collide
    /// with `(t, "SX") ++ ""`; prefixing both ends that.
    pub fn to_storage_bytes(&self) -> Vec<u8> {
        let tenant = self.tenant.as_bytes();
        let session = self.session.as_bytes();
        let mut out = Vec::with_capacity(4 + tenant.len() + 4 + session.len());
        out.extend_from_slice(&(tenant.len() as u32).to_be_bytes());
        out.extend_from_slice(tenant);
        out.extend_from_slice(&(session.len() as u32).to_be_bytes());
        out.extend_from_slice(session);
        out
    }

    /// Inverse of [`Self::to_storage_bytes`]. Returns `None` if
    /// `bytes` is shorter than either declared component length,
    /// has unconsumed trailing bytes, or if either component is
    /// invalid UTF-8.
    pub fn from_storage_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let tlen = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let after_tlen = &bytes[4..];
        if after_tlen.len() < tlen + 4 {
            return None;
        }
        let tenant = std::str::from_utf8(&after_tlen[..tlen]).ok()?.to_string();
        let rest = &after_tlen[tlen..];
        let slen = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        let body = &rest[4..];
        if body.len() != slen {
            return None;
        }
        let session = std::str::from_utf8(body).ok()?.to_string();
        if tenant.is_empty() {
            return None;
        }
        Some(Self { tenant, session })
    }
}

impl std::fmt::Display for SessionScope {
    /// Human-readable rendering for logs / errors. Not parseable —
    /// callers that need to round-trip must store the components
    /// directly. The form is intentionally not a valid encoded
    /// session id so it can't be confused with one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.session.is_empty() {
            write!(f, "{}/<global>", self.tenant)
        } else {
            write!(f, "{}/{}", self.tenant, self.session)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, SessionScope};

    #[test]
    fn backend_wire_strings_are_stable() {
        // These spellings are the wire + persisted + CLI contract; the
        // round-trip and the literals must hold so a variant rename
        // can't silently shift them.
        for be in [
            Backend::Souffle,
            Backend::SouffleInterpreted,
            Backend::Flowlog,
            Backend::Stub,
        ] {
            assert_eq!(
                Backend::from_wire(be.as_str()),
                Some(be),
                "round-trip {be:?}"
            );
        }
        assert_eq!(Backend::Souffle.as_str(), "souffle");
        assert_eq!(Backend::SouffleInterpreted.as_str(), "souffle-interpreted");
        assert_eq!(Backend::Flowlog.as_str(), "flowlog");
        assert_eq!(Backend::Stub.as_str(), "stub");
        assert_eq!(Backend::from_wire("bogus"), None);
    }

    #[test]
    #[should_panic]
    fn empty_tenant_panics() {
        SessionScope::new("", "s1");
    }

    #[test]
    fn global_is_empty_session() {
        let s = SessionScope::global("acme");
        assert_eq!(s.tenant(), "acme");
        assert_eq!(s.session(), "");
        assert!(s.is_global());
    }

    /// The collision the encoded approach has trouble with:
    /// tenant `"a"` + session `"cme-x"` versus tenant `"acme"` +
    /// session `"-x"`. With a tuple key, these are unambiguously
    /// distinct partitions regardless of how the strings might
    /// concatenate.
    #[test]
    fn collision_resistant_keys() {
        use std::collections::HashSet;
        let a = SessionScope::new("a", "cme-x");
        let b = SessionScope::new("acme", "-x");
        let mut set = HashSet::new();
        set.insert(a.clone());
        set.insert(b.clone());
        assert_eq!(set.len(), 2);
        assert_ne!(a, b);
    }

    /// Byte-level disjointness for composite keys of the shape
    /// `to_storage_bytes() ++ id`. Length-prefixing only the
    /// tenant would let `(t, "S") ++ "X"` and `(t, "SX") ++ ""`
    /// collide; length-prefixing both components rules that out.
    #[test]
    fn storage_bytes_disjoint_with_appended_id() {
        let a = {
            let mut k = SessionScope::new("t", "S").to_storage_bytes();
            k.extend_from_slice(b"X");
            k
        };
        let b = {
            let mut k = SessionScope::new("t", "SX").to_storage_bytes();
            k.extend_from_slice(b"");
            k
        };
        assert_ne!(a, b);

        // Same idea across tenants — a long tenant + short session
        // must not collide with a short tenant + long session.
        let c = {
            let mut k = SessionScope::new("acme", "-x").to_storage_bytes();
            k.extend_from_slice(b"id");
            k
        };
        let d = {
            let mut k = SessionScope::new("a", "cme-x").to_storage_bytes();
            k.extend_from_slice(b"id");
            k
        };
        assert_ne!(c, d);
    }

    #[test]
    fn storage_bytes_round_trip() {
        for s in [
            SessionScope::new("acme", "s1"),
            SessionScope::global("orgb"),
            SessionScope::new("t", ""),
            SessionScope::new("a\x00b", "c\x00d"),
        ] {
            let encoded = s.to_storage_bytes();
            assert_eq!(SessionScope::from_storage_bytes(&encoded), Some(s));
        }
    }

    #[test]
    fn matches_same_scope() {
        let s1 = SessionScope::new("acme", "s1");
        assert!(s1.matches(&SessionScope::new("acme", "s1")));
        assert!(!s1.matches(&SessionScope::new("acme", "s2")));
        assert!(!s1.matches(&SessionScope::new("orgb", "s1")));
    }

    #[test]
    fn global_matches_all_sessions_in_tenant() {
        let g = SessionScope::global("acme");
        assert!(g.matches(&SessionScope::new("acme", "s1")));
        assert!(g.matches(&SessionScope::new("acme", "s2")));
        assert!(g.matches(&SessionScope::global("acme")));
        // Cross-tenant: rejected even from global.
        assert!(!g.matches(&SessionScope::new("orgb", "s1")));
        assert!(!g.matches(&SessionScope::global("orgb")));
    }
}
