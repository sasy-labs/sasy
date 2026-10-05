//! Canonical names of the privileged roles the TCB checks at
//! authorization decision points.
//!
//! These are **config-data** values: an auth provider maps an entity to
//! role strings (see `config/auth_config.yaml`, `config/auth/*.json`)
//! and the server compares them at its decision points. Centralizing
//! the spellings keeps a typo from silently changing a trust decision —
//! a misspelled [`SERVICE_PROXY`] check fails *open* (delegation
//! silently disabled), and drift between the producer config and a
//! consumer literal re-enables raw wire trust. The string VALUES must
//! stay byte-stable to keep matching existing auth configs.

/// Full administrative access; bypasses session-ownership and other
/// gates. Mapped to the admin entities in `config/auth/*.json`.
pub const ADMIN: &str = "admin";

/// Trusted relay (a gateway, or a co-located reference monitor)
/// permitted to assert delegated `x-tenant` / `x-principal` / `x-roles`
/// on behalf of end users. This is the delegation trust gate — a typo
/// here fails *open* (the relay's delegation is silently ignored, or a
/// drifted literal lets an unauthenticated caller assert identities).
pub const SERVICE_PROXY: &str = "service-proxy";

/// Required to call the reference-monitor proxy RPCs
/// (`ProxyHTTP` / `CheckToolCall`).
pub const REFERENCE_MONITOR_USER: &str = "reference-monitor-user";

/// Read plaintext credentials via the credential server.
pub const CREDENTIAL_READER: &str = "credential-reader";

/// Write credentials via the credential server.
pub const CREDENTIAL_WRITER: &str = "credential-writer";

/// Read the observability graph (slices, traces, spans, state).
pub const OBSERVABILITY_READER: &str = "observability-reader";

/// Write to the observability graph (events, dependencies, computations).
pub const OBSERVABILITY_WRITER: &str = "observability-writer";
