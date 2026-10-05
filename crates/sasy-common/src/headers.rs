//! Canonical gRPC metadata header keys used at the auth / delegation
//! boundary.
//!
//! Each key is set by a producer (an SDK, a gateway, or a
//! relaying reference monitor) and read by a consumer (the auth
//! interceptor or a downstream service). Centralizing the literals
//! keeps the insert side and the read side from drifting apart. These
//! are **wire** keys — the values must stay byte-stable across services
//! and the Python/TS SDKs.

/// API-key credential (api-key auth provider).
pub const API_KEY: &str = "x-api-key";

/// Bearer / JWT credential. Standard HTTP header name (lowercased for
/// gRPC metadata, which requires lowercase keys).
pub const AUTHORIZATION: &str = "authorization";

/// mTLS client Common Name, forwarded by a TLS-terminating proxy that
/// the server is configured to trust.
pub const CLIENT_CN: &str = "x-client-cn";

/// User-supplied actor in the caller's domain (free-form; surfaced to
/// policies as `Entity`, never an auth identity).
pub const ENTITY: &str = "x-entity";

/// Delegated roles asserted by a [`crate::roles::SERVICE_PROXY`] caller.
pub const ROLES: &str = "x-roles";

/// Delegated end-user principal asserted by a
/// [`crate::roles::SERVICE_PROXY`] caller.
pub const PRINCIPAL: &str = "x-principal";

/// Delegated tenant asserted by a [`crate::roles::SERVICE_PROXY`] caller.
pub const TENANT: &str = "x-tenant";
