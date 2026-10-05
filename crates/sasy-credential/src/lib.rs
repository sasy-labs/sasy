//! Credential storage and lookup, behind one backend abstraction.
//!
//! Everything that injects a credential reads it through
//! [`CredentialSource`]. Three backends implement it:
//!
//! - [`SqliteSource`] — the local SQLite file ([`CredentialStore`]), the
//!   default;
//! - [`MemorySource`] — process memory, seeded from environment variables,
//!   for deployments whose secrets already arrive that way (a variable has no
//!   tenant, so what it seeds is injected for every tenant);
//! - `OpenBaoSource` — read-only lookups against an OpenBao KV v2 secret
//!   engine (behind the `openbao` cargo feature).
//!
//! [`CredentialService`] exposes the selected source over gRPC.

pub mod error;
pub mod memory;
#[cfg(feature = "openbao")]
pub mod openbao;
pub mod service;
pub mod source;
pub mod store;

pub use error::CredentialError;
pub use memory::MemorySource;
#[cfg(feature = "openbao")]
pub use openbao::{OpenBaoAuth, OpenBaoConfig, OpenBaoSource};
pub use service::CredentialService;
pub use source::{CredentialSource, CredentialView, Freshness, ResolvedCredentials, SqliteSource};
pub use store::{CredentialGeneration, CredentialStore, WILDCARD_ENTITY};
