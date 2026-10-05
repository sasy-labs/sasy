//! The credential backend abstraction.
//!
//! Everything that injects a credential goes through [`CredentialSource`].
//! The reference monitor never learns which backend it is talking to, so a
//! deployment can keep credentials in a local SQLite file, in process memory
//! seeded from the environment, or in an external secret manager without any
//! change to the injection path.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::error::CredentialError;
use crate::store::{CredentialGeneration, CredentialStore};

/// Which of the two views of a `(tenant, entity, service)` lookup is wanted.
///
/// The distinction is a security rule, not a convenience: a caller reading on
/// somebody's BEHALF needs the tenant's shared defaults merged in, because
/// that merge is what gets injected. A caller reading its OWN credentials must
/// not get them, or the role gate that protects the tenant-wide secrets
/// guards nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialView {
    /// Tenant-wide defaults with the entity's own entries laid over the top.
    /// This is the injection view, used by a `service-proxy` relay.
    Relay,
    /// Only the entries that belong to the entity itself.
    Own,
}

/// How long a resolved set of credentials may be reused before it has to be
/// read again.
///
/// Local and remote backends answer the question differently. A store this
/// process can watch answers it exactly —
/// the credentials are current as long as nothing has been written. A remote
/// secret manager cannot be watched, so its answer is a deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Current while the source's generation still equals this one. A write —
    /// by this process or another one — moves the generation, so a rotation or
    /// a revocation takes effect on the next lookup.
    Generation(CredentialGeneration),
    /// Current while both a writable store and its separate fallback store
    /// retain their generations. Used by the memory backend's relay view.
    LayeredGeneration {
        primary: CredentialGeneration,
        fallback: CredentialGeneration,
    },
    /// Current until this instant, and then read again.
    Until(Instant),
}

/// Credentials as a backend resolved them, with the token that says how long
/// they may be cached.
#[derive(Debug, Clone)]
pub struct ResolvedCredentials {
    /// Credential key → value, already merged per the requested
    /// [`CredentialView`].
    pub values: HashMap<String, String>,
    /// Hand this back to [`CredentialSource::is_current`] before reusing
    /// `values`.
    pub freshness: Freshness,
}

/// A place credentials are read from (and, for the local backends, written
/// to).
///
/// Implementations are shared behind an `Arc` across every request, so they
/// must be cheap to call concurrently.
#[tonic::async_trait]
pub trait CredentialSource: Send + Sync {
    /// Short name of this backend, for logs and operator-facing errors:
    /// `sqlite`, `memory`, `openbao`.
    fn backend(&self) -> &'static str;

    /// Where this backend keeps its credentials, for operator-facing errors —
    /// a file path or a server address. Never a secret.
    fn location(&self) -> String;

    /// Resolve the credentials for `(tenant, entity, service)` in the
    /// requested view.
    ///
    /// An entity with no credentials is an empty map, not an error.
    async fn resolve(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError>;

    /// Whether credentials stamped with `freshness` are still current.
    ///
    /// Callers that cache have to ask this before every reuse: it is the only
    /// thing standing between a revoked credential and its continued
    /// injection.
    fn is_current(&self, freshness: &Freshness) -> bool;

    /// Store credentials for `(tenant, entity, service)`.
    ///
    /// A read-only backend returns [`CredentialError::ReadOnly`]; the gRPC
    /// layer turns that into `FAILED_PRECONDITION` naming the backend, so an
    /// operator is told where the write actually belongs.
    async fn set_credentials(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError>;
}

/// Read a `(tenant, entity, service)` tuple out of a [`CredentialStore`].
///
/// The generation is read BEFORE the rows, not after. A write that lands in
/// between then makes the result immediately stale and the next lookup reads
/// again — the safe direction. Stamping with a generation read afterwards
/// would let a cache hold, as current, rows that a write had already
/// replaced.
pub(crate) fn resolve_from_store(
    store: &CredentialStore,
    tenant: &str,
    entity: &str,
    service: &str,
    view: CredentialView,
) -> Result<ResolvedCredentials, CredentialError> {
    let generation = store.generation();
    let values = match view {
        CredentialView::Relay => store.get_credentials_for_tenant(tenant, entity, service)?,
        CredentialView::Own => store.get_own_credentials_for_tenant(tenant, entity, service)?,
    };
    Ok(ResolvedCredentials {
        values,
        freshness: Freshness::Generation(generation),
    })
}

/// The default backend: the SQLite credential file.
///
/// A thin wrapper — the behaviour is entirely [`CredentialStore`]'s, so a
/// deployment that does not choose a backend keeps exactly what it had.
pub struct SqliteSource {
    store: Arc<CredentialStore>,
    /// The database path, kept for operator-facing messages.
    path: String,
}

impl SqliteSource {
    /// Open (creating if needed) the credential database at `db_path`.
    pub fn open(db_path: &str) -> Result<Self, CredentialError> {
        Ok(Self {
            store: Arc::new(CredentialStore::new(db_path)?),
            path: db_path.to_string(),
        })
    }

    /// The underlying store, for callers that need the concrete type
    /// (`init-credentials`, tests).
    pub fn store(&self) -> &Arc<CredentialStore> {
        &self.store
    }
}

#[tonic::async_trait]
impl CredentialSource for SqliteSource {
    fn backend(&self) -> &'static str {
        "sqlite"
    }

    fn location(&self) -> String {
        self.path.clone()
    }

    async fn resolve(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError> {
        resolve_from_store(&self.store, tenant, entity, service, view)
    }

    fn is_current(&self, freshness: &Freshness) -> bool {
        match freshness {
            Freshness::Generation(g) => self.store.generation() == *g,
            // A deadline says nothing about a store whose contents can change
            // at any moment; treat it as spent.
            Freshness::Until(_) | Freshness::LayeredGeneration { .. } => false,
        }
    }

    async fn set_credentials(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        self.store
            .set_credentials_for_tenant(tenant, entity, service, credentials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_source() -> (tempfile::TempDir, SqliteSource) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("creds.db").to_string_lossy().into_owned();
        let source = SqliteSource::open(&path).unwrap();
        (dir, source)
    }

    /// The source is the store's behaviour, unchanged: the tenant defaults
    /// are overlaid for the relay view and absent from the own view.
    #[tokio::test]
    async fn the_sqlite_source_keeps_the_two_views_the_store_defines() {
        let (_dir, source) = temp_source();
        source
            .set_credentials(
                "acme",
                crate::store::WILDCARD_ENTITY,
                "openai",
                vec![("platform_key".into(), "sk-shared".into())],
            )
            .await
            .unwrap();
        source
            .set_credentials(
                "acme",
                "agent",
                "openai",
                vec![("own_key".into(), "sk-mine".into())],
            )
            .await
            .unwrap();

        let relay = source
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            relay.values.get("platform_key").map(String::as_str),
            Some("sk-shared")
        );
        assert_eq!(
            relay.values.get("own_key").map(String::as_str),
            Some("sk-mine")
        );

        let own = source
            .resolve("acme", "agent", "openai", CredentialView::Own)
            .await
            .unwrap();
        assert!(
            !own.values.contains_key("platform_key"),
            "an entity reading itself must not receive the tenant's shared secret"
        );
    }

    /// Tenants stay disjoint through the source, as they do through the
    /// store.
    #[tokio::test]
    async fn the_sqlite_source_keeps_tenants_apart() {
        let (_dir, source) = temp_source();
        for (tenant, value) in [("tenant-a", "key-a"), ("tenant-b", "key-b")] {
            source
                .set_credentials(
                    tenant,
                    crate::store::WILDCARD_ENTITY,
                    "openai",
                    vec![("api_key".into(), value.into())],
                )
                .await
                .unwrap();
        }
        for (tenant, value) in [("tenant-a", "key-a"), ("tenant-b", "key-b")] {
            let got = source
                .resolve(tenant, "user", "openai", CredentialView::Relay)
                .await
                .unwrap();
            assert_eq!(got.values.get("api_key").map(String::as_str), Some(value));
        }
    }

    /// A write retires what was read before it, including a write made by
    /// another process against the same file.
    ///
    /// This is the generation counter the store's tests pin, reached through
    /// the source: without it a cache would keep injecting a credential that
    /// has been rotated or revoked.
    #[tokio::test]
    async fn a_write_retires_an_earlier_read_even_from_another_process() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("creds.db").to_string_lossy().into_owned();
        let serving = SqliteSource::open(&path).unwrap();

        let read = serving
            .resolve("acme", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(serving.is_current(&read.freshness));

        // A second handle on the same file, standing in for
        // `sasy init-credentials`.
        let loader = SqliteSource::open(&path).unwrap();
        loader
            .set_credentials(
                "acme",
                crate::store::WILDCARD_ENTITY,
                "openai",
                vec![("api_key".into(), "rotated".into())],
            )
            .await
            .unwrap();

        assert!(
            !serving.is_current(&read.freshness),
            "a write by another process must retire what the serving process read"
        );
    }

    /// A deadline from some other backend never counts as current here: the
    /// file can change at any moment, so only a generation can vouch for it.
    #[tokio::test]
    async fn the_sqlite_source_does_not_honour_a_deadline() {
        let (_dir, source) = temp_source();
        let far_future = Freshness::Until(Instant::now() + std::time::Duration::from_secs(3600));
        assert!(!source.is_current(&far_future));
    }
}
