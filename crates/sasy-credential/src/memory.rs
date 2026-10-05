//! In-process credential backend, seeded from the environment.
//!
//! For deployments whose secrets already arrive as environment variables, and
//! where copying them into a file on a persistent volume adds a durable
//! plaintext copy without adding any protection. Nothing here touches disk: the store
//! is SQLite's `:memory:` database, which lives and dies with the process.
//!
//! An environment variable belongs to the deployment, not to a tenant, so a
//! seeded credential is injected for every tenant that proxies through the
//! process — under whatever each tenant holds of its own. It is never handed
//! back to a caller reading its own credentials.

use std::sync::Arc;

use tracing::{debug, info};

use crate::error::CredentialError;
use crate::source::{
    resolve_from_store, CredentialSource, CredentialView, Freshness, ResolvedCredentials,
};
use crate::store::{CredentialStore, WILDCARD_ENTITY};

/// Well-known provider environment variables, seeded in addition to the
/// generic `SASY_CREDENTIAL_*` convention below.
const WELL_KNOWN_ENV_VARS: &[(&str, &str, &str)] = &[
    ("OPENAI_API_KEY", "openai", "api_key"),
    ("OPENFDA_API_KEY", "fda", "api_key"),
    ("BRAVE_API_KEY", "brave", "api_key"),
];

/// Prefix of the general convention, `SASY_CREDENTIAL_<SERVICE>_<KEY>`.
const GENERIC_PREFIX: &str = "SASY_CREDENTIAL_";

/// The tuple used by the environment seeding helper. MemorySource keeps
/// these rows in a separate store inaccessible to tenant credential writes;
/// the SQLite import command retains its ordinary `default` tenant mapping.
pub const SEEDED_TENANT: &str = "default";

/// Which `(service, key)` an environment variable names, if any.
///
/// The generic form splits at the FIRST underscore after the prefix, so
/// `SASY_CREDENTIAL_OPENAI_API_KEY` is the `api_key` of the service `openai`.
/// A service name therefore cannot itself contain an underscore — the
/// alternative is a rule nobody can read off the variable name.
pub fn credential_for_env_var(name: &str) -> Option<(String, String)> {
    for (var, service, key) in WELL_KNOWN_ENV_VARS {
        if name == *var {
            return Some(((*service).to_string(), (*key).to_string()));
        }
    }
    let rest = name.strip_prefix(GENERIC_PREFIX)?;
    let (service, key) = rest.split_once('_')?;
    if service.is_empty() || key.is_empty() {
        return None;
    }
    Some((service.to_lowercase(), key.to_lowercase()))
}

/// Apply the environment-variable convention to `vars`, writing what it
/// recognises into `store` under the tenant-wide entity `*`.
///
/// Returns how many credentials were written. Values are never logged: the
/// whole point of this backend is that the secret exists only in memory, and
/// a log line is a copy that outlives the process.
pub fn seed_store_from_env_vars<I>(
    store: &CredentialStore,
    vars: I,
) -> Result<usize, CredentialError>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut count = 0usize;
    for (name, value) in vars {
        let Some((service, key)) = credential_for_env_var(&name) else {
            continue;
        };
        if value.is_empty() {
            debug!(var = %name, "credential variable is set but empty, skipping");
            continue;
        }
        store.set_credentials_for_tenant(
            SEEDED_TENANT,
            WILDCARD_ENTITY,
            &service,
            vec![(key.clone(), value)],
        )?;
        debug!(var = %name, service = %service, key = %key, "seeded credential");
        count += 1;
    }
    Ok(count)
}

/// Credentials held in process memory only.
pub struct MemorySource {
    store: Arc<CredentialStore>,
    seeded: CredentialStore,
}

impl MemorySource {
    /// An empty in-memory store.
    pub fn new() -> Result<Self, CredentialError> {
        Ok(Self {
            store: Arc::new(CredentialStore::new(":memory:")?),
            seeded: CredentialStore::new(":memory:")?,
        })
    }

    /// Seed from `vars` (typically the process environment, and optionally a
    /// `.env`-style file the operator pointed at). Logs the count, never the
    /// values.
    pub fn seed<I>(&self, vars: I) -> Result<usize, CredentialError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let count = seed_store_from_env_vars(&self.seeded, vars)?;
        info!(count, "seeded in-memory credentials");
        Ok(count)
    }

    /// The writable tenant store. Operator seeds are held separately.
    pub fn store(&self) -> &Arc<CredentialStore> {
        &self.store
    }
}

#[tonic::async_trait]
impl CredentialSource for MemorySource {
    fn backend(&self) -> &'static str {
        "memory"
    }

    fn location(&self) -> String {
        "process memory".to_string()
    }

    /// Resolve credentials, adding deployment-wide environment seeds for relay
    /// lookups. Tenant-specific credentials override these seeds. The own view
    /// excludes seeds: shared values may be injected on a caller's behalf but
    /// must not be returned to that caller.
    async fn resolve(
        &self,
        tenant: &str,
        entity: &str,
        service: &str,
        view: CredentialView,
    ) -> Result<ResolvedCredentials, CredentialError> {
        if view == CredentialView::Own {
            return resolve_from_store(&self.store, tenant, entity, service, view);
        }
        // Stamp both layers before reading either. A concurrent write to
        // either store makes this result stale on the next cache lookup.
        let freshness = Freshness::LayeredGeneration {
            primary: self.store.generation(),
            fallback: self.seeded.generation(),
        };
        let mut values =
            self.seeded
                .get_credentials_for_tenant(SEEDED_TENANT, WILDCARD_ENTITY, service)?;
        values.extend(
            self.store
                .get_credentials_for_tenant(tenant, entity, service)?,
        );
        Ok(ResolvedCredentials { values, freshness })
    }

    fn is_current(&self, freshness: &Freshness) -> bool {
        match freshness {
            Freshness::Generation(g) => self.store.generation() == *g,
            Freshness::LayeredGeneration { primary, fallback } => {
                self.store.generation() == *primary && self.seeded.generation() == *fallback
            }
            Freshness::Until(_) => false,
        }
    }

    /// Writable, so `SetCredentials` still works — useful for demos and tests
    /// that add a credential to a running process. The write lives only in
    /// memory and is gone at the next restart.
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

    /// Each well-known variable name resolves to its service and key.
    #[test]
    fn the_well_known_variable_names_name_their_service() {
        assert_eq!(
            credential_for_env_var("OPENAI_API_KEY"),
            Some(("openai".into(), "api_key".into()))
        );
        assert_eq!(
            credential_for_env_var("OPENFDA_API_KEY"),
            Some(("fda".into(), "api_key".into()))
        );
        assert_eq!(
            credential_for_env_var("BRAVE_API_KEY"),
            Some(("brave".into(), "api_key".into()))
        );
    }

    /// The generic convention splits at the first underscore after the
    /// prefix, so the rest of the name is the credential key.
    #[test]
    fn the_generic_convention_reads_service_then_key() {
        assert_eq!(
            credential_for_env_var("SASY_CREDENTIAL_STRIPE_SECRET_KEY"),
            Some(("stripe".into(), "secret_key".into()))
        );
        assert_eq!(
            credential_for_env_var("SASY_CREDENTIAL_S3_ACCESS_KEY_ID"),
            Some(("s3".into(), "access_key_id".into()))
        );
        // Nothing to split, or nothing after the split: not a credential.
        assert_eq!(credential_for_env_var("SASY_CREDENTIAL_STRIPE"), None);
        assert_eq!(credential_for_env_var("SASY_CREDENTIAL_STRIPE_"), None);
        assert_eq!(credential_for_env_var("PATH"), None);
        assert_eq!(credential_for_env_var("SASY_API_KEY_SUFFIX"), None);
    }

    /// Seeding makes deployment defaults available to the injection view.
    #[tokio::test]
    async fn seeding_makes_the_credential_resolvable_for_any_entity() {
        let source = MemorySource::new().unwrap();
        let n = source
            .seed([
                ("OPENAI_API_KEY".to_string(), "sk-openai".to_string()),
                (
                    "SASY_CREDENTIAL_BRAVE_API_KEY".to_string(),
                    "brave-token".to_string(),
                ),
                // Set but empty — a secret variable that was never filled in.
                ("OPENFDA_API_KEY".to_string(), String::new()),
                ("UNRELATED".to_string(), "ignored".to_string()),
            ])
            .unwrap();
        assert_eq!(n, 2, "only the two non-empty credential variables count");

        let openai = source
            .resolve("default", "some-agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            openai.values.get("api_key").map(String::as_str),
            Some("sk-openai")
        );

        let brave = source
            .resolve("default", "some-agent", "brave", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            brave.values.get("api_key").map(String::as_str),
            Some("brave-token")
        );

        let fda = source
            .resolve("default", "some-agent", "fda", CredentialView::Relay)
            .await
            .unwrap();
        assert!(fda.values.is_empty(), "an empty variable seeds nothing");
    }

    /// An entity reading itself still does not receive the tenant-wide
    /// defaults — the view rule is a property of the source, not of one
    /// backend.
    #[tokio::test]
    async fn the_own_view_excludes_the_seeded_tenant_defaults() {
        let source = MemorySource::new().unwrap();
        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-shared".to_string())])
            .unwrap();

        let own = source
            .resolve("default", "agent", "openai", CredentialView::Own)
            .await
            .unwrap();
        assert!(own.values.is_empty());
    }

    /// Relay lookups include deployment-wide seeds for every tenant, including
    /// tenants other than the one used to store the seeds.
    #[tokio::test]
    async fn a_seeded_credential_is_injected_whatever_the_callers_tenant() {
        let source = MemorySource::new().unwrap();
        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-deployment".to_string())])
            .unwrap();

        for tenant in [SEEDED_TENANT, "globex", "acme"] {
            let got = source
                .resolve(tenant, "service-client", "openai", CredentialView::Relay)
                .await
                .unwrap();
            assert_eq!(
                got.values.get("api_key").map(String::as_str),
                Some("sk-deployment"),
                "tenant {tenant} must resolve the environment's credential"
            );
        }

        // A tenant's own value still wins over the seeded one, so a
        // credential written for one tenant is not overwritten by the
        // deployment-wide default.
        source
            .set_credentials(
                "globex",
                "service-client",
                "openai",
                vec![("api_key".into(), "sk-globex".into())],
            )
            .await
            .unwrap();
        let got = source
            .resolve("globex", "service-client", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("sk-globex")
        );
        let elsewhere = source
            .resolve("acme", "service-client", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            elsewhere.values.get("api_key").map(String::as_str),
            Some("sk-deployment"),
            "the write belongs to its own tenant and does not travel"
        );
    }

    /// The seeded credentials are still shared values injected on a caller's
    /// behalf, so no tenant can read them BACK through `GetCredentials`.
    #[tokio::test]
    async fn the_own_view_never_hands_out_the_seeded_credentials() {
        let source = MemorySource::new().unwrap();
        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-deployment".to_string())])
            .unwrap();

        for tenant in [SEEDED_TENANT, "globex"] {
            let own = source
                .resolve(tenant, "service-client", "openai", CredentialView::Own)
                .await
                .unwrap();
            assert!(
                own.values.is_empty(),
                "tenant {tenant} must not be able to read the shared key"
            );
        }
        // Nor by naming itself after the entity they are stored under.
        let own = source
            .resolve("globex", WILDCARD_ENTITY, "openai", CredentialView::Own)
            .await
            .unwrap();
        assert!(own.values.is_empty());
    }

    /// An operator reseed invalidates relay reads in every tenant.
    #[tokio::test]
    async fn reseeding_invalidates_every_tenants_relay_cache() {
        let source = MemorySource::new().unwrap();
        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-old".to_string())])
            .unwrap();
        let read = source
            .resolve("globex", "service-client", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(source.is_current(&read.freshness));

        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-new".to_string())])
            .unwrap();
        assert!(
            !source.is_current(&read.freshness),
            "a rotation must retire what another tenant read before it"
        );
        let again = source
            .resolve("globex", "service-client", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert_eq!(
            again.values.get("api_key").map(String::as_str),
            Some("sk-new")
        );
    }

    #[tokio::test]
    async fn tenant_defaults_overlay_seeds_only_within_their_own_tenant() {
        let source = MemorySource::new().unwrap();
        source
            .seed([("OPENAI_API_KEY".into(), "deployment".into())])
            .unwrap();
        let before = source
            .resolve("default", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        source
            .set_credentials(
                "default",
                WILDCARD_ENTITY,
                "openai",
                vec![
                    ("api_key".into(), "tenant-default".into()),
                    ("tenant_field".into(), "default-only".into()),
                ],
            )
            .await
            .unwrap();
        assert!(!source.is_current(&before.freshness));
        for (tenant, expected) in [("default", "tenant-default"), ("acme", "deployment")] {
            let relay = source
                .resolve(tenant, "agent", "openai", CredentialView::Relay)
                .await
                .unwrap();
            assert_eq!(relay.values["api_key"], expected);
            assert_eq!(
                relay.values.contains_key("tenant_field"),
                tenant == "default"
            );
            assert!(source.is_current(&relay.freshness));
            let own = source
                .resolve(tenant, "agent", "openai", CredentialView::Own)
                .await
                .unwrap();
            assert!(own.values.is_empty());
        }
        // Even the seed helper's internal tuple is an ordinary tenant tuple
        // in the public store; an Own read cannot reach the separate seeds.
        assert!(source
            .store()
            .get_own_credentials_for_tenant("acme", WILDCARD_ENTITY, "openai",)
            .unwrap()
            .is_empty());
    }

    /// Every file under `dir`, so two snapshots can be compared. Descends
    /// into subdirectories: a database created under `src/` would leak just
    /// as much as one in the crate root.
    fn files_under(dir: &std::path::Path) -> std::collections::BTreeSet<std::path::PathBuf> {
        let mut found = std::collections::BTreeSet::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(next) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&next) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                match entry.file_type() {
                    Ok(t) if t.is_dir() => stack.push(path),
                    Ok(_) => {
                        found.insert(path);
                    }
                    Err(_) => {}
                }
            }
        }
        found
    }

    /// Nothing reaches the filesystem. The backend exists to avoid a durable
    /// plaintext copy, so a stray file would defeat it entirely.
    ///
    /// Two checks, because either one alone can be satisfied by a store that
    /// does write: SQLite is asked which file it has open, and the working
    /// directory — where a relative path would land — is swept before and
    /// after.
    #[tokio::test]
    async fn seeding_writes_no_file() {
        let cwd = std::env::current_dir().unwrap();
        let before = files_under(&cwd);

        let source = MemorySource::new().unwrap();
        assert_eq!(source.seeded.database_file(), None);
        assert_eq!(
            source.store().database_file(),
            None,
            "the memory backend must have no database file open"
        );
        source
            .seed([("OPENAI_API_KEY".to_string(), "sk-openai".to_string())])
            .unwrap();
        source
            .set_credentials(
                "default",
                "agent",
                "openai",
                vec![("api_key".into(), "sk-written".into())],
            )
            .await
            .unwrap();

        let after = files_under(&cwd);
        let new: Vec<_> = after.difference(&before).collect();
        assert!(
            new.is_empty(),
            "the in-memory backend must not create files; these appeared: {new:?}"
        );
        assert_eq!(
            source.store().database_file(),
            None,
            "not even after a write"
        );
        // And the value is readable back, so the test above is not passing
        // because nothing happened.
        let got = source
            .resolve("default", "agent", "openai", CredentialView::Own)
            .await
            .unwrap();
        assert_eq!(
            got.values.get("api_key").map(String::as_str),
            Some("sk-written")
        );
    }

    /// A write invalidates what a cache is holding, exactly as it does for
    /// the file-backed store.
    #[tokio::test]
    async fn a_write_makes_an_earlier_read_stale() {
        let source = MemorySource::new().unwrap();
        let read = source
            .resolve("default", "agent", "openai", CredentialView::Relay)
            .await
            .unwrap();
        assert!(source.is_current(&read.freshness));

        source
            .set_credentials(
                "default",
                "agent",
                "openai",
                vec![("api_key".into(), "sk-rotated".into())],
            )
            .await
            .unwrap();
        assert!(
            !source.is_current(&read.freshness),
            "a rotation must retire what was read before it"
        );
    }
}
