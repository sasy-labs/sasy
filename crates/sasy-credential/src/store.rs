//! SQLite-backed credential store with hierarchical lookup.

use std::collections::HashMap;

use parking_lot::Mutex;
use rusqlite::Connection;

use crate::error::CredentialError;

/// In-process credential store backed by SQLite.
///
/// Credentials are keyed by `(tenant_id, entity, service, key)`.
/// The entity key the tenant-wide default credentials live under.
///
/// Named rather than spelled inline so the read guard and the lookup cannot
/// disagree about what the wildcard is.
pub const WILDCARD_ENTITY: &str = "*";

/// Schema version of the credential database, held in SQLite's
/// `PRAGMA user_version`. Bump it whenever the tables change shape; a file
/// stamped with anything else is refused at open rather than read through
/// columns that no longer mean what they did.
const SCHEMA_VERSION: i64 = 1;

/// A wildcard entity `"*"` provides global defaults that
/// can be overridden by entity-specific entries.
/// Default tenant_id is `"default"`.
/// Identifies a version of the credential store's contents. Compare for
/// equality; the parts have no meaning individually.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialGeneration {
    /// Writes made through this handle.
    local: u64,
    /// SQLite's `data_version`, which moves when another connection commits.
    data_version: i64,
}

/// Supplies a fresh, never-repeating stand-in when the data version cannot be
/// read, so the caller re-reads instead of trusting a cache entry.
static UNKNOWN_DATA_VERSION: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

pub struct CredentialStore {
    db: Mutex<Connection>,
    /// Bumped on every write. Readers that cache credentials compare it to
    /// decide whether what they hold is still current — see
    /// [`Self::generation`].
    generation: std::sync::atomic::AtomicU64,
}

// Safety: parking_lot::Mutex<Connection> is Send + Sync.
unsafe impl Send for CredentialStore {}
unsafe impl Sync for CredentialStore {}

/// Decide whether a credential database that is ALREADY on disk is safe to
/// fill with secrets. Returns a warning to log, or refuses outright.
///
/// Tightening the mode is all we can do for a file we created on an earlier
/// run — but a file owned by somebody ELSE is not that. It may have been
/// planted, and whoever planted it may still hold a descriptor that no
/// `chmod` revokes, so every secret written afterwards is readable by them.
/// Refuse rather than fill it.
///
/// A file that is ours but group- or world-readable gets the same treatment
/// in miniature: the caller's `chmod` closes it to future openers, and an
/// already-open descriptor it cannot close, so say so out loud.
///
/// Split out from the caller because the refusal case cannot be built in a
/// unit test: creating a file owned by another uid needs root.
fn vet_existing_db(
    db_path: &str,
    owner_uid: u32,
    mode: u32,
    our_uid: u32,
) -> Result<Option<String>, CredentialError> {
    if owner_uid != our_uid {
        return Err(CredentialError::Config(format!(
            "credential db {db_path} is owned by uid {owner_uid}, not by this \
             process (uid {our_uid}); refusing to write secrets into a file \
             whose access is not ours to control"
        )));
    }
    if mode & 0o077 != 0 {
        return Ok(Some(format!(
            "credential db {db_path} was already on disk with mode {:o}, readable \
             beyond its owner; tightening to 0600 now, but that does not revoke a \
             descriptor somebody already holds",
            mode & 0o7777
        )));
    }
    Ok(None)
}

impl CredentialStore {
    pub fn new(db_path: &str) -> Result<Self, CredentialError> {
        // Create the file ourselves, 0600, BEFORE SQLite opens it. Chmod-ing
        // afterwards leaves a window on first creation where the file exists
        // world-readable, and a descriptor opened in that window keeps its
        // access across the chmod — so one raced first boot yields durable
        // read access to every credential written later. SQLite gives the
        // lazily-created `-wal`/`-shm` sidecars the main file's bits, so they
        // are covered by getting this one right.
        // A URI path would defeat all of this: rusqlite enables
        // `SQLITE_OPEN_URI` by default, so SQLite would resolve `file:...` to
        // some other file while the pre-create and the chmod below treat the
        // string as a literal path and quietly act on nothing — leaving the
        // real database at the process umask with no warning. Refuse rather
        // than half-protect.
        if db_path.starts_with("file:") {
            return Err(CredentialError::Config(format!(
                "credential db path {db_path} looks like a SQLite URI; pass a plain \
                 filesystem path so its permissions can be set"
            )));
        }

        #[cfg(unix)]
        if db_path != ":memory:" {
            use std::os::unix::fs::OpenOptionsExt;
            // `symlink_metadata`, not `exists`: the latter follows links and
            // reports false for a DANGLING symlink, so a planted link would
            // skip the pre-create and let SQLite create the target at umask.
            match std::fs::symlink_metadata(db_path) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err(CredentialError::Config(format!(
                        "credential db path {db_path} is a symlink; refusing, because \
                         its permissions and its target are not ours to trust"
                    )));
                }
                // Already there. See `vet_existing_db`.
                Ok(meta) => {
                    use std::os::unix::fs::MetadataExt;
                    if let Some(warning) =
                        vet_existing_db(db_path, meta.uid(), meta.mode(), unsafe {
                            libc::geteuid()
                        })?
                    {
                        tracing::warn!(db_path, "{warning}");
                    }
                }
                Err(_) => {
                    if let Err(e) = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .open(db_path)
                    {
                        // Losing this race to another sasy process is benign —
                        // it creates 0600 too. Losing it to anything else is
                        // not, because a chmod does not revoke a descriptor
                        // already opened against the file, so warn rather than
                        // whisper.
                        tracing::warn!(
                            db_path, error = %e,
                            "could not create the credential db 0600; if another writer \
                             created it, its mode and any open handle are outside our control"
                        );
                    }
                }
            }
        }
        // Flags without `SQLITE_OPEN_URI`, so the path is unambiguously the
        // file whose permissions were just set.
        let conn = Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        // Owner-only. This file holds every tenant's upstream secrets in
        // plaintext, and it was being created with the process umask —
        // commonly world-readable, so any local user, co-tenant process,
        // backup or image layer could read them. WAL mode adds `-wal` and
        // `-shm` alongside it, which carry the same data and the same umask,
        // so they are tightened too. Best-effort: a store on a filesystem
        // without Unix permissions is not a reason to refuse to start.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for suffix in ["", "-wal", "-shm"] {
                let path = format!("{db_path}{suffix}");
                if let Ok(meta) = std::fs::metadata(&path) {
                    let mut perms = meta.permissions();
                    if perms.mode() & 0o077 != 0 {
                        perms.set_mode(0o600);
                        if let Err(e) = std::fs::set_permissions(&path, perms) {
                            tracing::warn!(path, error = %e, "could not restrict credential file mode");
                        }
                    }
                }
            }
        }
        let store = Self {
            db: Mutex::new(conn),
            generation: std::sync::atomic::AtomicU64::new(0),
        };
        store.init()?;
        Ok(store)
    }

    pub fn init(&self) -> Result<(), CredentialError> {
        let db = self.db.lock();
        // Agree with the file on its schema before touching it. `0` is what a
        // fresh database and every pre-stamp one report, and today's tables
        // ARE version 1, so stamping is a record rather than a guess. Any
        // other value was written by a build whose column shapes we do not
        // know: reading a credential out of it would return the wrong field
        // silently, so refuse instead.
        let found: i64 = db.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        match found {
            0 => db.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?,
            v if v == SCHEMA_VERSION => {}
            other => {
                let path = db.path().unwrap_or("<in-memory>");
                return Err(CredentialError::Config(format!(
                    "credential db {path} has schema version {other}, but this build \
                     reads version {SCHEMA_VERSION}; migrate it or point \
                     --credentials-db at a fresh file"
                )));
            }
        }
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS credentials(
                tenant_id TEXT NOT NULL DEFAULT 'default',
                entity    TEXT NOT NULL,
                service   TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                PRIMARY KEY(tenant_id, entity, service, key)
            );",
        )?;
        Ok(())
    }

    /// Store credentials for a `(tenant, entity, service)` tuple.
    pub fn set_credentials(
        &self,
        entity: &str,
        service: &str,
        credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        self.set_credentials_for_tenant("default", entity, service, credentials)
    }

    /// Store credentials with explicit tenant_id.
    pub fn set_credentials_for_tenant(
        &self,
        tenant_id: &str,
        entity: &str,
        service: &str,
        credentials: Vec<(String, String)>,
    ) -> Result<(), CredentialError> {
        // One transaction. Written as a bare loop, a failure partway left the
        // set half-updated — and a concurrent reader could see a mix of old
        // and new keys for one (tenant, entity, service), which for a
        // credential pair like id+secret is worse than either version.
        let mut db = self.db.lock();
        let tx = db.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO \
                 credentials(tenant_id, entity, service, key, value) \
                 VALUES(?1, ?2, ?3, ?4, ?5)",
            )?;
            for (key, value) in &credentials {
                stmt.execute(rusqlite::params![tenant_id, entity, service, key, value])?;
            }
        }
        tx.commit()?;
        // After the commit, so a reader that observes the new generation is
        // guaranteed to read the new rows.
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// A value that changes whenever any credential is written, by this
    /// process or another one.
    ///
    /// Callers that cache credentials hold this alongside them and discard
    /// what they have when it moves. Deliberately coarse — one value for the
    /// whole store rather than per key — because a rotation has to take
    /// effect immediately and the cost of an occasional unnecessary re-read
    /// is one SQLite point query. A per-key scheme would be cheaper and would
    /// also be a second place to get invalidation wrong.
    ///
    /// Two halves, because neither alone is enough. The in-process counter
    /// catches writes through this handle. `PRAGMA data_version` catches
    /// writes committed by any OTHER connection — and deliberately does not
    /// change for this connection's own writes, which is why both are needed.
    /// That second half is not hypothetical: the documented way to load
    /// credentials is `init-credentials`, a separate process writing the same
    /// file, so a counter that only saw in-process writes would leave a
    /// rotated key being injected by a running server indefinitely.
    pub fn generation(&self) -> CredentialGeneration {
        // Lock FIRST. Loading the counter before taking the mutex let a writer
        // that already held it commit and bump between the two, so this call
        // returned the pre-write pair — and `data_version` does not move for
        // this connection's own writes, so nothing else caught it. The cache
        // then hit on a credential the rotation had already replaced.
        let db = self.db.lock();
        let local = self.generation.load(std::sync::atomic::Ordering::Acquire);
        match db.query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0)) {
            Ok(data_version) => CredentialGeneration {
                local,
                data_version,
            },
            Err(e) => {
                // Cannot tell whether anything changed, so say "changed":
                // a value nothing can have cached forces a re-read. Failing
                // the other way would serve a possibly-revoked credential.
                tracing::warn!(error = %e, "reading the credential data version failed");
                CredentialGeneration {
                    local,
                    data_version: UNKNOWN_DATA_VERSION
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed),
                }
            }
        }
    }

    /// The file SQLite has open for this store, or `None` when the database
    /// lives in memory and none of it reaches the filesystem.
    ///
    /// Asked of SQLite rather than remembered from the path that was passed
    /// in, so it answers what is actually open. The in-memory backend exists
    /// precisely to leave no durable plaintext copy behind, and that claim
    /// needs something to check it against.
    pub fn database_file(&self) -> Option<String> {
        let db = self.db.lock();
        db.path()
            .filter(|path| !path.is_empty())
            .map(str::to_string)
    }

    /// Get credentials with hierarchical lookup:
    /// 1. Load global credentials (`entity = "*"`)
    /// 2. Overlay entity-specific credentials
    pub fn get_credentials(
        &self,
        entity: &str,
        service: &str,
    ) -> Result<HashMap<String, String>, CredentialError> {
        self.get_credentials_for_tenant("default", entity, service)
    }

    /// Get credentials with explicit tenant_id and hierarchical lookup:
    /// the tenant-wide `*` defaults with the entity's own rows overlaid.
    ///
    /// This is the injection view. A caller reading on someone's BEHALF wants
    /// it; a caller reading its OWN credentials must not get it — see
    /// [`Self::get_own_credentials_for_tenant`].
    pub fn get_credentials_for_tenant(
        &self,
        tenant_id: &str,
        entity: &str,
        service: &str,
    ) -> Result<HashMap<String, String>, CredentialError> {
        self.lookup(tenant_id, entity, service, true)
    }

    /// An entity's OWN credentials, without the tenant-wide `*` defaults.
    ///
    /// The read RPC confines a plain `credential-reader` to its own principal
    /// and says reading the wildcard needs `service-proxy` — but the
    /// hierarchical lookup merges `*` into every entity's result, so asking
    /// for your own credentials handed you the tenant's shared platform
    /// secrets anyway, and the wildcard check guarded nothing. The two views
    /// are now separate so the access rule and the query agree.
    pub fn get_own_credentials_for_tenant(
        &self,
        tenant_id: &str,
        entity: &str,
        service: &str,
    ) -> Result<HashMap<String, String>, CredentialError> {
        self.lookup(tenant_id, entity, service, false)
    }

    fn lookup(
        &self,
        tenant_id: &str,
        entity: &str,
        service: &str,
        include_wildcard: bool,
    ) -> Result<HashMap<String, String>, CredentialError> {
        let db = self.db.lock();
        let sql = if include_wildcard {
            "SELECT entity, key, value \
             FROM credentials \
             WHERE tenant_id = ?1 \
               AND entity IN ('*', ?2) \
               AND service = ?3 \
             ORDER BY CASE entity \
               WHEN '*' THEN 0 ELSE 1 END"
        } else {
            "SELECT entity, key, value \
             FROM credentials \
             WHERE tenant_id = ?1 \
               AND entity = ?2 \
               AND service = ?3"
        };
        let mut stmt = db.prepare(sql)?;

        let mut merged = HashMap::new();
        let rows = stmt.query_map(rusqlite::params![tenant_id, entity, service], |row| {
            let key: String = row.get(1)?;
            let value: String = row.get(2)?;
            Ok((key, value))
        })?;
        for row in rows {
            let (key, value) = row?;
            merged.insert(key, value);
        }
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn memory_store() -> CredentialStore {
        CredentialStore::new(":memory:").unwrap()
    }

    /// The rotation counter must not be read before the lock that guards the
    /// write it is supposed to describe.
    ///
    /// `generation()` loaded the in-process counter and only THEN took the db
    /// mutex. A writer already inside its transaction commits and bumps the
    /// counter while still holding that mutex, so the reader queued behind it
    /// returned the pre-write pair for a write that had already landed — and
    /// `PRAGMA data_version` does not move for this connection's own writes,
    /// so the second half of the generation did not catch it either. A cache
    /// holding that pair then hit on a credential the rotation had replaced.
    ///
    /// Staged by making the write hold the lock long enough for the reader to
    /// arrive mid-transaction. If the reader gets there first the run proves
    /// nothing, so the size grows until it does.
    #[test]
    fn the_generation_reflects_a_write_that_committed_while_it_waited() {
        for rows in [5_000usize, 50_000, 400_000] {
            let store = Arc::new(memory_store());
            store
                .set_credentials_for_tenant(
                    "acme",
                    "agent",
                    "openai",
                    vec![("key".into(), "v0".into())],
                )
                .unwrap();
            let before = store.generation();

            let big: Vec<(String, String)> = (0..rows)
                .map(|i| (format!("k{i}"), format!("v{i}")))
                .collect();

            let writer = {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    store
                        .set_credentials_for_tenant("acme", "agent", "openai", big)
                        .unwrap();
                })
            };

            // Let the writer get into its transaction — and therefore behind
            // the mutex — before the reader asks.
            std::thread::sleep(Duration::from_millis(5));
            let reader = {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let started = std::time::Instant::now();
                    let g = store.generation();
                    // Did we actually queue behind the writer? An uncontended
                    // `generation()` is one point query, microseconds; a flag
                    // set by the writer thread would be no good here, because
                    // it lands AFTER the mutex is released and the reader can
                    // already be through.
                    (g, started.elapsed() > Duration::from_millis(2))
                })
            };

            writer.join().unwrap();
            let (after, waited) = reader.join().unwrap();
            if !waited {
                // The write finished before the reader arrived: this run says
                // nothing about the ordering. Try a longer one.
                continue;
            }
            assert_ne!(
                before, after,
                "a generation read that waited out the write must not describe \
                 the state before it"
            );
            return;
        }
        panic!("could not stage a read that overlapped the write");
    }

    /// A credential db already on disk and owned by somebody else is refused,
    /// not adopted.
    ///
    /// The owning uid is the thing a `chmod` cannot fix: whoever placed the
    /// file may still hold an open descriptor, and every secret written after
    /// that is readable through it. The refusal branch cannot be built for
    /// real in a unit test — creating a file owned by another uid needs root —
    /// so the decision is tested directly.
    #[test]
    fn a_pre_existing_credential_db_owned_by_another_uid_is_refused() {
        let ours = 1000;

        let err = vet_existing_db("/var/lib/sasy/creds.db", 0, 0o600, ours)
            .expect_err("a root-owned db must not be adopted");
        let msg = format!("{err}");
        assert!(
            msg.contains("uid 0"),
            "the refusal must name the owner: {msg}"
        );

        assert!(
            vet_existing_db("/var/lib/sasy/creds.db", ours, 0o600, ours)
                .expect("our own 0600 db is exactly what we expect to find")
                .is_none(),
            "no warning is due for a file we own at 0600"
        );

        let warning = vet_existing_db("/var/lib/sasy/creds.db", ours, 0o100644, ours)
            .expect("our own db is still usable when it is too open")
            .expect("a group/world-readable db must warn");
        assert!(
            warning.contains("644"),
            "the warning must name the mode it found: {warning}"
        );
    }

    /// Reading your own credentials must not hand you the tenant's shared
    /// ones.
    ///
    /// The read RPC confines a plain `credential-reader` to its own principal
    /// and requires `service-proxy` to read the wildcard — but the
    /// hierarchical lookup merged `*` into every entity's result, so that
    /// second rule guarded nothing. A relay still gets the merged view,
    /// because that is what gets injected.
    #[test]
    fn an_entitys_own_view_excludes_the_tenant_wide_defaults() {
        let store = memory_store();
        store
            .set_credentials_for_tenant(
                "acme",
                "*",
                "openai",
                vec![("platform_key".into(), "sk-shared".into())],
            )
            .unwrap();
        store
            .set_credentials_for_tenant(
                "acme",
                "agent",
                "openai",
                vec![("own_key".into(), "sk-mine".into())],
            )
            .unwrap();

        let own = store
            .get_own_credentials_for_tenant("acme", "agent", "openai")
            .unwrap();
        assert_eq!(own.get("own_key").map(String::as_str), Some("sk-mine"));
        assert!(
            !own.contains_key("platform_key"),
            "an entity reading itself must not receive the tenant's shared secret"
        );

        let merged = store
            .get_credentials_for_tenant("acme", "agent", "openai")
            .unwrap();
        assert_eq!(
            merged.get("platform_key").map(String::as_str),
            Some("sk-shared"),
            "the injection view still overlays the tenant defaults"
        );
        assert_eq!(merged.get("own_key").map(String::as_str), Some("sk-mine"));
    }

    /// A write from ANOTHER process has to move the generation too.
    ///
    /// The documented way to load credentials is `init-credentials`, a
    /// separate process writing the same file — so a version that only
    /// counted writes through one handle left a running server injecting a
    /// rotated key indefinitely, which is the failure the version exists to
    /// prevent.
    #[test]
    fn a_write_from_another_connection_moves_the_generation() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("creds.db").to_string_lossy().into_owned();

        let serving = CredentialStore::new(&path).unwrap();
        // Read once so the connection has an established data version.
        let before = serving.generation();

        // A second handle on the same file, standing in for the CLI.
        let loader = CredentialStore::new(&path).unwrap();
        loader
            .set_credentials_for_tenant("acme", "*", "openai", vec![("k".into(), "rotated".into())])
            .unwrap();

        assert_ne!(
            serving.generation(),
            before,
            "a write by another connection must be visible to the serving handle"
        );
    }

    /// A write has to move the generation, or a cache keyed on it never
    /// notices a rotation — which is the whole point of the counter.
    #[test]
    fn every_write_moves_the_generation() {
        let store = memory_store();
        let before = store.generation();

        store
            .set_credentials_for_tenant("acme", "svc", "openai", vec![("k".into(), "v1".into())])
            .unwrap();
        let after_first = store.generation();
        assert_ne!(after_first, before, "a write must move the generation");

        store
            .set_credentials_for_tenant("acme", "svc", "openai", vec![("k".into(), "v2".into())])
            .unwrap();
        assert_ne!(
            store.generation(),
            after_first,
            "a rotation of the same key must move it too"
        );
    }

    /// A multi-key write is all-or-nothing, so a reader never sees half a
    /// credential pair.
    #[test]
    fn a_multi_key_write_is_atomic() {
        let store = memory_store();
        store
            .set_credentials_for_tenant(
                "acme",
                "svc",
                "s3",
                vec![
                    ("access_key".into(), "AK1".into()),
                    ("secret_key".into(), "SK1".into()),
                ],
            )
            .unwrap();
        let got = store
            .get_credentials_for_tenant("acme", "svc", "s3")
            .unwrap();
        assert_eq!(got.get("access_key").map(String::as_str), Some("AK1"));
        assert_eq!(got.get("secret_key").map(String::as_str), Some("SK1"));
    }

    #[test]
    fn set_and_get_credentials() {
        let store = memory_store();
        store
            .set_credentials(
                "alice",
                "openai",
                vec![
                    ("api_key".into(), "sk-123".into()),
                    ("org_id".into(), "org-1".into()),
                ],
            )
            .unwrap();

        let creds = store.get_credentials("alice", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "sk-123");
        assert_eq!(creds.get("org_id").unwrap(), "org-1");
    }

    #[test]
    fn empty_result_when_no_credentials() {
        let store = memory_store();
        let creds = store.get_credentials("bob", "openai").unwrap();
        assert!(creds.is_empty());
    }

    #[test]
    fn wildcard_fallback() {
        let store = memory_store();
        store
            .set_credentials("*", "openai", vec![("api_key".into(), "global-key".into())])
            .unwrap();

        let creds = store.get_credentials("alice", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "global-key");
    }

    #[test]
    fn entity_overrides_wildcard() {
        let store = memory_store();
        store
            .set_credentials(
                "*",
                "openai",
                vec![
                    ("api_key".into(), "global-key".into()),
                    ("org_id".into(), "global-org".into()),
                ],
            )
            .unwrap();
        store
            .set_credentials(
                "alice",
                "openai",
                vec![("api_key".into(), "alice-key".into())],
            )
            .unwrap();

        let creds = store.get_credentials("alice", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "alice-key");
        assert_eq!(creds.get("org_id").unwrap(), "global-org");
    }

    #[test]
    fn tenant_isolation() {
        let store = memory_store();
        store
            .set_credentials_for_tenant(
                "tenant-a",
                "*",
                "openai",
                vec![("api_key".into(), "key-a".into())],
            )
            .unwrap();
        store
            .set_credentials_for_tenant(
                "tenant-b",
                "*",
                "openai",
                vec![("api_key".into(), "key-b".into())],
            )
            .unwrap();

        let a = store
            .get_credentials_for_tenant("tenant-a", "user", "openai")
            .unwrap();
        let b = store
            .get_credentials_for_tenant("tenant-b", "user", "openai")
            .unwrap();
        assert_eq!(a.get("api_key").unwrap(), "key-a");
        assert_eq!(b.get("api_key").unwrap(), "key-b");

        // Default tenant sees nothing
        let d = store.get_credentials("user", "openai").unwrap();
        assert!(d.is_empty());
    }

    #[test]
    fn upsert_overwrites_existing_key() {
        let store = memory_store();
        store
            .set_credentials("alice", "openai", vec![("api_key".into(), "old".into())])
            .unwrap();
        store
            .set_credentials("alice", "openai", vec![("api_key".into(), "new".into())])
            .unwrap();

        let creds = store.get_credentials("alice", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "new");
    }

    #[test]
    fn concurrent_access() {
        use std::sync::Arc;

        let store = Arc::new(memory_store());
        let mut handles = vec![];

        for i in 0..10 {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let entity = format!("user-{i}");
                s.set_credentials(&entity, "svc", vec![("key".into(), format!("val-{i}"))])
                    .unwrap();
                let creds = s.get_credentials(&entity, "svc").unwrap();
                assert_eq!(creds.get("key").unwrap(), &format!("val-{i}"));
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
    /// A fresh database is stamped with the current schema version, so a
    /// later build that changes the tables has something to compare against
    /// instead of guessing at the shape of what it finds.
    #[test]
    fn a_fresh_store_reads_back_the_current_schema_version() {
        let store = memory_store();
        let db = store.db.lock();
        let found: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(found, SCHEMA_VERSION);
    }

    /// A database written by a build with different columns is refused rather
    /// than read through them — a credential fetched out of the wrong shape
    /// comes back wrong, quietly.
    #[test]
    fn a_store_from_another_schema_version_is_refused() {
        let store = memory_store();
        store
            .db
            .lock()
            .execute_batch("PRAGMA user_version = 7")
            .unwrap();

        let err = store
            .init()
            .expect_err("a version this build cannot read must not be opened");
        let msg = err.to_string();
        assert!(
            msg.contains("schema version 7") && msg.contains("version 1"),
            "the refusal must name both versions: {msg}"
        );
    }
}
