//! LRU cache for compiled Soufflé evaluator binaries.
//!
//! A full re-compile of the airline policy is ~7s — dominated by g++.
//! Re-uploading the same (or a comment-only-changed) policy pays the
//! cost again every time. This module memoizes the compiled ELF
//! keyed on a SHA-256 of the inputs that actually affect the output:
//! the desugared policy source, custom functor source, the Soufflé
//! word size, the magic-set transform setting, and the shim, evaluator
//! protocol, JSON codec, and common-functor sources bundled into every
//! binary, plus the selected Soufflé and C++ executable SHA-256 digests
//! and version strings (so a toolchain change misses the cache correctly
//! even when two executables report the same version).
//!
//! Activation: set `SASY_SOUFFLE_BUILD_CACHE_DIR=<path>`. The Rust
//! binary defaults this to `<data-dir>/souffle-build-cache` at
//! startup; tests and scripts can leave it unset to disable.
//!
//! Eviction: oldest-mtime first when the cache exceeds
//! `SASY_SOUFFLE_BUILD_CACHE_MAX_BYTES` (default 1 GiB), enforced on
//! every insert. The cache is content-addressed so two uploads that
//! produce the same bytes share the entry — and a partial / racy
//! populate is benign because identical keys imply identical content.
//!
//! Startup pruning ([`prune_at_startup`]): on boot the whole cache is
//! cleared when the toolchain/key-schema signature changed (every
//! entry's key embedded the old Soufflé/g++/arch/os/schema, so none
//! would hit again), and entries idle longer than
//! `SASY_SOUFFLE_BUILD_CACHE_MAX_AGE_DAYS` (default 30; a hit bumps
//! mtime) are removed. This complements the on-insert byte budget,
//! which alone never reclaims a stale-but-under-budget cache.
//!
//! Tenant keying: off by default. A `SASY_SOUFFLE_BUILD_CACHE_PER_TENANT=1`
//! opt-in adds the tenant id to the key for deployments that don't
//! want existence-of-identical-policy probing across tenants. The
//! signal leaked otherwise is narrow (cache hit ⇒ another tenant
//! uploaded the *exact* same desugared bytes), so we don't pay the
//! cross-tenant amortization loss by default.
//!
//! NOTE: the key only separates tenants if the caller passes a real
//! `tenant_id`. The runtime compile path currently passes `None` (the
//! compiler's public API isn't tenant-aware), so with the flag on the cache is
//! still shared — [`warn_per_tenant_not_wired`] warns once instead of failing
//! silently. Limitation: the flag has no effect until the compiler API takes a
//! tenant.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::hash::{hex, sha256_hex};

const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024; // 1 GiB
const DEFAULT_MAX_AGE_DAYS: u64 = 30;

/// Bumped whenever the cache-key schema changes. Embedded in both the
/// key and the on-disk build signature, so a schema change makes every
/// existing entry unreachable *and* gets the now-orphaned entries
/// pruned at startup rather than lingering until byte-budget eviction.
// v7: includes exact Soufflé and C++ executable bytes in cache identity.
const CACHE_SCHEMA_VERSION: &str = "v7";

/// Marker file recording the toolchain/schema signature the cached
/// entries were built under. Skipped by eviction + key lookups (it's a
/// dotfile; keys are 64-char hex).
const SIGNATURE_FILE: &str = ".build-signature";

/// The g++ flags that affect the emitted binary (paths, `-o`, `-lpthread`
/// and `-DRAM_DOMAIN_SIZE` excluded — the latter is keyed via `word_size`).
/// Single source of truth shared by the g++ invocation
/// ([`crate::compiler`]), the cache key, and the build signature, so an
/// edit here both misses stale entries *and* eagerly clears them at
/// startup.
pub(crate) const GXX_OUTPUT_FLAGS: &[&str] = &["-std=c++17", "-O2", "-D__EMBEDDED_SOUFFLE__"];

/// Inputs that uniquely identify a compiled binary.
pub struct CacheKeyInputs<'a> {
    pub desugared_source: &'a str,
    pub word_size: u8,
    pub magic_set: Option<&'a str>,
    pub functor_source: Option<&'a str>,
    pub evaluator_shim_source: &'a str,
    pub evaluator_protocol_source: &'a str,
    pub json_string_codec_source: &'a str,
    pub functors_common_source: &'a str,
    pub souffle_version: &'a str,
    pub souffle_sha256: &'a str,
    pub cxx_version: &'a str,
    pub cxx_sha256: &'a str,
    /// Used iff [`per_tenant`] is true. Pass `None` until the tenant
    /// id is plumbed through the upload path.
    pub tenant_id: Option<&'a str>,
}

/// Compute the cache key for a set of inputs.
pub fn cache_key(inputs: &CacheKeyInputs<'_>) -> String {
    let mut h = Sha256::new();
    let _ = writeln!(
        HashWrite(&mut h),
        "sasy-souffle-cache-{}",
        CACHE_SCHEMA_VERSION
    );
    let _ = writeln!(HashWrite(&mut h), "word_size={}", inputs.word_size);
    let _ = writeln!(
        HashWrite(&mut h),
        "magic={}",
        inputs.magic_set.unwrap_or("")
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "souffle_version={}",
        inputs.souffle_version
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "souffle_sha256={}",
        inputs.souffle_sha256
    );
    let _ = writeln!(HashWrite(&mut h), "gxx_version={}", inputs.cxx_version);
    let _ = writeln!(HashWrite(&mut h), "gxx_sha256={}", inputs.cxx_sha256);
    let _ = writeln!(
        HashWrite(&mut h),
        "gxx_flags={}",
        GXX_OUTPUT_FLAGS.join(" ")
    );
    // Two hosts with the same Soufflé/g++ version strings can still
    // produce incompatible ELFs across arch / OS. A shared cache
    // volume (or a stale cache after a host migration) must miss in
    // that case rather than hand back an unloadable binary.
    let _ = writeln!(HashWrite(&mut h), "target_arch={}", std::env::consts::ARCH);
    let _ = writeln!(HashWrite(&mut h), "target_os={}", std::env::consts::OS);
    if per_tenant() {
        // A set flag with no tenant id keys by "" — i.e. NO per-tenant
        // separation. The runtime compile path currently passes None, so warn
        // once rather than silently behaving as if the flag did nothing.
        if inputs.tenant_id.is_none() {
            warn_per_tenant_not_wired();
        }
        let _ = writeln!(
            HashWrite(&mut h),
            "tenant={}",
            inputs.tenant_id.unwrap_or("")
        );
    }
    let _ = writeln!(
        HashWrite(&mut h),
        "shim_sha={}",
        sha256_hex(inputs.evaluator_shim_source)
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "evaluator_protocol_sha={}",
        sha256_hex(inputs.evaluator_protocol_source)
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "json_string_codec_sha={}",
        sha256_hex(inputs.json_string_codec_source)
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "functors_common_sha={}",
        sha256_hex(inputs.functors_common_source)
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "functors_sha={}",
        sha256_hex(inputs.functor_source.unwrap_or(""))
    );
    let _ = writeln!(
        HashWrite(&mut h),
        "desugared_sha={}",
        sha256_hex(inputs.desugared_source)
    );
    hex(&h.finalize())
}

/// True iff caching is wired (env var present and non-empty).
pub fn enabled() -> bool {
    cache_root().is_some()
}

/// Try to fetch a cached binary. On hit, hardlink (or copy if cross-fs)
/// into ``dest`` and bump the cache entry's mtime. Returns true iff
/// the dest is now populated from the cache.
pub fn try_get(key: &str, dest: &Path) -> bool {
    try_get_in(None, key, dest)
}

/// [`try_get`] against an explicit cache directory.
///
/// `root` overrides what the environment names, and `None` falls back to it.
/// A caller that passes its own directory is not sharing a cache with anything
/// else in the process, which is what lets a test count the entries its own
/// compiles produced.
pub fn try_get_in(root: Option<&Path>, key: &str, dest: &Path) -> bool {
    let Some(root) = root.map(PathBuf::from).or_else(cache_root) else {
        return false;
    };
    let src = root.join(key);
    if !src.exists() {
        return false;
    }
    if let Some(parent) = dest.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            warn!(
                "souffle cache: dest mkdir {} failed: {}",
                parent.display(),
                e
            );
            return false;
        }
    }
    // Best-effort remove a stale dest (the upload-scoped temp dir is
    // single-writer so this is racy only against the very rare case
    // where a previous compile in this same dir produced a binary).
    let _ = std::fs::remove_file(dest);
    let used_hardlink = match std::fs::hard_link(&src, dest) {
        Ok(()) => true,
        Err(e) => {
            debug!(
                "souffle cache: hardlink {} -> {} failed ({}); falling back to copy",
                src.display(),
                dest.display(),
                e
            );
            if let Err(e2) = std::fs::copy(&src, dest) {
                warn!(
                    "souffle cache: copy {} -> {} failed: {}",
                    src.display(),
                    dest.display(),
                    e2
                );
                return false;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755));
            }
            false
        }
    };
    // Defence-in-depth: a truncated / non-executable cached binary
    // would hand Soufflé a path it can't exec. Treat that as a miss
    // and fall back to the compile path. Cheap: a single stat call.
    if !is_valid_executable(dest) {
        warn!(
            "souffle cache: populated {} is not a valid executable; treating as miss",
            dest.display()
        );
        let _ = std::fs::remove_file(dest);
        return false;
    }
    let _ = touch(&src);
    debug!(key, hardlink = used_hardlink, "souffle cache hit");
    true
}

/// True iff ``p`` is a non-empty file the current process can exec.
fn is_valid_executable(p: &Path) -> bool {
    let Ok(m) = std::fs::metadata(p) else {
        return false;
    };
    if !m.is_file() || m.len() == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Any execute bit set is enough — the kernel decides who
        // can actually run it; we just want to rule out the
        // "ELF accidentally written without +x" case.
        m.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Insert a freshly-compiled binary into the cache. No-op if disabled.
/// Eviction runs after successful insert and may free space to fit
/// the byte budget.
pub fn put(key: &str, built: &Path) {
    put_in(None, key, built)
}

/// [`put`] against an explicit cache directory; see [`try_get_in`].
pub fn put_in(root: Option<&Path>, key: &str, built: &Path) {
    let Some(root) = root.map(PathBuf::from).or_else(cache_root) else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(&root) {
        warn!("souffle cache: mkdir {} failed: {}", root.display(), e);
        return;
    }
    let dst = root.join(key);
    // Two concurrent puts with the same key produce byte-identical
    // content (key is a hash of the inputs), so EEXIST on hard_link
    // is a benign race-loss, not an error.
    match std::fs::hard_link(built, &dst) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => {
            // Cross-filesystem or EPERM. Fall back to copy via
            // tmpfile + rename so a reader never sees a partial file.
            let tmp = root.join(format!("{}.tmp.{}", key, std::process::id()));
            if let Err(e) = std::fs::copy(built, &tmp) {
                warn!(
                    "souffle cache: copy {} -> {} failed: {}",
                    built.display(),
                    tmp.display(),
                    e
                );
                return;
            }
            if let Err(e) = std::fs::rename(&tmp, &dst) {
                warn!(
                    "souffle cache: rename {} -> {} failed: {}",
                    tmp.display(),
                    dst.display(),
                    e
                );
                let _ = std::fs::remove_file(&tmp);
                return;
            }
        }
    }
    let _ = touch(&dst);
    debug!(key, "souffle cache populated");
    enforce_budget(&root, max_bytes());
}

// ------------------------------------------------------------------
// Internals
// ------------------------------------------------------------------

/// The directory `SASY_SOUFFLE_BUILD_CACHE_DIR` names, or `None` when the
/// cache is switched off. [`crate::assets`] materializes the embedded
/// compile-chain assets under the same directory, so the two on-disk artifacts
/// of a compile live in one place.
pub(crate) fn cache_root() -> Option<PathBuf> {
    // Re-read each call. Compile is rare (one env-var lookup per
    // upload), and not memoizing keeps tests trivial — they can
    // toggle the env between runs.
    std::env::var("SASY_SOUFFLE_BUILD_CACHE_DIR")
        .ok()
        .filter(|s| !s.is_empty() && !matches!(s.as_str(), "off" | "false" | "0"))
        .map(PathBuf::from)
}

fn max_bytes() -> u64 {
    std::env::var("SASY_SOUFFLE_BUILD_CACHE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_BYTES)
}

fn per_tenant() -> bool {
    sasy_common::env_flag("SASY_SOUFFLE_BUILD_CACHE_PER_TENANT")
}

/// One-time warning that the per-tenant flag is set but no tenant id reached the
/// cache key, so the build cache is still shared across tenants.
fn warn_per_tenant_not_wired() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "SASY_SOUFFLE_BUILD_CACHE_PER_TENANT is set but the build-cache key got \
             no tenant id (the runtime compile path passes None), so the cache stays \
             SHARED across tenants. Plumb tenant_id through the compiler to enable it."
        );
    });
}

/// Max idle age before a cached entry is age-pruned at startup. The
/// entry's mtime is bumped on every hit ([`try_get`]), so this only
/// reclaims binaries that haven't been used in this long. ``0`` disables
/// age pruning (the byte budget still bounds total size).
fn max_age() -> Option<std::time::Duration> {
    let days = std::env::var("SASY_SOUFFLE_BUILD_CACHE_MAX_AGE_DAYS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_AGE_DAYS);
    (days > 0).then(|| std::time::Duration::from_secs(days * 24 * 60 * 60))
}

/// Fingerprint of everything that makes existing entries *reachable*:
/// the key schema plus the toolchain/platform the keys embed. If this
/// changes, every entry built under the old signature is unreachable
/// (its key hashed the old values) — so the whole cache is stale.
fn build_signature() -> Result<String, String> {
    let (_, souffle_version, souffle_sha256) =
        crate::compiler::selected_tool_version_and_digest("souffle", "souffle")?;
    let (_, cxx_version, cxx_sha256) =
        crate::compiler::selected_tool_version_and_digest("cxx", "g++")?;
    Ok(build_signature_for_toolchain(
        &souffle_version,
        &souffle_sha256,
        &cxx_version,
        &cxx_sha256,
    ))
}

fn build_signature_for_toolchain(
    souffle_version: &str,
    souffle_sha256: &str,
    cxx_version: &str,
    cxx_sha256: &str,
) -> String {
    format!(
        "schema={CACHE_SCHEMA_VERSION}\nsouffle={}\nsouffle_sha256={}\ngxx={}\ngxx_sha256={}\ngxx_flags={}\narch={}\nos={}\n",
        souffle_version,
        souffle_sha256,
        cxx_version,
        cxx_sha256,
        GXX_OUTPUT_FLAGS.join(" "),
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

/// Prune stale cache entries at startup — the two cases the on-write
/// byte budget ([`enforce_budget`]) doesn't cover:
///
///  1. **Toolchain / key-schema change.** Every existing entry's key
///     hashed the old Soufflé/g++/arch/os/schema, so none will hit
///     again. Detected via the [`SIGNATURE_FILE`] marker; on mismatch
///     the whole cache is cleared (and the marker rewritten).
///  2. **Long-idle entries.** Removed when older than
///     `SASY_SOUFFLE_BUILD_CACHE_MAX_AGE_DAYS` (default 30). Hits bump
///     mtime, so actively-used entries survive.
///
/// No-op when the cache is disabled. Best-effort: every failure is
/// logged, never fatal.
pub fn prune_at_startup() {
    let Some(root) = cache_root() else {
        return;
    };
    let signature = match build_signature() {
        Ok(signature) => signature,
        Err(error) => {
            warn!("souffle cache: cannot authenticate toolchain: {error}");
            return;
        }
    };
    prune(&root, &signature, max_age());
}

/// Testable core of [`prune_at_startup`]: signature-mismatch clear +
/// age-based prune, with the toolchain signature and TTL passed in.
fn prune(root: &Path, signature: &str, max_age: Option<std::time::Duration>) {
    let _g = budget_lock().lock();
    if let Err(e) = std::fs::create_dir_all(root) {
        warn!(
            "souffle cache: prune mkdir {} failed: {}",
            root.display(),
            e
        );
        return;
    }
    let sig_path = root.join(SIGNATURE_FILE);
    let have = std::fs::read_to_string(&sig_path).unwrap_or_default();
    if have != signature {
        // Toolchain/schema changed (or first populated run): drop every
        // cached binary — they're keyed under the old signature and will
        // never be hit again. Keep the signature file itself. Unlike
        // `enforce_budget`, this also removes any abandoned
        // `<key>.tmp.<pid>` files: prune runs once at startup *before* the
        // gRPC listener binds, so there is no concurrent producer whose
        // in-flight tmp we could clobber.
        let mut cleared = 0u64;
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p == sig_path {
                    continue;
                }
                if e.file_type().map(|t| t.is_file()).unwrap_or(false)
                    && std::fs::remove_file(&p).is_ok()
                {
                    cleared += 1;
                }
            }
        }
        if let Err(e) = std::fs::write(&sig_path, signature) {
            warn!("souffle cache: write build signature failed: {}", e);
        }
        if cleared > 0 {
            debug!(
                cleared,
                "souffle cache: toolchain/schema changed; cleared stale entries"
            );
        }
        return;
    }
    // Same toolchain: age-prune long-idle entries.
    let Some(max_age) = max_age else {
        return;
    };
    let now = std::time::SystemTime::now();
    let mut pruned = 0u64;
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.contains(".tmp.") {
                continue;
            }
            let Ok(m) = e.metadata() else { continue };
            if !m.is_file() {
                continue;
            }
            let too_old = m
                .modified()
                .ok()
                .and_then(|mt| now.duration_since(mt).ok())
                .map(|age| age > max_age)
                .unwrap_or(false);
            if too_old && std::fs::remove_file(e.path()).is_ok() {
                pruned += 1;
            }
        }
    }
    if pruned > 0 {
        debug!(
            pruned,
            "souffle cache: startup age-prune removed idle entries"
        );
    }
}

/// Adapter so `writeln!` can stream into the SHA-256 hasher without
/// allocating intermediate strings.
struct HashWrite<'a>(&'a mut Sha256);

impl std::fmt::Write for HashWrite<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.update(s.as_bytes());
        Ok(())
    }
}

fn touch(p: &Path) -> std::io::Result<()> {
    let f = std::fs::File::open(p)?;
    f.set_modified(std::time::SystemTime::now())
}

/// Serialize budget enforcement so two concurrent puts can't race
/// each other into evicting the same entry twice.
fn budget_lock() -> &'static Mutex<()> {
    static M: OnceLock<Mutex<()>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(()))
}

fn enforce_budget(root: &Path, max_bytes: u64) {
    let _g = budget_lock().lock();
    let read = match std::fs::read_dir(root) {
        Ok(it) => it,
        Err(e) => {
            warn!("souffle cache: read_dir {} failed: {}", root.display(), e);
            return;
        }
    };
    let mut entries: Vec<(PathBuf, std::time::SystemTime, u64)> = read
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            if !m.is_file() {
                return None;
            }
            // Skip in-flight tmp files (`<key>.tmp.<pid>`, owned by the
            // producer) and internal dotfiles like the build signature —
            // evicting the latter would wipe the toolchain marker and
            // force a full recompile next boot. Real keys are 64-char hex.
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.contains(".tmp.") {
                return None;
            }
            let mt = m.modified().ok()?;
            Some((e.path(), mt, m.len()))
        })
        .collect();
    let total: u64 = entries.iter().map(|(_, _, sz)| sz).sum();
    if total <= max_bytes {
        return;
    }
    entries.sort_by_key(|(_, mt, _)| *mt);
    let mut remaining = total;
    for (path, _, sz) in entries {
        if remaining <= max_bytes {
            break;
        }
        if let Err(e) = std::fs::remove_file(&path) {
            warn!("souffle cache: evict {} failed: {}", path.display(), e);
            continue;
        }
        remaining = remaining.saturating_sub(sz);
        debug!(path = %path.display(), bytes = sz, "evicted from souffle cache");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    /// The cache reads its config once via `OnceLock` for the lifetime
    /// of the process. Tests share that state, so we serialize them
    /// here and reset the env vars per test. (The OnceLocks aren't
    /// resettable, so we exercise behavior through the public API
    /// without relying on re-reading env after first init.)
    static CACHE_ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn inputs_for(src: &str) -> CacheKeyInputs<'_> {
        CacheKeyInputs {
            desugared_source: src,
            word_size: 64,
            magic_set: None,
            functor_source: None,
            evaluator_shim_source: "shim",
            evaluator_protocol_source: "protocol",
            json_string_codec_source: "codec",
            functors_common_source: "fc",
            souffle_version: "souffle 2.4",
            souffle_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            cxx_version: "c++ 1.0",
            cxx_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            tenant_id: None,
        }
    }

    #[test]
    fn key_and_signature_cover_the_gxx_flags() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        // The g++ output flags are a single shared const folded into both
        // the cache key and the build signature, so editing them both
        // misses stale entries and eagerly clears them at startup.
        let flags = GXX_OUTPUT_FLAGS.join(" ");
        assert!(!flags.is_empty());
        assert!(
            build_signature()
                .unwrap()
                .contains(&format!("gxx_flags={flags}")),
            "build_signature must fold in the g++ flags so a flags edit re-keys the cache signature"
        );
    }

    #[test]
    fn cache_key_is_deterministic_and_input_sensitive() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let a = cache_key(&inputs_for("policy A"));
        let b = cache_key(&inputs_for("policy A"));
        let c = cache_key(&inputs_for("policy B"));
        assert_eq!(a, b, "same inputs ⇒ same key");
        assert_ne!(a, c, "different desugared source ⇒ different key");
    }

    #[test]
    fn compiler_executable_bytes_are_part_of_cache_identity() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        // These stand in for distinct compiler executables which both print
        // the same version. Only the non-output comment byte differs.
        let souffle_a_sha = sha256_hex("#!/bin/sh\necho same-version\n# compiler A\n");
        let souffle_b_sha = sha256_hex("#!/bin/sh\necho same-version\n# compiler B\n");
        let cxx_a_sha = sha256_hex("#!/bin/sh\necho same-version\n# compiler C\n");
        let cxx_b_sha = sha256_hex("#!/bin/sh\necho same-version\n# compiler D\n");

        let mut original = inputs_for("policy");
        original.souffle_sha256 = &souffle_a_sha;
        original.cxx_sha256 = &cxx_a_sha;
        let mut changed_souffle = inputs_for("policy");
        changed_souffle.souffle_sha256 = &souffle_b_sha;
        changed_souffle.cxx_sha256 = &cxx_a_sha;
        let mut changed_cxx = inputs_for("policy");
        changed_cxx.souffle_sha256 = &souffle_a_sha;
        changed_cxx.cxx_sha256 = &cxx_b_sha;

        assert_ne!(cache_key(&original), cache_key(&changed_souffle));
        assert_ne!(cache_key(&original), cache_key(&changed_cxx));
    }

    #[test]
    fn compiler_executable_bytes_are_part_of_startup_signature() {
        let original = build_signature_for_toolchain(
            "souffle same-version",
            &"a".repeat(64),
            "c++ same-version",
            &"c".repeat(64),
        );
        let changed = build_signature_for_toolchain(
            "souffle same-version",
            &"b".repeat(64),
            "c++ same-version",
            &"d".repeat(64),
        );

        assert_ne!(original, changed);
    }

    #[test]
    fn cache_key_changes_with_one_codec_byte() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let mut changed = inputs_for("policy");
        changed.json_string_codec_source = "codeD";
        assert_ne!(cache_key(&inputs_for("policy")), cache_key(&changed));
    }

    #[test]
    fn cache_key_changes_with_one_protocol_byte() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let mut changed = inputs_for("policy");
        changed.evaluator_protocol_source = "protocoM";
        assert_ne!(cache_key(&inputs_for("policy")), cache_key(&changed));
    }

    #[test]
    fn try_get_miss_when_disabled() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        // Cache root is read once via OnceLock. By the time these
        // tests run we're typically still uninitialized in the test
        // binary (no SASY_SOUFFLE_BUILD_CACHE_DIR set), so `enabled()`
        // is false and try_get returns false unconditionally.
        if !enabled() {
            let td = TempDir::new().unwrap();
            let dest = td.path().join("evaluator");
            assert!(!try_get("any-key", &dest));
            assert!(!dest.exists());
        }
    }

    #[test]
    fn enforce_budget_evicts_oldest_first() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let td = TempDir::new().unwrap();
        let root = td.path();
        // Three 100-byte entries with staggered mtimes.
        for (name, age) in [("oldest", 300), ("middle", 200), ("newest", 100)] {
            let p = root.join(name);
            std::fs::write(&p, vec![0u8; 100]).unwrap();
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(age);
            let f = std::fs::File::open(&p).unwrap();
            f.set_modified(when).unwrap();
        }
        // Budget = 250 bytes ⇒ must evict 1 entry (oldest).
        enforce_budget(root, 250);
        assert!(!root.join("oldest").exists(), "oldest should be evicted");
        assert!(root.join("middle").exists());
        assert!(root.join("newest").exists());
    }

    #[test]
    fn enforce_budget_no_op_under_budget() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let td = TempDir::new().unwrap();
        let root = td.path();
        std::fs::write(root.join("a"), vec![0u8; 100]).unwrap();
        std::fs::write(root.join("b"), vec![0u8; 100]).unwrap();
        enforce_budget(root, 1024);
        assert!(root.join("a").exists());
        assert!(root.join("b").exists());
    }

    #[test]
    fn prune_clears_all_entries_on_signature_change() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let td = TempDir::new().unwrap();
        let root = td.path();
        std::fs::write(root.join(SIGNATURE_FILE), "old-toolchain").unwrap();
        std::fs::write(root.join("a".repeat(64)), b"elf-a").unwrap();
        std::fs::write(root.join("b".repeat(64)), b"elf-b").unwrap();

        prune(root, "new-toolchain", None);

        // Every cached binary is unreachable under the new signature → gone.
        assert!(!root.join("a".repeat(64)).exists());
        assert!(!root.join("b".repeat(64)).exists());
        // Signature is rewritten to the current one.
        assert_eq!(
            std::fs::read_to_string(root.join(SIGNATURE_FILE)).unwrap(),
            "new-toolchain"
        );
    }

    #[test]
    fn prune_age_evicts_only_idle_entries_and_keeps_signature() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let td = TempDir::new().unwrap();
        let root = td.path();
        let sig = "matching-sig";
        std::fs::write(root.join(SIGNATURE_FILE), sig).unwrap();

        let stale = root.join("c".repeat(64));
        let fresh = root.join("d".repeat(64));
        std::fs::write(&stale, b"old").unwrap();
        std::fs::write(&fresh, b"new").unwrap();
        // Age the stale entry well past the TTL.
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 24 * 60 * 60);
        std::fs::File::open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();

        prune(
            root,
            sig,
            Some(std::time::Duration::from_secs(24 * 60 * 60)),
        );

        assert!(!stale.exists(), "idle entry past TTL should be age-pruned");
        assert!(fresh.exists(), "recently-used entry should survive");
        // The signature marker must never be age-pruned (it has no mtime bump).
        assert!(root.join(SIGNATURE_FILE).exists());
    }

    #[test]
    fn prune_age_disabled_keeps_everything() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let td = TempDir::new().unwrap();
        let root = td.path();
        let sig = "sig";
        std::fs::write(root.join(SIGNATURE_FILE), sig).unwrap();
        let entry = root.join("e".repeat(64));
        std::fs::write(&entry, b"x").unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(999 * 24 * 60 * 60);
        std::fs::File::open(&entry)
            .unwrap()
            .set_modified(old)
            .unwrap();

        prune(root, sig, None); // age pruning disabled

        assert!(entry.exists(), "age=0 disables age pruning");
    }
}
