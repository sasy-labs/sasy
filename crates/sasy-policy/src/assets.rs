//! Embedded Soufflé compile-chain assets.
//!
//! The compiler materializes these files in its build cache, so a standalone
//! binary uses the same assets regardless of its working directory. Cargo tracks
//! changes through `include_str!`. Operators can override the set with
//! `SASY_SOUFFLE_ASSETS`; see [`crate::compiler::SouffleAssets::discover`].
//!
//! The interpreted backend's `functors.cpp` and `interpreted_shim.cpp` are not
//! embedded. Build its separate executable and shared library with
//! `bash souffle/build-test-runtime.sh`.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::hash::hex;

/// The embedded compile-chain assets, `(file name, contents)`.
///
/// Paths are relative to this file: `crates/sasy-policy/src` → `.`
/// → `souffle`.
pub const ASSETS: &[(&str, &str)] = &[
    ("sugar.py", include_str!("../../../souffle/sugar.py")),
    (
        "evaluator_shim.cpp",
        include_str!("../../../souffle/evaluator_shim.cpp"),
    ),
    (
        "evaluator_protocol.h",
        include_str!("../../../souffle/evaluator_protocol.h"),
    ),
    (
        "json_string_codec.h",
        include_str!("../../../souffle/json_string_codec.h"),
    ),
    (
        "functors_common.cpp",
        include_str!("../../../souffle/functors_common.cpp"),
    ),
    (
        "common_policy.dl",
        include_str!("../../../souffle/common_policy.dl"),
    ),
];

/// Digest of an asset set: the first 16 hex chars of the SHA-256 over the
/// `name\0bytes\0` sequence, in the order given.
///
/// Every field is terminated, so no two different sets can encode to the same
/// byte string. Sixteen chars is a directory name, not a security boundary —
/// the bytes are re-verified after they are written.
pub fn set_hash(assets: &[(&str, &str)]) -> String {
    let mut h = Sha256::new();
    for (name, bytes) in assets {
        h.update(name.as_bytes());
        h.update([0u8]);
        h.update(bytes.as_bytes());
        h.update([0u8]);
    }
    hex(&h.finalize())[..16].to_string()
}

/// [`set_hash`] of the embedded set, computed once.
///
/// It keys the on-disk directory by content rather than by binary version, so
/// two builds carrying identical assets share one directory and a build that
/// changed one of them gets its own.
pub fn asset_set_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| set_hash(ASSETS))
}

/// Write `assets` into `dir`, creating it if needed, and verify the result.
///
/// Each file is written to a temp name of its own and renamed over `<name>`, so
/// a reader either sees the previous complete file or the new one — never a
/// half-written one. That is also why no lock is needed when two processes
/// materialize the same set at once: they write the same bytes, each rename is
/// atomic, and whichever lands last leaves the same content behind.
///
/// A file already present with the right bytes is left alone (materialization
/// is idempotent); one that differs is rewritten, which repairs a truncated
/// earlier write or a hand-edit. Files end up read-only (0o444) and the
/// directory 0o755.
///
/// Only `dir` and the files in it are touched. `dir`'s parent may be a
/// directory this process does not own — the operator's working directory, or
/// `/tmp` — so its mode is left alone; the cache-root layout hardens its own
/// private parent in [`materialize_under`].
pub fn materialize(dir: &Path, assets: &[(&str, &str)]) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    set_mode(dir, 0o755)?;

    for (name, bytes) in assets {
        let path = dir.join(name);
        let current_is_correct =
            matches!(std::fs::read(&path), Ok(have) if have == bytes.as_bytes());
        if !current_is_correct {
            write_atomic(&path, bytes.as_bytes())?;
        }
    }

    // Read every file back: a rename that reported success on a full or failing
    // disk still has to produce the bytes the compile is about to be handed.
    for (name, bytes) in assets {
        let path = dir.join(name);
        let have = std::fs::read(&path)?;
        if have != bytes.as_bytes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} does not match the embedded asset", path.display()),
            ));
        }
    }
    Ok(())
}

/// Write `bytes` to `path` through a private temp file and a rename.
///
/// The temp name carries the process id *and* a counter no other call in this
/// process repeats: two callers materializing at once — the compile path and
/// an `install-assets` in the same binary — would otherwise write the same
/// temp file over each other. It is left writable until it is in place; the
/// read-only mode is set on the destination after the rename, so a temp left
/// behind by a killed process is one a later run can simply overwrite rather
/// than an undeletable 0o444 file wedging every compile after it.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = temp_path_for(
        path,
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    );
    std::fs::write(&tmp, bytes)?;
    // The destination may be 0o444 from an earlier run; rename replaces it
    // regardless, since the permission that matters is the directory's.
    match std::fs::rename(&tmp, path) {
        Ok(()) => set_mode(path, 0o444),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

/// The temp file one write of `target` uses: the process id, and `seq` — a
/// number no other write in this process repeats.
///
/// Split out so a test can ask two calls for their names and compare them; the
/// collision it guards against is between concurrent writers, which no
/// sequence of writes can show.
pub(crate) fn temp_path_for(target: &Path, seq: u64) -> std::path::PathBuf {
    target.with_file_name(format!(
        "{}.tmp-{}-{seq}",
        target.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ))
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

/// The directory the embedded set is materialized into: `<root>/assets/<hash>`.
///
/// `root` is the build cache's root — the same directory the compiled
/// evaluators are cached in, which the server points at `<data-dir>/`. The
/// cache's own eviction only ever removes files, so an `assets/` subdirectory
/// under it is never swept; [`materialize_under`] sweeps the sets it left
/// behind itself.
pub fn materialized_dir(root: &Path) -> PathBuf {
    root.join("assets").join(asset_set_hash())
}

/// Materialize the embedded set into [`materialized_dir`] of `root` and return
/// the directory.
///
/// This is the cache-root layout, where `<root>/assets` is a directory this
/// code created for itself and nothing else writes, so it is made private
/// (0o700). [`materialize`] on its own hardens nothing above the directory it
/// is given, which is what an operator-named target needs.
pub fn materialize_under(root: &Path) -> io::Result<PathBuf> {
    let parent = root.join("assets");
    std::fs::create_dir_all(&parent)?;
    set_mode(&parent, 0o700)?;
    let dir = parent.join(asset_set_hash());
    materialize(&dir, ASSETS)?;
    // Say this set is in use before sweeping the others: the sweep dates
    // foreign sets by their directory mtime, and this is what keeps a live set
    // of another binary's from aging out under it.
    touch(&dir);
    prune_other_sets(&parent, asset_set_hash(), FOREIGN_SET_GRACE);
    Ok(dir)
}

/// How long a set directory that is not this binary's must have sat untouched
/// before it is swept.
///
/// One data directory can be shared by two binaries of different versions —
/// a rolling upgrade on a persistent volume, the layout `souffle_cache` is
/// built around. Both materialize on every compile, so a set in use is never
/// this old; a set left by a binary that is gone stops being touched and is
/// reclaimed a day later.
const FOREIGN_SET_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

/// Mark `dir` as used now, by setting its modification time.
///
/// Best effort: on a filesystem or a permission that refuses, the only cost is
/// that the sweep may take this set for an abandoned one after the grace
/// period — and the next compile writes it back.
fn touch(dir: &Path) {
    let times = std::fs::FileTimes::new().set_modified(std::time::SystemTime::now());
    if let Ok(handle) = std::fs::File::open(dir) {
        let _ = handle.set_times(times);
    }
}

/// Remove the set directories under `parent` that are not `keep` and have not
/// been used for `grace`.
///
/// An upgraded binary carries a different set, materializes it beside the old
/// one, and never looks at the old one again. Nothing else reclaims it: the
/// build cache's byte budget counts the compiled evaluators, not this
/// directory. But "not mine" is not "nobody's": another binary sharing this
/// data directory keeps its own set, and sweeping on ownership alone would
/// have two versions delete each other's assets on every compile. So a foreign
/// set goes only once its directory has sat unmodified for `grace`, and every
/// materialize touches its own set ([`touch`]) to say it is still in use. A
/// set whose age cannot be read is kept.
///
/// Only entries whose name is exactly a set hash — sixteen lowercase hex
/// characters, what [`set_hash`] produces — are removed, so an operator's own
/// file or directory that happens to sit here is left alone. A removal that
/// fails is not an error: the assets that matter are already in place, and
/// another process may be materializing the very set being swept.
fn prune_other_sets(parent: &Path, keep: &str, grace: Duration) {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let is_set_hash = name.len() == 16
            && name
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !is_set_hash || name == keep {
            continue;
        }
        let idle = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .map(|modified| modified.elapsed().unwrap_or_default());
        match idle {
            Ok(idle) if idle >= grace => {}
            // Younger than the grace period, or a clock or filesystem that
            // will not say: another binary may be running on it.
            _ => continue,
        }
        let path = entry.path();
        let _ = match entry.file_type() {
            Ok(kind) if kind.is_dir() => std::fs::remove_dir_all(&path),
            Ok(_) => std::fs::remove_file(&path),
            Err(_) => continue,
        };
    }
}

#[cfg(test)]
#[path = "assets_tests.rs"]
mod tests;
