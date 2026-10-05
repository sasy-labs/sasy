//! Repair for duplicated Soufflé header installs.
//!
//! Some Soufflé packagings ship the public headers twice: `<root>/souffle/*.h`
//! AND a byte-identical nested copy at `<root>/souffle/souffle/*.h` as
//! physically distinct files. The Homebrew `souffle` 2.5 bottle on macOS does
//! this. Soufflé's headers use `#pragma once` (deduped by file identity), and
//! `SouffleInterface.h` reaches its siblings via `#include "souffle/..."`,
//! which — evaluated from inside `<root>/souffle/` — resolves to the nested
//! copy. The top-level and nested copies are then both parsed, so every Soufflé
//! type (`isRamType`, `ramBitCast`, …) is defined twice and g++ fails.
//!
//! [`normalize`] detects that layout and returns an include root pointing at a
//! de-duplicated copy of the header tree (the nested `souffle/souffle/`
//! dropped), so every caller that DISCOVERS its assets works without users
//! hand-repairing their toolchain. A clean install is returned unchanged. The
//! assured runtime path never comes here: it copies the headers its manifest
//! declares into a private root, which cannot hold the nested copy.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};
use tracing::info;

/// Marks a fully-materialized sanitized include tree (vs. a half-copied one).
const READY_MARKER: &str = ".sasy-include-ready";

/// Return an include root safe to pass to `-I`, repairing a duplicated Soufflé
/// header install if one is detected. A normal install is returned unchanged
/// (no copying — just one existence check).
pub(crate) fn normalize(raw: PathBuf) -> std::io::Result<PathBuf> {
    // The pathology is a nested `<root>/souffle/souffle/` directory; without it
    // there is nothing to repair (the common case, and every clean Linux box).
    if !raw.join("souffle").join("souffle").is_dir() {
        return Ok(raw);
    }
    // Fail closed: a doubled install we can't sanitize won't compile anyway (it
    // would spew redefinition errors), so surface an actionable error rather
    // than silently handing back the broken tree.
    let sanitized = build_sanitized(&raw, &cache_root())?;
    info!(
        raw = %raw.display(),
        sanitized = %sanitized.display(),
        "Soufflé install ships duplicated headers; using a de-duplicated include tree"
    );
    Ok(sanitized)
}

/// Materialize `<raw>/souffle/` (minus the redundant nested `souffle/souffle/`)
/// under `cache_root`, returning the include root (the dir containing the
/// de-duplicated `souffle/`). Idempotent and concurrency-safe: a completed tree
/// is reused; concurrent builders stage into private dirs and the first to
/// finish wins.
fn build_sanitized(raw: &Path, cache_root: &Path) -> std::io::Result<PathBuf> {
    let fp = fingerprint(raw);
    let base = cache_root.join(&fp);
    if base.join(READY_MARKER).is_file() && base.join("souffle").is_dir() {
        return Ok(base);
    }
    std::fs::create_dir_all(cache_root)?;
    restrict_perms(cache_root);

    // Stage into a private dir, then atomically promote — so a concurrent
    // compile never observes a half-copied tree. The suffix is unique across
    // processes (pid) AND threads/tasks in this process (atomic counter), so
    // concurrent first-time builds can't clobber each other's staging dir.
    static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);
    let staging = cache_root.join(format!(
        "{fp}.staging.{}.{}",
        std::process::id(),
        STAGING_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&staging);
    copy_tree(
        &raw.join("souffle"),
        &staging.join("souffle"),
        Some("souffle"),
    )?;
    std::fs::write(staging.join(READY_MARKER), b"")?;

    match std::fs::rename(&staging, &base) {
        Ok(()) => Ok(base),
        // Lost the race (another process already promoted an identical tree) or
        // a stale `base` exists; prefer the finished tree when present.
        Err(_) if base.join(READY_MARKER).is_file() => {
            let _ = std::fs::remove_dir_all(&staging);
            Ok(base)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            Err(e)
        }
    }
}

/// Per-user, persistent location for sanitized include trees, kept OUTSIDE any
/// writable compile build dir so hostile functor C++ cannot rewrite headers it
/// later `#include`s. Honors `XDG_CACHE_HOME`, else `$HOME/.cache`, else temp.
fn cache_root() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("sasy").join("souffle-include")
}

/// Short, stable id for a raw include root + normalization-logic version, so
/// distinct roots (or a future logic change) never collide on a shared cache.
fn fingerprint(raw: &Path) -> String {
    let mut h = Sha256::new();
    h.update(raw.to_string_lossy().as_bytes());
    // Canonicalize the souffle dir so a Homebrew symlink retarget (e.g. a
    // `brew upgrade` from Cellar/2.5 to Cellar/2.6) re-keys the sanitized tree
    // even though `raw` is unchanged — otherwise we'd serve stale 2.5 headers
    // against a 2.6 toolchain.
    if let Ok(canon) = std::fs::canonicalize(raw.join("souffle")) {
        h.update(b"|");
        h.update(canon.to_string_lossy().as_bytes());
    }
    h.update(b"|norm-v2");
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Recursively copy `src` → `dst`, skipping a top-level child named `exclude`
/// (the redundant nested header dir) when one is named.
fn copy_tree(src: &Path, dst: &Path, exclude: Option<&str>) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if exclude.is_some_and(|name| entry.file_name().as_os_str() == OsStr::new(name)) {
            continue;
        }
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if entry.file_type()?.is_dir() {
            copy_tree(&from, &to, None)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn restrict_perms(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
}
#[cfg(not(unix))]
fn restrict_perms(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_header(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"#pragma once\n").unwrap();
    }

    #[test]
    fn clean_install_is_returned_unchanged() {
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path().to_path_buf();
        write_header(&root.join("souffle").join("RamTypes.h"));
        write_header(&root.join("souffle").join("io").join("IOSystem.h"));
        assert_eq!(
            normalize(root.clone()).unwrap(),
            root,
            "no nested dup ⇒ unchanged"
        );
    }

    #[test]
    fn doubled_install_is_deduplicated() {
        let td = tempfile::TempDir::new().unwrap();
        let root = td.path().join("include");
        // Top-level headers + a subdir + the redundant nested copy.
        write_header(&root.join("souffle").join("RamTypes.h"));
        write_header(&root.join("souffle").join("io").join("IOSystem.h"));
        write_header(&root.join("souffle").join("souffle").join("RamTypes.h"));
        write_header(
            &root
                .join("souffle")
                .join("souffle")
                .join("io")
                .join("IOSystem.h"),
        );

        let cache = td.path().join("cache");
        let out = build_sanitized(&root, &cache).unwrap();

        assert_ne!(out, root, "doubled install ⇒ a sanitized root is returned");
        assert!(out.join("souffle").join("RamTypes.h").is_file());
        assert!(out.join("souffle").join("io").join("IOSystem.h").is_file());
        assert!(
            !out.join("souffle").join("souffle").exists(),
            "the redundant nested souffle/souffle must be dropped"
        );
        // Idempotent: a second call reuses the same ready tree.
        assert_eq!(out, build_sanitized(&root, &cache).unwrap());
    }
}
