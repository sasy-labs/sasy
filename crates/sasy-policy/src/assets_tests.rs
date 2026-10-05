use std::os::unix::fs::PermissionsExt;

use tempfile::TempDir;

use super::*;

#[test]
fn the_embedded_set_carries_the_six_compile_chain_files() {
    let names: Vec<&str> = ASSETS.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        vec![
            "sugar.py",
            "evaluator_shim.cpp",
            "evaluator_protocol.h",
            "json_string_codec.h",
            "functors_common.cpp",
            "common_policy.dl",
        ]
    );
    for (name, bytes) in ASSETS {
        assert!(!bytes.is_empty(), "{name} embedded empty");
    }
}

#[test]
fn materializing_writes_every_asset_read_only_with_the_embedded_bytes() {
    let td = TempDir::new().unwrap();
    let dir = materialized_dir(td.path());
    materialize(&dir, ASSETS).unwrap();

    for (name, bytes) in ASSETS {
        let path = dir.join(name);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), *bytes);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o444, "{name} should be read-only");
    }
    let present: std::collections::BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let expected: std::collections::BTreeSet<String> =
        ASSETS.iter().map(|(name, _)| (*name).to_string()).collect();
    assert_eq!(
        present, expected,
        "the six assets and nothing else — no temp file may survive"
    );
}

#[test]
fn materializing_a_second_time_rewrites_nothing() {
    let td = TempDir::new().unwrap();
    let dir = materialized_dir(td.path());
    materialize(&dir, ASSETS).unwrap();

    let before: Vec<_> = ASSETS
        .iter()
        .map(|(name, _)| {
            std::fs::metadata(dir.join(name))
                .unwrap()
                .modified()
                .unwrap()
        })
        .collect();
    materialize(&dir, ASSETS).unwrap();
    let after: Vec<_> = ASSETS
        .iter()
        .map(|(name, _)| {
            std::fs::metadata(dir.join(name))
                .unwrap()
                .modified()
                .unwrap()
        })
        .collect();

    assert_eq!(before, after, "an unchanged asset must not be rewritten");
}

#[test]
fn materializing_repairs_a_tampered_asset() {
    let td = TempDir::new().unwrap();
    let dir = materialized_dir(td.path());
    materialize(&dir, ASSETS).unwrap();

    let victim = dir.join("common_policy.dl");
    std::fs::set_permissions(&victim, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::write(&victim, "// truncated by something else\n").unwrap();

    materialize(&dir, ASSETS).unwrap();
    let expected = ASSETS
        .iter()
        .find(|(name, _)| *name == "common_policy.dl")
        .unwrap()
        .1;
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), expected);
    let mode = std::fs::metadata(&victim).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o444);
}

#[test]
fn the_directory_is_keyed_by_the_asset_set_hash() {
    let modified: Vec<(&str, &str)> = ASSETS
        .iter()
        .map(|(name, bytes)| {
            if *name == "sugar.py" {
                (*name, "# a different sugar.py\n")
            } else {
                (*name, *bytes)
            }
        })
        .collect();

    assert_ne!(set_hash(ASSETS), set_hash(&modified));
    assert_eq!(set_hash(ASSETS), asset_set_hash());

    let td = TempDir::new().unwrap();
    let embedded = materialized_dir(td.path());
    let other = td.path().join("assets").join(set_hash(&modified));
    assert_ne!(embedded, other);

    materialize(&embedded, ASSETS).unwrap();
    materialize(&other, &modified).unwrap();
    assert_ne!(
        std::fs::read_to_string(embedded.join("sugar.py")).unwrap(),
        std::fs::read_to_string(other.join("sugar.py")).unwrap()
    );
}

#[test]
fn the_hash_separates_a_name_change_from_a_content_change() {
    // `name\0bytes\0` terminates every field, so moving bytes across the
    // boundary cannot produce the same digest.
    let a: Vec<(&str, &str)> = vec![("ab", "c")];
    let b: Vec<(&str, &str)> = vec![("a", "bc")];
    assert_ne!(set_hash(&a), set_hash(&b));
}

#[test]
fn materializing_leaves_the_directory_it_was_given_a_home_in_alone() {
    // The target's parent can be a directory this process does not own — the
    // operator's working directory, or /tmp — so nothing above `dir` is
    // created, chmodded, or otherwise touched.
    let td = TempDir::new().unwrap();
    let parent = td.path().join("operator-owned");
    std::fs::create_dir(&parent).unwrap();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();

    materialize(&parent.join("chain"), ASSETS).unwrap();

    let mode = std::fs::metadata(&parent).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o755,
        "the parent's mode must survive materialization"
    );
}

#[test]
fn the_cache_root_layout_keeps_its_own_assets_directory_private() {
    let td = TempDir::new().unwrap();
    let dir = materialize_under(td.path()).unwrap();

    assert_eq!(dir, materialized_dir(td.path()));
    let mode = std::fs::metadata(td.path().join("assets"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o700, "the cache's assets directory is ours alone");
    for (name, bytes) in ASSETS {
        assert_eq!(std::fs::read_to_string(dir.join(name)).unwrap(), *bytes);
    }
}

#[test]
fn two_writes_of_one_asset_use_temp_names_of_their_own() {
    // Sequential writes cannot show a collision — the first temp is renamed
    // away before the second exists — so what is asserted is the name itself:
    // two draws from the counter that names them differ.
    let dir = std::path::Path::new("/tmp/chain");
    let target = dir.join("sugar.py");

    let first = temp_path_for(&target, 0);
    let second = temp_path_for(&target, 1);
    assert_ne!(first, second, "two writes must not share a temp name");
    assert_eq!(
        first.parent(),
        Some(dir),
        "the temp is written beside its target"
    );
    for name in [&first, &second] {
        assert!(
            name.file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".tmp-"),
            "a temp must be recognizable as one: {}",
            name.display()
        );
    }
}

#[test]
fn two_threads_writing_one_asset_leave_one_whole_payload_and_no_temp() {
    // The collision the temp name exists to prevent: two writers, one target,
    // at the same time. Whoever renames last wins, but neither may be seen
    // half-written, and neither temp may survive.
    let td = TempDir::new().unwrap();
    let dir = td.path().join("chain");
    std::fs::create_dir(&dir).unwrap();
    let target = dir.join("sugar.py");

    let first = vec![b'a'; 64 * 1024];
    let second = vec![b'b'; 64 * 1024];
    std::thread::scope(|scope| {
        for payload in [&first, &second] {
            let target = target.clone();
            scope.spawn(move || write_atomic(&target, payload).expect("a write must succeed"));
        }
    });

    let landed = std::fs::read(&target).unwrap();
    assert!(
        landed == first || landed == second,
        "the file must hold exactly one payload intact, not a mix ({} bytes)",
        landed.len()
    );
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp files left behind: {leftovers:?}"
    );
}

#[test]
fn a_temp_file_left_behind_by_a_killed_process_does_not_wedge_a_later_write() {
    // The read-only mode goes on the destination after the rename, never on
    // the temp: a temp that outlived its process is a plain writable file a
    // later run can overwrite or remove.
    let td = TempDir::new().unwrap();
    let dir = td.path().join("chain");
    std::fs::create_dir(&dir).unwrap();
    let target = dir.join("sugar.py");
    let stale = dir.join("sugar.py.tmp-4242-0");
    std::fs::write(&stale, "half a file").unwrap();

    write_atomic(&target, b"complete").unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), b"complete");
    let mode = std::fs::metadata(&stale).unwrap().permissions().mode() & 0o200;
    assert_eq!(mode, 0o200, "a leftover temp must stay writable");
    std::fs::write(&stale, "and overwritable").unwrap();
}

/// Set `path`'s modification time that far into the past.
fn age(path: &Path, age: Duration) {
    let times = std::fs::FileTimes::new().set_modified(std::time::SystemTime::now() - age);
    std::fs::File::open(path).unwrap().set_times(times).unwrap();
}

/// How old a set directory looks now.
fn idle_for(path: &Path) -> Duration {
    std::fs::metadata(path)
        .unwrap()
        .modified()
        .unwrap()
        .elapsed()
        .unwrap_or_default()
}

#[test]
fn materializing_under_a_root_sweeps_the_abandoned_sets_and_keeps_the_live_ones() {
    let td = TempDir::new().unwrap();
    let parent = td.path().join("assets");
    std::fs::create_dir_all(&parent).unwrap();

    // A set no binary has touched for longer than the grace period, one a
    // second binary is still using, and two things that only look like a set.
    let abandoned = parent.join("0123456789abcdef");
    std::fs::create_dir(&abandoned).unwrap();
    std::fs::write(abandoned.join("sugar.py"), "old").unwrap();
    age(&abandoned, FOREIGN_SET_GRACE + Duration::from_secs(60));
    let in_use = parent.join("fedcba9876543210");
    std::fs::create_dir(&in_use).unwrap();
    std::fs::write(in_use.join("sugar.py"), "another binary's").unwrap();
    age(&in_use, FOREIGN_SET_GRACE - Duration::from_secs(600));
    let not_a_hash = parent.join("operator-notes");
    std::fs::create_dir(&not_a_hash).unwrap();
    let too_short = parent.join("0123456789abcde");
    std::fs::create_dir(&too_short).unwrap();

    let dir = materialize_under(td.path()).unwrap();

    assert!(dir.join("sugar.py").is_file());
    assert!(
        !abandoned.exists(),
        "a set idle past the grace must be swept"
    );
    assert!(
        in_use.is_dir(),
        "a set another binary is still touching must survive"
    );
    assert!(not_a_hash.is_dir(), "a name that is not a set hash is left");
    assert!(too_short.is_dir(), "a name that is not a set hash is left");
    assert!(dir.exists(), "the current set survives its own sweep");
}

#[test]
fn every_materialize_says_the_current_set_is_still_in_use() {
    // The other half of the grace period: a set this binary keeps compiling
    // against must never age into the sweep, however long the process runs.
    let td = TempDir::new().unwrap();
    let dir = materialize_under(td.path()).unwrap();
    age(&dir, FOREIGN_SET_GRACE * 2);
    assert!(idle_for(&dir) > FOREIGN_SET_GRACE);

    let again = materialize_under(td.path()).unwrap();

    assert_eq!(again, dir);
    assert!(
        idle_for(&dir) < Duration::from_secs(60),
        "materializing must touch the set it uses"
    );
    assert!(dir.join("sugar.py").is_file());
}

#[test]
fn a_foreign_set_goes_only_once_it_is_past_the_grace() {
    // The sweep dates a set by its directory mtime, and the grace is what
    // decides. Anything it cannot date — an entry that vanished under it, a
    // filesystem that will not say — is treated as in use: deleting another
    // binary's assets costs more than a stale directory does.
    let td = TempDir::new().unwrap();
    let parent = td.path().join("assets");
    std::fs::create_dir_all(&parent).unwrap();
    let foreign = parent.join("0123456789abcdef");
    std::fs::create_dir(&foreign).unwrap();
    age(&foreign, FOREIGN_SET_GRACE + Duration::from_secs(60));

    // A grace period nothing can have reached keeps every foreign set.
    prune_other_sets(&parent, asset_set_hash(), Duration::MAX);
    assert!(foreign.is_dir(), "nothing is old enough for this grace");

    prune_other_sets(&parent, asset_set_hash(), FOREIGN_SET_GRACE);
    assert!(!foreign.exists(), "past the grace, a foreign set goes");
}
