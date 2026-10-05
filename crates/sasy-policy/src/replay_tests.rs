use super::*;
use std::sync::Arc;

use parking_lot::Mutex;
use sasy_common::SessionScope;
use sasy_graph::PersistedPolicy;

use crate::engine::{InstallMode, SyncStatus};
use crate::evaluator::EvaluatorError;
use crate::runtime_conformance::{
    admit_runtime_inputs, admit_runtime_suite, assured_runtime_environment, execute_runtime_suite,
    execute_souffle_report, materialize_include_root, normalize_runtime_result,
    runtime_input_sha256, runtime_sha256, verify_generated_runtime_files, verify_runtime_inputs,
    AdmittedRuntimeInputs,
};
use crate::session_evaluator::EvaluatorFactory;

struct ScenarioRecordingEvaluator {
    events: Arc<Mutex<Vec<(&'static str, serde_json::Value)>>>,
    response: crate::evaluator::types::EvalAuthResponse,
}

#[tonic::async_trait]
impl crate::evaluator::Evaluator for ScenarioRecordingEvaluator {
    async fn update(&self, updates: Vec<crate::engine::GraphUpdate>) -> Result<(), EvaluatorError> {
        self.events
            .lock()
            .push(("update", serde_json::to_value(updates).unwrap()));
        Ok(())
    }

    async fn query(
        &self,
        request: crate::evaluator::types::EvalAuthRequest,
    ) -> Result<crate::evaluator::types::EvalAuthResponse, EvaluatorError> {
        self.events
            .lock()
            .push(("query", serde_json::to_value(request).unwrap()));
        Ok(self.response.clone())
    }

    async fn reset(&self) -> Result<(), EvaluatorError> {
        Ok(())
    }

    async fn set_metadata(
        &self,
        facts: Vec<crate::evaluator::types::PolicyMetadataFact>,
    ) -> Result<(), EvaluatorError> {
        self.events
            .lock()
            .push(("metadata", serde_json::to_value(facts).unwrap()));
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "scenario-recording"
    }
}

fn admitted_suite_fixture() -> serde_json::Value {
    let source = "IsAuthorized(idx) :- Actions(idx, _).\n";
    let result = |transform_ids: Vec<&str>| {
        serde_json::json!({
            "allow_passthrough": false,
            "authorized": true,
            "denial_reasons": [],
            "deny_if_unauthorized": true,
            "index": 0,
            "is_allowlisted": true,
            "is_authenticated": false,
            "is_denylisted": false,
            "requires_approval": false,
            "transform_ids": transform_ids
        })
    };
    let query = |action| {
        serde_json::json!({"Query": {
            "action_metadata": [], "actions": [action], "current_node_ids": [],
            "entity": null, "principal": null, "roles": [], "session_id": null,
            "tenant_id": null
        }})
    };
    serde_json::json!({
        "identity": {
            "composedDatalogSha256": "a".repeat(64),
            "familyId": "arbitrary-family",
            "publicFamilySha256": "b".repeat(64),
            "runtimeBaseDatalogSha256": "d".repeat(64),
            "runtimeMappingSha256": "c".repeat(64)
        },
        "probePolicy": {
            "sha256": runtime_sha256(source.as_bytes()),
            "source": source
        },
        "scenarios": [
            {
                "expectedResults": [result(vec!["alpha", "beta"])],
                "id": "beta",
                "queryRequest": query(serde_json::json!({"HttpRequest": {
                        "body": "catalog-body-雪",
                        "headers": [["x-catalog", "archive\\nvalue"]],
                        "url": "https://catalog.invalid/items?slot=7"
                    }})),
                "setupRequests": [
                    {"Update": {"updates": [{"NodeCreated": {
                        "agent": "catalog-agent", "content": "archive-message-😀",
                        "derived_from": null, "entity": "catalog-reader",
                        "id": "archive-1", "principal": "catalog-principal",
                        "role": "record", "session_id": "catalog-session", "tools": []
                    }}]}},
                    {"SetMetadata": {"facts": [{
                        "a": "shelf-7", "b": "retained", "rel": "catalog-entry"
                    }]}}
                ]
            },
            {
                "expectedResults": [result(Vec::new())],
                "id": "alpha",
                "queryRequest": query(serde_json::json!({"SendMessage": {
                        "agent": "catalog-agent", "agent_role": "indexer",
                        "content": "second-catalog-message", "entity": null,
                        "tool_calls": [["catalog-tool", "{\"slot\":7}"]]
                    }})),
                "setupRequests": []
            }
        ],
        "requiredSouffleWordSize": null,
        "schemaVersion": 1
    })
}

#[tokio::test]
async fn runtime_conformance_executor_executes_arbitrary_admitted_scenarios() {
    use std::sync::atomic::Ordering;

    let (bytes, digest) = canonical_document(&admitted_suite_fixture());
    let suite = admit_runtime_suite(&bytes, &digest).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let spawn_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let responses = Arc::new(Mutex::new(std::collections::VecDeque::from_iter(
        suite.scenarios.iter().map(|scenario| {
            let mut results = scenario.expected.clone();
            results.iter_mut().for_each(|result| {
                result.transform_ids.reverse();
                result.denial_reasons.reverse();
            });
            crate::evaluator::types::EvalAuthResponse { results }
        }),
    )));
    let factory_events = events.clone();
    let factory_spawn_count = spawn_count.clone();
    let factory: EvaluatorFactory = Arc::new(move || {
        let response = responses.lock().pop_front().unwrap();
        factory_spawn_count.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(ScenarioRecordingEvaluator {
            events: factory_events.clone(),
            response,
        }))
    });

    let observations = execute_runtime_suite("scenario-recording", &factory, &suite)
        .await
        .expect("execute scenario recording suite");
    let events = events.lock();
    let operations: Vec<_> = events.iter().map(|event| event.0).collect();
    assert_eq!(operations, ["update", "metadata", "query", "query"]);
    assert_eq!(spawn_count.load(Ordering::SeqCst), suite.scenarios.len());
    let update = &events[0].1;
    assert_eq!(update[0]["NodeCreated"]["content"], "archive-message-😀");
    assert_eq!(events[1].1[0]["rel"], "catalog-entry");
    for (event, scenario) in events[2..].iter().zip(&suite.scenarios) {
        assert_eq!(event.1, serde_json::to_value(&scenario.query).unwrap());
    }
    let expected = serde_json::to_value(&suite.scenarios[0].expected).unwrap();
    assert_eq!(
        serde_json::to_value(&observations[0].response.results).unwrap(),
        expected
    );
}

fn canonical_document(value: &serde_json::Value) -> (Vec<u8>, String) {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    let digest = runtime_sha256(&bytes);
    (bytes, digest)
}

/// Every single-file role the command's closed runtime-input vocabulary
/// contains, in the order admitted records are sorted into. Written out here on
/// purpose: if the production list changes, these tests must be revisited rather
/// than follow along silently. The Soufflé headers are declared separately.
/// The production bound on how many headers one manifest may declare, written
/// out for the same reason: a change to it must be noticed here.
const MAX_FIXTURE_INCLUDE_FILES: usize = 4096;

const FIXTURE_ROLES: [&str; 8] = [
    "commonPolicy",
    "evaluatorProtocol",
    "evaluatorShim",
    "functorsCommon",
    "interpretedAdapter",
    "interpretedFunctors",
    "jsonStringCodec",
    "sugarPy",
];

/// Production discovery takes both C++ headers from the evaluator shim's own
/// directory, so the fixture spells those three roles the way the compiler
/// names them. Every other role is named after itself.
fn fixture_file_name(role: &str) -> &str {
    match role {
        "evaluatorShim" => "evaluator_shim.cpp",
        "evaluatorProtocol" => "evaluator_protocol.h",
        "jsonStringCodec" => "json_string_codec.h",
        other => other,
    }
}

fn input_entry(role: &str, path: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "path": path.to_str().unwrap(),
        "role": role,
        "sha256": runtime_input_sha256(role, path).expect("digest fixture runtime input"),
    })
}

/// Materialize one real asset per closed role — a plain file for each, plus a
/// nested `souffle/` header tree for the include input — and describe them the
/// way a producer would.
fn runtime_input_fixture() -> (
    tempfile::TempDir,
    Vec<serde_json::Value>,
    std::path::PathBuf,
) {
    let fixture = tempfile::tempdir().expect("create runtime input fixture");
    // A declared path has to be the resolved spelling, and a temporary
    // directory usually sits under a symlinked prefix (on macOS `/var` is a
    // link to `/private/var`), so the fixture describes the resolved root.
    let root = std::fs::canonicalize(fixture.path()).expect("resolve the fixture root");
    let headers = root.join("include").join("souffle");
    std::fs::create_dir_all(headers.join("io")).expect("create fixture include tree");
    for (name, bytes) in [
        ("SouffleInterface.h", &b"fixture header\n"[..]),
        ("io/IOSystem.h", &b"fixture nested header\n"[..]),
    ] {
        std::fs::write(headers.join(name), bytes).expect("write fixture header");
    }
    let entries = FIXTURE_ROLES
        .iter()
        .map(|role| {
            let path = root.join(fixture_file_name(role));
            std::fs::write(&path, format!("{role} fixture bytes\n")).expect("write fixture input");
            input_entry(role, &path)
        })
        .collect();
    let include_root = headers.parent().expect("the include root").to_path_buf();
    (fixture, entries, include_root)
}

/// The declared path of one fixture role.
fn fixture_path(entries: &[serde_json::Value], role: &str) -> std::path::PathBuf {
    let entry = entries
        .iter()
        .find(|entry| entry["role"] == role)
        .unwrap_or_else(|| panic!("the fixture declares {role}"));
    std::path::PathBuf::from(entry["path"].as_str().expect("a declared path"))
}

/// Describe the header tree the way a producer would: one relative path per
/// regular file, each mapped to its digest. Directories carry no record, and a
/// link carries none either — the effective tree the compiler is given holds
/// only files, so only files are declared.
fn include_input(include_root: &std::path::Path) -> serde_json::Value {
    let mut files = serde_json::Map::new();
    let mut pending = vec![include_root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).expect("read the fixture include tree") {
            let path = entry.expect("a fixture include entry").path();
            let name = path
                .strip_prefix(include_root)
                .expect("a path under the include root")
                .to_str()
                .expect("a UTF-8 header name")
                .to_string();
            let kind = std::fs::symlink_metadata(&path).expect("inspect a fixture header");
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                let sha256 = runtime_input_sha256(&name, &path).expect("digest a fixture header");
                files.insert(name, serde_json::json!(sha256));
            }
        }
    }
    serde_json::json!({
        "files": files,
        "sourceRoot": include_root.to_str().expect("a UTF-8 include root"),
    })
}

fn manifest_document(
    entries: &[serde_json::Value],
    include_root: &std::path::Path,
) -> serde_json::Value {
    serde_json::json!({
        "inputs": entries,
        "schemaVersion": 2,
        "souffleInclude": include_input(include_root),
    })
}

fn admit_fixture_manifest(
    entries: &[serde_json::Value],
    include_root: &std::path::Path,
) -> Result<AdmittedRuntimeInputs, String> {
    let (bytes, digest) = canonical_document(&manifest_document(entries, include_root));
    admit_runtime_inputs(&bytes, &digest)
}

#[test]
fn runtime_input_manifest_admits_the_closed_role_set() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let (bytes, digest) = canonical_document(&manifest_document(&entries, &include_root));
    let admitted = admit_runtime_inputs(&bytes, &digest).expect("admit the fixture manifest");

    assert_eq!(admitted.sha256, digest);
    let roles: Vec<&str> = admitted
        .records
        .iter()
        .map(|record| record.role.as_str())
        .collect();
    assert_eq!(roles, FIXTURE_ROLES);
    for (record, entry) in admitted.records.iter().zip(&entries) {
        assert_eq!(record.path, entry["path"].as_str().unwrap());
        assert_eq!(record.sha256, entry["sha256"].as_str().unwrap());
    }
    // The headers are admitted with them, one relative path at a time.
    let declared: Vec<&str> = admitted
        .souffle_include
        .files
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        declared,
        ["souffle/SouffleInterface.h", "souffle/io/IOSystem.h"]
    );
    assert_eq!(
        admitted.souffle_include.source_root,
        include_root.to_str().unwrap()
    );
}

#[test]
fn runtime_input_manifest_rejects_missing_extra_and_duplicate_roles() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let admit = |entries: &[serde_json::Value]| admit_fixture_manifest(entries, &include_root);

    let mut missing = entries.clone();
    missing.remove(0);
    assert!(admit(&missing)
        .unwrap_err()
        .contains("does not declare every role"));

    let mut extra = entries.clone();
    let mut undeclared = entries[0].clone();
    undeclared["role"] = serde_json::json!("souffleProfiler");
    extra.push(undeclared);
    assert!(admit(&extra)
        .unwrap_err()
        .contains("is not a declared role"));

    let mut duplicate = entries.clone();
    duplicate.push(entries[0].clone());
    assert!(admit(&duplicate).unwrap_err().contains("is declared twice"));

    let mut unknown_field = entries;
    unknown_field[0]["note"] = serde_json::json!("extra");
    assert!(admit(&unknown_field)
        .unwrap_err()
        .contains("input manifest schema"));
}

#[test]
fn runtime_input_manifest_rejects_changed_digests_and_substituted_paths() {
    let (fixture, entries, include_root) = runtime_input_fixture();
    let admit = |entries: &[serde_json::Value]| admit_fixture_manifest(entries, &include_root);

    // The manifest itself must be the one the caller pinned.
    let (bytes, _) = canonical_document(&manifest_document(&entries, &include_root));
    assert!(admit_runtime_inputs(&bytes, &"0".repeat(64))
        .unwrap_err()
        .contains("input manifest digest mismatch"));

    // A declared file digest that does not match the bytes on disk.
    let mut wrong_digest = entries.clone();
    wrong_digest[0]["sha256"] = serde_json::json!("f".repeat(64));
    assert!(admit(&wrong_digest)
        .unwrap_err()
        .contains("runtime input commonPolicy digest mismatch"));

    // A substituted path: a real, readable file that is not the pinned one.
    let mut substituted = entries.clone();
    substituted[0]["path"] = substituted[2]["path"].clone();
    assert!(admit(&substituted)
        .unwrap_err()
        .contains("runtime input commonPolicy digest mismatch"));

    // A path that could name two different assets.
    let mut traversing = entries.clone();
    traversing[0]["path"] = serde_json::json!("souffle/../souffle/common_policy.dl");
    assert!(admit(&traversing)
        .unwrap_err()
        .contains("path is not canonical"));

    // A future schema fails closed instead of being read with today's rules.
    let mut future = manifest_document(&entries, &include_root);
    future["schemaVersion"] = serde_json::json!(3);
    let (bytes, digest) = canonical_document(&future);
    assert!(admit_runtime_inputs(&bytes, &digest)
        .unwrap_err()
        .contains("schema version is unsupported"));

    // An asset mutated after admission is caught by the boundary re-check.
    let admitted = admit(&entries).expect("admit the fixture manifest");
    let effective_root = fixture.path().join("effective");
    materialize_include_root(&admitted.souffle_include, &effective_root)
        .expect("materialize the declared headers");
    std::fs::write(fixture.path().join("evaluator_shim.cpp"), b"swapped\n").expect("mutate input");
    assert!(verify_runtime_inputs(&admitted, &effective_root)
        .unwrap_err()
        .contains("runtime input evaluatorShim digest mismatch"));
}

/// The header set is a JSON object, so a repeated path is not a rule this module
/// enforces by hand: re-encoding the parsed document keeps one copy, which no
/// longer reproduces the pinned bytes.
#[test]
fn runtime_input_manifest_refuses_a_repeated_include_record() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let (bytes, _) = canonical_document(&manifest_document(&entries, &include_root));
    let declared = r#""souffle/SouffleInterface.h":"#;
    let repeated = String::from_utf8(bytes).expect("a UTF-8 manifest").replace(
        declared,
        &format!(r#"{declared}"{}",{declared}"#, "b".repeat(64)),
    );
    let bytes = repeated.into_bytes();
    let digest = runtime_sha256(&bytes);
    assert!(admit_runtime_inputs(&bytes, &digest)
        .unwrap_err()
        .contains("not canonical JSON"));
}

/// A relative header path must name exactly one file. Every spelling below is a
/// second way to write a path the fixture already declares, or a way to leave the
/// source root altogether, so what is refused is the spelling itself.
#[test]
fn runtime_input_manifest_refuses_unsafe_include_spellings() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let declared = manifest_document(&entries, &include_root);
    let digest = declared["souffleInclude"]["files"]["souffle/SouffleInterface.h"].clone();
    let with_files = |files: serde_json::Map<String, serde_json::Value>| {
        let mut document = declared.clone();
        document["souffleInclude"]["files"] = serde_json::Value::Object(files);
        let (bytes, digest) = canonical_document(&document);
        admit_runtime_inputs(&bytes, &digest).unwrap_err()
    };

    for spelling in [
        "../SouffleInterface.h",
        "/souffle/SouffleInterface.h",
        "./souffle/SouffleInterface.h",
        "souffle//SouffleInterface.h",
        "souffle/",
        "souffle/../souffle/SouffleInterface.h",
        "",
        "..",
    ] {
        let mut files = serde_json::Map::new();
        files.insert(spelling.to_string(), digest.clone());
        let refusal = with_files(files);
        assert!(
            refusal.contains(&format!("souffleInclude {spelling} path is not canonical")),
            "{spelling} must be refused, got: {refusal}"
        );
    }

    // A digest that is not a digest, and a set that is empty or unbounded.
    let mut malformed = serde_json::Map::new();
    malformed.insert(
        "souffle/SouffleInterface.h".to_string(),
        serde_json::json!("NOT-A-DIGEST"),
    );
    assert!(with_files(malformed).contains("digest is invalid"));
    assert!(with_files(serde_json::Map::new()).contains("no or too many files"));
    let crowded = (0..=MAX_FIXTURE_INCLUDE_FILES)
        .map(|index| (format!("souffle/header{index}.h"), digest.clone()))
        .collect();
    assert!(with_files(crowded).contains("no or too many files"));
}

/// A declared path must be self-identifying: one absolute spelling, resolved
/// the way the filesystem resolves it. Every spelling here names the same
/// readable file the fixture already declares, so nothing but the spelling
/// itself is what gets refused.
#[test]
fn runtime_input_manifest_requires_absolute_resolved_paths() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let declared = fixture_path(&entries, "commonPolicy");
    let directory = declared.parent().expect("the fixture root").to_owned();
    let directory = directory.to_str().expect("a UTF-8 fixture root");

    let leaf = declared
        .parent()
        .and_then(std::path::Path::file_name)
        .and_then(std::ffi::OsStr::to_str)
        .expect("the fixture root directory name");
    let mut spellings = vec![
        "commonPolicy".to_string(),            // relative to the caller's cwd
        format!("{directory}/./commonPolicy"), // a dot segment
        format!("{directory}//commonPolicy"),  // a repeated separator
        format!("{directory}/commonPolicy/"),  // a trailing separator
        format!("{directory}/../{leaf}/commonPolicy"), // a parent segment
        "/".to_string(),                       // a bare root
    ];
    // A symbolic link spelling resolves to the pinned bytes, so only comparing
    // against the resolved path catches it.
    #[cfg(unix)]
    {
        let link = format!("{directory}/commonPolicyLink");
        std::os::unix::fs::symlink(&declared, &link).expect("link the fixture input");
        spellings.push(link);
    }

    for spelling in spellings {
        let mut spelled = entries.clone();
        spelled[0]["path"] = serde_json::json!(spelling);
        let refusal = admit_fixture_manifest(&spelled, &include_root).unwrap_err();
        assert!(
            refusal.contains("runtime input commonPolicy path is not canonical"),
            "{spelling} must be refused, got: {refusal}"
        );
    }
}

/// The compiler is given a private copy of exactly the declared headers, so a
/// file the run did not authenticate cannot be on its search path — including a
/// sibling of the header directory, which is what a shared prefix such as
/// `/opt/homebrew/include` is full of, and which would shadow a standard header
/// the functor sources include by name.
#[test]
fn effective_include_root_holds_only_the_declared_headers() {
    let (fixture, entries, include_root) = runtime_input_fixture();
    let admitted =
        admit_fixture_manifest(&entries, &include_root).expect("admit the fixture manifest");
    // Both planted after the manifest was pinned, so neither is declared.
    std::fs::write(include_root.join("cassert"), b"#error shadowed\n").expect("plant a sibling");
    std::fs::write(
        include_root.join("souffle").join("Extra.h"),
        b"undeclared\n",
    )
    .expect("plant an undeclared header");

    let effective_root = fixture.path().join("effective");
    materialize_include_root(&admitted.souffle_include, &effective_root)
        .expect("materialize the declared headers");
    let souffle = effective_root.join("souffle");
    assert!(souffle.join("SouffleInterface.h").is_file());
    assert!(souffle.join("io").join("IOSystem.h").is_file());
    assert!(
        !effective_root.join("cassert").exists(),
        "a sibling of the header directory was copied"
    );
    assert!(
        !souffle.join("Extra.h").exists(),
        "an undeclared header was copied"
    );
    verify_runtime_inputs(&admitted, &effective_root).expect("the copy is the declared set");
}

/// The effective root is what every tool reads, and a compile stage writes into
/// the same work directory, so each boundary re-check has to catch a file added,
/// replaced, or removed while a tool was running.
#[test]
fn effective_include_root_is_rechecked_at_every_boundary() {
    let (fixture, entries, include_root) = runtime_input_fixture();
    let admitted =
        admit_fixture_manifest(&entries, &include_root).expect("admit the fixture manifest");
    let effective_root = fixture.path().join("effective");
    materialize_include_root(&admitted.souffle_include, &effective_root)
        .expect("materialize the declared headers");
    let declared = effective_root.join("souffle").join("SouffleInterface.h");
    let planted = effective_root.join("souffle").join("cassert");

    std::fs::write(&planted, b"#error shadowed\n").expect("plant an extra header");
    assert!(verify_runtime_inputs(&admitted, &effective_root)
        .unwrap_err()
        .contains("effective include souffle/cassert is not a declared file"));
    std::fs::remove_file(&planted).expect("remove the extra header");
    verify_runtime_inputs(&admitted, &effective_root).expect("the tree is declared again");

    std::fs::write(&declared, b"rewritten header\n").expect("rewrite a declared header");
    assert!(verify_runtime_inputs(&admitted, &effective_root)
        .unwrap_err()
        .contains("effective include souffle/SouffleInterface.h digest mismatch"));

    std::fs::remove_file(&declared).expect("remove a declared header");
    assert!(verify_runtime_inputs(&admitted, &effective_root)
        .unwrap_err()
        .contains("missing a declared file"));
}

/// The run writes two files of its own into the work directory a compile stage
/// then writes into: the verified probe functors, and the mapping policy the
/// suite pinned. Each boundary re-check has to accept the untouched pair and
/// refuse either one rewritten or removed while a tool was running.
#[test]
fn generated_work_files_are_rechecked_at_every_boundary() {
    let fixture = tempfile::tempdir().expect("create the generated file fixture");
    let functors = fixture.path().join("runtime_probe_functors.cpp");
    let policy = fixture.path().join("mapping_probe.dl");
    let functor_bytes = &b"probe functor bytes\n"[..];
    let policy_bytes = &b".decl Probe(x:symbol)\n"[..];
    std::fs::write(&functors, functor_bytes).expect("write the probe functors");
    std::fs::write(&policy, policy_bytes).expect("write the mapping policy");
    let generated = [
        (
            "probe functors",
            functors.clone(),
            runtime_sha256(functor_bytes),
        ),
        (
            "mapping policy",
            policy.clone(),
            runtime_sha256(policy_bytes),
        ),
    ];
    verify_generated_runtime_files(&generated, "the compiled build")
        .expect("the untouched pair is what was written");

    // A stage that rewrote the policy the compilers are about to read.
    std::fs::write(&policy, b".decl Probe(x:symbol)\nProbe(\"planted\").\n")
        .expect("rewrite the mapping policy");
    assert_eq!(
        verify_generated_runtime_files(&generated, "the compiled build").unwrap_err(),
        "verified mapping policy changed during the compiled build"
    );

    std::fs::write(&policy, policy_bytes).expect("restore the mapping policy");
    std::fs::write(&functors, b"planted functor bytes\n").expect("rewrite the functors");
    assert_eq!(
        verify_generated_runtime_files(&generated, "the interpreted build").unwrap_err(),
        "verified probe functors changed during the interpreted build"
    );

    // A file removed between stages is refused before whatever replaced it
    // could be read: the digest is taken through the same bounded inspection.
    std::fs::remove_file(&functors).expect("remove the probe functors");
    assert!(verify_generated_runtime_files(&generated, "execution")
        .unwrap_err()
        .contains("inspect runtime input probe functors"));
}

/// A link is followed by the compiler and would deliver bytes no record covers,
/// so it is refused on both sides: as a declared source, and as an entry in the
/// effective tree. Changed source bytes are refused with them.
#[cfg(unix)]
#[test]
fn include_links_and_changed_sources_are_refused() {
    let (fixture, entries, include_root) = runtime_input_fixture();
    let header = include_root.join("souffle").join("SouffleInterface.h");
    let admitted =
        admit_fixture_manifest(&entries, &include_root).expect("admit the fixture manifest");

    // A source header whose bytes changed after the manifest was pinned.
    std::fs::write(&header, b"rewritten header\n").expect("rewrite the source header");
    let refusal =
        materialize_include_root(&admitted.souffle_include, &fixture.path().join("changed"))
            .unwrap_err();
    assert!(
        refusal.contains("souffleInclude souffle/SouffleInterface.h digest mismatch"),
        "{refusal}"
    );

    // A source header replaced by a link to bytes the manifest never described.
    std::fs::remove_file(&header).expect("remove the source header");
    std::os::unix::fs::symlink(fixture_path(&entries, "commonPolicy"), &header)
        .expect("link the source header");
    let refusal =
        materialize_include_root(&admitted.souffle_include, &fixture.path().join("linked"))
            .unwrap_err();
    assert!(
        refusal.contains("souffleInclude souffle/SouffleInterface.h is not a plain bounded file"),
        "{refusal}"
    );

    // And a link planted inside the effective tree after it was written.
    let (fixture, entries, include_root) = runtime_input_fixture();
    let admitted =
        admit_fixture_manifest(&entries, &include_root).expect("admit the fixture manifest");
    let effective_root = fixture.path().join("effective");
    materialize_include_root(&admitted.souffle_include, &effective_root)
        .expect("materialize the declared headers");
    std::os::unix::fs::symlink(
        fixture_path(&entries, "commonPolicy"),
        effective_root.join("souffle").join("cassert"),
    )
    .expect("plant a link");
    assert!(verify_runtime_inputs(&admitted, &effective_root)
        .unwrap_err()
        .contains("effective include souffle/cassert is not a declared file"));
}

/// A role that names a file has to name a bounded regular file. Both paths below
/// are ones the manifest layer would accept — absolute, resolved, spelled once —
/// so what refuses them is the inspection that happens before anything is opened.
#[cfg(unix)]
#[test]
fn runtime_input_digest_refuses_a_device_and_an_oversized_file() {
    let device = std::path::Path::new("/dev/null");
    // Reading it would have succeeded, and digested as the empty file.
    assert!(std::fs::read(device).is_ok());
    let refusal = runtime_input_sha256("commonPolicy", device).unwrap_err();
    assert!(
        refusal.contains("runtime input commonPolicy is not a plain bounded file"),
        "{refusal}"
    );

    // The whole manifest layer would otherwise take it: the spelling is
    // canonical, and the digest a producer would pin is the one an empty read
    // returns. The refusal above is what stops it, and it stops it first.
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let mut declared_device = entries;
    declared_device[0]["path"] = serde_json::json!("/dev/null");
    declared_device[0]["sha256"] = serde_json::json!(runtime_sha256(b""));
    let refusal = admit_fixture_manifest(&declared_device, &include_root).unwrap_err();
    assert!(
        refusal.contains("runtime input commonPolicy is not a plain bounded file"),
        "{refusal}"
    );

    let fixture = tempfile::tempdir().expect("create the oversized input fixture");
    let oversized = fixture.path().join("oversized");
    std::fs::File::create(&oversized)
        .and_then(|file| file.set_len((64 << 20) + 1))
        .expect("declare a file one byte past the bound");
    let refusal = runtime_input_sha256("commonPolicy", &oversized).unwrap_err();
    assert!(
        refusal.contains("runtime input commonPolicy is not a plain bounded file"),
        "{refusal}"
    );
}

/// Production discovery takes both C++ headers from the evaluator shim's own
/// directory, so an authenticated set that pins either one somewhere else is a
/// set a client could never have discovered. Each copy here is real, pinned,
/// and admitted by the manifest layer; the asset constructor is what refuses
/// it, and it refuses before the toolchain is selected — which is exactly what
/// the untouched set goes on to fail.
#[cfg(feature = "compiler")]
#[test]
fn authenticated_assets_require_the_headers_beside_the_shim() {
    let _guard = RUNTIME_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (fixture, entries, include_root) = runtime_input_fixture();
    let attempt = std::sync::atomic::AtomicUsize::new(0);
    let build = |entries: &[serde_json::Value]| -> String {
        let admitted =
            admit_fixture_manifest(entries, &include_root).expect("admit the fixture manifest");
        // Each attempt gets its own effective root: a run always materializes
        // into a directory it just created.
        let index = attempt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let effective = fixture_path(entries, "commonPolicy").with_file_name(format!("out{index}"));
        std::env::set_var("SASY_ASSURED_MODE", "1");
        let built = crate::runtime_conformance::assured_souffle_assets(&admitted, &effective);
        std::env::remove_var("SASY_ASSURED_MODE");
        match built {
            Err(refusal) => refusal,
            Ok(_) => panic!("the fixture toolchain is not pinned, so no set can build"),
        }
    };

    let toolchain = build(&entries);
    assert!(toolchain.contains("SASY_TOOL_SOUFFLE_PATH"), "{toolchain}");

    let elsewhere = std::fs::canonicalize(fixture.path())
        .expect("resolve the fixture root")
        .join("elsewhere");
    std::fs::create_dir(&elsewhere).expect("create a second directory to pin from");
    for (role, name) in [
        ("evaluatorProtocol", "evaluator_protocol.h"),
        ("jsonStringCodec", "json_string_codec.h"),
    ] {
        let copy = elsewhere.join(name);
        std::fs::copy(fixture_path(&entries, role), &copy).expect("copy the header aside");
        let mut moved = entries.clone();
        let slot = moved
            .iter_mut()
            .find(|entry| entry["role"] == role)
            .expect("the fixture declares the role");
        *slot = input_entry(role, &copy);
        let refusal = build(&moved);
        assert!(
            refusal.contains(&format!("{role} must be {name} beside the shim")),
            "{refusal}"
        );
    }
}

/// Serializing env changes keeps these tests from reading each other's writes
/// — including the ones in `compiler_tests`, which set the same variables, so
/// the lock is the crate's one and not this file's.
use crate::TEST_ENV_LOCK as RUNTIME_ENV_LOCK;

#[test]
fn assured_runtime_environment_requires_assured_mode() {
    let _guard = RUNTIME_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    std::env::remove_var("SASY_ASSURED_MODE");
    let refusal = assured_runtime_environment().unwrap_err();
    assert!(refusal.contains("SASY_ASSURED_MODE must be exactly 1"));

    std::env::set_var("SASY_ASSURED_MODE", "true");
    let refusal = assured_runtime_environment().unwrap_err();
    std::env::remove_var("SASY_ASSURED_MODE");
    assert!(refusal.contains("SASY_ASSURED_MODE must be exactly 1"));
}

#[test]
fn assured_runtime_environment_refuses_behavior_changing_overrides() {
    let _guard = RUNTIME_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    // Both kinds: the ones that would change what this command builds, and the
    // ones that would put an unauthenticated directory on a compiler's header
    // search path without any manifest record naming it.
    for name in [
        "SASY_SOUFFLE_INCLUDE",
        "SASY_SOUFFLE_MAGIC_SET",
        "SASY_SOUFFLE_BUILD_CACHE_DIR",
        "SASY_SOUFFLE_ASSETS",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
        "OBJCPLUS_INCLUDE_PATH",
        "SDKROOT",
        "CPPFLAGS",
        "CXXFLAGS",
        "SOUFFLE_INCLUDE_PATH",
    ] {
        std::env::set_var(name, "1");
        let refusal = assured_runtime_environment().unwrap_err();
        std::env::remove_var(name);
        assert!(
            refusal.contains(name),
            "{name} must be refused, got: {refusal}"
        );
    }
}

/// A work root that already exists could hold a header tree or a build product
/// someone else put there, so the run has to create its own. The check runs
/// inside the real command, driven on a runtime this test owns so the
/// environment lock is never held across an await.
#[cfg(feature = "compiler")]
#[test]
fn runtime_conformance_refuses_a_work_root_it_did_not_create() {
    let _guard = RUNTIME_ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (fixture, entries, include_root) = runtime_input_fixture();
    let (manifest_bytes, manifest_sha256) =
        canonical_document(&manifest_document(&entries, &include_root));
    let manifest_path = fixture.path().join("input_manifest.json");
    std::fs::write(&manifest_path, &manifest_bytes).expect("write the fixture manifest");

    // The suite pins the runtime base Datalog it was generated against, which
    // is the manifest's own `commonPolicy` asset.
    let runtime_base =
        std::fs::read(fixture_path(&entries, "commonPolicy")).expect("read the fixture base");
    let mut suite = admitted_suite_fixture();
    suite["identity"]["runtimeBaseDatalogSha256"] =
        serde_json::json!(runtime_sha256(&runtime_base));
    let (suite_bytes, suite_sha256) = canonical_document(&suite);
    let suite_path = fixture.path().join("suite.json");
    std::fs::write(&suite_path, &suite_bytes).expect("write the fixture suite");

    let probe_functors = fixture.path().join("probe_functors.cpp");
    std::fs::write(&probe_functors, b"// probe\n").expect("write the fixture probe functors");
    let probe_sha256 = runtime_sha256(b"// probe\n");

    let planted = fixture.path().join("planted");
    std::fs::create_dir(&planted).expect("plant a work root the run did not create");
    let fresh = planted.join("conformance");
    let run = |work_root: std::path::PathBuf| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build a runtime for the conformance command")
            .block_on(execute_souffle_report(
                &suite_path,
                &suite_sha256,
                &probe_functors,
                &probe_sha256,
                &manifest_path,
                &manifest_sha256,
                &work_root,
            ))
            .expect_err("the fixture assets cannot build a real evaluator")
    };

    std::env::set_var("SASY_ASSURED_MODE", "1");
    let refused = run(planted.clone());
    // A path the run creates itself gets past the work-root check; what stops
    // it afterwards is the fixture toolchain, not the directory.
    let admitted = run(fresh.clone());
    std::env::remove_var("SASY_ASSURED_MODE");

    assert!(
        refused.contains("create fresh conformance work directory"),
        "{refused}"
    );
    assert!(
        !admitted.contains("create fresh conformance work directory"),
        "{admitted}"
    );
    assert!(fresh.is_dir(), "the run creates the work root it was given");
}

#[test]
fn runtime_execution_report_carries_every_selected_input() {
    let (_fixture, entries, include_root) = runtime_input_fixture();
    let admitted =
        admit_fixture_manifest(&entries, &include_root).expect("admit the fixture manifest");
    let report = crate::runtime_conformance::RuntimeExecutionReport {
        schema_version: 3,
        suite_sha256: "1".repeat(64),
        probe_functors_sha256: "2".repeat(64),
        input_manifest_sha256: admitted.sha256.clone(),
        inputs: admitted.records.clone(),
        backends: Vec::new(),
    };

    let encoded = serde_json::to_value(&report).expect("serialize execution report");
    assert_eq!(encoded["schemaVersion"], 3);
    assert_eq!(encoded["inputManifestSha256"], admitted.sha256);
    let reported = encoded["inputs"].as_array().expect("reported inputs");
    assert_eq!(reported.len(), FIXTURE_ROLES.len());
    for (record, entry) in reported.iter().zip(&entries) {
        assert_eq!(record["role"], entry["role"]);
        assert_eq!(record["path"], entry["path"]);
        assert_eq!(record["sha256"], entry["sha256"]);
    }
}

#[test]
fn runtime_observations_normalize_unordered_relation_rows() {
    let mut result: crate::evaluator::types::EvalActionResult =
        serde_json::from_value(serde_json::json!({
            "allow_passthrough": false,
            "authorized": false,
            "denial_reasons": [
                {"kind": "ask", "reason": "z", "suggestion": "last"},
                {"kind": "ask", "reason": "a", "suggestion": "first"}
            ],
            "deny_if_unauthorized": true,
            "index": 0,
            "is_allowlisted": true,
            "is_authenticated": false,
            "is_denylisted": false,
            "requires_approval": true,
            "transform_ids": ["zeta", "alpha"]
        }))
        .unwrap();

    normalize_runtime_result(&mut result);

    assert_eq!(result.transform_ids, ["alpha", "zeta"]);
    assert_eq!(result.denial_reasons[0].reason, "a");
    assert_eq!(result.denial_reasons[1].reason, "z");
}

#[test]
fn runtime_suite_admission_rejects_substitution_and_loss() {
    let valid = admitted_suite_fixture();
    let (bytes, digest) = canonical_document(&valid);
    // The mapping policy is carried with the digest the suite pinned over it,
    // so the copy this run writes to disk can be re-checked at every boundary.
    let admitted = admit_runtime_suite(&bytes, &digest).expect("admit the fixture suite");
    assert_eq!(
        admitted.policy_sha256,
        runtime_sha256(admitted.policy_source.as_bytes())
    );
    assert_eq!(
        admitted.policy_sha256,
        valid["probePolicy"]["sha256"].as_str().unwrap()
    );
    assert!(admit_runtime_suite(&bytes, &"0".repeat(64)).is_err());

    let mut unknown = valid.clone();
    unknown["scenarios"][0]["queryRequest"]["Query"]["unknown"] = serde_json::json!(true);
    let (bytes, digest) = canonical_document(&unknown);
    assert!(admit_runtime_suite(&bytes, &digest)
        .unwrap_err()
        .contains("exact typed round trip"));

    let mut lossy = valid.clone();
    lossy["scenarios"][0]["setupRequests"] = serde_json::json!([{
        "Update": {"updates": [{"NodeCreated": {
            "agent": null,
            "content": null,
            "derived_from": null,
            "entity": null,
            "id": "row",
            "principal": null,
            "role": null,
            "session_id": "",
            "tools": [{
                "arguments": "probe",
                "discarded": "lossy compact fixture field",
                "name": "fixture"
            }]
        }}]}
    }]);
    let (bytes, digest) = canonical_document(&lossy);
    assert!(admit_runtime_suite(&bytes, &digest)
        .unwrap_err()
        .contains("exact typed round trip"));

    let mut unsupported_word_size = valid.clone();
    unsupported_word_size["requiredSouffleWordSize"] = serde_json::json!(32);
    let (bytes, digest) = canonical_document(&unsupported_word_size);
    assert!(admit_runtime_suite(&bytes, &digest)
        .unwrap_err()
        .contains("identity or probe digest is invalid"));

    let mut duplicate = valid;
    duplicate["scenarios"][0]["expectedResults"][0]["transform_ids"] =
        serde_json::json!(["alpha", "alpha"]);
    let (bytes, digest) = canonical_document(&duplicate);
    assert!(admit_runtime_suite(&bytes, &digest)
        .unwrap_err()
        .contains("not canonical"));
}

/// Engine that records every `lazy_bind_session` /
/// `lazy_set_default` / `install_policy` call. Lets us verify
/// the rehydrate pass doesn't compile and the eager path does.
#[derive(Default)]
struct RecordingEngine {
    binds: Mutex<Vec<(SessionScope, String)>>,
    defaults: Mutex<Vec<(String, String)>>,
    installs: Mutex<Vec<(String, String, InstallMode)>>,
    ids: Mutex<std::collections::HashMap<(String, String), String>>,
}

impl crate::engine::Engine for RecordingEngine {
    fn apply_graph_updates(&self, _: Vec<crate::engine::GraphUpdate>) -> Result<(), anyhow::Error> {
        Ok(())
    }
    fn check_authorization(
        &self,
        _: &[String],
        _: &[sasy_common::policy_engine::Action],
        _: Option<&str>,
        _: &[String],
        _: &SessionScope,
        _: Option<&str>,
        _: Option<&str>,
    ) -> Result<sasy_common::policy_engine::AuthorizationResponse, anyhow::Error> {
        Ok(sasy_common::policy_engine::AuthorizationResponse {
            results: vec![],
            timing: None,
        })
    }
    fn reset(&self) -> Result<(), anyhow::Error> {
        Ok(())
    }
    fn load_rule_metadata(&self, _: &std::path::Path) -> Result<(), anyhow::Error> {
        Ok(())
    }
    fn get_sync_status(&self) -> SyncStatus {
        SyncStatus {
            current_sequence: 0,
            node_count: 0,
            edge_count: 0,
            connected: true,
        }
    }
    fn set_connected(&self, _: bool) {}
    fn set_sequence(&self, _: i64) {}

    fn lazy_bind_session(&self, scope: SessionScope, content_hash: &str) {
        self.binds.lock().push((scope, content_hash.to_string()));
    }
    fn lazy_set_default(&self, tenant: &str, content_hash: &str) {
        self.defaults
            .lock()
            .push((tenant.to_string(), content_hash.to_string()));
    }

    fn install_policy(
        &self,
        tenant: &str,
        content_hash: &str,
        _factory: EvaluatorFactory,
        _backend_name: String,
        mode: InstallMode,
    ) -> Result<String, anyhow::Error> {
        self.installs
            .lock()
            .push((tenant.to_string(), content_hash.to_string(), mode));
        let key = (tenant.to_string(), content_hash.to_string());
        let mut ids = self.ids.lock();
        let next_id = format!("policy-{}", ids.len());
        let id = ids.entry(key).or_insert(next_id).clone();
        Ok(id)
    }

    fn lookup_policy_by_content(&self, tenant: &str, content_hash: &str) -> Option<String> {
        self.ids
            .lock()
            .get(&(tenant.to_string(), content_hash.to_string()))
            .cloned()
    }

    fn set_session_policy(
        &self,
        _scope: &SessionScope,
        _policy_id: &str,
    ) -> Result<bool, anyhow::Error> {
        Ok(true)
    }
}

/// Lazy rehydrate writes bindings + defaults into the engine's
/// in-memory maps and never calls the install path. That's the
/// whole point: startup is O(1) regardless of source compile
/// cost.
#[tokio::test]
async fn lazy_rehydrate_does_not_compile() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    let engine = RecordingEngine::default();

    let bundle = PersistedPolicy {
        policy_source: "IsAuthorized(idx) :- Actions(idx, _).".into(),
        functor_source: String::new(),
        backend: "souffle".into(),
        functor_admission: Default::default(),
    };
    graph.put_policy_source("hash-alpha", &bundle).unwrap();
    graph.put_policy_source("hash-beta", &bundle).unwrap();

    graph
        .put_policy_binding(&SessionScope::new("acme", "A1"), "hash-alpha")
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "A2"), "hash-alpha")
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("orgb", "B1"), "hash-beta")
        .unwrap();
    graph
        .put_tenant_default_policy("acme", "hash-alpha")
        .unwrap();

    let stats = rehydrate_persisted_bindings(&graph, &engine).unwrap();

    assert_eq!(stats.bindings_restored, 3);
    assert_eq!(stats.defaults_restored, 1);
    assert_eq!(stats.orphans_no_source, 0);
    // Lazy = no install attempts at boot.
    assert_eq!(stats.policies_installed_eager, 0);
    assert!(engine.installs.lock().is_empty());
    // But the engine did get rehydration calls for each binding
    // and default.
    assert_eq!(engine.binds.lock().len(), 3);
    assert_eq!(engine.defaults.lock().len(), 1);
}

/// Bindings whose source bundle is missing on disk are still
/// rehydrated (the binding stays valid for a future re-upload),
/// but the orphan count is reported so operators can spot the
/// gap.
#[tokio::test]
async fn lazy_rehydrate_reports_orphan_sources() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    let engine = RecordingEngine::default();

    graph
        .put_policy_binding(&SessionScope::new("acme", "X"), "ghost-hash")
        .unwrap();
    // No put_policy_source for ghost-hash.

    let stats = rehydrate_persisted_bindings(&graph, &engine).unwrap();
    assert_eq!(stats.bindings_restored, 1);
    assert_eq!(stats.orphans_no_source, 1);
    // Binding still rehydrated — first dispatch will return a
    // clear "missing source" error.
    assert_eq!(engine.binds.lock().len(), 1);
}

/// Empty store: zero work, no error.
#[tokio::test]
async fn lazy_rehydrate_no_op_on_empty_store() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    let engine = RecordingEngine::default();
    let stats = rehydrate_persisted_bindings(&graph, &engine).unwrap();
    assert_eq!(stats.bindings_restored, 0);
    assert_eq!(stats.defaults_restored, 0);
    assert!(engine.binds.lock().is_empty());
}

// ── The functor gate at boot ────────────────────────────────────────
//
// The eager replay path compiles every persisted policy at startup, functor
// source included, so it is a LOAD of caller-supplied native code and takes
// the same gate the upload took — against the settings this process was
// started with, which need not be the ones that admitted the upload.
//
// "Compiled" here means "got past the gate". The compile itself needs
// souffle and g++ and is not what these tests are about, so they assert on
// `skipped_functor_gate`: zero means the gate let it through to the compiler.

/// A persisted bundle carrying functor source admitted as `admission`, bound
/// to one session and to the tenant default.
fn functor_bundle(admission: sasy_graph::FunctorAdmission) -> PersistedPolicy {
    PersistedPolicy {
        policy_source: "IsAuthorized(idx) :- Actions(idx, _).".into(),
        functor_source: "extern \"C\" const char* my_functor() { return \"\"; }".into(),
        backend: "souffle".into(),
        functor_admission: admission,
    }
}

fn graph_with(bundle: &PersistedPolicy) -> sasy_graph::GraphStore {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    graph.put_policy_source("hash-fn", bundle).unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "S1"), "hash-fn")
        .unwrap();
    graph
}

fn config_with(
    user_functors: crate::service::UserFunctors,
    sandbox_available: Option<bool>,
) -> crate::service::PolicyServiceConfig {
    crate::service::PolicyServiceConfig {
        user_functors,
        sandbox_available,
    }
}

/// The opt-in is off in this process, so functor source that was admitted as
/// user-supplied is not compiled at boot — whatever the process that took the
/// upload allowed. The entry is skipped; the session bound to it is still
/// rehydrated and fails closed on its first dispatch.
#[tokio::test]
async fn a_user_admitted_policy_is_not_compiled_at_boot_when_the_opt_in_is_off() {
    let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::User));
    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Refuse, None),
    )
    .unwrap();
    assert_eq!(stats.skipped_functor_gate, 1);
    assert_eq!(stats.policies_installed_eager, 0);
    assert!(
        engine.installs.lock().is_empty(),
        "a refused policy must not reach the registry"
    );
    assert_eq!(
        engine.binds.lock().len(),
        1,
        "the binding is still restored: the session fails closed, it is not silently \
         moved to another policy"
    );
}

/// With the unsandboxed opt-in the same bundle is admitted and handed to the
/// compiler, on any host.
#[tokio::test]
async fn a_user_admitted_policy_is_compiled_at_boot_when_the_opt_in_is_unsandboxed() {
    let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::User));
    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Unsandboxed, Some(false)),
    )
    .unwrap();
    assert_eq!(
        stats.skipped_functor_gate, 0,
        "the unsandboxed opt-in admits it even with no sandbox on the host"
    );
}

/// Under the sandboxed opt-in the answer is the sandbox probe's. Injected
/// here, so the test says the same thing on a Linux host with bubblewrap and
/// on a Mac without it.
#[tokio::test]
async fn a_user_admitted_policy_follows_the_sandbox_probe_under_the_sandboxed_opt_in() {
    for (sandbox, expected_skips) in [(true, 0), (false, 1)] {
        let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::User));
        let engine = RecordingEngine::default();
        let stats = replay_persisted_policies_eager(
            &graph,
            &engine,
            &config_with(crate::service::UserFunctors::Sandboxed, Some(sandbox)),
        )
        .unwrap();
        assert_eq!(
            stats.skipped_functor_gate, expected_skips,
            "sandbox_available={sandbox} should give {expected_skips} skip(s)"
        );
    }
}

/// Functor source an admin supplied loads whatever the flag says: the role
/// already carried the authority to decide what the engine runs, and no
/// opt-in was needed to admit it in the first place.
#[tokio::test]
async fn an_admin_admitted_policy_loads_at_boot_whatever_the_flag_says() {
    for config in [
        config_with(crate::service::UserFunctors::Refuse, Some(false)),
        config_with(crate::service::UserFunctors::Sandboxed, Some(false)),
        config_with(crate::service::UserFunctors::Unsandboxed, Some(false)),
    ] {
        let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::Admin));
        let engine = RecordingEngine::default();
        let stats = replay_persisted_policies_eager(&graph, &engine, &config).unwrap();
        assert_eq!(
            stats.skipped_functor_gate, 0,
            "an admin-admitted policy is never held back by the opt-in"
        );
    }
}

/// A policy with no functor source has nothing to admit, so the gate never
/// applies to it — not even with the opt-in off.
#[tokio::test]
async fn a_policy_without_functor_source_is_never_held_back_at_boot() {
    let bundle = PersistedPolicy {
        functor_source: String::new(),
        ..functor_bundle(sasy_graph::FunctorAdmission::User)
    };
    let graph = graph_with(&bundle);
    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .unwrap();
    assert_eq!(stats.skipped_functor_gate, 0);
}

// ── The admission class is monotone toward least privilege ──────────
//
// The store keys a persisted source by (content hash, admission class), so
// the two classes are independent records: no upload can widen a record
// already stored as user-supplied, and none can take an admin's record away.
// The load paths ask within the class in force, least-privileged first.

/// A non-admin uploads content; an admin later uploads the same content. The
/// user record keeps its class, so while the opt-in admits user-supplied
/// functors that is what loads — the admin's upload has not quietly turned
/// yesterday's user upload into an admin-admitted one.
#[tokio::test]
async fn an_admin_upload_does_not_widen_a_record_already_stored_as_user_supplied() {
    let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::User));
    graph
        .put_policy_source(
            "hash-fn",
            &functor_bundle(sasy_graph::FunctorAdmission::Admin),
        )
        .unwrap();
    let (class, _record) = crate::replay::admitted_policy_source(
        &graph,
        "hash-fn",
        &config_with(crate::service::UserFunctors::Unsandboxed, Some(false)),
    )
    .expect("the opt-in admits user-supplied functors, so a record loads")
    .expect("a record is stored under this hash");
    assert_eq!(
        class,
        sasy_graph::FunctorAdmission::User,
        "the admin's upload of the same content must not make the user's record load \
         as admin-admitted"
    );
}

/// Eager replay gates on the class the record was FOUND under, so a record
/// whose key and value disagree means the same thing on both load paths.
///
/// Here the key says admin and the value says user. With the opt-in off, admin
/// is the only class in force: the record is admitted and replay goes on to
/// compile it. A gate that read the value field instead would refuse it, and
/// the same record would then load lazily (which asks
/// `admitted_policy_source`) but not at boot.
#[tokio::test]
async fn eager_replay_admits_a_record_whose_key_says_admin_and_whose_value_says_user() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    graph
        .put_policy_source_in_class(
            "hash-fn",
            sasy_graph::FunctorAdmission::Admin,
            &functor_bundle(sasy_graph::FunctorAdmission::User),
        )
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "S1"), "hash-fn")
        .unwrap();

    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .unwrap();
    assert_eq!(
        stats.skipped_functor_gate, 0,
        "the record is stored under the admin key, and admin is in force, so no \
         functor gate may hold it back at boot"
    );
}

/// The mirror: the key says user and the value says admin. With the opt-in
/// off, that record is refused at boot — the value's claim buys it nothing.
#[tokio::test]
async fn eager_replay_refuses_a_record_whose_key_says_user_and_whose_value_says_admin() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    graph
        .put_policy_source_in_class(
            "hash-fn",
            sasy_graph::FunctorAdmission::User,
            &functor_bundle(sasy_graph::FunctorAdmission::Admin),
        )
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "S1"), "hash-fn")
        .unwrap();

    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .unwrap();
    assert_eq!(
        stats.skipped_functor_gate, 1,
        "the record sits under the user key and the opt-in is off, so the functor \
         gate holds it back"
    );
    assert!(
        engine.installs.lock().is_empty(),
        "nothing may be compiled or installed for a refused record"
    );
}

/// The class a record is read back in is the one ITS KEY encodes, not the
/// `functor_admission` field inside the record.
///
/// The key is the half the store's write path makes unforgeable: a later
/// uploader can add a record in its own class but cannot rewrite another
/// class's key. A record whose value claims `Admin` while sitting under the
/// user key is therefore user-supplied, and with the opt-in off it is refused
/// — a gate that read the field would compile it.
#[tokio::test]
async fn the_key_decides_the_class_not_the_field_inside_the_record() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    graph
        .put_policy_source_in_class(
            "hash-fn",
            sasy_graph::FunctorAdmission::User,
            &functor_bundle(sasy_graph::FunctorAdmission::Admin),
        )
        .unwrap();

    let refusal = crate::replay::admitted_policy_source(
        &graph,
        "hash-fn",
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .expect_err("the only record is under the user key, and the opt-in is off");
    assert!(
        refusal.starts_with(crate::service::FUNCTOR_REFUSED_AT_LOAD),
        "the refusal must be the functor gate's, recognisably: {refusal}"
    );

    // ...and with the opt-in on it loads, as the user-supplied record it is.
    let (class, _record) = crate::replay::admitted_policy_source(
        &graph,
        "hash-fn",
        &config_with(crate::service::UserFunctors::Unsandboxed, Some(false)),
    )
    .expect("the opt-in admits user-supplied functors")
    .expect("a record is stored under this hash");
    assert_eq!(
        class,
        sasy_graph::FunctorAdmission::User,
        "the key it was found under decides the class it loads in"
    );
}

/// The other order: an admin uploads content, a non-admin later uploads the
/// same content. The admin record is not downgraded — with the opt-in off,
/// where the user record is refused, the admin record still loads.
#[tokio::test]
async fn a_user_upload_does_not_downgrade_an_existing_admin_record() {
    let graph = graph_with(&functor_bundle(sasy_graph::FunctorAdmission::Admin));
    graph
        .put_policy_source(
            "hash-fn",
            &functor_bundle(sasy_graph::FunctorAdmission::User),
        )
        .unwrap();
    let (class, _record) = crate::replay::admitted_policy_source(
        &graph,
        "hash-fn",
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .expect("the admin record survives the later user upload and is admitted")
    .expect("a record is stored under this hash");
    assert_eq!(
        class,
        sasy_graph::FunctorAdmission::Admin,
        "a later non-admin upload of the same content must not take the admin record away"
    );
}

/// Boot replay honours the class of each record, in one pass over one store.
/// With the opt-in off, a hash an admin really did upload gets past the gate
/// even though a non-admin uploaded the same content afterwards, while a hash
/// that exists only as user-supplied is skipped.
#[tokio::test]
async fn boot_replay_honours_the_admission_class_of_each_record() {
    let graph = sasy_graph::GraphStore::new(None).unwrap();
    // "hash-both": an admin uploaded it, then a non-admin uploaded the same
    // content. Both records stand.
    graph
        .put_policy_source(
            "hash-both",
            &functor_bundle(sasy_graph::FunctorAdmission::Admin),
        )
        .unwrap();
    graph
        .put_policy_source(
            "hash-both",
            &functor_bundle(sasy_graph::FunctorAdmission::User),
        )
        .unwrap();
    // "hash-user": a non-admin uploaded it, and nobody else.
    graph
        .put_policy_source(
            "hash-user",
            &functor_bundle(sasy_graph::FunctorAdmission::User),
        )
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "S1"), "hash-both")
        .unwrap();
    graph
        .put_policy_binding(&SessionScope::new("acme", "S2"), "hash-user")
        .unwrap();

    let engine = RecordingEngine::default();
    let stats = replay_persisted_policies_eager(
        &graph,
        &engine,
        &config_with(crate::service::UserFunctors::Refuse, Some(false)),
    )
    .unwrap();
    assert_eq!(
        stats.skipped_functor_gate, 1,
        "exactly the user-only hash is held back: the later non-admin upload of \
         hash-both did not take away the admin record that admits it"
    );
}
