use std::io::Write;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::session_evaluator::EvaluatorFactory;

/// Schema version of the runtime input manifest this command accepts.
const RUNTIME_INPUT_SCHEMA_VERSION: u8 = 2;
/// Schema version of the execution report this command writes.
const RUNTIME_REPORT_SCHEMA_VERSION: u8 = 3;

/// The closed set of single-file runtime asset roles this command version may
/// use: generic Soufflé build and adapter assets, owned by the command rather
/// than by any policy family. The headers are a set, so they are declared apart
/// from these. A new asset fails closed until a schema change adds it. Sorted.
const RUNTIME_INPUT_ROLES: [&str; 8] = [
    "commonPolicy",
    "evaluatorProtocol",
    "evaluatorShim",
    "functorsCommon",
    "interpretedAdapter",
    "interpretedFunctors",
    "jsonStringCodec",
    "sugarPy",
];

/// Variables that would change what the assured path builds, serve a build
/// product from outside it, or add a header search path no record describes:
/// the manifest, not the environment, decides the build. Sorted.
#[rustfmt::skip]
const REFUSED_RUNTIME_OVERRIDES: [&str; 16] = [
    "CPATH", "CPLUS_INCLUDE_PATH", "CPPFLAGS", "CXXFLAGS", "C_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "SASY_SOUFFLE_ASSETS",
    "SASY_SOUFFLE_BUILD_CACHE_DIR",
    "SASY_SOUFFLE_BUILD_CACHE_MAX_AGE_DAYS", "SASY_SOUFFLE_BUILD_CACHE_MAX_BYTES",
    "SASY_SOUFFLE_BUILD_CACHE_PER_TENANT", "SASY_SOUFFLE_INCLUDE",
    "SASY_SOUFFLE_MAGIC_SET", "SDKROOT", "SOUFFLE_INCLUDE_PATH",
];

/// Far larger than any Soufflé install: refuse rather than copy an unbounded set.
const MAX_INCLUDE_FILES: usize = 4096;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RuntimeConformanceSuite {
    schema_version: u8,
    identity: RuntimeConformanceIdentity,
    probe_policy: RuntimeProbePolicy,
    required_souffle_word_size: Option<u8>,
    scenarios: Vec<RuntimeScenario>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RuntimeConformanceIdentity {
    family_id: String,
    public_family_sha256: String,
    runtime_base_datalog_sha256: String,
    runtime_mapping_sha256: String,
    composed_datalog_sha256: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeProbePolicy {
    source: String,
    sha256: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RuntimeScenario {
    id: String,
    setup_requests: Vec<serde_json::Value>,
    query_request: serde_json::Value,
    expected_results: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RuntimeInputManifest {
    schema_version: u8,
    inputs: Vec<RuntimeInputRecord>,
    souffle_include: SouffleIncludeInput,
}

/// One runtime asset: declared by the manifest, reported as selected.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeInputRecord {
    pub(crate) role: String,
    pub(crate) path: String,
    pub(crate) sha256: String,
}

/// The Soufflé headers, named one file at a time rather than as one opaque
/// directory: `files` maps each allowed path under `source_root` to its digest.
/// Only these are copied into the directory the compiler searches, so no sibling
/// of theirs can shadow a standard header. A JSON object makes the set sorted and
/// repetition-free under the canonical-encoding rule.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub(crate) struct SouffleIncludeInput {
    pub(crate) source_root: String,
    pub(crate) files: std::collections::BTreeMap<String, String>,
}

/// Digest, schema, roles, paths, and digests all checked: the only way to name an asset.
#[derive(Debug)]
pub(crate) struct AdmittedRuntimeInputs {
    pub(crate) sha256: String,
    pub(crate) records: Vec<RuntimeInputRecord>,
    pub(crate) souffle_include: SouffleIncludeInput,
}

impl AdmittedRuntimeInputs {
    fn record(&self, role: &str) -> Result<&RuntimeInputRecord, String> {
        self.records
            .iter()
            .find(|record| record.role == role)
            .ok_or_else(|| format!("runtime input role {role} is not admitted"))
    }

    fn path(&self, role: &str) -> Result<PathBuf, String> {
        self.record(role).map(|record| PathBuf::from(&record.path))
    }
}

#[derive(Debug)]
pub(crate) struct AdmittedRuntimeScenario {
    pub(crate) id: String,
    pub(crate) setup: Vec<AdmittedRuntimeSetup>,
    pub(crate) query: crate::evaluator::types::EvalAuthRequest,
    pub(crate) expected: Vec<crate::evaluator::types::EvalActionResult>,
}

#[derive(Debug)]
pub(crate) enum AdmittedRuntimeSetup {
    Update(Vec<crate::engine::GraphUpdate>),
    SetMetadata(Vec<crate::evaluator::types::PolicyMetadataFact>),
}

#[derive(Debug)]
pub(crate) struct AdmittedRuntimeSuite {
    pub(crate) policy_source: String,
    /// The digest the suite pinned over `policy_source`, carried so the copy
    /// this run writes to disk can be re-checked against it later.
    pub(crate) policy_sha256: String,
    pub(crate) runtime_base_datalog_sha256: String,
    pub(crate) required_souffle_word_size: Option<u8>,
    pub(crate) scenarios: Vec<AdmittedRuntimeScenario>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeExecutionReport {
    pub(crate) schema_version: u8,
    pub(crate) suite_sha256: String,
    pub(crate) probe_functors_sha256: String,
    pub(crate) input_manifest_sha256: String,
    pub(crate) inputs: Vec<RuntimeInputRecord>,
    pub(crate) backends: Vec<RuntimeBackendReport>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeBackendReport {
    backend_role: String,
    executable_sha256: String,
    runtime_executable_sha256: Option<String>,
    support_library_sha256: Option<String>,
    observations: Vec<RuntimeObservation>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeObservation {
    pub(crate) scenario_id: String,
    pub(crate) response: crate::evaluator::types::EvalAuthResponse,
}

pub(crate) fn runtime_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    crate::hash::hex(&Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn typed_runtime_request(
    raw: serde_json::Value,
    label: &str,
) -> Result<crate::evaluator::types::EvalRequest, String> {
    let typed = serde_json::from_value(raw.clone())
        .map_err(|error| format!("{label} is not an EvalRequest: {error}"))?;
    let round_trip = serde_json::to_value(&typed)
        .map_err(|error| format!("{label} cannot be reserialized: {error}"))?;
    if round_trip != raw {
        return Err(format!("{label} is not an exact typed round trip"));
    }
    Ok(typed)
}

fn typed_runtime_result(
    raw: serde_json::Value,
    label: &str,
) -> Result<crate::evaluator::types::EvalActionResult, String> {
    let typed = serde_json::from_value(raw.clone())
        .map_err(|error| format!("{label} is not an EvalActionResult: {error}"))?;
    let round_trip = serde_json::to_value(&typed)
        .map_err(|error| format!("{label} cannot be reserialized: {error}"))?;
    if round_trip != raw {
        return Err(format!("{label} is not an exact typed round trip"));
    }
    Ok(typed)
}

pub(crate) fn normalize_runtime_result(result: &mut crate::evaluator::types::EvalActionResult) {
    result.transform_ids.sort();
    result.transform_ids.dedup();
    result.denial_reasons.sort_by(|left, right| {
        (&left.kind, &left.reason, &left.suggestion).cmp(&(
            &right.kind,
            &right.reason,
            &right.suggestion,
        ))
    });
}

/// Refuse to run outside assured mode, or with any override that would let the
/// build reach past the manifest. Refusals are collected, not reported one by one.
pub(crate) fn assured_runtime_environment() -> Result<(), String> {
    let mut refusals: Vec<String> = REFUSED_RUNTIME_OVERRIDES
        .iter()
        .filter(|name| std::env::var_os(name).is_some())
        .map(|name| format!("{name} would override an authenticated runtime input"))
        .collect();
    if std::env::var("SASY_ASSURED_MODE").ok().as_deref() != Some("1") {
        refusals.push("SASY_ASSURED_MODE must be exactly 1".to_string());
    }
    if refusals.is_empty() {
        return Ok(());
    }
    Err(format!(
        "runtime conformance refused this environment: {}",
        refusals.join("; ")
    ))
}

/// Reject anything but a lexically canonical path spelled exactly one way: `.`,
/// `..`, an empty, repeated, or trailing separator, and a bare root each give one
/// asset two spellings. An absolute path must also already be resolved, or a link
/// would misname what was read; a relative one is only joined onto a resolved
/// root. Re-spelling the components and comparing them against the declared text
/// is what catches all of this: `Path` equality normalizes exactly these away.
fn check_canonical_path(role: &str, path: &str, absolute: bool) -> Result<(), String> {
    let candidate = Path::new(path);
    let spelled_once = candidate.is_absolute() == absolute
        && candidate.file_name().is_some()
        && candidate.components().all(|part| {
            matches!(part, Component::Normal(_)) || (absolute && part == Component::RootDir)
        })
        && candidate.components().collect::<PathBuf>().to_str() == Some(path)
        && (!absolute || std::fs::canonicalize(candidate).ok().as_deref() == Some(candidate));
    if !spelled_once {
        return Err(format!("runtime input {role} path is not canonical"));
    }
    Ok(())
}

/// Read one declared asset, inspecting it before it is opened: a device, socket,
/// or link hands back bytes no digest can describe. The bound is the tool pin's.
fn bounded_runtime_input(label: &str, path: &Path) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect runtime input {label}: {error}"))?;
    if !metadata.is_file() || metadata.len() > 64 << 20 {
        return Err(format!("runtime input {label} is not a plain bounded file"));
    }
    std::fs::read(path).map_err(|error| format!("read runtime input {label}: {error}"))
}

pub(crate) fn runtime_input_sha256(role: &str, path: &Path) -> Result<String, String> {
    bounded_runtime_input(role, path).map(|bytes| runtime_sha256(&bytes))
}

/// Admit the declared header set: bounded in size, every path a safe relative
/// spelling, every digest well formed.
fn admit_souffle_include(include: &SouffleIncludeInput) -> Result<(), String> {
    check_canonical_path("souffleInclude", &include.source_root, true)?;
    if include.files.is_empty() || include.files.len() > MAX_INCLUDE_FILES {
        return Err("runtime input souffleInclude declares no or too many files".into());
    }
    for (path, sha256) in &include.files {
        check_canonical_path(&format!("souffleInclude {path}"), path, false)?;
        if !is_sha256(sha256) {
            return Err(format!("souffleInclude {path} digest is invalid"));
        }
    }
    Ok(())
}

/// Copy exactly the declared headers into a fresh private root inside the run's
/// own work directory, checking each source file's bytes before they are written,
/// then check the root itself. It is the only include path any tool is given.
pub(crate) fn materialize_include_root(
    include: &SouffleIncludeInput,
    effective_root: &Path,
) -> Result<(), String> {
    let source_root = Path::new(&include.source_root);
    std::fs::create_dir(effective_root)
        .map_err(|error| format!("create effective include root: {error}"))?;
    for (path, sha256) in &include.files {
        let label = format!("souffleInclude {path}");
        let bytes = bounded_runtime_input(&label, &source_root.join(path))?;
        if runtime_sha256(&bytes) != *sha256 {
            return Err(format!("runtime input {label} digest mismatch"));
        }
        let target = effective_root.join(path);
        std::fs::create_dir_all(target.parent().ok_or("include file has no parent")?)
            .map_err(|error| format!("create effective include directory: {error}"))?;
        write_bytes_create_new(&bytes, &target, "effective include file")?;
    }
    verify_effective_include(include, effective_root)
}

/// Check the directory the compiler is actually given: every entry is a declared
/// file holding the declared bytes, or a directory leading to one, and none is
/// missing. A non-UTF-8 name cannot equal a declared path, and `symlink_metadata`
/// reports a link as neither file nor directory, so one comparison refuses both.
fn verify_effective_include(
    include: &SouffleIncludeInput,
    effective_root: &Path,
) -> Result<(), String> {
    let mut found = 0usize;
    let mut pending = std::collections::VecDeque::from([effective_root.to_path_buf()]);
    while let Some(directory) = pending.pop_front() {
        let listing = std::fs::read_dir(&directory)
            .map_err(|error| format!("read effective include: {error}"))?;
        for entry in listing {
            let entry = entry.map_err(|error| format!("list effective include: {error}"))?;
            let path = entry.path();
            let name = path.strip_prefix(effective_root).unwrap_or(&path);
            let name = name.to_string_lossy().into_owned();
            let kind = std::fs::symlink_metadata(&path)
                .map_err(|error| format!("inspect effective include {name}: {error}"))?;
            let prefix = format!("{name}/");
            if kind.is_dir() && include.files.keys().any(|file| file.starts_with(&prefix)) {
                pending.push_back(path);
                continue;
            }
            let expected = include
                .files
                .get(&name)
                .filter(|_| kind.is_file())
                .ok_or_else(|| format!("effective include {name} is not a declared file"))?;
            if runtime_input_sha256(&format!("effective include {name}"), &path)? != *expected {
                return Err(format!("effective include {name} digest mismatch"));
            }
            found += 1;
        }
    }
    (found == include.files.len())
        .then_some(())
        .ok_or_else(|| "effective include tree is missing a declared file".into())
}

/// Re-check the files this run wrote into its own work directory against the
/// digests they were bound to when they were written. A compile stage writes
/// into that same directory, so one that rewrote the verified probe functors or
/// the generated mapping policy would otherwise reach the next stage unnoticed.
pub(crate) fn verify_generated_runtime_files(
    generated: &[(&str, PathBuf, String)],
    stage: &str,
) -> Result<(), String> {
    for (label, path, expected) in generated {
        if runtime_input_sha256(label, path)? != *expected {
            return Err(format!("verified {label} changed during {stage}"));
        }
    }
    Ok(())
}

/// Re-check every declared asset, and the headers the compiler will actually
/// search, against the digests the manifest pinned. Called at each security
/// boundary, so an asset swapped between builds is caught rather than compiled.
pub(crate) fn verify_runtime_inputs(
    inputs: &AdmittedRuntimeInputs,
    effective_include_root: &Path,
) -> Result<(), String> {
    for record in &inputs.records {
        if runtime_input_sha256(&record.role, Path::new(&record.path))? != record.sha256 {
            return Err(format!("runtime input {} digest mismatch", record.role));
        }
    }
    verify_effective_include(&inputs.souffle_include, effective_include_root)
}

pub(crate) fn admit_runtime_inputs(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<AdmittedRuntimeInputs, String> {
    if !is_sha256(expected_sha256) || runtime_sha256(bytes) != expected_sha256 {
        return Err("runtime input manifest digest mismatch".into());
    }
    let raw: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("input manifest JSON: {error}"))?;
    let mut canonical =
        serde_json::to_vec(&raw).map_err(|error| format!("canonical input manifest: {error}"))?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("runtime input manifest is not canonical JSON".into());
    }
    let manifest: RuntimeInputManifest =
        serde_json::from_value(raw).map_err(|error| format!("input manifest schema: {error}"))?;
    if manifest.schema_version != RUNTIME_INPUT_SCHEMA_VERSION {
        return Err("runtime input manifest schema version is unsupported".into());
    }
    admit_souffle_include(&manifest.souffle_include)?;
    let mut records = manifest.inputs;
    let mut declared = std::collections::BTreeSet::new();
    for record in &records {
        let role = &record.role;
        if !RUNTIME_INPUT_ROLES.contains(&role.as_str()) {
            return Err(format!("runtime input role {role} is not a declared role"));
        }
        if !declared.insert(role.clone()) {
            return Err(format!("runtime input role {role} is declared twice"));
        }
        if !is_sha256(&record.sha256) {
            return Err(format!("runtime input {role} digest is invalid"));
        }
        check_canonical_path(role, &record.path, true)?;
        if runtime_input_sha256(role, Path::new(&record.path))? != record.sha256 {
            return Err(format!("runtime input {role} digest mismatch"));
        }
    }
    // Roles are closed and never repeat, so a size match is exact set equality.
    if declared.len() != RUNTIME_INPUT_ROLES.len() {
        return Err("runtime input manifest does not declare every role".into());
    }
    records.sort_by(|left, right| left.role.cmp(&right.role));
    Ok(AdmittedRuntimeInputs {
        sha256: expected_sha256.to_string(),
        records,
        souffle_include: manifest.souffle_include,
    })
}

pub(crate) fn admit_runtime_suite(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<AdmittedRuntimeSuite, String> {
    if !is_sha256(expected_sha256) || runtime_sha256(bytes) != expected_sha256 {
        return Err("runtime suite digest mismatch".into());
    }
    let raw: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("suite JSON: {error}"))?;
    let mut canonical =
        serde_json::to_vec(&raw).map_err(|error| format!("canonical suite: {error}"))?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("runtime suite is not canonical JSON".into());
    }
    let suite: RuntimeConformanceSuite =
        serde_json::from_value(raw).map_err(|error| format!("suite schema: {error}"))?;
    let identity = &suite.identity;
    if suite.schema_version != 1
        || !matches!(suite.required_souffle_word_size, None | Some(64))
        || identity.family_id.is_empty()
        || !is_sha256(&identity.public_family_sha256)
        || !is_sha256(&identity.runtime_base_datalog_sha256)
        || !is_sha256(&identity.runtime_mapping_sha256)
        || !is_sha256(&identity.composed_datalog_sha256)
        || runtime_sha256(suite.probe_policy.source.as_bytes()) != suite.probe_policy.sha256
    {
        return Err("runtime suite identity or probe digest is invalid".into());
    }

    let runtime_base_datalog_sha256 = identity.runtime_base_datalog_sha256.clone();
    let mut ids = std::collections::HashSet::new();
    let mut admitted = Vec::with_capacity(suite.scenarios.len());
    for scenario in suite.scenarios {
        if scenario.id.is_empty() || !ids.insert(scenario.id.clone()) {
            return Err("runtime scenario IDs must be nonempty and unique".into());
        }
        let mut setup = Vec::with_capacity(scenario.setup_requests.len());
        for (index, raw) in scenario.setup_requests.into_iter().enumerate() {
            match typed_runtime_request(raw, &format!("{} setup {index}", scenario.id))? {
                crate::evaluator::types::EvalRequest::Update { updates } => {
                    setup.push(AdmittedRuntimeSetup::Update(updates));
                }
                crate::evaluator::types::EvalRequest::SetMetadata { facts } => {
                    setup.push(AdmittedRuntimeSetup::SetMetadata(facts));
                }
                _ => {
                    return Err(format!(
                        "{} setup is not Update or SetMetadata",
                        scenario.id
                    ))
                }
            }
        }
        let query =
            match typed_runtime_request(scenario.query_request, &format!("{} query", scenario.id))?
            {
                crate::evaluator::types::EvalRequest::Query(request) => request,
                _ => return Err(format!("{} query is not a query request", scenario.id)),
            };
        if scenario.expected_results.len() != query.actions.len() {
            return Err(format!(
                "{} result count does not match actions",
                scenario.id
            ));
        }
        let mut expected_results = Vec::with_capacity(scenario.expected_results.len());
        for (index, raw) in scenario.expected_results.into_iter().enumerate() {
            let expected =
                typed_runtime_result(raw, &format!("{} expected result {index}", scenario.id))?;
            let mut normalized = expected.clone();
            normalize_runtime_result(&mut normalized);
            if expected.index != index as u32
                || serde_json::to_value(&normalized).expect("serialize normalized result")
                    != serde_json::to_value(&expected).expect("serialize expected result")
            {
                return Err(format!(
                    "{} expected results are not canonical",
                    scenario.id
                ));
            }
            expected_results.push(expected);
        }
        admitted.push(AdmittedRuntimeScenario {
            id: scenario.id,
            setup,
            query,
            expected: expected_results,
        });
    }
    if admitted.is_empty() {
        return Err("runtime suite has no scenarios".into());
    }
    Ok(AdmittedRuntimeSuite {
        policy_source: suite.probe_policy.source,
        policy_sha256: suite.probe_policy.sha256,
        runtime_base_datalog_sha256,
        required_souffle_word_size: suite.required_souffle_word_size,
        scenarios: admitted,
    })
}

pub(crate) async fn execute_runtime_suite(
    name: &str,
    factory: &EvaluatorFactory,
    suite: &AdmittedRuntimeSuite,
) -> Result<Vec<RuntimeObservation>, String> {
    let mut observations = Vec::with_capacity(suite.scenarios.len());
    for scenario in &suite.scenarios {
        let evaluator = factory()
            .map_err(|error| format!("{name} could not start conformance evaluator: {error}"))?;
        if evaluator.backend_name() != name {
            return Err(format!("{name} adapter identity changed"));
        }
        for setup in &scenario.setup {
            match setup {
                AdmittedRuntimeSetup::Update(updates) => {
                    evaluator.update(updates.clone()).await.map_err(|error| {
                        format!("{name} could not apply conformance update: {error}")
                    })?
                }
                AdmittedRuntimeSetup::SetMetadata(facts) => evaluator
                    .set_metadata(facts.clone())
                    .await
                    .map_err(|error| {
                        format!("{name} could not apply conformance metadata: {error}")
                    })?,
            }
        }
        let mut response = evaluator
            .query(scenario.query.clone())
            .await
            .map_err(|error| format!("{name} could not execute conformance query: {error}"))?;
        if response.results.len() != scenario.expected.len() {
            return Err(format!("{name} runtime scenario result count differs"));
        }
        response
            .results
            .iter_mut()
            .for_each(normalize_runtime_result);
        for (actual, expected) in response.results.iter().zip(&scenario.expected) {
            let actual = serde_json::to_value(actual)
                .map_err(|error| format!("serialize {name} result: {error}"))?;
            let expected = serde_json::to_value(expected)
                .map_err(|error| format!("serialize expected result: {error}"))?;
            if actual != expected {
                return Err(format!(
                    "{name} runtime scenario {} result {} differs: actual={actual}, expected={expected}",
                    scenario.id, expected["index"],
                ));
            }
        }
        observations.push(RuntimeObservation {
            scenario_id: scenario.id.clone(),
            response,
        });
    }
    Ok(observations)
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|error| format!("serialize JSON: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_bytes_create_new(bytes: &[u8], output_path: &Path, label: &str) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(|error| format!("create {label}: {error}"))?;
    file.write_all(bytes)
        .map_err(|error| format!("write {label}: {error}"))
}

pub fn write_execution_report_create_new(
    report: &RuntimeExecutionReport,
    output_path: &Path,
) -> Result<(), String> {
    let bytes = canonical_json_bytes(report)?;
    write_bytes_create_new(&bytes, output_path, "execution report")
}

#[cfg(feature = "compiler")]
fn authenticate_launch_artifacts(
    role: &str,
    launch: &crate::evaluator::factory::SouffleLaunchArtifacts,
) -> Result<(String, Option<String>, Option<String>), String> {
    let digest = |path: &Path| runtime_input_sha256(role, path);
    Ok((
        digest(&launch.executable_path)?,
        launch
            .runtime_executable_path
            .as_deref()
            .map(digest)
            .transpose()?,
        launch
            .support_library_path
            .as_deref()
            .map(digest)
            .transpose()?,
    ))
}

#[cfg(feature = "compiler")]
async fn execute_backend_report(
    build: &crate::evaluator::factory::SouffleFactoryBuild,
    suite: &AdmittedRuntimeSuite,
) -> Result<RuntimeBackendReport, String> {
    let role = &build.backend_name;
    let before = authenticate_launch_artifacts(role, &build.launch_artifacts)?;
    let observations = execute_runtime_suite(&build.backend_name, &build.factory, suite).await?;
    let after = authenticate_launch_artifacts(role, &build.launch_artifacts)?;
    if after != before {
        return Err(format!("{role} launch artifacts changed during execution"));
    }
    Ok(RuntimeBackendReport {
        backend_role: role.clone(),
        executable_sha256: before.0,
        runtime_executable_sha256: before.1,
        support_library_sha256: before.2,
        observations,
    })
}

/// Build the whole Soufflé asset set from the manifest and nothing else: no
/// candidate-path search, no fallback, every path from a record whose digest was
/// already checked, and the declared headers copied into the caller's own fresh
/// effective root. Production discovery takes both C++ headers from the evaluator
/// shim's own directory, and so does this, before any tool runs.
#[cfg(feature = "compiler")]
pub(crate) fn assured_souffle_assets(
    inputs: &AdmittedRuntimeInputs,
    effective_include_root: &Path,
) -> Result<crate::evaluator::factory::AuthenticatedSouffleAssets, String> {
    let shim = inputs.path("evaluatorShim")?;
    for (role, name) in [
        ("evaluatorProtocol", "evaluator_protocol.h"),
        ("jsonStringCodec", "json_string_codec.h"),
    ] {
        if inputs.path(role)? != shim.with_file_name(name) {
            return Err(format!("{role} must be {name} beside the shim"));
        }
    }
    materialize_include_root(&inputs.souffle_include, effective_include_root)?;
    Ok(crate::evaluator::factory::AuthenticatedSouffleAssets {
        compiler: crate::compiler::SouffleAssets {
            include_dir: effective_include_root.to_path_buf(),
            sugar_py: inputs.path("sugarPy")?,
            evaluator_shim: inputs.path("evaluatorShim")?,
            evaluator_protocol: inputs.path("evaluatorProtocol")?,
            json_string_codec: inputs.path("jsonStringCodec")?,
            functors_common: inputs.path("functorsCommon")?,
            common_policy: Some(inputs.path("commonPolicy")?),
            // Manifest paths, not a directory this binary owns: the shim and
            // its two headers are forced to share a directory above, but that
            // directory is the operator's and holds other files. Bound file by
            // file.
            exclusive_dir: None,
            word_size: crate::compiler::detect_souffle_word_size()
                .map_err(|error| format!("detect Soufflé word size: {error}"))?,
        },
        interpreted_adapter: inputs.path("interpretedAdapter")?,
        interpreted_functors: inputs.path("interpretedFunctors")?,
    })
}

#[cfg(feature = "compiler")]
pub async fn execute_souffle_report(
    suite_path: &Path,
    expected_suite_sha256: &str,
    probe_functors: &Path,
    expected_probe_functors_sha256: &str,
    input_manifest_path: &Path,
    expected_input_manifest_sha256: &str,
    work_root: &Path,
) -> Result<RuntimeExecutionReport, String> {
    assured_runtime_environment()?;
    let manifest_bytes = std::fs::read(input_manifest_path)
        .map_err(|error| format!("read runtime input manifest: {error}"))?;
    let inputs = admit_runtime_inputs(&manifest_bytes, expected_input_manifest_sha256)?;
    let suite_bytes =
        std::fs::read(suite_path).map_err(|error| format!("read conformance suite: {error}"))?;
    let suite = admit_runtime_suite(&suite_bytes, expected_suite_sha256)?;
    let runtime_base_path = inputs.path("commonPolicy")?;
    let runtime_base =
        std::fs::read(&runtime_base_path).map_err(|error| format!("read runtime base: {error}"))?;
    if runtime_sha256(&runtime_base) != suite.runtime_base_datalog_sha256 {
        return Err("runtime base Datalog does not match the generated conformance suite".into());
    }
    if !is_sha256(expected_probe_functors_sha256) {
        return Err("runtime conformance probe digest is invalid".into());
    }
    let probe_functor_bytes = bounded_runtime_input("probe functors", probe_functors)?;
    if runtime_sha256(&probe_functor_bytes) != expected_probe_functors_sha256 {
        return Err("runtime conformance probe functor digest mismatch".into());
    }

    // An existing work root could hold a planted include tree or build product.
    std::fs::create_dir(work_root)
        .map_err(|error| format!("create fresh conformance work directory: {error}"))?;
    let effective_include_root = work_root.join("include");
    let assets = assured_souffle_assets(&inputs, &effective_include_root)?;
    if let Some(required) = suite.required_souffle_word_size {
        let actual = assets.compiler.word_size;
        if actual != required {
            return Err(format!(
                "runtime primitive specimens require {required}-bit Souffle; found {actual}-bit"
            ));
        }
    }
    let verified_probe_functors = work_root.join("runtime_probe_functors.cpp");
    write_bytes_create_new(
        &probe_functor_bytes,
        &verified_probe_functors,
        "verified runtime probe functors",
    )?;
    let policy_path = work_root.join("mapping_probe.dl");
    write_bytes_create_new(
        suite.policy_source.as_bytes(),
        &policy_path,
        "generated mapping policy",
    )?;
    // Everything the next stage reads: the files this run generated, then the
    // declared assets and the headers the compiler is given.
    let generated = [
        (
            "probe functors",
            verified_probe_functors.clone(),
            expected_probe_functors_sha256.to_string(),
        ),
        (
            "mapping policy",
            policy_path.clone(),
            suite.policy_sha256.clone(),
        ),
    ];
    let boundary = |stage: &str| -> Result<(), String> {
        verify_generated_runtime_files(&generated, stage)?;
        verify_runtime_inputs(&inputs, &effective_include_root)
    };

    let compiled_root = work_root.join("compiled");
    let interpreted_root = work_root.join("interpreted");
    std::fs::create_dir_all(&compiled_root)
        .map_err(|error| format!("create compiled work directory: {error}"))?;
    std::fs::create_dir_all(&interpreted_root)
        .map_err(|error| format!("create interpreted work directory: {error}"))?;
    // Bracket both compiler stages: this is the check before the first one.
    boundary("work directory setup")?;

    let compiled = crate::evaluator::factory::build_souffle_factory_with_artifacts(
        sasy_common::Backend::Souffle,
        &compiled_root,
        &policy_path,
        Some(&verified_probe_functors),
        Some(&assets),
    )?;
    boundary("the compiled build")?;
    let interpreted = crate::evaluator::factory::build_souffle_factory_with_artifacts(
        sasy_common::Backend::SouffleInterpreted,
        &interpreted_root,
        &policy_path,
        Some(&verified_probe_functors),
        Some(&assets),
    )?;
    boundary("the interpreted build")?;

    let compiled_report = execute_backend_report(&compiled, &suite).await;
    let interpreted_report = execute_backend_report(&interpreted, &suite).await;
    if compiled_report.is_err() || interpreted_report.is_err() {
        return Err(format!(
            "compiled adapter: {}; interpreted adapter: {}",
            compiled_report
                .as_ref()
                .err()
                .map_or("passed", String::as_str),
            interpreted_report
                .as_ref()
                .err()
                .map_or("passed", String::as_str),
        ));
    }
    let compiled_report = compiled_report.expect("compiled report status was checked");
    let interpreted_report = interpreted_report.expect("interpreted report status was checked");
    // The adapter is built before this command runs and then declared as a
    // runtime input, so its executable has to be exactly the declared bytes.
    if interpreted_report.executable_sha256 != inputs.record("interpretedAdapter")?.sha256 {
        return Err("interpreted adapter is not the declared runtime input".into());
    }
    let compiled_observations = serde_json::to_value(&compiled_report.observations)
        .map_err(|error| format!("serialize compiled observations: {error}"))?;
    let interpreted_observations = serde_json::to_value(&interpreted_report.observations)
        .map_err(|error| format!("serialize interpreted observations: {error}"))?;
    if compiled_observations != interpreted_observations {
        return Err("adapter parity mismatch".into());
    }
    boundary("execution")?;

    Ok(RuntimeExecutionReport {
        schema_version: RUNTIME_REPORT_SCHEMA_VERSION,
        suite_sha256: expected_suite_sha256.to_string(),
        probe_functors_sha256: expected_probe_functors_sha256.to_string(),
        input_manifest_sha256: inputs.sha256.clone(),
        inputs: inputs.records.clone(),
        backends: vec![compiled_report, interpreted_report],
    })
}
