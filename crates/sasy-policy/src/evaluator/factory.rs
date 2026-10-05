//! Shared construction of source-backed Soufflé evaluator factories.

use std::path::Path;
use std::sync::Arc;

use sasy_common::Backend;

use super::manager::{EvaluatorProcess, EvaluatorProcessConfig};
use super::Evaluator;
use crate::session_evaluator::EvaluatorFactory;

#[cfg(target_os = "macos")]
const FUNCTOR_LIBRARY_FILENAME: &str = "libfunctors.dylib";
#[cfg(not(target_os = "macos"))]
const FUNCTOR_LIBRARY_FILENAME: &str = "libfunctors.so";

#[cfg(target_os = "macos")]
const FUNCTOR_LIBRARY_CANDIDATES: &[&str] = &[
    "souffle/libfunctors.dylib",
    "souffle/libfunctors.dylib",
    "/opt/homebrew/lib/libfunctors.dylib",
    "/usr/local/lib/libfunctors.dylib",
];
#[cfg(not(target_os = "macos"))]
const FUNCTOR_LIBRARY_CANDIDATES: &[&str] = &[
    "souffle/libfunctors.so",
    "souffle/libfunctors.so",
    "/opt/homebrew/lib/libfunctors.so",
    "/usr/local/lib/libfunctors.so",
];

/// Find the first existing path from a list of candidates.
fn find_path(candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| Path::new(candidate).exists())
        .map(|candidate| candidate.to_string())
}

/// Extra runtime assets for interpreted mode (not in compiler module).
struct InterpretedAssets {
    bin: std::path::PathBuf,
    functor_lib: Option<String>,
    functor_source: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct SouffleLaunchArtifacts {
    pub(crate) executable_path: std::path::PathBuf,
    pub(crate) runtime_executable_path: Option<std::path::PathBuf>,
    pub(crate) support_library_path: Option<std::path::PathBuf>,
}

pub(crate) struct SouffleFactoryBuild {
    pub(crate) factory: EvaluatorFactory,
    pub(crate) backend_name: String,
    pub(crate) launch_artifacts: SouffleLaunchArtifacts,
}

/// Every source-backed asset a Soufflé factory needs, already authenticated by
/// the caller. Passing this in replaces asset discovery outright: no candidate
/// path list, no `PATH` search, no share directory.
#[derive(Clone)]
pub(crate) struct AuthenticatedSouffleAssets {
    pub(crate) compiler: crate::compiler::SouffleAssets,
    pub(crate) interpreted_adapter: std::path::PathBuf,
    pub(crate) interpreted_functors: std::path::PathBuf,
}

impl InterpretedAssets {
    fn discover() -> Result<Self, String> {
        if let Some(runtime) = crate::nix_runtime::configured()? {
            let bin = runtime
                .tool("souffle-interpreted")
                .ok_or("Nix runtime manifest has no pinned interpreted adapter")?
                .to_path_buf();
            let source = crate::compiler::assets_dir()
                .map_err(|error| format!("Find Nix interpreted assets: {error}"))?
                .join("functors.cpp");
            runtime.check_mount(&source)?;
            if !source.is_file() {
                return Err("Nix compiler assets are missing interpreted functors.cpp".into());
            }
            return Ok(Self {
                bin,
                functor_lib: None,
                functor_source: Some(source.to_string_lossy().into_owned()),
            });
        }
        let bin = find_path(&[
            "souffle/souffle-interpreted",
            "souffle/souffle-interpreted",
            "/usr/local/bin/souffle-interpreted",
            "/opt/homebrew/bin/souffle-interpreted",
        ])
        .unwrap_or_else(|| "souffle-interpreted".into());
        Ok(Self {
            // The adapter is a checked-in input, not a gate tool and not a
            // build product. Resolve it here; the runtime receipt
            // authenticates the exact bytes that are run.
            bin: crate::compiler::resolve_path_tool("souffle-interpreted", &bin)
                .map_err(|error| format!("Resolve interpreted adapter: {error}"))?,
            functor_lib: find_path(FUNCTOR_LIBRARY_CANDIDATES),
            // The interpreted backend's own functor source, still searched for
            // on disk. Unlike the compiled backend's assets, which moved
            // inside the binary (see `crate::assets`), this file is not
            // carried: it stays a build-time artifact of
            // `bash souffle/build-test-runtime.sh`, which is what supplies the
            // prebuilt library the first candidate list looks for.
            //
            // Be clear about what that costs. When no prebuilt library is
            // found (or a custom functor is supplied), `ensure_functor_lib`
            // compiles this source with the authenticated C++ compiler and the
            // evaluator dlopens the result, so its object code runs inside the
            // process that makes policy decisions. The first two candidates in
            // both lists are relative, so they resolve against the process's
            // working directory: `--evaluator souffle-interpreted` started
            // from inside a checkout picks up that checkout's `functors.cpp`
            // (or its prebuilt `libfunctors`), which is exactly the
            // wrong-checkout failure the embedded set removed for the compiled
            // backend. Name `--souffle-functor-lib`, or run the authenticated
            // path (`InterpretedAssets::authenticated`), to avoid it.
            functor_source: find_path(&[
                "souffle/functors.cpp",
                "souffle/functors.cpp",
                "/usr/local/share/souffle/functors.cpp",
                "/etc/policy-engine/functors.cpp",
            ]),
        })
    }

    /// The same assets taken from an authenticated set: the adapter is named
    /// outright, and the functor library is always rebuilt from the declared
    /// source rather than picked up prebuilt.
    fn authenticated(assets: &AuthenticatedSouffleAssets) -> Self {
        Self {
            bin: assets.interpreted_adapter.clone(),
            functor_lib: None,
            functor_source: Some(assets.interpreted_functors.to_string_lossy().into_owned()),
        }
    }
}

fn interpreted_functor_ro_binds<'a>(
    assets: &'a crate::compiler::SouffleAssets,
    interpreted_functors: &'a Path,
    custom_functor_path: Option<&'a Path>,
) -> Vec<&'a Path> {
    let mut paths = vec![assets.include_dir.as_path(), interpreted_functors];
    if let Some(custom) = custom_functor_path {
        paths.push(custom);
    }
    paths
}

/// Build an isolated interpreted functor library when no prebuilt one applies.
fn ensure_functor_lib(
    interpreted: &InterpretedAssets,
    assets: &crate::compiler::SouffleAssets,
    work_root: &Path,
    custom_functor_path: Option<&Path>,
) -> Result<std::path::PathBuf, String> {
    if custom_functor_path.is_none() {
        if let Some(ref lib) = interpreted.functor_lib {
            // Materialize the selected bytes under the name passed to -l.
            // Canonicalizing a versioned symlink alone can discard the
            // libfunctors.so/dylib name that the runtime loader searches for.
            let directory = work_root.join(format!("interpreted-library-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&directory)
                .map_err(|error| format!("Create runtime library directory: {error}"))?;
            let library = directory.join(FUNCTOR_LIBRARY_FILENAME);
            std::fs::copy(lib, &library)
                .map_err(|error| format!("Stage interpreted library: {error}"))?;
            return Ok(library);
        }
    }

    let interpreted_functors = interpreted
        .functor_source
        .as_deref()
        .map(Path::new)
        .ok_or("interpreted functors.cpp was not found")?;

    // Custom source is untrusted, and the certification gate must not inherit
    // a library built by an earlier or substituted compiler. Build once in the
    // caller's isolated workspace with the currently authenticated compiler.
    let lib_dir = work_root.join("interpreted_functors");
    std::fs::create_dir_all(&lib_dir)
        .map_err(|error| format!("Create functor lib dir: {error}"))?;
    let library_path = lib_dir.join(FUNCTOR_LIBRARY_FILENAME);

    let ram_domain_def = format!("-DRAM_DOMAIN_SIZE={}", assets.word_size);
    let mut args = vec!["-std=c++17", "-O2", "-shared", "-fPIC", &ram_domain_def];
    let include = format!("-I{}", assets.include_dir.display());
    args.push(&include);
    let interpreted_source = interpreted_functors.to_string_lossy().to_string();
    args.push(&interpreted_source);
    if let Some(custom) = custom_functor_path {
        args.push(custom.to_str().ok_or("custom functor path is not UTF-8")?);
    }
    let library = library_path.to_string_lossy().to_string();
    args.extend_from_slice(&["-o", &library]);

    let read_only = interpreted_functor_ro_binds(assets, interpreted_functors, custom_functor_path);
    let cxx = crate::compiler::selected_tool("cxx", "g++")?;
    let build = crate::sandbox::sandboxed_command(
        cxx.to_str().ok_or("authenticated C++ path is not UTF-8")?,
        &args,
        &lib_dir,
        true,
        &read_only,
    )?;
    let build = crate::compiler::output_bounded(build, "functor compile")
        .map_err(|error| format!("Build libfunctors.so: {error}"))?;
    if !build.status.success() {
        return Err(format!(
            "Functor compilation failed:\n{}",
            String::from_utf8_lossy(&build.stderr)
        ));
    }
    Ok(library_path)
}

/// Build a compiled or interpreted Soufflé evaluator through one pipeline.
pub(crate) fn build_souffle_factory(
    backend: Backend,
    work_root: &Path,
    policy_path: &Path,
    custom_functor_path: Option<&Path>,
) -> Result<(EvaluatorFactory, String), String> {
    let build = build_souffle_factory_with_artifacts(
        backend,
        work_root,
        policy_path,
        custom_functor_path,
        None,
    )?;
    Ok((build.factory, build.backend_name))
}

/// Build a factory, optionally from assets the caller already authenticated.
/// `authenticated` is `None` for production discovery, which keeps the
/// candidate-path search that lets the engine run against a system install.
pub(crate) fn build_souffle_factory_with_artifacts(
    backend: Backend,
    work_root: &Path,
    policy_path: &Path,
    custom_functor_path: Option<&Path>,
    authenticated: Option<&AuthenticatedSouffleAssets>,
) -> Result<SouffleFactoryBuild, String> {
    match backend {
        Backend::SouffleInterpreted => {
            build_interpreted_factory(work_root, policy_path, custom_functor_path, authenticated)
        }
        Backend::Souffle => {
            build_compiled_factory(work_root, policy_path, custom_functor_path, authenticated)
        }
        Backend::Flowlog | Backend::Stub => Err(format!(
            "backend '{}' does not build a Soufflé evaluator",
            backend.as_str()
        )),
    }
}

/// The caller's authenticated compiler assets when given, else discovery.
fn selected_compiler_assets(
    authenticated: Option<&AuthenticatedSouffleAssets>,
) -> Result<crate::compiler::SouffleAssets, String> {
    match authenticated {
        Some(authenticated) => Ok(authenticated.compiler.clone()),
        None => crate::compiler::SouffleAssets::discover()
            .map_err(|error| format!("Discover assets: {error}")),
    }
}

fn build_interpreted_factory(
    work_root: &Path,
    policy_path: &Path,
    custom_functor_path: Option<&Path>,
    authenticated: Option<&AuthenticatedSouffleAssets>,
) -> Result<SouffleFactoryBuild, String> {
    let python = crate::compiler::selected_tool("python3", "python3")
        .map_err(|error| format!("Select Python tool: {error}"))?;
    let interpreted = match authenticated {
        Some(authenticated) => InterpretedAssets::authenticated(authenticated),
        None => InterpretedAssets::discover()?,
    };
    let interpreted_bin = interpreted.bin.clone();
    let souffle = crate::compiler::selected_tool("souffle", "souffle")
        .map_err(|error| format!("Select Soufflé tool: {error}"))?;
    let assets = selected_compiler_assets(authenticated)?;
    let policy_source =
        std::fs::read_to_string(policy_path).map_err(|error| format!("Read policy: {error}"))?;
    let full_source = crate::compiler::prepend_common_policy(&policy_source, &assets)
        .map_err(|error| format!("Prepend common policy: {error}"))?;

    let combined = work_root.join("policy_combined.dl");
    std::fs::write(&combined, &full_source).map_err(|error| format!("Write combined: {error}"))?;
    let desugared = work_root.join("policy_desugared.dl");
    let preprocess = crate::sandbox::sandboxed_command(
        python
            .to_str()
            .ok_or("authenticated Python path is not UTF-8")?,
        &[
            assets.sugar_py.to_str().unwrap(),
            "--resolve-includes",
            combined.to_str().unwrap(),
        ],
        work_root,
        true,
        &[assets.sugar_py.as_path()],
    )?;
    let preprocess = crate::compiler::output_bounded(preprocess, "sugar.py preprocessing")
        .map_err(|error| format!("Preprocess: {error}"))?;
    if !preprocess.status.success() {
        return Err(format!(
            "Sugar preprocessor failed:\n{}",
            String::from_utf8_lossy(&preprocess.stderr)
        ));
    }
    let inferred =
        crate::compiler::apply_policy_inference(&String::from_utf8_lossy(&preprocess.stdout))?;
    std::fs::write(&desugared, inferred).map_err(|error| format!("Write desugared: {error}"))?;

    let support_library_path =
        ensure_functor_lib(&interpreted, &assets, work_root, custom_functor_path)?
            .canonicalize()
            .map_err(|error| format!("Resolve functor library: {error}"))?;
    let desugared = desugared
        .canonicalize()
        .map_err(|error| format!("Resolve desugared policy: {error}"))?;
    // The adapter's directory is not the policy workspace. With bwrap, /tmp
    // is private, so these exact selected files need separate read-only binds.
    // Never admit their parent directories or arbitrary paths from policy argv.
    let runtime_files = vec![
        desugared.clone(),
        support_library_path.clone(),
        souffle.clone(),
    ];
    let lib_dir = support_library_path
        .parent()
        .unwrap_or(Path::new("/tmp"))
        .to_string_lossy()
        .to_string();
    let config = EvaluatorProcessConfig {
        program: interpreted_bin.to_string_lossy().into_owned(),
        args: vec![
            desugared.to_string_lossy().into_owned(),
            souffle
                .to_str()
                .ok_or("authenticated Soufflé path is not UTF-8")?
                .to_string(),
            "functors".to_string(),
            lib_dir.clone(),
        ],
        backend: "souffle-interpreted".to_string(),
        env: vec![
            ("LD_LIBRARY_PATH".into(), lib_dir.clone()),
            ("DYLD_LIBRARY_PATH".into(), lib_dir),
        ],
    };
    Ok(factory_build_from_config(
        config,
        SouffleLaunchArtifacts {
            executable_path: interpreted_bin,
            runtime_executable_path: Some(souffle),
            support_library_path: Some(support_library_path),
        },
        runtime_files,
    ))
}

fn build_compiled_factory(
    work_root: &Path,
    policy_path: &Path,
    custom_functor_path: Option<&Path>,
    authenticated: Option<&AuthenticatedSouffleAssets>,
) -> Result<SouffleFactoryBuild, String> {
    let policy_source =
        std::fs::read_to_string(policy_path).map_err(|error| format!("Read policy: {error}"))?;
    let assets = selected_compiler_assets(authenticated)?;
    let custom_source = custom_functor_path
        .map(std::fs::read_to_string)
        .transpose()
        .map_err(|error| format!("Read custom functors: {error}"))?;
    let result = crate::compiler::compile_souffle_with_assets(
        &policy_source,
        custom_source.as_deref(),
        &work_root.join("souffle_build"),
        &assets,
    )
    .map_err(|error| format!("{error}"))?;
    let config = EvaluatorProcessConfig {
        program: result.binary_path.to_string_lossy().into_owned(),
        args: vec!["policy_program".to_string()],
        backend: "souffle".to_string(),
        env: Vec::new(),
    };
    Ok(factory_build_from_config(
        config,
        SouffleLaunchArtifacts {
            executable_path: result.binary_path,
            runtime_executable_path: None,
            support_library_path: None,
        },
        Vec::new(),
    ))
}

fn factory_build_from_config(
    config: EvaluatorProcessConfig,
    launch_artifacts: SouffleLaunchArtifacts,
    read_only_files: Vec<std::path::PathBuf>,
) -> SouffleFactoryBuild {
    let backend = config.backend.clone();
    let factory: EvaluatorFactory = Arc::new(move || {
        EvaluatorProcess::new_with_read_only_files(config.clone(), read_only_files.clone())
            .map(|evaluator| Arc::new(evaluator) as Arc<dyn Evaluator>)
    });
    SouffleFactoryBuild {
        factory,
        backend_name: backend,
        launch_artifacts,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn test_assets(word_size: u8) -> crate::compiler::SouffleAssets {
        let root = PathBuf::from("/authenticated-assets");
        crate::compiler::SouffleAssets {
            include_dir: root.join("include"),
            sugar_py: root.join("sugar.py"),
            evaluator_shim: root.join("evaluator_shim.cpp"),
            evaluator_protocol: root.join("evaluator_protocol.h"),
            json_string_codec: root.join("json_string_codec.h"),
            functors_common: root.join("functors_common.cpp"),
            common_policy: None,
            exclusive_dir: None,
            word_size,
        }
    }

    #[test]
    fn interpreted_functor_sandbox_binds_interpreted_and_custom_sources() {
        let assets = test_assets(64);
        let interpreted = Path::new("/runtime/functors.cpp");
        let custom = Path::new("/policy/custom_functors.cpp");
        let binds = interpreted_functor_ro_binds(&assets, interpreted, Some(custom));
        assert!(binds.contains(&interpreted));
        assert!(binds.contains(&custom));
    }

    #[cfg(unix)]
    #[test]
    fn versioned_prebuilt_library_keeps_its_loader_name_in_the_workspace() {
        let assets = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let versioned = assets.path().join("versioned-library.1");
        let selected = assets.path().join(FUNCTOR_LIBRARY_FILENAME);
        std::fs::write(&versioned, b"selected library bytes").unwrap();
        std::os::unix::fs::symlink(&versioned, &selected).unwrap();
        let interpreted = InterpretedAssets {
            bin: assets.path().join("adapter"),
            functor_lib: Some(selected.display().to_string()),
            functor_source: None,
        };
        let staged =
            ensure_functor_lib(&interpreted, &test_assets(64), workspace.path(), None).unwrap();
        assert!(staged.starts_with(workspace.path()));
        assert_eq!(staged.file_name().unwrap(), FUNCTOR_LIBRARY_FILENAME);
        assert!(!staged.is_symlink());
        std::fs::write(versioned, b"later replacement").unwrap();
        assert_eq!(std::fs::read(staged).unwrap(), b"selected library bytes");
    }
}
