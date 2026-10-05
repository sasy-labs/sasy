//! Build-time tool: precompile curated policies into an evaluator pack.
//!
//! For each curated profile it runs the normal Soufflé compile pipeline
//! (sugar.py → souffle -g → g++) ONCE, at build time, and emits a
//! self-contained evaluator ELF plus the metadata the restricted binary needs
//! to pre-install the profile without ever invoking the toolchain at runtime:
//!
//!   <out>/<profile>.evaluator   the compiled ELF (spawned via IPC at runtime)
//!   <out>/<profile>.dl          raw source (rule metadata + content-hash input)
//!   <out>/manifest.json         [{name, content_hash, program, default}]
//!
//! The `content_hash` MUST match what `PolicyService::set_policy` computes for
//! the same source (service.rs), so a client's `SetPolicy(session, <source>)`
//! dedup-hits the pre-installed entry. Both call the one formula in
//! `sasy_policy::hash::upload_content_hash` (raw source, backend "souffle",
//! magic from env, empty functor source), so the two cannot drift apart.
//!
//! Usage: pack-policies <out-dir> <default-profile> <profile.dl>...

use std::path::{Path, PathBuf};

fn content_hash(source: &str) -> String {
    // The same function crates/sasy-policy/src/service.rs set_policy calls:
    // backend resolves to the engine's "souffle"; functor source is empty for
    // the curated profiles; magic from SASY_SOUFFLE_MAGIC_SET (default "").
    let magic = std::env::var("SASY_SOUFFLE_MAGIC_SET").unwrap_or_default();
    sasy_policy::hash::upload_content_hash("souffle", &magic, source, "")
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: pack-policies <out-dir> <default-profile> <profile.dl>...");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[1]);
    let default_profile = args[2].clone();
    let profile_paths: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();

    std::fs::create_dir_all(&out_dir)?;
    let build_dir = out_dir.join("build");
    std::fs::create_dir_all(&build_dir)?;

    let assets = sasy_policy::compiler::SouffleAssets::discover().map_err(|e| {
        anyhow::anyhow!("Soufflé build assets not found (toolchain required at BUILD time): {e}")
    })?;

    let mut manifest = Vec::new();
    let mut saw_default = false;

    for path in &profile_paths {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("bad profile path: {}", path.display()))?
            .to_string();
        let source = std::fs::read_to_string(path)?;
        let hash = content_hash(&source);

        eprintln!("[pack] compiling {name} ...");
        let functors = sasy_policy::compiler::find_functors(path);
        let result = sasy_policy::compiler::compile_souffle_from_file(
            path,
            functors.as_deref(),
            &build_dir,
            &assets,
        )
        .map_err(|e| anyhow::anyhow!("compile {name}: {e}"))?;

        let elf_dest = out_dir.join(format!("{name}.evaluator"));
        std::fs::copy(&result.binary_path, &elf_dest)?;
        std::fs::write(out_dir.join(format!("{name}.dl")), &source)?;

        let is_default = name == default_profile;
        saw_default |= is_default;
        manifest.push(serde_json::json!({
            "name": name,
            "content_hash": hash,
            "program": "policy_program",
            "evaluator": format!("{name}.evaluator"),
            "source": format!("{name}.dl"),
            "default": is_default,
        }));
        eprintln!("[pack]   -> {} (hash {})", elf_dest.display(), &hash[..16]);
    }

    if !saw_default {
        anyhow::bail!("default profile '{default_profile}' not among the supplied profiles");
    }

    let manifest_json = serde_json::to_string_pretty(&serde_json::json!({
        "version": 1,
        "default": default_profile,
        "profiles": manifest,
    }))?;
    std::fs::write(out_dir.join("manifest.json"), manifest_json)?;
    // The build dir is only scratch; leave it for cache reuse but note it.
    let _ = Path::new(&build_dir);
    eprintln!("[pack] wrote {}/manifest.json", out_dir.display());
    Ok(())
}
