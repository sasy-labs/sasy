//! Release corpus gate: prerequisites and current public policies are required.
//! Run with `cargo test -p sasy-policy-analysis --test public_policy_corpus`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sasy_policy_analysis::analysis::{
    parse, parse_with_options,
    pipeline::{analyze_raw, AnalyzeError},
    resolve_dots::resolve_dots,
    ParseOptions,
};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn combined(root: &Path, path: &str, common: &str) -> String {
    let source = std::fs::read_to_string(root.join(path)).expect("required public policy");
    if path == "policies/common_policy.dl" || path == "souffle/common_policy.dl" {
        return source;
    }
    let user = source
        .lines()
        .filter(|line| {
            if line.trim_start().starts_with("#include") {
                assert!(
                    line.contains("common_policy.dl"),
                    "corpus needs an explicit include fixture: {line}"
                );
                false
            } else {
                true
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{common}\n// === USER_POLICY_BEGIN ===\n{user}")
}

fn corpus_worker() {
    let root = repo();
    let selected: Vec<String> =
        serde_json::from_str(include_str!("fixtures/public-policy-corpus.json")).unwrap();
    let policies: BTreeSet<&str> = selected.iter().map(String::as_str).collect();
    assert_eq!(policies.len(), selected.len(), "duplicate corpus entry");
    let required = "souffle/common_policy.dl";
    assert!(
        policies.contains(required),
        "public corpus omitted {required}"
    );
    let common = std::fs::read_to_string(root.join(required)).unwrap();
    for path in &policies {
        let raw = combined(&root, path, &common);
        let output = Command::new("python3")
            .arg(root.join("souffle/sugar.py"))
            .arg("--resolve-includes")
            .arg(root.join(path))
            .env_remove("COMMON_POLICY_DL")
            .output()
            .expect("Python 3 is required for the release corpus gate");
        assert!(
            output.status.success(),
            "preprocess {path}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cooked = String::from_utf8(output.stdout).unwrap();
        for (mode, source) in [("raw", &raw), ("python", &cooked)] {
            let mut program =
                parse(source, *path).unwrap_or_else(|e| panic!("{path} ({mode}): {e}"));
            resolve_dots(&mut program).unwrap_or_else(|e| panic!("{path} ({mode}) dots: {e}"));
            parse_with_options(
                source,
                *path,
                ParseOptions {
                    expand_body_disjunctions: false,
                    reject_duplicate_relations: true,
                    full_rule_spans: true,
                    preserve_aggregates: true,
                },
            )
            .unwrap_or_else(|e| panic!("{path} ({mode}) bridge: {e}"));
            println!("{path} ({mode}): {} rules", program.rules.len());
        }
    }
    let builtin_policy = ".decl Length(n:number)\nLength(n) :- n = strlen(\"example\").";
    assert!(matches!(
        analyze_raw(builtin_policy, "builtin-limit.dl", &[]),
        Err(AnalyzeError::UnsupportedBuiltin { name, .. }) if name == "strlen"
    ), "builtin syntax is supported; full analysis must remain explicitly unavailable until budgeted");
    println!("builtin analysis: explicit UnsupportedBuiltin (no unbounded analysis)");
}

#[test]
fn current_public_corpus_parses_and_builtin_analysis_is_bounded() {
    const WORKER: &str = "SASY_PUBLIC_CORPUS_TEST_WORKER";
    if std::env::var_os(WORKER).is_some() {
        corpus_worker();
        return;
    }
    // The analysis API is synchronous. An isolated test process makes the
    // timeout enforceable even if a parser/analysis regression never yields.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "current_public_corpus_parses_and_builtin_analysis_is_bounded",
            "--nocapture",
        ])
        .env(WORKER, "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success(),
                "public corpus/analysis worker failed: {status}"
            );
            return;
        }
        if Instant::now() >= deadline {
            child.kill().expect("stop over-budget analysis worker");
            child.wait().expect("reap analysis worker");
            panic!("public corpus and builtin analysis exceeded 60 seconds");
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}
