//! Top-level entry point bundling all four policy
//! analyses into one [`AnalysisReport`].
//!
//! The expected pipeline (preprocessing happens upstream
//! and is shared with `ValidatePolicy`):
//!
//! ```text
//!   raw .dl source
//!     → prepend common_policy.dl
//!     → sugar.py    (runs in the existing service)
//!     → analyze_desugared()  ← this module
//! ```
//!
//! `analyze_desugared` parses the desugared text, scans
//! for `@broad` rule annotations using
//! [`crate::rule_metadata::RuleMetadataParser`], and runs
//! the four analyses (contradiction, redundancy,
//! subsumption-driven, reachability) on the resulting
//! AST. The report carries findings with source spans
//! that line up with the desugared text.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::ast::{format_rule, Literal, Program, Term};
use super::contradiction::{self, AllowAnnotations};
use super::reachability::{self, ReachabilityResult};
use super::resolve_dots::{resolve_dots, DotResolveError};
use super::rewrite_annotations::rewrite_deny_annotations;
use super::subsumption;
use super::unfold::OpaqueSet;
use super::ParseError;
use crate::rule_metadata::RuleMetadataParser;

/// Bundle of findings from all analyses on one policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub contradictions: Vec<ContradictionView>,
    pub redundancies: Vec<RedundancyView>,
    pub reachability: Vec<ReachabilityView>,
    /// Names of rules the analyzer identified as `@broad`
    /// (used to suppress contradiction findings).
    pub broad_rules: Vec<String>,
}

/// Wire-friendly contradiction finding (string fields
/// only — no `Term` AST in transit). The rule bodies are
/// pre-formatted so the auditor can read the conflicting
/// rules inline without cross-referencing the source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionView {
    pub category: String,
    pub allow_location: String,
    pub deny_location: String,
    pub message: String,
    pub allow_body: String,
    pub deny_body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyView {
    pub head_relation: String,
    pub redundant_location: String,
    pub covered_by_location: String,
    pub redundant_body: String,
    pub covered_by_body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityView {
    pub target: String,
    pub disjuncts: Vec<String>,
    pub opaque: Vec<String>,
    pub pruned: usize,
}

/// Targets to run reachability for by default. Callers
/// can extend the list via `extra_reach_targets`.
const DEFAULT_REACH_TARGETS: &[&str] = &[
    "IsAuthorized",
    "Unauthorized",
    "Authorized",
    "DenyUnauthorized",
    "AllowPassthrough",
];

/// Errors from the AST-level pipeline (`analyze_raw`).
#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error("parse: {0}")]
    Parse(#[from] ParseError),
    #[error("resolve dots: {0}")]
    Resolve(#[from] DotResolveError),
    #[error("full static analysis is unavailable for builtin `{name}` at {location}; parsing and runtime evaluation remain supported")]
    UnsupportedBuiltin { name: String, location: String },
}

/// Run the full analysis pipeline on a *post-sugar* `.dl`
/// source. The `file_path` is used only for source-span
/// reporting on findings.
pub fn analyze_desugared(
    desugared: &str,
    file_path: &str,
    extra_reach_targets: &[&str],
) -> Result<AnalysisReport, AnalyzeError> {
    let prog = super::parse(desugared, file_path)?;
    ensure_full_analysis_supported(&prog)?;
    Ok(analyze_program(
        &prog,
        desugared,
        file_path,
        extra_reach_targets,
    ))
}

// Accepting syntax must not silently admit previously rejected programs to
// unbounded whole-program analysis. The current guard policy takes more than
// 60 seconds in that pipeline. Keep full analysis explicitly unavailable for
// the newly parsed builtin subset until it has enforceable work budgets;
// parser/dot visitors and directly invoked small analyses remain usable.
fn ensure_full_analysis_supported(program: &Program) -> Result<(), AnalyzeError> {
    fn in_term(term: &Term) -> Option<&str> {
        match term {
            Term::Builtin { name, .. } => Some(name),
            Term::Constructor { args, .. } | Term::Functor { args, .. } | Term::RecordLit(args) => {
                args.iter().find_map(in_term)
            }
            Term::Arith { left, right, .. } => in_term(left).or_else(|| in_term(right)),
            _ => None,
        }
    }
    fn in_literal(literal: &Literal) -> Option<&str> {
        match literal {
            Literal::Pos(atom) | Literal::Neg(atom) => atom.args.iter().find_map(in_term),
            Literal::Compare { left, right, .. } => in_term(left).or_else(|| in_term(right)),
            Literal::Negation { literal, .. } => in_literal(literal),
            Literal::Disjunction { alternatives, .. } => {
                alternatives.iter().flatten().find_map(in_literal)
            }
        }
    }
    for rule in &program.rules {
        if let Some(name) = rule
            .head
            .args
            .iter()
            .find_map(in_term)
            .or_else(|| rule.body.iter().find_map(in_literal))
        {
            return Err(AnalyzeError::UnsupportedBuiltin {
                name: name.to_string(),
                location: rule.span.as_location(),
            });
        }
    }
    for fact in &program.facts {
        if let Some(name) = fact.atom.args.iter().find_map(in_term) {
            return Err(AnalyzeError::UnsupportedBuiltin {
                name: name.to_string(),
                location: fact.atom.span.as_location(),
            });
        }
    }
    Ok(())
}

/// The shared analysis tail behind [`analyze_desugared`] and [`analyze_raw`]:
/// broad-`@allow` detection, contradiction, redundancy, and reachability over
/// an already-parsed program. `source` is the text `prog` was parsed from
/// (used to locate `@broad` annotation spans). Infallible — the only fallible
/// steps (parse, dot-resolution) happen in the callers before this point.
fn analyze_program(
    prog: &Program,
    source: &str,
    file_path: &str,
    extra_reach_targets: &[&str],
) -> AnalysisReport {
    let broad_indices = collect_broad_annotations(source, file_path, prog);
    let broad_names = broad_rule_names(prog, &broad_indices);

    let annotations = AllowAnnotations {
        broad_rule_indices: broad_indices,
    };
    let opaque = OpaqueSet::default();

    let contradictions = contradiction::analyze(prog, &annotations, &opaque)
        .into_iter()
        .map(|f| ContradictionView {
            category: f.category.as_str().to_string(),
            allow_location: f.allow_span.as_location(),
            deny_location: f.deny_span.as_location(),
            message: f.message,
            allow_body: format_rule(&prog.rules[f.allow_index]),
            deny_body: format_rule(&prog.rules[f.deny_index]),
        })
        .collect();

    let redundancies = subsumption::find_redundancies(prog, &opaque)
        .into_iter()
        .map(|r| RedundancyView {
            head_relation: r.head_relation,
            redundant_location: r.redundant_span.as_location(),
            covered_by_location: r.covered_by_span.as_location(),
            redundant_body: format_rule(&prog.rules[r.redundant_index]),
            covered_by_body: format_rule(&prog.rules[r.covered_by_index]),
        })
        .collect();

    let mut targets: Vec<&str> = DEFAULT_REACH_TARGETS.to_vec();
    for t in extra_reach_targets {
        if !targets.contains(t) {
            targets.push(t);
        }
    }
    // Only run reachability on targets the program
    // actually defines, to avoid empty disjunct lists in
    // the report.
    let defined_heads: HashSet<&str> = prog
        .rules
        .iter()
        .map(|r| r.head.relation.as_str())
        .collect();
    let reachability: Vec<ReachabilityView> = targets
        .into_iter()
        .filter(|t| defined_heads.contains(t))
        .map(|t| reach_view(prog, t, &opaque))
        .collect();

    AnalysisReport {
        contradictions,
        redundancies,
        reachability,
        broad_rules: broad_names,
    }
}

/// Run the analysis pipeline on a *raw* (sugar-bearing)
/// `.dl` source. Builtin-containing programs return an explicit unsupported
/// error until full analysis has enforceable work budgets. Parses the user's source directly,
/// applies AST-level dot-resolution
/// ([`super::resolve_dots`]), then runs the four
/// analyses. This path keeps source spans on the user's
/// own line numbers — no desugared-text indirection.
///
/// Callers are responsible for prepending common_policy
/// and stripping any `#include "common_policy.dl"`
/// lines from the user source before passing the
/// combined text in.
pub fn analyze_raw(
    source: &str,
    file_path: &str,
    extra_reach_targets: &[&str],
) -> Result<AnalysisReport, AnalyzeError> {
    let mut prog = super::parse(source, file_path)?;
    ensure_full_analysis_supported(&prog)?;
    resolve_dots(&mut prog)?;
    // Match sugar.py's behavior: `@deny_message`-tagged
    // Unauthorized rules become DenialReason rules so
    // analyses see one shape for denies regardless of
    // which path produced the program.
    rewrite_deny_annotations(&mut prog, source, file_path);
    Ok(analyze_program(
        &prog,
        source,
        file_path,
        extra_reach_targets,
    ))
}

fn reach_view(prog: &Program, target: &str, opaque: &OpaqueSet) -> ReachabilityView {
    let r: ReachabilityResult = reachability::reach(prog, target, opaque);
    let disjuncts = r.clauses.iter().map(|c| format!("{}", c)).collect();
    let mut opaque_names: Vec<String> = r.opaque.into_iter().collect();
    opaque_names.sort();
    ReachabilityView {
        target: r.target,
        disjuncts,
        opaque: opaque_names,
        pruned: r.pruned,
    }
}

/// Walk the `@broad` annotations in the source and map
/// them to rule indices in the parsed program. Annotations
/// are matched to rules by source line: an annotation on
/// line N attaches to the next rule whose span starts at
/// N or later.
fn collect_broad_annotations(source: &str, file_path: &str, prog: &Program) -> HashSet<usize> {
    let parser = RuleMetadataParser::parse_content(source, file_path.to_string());
    let mut out = HashSet::new();
    for (i, rule) in prog.rules.iter().enumerate() {
        if let Some(meta) = parser.get_metadata(rule.span.start_line) {
            if meta.custom.contains_key("broad") {
                out.insert(i);
            }
        }
    }
    out
}

fn broad_rule_names(prog: &Program, indices: &HashSet<usize>) -> Vec<String> {
    let mut names: Vec<String> = indices
        .iter()
        .map(|&i| {
            format!(
                "{}@{}",
                prog.rules[i].head.relation,
                prog.rules[i].span.as_location()
            )
        })
        .collect();
    names.sort();
    names
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_includes_all_analyses() {
        let src = r#"
            .decl Actions(idx: unsigned, a: symbol)
            .input Actions
            IsAuthorized(idx) :- Actions(idx, _).
            IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("submit", _).
            Unauthorized(idx) :- Actions(idx, a), a = $CallTool("dangerous", _).
        "#;
        let report = analyze_desugared(src, "test.dl", &[]).unwrap();
        // Redundancy: rule 2 is subsumed by rule 1.
        assert!(
            report
                .redundancies
                .iter()
                .any(|r| r.head_relation == "IsAuthorized"),
            "got {:?}",
            report.redundancies
        );
        // Reachability: at least IsAuthorized is reachable.
        assert!(
            report
                .reachability
                .iter()
                .any(|r| r.target == "IsAuthorized" && !r.disjuncts.is_empty()),
            "got {:?}",
            report.reachability
        );
        // Contradiction: at least one finding (broad allow
        // overlapping the deny).
        assert!(
            !report.contradictions.is_empty(),
            "got {:?}",
            report.contradictions
        );
    }

    #[test]
    fn broad_annotation_recognized_from_source() {
        let src = r#"
.decl Actions(idx: unsigned, a: symbol)
.input Actions
// @broad: default-allow umbrella for tool calls
IsAuthorized(idx) :- Actions(idx, _).
Unauthorized(idx) :- Actions(idx, a), a = $CallTool("dangerous", _).
"#;
        let report = analyze_desugared(src, "test.dl", &[]).unwrap();
        assert_eq!(report.broad_rules.len(), 1, "got {:?}", report.broad_rules);
        // The contradiction finding should be classified
        // as ExpectedCarveOut, not DenyCrossesSpecificAllow,
        // because the allow is marked @broad.
        for f in &report.contradictions {
            assert!(
                f.category == "ExpectedCarveOut",
                "expected ExpectedCarveOut, got {:?}",
                f
            );
        }
    }

    /// Equivalence check: for each repo policy, run
    /// both the AST-level pipeline (`analyze_raw`) and
    /// the sugar.py-then-desugared pipeline
    /// (`analyze_desugared`) and verify the analyses
    /// produce the same per-relation coverage. We compare
    /// counts (contradictions, redundancies) and
    /// reachability shapes (targets + disjunct counts +
    /// opaque sets), because exact source locations
    /// differ — by design — between the two paths.
    #[test]
    fn ast_pipeline_matches_sugar_pipeline() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let sugar = format!("{manifest}/../../souffle/sugar.py");
        assert!(
            std::path::Path::new(&sugar).exists(),
            "the desugaring preprocessor is missing: {sugar}"
        );
        let common_path = format!("{manifest}/../../policies/common_policy.dl");
        let common_src = std::fs::read_to_string(&common_path)
            .unwrap_or_else(|e| panic!("common policy missing at {common_path}: {e}"));
        let candidates = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/adk-separation-of-duties/payment_policy.dl"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/langchain-information-flow/policy.dl"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/message-flow/policy.dl"
            ),
        ];

        let mut analysed = 0usize;
        for path in &candidates {
            assert!(
                std::path::Path::new(path).exists(),
                "example policy missing from the repository: {path}"
            );
            let user_src = std::fs::read_to_string(path).unwrap();
            let stripped: String = user_src
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !(t.starts_with("#include") && t.contains("common_policy"))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let combined = format!("{}\n{}", common_src, stripped);

            // AST pipeline: parse raw + resolve dots.
            let ast_report = match analyze_raw(&combined, path, &[]) {
                Ok(r) => r,
                Err(e) => panic!("analyze_raw {}: {}", path, e),
            };

            // Sugar pipeline: run sugar.py, parse desugared.
            let tmp = std::env::temp_dir().join(format!(
                "sasy-equiv-{}.dl",
                std::path::Path::new(path)
                    .parent()
                    .and_then(|d| d.file_name())
                    .map(|d| d.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ));
            std::fs::write(&tmp, &combined).unwrap();
            let out = match std::process::Command::new("python3")
                .arg(&sugar)
                .arg(&tmp)
                .output()
            {
                Ok(o) => o,
                Err(_) => return,
            };
            assert!(
                out.status.success(),
                "sugar.py failed for {}: {}",
                path,
                String::from_utf8_lossy(&out.stderr)
            );
            let desugared = String::from_utf8(out.stdout).unwrap();
            let sugar_report = analyze_desugared(&desugared, path, &[]).unwrap();

            // Compare counts. Locations differ (one
            // points at user source, one at desugared)
            // but the analyses should observe the same
            // structure.
            assert_eq!(
                ast_report.contradictions.len(),
                sugar_report.contradictions.len(),
                "contradiction count mismatch for {}: ast {} vs sugar {}",
                path,
                ast_report.contradictions.len(),
                sugar_report.contradictions.len()
            );
            assert_eq!(
                ast_report.redundancies.len(),
                sugar_report.redundancies.len(),
                "redundancy count mismatch for {}: ast {} vs sugar {}",
                path,
                ast_report.redundancies.len(),
                sugar_report.redundancies.len()
            );

            // Reachability: same set of targets, same
            // disjunct counts per target, same opaque
            // sets.
            let ast_targets: std::collections::HashMap<&str, &ReachabilityView> = ast_report
                .reachability
                .iter()
                .map(|r| (r.target.as_str(), r))
                .collect();
            for sugar_r in &sugar_report.reachability {
                let ast_r = ast_targets.get(sugar_r.target.as_str()).unwrap_or_else(|| {
                    panic!(
                        "AST pipeline missing target {} for {}",
                        sugar_r.target, path
                    )
                });
                assert_eq!(
                    ast_r.disjuncts.len(),
                    sugar_r.disjuncts.len(),
                    "{} :: {} disjunct count: ast {} vs sugar {}",
                    path,
                    sugar_r.target,
                    ast_r.disjuncts.len(),
                    sugar_r.disjuncts.len()
                );
                assert_eq!(
                    ast_r.opaque, sugar_r.opaque,
                    "{} :: {} opaque set differs",
                    path, sugar_r.target
                );
            }
            eprintln!(
                "OK {}: {} contradictions, {} redundancies, {} reach targets",
                std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy(),
                ast_report.contradictions.len(),
                ast_report.redundancies.len(),
                ast_report.reachability.len()
            );
            analysed += 1;
        }
        assert!(analysed > 0, "no policy was analysed");
    }

    #[test]
    fn skips_undefined_reachability_targets() {
        let src = r#"
            .decl Actions(idx: unsigned, a: symbol)
            .input Actions
            IsAuthorized(idx) :- Actions(idx, _).
        "#;
        let report = analyze_desugared(src, "test.dl", &[]).unwrap();
        // Only IsAuthorized has rules; Unauthorized,
        // Authorized, DenyUnauthorized, AllowPassthrough
        // shouldn't appear.
        let targets: Vec<&str> = report
            .reachability
            .iter()
            .map(|r| r.target.as_str())
            .collect();
        assert_eq!(targets, vec!["IsAuthorized"]);
    }
}
