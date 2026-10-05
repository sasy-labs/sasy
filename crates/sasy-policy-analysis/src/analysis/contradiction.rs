//! Contradiction detection over Allow/Deny rule pairs.
//!
//! The policy language gives the verdict as `Allow` minus
//! `Deny` (modulo authentication). Overlap between an
//! Allow rule's body and a Deny rule's body is therefore
//! *intentional* whenever the deny is meant to carve an
//! exception out of a broader allow — the whole point of
//! denylist semantics. The interesting case for the
//! analyzer is overlap with a *specific* allow rule that
//! the author probably did not intend to also be denied.
//!
//! # Pipeline
//!
//! 1. Locate `IsAuthorized` rules (allows) and the union
//!    of `Unauthorized`, `DenialReason`-derived `Unauthorized`,
//!    and `DenyUnauthorized` rules (denies).
//! 2. For each pair, unfold non-recursive IDBs in both
//!    bodies (recursive IDBs stay opaque).
//! 3. Across the cartesian product of unfolded
//!    alternatives, run the FD chase: assert both bodies
//!    in fresh scopes, unify the heads positionally,
//!    saturate, and observe whether the conjunction is
//!    satisfiable.
//! 4. Classify any satisfiable overlap as expected or
//!    suspicious by the broad/specific rule:
//!    - `@broad` annotation on the allow → expected.
//!    - Subsumption fallback (every allowed action of A
//!      is also allowed by some other allow A' that
//!      overlaps the deny) → expected.
//!    - Otherwise: suspicious finding.
//!
//! Output is a list of [`Finding`]s with source spans
//! pointing at the conflicting rules and a short label
//! for the conflict category.

use std::collections::HashSet;

use super::ast::*;
use super::chase::{Chase, Conflict};
use super::subsumption::rule_subsumes;
use super::unfold::{unfold_body, FreshVars, OpaqueSet};

/// Relations that produce an Allow verdict on `idx`.
const ALLOW_HEADS: &[&str] = &["IsAuthorized"];

/// Relations that produce a Deny verdict on `idx`. We
/// include both the directly-written `Unauthorized` and
/// `DenialReason` (which sugar.py also rewrites to
/// `Unauthorized`); a same-shaped pair is detected
/// regardless of which head the rule uses.
const DENY_HEADS: &[&str] = &["Unauthorized", "DenialReason"];

/// A contradiction or overlap finding.
#[derive(Debug, Clone)]
pub struct Finding {
    pub allow_span: SourceSpan,
    pub deny_span: SourceSpan,
    pub allow_index: usize,
    pub deny_index: usize,
    pub category: Category,
    /// Free-form reason summarizing the witness, useful
    /// for human-readable output.
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Category {
    /// Allow and deny overlap; allow has no `@broad`
    /// annotation and is not subsumed by any broader
    /// allow that also overlaps the deny. Likely an
    /// authoring mistake.
    DenyCrossesSpecificAllow,
    /// Allow body strictly implies deny body (every
    /// action the allow permits is also denied) — the
    /// allow rule is dead under deny semantics.
    DenySwallowsAllow,
    /// Overlap exists but is judged intentional (allow
    /// is annotated `@broad` or is subsumed by another
    /// allow that also overlaps the deny). Recorded for
    /// audit but not flagged as suspicious.
    ExpectedCarveOut,
}

impl Category {
    /// Stable wire string for this category. The values match the
    /// variant names and are the contract consumed by the TypeScript
    /// SDK and the `policy_analyze` output, so they must not change even if a variant is renamed.
    /// Defining them explicitly — rather than Debug-deriving the wire
    /// string — is the point: a future rename forces an intentional
    /// edit here instead of silently shifting the wire value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DenyCrossesSpecificAllow => "DenyCrossesSpecificAllow",
            Self::DenySwallowsAllow => "DenySwallowsAllow",
            Self::ExpectedCarveOut => "ExpectedCarveOut",
        }
    }
}

/// Allow/deny rule annotations relevant to the
/// classification step. The single recognized
/// annotation is `@broad`, which means "this rule's
/// overlap with the other side is intentional, don't
/// flag." Applies on either side:
/// - on an *allow* rule: deny carves from this broad
///   umbrella by design.
/// - on a *deny* rule: this deny is the catch-all that
///   carves from any allow (e.g., common_policy's
///   `Unauthorized :- DenialReason`).
///
/// Callers populate the index sets after parsing
/// annotations via `RuleMetadataParser`.
#[derive(Debug, Clone, Default)]
pub struct AllowAnnotations {
    pub broad_rule_indices: HashSet<usize>,
}

/// Run contradiction detection over the program. Returns
/// findings with source spans, the allow/deny rule
/// indices, and the category. `opaque` lists relations
/// to keep symbolic during unfold (recursive IDBs are
/// added automatically); pass [`OpaqueSet::default()`]
/// for the standard policy.
pub fn analyze(
    program: &Program,
    annotations: &AllowAnnotations,
    extra_opaque: &OpaqueSet,
) -> Vec<Finding> {
    let mut opaque = OpaqueSet::from_program(program);
    for name in &extra_opaque.names {
        opaque.force(name);
    }

    let allow_rules = collect_rule_indices(program, ALLOW_HEADS);
    let deny_rules = collect_rule_indices(program, DENY_HEADS);

    let mut findings = Vec::new();
    for &ai in &allow_rules {
        let allow = &program.rules[ai];
        for &di in &deny_rules {
            let deny = &program.rules[di];
            if let Some(category) = analyze_pair(
                program,
                allow,
                ai,
                deny,
                di,
                &opaque,
                annotations,
                &allow_rules,
            ) {
                let message = describe_pair(allow, deny, &category);
                findings.push(Finding {
                    allow_span: allow.span.clone(),
                    deny_span: deny.span.clone(),
                    allow_index: ai,
                    deny_index: di,
                    category,
                    message,
                });
            }
        }
    }
    findings
}

fn collect_rule_indices(program: &Program, heads: &[&str]) -> Vec<usize> {
    program
        .rules
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            if heads.iter().any(|h| *h == r.head.relation) {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Test a single (allow, deny) pair. Returns the finding
/// category if the bodies overlap (under any unfolded
/// alternative pairing), or `None` if the chase
/// determined every alternative pairing UNSAT.
#[allow(clippy::too_many_arguments)] // inherent: a pair-analysis needs both rules + their indices + context
fn analyze_pair(
    program: &Program,
    allow: &Rule,
    allow_index: usize,
    deny: &Rule,
    deny_index: usize,
    opaque: &OpaqueSet,
    annotations: &AllowAnnotations,
    all_allow_indices: &[usize],
) -> Option<Category> {
    if allow.head.args.len() != deny.head.args.len() {
        // Heads of different arities can't be
        // contradicting — nothing to align.
        return None;
    }
    let mut var_gen = FreshVars::new();
    let allow_alts = unfold_body(&allow.body, program, opaque, &mut var_gen);
    let deny_alts = unfold_body(&deny.body, program, opaque, &mut var_gen);

    let mut any_overlap = false;
    for a_body in &allow_alts {
        for d_body in &deny_alts {
            if let OverlapVerdict::Sat = check_overlap(allow, a_body, deny, d_body) {
                any_overlap = true;
                break;
            }
        }
        if any_overlap {
            break;
        }
    }
    if !any_overlap {
        return None;
    }

    // Broad/specific classification. Two suppression
    // paths are intentional:
    //  1. The author marked the allow `@broad` — the
    //     deny is meant to carve from it, by design.
    //  2. Some other allow rule A' is broader than this
    //     one (this allow is subsumed by A') AND A' also
    //     overlaps the deny. The deny is naturally read
    //     as carving from A'; this allow's overlap is
    //     incidental.
    if annotations.broad_rule_indices.contains(&allow_index) {
        return Some(Category::ExpectedCarveOut);
    }
    // @broad on the deny side: this deny is a catch-all
    // that carves from any allow by design (e.g.,
    // common_policy's `Unauthorized :- DenialReason`).
    if annotations.broad_rule_indices.contains(&deny_index) {
        return Some(Category::ExpectedCarveOut);
    }
    if has_broader_overlapping_allow(program, allow, allow_index, deny, opaque, all_allow_indices) {
        return Some(Category::ExpectedCarveOut);
    }

    Some(Category::DenyCrossesSpecificAllow)
}

/// Subsumption fallback: this allow A is covered by some
/// strictly-broader allow A' that also overlaps the
/// deny. In that case, the deny is interpretable as
/// carving from A', and A's overlap with deny is
/// incidental.
fn has_broader_overlapping_allow(
    program: &Program,
    allow: &Rule,
    allow_index: usize,
    deny: &Rule,
    opaque: &OpaqueSet,
    all_allow_indices: &[usize],
) -> bool {
    for &other_index in all_allow_indices {
        if other_index == allow_index {
            continue;
        }
        let other = &program.rules[other_index];
        if other.head.relation != allow.head.relation {
            continue;
        }
        // A is "broader" than other? Skip.
        // We want: this allow ⊆ other AND NOT other ⊆ this.
        // Strict subsumption — equal rules don't count.
        if !rule_subsumes(allow, other, program, opaque) {
            continue;
        }
        if rule_subsumes(other, allow, program, opaque) {
            // Equivalent rules; not a strict broader cover.
            continue;
        }
        // Other must also overlap the deny — otherwise
        // the deny isn't carving from other, and the
        // overlap with this allow remains the only
        // contact point.
        let mut var_gen = FreshVars::new();
        let other_alts = unfold_body(&other.body, program, opaque, &mut var_gen);
        let deny_alts = unfold_body(&deny.body, program, opaque, &mut var_gen);
        for o_body in &other_alts {
            for d_body in &deny_alts {
                if let OverlapVerdict::Sat = check_overlap(other, o_body, deny, d_body) {
                    return true;
                }
            }
        }
    }
    false
}

enum OverlapVerdict {
    Sat,
    /// UNSAT — kept with the conflict for future
    /// witness-rendering (current callers only check
    /// SAT/UNSAT, but the conflict carries the reason
    /// for diagnostic output once we surface it).
    #[allow(dead_code)]
    Unsat(Conflict),
}

/// Run the FD chase on the conjunction of an unfolded
/// allow alternative and an unfolded deny alternative,
/// with the heads aligned positionally.
fn check_overlap(
    allow: &Rule,
    allow_body: &[Literal],
    deny: &Rule,
    deny_body: &[Literal],
) -> OverlapVerdict {
    let mut chase = Chase::new();
    let av = match chase.assert_body(allow_body) {
        Ok(v) => v,
        Err(c) => return OverlapVerdict::Unsat(c),
    };
    let dv = match chase.assert_body(deny_body) {
        Ok(v) => v,
        Err(c) => return OverlapVerdict::Unsat(c),
    };
    // Align head args: the allow and deny rule heads
    // typically share an `idx` argument. We unify
    // positionally on whatever variables the rule uses.
    for (a_head_arg, d_head_arg) in allow.head.args.iter().zip(deny.head.args.iter()) {
        if let (Some((a_name, _)), Some((d_name, _))) = (var_of(a_head_arg), var_of(d_head_arg)) {
            if let (Some(&at), Some(&dt)) = (av.get(a_name), dv.get(d_name)) {
                if let Err(c) = chase.unify(at, dt) {
                    return OverlapVerdict::Unsat(c);
                }
            }
        }
    }
    match chase.check() {
        Ok(()) => OverlapVerdict::Sat,
        Err(c) => OverlapVerdict::Unsat(c),
    }
}

fn var_of(t: &Term) -> Option<(&str, ())> {
    match t {
        Term::Var(s) => Some((s.as_str(), ())),
        _ => None,
    }
}

fn describe_pair(allow: &Rule, deny: &Rule, category: &Category) -> String {
    match category {
        Category::DenyCrossesSpecificAllow => format!(
            "deny rule {} overlaps specific allow rule {} — likely \
             unintentional carve-out",
            deny.span.as_location(),
            allow.span.as_location()
        ),
        Category::DenySwallowsAllow => format!(
            "deny rule {} fully suppresses allow rule {} — \
             allow is dead under deny semantics",
            deny.span.as_location(),
            allow.span.as_location()
        ),
        Category::ExpectedCarveOut => format!(
            "deny rule {} carves from broad allow rule {} — recorded as expected",
            deny.span.as_location(),
            allow.span.as_location()
        ),
    }
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    fn analyze_src(src: &str) -> Vec<Finding> {
        let prog = parse(src, "test.dl").unwrap();
        let ann = AllowAnnotations::default();
        let extra = OpaqueSet::default();
        analyze(&prog, &ann, &extra)
    }

    #[test]
    fn raw_constructor_contradiction_flagged() {
        // Allow and deny with concrete tool names that
        // can't both be true for the same action.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("submit", w).
                Unauthorized(idx) :- Actions(idx, a), a = $CallTool("delete", w).
            "#,
        );
        // No overlap — Actions FD merges the action terms,
        // injectivity merges constants → UNSAT. So no finding.
        assert!(findings.is_empty(), "got: {:?}", findings);
    }

    #[test]
    fn overlapping_specific_allow_flagged() {
        // Allow tool=submit + role=admin; deny tool=submit
        // unconditional. They overlap on (submit, admin)
        // actions. Allow is specific (it pins a tool),
        // deny is a carve. Without @broad, this is flagged.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :-
                    Actions(idx, a),
                    a = $CallTool("submit", w),
                    HasRole("admin").
                Unauthorized(idx) :-
                    Actions(idx, a),
                    a = $CallTool("submit", w).
            "#,
        );
        assert_eq!(findings.len(), 1, "got: {:?}", findings);
        assert_eq!(findings[0].category, Category::DenyCrossesSpecificAllow);
    }

    #[test]
    fn broad_annotation_suppresses_finding() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, _).
                Unauthorized(idx) :- Actions(idx, a), a = $CallTool("dangerous", _).
            "#,
            "test.dl",
        )
        .unwrap();
        let allow_idx = prog
            .rules
            .iter()
            .position(|r| r.head.relation == "IsAuthorized")
            .unwrap();
        let mut ann = AllowAnnotations::default();
        ann.broad_rule_indices.insert(allow_idx);
        let findings = analyze(&prog, &ann, &OpaqueSet::default());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].category, Category::ExpectedCarveOut);
    }

    #[test]
    fn idb_unfolds_to_expose_constructor_conflict() {
        // Without unfolding IsTool, the analyzer would see
        // two opaque atoms IsTool(a, "X") and IsTool(a, "Y")
        // and judge them satisfiable. Unfolding pulls in
        // the underlying $CallTool pattern, which the FD
        // chase + injectivity catches.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).
                IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "submit").
                Unauthorized(idx) :- Actions(idx, a), IsTool(a, "delete").
            "#,
        );
        assert!(
            findings.is_empty(),
            "expected unfolding to expose UNSAT, got {:?}",
            findings
        );
    }

    #[test]
    fn distinct_tool_allow_and_deny_no_overlap() {
        // Sanity: two completely different tool names
        // give no overlap whatsoever.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("read", w).
                Unauthorized(idx) :- Actions(idx, a), a = $CallTool("write", w).
            "#,
        );
        assert!(findings.is_empty(), "got: {:?}", findings);
    }

    /// End-to-end: run the analyzer on every real policy
    /// in the repo, after running it through the production
    /// preprocessor. Reports findings to stderr; the test
    /// asserts the analyzer terminates cleanly on every
    /// policy listed. Only a missing python3 skips it.
    #[test]
    fn analyzes_repo_policies() {
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
            // Strip `#include "common_policy.dl"` lines —
            // we prepend common_policy ourselves, sugar.py
            // doesn't expand includes, and Soufflé would
            // see two copies otherwise.
            let stripped: String = user_src
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !(t.starts_with("#include") && t.contains("common_policy"))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let combined = format!("{}\n{}", common_src, stripped);
            let tmp = std::env::temp_dir().join(format!(
                "sasy-contra-smoke-{}.dl",
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
                "sugar.py failed for {path}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let desugared = String::from_utf8(out.stdout).unwrap();
            let prog = match parse(&desugared, *path) {
                Ok(p) => p,
                Err(e) => panic!("parse {path}: {e}"),
            };
            let findings = analyze(&prog, &AllowAnnotations::default(), &OpaqueSet::default());
            eprintln!("\n=== {} — {} finding(s) ===", path, findings.len());
            for f in &findings {
                eprintln!(
                    "  [{:?}] allow@{} vs deny@{}",
                    f.category,
                    f.allow_span.as_location(),
                    f.deny_span.as_location()
                );
            }
            analysed += 1;
        }
        assert!(analysed > 0, "no policy was analysed");
    }

    #[test]
    fn two_principals_for_one_edge_cannot_both_hold() {
        // One edge carries at most one principal, so an allow that
        // wants the gateway to have asserted it and a deny that wants
        // the intern to have asserted it never hold together. Without
        // the EdgePrincipal functional dependency the analyzer takes
        // both rows as possible at once and reports the deny as
        // crossing the allow.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                .decl EdgePrincipal(source_id: symbol, dest_id: symbol, principal: symbol)
                .input EdgePrincipal
                IsAuthorized(idx) :-
                    Actions(idx, _), Current(d), EdgePrincipal("m1", d, "gateway").
                Unauthorized(idx) :-
                    Actions(idx, _), Current(d), EdgePrincipal("m1", d, "intern").
            "#,
        );
        assert!(findings.is_empty(), "got: {findings:?}");
    }

    #[test]
    fn two_entities_for_one_edge_cannot_both_hold() {
        // The same for the client-supplied entity: one row per edge.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                .decl EdgeEntity(source_id: symbol, dest_id: symbol, entity: symbol)
                .input EdgeEntity
                IsAuthorized(idx) :-
                    Actions(idx, _), Current(d), EdgeEntity("m1", d, "staff").
                Unauthorized(idx) :-
                    Actions(idx, _), Current(d), EdgeEntity("m1", d, "contractor").
            "#,
        );
        assert!(findings.is_empty(), "got: {findings:?}");
    }

    #[test]
    fn different_edges_still_take_different_principals() {
        // The dependency keys on the edge, not on the principal
        // column alone: two rules that name different edges stay
        // satisfiable together, so their overlap is still reported.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                .decl EdgePrincipal(source_id: symbol, dest_id: symbol, principal: symbol)
                .input EdgePrincipal
                IsAuthorized(idx) :-
                    Actions(idx, _), Current(d), EdgePrincipal("m1", d, "gateway").
                Unauthorized(idx) :-
                    Actions(idx, _), Current(d), EdgePrincipal("m9", d, "intern").
            "#,
        );
        assert_eq!(findings.len(), 1, "got: {findings:?}");
    }

    #[test]
    fn recursive_idb_stays_opaque() {
        // Supervises is recursive; the analyzer keeps it
        // opaque, so an overlap mediated only by the IDB
        // is conservatively reported.
        let findings = analyze_src(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                Supervises(s, e) :- Manages(s, e).
                Supervises(s, e) :- Manages(s, m), Supervises(m, e).
                IsAuthorized(idx) :- Actions(idx, _), Supervises("alice", "bob").
                Unauthorized(idx) :- Actions(idx, _), Supervises("alice", "bob").
            "#,
        );
        assert_eq!(findings.len(), 1, "got: {:?}", findings);
    }
}
