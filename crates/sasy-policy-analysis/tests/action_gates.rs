use sasy_policy_analysis::action_gates::{infer_action_gates, ActionGateTransform, GateDecision};
const COMMON: &str = r#"
.type Action = CallTool { name: symbol, args: symbol } | SendAttempt { message: symbol } | HTTPRequest { url: symbol, body: symbol, headers: symbol }
.decl Actions(i:unsigned, a:Action)
.input Actions
.decl Edge(a:symbol,b:symbol)
.input Edge
.decl CurrentDependsPolicyRelevant()
.decl ReachableFromPolicyRelevant()
.decl CurrentDepends(x:symbol)
.output CurrentDepends
.decl ReachableFrom(x:symbol,y:symbol)
.output ReachableFrom
CurrentDepends(x) :- CurrentDependsPolicyRelevant(), Edge(x,_).
CurrentDepends(x) :- CurrentDepends(y), Edge(x,y).
ReachableFrom(x,y) :- ReachableFromPolicyRelevant(), CurrentDepends(x), Edge(x,y).
ReachableFrom(x,y) :- ReachableFrom(x,z), Edge(z,y).
CurrentDependsPolicyRelevant() :- ReachableFromPolicyRelevant().
.decl IsTool(a:Action,n:symbol)
IsTool(a,n) :- Actions(_,a), a = $CallTool(n,_).
.decl Unauthorized(i:unsigned)
// === USER_POLICY_BEGIN ===
"#;
fn source(user: &str) -> String {
    format!("{COMMON}{user}\n// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\nCurrentDependsPolicyRelevant().\n// SASY_AUTO_GATE_DEFAULT: ReachableFromPolicyRelevant\nReachableFromPolicyRelevant().\n")
}
fn run(user: &str) -> ActionGateTransform {
    let s = source(user);
    infer_action_gates(&s, &s)
}
fn current(out: &ActionGateTransform) -> &GateDecision {
    &out.report.gates[0]
}
#[test]
fn direct_action_binding_preserves_source() {
    let user="// exact source spacing\nUnauthorized(i) :- Actions(i, a), a = $CallTool(\"Write\", _), CurrentDepends(x).";
    let out = run(user);
    assert_eq!(out.report.fallback, None);
    assert_eq!(current(&out).action_patterns, ["$CallTool(\"Write\", _)"]);
    assert!(out.source.contains(user));
    assert!(out
        .source
        .contains("CurrentDependsPolicyRelevant() :- Actions(_, $CallTool(\"Write\", _))."));
    assert_eq!(out.report.gates[1].outcome, "unused");
}
#[test]
fn unrelated_constructor_is_not_an_action_binding() {
    assert_eq!(
        current(&run(
            "Unauthorized(i) :- Actions(i,a), b=$CallTool(\"Write\",_), CurrentDepends(x)."
        ))
        .outcome,
        "unconditional"
    );
}
#[test]
fn nonrecursive_helper_parameters_and_aliases() {
    let out=run(".decl W(i:unsigned)\nW(i) :- Actions(i,a), b=a, IsTool(b,\"Write\").\nUnauthorized(i) :- W(i), CurrentDepends(x).");
    assert_eq!(current(&out).conditions, ["W(_)"]);
}
#[test]
fn shared_helpers_union_demand_and_unscoped_consumers_widen() {
    let shared=".decl H(x:symbol)\nH(x) :- CurrentDepends(x).\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), H(x).\nUnauthorized(i) :- Actions(i,$CallTool(\"Read\",_)), H(x).";
    assert_eq!(current(&run(shared)).action_patterns.len(), 2);
    assert_eq!(
        current(&run(&format!("{shared}\nUnauthorized(0) :- H(x)."))).outcome,
        "unconditional"
    );
}
#[test]
fn negative_dependencies_are_preserved() {
    assert_eq!(
        current(&run(
            "Unauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), !CurrentDepends(\"x\")."
        ))
        .action_patterns,
        ["$CallTool(\"Write\", _)"]
    );
    assert_eq!(
        current(&run(
            "Unauthorized(0) :- !CurrentDepends(\"x\"), !IsTool(_,\"Write\")."
        ))
        .outcome,
        "unconditional"
    );
}
#[test]
fn mutual_recursion_propagates_and_is_reported() {
    let out=run(".decl H(x:symbol)\n.decl J(x:symbol)\nH(x) :- J(x).\nJ(x) :- H(x).\nJ(x) :- CurrentDepends(x).\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), H(x).");
    assert_eq!(current(&out).action_patterns, ["$CallTool(\"Write\", _)"]);
    assert!(out
        .report
        .recursive_groups
        .contains(&vec!["H".into(), "J".into()]));
}
#[test]
fn action_batches_are_never_intersected() {
    let out=run("Unauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), Actions(j,$CallTool(\"Read\",_)), CurrentDepends(x).");
    assert_eq!(current(&out).action_patterns.len(), 2);
}
#[test]
fn explicit_and_legacy_gates_are_preserved() {
    let user="CurrentDependsPolicyRelevant() :- Actions(_,$CallTool(\"Custom\",_)).\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).";
    let out = run(user);
    assert_eq!(current(&out).outcome, "preserved");
    assert!(out.source.contains(user));
    let legacy = source(user).replace("// SASY_AUTO_GATE_DEFAULT:", "// old default:");
    assert_eq!(infer_action_gates(&legacy, &legacy).source, legacy);
}
#[test]
fn user_outputs_and_diagnostics_are_observable() {
    assert_eq!(current(&run(".output CurrentDepends\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).")).outcome,"unconditional");
    assert_eq!(
        current(&run(
            ".decl D(x:symbol)\n.printsize D\nD(x) :- CurrentDepends(x)."
        ))
        .outcome,
        "unconditional"
    );
    assert_eq!(current(&run(".decl SasyAllowRoute(i:unsigned)\nSasyAllowRoute(i) :- Actions(i,$CallTool(\"Read\",_)), CurrentDepends(x).")).action_patterns,["$CallTool(\"Read\", _)"]);
}
#[test]
fn disjunction_demand_is_not_cartesian_expanded() {
    assert_eq!(current(&run("Unauthorized(i) :- (Actions(i,$CallTool(\"Read\",_)); Actions(i,$CallTool(\"Write\",_))), CurrentDepends(x).")).action_patterns.len(),2);
    assert_eq!(
        current(&run(
            "Unauthorized(i) :- (Actions(i,$CallTool(\"Read\",_)); i=0), CurrentDepends(x)."
        ))
        .outcome,
        "unconditional"
    );
}
#[test]
fn unknown_syntax_and_limits_leave_true_defaults() {
    for user in [
        ".component Unknown {}",
        "Unauthorized(0) :- 1 = count : { CurrentDepends(_) }.",
    ] {
        let s = source(user);
        let out = infer_action_gates(&s, &s);
        assert!(out.report.fallback.is_some());
        assert_eq!(out.source, s);
    }
    let s = source(&" ".repeat(1024 * 1024));
    assert_eq!(infer_action_gates(&s, &s).source, s);
}
#[test]
fn runtime_functors_are_neither_called_nor_emitted() {
    let out = run("Unauthorized(i) :- Actions(i,$CallTool(@name(),_)), CurrentDepends(x).");
    assert_eq!(current(&out).action_patterns, ["$CallTool(_, _)"]);
    assert!(!out
        .source
        .lines()
        .filter(|l| l.starts_with("CurrentDependsPolicyRelevant() :- Actions"))
        .any(|l| l.contains('@')));
}
#[test]
fn observable_gate_values_are_preserved_even_without_helper_consumers() {
    for user in [
        ".output CurrentDependsPolicyRelevant",
        "Unauthorized(i) :- CurrentDependsPolicyRelevant(), Actions(i,_).",
    ] {
        let s = source(user);
        let out = infer_action_gates(&s, &s);
        assert_eq!(current(&out).outcome, "preserved");
        assert!(out.source.contains("CurrentDependsPolicyRelevant()."));
    }
}
#[test]
fn principal_output_consumed_by_runtime_is_observable() {
    let out = run(".decl HasPrincipal(i:unsigned)\nHasPrincipal(0) :- CurrentDepends(x).");
    assert_eq!(current(&out).outcome, "unconditional");
}
#[test]
fn nonfaithful_string_escapes_keep_original_defaults() {
    for escape in [r"\b", r"\v", r"\f", r"\a", r"\x41", r"\101", r"\u0041"] {
        let s = source(&format!(
            r#"Unauthorized(i) :- Actions(i,$CallTool("{escape}",_)), CurrentDepends(x)."#
        ));
        let out = infer_action_gates(&s, &s);
        assert_eq!(out.source, s);
        assert!(out.report.fallback.unwrap().contains("escape"));
    }
    let out=run("// ignored comment \\b\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).");
    assert_eq!(current(&out).outcome, "inferred");
}
#[test]
fn authored_actions_cannot_create_a_gate_dependency_cycle() {
    for definition in [
        "Actions(0,$CallTool(\"Write\",\"\")) :- CurrentDepends(\"x\").",
        "Actions(0,$CallTool(\"Write\",\"\")).",
    ] {
        let s=source(&format!("{definition}\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x)."));
        let out = infer_action_gates(&s, &s);
        assert_eq!(out.source, s);
        assert!(out.report.fallback.unwrap().contains("Actions"));
    }
}
#[test]
fn marker_text_inside_block_comments_is_not_provenance() {
    let fake="/*\n// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\nCurrentDependsPolicyRelevant().\n*/\n";
    let s=format!("{COMMON}{fake}Unauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).\nCurrentDependsPolicyRelevant().\n");
    assert_eq!(infer_action_gates(&s, &s).source, s);
    let fake = format!("{fake}.output CurrentDepends\n");
    let s = source(&fake);
    let out = infer_action_gates(&s, &s);
    assert_eq!(current(&out).outcome, "unconditional");
    assert!(out.source.contains(&fake));
}
#[test]
fn isolated_authored_source_cannot_hide_outputs_with_a_boundary_comment() {
    let user=".output CurrentDepends\n// === USER_POLICY_BEGIN ===\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).";
    let s = source(user);
    let out = infer_action_gates(&s, user);
    assert_eq!(current(&out).outcome, "unconditional");
}

#[test]
fn positive_state_and_nonrecursive_idb_are_retained_without_functor_copying() {
    let out = run(r#"
.decl Payload(x:symbol)
.input Payload
.decl HiddenUnicodeSource(x:symbol)
HiddenUnicodeSource(x) :- Payload(x), @hidden(x) = 1.
.decl IsPush(i:unsigned)
IsPush(i) :- Actions(i,$CallTool("Bash",args)), @push(args) = 1.
Unauthorized(i) :- IsPush(i), HiddenUnicodeSource(x), CurrentDepends(x).
"#);
    assert_eq!(out.report.fallback, None);
    let clauses = &current(&out).conditions;
    assert_eq!(clauses.len(), 1);
    assert!(clauses[0].contains("IsPush(_)"));
    assert!(clauses[0].contains("HiddenUnicodeSource(_)"));
    assert_eq!(clauses[0], "HiddenUnicodeSource(_), IsPush(_)");
    assert!(!clauses[0].contains('@'));
    assert!(!clauses[0].contains("CurrentDepends("));
}

#[test]
fn different_action_witnesses_remain_separate_existentials_in_conjunction() {
    let out = run(r#"
.decl Ready(i:unsigned)
.input Ready
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), Actions(j,$CallTool("Read",_)), Ready(j), CurrentDepends(x), i != j.
"#);
    assert_eq!(current(&out).conditions.len(), 1);
    assert!(current(&out).conditions[0].contains("Actions(_, $CallTool(\"Write\", _))"));
    assert!(current(&out).conditions[0].contains("Actions(_, $CallTool(\"Read\", _))"));
    assert!(current(&out).conditions[0].contains("Ready(_)"));
    assert_eq!(
        current(&out).conditions[0],
        "Actions(_, $CallTool(\"Read\", _)), Actions(_, $CallTool(\"Write\", _)), Ready(_)"
    );
}

#[test]
fn sibling_rule_scopes_and_disjunctive_paths_do_not_share_variables() {
    let out = run(r#"
.decl State(x:symbol)
.input State
.decl Other(x:symbol)
.input Other
.decl H(x:symbol)
H(x) :- CurrentDepends(x), Other("inner").
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), State("outer"), H(x).
Unauthorized(i) :- Actions(i,$CallTool("Read",_)), State("second"), H(x).
"#);
    assert_eq!(current(&out).conditions.len(), 2);
    for clause in &current(&out).conditions {
        assert!(clause.contains("Other(\"inner\")"));
        assert_ne!(
            clause.contains("State(\"outer\")"),
            clause.contains("State(\"second\")")
        );
    }
}

#[test]
fn all_definitions_and_signed_dependencies_must_be_independent() {
    for definition in [
        "P(x) :- CurrentDepends(x).",
        "P(x) :- Seed(x), !CurrentDepends(x).",
        "P(x) :- ReachableFrom(x,_).",
        "P(x) :- ReachableFromPolicyRelevant(), Seed(x).",
        "P(x) :- Q(x). Q(x) :- P(x).",
    ] {
        let out = run(&format!(
            r#"
.decl Seed(x:symbol)
.input Seed
.decl P(x:symbol)
.decl Q(x:symbol)
P(x) :- Seed(x).
{definition}
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), P(x), CurrentDepends(x).
"#
        ));
        assert_eq!(out.report.fallback, None);
        assert!(
            current(&out)
                .conditions
                .iter()
                .all(|c| !c.contains("P(") && !c.contains("Q(")),
            "{definition}: {:?}",
            current(&out)
        );
    }
    let out = run(r#"
.decl P(x:symbol)
P(x) :- CurrentDepends(x).
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), P(x), ReachableFrom(x,_).
"#);
    assert!(out.report.gates[1]
        .conditions
        .iter()
        .all(|c| !c.contains("P(")));
}

#[test]
fn negative_state_is_not_reinterpreted_as_positive_existence() {
    let out = run(r#"
.decl Ready(x:symbol)
.input Ready
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), !Ready("approval"), !CurrentDepends("source").
"#);
    assert_eq!(
        current(&out).conditions,
        ["Actions(_, $CallTool(\"Write\", _))"]
    );
}

#[test]
fn opaque_terms_in_retained_atoms_widen_without_becoming_code() {
    let out = run(r#"
.decl State(x:symbol)
.input State
Unauthorized(i) :- Actions(i,$CallTool("Write",_)), State(@external("x")), CurrentDepends(x).
"#);
    assert!(current(&out).conditions[0].contains("State(_)"));
    assert!(!current(&out).conditions[0].contains('@'));
}

#[test]
fn conjunctive_clause_limits_widen_instead_of_dropping_consumers() {
    let mut user = String::new();
    let mut body = Vec::new();
    for i in 0..40 {
        user.push_str(&format!(".decl P{i}()\n.input P{i}\n"));
        body.push(format!("P{i}()"));
    }
    user.push_str(&format!(
        "Unauthorized(0) :- {}, CurrentDepends(x).",
        body.join(", ")
    ));
    let out = run(&user);
    assert!(
        out.report.fallback.is_some()
            || current(&out)
                .conditions
                .iter()
                .all(|c| c.split("), ").count() <= 32)
    );
    assert!(out.source.contains(&user));
}

#[test]
fn erased_relation_qualifiers_and_unsupported_execution_directives_preserve_defaults() {
    for qualifier in [
        "inline",
        "no_inline",
        "brie",
        "btree",
        "btree_delete",
        "eqrel",
        "override",
        "magic",
        "no_magic",
        "choice-domain x",
    ] {
        let s = source(&format!(".decl Signal(x:symbol) {qualifier}\nSignal(x) :- Edge(x,_), @stateful(x)=1.\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), Signal(x), CurrentDepends(x)."));
        let out = infer_action_gates(&s, &s);
        assert!(out.report.fallback.is_some(), "{qualifier}");
        assert_eq!(out.source, s, "{qualifier}");
    }
    let s = source(".pragma \"magic-transform\" \"*\"\nUnauthorized(i) :- Actions(i,$CallTool(\"Write\",_)), CurrentDepends(x).");
    let out = infer_action_gates(&s, &s);
    assert!(out.report.fallback.is_some());
    assert_eq!(out.source, s);
    // Keyword text inside a string or comment is not a relation qualifier.
    let out = run("// inline eqrel magic\nUnauthorized(i) :- Actions(i,$CallTool(\"inline\",_)), CurrentDepends(x).");
    assert_eq!(out.report.fallback, None);
}
