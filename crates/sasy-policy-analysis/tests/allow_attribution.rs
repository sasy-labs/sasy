use sasy_policy_analysis::allow_attribution::{transform_desugared, MAX_ROUTES, MAX_SOURCE_BYTES};

const INPUTS: &str = r#"
.type Action = CallTool { name:symbol, args:symbol } | HTTPRequest { url:symbol, body:symbol, headers:symbol }
.decl Actions(idx:unsigned, action:Action)
.input Actions
.decl Principal(p:symbol)
.input Principal
.decl PrincipalRole(p:symbol, role:symbol)
.input PrincipalRole
.decl ActionMetadata(idx:unsigned, rel:symbol, a:symbol, b:symbol)
.input ActionMetadata
.decl IsAuthorized(idx:unsigned)
.output IsAuthorized
"#;
fn source(rules: &str) -> String {
    format!("{INPUTS}{rules}\n")
}
fn generated(text: &str) -> String {
    let result = transform_desugared(text);
    assert!(
        result.fallback_reason.is_none(),
        "{:?}",
        result.fallback_reason
    );
    assert!(result.source.starts_with(text));
    result.source[text.len()..].to_string()
}

#[test]
fn freezes_only_builtin_conditions_and_keeps_join_witnesses() {
    let src = source(
        r#"
.decl Approval(idx:unsigned)
// @deny_message: Obtain the required approval
// @suggestion: Ask the appropriate reviewer
IsAuthorized(i) :- Actions(i,a), a=$CallTool("deploy",_), Principal(p), PrincipalRole(p,"operator"), Approval(i).
"#,
    );
    let result = transform_desugared(&src);
    let tail = generated(&src);
    assert!(tail.contains(
        "Actions(i,a), a=$CallTool(\"deploy\",_), Principal(p), PrincipalRole(p,\"operator\")"
    ));
    assert!(!tail.contains("Approval("));
    assert!(tail.contains("\"possible\""));
    assert!(tail.contains("\"blocked\""));
    assert!(!tail.contains("\"unknown\""));
    assert_eq!(result.routes[0].message, "Obtain the required approval");
    assert_eq!(
        result.routes[0].suggestions,
        ["Ask the appropriate reviewer"]
    );
    assert!(tail.contains("Obtain the required approval"));
    assert!(!tail.contains("DenialReason("));
}

#[test]
fn custom_recursive_and_negative_relations_are_not_snapshot_pruners() {
    let src = source(
        r#"
.decl Approved(idx:unsigned)
.decl Revoked(idx:unsigned)
Approved(i) :- Approved(i).
IsAuthorized(i) :- Actions(i,_), Approved(i), !Revoked(i).
"#,
    );
    let tail = generated(&src);
    assert!(!tail.contains("Approved("));
    assert!(!tail.contains("Revoked("));
    assert!(tail.contains("\"possible\""));
}

#[test]
fn preserves_correlated_metadata_witness_and_safe_negation() {
    let src = source(
        r#"IsAuthorized(i) :- Actions(i,_), ActionMetadata(i,"left",w,"yes"), ActionMetadata(i,"right",w,"yes"), !Principal("blocked")."#,
    );
    let tail = generated(&src);
    assert!(tail.contains("ActionMetadata(i,\"left\",w,\"yes\"), ActionMetadata(i,\"right\",w,\"yes\"), !Principal(\"blocked\")"));
    assert!(tail.contains("\"possible\""));
}

#[test]
fn overridden_common_helpers_are_changeable_not_frozen() {
    let canonical = r#"
.decl HasRole(role:symbol)
HasRole(role) :- Principal(p), PrincipalRole(p, role).
IsAuthorized(i) :- Actions(i,_), HasRole("operator").
"#;
    assert!(generated(&source(canonical)).contains("HasRole(\"operator\")"));
    let override_source = source(&format!("{canonical}\nHasRole(\"extra\")."));
    let tail = generated(&override_source);
    assert!(!tail.contains("HasRole("));
    assert!(tail.contains("\"possible\""));
    let changed = source(&canonical.replace("PrincipalRole(p, role)", "PrincipalRole(_, role)"));
    assert!(!generated(&changed).contains("HasRole("));
}

#[test]
fn exact_pure_tool_helper_is_retained_without_copying_its_definition() {
    let src = source(
        r#"
.decl IsTool(action:Action, name:symbol)
IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).
IsAuthorized(i) :- Actions(i,a), IsTool(a,"deploy").
"#,
    );
    let tail = generated(&src);
    assert!(tail.contains("IsTool(a,\"deploy\")"));
    assert!(!tail.contains("$CallTool"));
}

#[test]
fn ancestor_and_external_function_conditions_remain_unknown_without_calls() {
    for body in [
        "CurrentDepends(x)",
        "ReachableFrom(x,y)",
        "QueriesHost(a,h)",
        "@llm_check_fn(\"p\",\"c\") = 1",
        "strlen(\"a\") = 1",
    ] {
        let src = source(&format!("IsAuthorized(i) :- Actions(i,a), {body}."));
        let tail = generated(&src);
        assert!(tail.contains("\"unknown\""), "{body}: {tail}");
        assert!(!tail.contains(body));
        assert!(!tail.contains('@'));
    }
}

#[test]
fn disjunction_cast_and_aggregate_routes_are_unknown_without_expansion() {
    for body in [
        "(Principal(\"a\"); Principal(\"b\"))",
        "as(i,unsigned)=i",
        "n = count : {Principal(_)}, n > 0",
    ] {
        let src = source(&format!("IsAuthorized(i) :- Actions(i,_), {body}."));
        let tail = generated(&src);
        assert!(tail.contains("\"unknown\""), "{body}: {tail}");
        assert!(!tail.contains(body));
    }
}

#[test]
fn preserves_literal_bytes_and_multiline_statement_annotations() {
    let src = source("// @deny_message: Escaped \\\"hint\\\"\nIsAuthorized(i) :-\n Actions(i,a), a=$CallTool(\"a\\\\b\\t\\\"c\",_), Principal(\"☃\").\n");
    let tail = generated(&src);
    assert!(tail.contains("a=$CallTool(\"a\\\\b\\t\\\"c\",_)"));
    assert!(tail.contains("Principal(\"☃\")"));
    assert!(transform_desugared(&src).source.starts_with(&src));
}

#[test]
fn route_ids_are_deterministic_unique_and_metadata_does_not_cross_same_line() {
    let src = source("// @suggestion: first only\nIsAuthorized(i) :- Actions(i,_). IsAuthorized(i) :- Actions(i,_).\n");
    let one = transform_desugared(&src);
    let two = transform_desugared(&src);
    assert_eq!(one.source, two.source);
    assert_eq!(one.routes.len(), 2);
    assert_ne!(one.routes[0].rule_id, one.routes[1].rule_id);
    assert!(one.routes.iter().all(|r| r.suggestions.is_empty()));
}

#[test]
fn bounded_fallback_preserves_input_and_never_hides_omitted_routes() {
    for src in [
        source(&"IsAuthorized(i) :- Actions(i,_).\n".repeat(MAX_ROUTES + 1)),
        source(".decl SasyAllowFixed0(i:unsigned)\nIsAuthorized(i) :- Actions(i,_)."),
        source("Actions(0,$CallTool(\"x\",\"\")).\nIsAuthorized(i) :- Actions(i,_)."),
        source("#include \"missing.dl\"\n"),
        " ".repeat(MAX_SOURCE_BYTES + 1),
        source(&format!(
            "IsAuthorized(i) :- Actions(i,{}0{}).",
            "[".repeat(64),
            "]".repeat(64)
        )),
    ] {
        let result = transform_desugared(&src);
        assert_eq!(result.source, src);
        assert!(!result.complete);
        assert!(result.routes.is_empty());
        assert!(result.fallback_reason.is_some());
    }
}

#[test]
fn all_literal_work_and_author_hints_are_capped() {
    let src = source(&format!(
        "// @deny_message: {}\n{}\nIsAuthorized(i) :- Actions(i,_).",
        "é".repeat(1500),
        (0..10)
            .map(|i| format!("// @suggestion: suggestion{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    ));
    let result = transform_desugared(&src);
    assert_eq!(result.routes[0].message.len(), 1024);
    assert_eq!(result.routes[0].suggestions.len(), 4);
    let src = source(&format!(
        "IsAuthorized(i) :- Actions(i,_), {}.",
        vec!["Principal(\"a\")"; 40].join(",")
    ));
    assert!(generated(&src).contains("\"unknown\""));
}

#[test]
fn reserved_namespace_detection_survives_parser_and_size_fallback() {
    use sasy_policy_analysis::allow_attribution::reserved_namespace_collision;
    assert!(!reserved_namespace_collision(
        "// SasyAllowRoute\n/* SasyAllowFixed0 */\nR(\"SasyAllowRoute\").\nOtherSasyAllowRoute()."
    ));
    for source in [
        ".decl SasyAllowRoute(x:unsigned)",
        "R(x) :- SasyAllowOther(x).",
        "R(SasyAllowVariable).",
        "unsupported syntax ??? SasyAllowRoute()",
    ] {
        assert!(reserved_namespace_collision(source));
    }
    assert!(reserved_namespace_collision(&format!(
        "{} SasyAllowRoute()",
        " ".repeat(MAX_SOURCE_BYTES + 1)
    )));
}

#[test]
fn wildcard_patterns_cannot_ground_dropped_custom_witnesses() {
    for equality in [
        "a=$CallTool(_,_)",
        "$CallTool(_,_) = a",
        "a=[_,_]",
        "a != $CallTool(_,_)",
    ] {
        let src = source(&format!(
            ".decl Custom(a:Action)\nIsAuthorized(i) :- Actions(i,_), Custom(a), {equality}."
        ));
        let tail = generated(&src);
        assert!(tail.contains("\"unknown\""), "{equality}: {tail}");
        assert!(!tail.contains(equality));
    }
    let src = source("IsAuthorized(i) :- Actions(i,a), a=$CallTool(name,_), name=\"deploy\".");
    let tail = generated(&src);
    assert!(tail.contains("a=$CallTool(name,_), name=\"deploy\""));
    assert!(tail.contains("\"possible\""));
}
