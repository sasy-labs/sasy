use sasy_policy_analysis::analysis::{
    ast::{format_rule, Literal, Term},
    chase::Chase,
    parse, parse_with_options,
    resolve_dots::resolve_dots,
    subsumption::rule_subsumes,
    unfold::OpaqueSet,
    ParseOptions,
};

#[test]
fn nested_builtins_keep_their_kind_through_parsing_and_dot_resolution() {
    let source = r#"
.type Message = [contents:symbol]
.decl Input(m:Message)
.decl Output(x:symbol)
Output(cat(m.contents, "/")) :- Input(m), strlen(m.contents) > 0,
    substr(m.contents, strlen(m.contents) - 1, 1) != "/",
    @strlen(m.contents) = 4, to_number(m.contents) = 3.
"#;
    let mut program = parse(source, "builtin.dl").unwrap();
    assert!(matches!(
        program.rules[0].head.args[0],
        Term::Builtin { .. }
    ));
    let serialized = serde_json::to_value(
        parse_with_options(source, "builtin.dl", ParseOptions::default()).unwrap(),
    )
    .unwrap();
    assert!(serialized.to_string().contains("\"Builtin\""));
    assert!(serialized.to_string().contains("\"Functor\""));
    resolve_dots(&mut program).unwrap();
    let formatted = format_rule(&program.rules[0]);
    assert!(formatted.starts_with("Output(cat("), "{formatted}");
    assert!(formatted.contains("substr("), "{formatted}");
    assert!(formatted.contains("strlen("), "{formatted}");
    assert!(formatted.contains("@strlen("), "{formatted}");
    assert!(!formatted.contains("m.contents"), "{formatted}");
    assert!(parse(&formatted, "formatted.dl").is_ok(), "{formatted}");
}

#[test]
fn malformed_and_unsupported_calls_are_rejected_without_changing_relation_atoms() {
    for call in [
        "strlen()",
        "to_number()",
        "strlen(x,y)",
        "substr(x,0)",
        "cat(,x)",
        "strlen(x,)",
        "unknown(x)",
        "range(0,2)",
        "autoinc()",
    ] {
        let error = parse(&format!("R(x) :-\n S(x), x = {call}."), "invalid.dl").expect_err(call);
        assert!(error.to_string().contains("line 2"), "{call}: {error}");
    }
    let p = parse("R(x) :- StringLength(x), x = @strlen(x).", "relations.dl").unwrap();
    assert!(matches!(&p.rules[0].body[0], Literal::Pos(atom) if atom.relation == "StringLength"));
    // Soufflé cat accepts empty, unary and variadic argument lists.
    assert!(parse("R(x) :- x = cat().", "empty-cat.dl").is_ok());
    assert!(parse("R(x) :- x = cat(\"a\").", "unary-cat.dl").is_ok());
    assert!(parse("R(x) :- x = cat(\"a\",\"b\",\"c\").", "variadic.dl").is_ok());
}

#[test]
fn builtin_determinism_never_unifies_same_named_external_functions() {
    let p = parse("R(x) :- S(x), strlen(x) = 1, strlen(x) = 2.", "same.dl").unwrap();
    let mut chase = Chase::new();
    assert!(chase.assert_body(&p.rules[0].body).is_err());

    let p = parse(
        "R(x) :- S(x), strlen(x) = 1, @strlen(x) = 2.",
        "different.dl",
    )
    .unwrap();
    let mut chase = Chase::new();
    chase.assert_body(&p.rules[0].body).unwrap();
    chase.check().unwrap();
}

#[test]
fn subsumption_and_unfolding_preserve_builtin_identity() {
    let p = parse(
        r#"
.decl Input(x:symbol)
.input Input
.decl Helper(x:symbol)
.decl R(x:symbol)
Helper(x) :- Input(x), strlen(x) = 1.
R(x) :- Helper(x).
R(x) :- Input(x), strlen(x) = 1.
R(x) :- Input(x), @strlen(x) = 1.
"#,
        "subsumption.dl",
    )
    .unwrap();
    let rules = p.rules_for("R").collect::<Vec<_>>();
    let opaque = OpaqueSet::default();
    assert!(rule_subsumes(rules[0], rules[1], &p, &opaque));
    assert!(rule_subsumes(rules[1], rules[0], &p, &opaque));
    assert!(!rule_subsumes(rules[0], rules[2], &p, &opaque));
    assert!(!rule_subsumes(rules[2], rules[0], &p, &opaque));
}

#[test]
fn full_analysis_reports_unavailable_instead_of_running_unbounded_work() {
    use sasy_policy_analysis::analysis::pipeline::{analyze_desugared, analyze_raw, AnalyzeError};
    for source in [
        "R(cat(\"a\",\"b\")).",
        "R(x) :- S(x), x = @outer(strlen(\"a\")).",
        "R(x) :- S(x), (x = to_number(\"1\"); x = 2).",
        "R(x) :- S(x), !T(substr(x,0,1)).",
    ] {
        assert!(
            matches!(
                analyze_raw(source, "builtin.dl", &[]),
                Err(AnalyzeError::UnsupportedBuiltin { .. })
            ),
            "{source}"
        );
        assert!(
            matches!(
                analyze_desugared(source, "builtin.dl", &[]),
                Err(AnalyzeError::UnsupportedBuiltin { .. })
            ),
            "{source}"
        );
    }
    assert!(analyze_raw("R(x) :- S(x), x = @strlen(x).", "external.dl", &[]).is_ok());
}
