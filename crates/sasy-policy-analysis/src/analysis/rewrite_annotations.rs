//! Rewrite `@deny_message`-annotated `Unauthorized`
//! rules to `DenialReason` rules.
//!
//! `sugar.py` does this textually; we do it at the AST
//! level so analyses see the same shape regardless of
//! which path produced the program. The transformation:
//!
//! ```text
//!   // @deny_message: M
//!   // @suggestion: S
//!   Unauthorized(idx) :- body.
//! ```
//!
//! becomes
//!
//! ```text
//!   DenialReason(idx, "block", "M", "S") :- body.
//! ```
//!
//! (the second arg is the denial kind — "ask" for an `@ask` rule, else "block" —
//! matching sugar.py and the `Unauthorized :- DenialReason(idx, "block", _, _)`
//! derivation in common_policy.dl).
//!
//! Annotations are looked up via
//! [`crate::rule_metadata::RuleMetadataParser`] using
//! the rule's source span. Rules without
//! `@deny_message` are left untouched (they're plain
//! Unauthorized rules — typically `Unauthorized(idx) :-
//! DenialReason(idx, _, _).` from `common_policy.dl`).

use super::ast::*;
use crate::rule_metadata::RuleMetadataParser;

/// Rewrite annotated Unauthorized rules in `program`.
/// `source` is the raw `.dl` text from which `program`
/// was parsed (the parser doesn't carry comments
/// through, so we re-scan for annotations).
pub fn rewrite_deny_annotations(program: &mut Program, source: &str, file_path: &str) {
    let parser = RuleMetadataParser::parse_content(source, file_path.to_string());
    for rule in &mut program.rules {
        if rule.head.relation != "Unauthorized" {
            continue;
        }
        let Some(meta) = parser.get_metadata(rule.span.start_line) else {
            continue;
        };
        let Some(deny_message) = meta.deny_message.as_ref() else {
            continue;
        };
        let suggestion = meta.suggestions.first().cloned().unwrap_or_default();
        // Match sugar.py exactly: DenialReason(<idx>, "block"|"ask", "msg", "fix").
        // The second arg is the denial KIND — "ask" for a soft (@ask) denial,
        // "block" otherwise. It is NOT optional: common_policy derives
        // `Unauthorized(idx) :- DenialReason(idx, "block", _, _)`, so a 3-arg head
        // (missing the kind) has the wrong arity and the derivation never unfolds
        // these rules — the AST pipeline then sees 1 Unauthorized disjunct where
        // sugar sees dozens. `@ask` (a value-less annotation) lands in `custom`.
        let kind = if meta.custom.contains_key("ask") {
            "ask"
        } else {
            "block"
        };
        let mut new_args = rule.head.args.clone();
        new_args.push(Term::StringLit(kind.to_string()));
        new_args.push(Term::StringLit(deny_message.clone()));
        new_args.push(Term::StringLit(suggestion));
        rule.head = Atom {
            relation: "DenialReason".to_string(),
            args: new_args,
            span: rule.head.span.clone(),
        };
    }
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    #[test]
    fn rewrites_annotated_unauthorized_to_denial_reason() {
        let src = r#"
            .decl Actions(idx: unsigned, a: symbol)
            .input Actions
// @deny_message: Wrong order
// @suggestion: Use the original payment method
Unauthorized(idx) :- Actions(idx, a), a = $CallTool("modify", _).
        "#;
        let mut prog = parse(src, "test.dl").unwrap();
        rewrite_deny_annotations(&mut prog, src, "test.dl");
        // Original Unauthorized rule should now be DenialReason.
        let dr = prog
            .rules
            .iter()
            .find(|r| r.head.relation == "DenialReason");
        assert!(
            dr.is_some(),
            "expected DenialReason rule, got {:?}",
            prog.rules
        );
        let dr = dr.unwrap();
        assert_eq!(
            dr.head.args.len(),
            4,
            "DenialReason is (idx, kind, msg, fix)"
        );
        match &dr.head.args[1] {
            Term::StringLit(s) => assert_eq!(s, "block", "kind defaults to block without @ask"),
            other => panic!("expected kind string, got {:?}", other),
        }
        match &dr.head.args[2] {
            Term::StringLit(s) => assert_eq!(s, "Wrong order"),
            other => panic!("expected message string, got {:?}", other),
        }
        match &dr.head.args[3] {
            Term::StringLit(s) => assert_eq!(s, "Use the original payment method"),
            other => panic!("expected suggestion string, got {:?}", other),
        }
    }

    #[test]
    fn ask_annotation_sets_kind_to_ask() {
        let src = r#"
            .decl Actions(idx: unsigned, a: symbol)
            .input Actions
// @ask
// @deny_message: Needs confirmation
// @suggestion: Confirm first
Unauthorized(idx) :- Actions(idx, a), a = $CallTool("modify", _).
        "#;
        let mut prog = parse(src, "test.dl").unwrap();
        rewrite_deny_annotations(&mut prog, src, "test.dl");
        let dr = prog
            .rules
            .iter()
            .find(|r| r.head.relation == "DenialReason")
            .expect("expected DenialReason rule");
        assert_eq!(dr.head.args.len(), 4);
        match &dr.head.args[1] {
            Term::StringLit(s) => assert_eq!(s, "ask", "@ask makes the denial soft"),
            other => panic!("expected kind string, got {:?}", other),
        }
    }

    #[test]
    fn leaves_unannotated_unauthorized_alone() {
        let src = r#"
            .decl DenialReason(idx: unsigned, reason: symbol, suggestion: symbol)
            Unauthorized(idx) :- DenialReason(idx, _, _).
        "#;
        let mut prog = parse(src, "test.dl").unwrap();
        rewrite_deny_annotations(&mut prog, src, "test.dl");
        let unauth = prog
            .rules
            .iter()
            .find(|r| r.head.relation == "Unauthorized");
        assert!(unauth.is_some());
    }
}
