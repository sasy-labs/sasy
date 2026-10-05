//! Resolve `var.field` access sugar into positional
//! record unpacks.
//!
//! After parsing, rule bodies may contain
//! [`Term::FieldAccess { record: var, field }`] terms
//! where `var` is bound to a record-typed value. This
//! pass mirrors what `sugar.py` does, at the AST level:
//! for each rule, group the field accesses by record
//! variable, look up the variable's type via the
//! enclosing atoms' relation signatures, and rewrite the
//! body to:
//!
//! ```text
//!   var = [_, _, fresh_field_a, _, fresh_field_b],
//!   <substituted body using fresh_field_a, fresh_field_b>
//! ```
//!
//! The auditor's source-line spans stay intact because
//! we work directly on the parsed AST — no intermediate
//! desugared text and no line-number remapping.
//!
//! Limitations / contract:
//! - We resolve dot-notation against `.type` *record*
//!   declarations only. ADTs (e.g., `Action = …`) don't
//!   support dot access; field access on ADT-typed
//!   vars is left as `FieldAccess` and the caller will
//!   error out downstream.
//! - We need the variable's type to be inferable from
//!   *some* positive atom whose relation has a typed
//!   parameter. Variables that only appear on the right
//!   side of a `Compare` aren't typeable here.

use std::collections::HashMap;

use super::ast::*;
use super::unfold::FreshVars;

#[derive(Debug, thiserror::Error)]
pub enum DotResolveError {
    #[error("cannot infer record type for variable `{var}` (rule at {span})")]
    UnknownType { var: String, span: String },
    #[error("variable `{var}` has type `{ty}`, which is not a record (rule at {span})")]
    NotARecord {
        var: String,
        ty: String,
        span: String,
    },
    #[error("record type `{ty}` has no field `{field}` (referenced as `{var}.{field}` at {span})")]
    UnknownField {
        var: String,
        field: String,
        ty: String,
        span: String,
    },
}

/// Apply dot-resolution to every rule in the program.
/// Mutates `program.rules` in place.
pub fn resolve_dots(program: &mut Program) -> Result<(), DotResolveError> {
    let registry = TypeRegistry::build(program);
    let mut var_gen = FreshVars::new();
    let mut wildcard_counter: u32 = 1_000_000_000;
    let mut new_rules = Vec::with_capacity(program.rules.len());
    for rule in std::mem::take(&mut program.rules) {
        new_rules.push(resolve_in_rule(
            rule,
            &registry,
            &mut var_gen,
            &mut wildcard_counter,
        )?);
    }
    program.rules = new_rules;
    Ok(())
}

/* ────────────────────────────────────────────────────────
Type registry
──────────────────────────────────────────────────────── */

struct TypeRegistry {
    /// Record-type definitions: type name → ordered list
    /// of (field_name, field_type).
    records: HashMap<String, Vec<(String, TypeRef)>>,
    /// Relation declarations: relation name → ordered
    /// list of arg types.
    relations: HashMap<String, Vec<TypeRef>>,
    /// Type aliases: `T <: U` → `U`. Recursive resolution
    /// is the caller's job.
    aliases: HashMap<String, TypeRef>,
}

impl TypeRegistry {
    fn build(program: &Program) -> Self {
        let mut records: HashMap<String, Vec<(String, TypeRef)>> = HashMap::new();
        let mut aliases: HashMap<String, TypeRef> = HashMap::new();
        for (name, decl) in &program.types {
            match decl {
                TypeDecl::Record { fields, .. } => {
                    records.insert(name.clone(), fields.clone());
                }
                TypeDecl::Alias { of, .. } => {
                    aliases.insert(name.clone(), of.clone());
                }
                TypeDecl::Adt { .. } => {
                    // ADTs aren't dot-accessible.
                }
            }
        }
        let mut relations: HashMap<String, Vec<TypeRef>> = HashMap::new();
        for (name, rd) in &program.relations {
            relations.insert(
                name.clone(),
                rd.params.iter().map(|(_, t)| t.clone()).collect(),
            );
        }
        Self {
            records,
            relations,
            aliases,
        }
    }

    /// Resolve a `TypeRef` to a record type's field
    /// list, following aliases. Returns `None` if the
    /// type isn't a record (or chain ends in a primitive).
    fn record_fields(&self, t: &TypeRef) -> Option<&Vec<(String, TypeRef)>> {
        match t {
            TypeRef::Named(n) => {
                if let Some(fields) = self.records.get(n) {
                    Some(fields)
                } else if let Some(alias) = self.aliases.get(n) {
                    self.record_fields(alias)
                } else {
                    None
                }
            }
            TypeRef::Prim(_) => None,
        }
    }

    fn type_display(&self, t: &TypeRef) -> String {
        match t {
            TypeRef::Named(n) => n.clone(),
            TypeRef::Prim(p) => format!("{:?}", p),
        }
    }
}

/* ────────────────────────────────────────────────────────
Per-rule resolution
──────────────────────────────────────────────────────── */

fn resolve_in_rule(
    rule: Rule,
    registry: &TypeRegistry,
    var_gen: &mut FreshVars,
    wildcard_counter: &mut u32,
) -> Result<Rule, DotResolveError> {
    // Collect field accesses anywhere in the rule.
    let mut accesses: Vec<(String, String)> = Vec::new();
    collect_accesses_in_atom(&rule.head, &mut accesses);
    for lit in &rule.body {
        collect_accesses_in_literal(lit, &mut accesses);
    }
    if accesses.is_empty() {
        return Ok(rule);
    }
    // Deduplicate while preserving insertion order.
    accesses.sort();
    accesses.dedup();

    // Allocate fresh names for each (var, field) pair.
    let mut subst: HashMap<(String, String), String> = HashMap::new();
    let mut by_var: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (var, field) in &accesses {
        let fresh = var_gen.fresh(&format!("{}_{}", var, field));
        subst.insert((var.clone(), field.clone()), fresh.clone());
        by_var
            .entry(var.clone())
            .or_default()
            .push((field.clone(), fresh));
    }

    // Infer types of vars.
    let var_types = infer_var_types(&rule, registry);

    // Build positional unpack literals.
    let mut unpack: Vec<Literal> = Vec::new();
    for (var, field_pairs) in &by_var {
        let ty = var_types
            .get(var)
            .ok_or_else(|| DotResolveError::UnknownType {
                var: var.clone(),
                span: rule.span.as_location(),
            })?;
        let fields = registry
            .record_fields(ty)
            .ok_or_else(|| DotResolveError::NotARecord {
                var: var.clone(),
                ty: registry.type_display(ty),
                span: rule.span.as_location(),
            })?;
        // Each rule field gets either the fresh var
        // (when accessed) or a fresh wildcard.
        let record_args: Vec<Term> = fields
            .iter()
            .map(|(field_name, _)| {
                if let Some((_, fresh)) = field_pairs.iter().find(|(f, _)| f == field_name) {
                    Ok(Term::Var(fresh.clone()))
                } else {
                    let id = *wildcard_counter;
                    *wildcard_counter += 1;
                    Ok(Term::Wildcard(id))
                }
            })
            .collect::<Result<Vec<_>, DotResolveError>>()?;
        // Validate field names: every requested field
        // must exist on the record type.
        for (field_name, _) in field_pairs {
            if !fields.iter().any(|(n, _)| n == field_name) {
                return Err(DotResolveError::UnknownField {
                    var: var.clone(),
                    field: field_name.clone(),
                    ty: registry.type_display(ty),
                    span: rule.span.as_location(),
                });
            }
        }
        unpack.push(Literal::Compare {
            op: CompareOp::Eq,
            left: Term::Var(var.clone()),
            right: Term::RecordLit(record_args),
            span: rule.span.clone(),
        });
    }

    // Substitute FieldAccess in head and body.
    let head = sub_atom(&rule.head, &subst);
    let mut body: Vec<Literal> = Vec::with_capacity(unpack.len() + rule.body.len());
    body.extend(unpack);
    for lit in &rule.body {
        body.push(sub_literal(lit, &subst));
    }
    Ok(Rule {
        head,
        body,
        span: rule.span,
    })
}

/* ────────────────────────────────────────────────────────
Type inference
──────────────────────────────────────────────────────── */

fn infer_var_types(rule: &Rule, registry: &TypeRegistry) -> HashMap<String, TypeRef> {
    let mut out: HashMap<String, TypeRef> = HashMap::new();
    bind_from_atom(&rule.head, registry, &mut out);
    for lit in &rule.body {
        match lit {
            Literal::Pos(a) | Literal::Neg(a) => bind_from_atom(a, registry, &mut out),
            _ => {}
        }
    }
    out
}

fn bind_from_atom(atom: &Atom, registry: &TypeRegistry, out: &mut HashMap<String, TypeRef>) {
    let Some(types) = registry.relations.get(&atom.relation) else {
        return;
    };
    for (i, arg) in atom.args.iter().enumerate() {
        if i >= types.len() {
            break;
        }
        if let Term::Var(name) = arg {
            // First-binding wins. Conflicts (a var
            // appearing in two relations with different
            // types) would be a Soufflé type error and
            // surface there.
            out.entry(name.clone()).or_insert_with(|| types[i].clone());
        }
    }
}

/* ────────────────────────────────────────────────────────
FieldAccess collection
──────────────────────────────────────────────────────── */

fn collect_accesses_in_atom(atom: &Atom, out: &mut Vec<(String, String)>) {
    for arg in &atom.args {
        collect_accesses_in_term(arg, out);
    }
}

fn collect_accesses_in_literal(lit: &Literal, out: &mut Vec<(String, String)>) {
    match lit {
        Literal::Pos(a) | Literal::Neg(a) => collect_accesses_in_atom(a, out),
        Literal::Negation { literal, .. } => {
            collect_accesses_in_literal(literal, out);
        }
        Literal::Compare { left, right, .. } => {
            collect_accesses_in_term(left, out);
            collect_accesses_in_term(right, out);
        }
        Literal::Disjunction { alternatives, .. } => {
            for alt in alternatives {
                for l in alt {
                    collect_accesses_in_literal(l, out);
                }
            }
        }
    }
}

fn collect_accesses_in_term(t: &Term, out: &mut Vec<(String, String)>) {
    match t {
        Term::FieldAccess { record, field } => {
            out.push((record.clone(), field.clone()));
        }
        Term::Constructor { args, .. }
        | Term::Functor { args, .. }
        | Term::Builtin { args, .. } => {
            for a in args {
                collect_accesses_in_term(a, out);
            }
        }
        Term::RecordLit(fs) => {
            for f in fs {
                collect_accesses_in_term(f, out);
            }
        }
        Term::Arith { left, right, .. } => {
            collect_accesses_in_term(left, out);
            collect_accesses_in_term(right, out);
        }
        Term::Var(_)
        | Term::Wildcard(_)
        | Term::StringLit(_)
        | Term::NumberLit(_)
        | Term::UnsignedLit(_) => {}
    }
}

/* ────────────────────────────────────────────────────────
Substitution
──────────────────────────────────────────────────────── */

type Subst = HashMap<(String, String), String>;

fn sub_atom(atom: &Atom, subst: &Subst) -> Atom {
    Atom {
        relation: atom.relation.clone(),
        args: atom.args.iter().map(|t| sub_term(t, subst)).collect(),
        span: atom.span.clone(),
    }
}

fn sub_literal(lit: &Literal, subst: &Subst) -> Literal {
    match lit {
        Literal::Pos(a) => Literal::Pos(sub_atom(a, subst)),
        Literal::Neg(a) => Literal::Neg(sub_atom(a, subst)),
        Literal::Negation { literal, span } => Literal::Negation {
            literal: Box::new(sub_literal(literal, subst)),
            span: span.clone(),
        },
        Literal::Compare {
            op,
            left,
            right,
            span,
        } => Literal::Compare {
            op: *op,
            left: sub_term(left, subst),
            right: sub_term(right, subst),
            span: span.clone(),
        },
        Literal::Disjunction { alternatives, span } => Literal::Disjunction {
            alternatives: alternatives
                .iter()
                .map(|alt| alt.iter().map(|l| sub_literal(l, subst)).collect())
                .collect(),
            span: span.clone(),
        },
    }
}

fn sub_term(t: &Term, subst: &Subst) -> Term {
    match t {
        Term::FieldAccess { record, field } => {
            if let Some(fresh) = subst.get(&(record.clone(), field.clone())) {
                Term::Var(fresh.clone())
            } else {
                t.clone()
            }
        }
        Term::Constructor { name, args } => Term::Constructor {
            name: name.clone(),
            args: args.iter().map(|a| sub_term(a, subst)).collect(),
        },
        Term::Functor { name, args } => Term::Functor {
            name: name.clone(),
            args: args.iter().map(|a| sub_term(a, subst)).collect(),
        },
        Term::Builtin { name, args } => Term::Builtin {
            name: name.clone(),
            args: args.iter().map(|a| sub_term(a, subst)).collect(),
        },
        Term::RecordLit(fs) => Term::RecordLit(fs.iter().map(|f| sub_term(f, subst)).collect()),
        Term::Arith { op, left, right } => Term::Arith {
            op: *op,
            left: Box::new(sub_term(left, subst)),
            right: Box::new(sub_term(right, subst)),
        },
        Term::Var(_)
        | Term::Wildcard(_)
        | Term::StringLit(_)
        | Term::NumberLit(_)
        | Term::UnsignedLit(_) => t.clone(),
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
    fn resolves_single_dot_access() {
        let mut prog = parse(
            r#"
                .type Message = [contents: symbol, agent: symbol, role: symbol]
                .decl SentMessage(id: symbol, msg: Message)
                .input SentMessage
                R(id) :- SentMessage(id, msg), msg.agent = "FDAHandler".
            "#,
            "test.dl",
        )
        .unwrap();
        resolve_dots(&mut prog).unwrap();
        let body = &prog.rules[0].body;
        // Body should now have:
        // 1. Compare(msg = [_, fresh_agent, _])  (the unpack)
        // 2. Pos(SentMessage(id, msg))
        // 3. Compare(fresh_agent = "FDAHandler")
        assert_eq!(body.len(), 3, "got {:?}", body);
        let has_unpack = body.iter().any(|lit| match lit {
            Literal::Compare {
                op: CompareOp::Eq,
                left: Term::Var(v),
                right: Term::RecordLit(_),
                ..
            } => v == "msg",
            _ => false,
        });
        assert!(has_unpack, "expected positional unpack, got {:?}", body);
        // No FieldAccess remaining anywhere.
        let has_field = body.iter().any(|lit| match lit {
            Literal::Compare { left, right, .. } => {
                matches!(left, Term::FieldAccess { .. })
                    || matches!(right, Term::FieldAccess { .. })
            }
            _ => false,
        });
        assert!(
            !has_field,
            "expected no FieldAccess remaining, got {:?}",
            body
        );
    }

    #[test]
    fn resolves_multiple_fields_into_one_unpack() {
        let mut prog = parse(
            r#"
                .type Message = [contents: symbol, agent: symbol, role: symbol]
                .decl SentMessage(id: symbol, msg: Message)
                .input SentMessage
                R(id) :- SentMessage(id, msg), msg.agent = "X", msg.contents = "Y".
            "#,
            "test.dl",
        )
        .unwrap();
        resolve_dots(&mut prog).unwrap();
        let body = &prog.rules[0].body;
        // Single unpack literal for msg should be added.
        let unpacks: Vec<_> = body
            .iter()
            .filter(|lit| {
                matches!(
                    lit,
                    Literal::Compare {
                        left: Term::Var(v),
                        right: Term::RecordLit(_),
                        ..
                    } if v == "msg"
                )
            })
            .collect();
        assert_eq!(unpacks.len(), 1, "expected one unpack, got {:?}", unpacks);
    }

    #[test]
    fn errors_on_unknown_type() {
        let mut prog = parse(
            r#"
                .decl R(idx: unsigned)
                R(idx) :- foo.bar = "x".
            "#,
            "test.dl",
        )
        .unwrap();
        let res = resolve_dots(&mut prog);
        assert!(matches!(res, Err(DotResolveError::UnknownType { .. })));
    }

    #[test]
    fn errors_on_unknown_field() {
        let mut prog = parse(
            r#"
                .type Message = [contents: symbol, agent: symbol]
                .decl SentMessage(id: symbol, msg: Message)
                .input SentMessage
                R(id) :- SentMessage(id, msg), msg.nonexistent = "x".
            "#,
            "test.dl",
        )
        .unwrap();
        let res = resolve_dots(&mut prog);
        assert!(matches!(res, Err(DotResolveError::UnknownField { .. })));
    }

    #[test]
    fn no_op_when_no_dots() {
        let mut prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                R(idx) :- Actions(idx, a), a = $CallTool("X", _).
            "#,
            "test.dl",
        )
        .unwrap();
        let original = prog.rules[0].body.len();
        resolve_dots(&mut prog).unwrap();
        assert_eq!(prog.rules[0].body.len(), original);
    }
}
