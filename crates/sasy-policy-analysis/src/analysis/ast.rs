//! Abstract syntax tree for the Soufflé Datalog subset
//! used by SASY policies.
//!
//! This is the post-sugar form: `sugar.py` has already
//! desugared dot-notation field accesses into positional
//! record unpacks and rewritten annotated `Unauthorized`
//! rules into `DenialReason` rules. We parse what Soufflé
//! itself accepts.

use std::collections::HashMap;

use serde::Serialize;

/// Source location for an AST node — the file it came
/// from and the 1-indexed line range it spans.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct SourceSpan {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

impl SourceSpan {
    pub fn single(file: impl Into<String>, line: u32) -> Self {
        Self {
            file: file.into(),
            start_line: line,
            end_line: line,
        }
    }

    /// Format as the `file:line` form used elsewhere
    /// in the crate (rule_metadata, denial traces).
    pub fn as_location(&self) -> String {
        if self.start_line == self.end_line {
            format!("{}:{}", self.file, self.start_line)
        } else {
            format!("{}:{}-{}", self.file, self.start_line, self.end_line)
        }
    }
}

/// A Soufflé primitive type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum PrimType {
    Symbol,
    Number,
    Unsigned,
    Float,
}

/// A type reference — either a built-in primitive or a
/// user-declared name resolved later against the program's
/// type table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum TypeRef {
    Prim(PrimType),
    Named(String),
}

impl TypeRef {
    #[allow(clippy::should_implement_trait)] // intentional inherent infallible parser; FromStr churn not warranted
    pub fn from_str(s: &str) -> Self {
        match s {
            "symbol" => TypeRef::Prim(PrimType::Symbol),
            "number" => TypeRef::Prim(PrimType::Number),
            "unsigned" => TypeRef::Prim(PrimType::Unsigned),
            "float" => TypeRef::Prim(PrimType::Float),
            other => TypeRef::Named(other.to_string()),
        }
    }
}

/// A user-defined type declaration: `.type T = ...`.
#[derive(Debug, Clone, Serialize)]
pub enum TypeDecl {
    /// Subtype alias: `.type T <: symbol` or `.type T = number`.
    Alias { name: String, of: TypeRef },
    /// Record type: `.type T = [f1: T1, f2: T2, ...]`.
    Record {
        name: String,
        fields: Vec<(String, TypeRef)>,
    },
    /// Algebraic data type with one or more branches.
    Adt {
        name: String,
        branches: Vec<AdtBranch>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct AdtBranch {
    pub name: String,
    pub fields: Vec<(String, TypeRef)>,
}

impl TypeDecl {
    pub fn name(&self) -> &str {
        match self {
            TypeDecl::Alias { name, .. } => name,
            TypeDecl::Record { name, .. } => name,
            TypeDecl::Adt { name, .. } => name,
        }
    }
}

/// A relation declaration: `.decl R(x: T, y: U)` plus
/// optional `.input` / `.output` markers.
#[derive(Debug, Clone, Serialize)]
pub struct RelationDecl {
    pub name: String,
    pub params: Vec<(String, TypeRef)>,
    pub is_input: bool,
    pub is_output: bool,
    pub span: SourceSpan,
}

/// A functor declaration: `.functor f(x: T): U stateful`.
#[derive(Debug, Clone, Serialize)]
pub struct FunctorDecl {
    pub name: String,
    pub params: Vec<TypeRef>,
    pub return_type: TypeRef,
    pub stateful: bool,
    pub span: SourceSpan,
}

/// A term — a value appearing as an argument to an atom
/// or as one side of a comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum Term {
    /// Plain variable: `idx`, `user`, `mid`.
    Var(String),
    /// Anonymous wildcard `_`. Each occurrence is a
    /// fresh variable; the parser assigns unique names.
    Wildcard(u32),
    /// String literal: `"submit_fda_request"`.
    StringLit(String),
    /// Integer literal: `42`, `-5`.
    NumberLit(i64),
    /// Unsigned literal: written like a number but used
    /// in unsigned positions. Soufflé is permissive here;
    /// we keep the source spelling and let type-checking
    /// downstream of the parser decide.
    UnsignedLit(u64),
    /// ADT branch construction: `$CallTool(name, args)`.
    Constructor { name: String, args: Vec<Term> },
    /// Record literal: `[c, t, agent, role, e]`.
    RecordLit(Vec<Term>),
    /// Functor call: `@json_get_str(json, field)`.
    Functor { name: String, args: Vec<Term> },
    /// Deterministic Soufflé builtin, distinct from an `@` user functor.
    /// Initially covers the string operations used by the public guard.
    Builtin { name: String, args: Vec<Term> },
    /// Sugar-level field access on a record-typed
    /// variable: `msg.agent`. Resolved by
    /// [`super::resolve_dots`] to a positional unpack
    /// of the record before downstream analyses run.
    FieldAccess { record: String, field: String },
    /// Arithmetic: `t1 + t2`, `t1 * t2`. We keep these
    /// as opaque binary nodes for the FD chase, since
    /// arithmetic only feeds comparisons and equality
    /// in the rule fragments we analyze.
    Arith {
        op: ArithOp,
        left: Box<Term>,
        right: Box<Term>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Comparison operator used in body literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A rule head or a body atom.
#[derive(Debug, Clone, Serialize)]
pub struct Atom {
    pub relation: String,
    pub args: Vec<Term>,
    pub span: SourceSpan,
}

/// A body literal — a single conjunct of a rule body.
#[derive(Debug, Clone, Serialize)]
pub enum Literal {
    /// `R(t1, ..., tn)` — positive atom.
    Pos(Atom),
    /// `!R(...)` or `not R(...)` — negative atom.
    Neg(Atom),
    /// Bridge-only preservation of negation around a
    /// grouped literal such as `!(A(x); B(x))`.
    Negation {
        literal: Box<Literal>,
        span: SourceSpan,
    },
    /// `t1 OP t2` — comparison or (dis)equality.
    Compare {
        op: CompareOp,
        left: Term,
        right: Term,
        span: SourceSpan,
    },
    /// `(b1 ; b2 ; ...)` — disjunction of conjunctive
    /// sub-bodies. Soufflé permits arbitrary nesting,
    /// but in practice policies only use a single level.
    /// The parse-program step expands rules containing
    /// disjunctions into multiple conjunctive rules so
    /// downstream analyses see only [`Literal::Pos`],
    /// [`Literal::Neg`], and [`Literal::Compare`].
    Disjunction {
        alternatives: Vec<Vec<Literal>>,
        span: SourceSpan,
    },
}

impl Literal {
    pub fn span(&self) -> &SourceSpan {
        match self {
            Literal::Pos(a) | Literal::Neg(a) => &a.span,
            Literal::Negation { span, .. } => span,
            Literal::Compare { span, .. } => span,
            Literal::Disjunction { span, .. } => span,
        }
    }
}

/// A Datalog rule: `Head(...) :- body1, body2, ...`.
#[derive(Debug, Clone, Serialize)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Literal>,
    pub span: SourceSpan,
}

/// A ground or partially-ground fact: `R("submit", "x").`.
/// Modelled as a rule with an empty body in callers that
/// don't need to distinguish them.
#[derive(Debug, Clone, Serialize)]
pub struct Fact {
    pub atom: Atom,
}

/// A complete program — types, relations, functors,
/// rules, and facts collected from one or more `.dl`
/// files (with `#include`s already inlined).
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub types: HashMap<String, TypeDecl>,
    pub relations: HashMap<String, RelationDecl>,
    pub functors: HashMap<String, FunctorDecl>,
    pub rules: Vec<Rule>,
    pub facts: Vec<Fact>,
}

impl Program {
    /// Rules whose head produces the given relation.
    pub fn rules_for<'a>(&'a self, relation: &'a str) -> impl Iterator<Item = &'a Rule> + 'a {
        self.rules
            .iter()
            .filter(move |r| r.head.relation == relation)
    }

    /// Whether a relation is declared as `.input`.
    pub fn is_edb(&self, relation: &str) -> bool {
        self.relations
            .get(relation)
            .map(|r| r.is_input)
            .unwrap_or(false)
    }
}

/* ────────────────────────────────────────────────────────
Pretty-printing helpers
──────────────────────────────────────────────────────── */

/// Format a rule as a one-line `head :- body.` string.
/// Used by analyses to surface the rule text inline with
/// findings so the auditor doesn't have to cross-reference
/// source by line number.
pub fn format_rule(rule: &Rule) -> String {
    let body: Vec<String> = rule.body.iter().map(format_literal).collect();
    if body.is_empty() {
        format!("{}.", format_atom(&rule.head))
    } else {
        format!("{} :- {}.", format_atom(&rule.head), body.join(", "))
    }
}

/// Format a single atom: `Relation(arg1, arg2, ...)`.
pub fn format_atom(atom: &Atom) -> String {
    let args: Vec<String> = atom.args.iter().map(format_term).collect();
    format!("{}({})", atom.relation, args.join(", "))
}

/// Format a body literal.
pub fn format_literal(lit: &Literal) -> String {
    match lit {
        Literal::Pos(a) => format_atom(a),
        Literal::Neg(a) => format!("!{}", format_atom(a)),
        Literal::Negation { literal, .. } => {
            format!("!{}", format_literal(literal))
        }
        Literal::Compare {
            op, left, right, ..
        } => format!(
            "{} {} {}",
            format_term(left),
            format_compare_op(*op),
            format_term(right)
        ),
        Literal::Disjunction { alternatives, .. } => {
            let alts: Vec<String> = alternatives
                .iter()
                .map(|alt| {
                    alt.iter()
                        .map(format_literal)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .collect();
            format!("({})", alts.join(" ; "))
        }
    }
}

/// Format a single term.
pub fn format_term(t: &Term) -> String {
    match t {
        Term::Var(n) => n.clone(),
        Term::Wildcard(_) => "_".into(),
        Term::StringLit(s) => format!("\"{}\"", s),
        Term::NumberLit(n) => n.to_string(),
        Term::UnsignedLit(n) => n.to_string(),
        Term::Constructor { name, args } => {
            let inner: Vec<String> = args.iter().map(format_term).collect();
            format!("${}({})", name, inner.join(", "))
        }
        Term::Functor { name, args } => {
            let inner: Vec<String> = args.iter().map(format_term).collect();
            format!("@{}({})", name, inner.join(", "))
        }
        Term::Builtin { name, args } => {
            let inner: Vec<String> = args.iter().map(format_term).collect();
            format!("{}({})", name, inner.join(", "))
        }
        Term::RecordLit(fs) => {
            let inner: Vec<String> = fs.iter().map(format_term).collect();
            format!("[{}]", inner.join(", "))
        }
        Term::Arith { op, left, right } => {
            let s = match op {
                ArithOp::Add => "+",
                ArithOp::Sub => "-",
                ArithOp::Mul => "*",
                ArithOp::Div => "/",
                ArithOp::Mod => "%",
            };
            format!("({} {} {})", format_term(left), s, format_term(right))
        }
        Term::FieldAccess { record, field } => format!("{}.{}", record, field),
    }
}

fn format_compare_op(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "=",
        CompareOp::Ne => "!=",
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}
