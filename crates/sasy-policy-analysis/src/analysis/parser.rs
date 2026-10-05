//! Recursive-descent parser for the Soufflé .dl subset
//! used in SASY policies.
//!
//! Consumes the token stream from [`super::lexer`] and
//! produces an [`super::ast::Program`]. Targets the
//! post-sugar form: dot-notation access has been
//! desugared upstream by `sugar.py`.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::ser::Error as _;
use serde::{Serialize, Serializer};

use super::ast::*;
use super::lexer::{Spanned, Token};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("lex error: {0}")]
    Lex(#[from] super::lexer::LexError),
    #[error("expected {expected}, got {got} at line {line}")]
    Unexpected {
        expected: String,
        got: String,
        line: u32,
    },
    #[error("unexpected end of input (expected {expected})")]
    Eof { expected: String },
    #[error("duplicate type declaration: {name}")]
    DuplicateType { name: String },
}

/// Diagnostics emitted only by the additive bridge parser.
///
/// The legacy [`ParseError`] API intentionally remains
/// unchanged for existing `parse()` consumers.
#[derive(Debug, thiserror::Error)]
pub enum BridgeParseError {
    #[error("{0}")]
    Parse(#[from] ParseError),
    #[error("unexpected end of input at line {line} (expected {expected})")]
    Eof { expected: String, line: u32 },
    #[error("duplicate type declaration at line {line}: {name}")]
    DuplicateType { name: String, line: u32 },
    #[error("duplicate relation declaration at line {line}: {name}")]
    DuplicateRelation { name: String, line: u32 },
}

/// Controls optional parser normalization and validation.
///
/// The default preserves the behavior of [`parse`]:
/// disjunctive bodies are expanded and repeated relation
/// declarations replace earlier declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseOptions {
    pub expand_body_disjunctions: bool,
    pub reject_duplicate_relations: bool,
    pub full_rule_spans: bool,
    pub preserve_aggregates: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            expand_body_disjunctions: true,
            reject_duplicate_relations: false,
            full_rule_spans: false,
            preserve_aggregates: false,
        }
    }
}

#[derive(Debug, Clone)]
struct PreservedAggregate {
    raw: String,
    span: SourceSpan,
}

/// A bridge-only parsed program.
///
/// Its custom serializer sorts all map keys and restores
/// preserved aggregate nodes without changing the legacy
/// [`Program`] data structures.
#[derive(Debug, Clone)]
pub struct BridgeProgram {
    program: Program,
    aggregates: BTreeMap<String, PreservedAggregate>,
    aggregate_marker: String,
}

#[derive(Serialize)]
struct StableProgram<'a> {
    types: BTreeMap<&'a str, &'a TypeDecl>,
    relations: BTreeMap<&'a str, &'a RelationDecl>,
    functors: BTreeMap<&'a str, &'a FunctorDecl>,
    rules: &'a [Rule],
    facts: &'a [Fact],
}

impl BridgeProgram {
    pub(crate) fn into_program(self) -> Program {
        self.program
    }
}

impl Serialize for BridgeProgram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let stable = StableProgram {
            types: self
                .program
                .types
                .iter()
                .map(|(name, decl)| (name.as_str(), decl))
                .collect(),
            relations: self
                .program
                .relations
                .iter()
                .map(|(name, decl)| (name.as_str(), decl))
                .collect(),
            functors: self
                .program
                .functors
                .iter()
                .map(|(name, decl)| (name.as_str(), decl))
                .collect(),
            rules: &self.program.rules,
            facts: &self.program.facts,
        };
        let mut value = serde_json::to_value(stable).map_err(S::Error::custom)?;
        restore_aggregate_nodes(&mut value, &self.aggregates, &self.aggregate_marker);
        value.serialize(serializer)
    }
}

fn restore_aggregate_nodes(
    value: &mut serde_json::Value,
    aggregates: &BTreeMap<String, PreservedAggregate>,
    marker: &str,
) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                restore_aggregate_nodes(value, aggregates, marker);
            }
        }
        serde_json::Value::Object(fields) => {
            let aggregate_id = fields
                .get("Functor")
                .and_then(serde_json::Value::as_object)
                .filter(|functor| {
                    functor.get("name").and_then(serde_json::Value::as_str) == Some(marker)
                })
                .and_then(|functor| functor.get("args"))
                .and_then(serde_json::Value::as_array)
                .and_then(|args| args.first())
                .and_then(|arg| arg.get("Var"))
                .and_then(serde_json::Value::as_str)
                .and_then(|id| aggregates.get(id).map(|aggregate| (id, aggregate)));
            if let Some((_id, aggregate)) = aggregate_id {
                *value = serde_json::json!({
                    "Aggregate": {
                        "raw": aggregate.raw,
                        "span": aggregate.span,
                    }
                });
                return;
            }
            for value in fields.values_mut() {
                restore_aggregate_nodes(value, aggregates, marker);
            }
        }
        _ => {}
    }
}

pub struct Parser<'a> {
    tokens: &'a [Spanned],
    pos: usize,
    file: String,
    /// Source-relative wildcard counter, so anonymous
    /// `_` arguments get unique names without colliding
    /// across rules.
    wildcard_id: u32,
    options: ParseOptions,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Spanned], file: impl Into<String>) -> Self {
        Self::new_with_options(tokens, file, ParseOptions::default())
    }

    fn new_with_options(
        tokens: &'a [Spanned],
        file: impl Into<String>,
        options: ParseOptions,
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            file: file.into(),
            wildcard_id: 0,
            options,
        }
    }

    /// Convenience: parse a complete program from source
    /// text. Wraps tokenize + parse.
    pub fn parse_str(src: &str, file: impl Into<String>) -> Result<Program, ParseError> {
        let tokens = super::lexer::tokenize(src)?;
        let mut p = Parser::new(&tokens, file);
        p.parse_program()
    }

    /* ── token helpers ──────────────────────────────────── */

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn peek_line(&self) -> u32 {
        self.tokens[self.pos].line
    }

    fn bump(&mut self) -> &Spanned {
        let t = &self.tokens[self.pos];
        if !matches!(t.token, Token::Eof) {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, want: &Token) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(want) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, want: &Token, what: &str) -> Result<(), ParseError> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(want) {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::Unexpected {
                expected: what.to_string(),
                got: format!("{}", self.peek()),
                line: self.peek_line(),
            })
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<String, ParseError> {
        let line = self.peek_line();
        match self.peek().clone() {
            Token::Ident(s) => {
                self.bump();
                Ok(s)
            }
            other => Err(ParseError::Unexpected {
                expected: what.to_string(),
                got: format!("{}", other),
                line,
            }),
        }
    }

    fn span_at(&self, line: u32) -> SourceSpan {
        SourceSpan::single(self.file.clone(), line)
    }

    fn span_between(&self, start_line: u32, end_line: u32) -> SourceSpan {
        SourceSpan {
            file: self.file.clone(),
            start_line,
            end_line,
        }
    }

    fn next_wildcard(&mut self) -> Term {
        let id = self.wildcard_id;
        self.wildcard_id += 1;
        Term::Wildcard(id)
    }

    /* ── top level ──────────────────────────────────────── */

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut program = Program::default();
        while !matches!(self.peek(), Token::Eof) {
            self.parse_top_item(&mut program)?;
        }
        if self.options.expand_body_disjunctions {
            program.rules = expand_disjunctions(program.rules);
        }
        Ok(program)
    }

    fn parse_top_item(&mut self, program: &mut Program) -> Result<(), ParseError> {
        match self.peek().clone() {
            Token::DotType => self.parse_type_decl(program),
            Token::DotDecl => self.parse_relation_decl(program),
            Token::DotInput => self.parse_io_marker(program, true, false),
            Token::DotOutput => self.parse_io_marker(program, false, true),
            Token::DotFunctor => self.parse_functor_decl(program),
            Token::DotPrintsize => {
                // .printsize R — skip the argument
                self.bump();
                let _ = self.expect_ident("relation name");
                Ok(())
            }
            Token::Ident(_) => self.parse_rule_or_fact(program),
            other => Err(ParseError::Unexpected {
                expected: "top-level declaration".to_string(),
                got: format!("{}", other),
                line: self.peek_line(),
            }),
        }
    }

    /* ── .type ──────────────────────────────────────────── */

    fn parse_type_decl(&mut self, program: &mut Program) -> Result<(), ParseError> {
        self.expect(&Token::DotType, ".type")?;
        let name = self.expect_ident("type name")?;
        match self.peek().clone() {
            Token::SubtypeArrow => {
                self.bump();
                let target = self.expect_ident("base type")?;
                self.insert_type(
                    program,
                    TypeDecl::Alias {
                        name,
                        of: TypeRef::from_str(&target),
                    },
                )
            }
            Token::Eq => {
                self.bump();
                match self.peek().clone() {
                    Token::LBracket => {
                        let fields = self.parse_field_list()?;
                        self.insert_type(program, TypeDecl::Record { name, fields })
                    }
                    Token::Ident(_) => self.parse_adt_decl(program, name),
                    other => Err(ParseError::Unexpected {
                        expected: "[ for record or branch name for ADT".into(),
                        got: format!("{}", other),
                        line: self.peek_line(),
                    }),
                }
            }
            other => Err(ParseError::Unexpected {
                expected: "<: or =".into(),
                got: format!("{}", other),
                line: self.peek_line(),
            }),
        }
    }

    fn insert_type(&self, program: &mut Program, decl: TypeDecl) -> Result<(), ParseError> {
        let name = decl.name().to_string();
        if program.types.contains_key(&name) {
            return Err(ParseError::DuplicateType { name });
        }
        program.types.insert(name, decl);
        Ok(())
    }

    fn parse_field_list(&mut self) -> Result<Vec<(String, TypeRef)>, ParseError> {
        self.expect(&Token::LBracket, "[")?;
        let mut fields = Vec::new();
        if !matches!(self.peek(), Token::RBracket) {
            loop {
                let fname = self.expect_ident("field name")?;
                self.expect(&Token::Colon, ":")?;
                let tname = self.expect_ident("type")?;
                fields.push((fname, TypeRef::from_str(&tname)));
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RBracket, "]")?;
        Ok(fields)
    }

    fn parse_adt_decl(&mut self, program: &mut Program, name: String) -> Result<(), ParseError> {
        let mut branches = Vec::new();
        loop {
            let branch_name = self.expect_ident("branch name")?;
            self.expect(&Token::LBrace, "{")?;
            let mut fields = Vec::new();
            if !matches!(self.peek(), Token::RBrace) {
                loop {
                    let fname = self.expect_ident("branch field name")?;
                    self.expect(&Token::Colon, ":")?;
                    let tname = self.expect_ident("branch field type")?;
                    fields.push((fname, TypeRef::from_str(&tname)));
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Token::RBrace, "}")?;
            branches.push(AdtBranch {
                name: branch_name,
                fields,
            });
            if !self.eat(&Token::Pipe) {
                break;
            }
        }
        self.insert_type(program, TypeDecl::Adt { name, branches })
    }

    /* ── .decl, .input, .output ─────────────────────────── */

    fn parse_relation_decl(&mut self, program: &mut Program) -> Result<(), ParseError> {
        let line = self.peek_line();
        self.expect(&Token::DotDecl, ".decl")?;
        let name = self.expect_ident("relation name")?;
        self.expect(&Token::LParen, "(")?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Token::RParen) {
            loop {
                let pname = self.expect_ident("parameter name")?;
                self.expect(&Token::Colon, ":")?;
                let ptype = self.expect_ident("parameter type")?;
                params.push((pname, TypeRef::from_str(&ptype)));
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RParen, ")")?;
        // Optional Soufflé qualifiers immediately after
        // the parameter list. Match a known set so we
        // don't accidentally swallow the next rule head.
        loop {
            let consume = match self.peek() {
                Token::Ident(s) => matches!(
                    s.as_str(),
                    "inline"
                        | "no_inline"
                        | "brie"
                        | "btree"
                        | "btree_delete"
                        | "eqrel"
                        | "override"
                        | "magic"
                        | "no_magic"
                ),
                _ => false,
            };
            if !consume {
                break;
            }
            self.bump();
        }
        let span = self.span_at(line);
        let prev = program.relations.remove(&name);
        let (is_input, is_output) = prev
            .map(|r| (r.is_input, r.is_output))
            .unwrap_or((false, false));
        program.relations.insert(
            name.clone(),
            RelationDecl {
                name,
                params,
                is_input,
                is_output,
                span,
            },
        );
        Ok(())
    }

    fn parse_io_marker(
        &mut self,
        program: &mut Program,
        input: bool,
        output: bool,
    ) -> Result<(), ParseError> {
        let line = self.peek_line();
        self.bump();
        let name = self.expect_ident("relation name")?;
        // Transport options describe how Souffle reads/writes the relation,
        // not its logical tuples. Validate and consume them while retaining
        // only the input/output flags in this analysis AST.
        if self.eat(&Token::LParen) {
            if !matches!(self.peek(), Token::RParen) {
                loop {
                    self.expect_ident("I/O option name")?;
                    self.expect(&Token::Eq, "=")?;
                    match self.peek() {
                        Token::Ident(_)
                        | Token::String(_)
                        | Token::Int(_)
                        | Token::Uint(_)
                        | Token::KwTrue
                        | Token::KwFalse => {
                            self.bump();
                        }
                        other => {
                            return Err(ParseError::Unexpected {
                                expected: "scalar I/O option value".to_string(),
                                got: format!("{}", other),
                                line: self.peek_line(),
                            });
                        }
                    }
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Token::RParen, ")")?;
        }
        let entry = program
            .relations
            .entry(name.clone())
            .or_insert(RelationDecl {
                name,
                params: Vec::new(),
                is_input: false,
                is_output: false,
                span: self.span_at(line),
            });
        if input {
            entry.is_input = true;
        }
        if output {
            entry.is_output = true;
        }
        Ok(())
    }

    /* ── .functor ───────────────────────────────────────── */

    fn parse_functor_decl(&mut self, program: &mut Program) -> Result<(), ParseError> {
        let line = self.peek_line();
        self.expect(&Token::DotFunctor, ".functor")?;
        let name = self.expect_ident("functor name")?;
        self.expect(&Token::LParen, "(")?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Token::RParen) {
            loop {
                let _pname = self.expect_ident("functor parameter name")?;
                self.expect(&Token::Colon, ":")?;
                let ptype = self.expect_ident("parameter type")?;
                params.push(TypeRef::from_str(&ptype));
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RParen, ")")?;
        self.expect(&Token::Colon, ":")?;
        let ret = self.expect_ident("return type")?;
        let mut stateful = false;
        if matches!(self.peek(), Token::KwStateful) {
            self.bump();
            stateful = true;
        }
        program.functors.insert(
            name.clone(),
            FunctorDecl {
                name,
                params,
                return_type: TypeRef::from_str(&ret),
                stateful,
                span: self.span_at(line),
            },
        );
        Ok(())
    }

    /* ── rules and facts ────────────────────────────────── */

    fn parse_rule_or_fact(&mut self, program: &mut Program) -> Result<(), ParseError> {
        let line = self.peek_line();
        let head = self.parse_atom()?;
        match self.peek() {
            Token::Dot => {
                self.bump();
                program.facts.push(Fact { atom: head });
                Ok(())
            }
            Token::ColonDash => {
                self.bump();
                let body = self.parse_body()?;
                let end_line = self.peek_line();
                self.expect(&Token::Dot, ".")?;
                let span = if self.options.full_rule_spans {
                    self.span_between(line, end_line)
                } else {
                    self.span_at(line)
                };
                program.rules.push(Rule { head, body, span });
                Ok(())
            }
            other => Err(ParseError::Unexpected {
                expected: ". or :-".into(),
                got: format!("{}", other),
                line: self.peek_line(),
            }),
        }
    }

    fn parse_atom(&mut self) -> Result<Atom, ParseError> {
        let line = self.peek_line();
        let name = self.expect_ident("relation name")?;
        self.expect(&Token::LParen, "(")?;
        let args = self.parse_term_list()?;
        self.expect(&Token::RParen, ")")?;
        Ok(Atom {
            relation: name,
            args,
            span: self.span_at(line),
        })
    }

    fn parse_body(&mut self) -> Result<Vec<Literal>, ParseError> {
        let mut out = Vec::new();
        loop {
            let lit = self.parse_literal()?;
            out.push(lit);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(out)
    }

    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        let line = self.peek_line();
        // Negation prefix.
        let negated = match self.peek() {
            Token::Bang | Token::KwNot => {
                self.bump();
                true
            }
            _ => false,
        };

        // After negation an atom must follow.
        if negated {
            if !self.options.expand_body_disjunctions
                && matches!(self.peek(), Token::LParen)
                && matches!(self.classify_lparen()?, LParenKind::Disjunction)
            {
                let literal = self.parse_disjunction(line)?;
                let span = literal.span().clone();
                return Ok(Literal::Negation {
                    literal: Box::new(literal),
                    span,
                });
            }
            let atom = self.parse_atom()?;
            return Ok(Literal::Neg(atom));
        }

        // `(...)` at literal start: could be a disjunction
        // group `(a, b ; c, d)`, a degenerate parenthesized
        // conjunction `(a, b)`, or a parenthesized term in
        // a comparison `(t1 + t2) > t3`. Disambiguate via
        // depth-0 lookahead.
        if matches!(self.peek(), Token::LParen) {
            match self.classify_lparen()? {
                LParenKind::Disjunction => return self.parse_disjunction(line),
                LParenKind::TermGroup => {
                    let left = self.parse_term()?;
                    let op = self.parse_compare_op()?;
                    let right = self.parse_term()?;
                    return Ok(Literal::Compare {
                        op,
                        left,
                        right,
                        span: self.span_at(line),
                    });
                }
            }
        }

        // Lookahead: is this an atom (Ident followed by `(`)
        // at the head of a positive literal, or a comparison
        // expression?
        let is_relation_head = matches!(self.peek(), Token::Ident(name) if builtin_arity(name).is_none())
            && matches!(self.peek_n(1), Some(Token::LParen));
        if is_relation_head {
            let atom = self.parse_atom()?;
            return Ok(Literal::Pos(atom));
        }

        // Otherwise it's a comparison: term OP term.
        let left = self.parse_term()?;
        let op = self.parse_compare_op()?;
        let right = self.parse_term()?;
        Ok(Literal::Compare {
            op,
            left,
            right,
            span: self.span_at(line),
        })
    }

    /// Look forward through the matching `)` to decide
    /// whether a `(...)` at literal-start is a disjunction
    /// group (containing `;` or `,` at the same depth) or
    /// a parenthesized term participating in a comparison.
    /// We commit to `Disjunction` when we see `;` or `,`
    /// at depth 0 (covering both `(a;b)` and the
    /// degenerate `(a, b)` grouping); otherwise the parens
    /// surround a single expression in a comparison.
    fn classify_lparen(&self) -> Result<LParenKind, ParseError> {
        debug_assert!(matches!(self.peek(), Token::LParen));
        let mut depth = 0i32;
        let mut i = self.pos;
        loop {
            let tok = self
                .tokens
                .get(i)
                .map(|s| &s.token)
                .ok_or_else(|| ParseError::Eof {
                    expected: ") closing literal group".into(),
                })?;
            match tok {
                Token::LParen => depth += 1,
                Token::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(LParenKind::TermGroup);
                    }
                }
                Token::Semicolon | Token::Comma if depth == 1 => {
                    return Ok(LParenKind::Disjunction);
                }
                Token::Eof => {
                    return Err(ParseError::Eof {
                        expected: ") closing literal group".into(),
                    });
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn parse_disjunction(&mut self, line: u32) -> Result<Literal, ParseError> {
        self.expect(&Token::LParen, "(")?;
        let mut alternatives: Vec<Vec<Literal>> = Vec::new();
        let mut current: Vec<Literal> = Vec::new();
        let end_line;
        loop {
            current.push(self.parse_literal()?);
            match self.peek() {
                Token::Comma => {
                    self.bump();
                }
                Token::Semicolon => {
                    self.bump();
                    alternatives.push(std::mem::take(&mut current));
                }
                Token::RParen => {
                    end_line = self.peek_line();
                    self.bump();
                    alternatives.push(current);
                    break;
                }
                other => {
                    return Err(ParseError::Unexpected {
                        expected: ", or ; or )".into(),
                        got: format!("{}", other),
                        line: self.peek_line(),
                    });
                }
            }
        }
        Ok(Literal::Disjunction {
            alternatives,
            span: if self.options.full_rule_spans {
                self.span_between(line, end_line)
            } else {
                self.span_at(line)
            },
        })
    }

    fn parse_compare_op(&mut self) -> Result<CompareOp, ParseError> {
        let op = match self.peek() {
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            other => {
                return Err(ParseError::Unexpected {
                    expected: "comparison operator".into(),
                    got: format!("{}", other),
                    line: self.peek_line(),
                });
            }
        };
        self.bump();
        Ok(op)
    }

    fn peek_n(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n).map(|s| &s.token)
    }
}

/// Distinguishes how an opening `(` at literal-start is
/// being used in the body. Returned by
/// [`Parser::classify_lparen`].
#[derive(Debug, Clone, Copy)]
enum LParenKind {
    /// `(b1[, b2[...]] [; ...])` — disjunction or grouped
    /// conjunction.
    Disjunction,
    /// `(t1 op_or_arith t2)` — parenthesized expression
    /// participating in an outer comparison.
    TermGroup,
}

impl<'a> Parser<'a> {
    /* ── terms ──────────────────────────────────────────── */

    fn parse_term_list(&mut self) -> Result<Vec<Term>, ParseError> {
        let mut out = Vec::new();
        if !matches!(self.peek(), Token::RParen | Token::RBracket) {
            loop {
                out.push(self.parse_term()?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Parse an additive expression: `t (+|-) t (+|-) t ...`.
    fn parse_term(&mut self) -> Result<Term, ParseError> {
        let mut left = self.parse_term_mul()?;
        loop {
            let op = match self.peek() {
                Token::Plus => ArithOp::Add,
                Token::Minus => ArithOp::Sub,
                _ => break,
            };
            self.bump();
            let right = self.parse_term_mul()?;
            left = Term::Arith {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_term_mul(&mut self) -> Result<Term, ParseError> {
        let mut left = self.parse_term_atom()?;
        loop {
            let op = match self.peek() {
                Token::Star => ArithOp::Mul,
                Token::Slash => ArithOp::Div,
                Token::Percent => ArithOp::Mod,
                _ => break,
            };
            self.bump();
            let right = self.parse_term_atom()?;
            left = Term::Arith {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_term_atom(&mut self) -> Result<Term, ParseError> {
        match self.peek().clone() {
            Token::Ident(s) if s == "_" => {
                self.bump();
                Ok(self.next_wildcard())
            }
            Token::Ident(s)
                if builtin_arity(&s).is_some() && matches!(self.peek_n(1), Some(Token::LParen)) =>
            {
                let line = self.peek_line();
                self.bump();
                self.expect(&Token::LParen, "(")?;
                let args = self.parse_term_list()?;
                self.expect(&Token::RParen, ")")?;
                let (minimum, maximum) = builtin_arity(&s).unwrap();
                if args.len() < minimum || maximum.is_some_and(|maximum| args.len() > maximum) {
                    return Err(ParseError::Unexpected {
                        expected: match maximum {
                            Some(maximum) => format!("{maximum} arguments for {s}"),
                            None => format!("at least {minimum} arguments for {s}"),
                        },
                        got: format!("{} arguments", args.len()),
                        line,
                    });
                }
                Ok(Term::Builtin { name: s, args })
            }
            Token::Ident(s) => {
                let var_line = self.peek_line();
                self.bump();
                // Sugar-level field access: `var.field`.
                // Only treat `Ident . Ident` as
                // FieldAccess when all three tokens are
                // on the same source line — otherwise
                // the `.` is the rule terminator and the
                // following Ident is the next rule's head.
                let dot_line = self.peek_line();
                let field_line = self.tokens.get(self.pos + 1).map(|s| s.line).unwrap_or(0);
                let dot_after_ident = matches!(self.peek(), Token::Dot)
                    && matches!(self.peek_n(1), Some(Token::Ident(_)));
                if dot_after_ident && dot_line == var_line && field_line == var_line {
                    self.bump(); // consume Dot
                    let field = self.expect_ident("field name")?;
                    return Ok(Term::FieldAccess { record: s, field });
                }
                Ok(Term::Var(s))
            }
            Token::String(s) => {
                self.bump();
                Ok(Term::StringLit(s))
            }
            Token::Int(n) => {
                self.bump();
                // Fits in i64 (covers all negatives) → NumberLit;
                // otherwise it's a large positive → UnsignedLit.
                if n <= i64::MAX as i128 {
                    Ok(Term::NumberLit(n as i64))
                } else {
                    Ok(Term::UnsignedLit(n as u64))
                }
            }
            Token::Uint(n) => {
                self.bump();
                Ok(Term::UnsignedLit(n))
            }
            Token::KwNil => {
                self.bump();
                // Nil is the empty record. Soufflé treats
                // it as a constant; we model it as an
                // empty RecordLit so existing record-aware
                // matching works.
                Ok(Term::RecordLit(Vec::new()))
            }
            Token::KwTrue => {
                self.bump();
                Ok(Term::NumberLit(1))
            }
            Token::KwFalse => {
                self.bump();
                Ok(Term::NumberLit(0))
            }
            Token::KwAs => {
                // Soufflé `as(x, MyType)` casts a value;
                // for our analysis purposes the cast is
                // erased — the inner value's class
                // semantics already apply.
                self.bump();
                self.expect(&Token::LParen, "(")?;
                let inner = self.parse_term()?;
                self.expect(&Token::Comma, ",")?;
                let _ty = self.expect_ident("cast target type")?;
                self.expect(&Token::RParen, ")")?;
                Ok(inner)
            }
            Token::Minus => {
                self.bump();
                let inner = self.parse_term_atom()?;
                match inner {
                    Term::NumberLit(n) => Ok(Term::NumberLit(-n)),
                    other => Ok(Term::Arith {
                        op: ArithOp::Sub,
                        left: Box::new(Term::NumberLit(0)),
                        right: Box::new(other),
                    }),
                }
            }
            Token::Dollar => {
                self.bump();
                let name = self.expect_ident("constructor name")?;
                self.expect(&Token::LParen, "(")?;
                let args = self.parse_term_list()?;
                self.expect(&Token::RParen, ")")?;
                Ok(Term::Constructor { name, args })
            }
            Token::At => {
                self.bump();
                let name = self.expect_ident("functor name")?;
                self.expect(&Token::LParen, "(")?;
                let args = self.parse_term_list()?;
                self.expect(&Token::RParen, ")")?;
                Ok(Term::Functor { name, args })
            }
            Token::LBracket => {
                self.bump();
                let elems = self.parse_term_list()?;
                self.expect(&Token::RBracket, "]")?;
                Ok(Term::RecordLit(elems))
            }
            Token::LParen => {
                self.bump();
                let inner = self.parse_term()?;
                self.expect(&Token::RParen, ")")?;
                Ok(inner)
            }
            other => Err(ParseError::Unexpected {
                expected: "term".into(),
                got: format!("{}", other),
                line: self.peek_line(),
            }),
        }
    }
}

/// Only deterministic, single-result builtins supported by this analysis
/// subset. Generators such as range/autoinc must not acquire functor
/// determinism merely because they have function-call syntax.
fn builtin_arity(name: &str) -> Option<(usize, Option<usize>)> {
    match name {
        "strlen" | "to_number" => Some((1, Some(1))),
        "substr" => Some((3, Some(3))),
        "cat" => Some((0, None)),
        _ => None,
    }
}

// Public re-exports of helpers callers might want.
pub use ParseError as Error;

/// Parse a `.dl` source string into a [`Program`]. The
/// `file` argument is used only for source-location
/// reporting on the resulting AST nodes.
pub fn parse(src: &str, file: impl Into<String>) -> Result<Program, ParseError> {
    Parser::parse_str(src, file)
}

/// Parse a `.dl` source string using explicit options.
///
/// This additive entry point lets admission bridges retain
/// syntax that the default parser normalizes away.
pub fn parse_with_options(
    src: &str,
    file: impl Into<String>,
    options: ParseOptions,
) -> Result<BridgeProgram, BridgeParseError> {
    let file = file.into();
    let source_tokens = super::lexer::tokenize(src)
        .map_err(ParseError::from)
        .map_err(BridgeParseError::from)?;
    if options.reject_duplicate_relations {
        if let Some((name, line)) = first_duplicate_relation(&source_tokens) {
            return Err(BridgeParseError::DuplicateRelation { name, line });
        }
    }
    let (prepared, aggregates, aggregate_marker) = if options.preserve_aggregates {
        prepare_aggregates(src, &file)
    } else {
        (src.to_string(), BTreeMap::new(), String::new())
    };
    let tokens = bridge_tokens(&prepared)
        .map_err(ParseError::from)
        .map_err(BridgeParseError::from)?;
    let mut parser = Parser::new_with_options(&tokens, file, options);
    let program = parser
        .parse_program()
        .map_err(|error| bridge_error(error, &source_tokens))?;
    Ok(BridgeProgram {
        program,
        aggregates,
        aggregate_marker,
    })
}

fn bridge_tokens(source: &str) -> Result<Vec<Spanned>, super::lexer::LexError> {
    let tokens = super::lexer::tokenize(source)?;
    let mut out = Vec::with_capacity(tokens.len());
    for token in tokens {
        if let Token::Int(value) = token.token {
            if out.last().is_some_and(token_can_end_term) {
                if let Some(positive) = value.checked_neg().filter(|_| value < 0) {
                    out.push(Spanned {
                        token: Token::Minus,
                        line: token.line,
                    });
                    out.push(Spanned {
                        token: Token::Int(positive),
                        line: token.line,
                    });
                    continue;
                }
            }
            out.push(Spanned {
                token: Token::Int(value),
                line: token.line,
            });
        } else {
            out.push(token);
        }
    }
    Ok(out)
}

fn token_can_end_term(token: &Spanned) -> bool {
    matches!(
        token.token,
        Token::Ident(_)
            | Token::String(_)
            | Token::Int(_)
            | Token::Uint(_)
            | Token::KwNil
            | Token::KwTrue
            | Token::KwFalse
            | Token::RParen
            | Token::RBracket
    )
}

fn bridge_error(error: ParseError, tokens: &[Spanned]) -> BridgeParseError {
    let eof_line = tokens.last().map(|token| token.line).unwrap_or(1);
    match error {
        ParseError::Eof { expected } => BridgeParseError::Eof {
            expected,
            line: eof_line,
        },
        ParseError::DuplicateType { name } => BridgeParseError::DuplicateType {
            line: duplicate_type_line(tokens, &name).unwrap_or(eof_line),
            name,
        },
        other => BridgeParseError::Parse(other),
    }
}

fn first_duplicate_relation(tokens: &[Spanned]) -> Option<(String, u32)> {
    let mut declared = HashSet::new();
    for pair in tokens.windows(2) {
        if matches!(pair[0].token, Token::DotDecl) {
            if let Token::Ident(name) = &pair[1].token {
                if !declared.insert(name.clone()) {
                    return Some((name.clone(), pair[0].line));
                }
            }
        }
    }
    None
}

fn duplicate_type_line(tokens: &[Spanned], duplicate_name: &str) -> Option<u32> {
    let mut seen = false;
    for pair in tokens.windows(2) {
        if matches!(pair[0].token, Token::DotType) {
            if let Token::Ident(name) = &pair[1].token {
                if name == duplicate_name {
                    if seen {
                        return Some(pair[0].line);
                    }
                    seen = true;
                }
            }
        }
    }
    None
}

fn prepare_aggregates(
    source: &str,
    file: &str,
) -> (String, BTreeMap<String, PreservedAggregate>, String) {
    let ranges = aggregate_ranges(source);
    if ranges.is_empty() {
        return (source.to_string(), BTreeMap::new(), String::new());
    }

    let mut marker = "__sasy_bridge_preserved_aggregate".to_string();
    while source.contains(&marker) {
        marker.push('_');
    }
    let mut prepared = String::with_capacity(source.len());
    let mut aggregates = BTreeMap::new();
    let mut copied_until = 0;
    for (index, (start, end, start_line, end_line)) in ranges.into_iter().enumerate() {
        prepared.push_str(&source[copied_until..start]);
        let id = format!("__sasy_bridge_aggregate_{index}");
        prepared.push('@');
        prepared.push_str(&marker);
        prepared.push('(');
        prepared.push_str(&id);
        prepared.push(')');
        for byte in source[start..end].bytes() {
            if byte == b'\n' {
                prepared.push('\n');
            }
        }
        aggregates.insert(
            id,
            PreservedAggregate {
                raw: source[start..end].to_string(),
                span: SourceSpan {
                    file: file.to_string(),
                    start_line,
                    end_line,
                },
            },
        );
        copied_until = end;
    }
    prepared.push_str(&source[copied_until..]);
    (prepared, aggregates, marker)
}

fn aggregate_ranges(source: &str) -> Vec<(usize, usize, u32, u32)> {
    let bytes = source.as_bytes();
    let mut ranges = Vec::new();
    let mut index = 0;
    let mut line = 1;

    while index < bytes.len() {
        match bytes[index] {
            b'"' => skip_string(bytes, &mut index, &mut line),
            b'/' if bytes.get(index + 1) == Some(&b'/') => skip_line_comment(bytes, &mut index),
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                skip_block_comment(bytes, &mut index, &mut line)
            }
            byte if is_ident_start(byte) => {
                let start = index;
                index += 1;
                while index < bytes.len() && is_ident_continue(bytes[index]) {
                    index += 1;
                }
                let name = &source[start..index];
                if matches!(name, "count" | "sum" | "min" | "max" | "mean") {
                    if let Some(end) = aggregate_end(bytes, index) {
                        let start_line = line;
                        let end_line = start_line
                            + source[start..end]
                                .bytes()
                                .filter(|byte| *byte == b'\n')
                                .count() as u32;
                        ranges.push((start, end, start_line, end_line));
                        line = end_line;
                        index = end;
                    }
                }
            }
            b'\n' => {
                line += 1;
                index += 1;
            }
            _ => index += 1,
        }
    }
    ranges
}

fn aggregate_end(bytes: &[u8], mut index: usize) -> Option<usize> {
    while index < bytes.len() && bytes[index] != b':' {
        if matches!(bytes[index], b',' | b'.' | b';') {
            return None;
        }
        index += 1;
    }
    if bytes.get(index) != Some(&b':') {
        return None;
    }
    index += 1;
    while matches!(bytes.get(index), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        index += 1;
    }
    if bytes.get(index) != Some(&b'{') {
        return None;
    }

    let mut depth = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let mut ignored_line = 0;
                skip_string(bytes, &mut index, &mut ignored_line);
            }
            b'/' if bytes.get(index + 1) == Some(&b'/') => skip_line_comment(bytes, &mut index),
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                let mut ignored_line = 0;
                skip_block_comment(bytes, &mut index, &mut ignored_line);
            }
            b'{' => {
                depth += 1;
                index += 1;
            }
            b'}' => {
                depth -= 1;
                index += 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => index += 1,
        }
    }
    None
}

fn skip_string(bytes: &[u8], index: &mut usize, line: &mut u32) {
    *index += 1;
    while *index < bytes.len() {
        match bytes[*index] {
            b'\\' => *index = (*index + 2).min(bytes.len()),
            b'"' => {
                *index += 1;
                return;
            }
            b'\n' => {
                *line += 1;
                *index += 1;
            }
            _ => *index += 1,
        }
    }
}

fn skip_line_comment(bytes: &[u8], index: &mut usize) {
    *index += 2;
    while *index < bytes.len() && bytes[*index] != b'\n' {
        *index += 1;
    }
}

fn skip_block_comment(bytes: &[u8], index: &mut usize, line: &mut u32) {
    *index += 2;
    while *index < bytes.len() {
        if bytes[*index] == b'\n' {
            *line += 1;
        }
        if bytes[*index] == b'*' && bytes.get(*index + 1) == Some(&b'/') {
            *index += 2;
            return;
        }
        *index += 1;
    }
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_ident_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Expand any [`Literal::Disjunction`] in a rule's body
/// into multiple conjunctive rules, so downstream analyses
/// only see [`Literal::Pos`] / [`Literal::Neg`] /
/// [`Literal::Compare`]. The expansion is the cartesian
/// product over disjunctions; each generated rule carries
/// the original rule's source span.
pub fn expand_disjunctions(rules: Vec<Rule>) -> Vec<Rule> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let bodies = expand_body(&rule.body);
        for body in bodies {
            out.push(Rule {
                head: rule.head.clone(),
                body,
                span: rule.span.clone(),
            });
        }
    }
    out
}

/// Cartesian-product expand a body containing
/// disjunctions. Returns a vec of conjunctive bodies.
fn expand_body(body: &[Literal]) -> Vec<Vec<Literal>> {
    let mut acc: Vec<Vec<Literal>> = vec![Vec::new()];
    for lit in body {
        match lit {
            Literal::Disjunction { alternatives, .. } => {
                let mut next: Vec<Vec<Literal>> = Vec::new();
                for prefix in &acc {
                    for alt in alternatives {
                        for sub_body in expand_body(alt) {
                            let mut combined = prefix.clone();
                            combined.extend(sub_body);
                            next.push(combined);
                        }
                    }
                }
                acc = next;
            }
            other => {
                for body in &mut acc {
                    body.push(other.clone());
                }
            }
        }
    }
    acc
}

/// Index rules by head relation name for fast lookup.
pub fn index_rules_by_head(program: &Program) -> HashMap<String, Vec<usize>> {
    let mut out: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in program.rules.iter().enumerate() {
        out.entry(r.head.relation.clone()).or_default().push(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(src: &str) -> Program {
        parse(src, "test.dl").expect("parse")
    }

    /// Pin down the parser's line tracking with a few
    /// known rules at specific line offsets, including
    /// rules separated by blank lines and comments.
    #[test]
    fn rule_spans_match_source_lines() {
        let src = "\
// line 1: comment
// line 2: comment
A(x) :- B(x).
\n\
// line 5: comment

C(x) :-
    D(x),
    E(x).
\n\
F(x) :- G(x).
";
        let prog = parse(src, "test.dl").unwrap();
        let lines: Vec<u32> = prog.rules.iter().map(|r| r.span.start_line).collect();
        // Expected: A at 3, C at 7, F at 11.
        assert_eq!(lines, vec![3, 7, 11], "got {:?}", lines);
        assert!(
            prog.rules
                .iter()
                .all(|rule| rule.span.end_line == rule.span.start_line),
            "legacy parse() must keep single-line rule spans: {:?}",
            prog.rules
        );
    }

    #[test]
    fn legacy_parse_errors_keep_their_original_shapes_and_messages() {
        let error = parse(
            ".type Request <: symbol\n.type Request <: symbol\n",
            "test.dl",
        )
        .unwrap_err();

        assert!(matches!(
            &error,
            ParseError::DuplicateType { name } if name == "Request"
        ));
        assert_eq!(error.to_string(), "duplicate type declaration: Request");
        assert_eq!(
            ParseError::Eof {
                expected: "term".into()
            }
            .to_string(),
            "unexpected end of input (expected term)"
        );
    }

    #[test]
    fn default_parse_replaces_duplicate_relation_declaration() {
        let program = parse(
            ".decl Host(r: symbol, value: symbol)\n\
             .decl Host(request: symbol)\n",
            "test.dl",
        )
        .unwrap();

        let host = program.relations.get("Host").unwrap();
        assert_eq!(host.params.len(), 1);
        assert_eq!(host.params[0].0, "request");
    }

    #[test]
    fn relation_decl_after_io_marker_is_not_a_duplicate() {
        let program = parse(
            ".input Host\n.decl Host(r: symbol, value: symbol)\n",
            "test.dl",
        )
        .unwrap();

        let host = program.relations.get("Host").unwrap();
        assert!(host.is_input);
        assert_eq!(host.params.len(), 2);
    }

    #[test]
    fn options_preserve_body_disjunction_without_rule_duplication() {
        let program = parse_with_options(
            "block(R) :- (http_host(R, \"x\"); uri_path(R, \"/admin\")).\n",
            "test.dl",
            ParseOptions {
                expand_body_disjunctions: false,
                ..ParseOptions::default()
            },
        )
        .unwrap();

        assert_eq!(program.program.rules.len(), 1);
        assert!(matches!(
            program.program.rules[0].body.as_slice(),
            [Literal::Disjunction { alternatives, .. }]
                if alternatives.len() == 2
        ));
    }

    #[test]
    fn options_preserve_negation_over_grouped_disjunction() {
        let program = parse_with_options(
            "block(R) :- !(helper_a(R); helper_b(R)).\n",
            "test.dl",
            ParseOptions {
                expand_body_disjunctions: false,
                full_rule_spans: true,
                ..ParseOptions::default()
            },
        )
        .unwrap();

        assert!(matches!(
            program.program.rules[0].body.as_slice(),
            [Literal::Negation { literal, span }]
                if matches!(literal.as_ref(), Literal::Disjunction { .. })
                    && span.start_line == 1
                    && span.end_line == 1
        ));
    }

    #[test]
    fn bridge_only_subtraction_token_repair_preserves_plain_parse() {
        let options = ParseOptions {
            expand_body_disjunctions: false,
            ..ParseOptions::default()
        };
        for source in ["block(R) :- X-1 = 2.\n", "block(R) :- X - 1 = 2.\n"] {
            let program = parse_with_options(source, "test.dl", options).unwrap();
            assert!(matches!(
                program.program.rules[0].body.as_slice(),
                [Literal::Compare {
                    left: Term::Arith {
                        op: ArithOp::Sub,
                        ..
                    },
                    ..
                }]
            ));
        }

        assert!(parse("block(R) :- X-1 = 2.\n", "test.dl").is_err());
    }

    #[test]
    fn default_parse_still_expands_body_disjunction() {
        let program = parse(
            "block(R) :- (http_host(R, \"x\"); uri_path(R, \"/admin\")).\n",
            "test.dl",
        )
        .unwrap();

        assert_eq!(program.rules.len(), 2);
        assert!(program.rules.iter().all(|rule| {
            rule.body
                .iter()
                .all(|literal| !matches!(literal, Literal::Disjunction { .. }))
        }));
    }

    #[test]
    fn options_reject_duplicate_relation_declaration() {
        let error = parse_with_options(
            ".decl Host(r: symbol, value: symbol)\n\
             .decl Host(r: symbol, value: symbol)\n",
            "test.dl",
            ParseOptions {
                reject_duplicate_relations: true,
                ..ParseOptions::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            BridgeParseError::DuplicateRelation {
                name,
                line: 2
            } if name == "Host"
        ));
    }

    #[test]
    fn options_eof_error_carries_line_after_trailing_newline() {
        let error =
            parse_with_options("R(x) :- (S(x)\n", "test.dl", ParseOptions::default()).unwrap_err();

        assert!(matches!(error, BridgeParseError::Eof { line: 2, .. }));
    }

    #[test]
    fn parses_field_access_term() {
        // Sugar-level dot notation: `msg.agent` produces
        // a FieldAccess term that the resolve_dots pass
        // will rewrite to a positional record unpack.
        let prog = p(r#"R(x) :- SentMessage(id, msg), msg.agent = "FDAHandler"."#);
        let body = &prog.rules[0].body;
        // Find the Compare with FieldAccess on left.
        let found = body.iter().any(|lit| {
            matches!(lit,
            Literal::Compare {
                op: CompareOp::Eq,
                left: Term::FieldAccess { record, field },
                ..
            } if record == "msg" && field == "agent")
        });
        assert!(found, "expected FieldAccess term, got {:?}", body);
    }

    #[test]
    fn rule_terminator_after_ident_not_field_access() {
        // The `.` after `R(x)` is the rule terminator,
        // not the start of a field access — even though
        // the next token is an Ident on a later line.
        let prog = p(r#"
            S(x) :- R(x).
            T(y) :- R(y).
        "#);
        assert_eq!(prog.rules.len(), 2);
        assert!(prog.rules.iter().all(|r| r.body.iter().all(|lit| !matches!(
            lit,
            Literal::Compare {
                left: Term::FieldAccess { .. },
                ..
            }
        ))));
    }

    #[test]
    fn parses_simple_rule() {
        let prog = p(r#"IsAuthorized(idx) :- Actions(idx, _)."#);
        assert_eq!(prog.rules.len(), 1);
        let r = &prog.rules[0];
        assert_eq!(r.head.relation, "IsAuthorized");
        assert_eq!(r.body.len(), 1);
    }

    #[test]
    fn parses_constructor_pattern() {
        let src = r#"IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _)."#;
        let prog = p(src);
        let body = &prog.rules[0].body;
        assert_eq!(body.len(), 2);
        let lit = &body[1];
        match lit {
            Literal::Compare {
                op: CompareOp::Eq,
                right: Term::Constructor { name, args },
                ..
            } => {
                assert_eq!(name, "CallTool");
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected constructor compare, got {:?}", other),
        }
    }

    #[test]
    fn parses_functor_call_in_rule() {
        let src = r#"R(host) :- a = $HTTPRequest(url, _, _), host = @url_host(url), host != ""."#;
        let prog = p(src);
        let r = &prog.rules[0];
        assert!(matches!(
            &r.body[1],
            Literal::Compare {
                op: CompareOp::Eq,
                right: Term::Functor { .. },
                ..
            }
        ));
    }

    #[test]
    fn parses_negated_atom() {
        let src = r#"R(x) :- S(x), !T(x), not U(x)."#;
        let prog = p(src);
        let body = &prog.rules[0].body;
        assert_eq!(body.len(), 3);
        assert!(matches!(body[1], Literal::Neg(_)));
        assert!(matches!(body[2], Literal::Neg(_)));
    }

    #[test]
    fn parses_type_decls() {
        let src = r#"
            .type Action = SendAttempt {} | CallTool { fn_name: symbol, args: symbol }
            .type Message = [c: symbol, t: symbol]
            .type AgentRole <: symbol
        "#;
        let prog = p(src);
        assert!(
            matches!(prog.types.get("Action"), Some(TypeDecl::Adt { branches, .. }) if branches.len() == 2)
        );
        assert!(
            matches!(prog.types.get("Message"), Some(TypeDecl::Record { fields, .. }) if fields.len() == 2)
        );
        assert!(matches!(
            prog.types.get("AgentRole"),
            Some(TypeDecl::Alias { .. })
        ));
    }

    #[test]
    fn io_options_preserve_flags_and_leave_following_rules_intact() {
        let src = r#"
            .input Edge(rfc4180=true, delimiter="\t", headers=false)
            .decl Edge(src: symbol, dst: symbol)
            .decl Reach(src: symbol, dst: symbol)
            .output Reach(IO=stdout, filename="out.csv", jobs=2)
            Reach(x, y) :- Edge(x, y).
            Edge("a", "b").
        "#;
        let bridge = parse_with_options(
            src,
            "io.dl",
            ParseOptions {
                reject_duplicate_relations: true,
                ..ParseOptions::default()
            },
        )
        .unwrap();
        for program in [parse(src, "io.dl").unwrap(), bridge.program] {
            let edge = &program.relations["Edge"];
            assert!(edge.is_input && !edge.is_output);
            assert_eq!(edge.params.len(), 2);
            let reach = &program.relations["Reach"];
            assert!(reach.is_output && !reach.is_input);
            assert_eq!(program.rules.len(), 1);
            assert_eq!(program.rules[0].head.relation, "Reach");
            assert_eq!(program.facts.len(), 1);
        }
    }

    #[test]
    fn malformed_io_options_are_rejected_without_consuming_the_next_declaration() {
        for src in [
            ".input R(rfc4180=true",
            ".input R(rfc4180)\n.decl R(x: symbol)",
            ".input R(rfc4180=)\n.decl R(x: symbol)",
            ".input R(rfc4180=true,)\n.decl R(x: symbol)",
            ".input R(rfc4180=true\n.decl R(x: symbol)",
            ".output R(IO=stdout filename=\"x\")",
            ".output R(filename=f(1))",
        ] {
            assert!(parse(src, "io.dl").is_err(), "must refuse: {src}");
            assert!(
                parse_with_options(src, "io.dl", ParseOptions::default()).is_err(),
                "bridge must refuse: {src}"
            );
        }
    }

    #[test]
    fn parses_relation_input_output() {
        let src = r#"
            .decl Edge(s: symbol, d: symbol)
            .input Edge
            .decl Authorized(idx: unsigned)
            .output Authorized
        "#;
        let prog = p(src);
        let edge = prog.relations.get("Edge").unwrap();
        assert!(edge.is_input && !edge.is_output);
        let auth = prog.relations.get("Authorized").unwrap();
        assert!(auth.is_output && !auth.is_input);
    }

    #[test]
    fn parses_functor_decl() {
        let src = r#".functor json_get_str(json: symbol, field: symbol): symbol stateful"#;
        let prog = p(src);
        let f = prog.functors.get("json_get_str").unwrap();
        assert_eq!(f.params.len(), 2);
        assert!(f.stateful);
    }

    #[test]
    fn parses_fact() {
        let src = r#"ToolResultFieldName("status")."#;
        let prog = p(src);
        assert_eq!(prog.facts.len(), 1);
        assert_eq!(prog.facts[0].atom.relation, "ToolResultFieldName");
    }

    #[test]
    fn parses_hex_and_binary_literals() {
        let prog = p(r#"R(x) :- x = 0xFF, y = 0b101, x = y."#);
        let body = &prog.rules[0].body;
        let mut found_hex = false;
        let mut found_bin = false;
        for lit in body {
            if let Literal::Compare {
                right: Term::NumberLit(n),
                ..
            } = lit
            {
                if *n == 0xFF {
                    found_hex = true;
                }
                if *n == 0b101 {
                    found_bin = true;
                }
            }
        }
        assert!(found_hex && found_bin, "got {:?}", body);
    }

    #[test]
    fn parses_unsigned_suffix() {
        let prog = p(r#"R(x) :- x = 5u, y = 0xFFu, x = y."#);
        let body = &prog.rules[0].body;
        let count = body
            .iter()
            .filter(|lit| {
                matches!(
                    lit,
                    Literal::Compare {
                        right: Term::UnsignedLit(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(count, 2, "expected two unsigned compares, got {:?}", body);
    }

    #[test]
    fn parses_nil_true_false_keywords() {
        let prog = p(r#"R(x) :- x = nil, y = true, z = false, x = y, y = z."#);
        let body = &prog.rules[0].body;
        // nil → empty RecordLit, true → 1, false → 0.
        let mut nil_seen = false;
        let mut t_seen = false;
        let mut f_seen = false;
        for lit in body {
            if let Literal::Compare { right, .. } = lit {
                match right {
                    Term::RecordLit(fs) if fs.is_empty() => nil_seen = true,
                    Term::NumberLit(1) => t_seen = true,
                    Term::NumberLit(0) => f_seen = true,
                    _ => {}
                }
            }
        }
        assert!(nil_seen && t_seen && f_seen, "got {:?}", body);
    }

    #[test]
    fn parses_as_cast_erased() {
        // Soufflé's `as(x, T)` is a type cast we treat as
        // an identity at the AST level — the inner term
        // appears unchanged.
        let prog = p(r#"R(idx) :- Actions(idx, a), b = as(a, Action)."#);
        let body = &prog.rules[0].body;
        let has_cast = body.iter().any(|lit| {
            matches!(lit,
            Literal::Compare {
                right: Term::Var(name),
                ..
            } if name == "a")
        });
        assert!(
            has_cast,
            "expected `as(a, Action)` to erase to `a`, got {:?}",
            body
        );
    }

    #[test]
    fn parses_arithmetic_in_comparison() {
        let src = r#"R(idx) :- A(idx, n), n + 1 > 5."#;
        let prog = p(src);
        let cmp = &prog.rules[0].body[1];
        match cmp {
            Literal::Compare {
                op: CompareOp::Gt,
                left: Term::Arith { .. },
                ..
            } => {}
            other => panic!("expected arith compare, got {:?}", other),
        }
    }

    #[test]
    fn wildcards_are_unique() {
        let src = r#"R(x) :- A(_, x), B(_, x)."#;
        let prog = p(src);
        let body = &prog.rules[0].body;
        let extract_first_arg = |lit: &Literal| match lit {
            Literal::Pos(a) => a.args[0].clone(),
            _ => panic!(),
        };
        let w1 = extract_first_arg(&body[0]);
        let w2 = extract_first_arg(&body[1]);
        match (w1, w2) {
            (Term::Wildcard(a), Term::Wildcard(b)) => assert_ne!(a, b),
            _ => panic!("expected wildcards"),
        }
    }

    /// Smoke-test the parser against the real `.dl`
    /// files in the repo, after running them through
    /// `sugar.py` (the production preprocessor). Every
    /// policy listed must exist and parse; only a missing
    /// python3 skips the test.
    #[test]
    fn parses_repo_policies() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let sugar = format!("{manifest}/../../souffle/sugar.py");
        assert!(
            std::path::Path::new(&sugar).exists(),
            "the desugaring preprocessor is missing: {sugar}"
        );
        let candidates = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../policies/common_policy.dl"
            ),
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
        let common_policy = format!("{manifest}/../../policies/common_policy.dl");
        let common_src = std::fs::read_to_string(&common_policy)
            .unwrap_or_else(|e| panic!("common policy missing at {common_policy}: {e}"));
        let mut analysed = 0usize;
        for path in &candidates {
            assert!(
                std::path::Path::new(path).exists(),
                "example policy missing from the repository: {path}"
            );
            let src = std::fs::read_to_string(path).unwrap();
            // The production pipeline prepends common_policy
            // before sugar.py so it sees the type table.
            // Mirror that here, then inline any remaining
            // #include lines.
            let stripped: String = src
                .lines()
                .filter(|l| {
                    let t = l.trim_start();
                    !(t.starts_with("#include") && t.contains("common_policy"))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let combined = if path.contains("common_policy.dl") {
                stripped
            } else {
                format!("{}\n{}", common_src, stripped)
            };
            let inlined = inline_includes(&combined, path);
            let tmp = std::env::temp_dir().join(format!(
                "sasy-parser-smoke-{}.dl",
                std::path::Path::new(path)
                    .parent()
                    .and_then(|d| d.file_name())
                    .map(|d| d.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ));
            std::fs::write(&tmp, &inlined).unwrap();
            let out = match std::process::Command::new("python3")
                .arg(&sugar)
                .arg(&tmp)
                .output()
            {
                Ok(o) => o,
                Err(_) => {
                    eprintln!("skipping: python3 not available");
                    return;
                }
            };
            if !out.status.success() {
                panic!(
                    "sugar.py failed for {path}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            let desugared = String::from_utf8(out.stdout).unwrap();
            match parse(&desugared, *path) {
                Ok(prog) => {
                    eprintln!(
                        "{}: {} rules, {} relations, {} functors, {} facts",
                        path,
                        prog.rules.len(),
                        prog.relations.len(),
                        prog.functors.len(),
                        prog.facts.len()
                    );
                }
                Err(e) => panic!("parse failed for {path}: {e}"),
            }
            analysed += 1;
        }
        assert!(analysed > 0, "no policy was analysed");
    }

    fn inline_includes(src: &str, path: &str) -> String {
        let dir = std::path::Path::new(path).parent().unwrap();
        let mut out = String::new();
        for line in src.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let rest = rest.trim();
                if let Some(name) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    let p = dir.join(name);
                    if let Ok(content) = std::fs::read_to_string(&p) {
                        out.push_str(&inline_includes(&content, p.to_str().unwrap()));
                        out.push('\n');
                        continue;
                    }
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    #[test]
    fn parses_common_policy_authorized() {
        let src = r#"
            Authorized(idx) :-
                HasAuthenticatedEntity(),
                IsAuthorized(idx),
                !Unauthorized(idx).
        "#;
        let prog = p(src);
        assert_eq!(prog.rules.len(), 1);
        assert_eq!(prog.rules[0].body.len(), 3);
    }
}
