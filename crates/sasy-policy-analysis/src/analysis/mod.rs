//! Static analyses on Datalog policies.
//!
//! Implements the four analyses described in the SASY
//! paper: contradiction detection, redundancy detection,
//! subsumption detection, and conditional reachability.
//! All analyses operate on a parsed [`ast::Program`] and
//! share a common substrate-aware satisfiability core
//! ([`chase`]) that handles functional dependencies,
//! ADT constructor reasoning, and functor determinism.
//!
//! Module layout:
//! - [`ast`]: the AST that downstream analyses consume.
//! - [`lexer`] + [`parser`]: tokenizer and recursive-
//!   descent parser for the post-sugar `.dl` subset.
//! - [`chase`]: FD chase + congruence closure for body
//!   satisfiability and overlap-witness extraction.
//! - [`contradiction`]: pairwise (allow, deny) overlap
//!   analysis with broad/specific classification.
//! - [`subsumption`]: query-containment via canonical-DB
//!   homomorphism check, drives redundancy detection.
//! - [`reachability`]: backward unfold to DNF over
//!   substrate atoms, with symbolic recursion.

pub mod ast;
pub mod chase;
pub mod contradiction;
pub mod lexer;
pub mod parser;
pub mod pipeline;
pub mod reachability;
pub mod resolve_dots;
pub mod rewrite_annotations;
pub mod subsumption;
pub mod unfold;

pub use pipeline::{analyze_desugared, AnalysisReport};

pub use ast::{Atom, CompareOp, Fact, Literal, Program, Rule, SourceSpan, Term, TypeDecl, TypeRef};
pub use parser::{
    parse, parse_with_options, BridgeParseError, BridgeProgram, ParseError, ParseOptions,
};
