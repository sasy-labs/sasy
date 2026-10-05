//! Non-recursive IDB unfolding.
//!
//! Most real Allow/Deny rules in SASY policies invoke
//! helper IDBs — `IsTool`, `QueriesHost`, `ToolResultField`,
//! `IsToolCall`, and the like — rather than match raw
//! constructor patterns. The chase reasons about
//! constructor injectivity and substrate FDs, but treats
//! every unknown relation as opaque. So a body literal
//! `IsTool(a, "submit")` and `IsTool(a, "delete")` both
//! look satisfiable to the chase: two opaque atoms with
//! no FD relating their arguments.
//!
//! To expose the structural conflict, we perform a
//! preprocessing pass that *unfolds* non-recursive IDB
//! calls — substitutes each call with the disjunction of
//! its rule bodies, with fresh variables for each body's
//! existentials. Recursive IDBs (`Supervises`,
//! `CurrentDepends`, `CurrentDependsSameAgent`) stay
//! opaque: recursion is not unfolded.
//!
//! The unfolding is one-step at the boundary between
//! recursive and non-recursive: every reachable
//! non-recursive IDB call is replaced by its definition,
//! which may itself contain calls — those are unfolded
//! too, transitively, until only EDB atoms, recursive IDB
//! calls, comparisons, constructor patterns, and functor
//! calls remain.

use std::collections::{HashMap, HashSet};

use super::ast::*;

/// Set of relation names treated as opaque even when the
/// program defines rules for them. The classifier marks
/// a relation as recursive iff it lies in a non-trivial
/// call-graph SCC (or has a self-loop). Callers may add
/// names to this set to force opacity (e.g., for IDBs
/// they want to keep symbolic in reachability output).
#[derive(Debug, Clone, Default)]
pub struct OpaqueSet {
    pub names: HashSet<String>,
}

impl OpaqueSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_program(program: &Program) -> Self {
        let mut s = Self::new();
        s.names.extend(recursive_idbs(program));
        s
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn force(&mut self, name: impl Into<String>) {
        self.names.insert(name.into());
    }
}

/// Identify recursive IDBs by computing strongly-connected
/// components of the rule-body → rule-head call graph and
/// returning relations that either (a) lie in an SCC of
/// size > 1, or (b) appear in their own rule's body
/// (self-loop).
pub fn recursive_idbs(program: &Program) -> HashSet<String> {
    // Build adjacency: for each head relation, the set of
    // body relations it calls.
    let mut adj: HashMap<String, HashSet<String>> = HashMap::new();
    for rule in &program.rules {
        let head = &rule.head.relation;
        let entry = adj.entry(head.clone()).or_default();
        for lit in &rule.body {
            if let Some(name) = literal_relation(lit) {
                entry.insert(name.to_string());
            }
        }
    }
    // Tarjan's SCC algorithm.
    let nodes: Vec<String> = adj.keys().cloned().collect();
    let mut index_of: HashMap<String, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        index_of.insert(n.clone(), i);
    }

    #[derive(Default)]
    struct Tarjan {
        index: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        next_idx: usize,
        sccs: Vec<Vec<usize>>,
    }

    let n = nodes.len();
    let mut t = Tarjan {
        index: vec![None; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        next_idx: 0,
        sccs: Vec::new(),
    };

    fn strong(
        v: usize,
        nodes: &[String],
        index_of: &HashMap<String, usize>,
        adj: &HashMap<String, HashSet<String>>,
        t: &mut Tarjan,
    ) {
        t.index[v] = Some(t.next_idx);
        t.lowlink[v] = t.next_idx;
        t.next_idx += 1;
        t.stack.push(v);
        t.on_stack[v] = true;
        if let Some(callees) = adj.get(&nodes[v]) {
            for callee in callees {
                if let Some(&w) = index_of.get(callee) {
                    if t.index[w].is_none() {
                        strong(w, nodes, index_of, adj, t);
                        t.lowlink[v] = t.lowlink[v].min(t.lowlink[w]);
                    } else if t.on_stack[w] {
                        t.lowlink[v] = t.lowlink[v].min(t.index[w].expect("indexed"));
                    }
                }
            }
        }
        if t.lowlink[v] == t.index[v].expect("indexed") {
            let mut scc = Vec::new();
            loop {
                let w = t.stack.pop().expect("nonempty");
                t.on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            t.sccs.push(scc);
        }
    }

    for v in 0..n {
        if t.index[v].is_none() {
            strong(v, &nodes, &index_of, &adj, &mut t);
        }
    }

    let mut recursive: HashSet<String> = HashSet::new();
    for scc in &t.sccs {
        if scc.len() > 1 {
            for &v in scc {
                recursive.insert(nodes[v].clone());
            }
        } else {
            // Single-node SCC: self-loop iff the relation
            // calls itself.
            let v = scc[0];
            let name = &nodes[v];
            if let Some(callees) = adj.get(name) {
                if callees.contains(name) {
                    recursive.insert(name.clone());
                }
            }
        }
    }
    recursive
}

fn literal_relation(lit: &Literal) -> Option<&str> {
    match lit {
        Literal::Pos(a) | Literal::Neg(a) => Some(&a.relation),
        _ => None,
    }
}

/* ────────────────────────────────────────────────────────
Unfolding
──────────────────────────────────────────────────────── */

/// Unfold non-recursive IDB calls in a body. Each
/// non-recursive IDB literal is replaced by the
/// (disjunction of) its rule bodies, with variables
/// renamed apart so the existentials of different
/// occurrences don't clash. The result is a list of
/// conjunctive bodies (one per disjunct).
///
/// `program` is the source of rule definitions; `opaque`
/// names relations that should not be unfolded
/// (recursive IDBs, EDBs by virtue of having no rules,
/// or any relation a caller chose to keep symbolic).
///
/// `var_gen` is a counter for fresh-variable generation,
/// so multiple calls share a global namespace.
pub fn unfold_body(
    body: &[Literal],
    program: &Program,
    opaque: &OpaqueSet,
    var_gen: &mut FreshVars,
) -> Vec<Vec<Literal>> {
    let mut acc: Vec<Vec<Literal>> = vec![Vec::new()];
    for lit in body {
        match lit {
            Literal::Pos(atom) => {
                if should_unfold(atom, program, opaque) {
                    let alts = unfold_atom_pos(atom, program, opaque, var_gen);
                    acc = cross(acc, alts);
                } else {
                    for body in &mut acc {
                        body.push(Literal::Pos(atom.clone()));
                    }
                }
            }
            Literal::Neg(atom) => {
                // Negative IDB calls stay symbolic in the
                // output. The auditor reads positive
                // reachability of the negated relation
                // separately to see its conditions; in-line
                // unfolding only swaps which symbolic name
                // appears (e.g., `!Unauthorized` →
                // `!DenialReason`) without exposing further
                // structure.
                for body in &mut acc {
                    body.push(Literal::Neg(atom.clone()));
                }
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

fn should_unfold(atom: &Atom, program: &Program, opaque: &OpaqueSet) -> bool {
    if opaque.contains(&atom.relation) {
        return false;
    }
    // EDBs (declared .input or no rules defining them)
    // stay as is; the chase already handles them via FDs.
    if program.is_edb(&atom.relation) {
        return false;
    }
    // Has at least one defining rule? If not, leave
    // opaque (likely an EDB without an explicit .input).
    program.rules_for(&atom.relation).next().is_some()
}

fn unfold_atom_pos(
    atom: &Atom,
    program: &Program,
    opaque: &OpaqueSet,
    var_gen: &mut FreshVars,
) -> Vec<Vec<Literal>> {
    let mut alternatives: Vec<Vec<Literal>> = Vec::new();
    for rule in program.rules_for(&atom.relation) {
        let mapping = make_renaming(rule, var_gen);
        if rule.head.args.len() != atom.args.len() {
            return vec![vec![Literal::Pos(atom.clone())]];
        }
        // For each Var head arg, build a substitution
        // mapping renamed name → call arg, so the head-arg
        // unification disappears from the body. Non-Var
        // head args (e.g., string literals in the rule's
        // head) keep an explicit `Compare(...)` literal.
        let mut head_subst: HashMap<String, Term> = HashMap::new();
        let mut residual_compares: Vec<Literal> = Vec::new();
        for (h, c) in rule.head.args.iter().zip(atom.args.iter()) {
            match h {
                Term::Var(name) => {
                    if let Some(renamed) = mapping.get(name) {
                        head_subst.insert(renamed.clone(), c.clone());
                    }
                }
                _ => {
                    let h_renamed = rename_term(h, &mapping);
                    residual_compares.push(Literal::Compare {
                        op: CompareOp::Eq,
                        left: h_renamed,
                        right: c.clone(),
                        span: atom.span.clone(),
                    });
                }
            }
        }
        let renamed_body: Vec<Literal> = rule
            .body
            .iter()
            .map(|l| rename_literal(l, &mapping))
            .map(|l| substitute_in_literal(&l, &head_subst))
            .collect();
        let inner = unfold_body(&renamed_body, program, opaque, var_gen);
        for sub in inner {
            let mut combined = residual_compares.clone();
            combined.extend(sub);
            alternatives.push(combined);
        }
    }
    if alternatives.is_empty() {
        return vec![vec![Literal::Pos(atom.clone())]];
    }
    alternatives
}

fn substitute_in_literal(lit: &Literal, subst: &HashMap<String, Term>) -> Literal {
    match lit {
        Literal::Pos(a) => Literal::Pos(substitute_in_atom(a, subst)),
        Literal::Neg(a) => Literal::Neg(substitute_in_atom(a, subst)),
        Literal::Negation { literal, span } => Literal::Negation {
            literal: Box::new(substitute_in_literal(literal, subst)),
            span: span.clone(),
        },
        Literal::Compare {
            op,
            left,
            right,
            span,
        } => Literal::Compare {
            op: *op,
            left: substitute_in_term(left, subst),
            right: substitute_in_term(right, subst),
            span: span.clone(),
        },
        Literal::Disjunction { alternatives, span } => Literal::Disjunction {
            alternatives: alternatives
                .iter()
                .map(|alt| {
                    alt.iter()
                        .map(|l| substitute_in_literal(l, subst))
                        .collect()
                })
                .collect(),
            span: span.clone(),
        },
    }
}

fn substitute_in_atom(atom: &Atom, subst: &HashMap<String, Term>) -> Atom {
    Atom {
        relation: atom.relation.clone(),
        args: atom
            .args
            .iter()
            .map(|t| substitute_in_term(t, subst))
            .collect(),
        span: atom.span.clone(),
    }
}

fn substitute_in_term(t: &Term, subst: &HashMap<String, Term>) -> Term {
    match t {
        Term::Var(n) => subst.get(n).cloned().unwrap_or_else(|| t.clone()),
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => {
            t.clone()
        }
        Term::Constructor { name, args } => Term::Constructor {
            name: name.clone(),
            args: args.iter().map(|a| substitute_in_term(a, subst)).collect(),
        },
        Term::Functor { name, args } => Term::Functor {
            name: name.clone(),
            args: args.iter().map(|a| substitute_in_term(a, subst)).collect(),
        },
        Term::Builtin { name, args } => Term::Builtin {
            name: name.clone(),
            args: args.iter().map(|a| substitute_in_term(a, subst)).collect(),
        },
        Term::RecordLit(fs) => {
            Term::RecordLit(fs.iter().map(|f| substitute_in_term(f, subst)).collect())
        }
        Term::Arith { op, left, right } => Term::Arith {
            op: *op,
            left: Box::new(substitute_in_term(left, subst)),
            right: Box::new(substitute_in_term(right, subst)),
        },
        Term::FieldAccess { record, field } => Term::FieldAccess {
            record: record.clone(),
            field: field.clone(),
        },
    }
}

fn cross(a: Vec<Vec<Literal>>, b: Vec<Vec<Literal>>) -> Vec<Vec<Literal>> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for prefix in &a {
        for suffix in &b {
            let mut combined = prefix.clone();
            combined.extend(suffix.iter().cloned());
            out.push(combined);
        }
    }
    out
}

/* ────────────────────────────────────────────────────────
Variable renaming
──────────────────────────────────────────────────────── */

/// Generator for fresh variable names. One generator
/// instance shared across an entire unfold call, so
/// repeated unfolds of the same rule produce distinct
/// variables each time.
#[derive(Debug, Clone, Default)]
pub struct FreshVars {
    next: u64,
}

impl FreshVars {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fresh(&mut self, hint: &str) -> String {
        let id = self.next;
        self.next += 1;
        format!("__u{}_{}", id, hint)
    }
}

fn make_renaming(rule: &Rule, var_gen: &mut FreshVars) -> HashMap<String, String> {
    let mut mapping: HashMap<String, String> = HashMap::new();
    collect_vars_atom(&rule.head, &mut mapping, var_gen);
    for lit in &rule.body {
        collect_vars_lit(lit, &mut mapping, var_gen);
    }
    mapping
}

fn collect_vars_atom(atom: &Atom, mapping: &mut HashMap<String, String>, var_gen: &mut FreshVars) {
    for arg in &atom.args {
        collect_vars_term(arg, mapping, var_gen);
    }
}

fn collect_vars_lit(lit: &Literal, mapping: &mut HashMap<String, String>, var_gen: &mut FreshVars) {
    match lit {
        Literal::Pos(a) | Literal::Neg(a) => collect_vars_atom(a, mapping, var_gen),
        Literal::Negation { literal, .. } => {
            collect_vars_lit(literal, mapping, var_gen);
        }
        Literal::Compare { left, right, .. } => {
            collect_vars_term(left, mapping, var_gen);
            collect_vars_term(right, mapping, var_gen);
        }
        Literal::Disjunction { alternatives, .. } => {
            for alt in alternatives {
                for l in alt {
                    collect_vars_lit(l, mapping, var_gen);
                }
            }
        }
    }
}

fn collect_vars_term(term: &Term, mapping: &mut HashMap<String, String>, var_gen: &mut FreshVars) {
    match term {
        Term::Var(name) => {
            mapping
                .entry(name.clone())
                .or_insert_with(|| var_gen.fresh(name));
        }
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => {}
        Term::Constructor { args, .. }
        | Term::Functor { args, .. }
        | Term::Builtin { args, .. } => {
            for a in args {
                collect_vars_term(a, mapping, var_gen);
            }
        }
        Term::RecordLit(fs) => {
            for f in fs {
                collect_vars_term(f, mapping, var_gen);
            }
        }
        Term::Arith { left, right, .. } => {
            collect_vars_term(left, mapping, var_gen);
            collect_vars_term(right, mapping, var_gen);
        }
        Term::FieldAccess { record, .. } => {
            mapping
                .entry(record.clone())
                .or_insert_with(|| var_gen.fresh(record));
        }
    }
}

fn rename_atom(atom: &Atom, mapping: &HashMap<String, String>) -> Atom {
    Atom {
        relation: atom.relation.clone(),
        args: atom.args.iter().map(|a| rename_term(a, mapping)).collect(),
        span: atom.span.clone(),
    }
}

fn rename_literal(lit: &Literal, mapping: &HashMap<String, String>) -> Literal {
    match lit {
        Literal::Pos(a) => Literal::Pos(rename_atom(a, mapping)),
        Literal::Neg(a) => Literal::Neg(rename_atom(a, mapping)),
        Literal::Negation { literal, span } => Literal::Negation {
            literal: Box::new(rename_literal(literal, mapping)),
            span: span.clone(),
        },
        Literal::Compare {
            op,
            left,
            right,
            span,
        } => Literal::Compare {
            op: *op,
            left: rename_term(left, mapping),
            right: rename_term(right, mapping),
            span: span.clone(),
        },
        Literal::Disjunction { alternatives, span } => Literal::Disjunction {
            alternatives: alternatives
                .iter()
                .map(|alt| alt.iter().map(|l| rename_literal(l, mapping)).collect())
                .collect(),
            span: span.clone(),
        },
    }
}

fn rename_term(term: &Term, mapping: &HashMap<String, String>) -> Term {
    match term {
        Term::Var(name) => Term::Var(mapping.get(name).cloned().unwrap_or_else(|| name.clone())),
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => {
            term.clone()
        }
        Term::Constructor { name, args } => Term::Constructor {
            name: name.clone(),
            args: args.iter().map(|a| rename_term(a, mapping)).collect(),
        },
        Term::Functor { name, args } => Term::Functor {
            name: name.clone(),
            args: args.iter().map(|a| rename_term(a, mapping)).collect(),
        },
        Term::Builtin { name, args } => Term::Builtin {
            name: name.clone(),
            args: args.iter().map(|a| rename_term(a, mapping)).collect(),
        },
        Term::RecordLit(fs) => {
            Term::RecordLit(fs.iter().map(|f| rename_term(f, mapping)).collect())
        }
        Term::Arith { op, left, right } => Term::Arith {
            op: *op,
            left: Box::new(rename_term(left, mapping)),
            right: Box::new(rename_term(right, mapping)),
        },
        Term::FieldAccess { record, field } => Term::FieldAccess {
            record: mapping
                .get(record)
                .cloned()
                .unwrap_or_else(|| record.clone()),
            field: field.clone(),
        },
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
    fn detects_self_recursive_idb() {
        let prog = parse(
            r#"
                Path(a, b) :- Edge(a, b).
                Path(a, b) :- Edge(a, m), Path(m, b).
            "#,
            "test.dl",
        )
        .unwrap();
        let recs = recursive_idbs(&prog);
        assert!(recs.contains("Path"), "Path is recursive, got {:?}", recs);
    }

    #[test]
    fn detects_mutually_recursive_idbs() {
        let prog = parse(
            r#"
                A(x) :- B(x).
                B(x) :- A(x).
            "#,
            "test.dl",
        )
        .unwrap();
        let recs = recursive_idbs(&prog);
        assert!(recs.contains("A") && recs.contains("B"));
    }

    #[test]
    fn non_recursive_idb_not_flagged() {
        let prog = parse(
            r#"
                IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).
                Allow(idx) :- Actions(idx, a), IsTool(a, "submit").
            "#,
            "test.dl",
        )
        .unwrap();
        let recs = recursive_idbs(&prog);
        assert!(!recs.contains("IsTool"));
        assert!(!recs.contains("Allow"));
    }

    #[test]
    fn unfolds_non_recursive_idb_into_constructor_pattern() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).
                Allow(idx) :- IsTool(a, "submit"), Actions(idx, a).
            "#,
            "test.dl",
        )
        .unwrap();
        let opaque = OpaqueSet::from_program(&prog);
        let allow_body = &prog
            .rules
            .iter()
            .find(|r| r.head.relation == "Allow")
            .unwrap()
            .body;
        let mut vg = FreshVars::new();
        let unfolded = unfold_body(allow_body, &prog, &opaque, &mut vg);
        assert_eq!(unfolded.len(), 1, "single rule for IsTool, single alt");
        let body = &unfolded[0];
        // Should now contain a Constructor pattern instead
        // of the IsTool call.
        let has_constructor = body.iter().any(|lit| match lit {
            Literal::Compare {
                right: Term::Constructor { name, .. },
                ..
            } => name == "CallTool",
            _ => false,
        });
        assert!(
            has_constructor,
            "expected unfolded constructor, got {:?}",
            body
        );
        // And no IsTool call left.
        let has_istool = body.iter().any(|lit| match lit {
            Literal::Pos(a) => a.relation == "IsTool",
            _ => false,
        });
        assert!(!has_istool);
    }

    #[test]
    fn keeps_recursive_idb_opaque() {
        let prog = parse(
            r#"
                Path(a, b) :- Edge(a, b).
                Path(a, b) :- Edge(a, m), Path(m, b).
                Allow(x) :- Path("alice", x).
            "#,
            "test.dl",
        )
        .unwrap();
        let opaque = OpaqueSet::from_program(&prog);
        let allow_body = &prog
            .rules
            .iter()
            .find(|r| r.head.relation == "Allow")
            .unwrap()
            .body;
        let mut vg = FreshVars::new();
        let unfolded = unfold_body(allow_body, &prog, &opaque, &mut vg);
        assert_eq!(unfolded.len(), 1);
        let has_path = unfolded[0].iter().any(|lit| match lit {
            Literal::Pos(a) => a.relation == "Path",
            _ => false,
        });
        assert!(has_path, "Path should stay opaque, got {:?}", unfolded[0]);
    }

    #[test]
    fn unfolds_idb_with_multiple_rules_into_disjuncts() {
        let prog = parse(
            r#"
                ReservationHasFlownFlight(r) :- ToolResult(_, "x", _), r = "r1".
                ReservationHasFlownFlight(r) :- ToolResult(_, "y", _), r = "r2".
                Use(r) :- ReservationHasFlownFlight(r).
            "#,
            "test.dl",
        )
        .unwrap();
        let opaque = OpaqueSet::from_program(&prog);
        let body = &prog
            .rules
            .iter()
            .find(|r| r.head.relation == "Use")
            .unwrap()
            .body;
        let mut vg = FreshVars::new();
        let unfolded = unfold_body(body, &prog, &opaque, &mut vg);
        assert_eq!(unfolded.len(), 2, "two rules → two alternatives");
    }
}
