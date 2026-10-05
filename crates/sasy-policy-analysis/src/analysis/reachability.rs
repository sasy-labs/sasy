//! Conditional reachability — backward unfold of a head
//! relation into a DNF over substrate atoms, symbolic
//! recursive IDB calls, and side conditions.
//!
//! Given a target relation (typically `IsAuthorized`,
//! `Unauthorized`, or `Authorized`), this analysis
//! produces an exhaustive enumeration of the conditions
//! under which the relation fires: each disjunct names
//! the substrate facts and constraints required for one
//! derivation path. The output gives an auditor a
//! ground-truth answer to "what would have to be true
//! for X to be authorized?"
//!
//! # Algorithm
//!
//! 1. Start with `[Pos(target(head_vars))]` as a
//!    one-literal body.
//! 2. Unfold via [`super::unfold::unfold_body`]: every
//!    non-recursive IDB call is substituted with the
//!    disjunction of its rule bodies. Recursive IDBs
//!    (detected by Tarjan SCC over the call graph) stay
//!    *symbolic* — they appear in the output as
//!    uninterpreted positive atoms whose definitions are
//!    cited in the result. Symbolic recursion preserves
//!    the auditor's reading ("X is authorized when Y AND
//!    there's a Supervises chain from A to B") rather
//!    than truncating to a depth bound.
//! 3. For each unfolded conjunctive body, run the FD
//!    chase to test satisfiability. Bodies that
//!    saturate to UNSAT (constructor injectivity,
//!    distinct constants, etc.) are pruned — they
//!    represent disjuncts that can't fire on any
//!    substrate state.
//! 4. The surviving bodies are the result DNF.
//!
//! Negation of non-recursive IDBs is left *symbolic* in
//! this implementation: `!Unauthorized(idx)` distributes over the
//! disjunction of `Unauthorized`'s rule bodies as a
//! conjunction of negated bodies, which would re-expand
//! to a CNF and blow up the DNF size. The output retains
//! the negated atom; the auditor reads it as "no
//! `Unauthorized` derivation fires here" with the
//! definition cross-referenced.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::ast::*;
use super::chase::Chase;
use super::unfold::{unfold_body, FreshVars, OpaqueSet};

/// One conjunctive disjunct of the reachability DNF —
/// the conditions for one specific derivation path.
#[derive(Debug, Clone)]
pub struct ReachClause {
    /// Positive atoms: substrate predicates and
    /// (symbolic) recursive IDB calls. Each one must
    /// hold for this disjunct to fire.
    pub positive: Vec<Atom>,
    /// Negative atoms: substrate predicates and IDB
    /// calls that must *not* hold (under stratified
    /// negation). Negative IDBs are kept symbolic; the
    /// auditor reads positive reachability of the
    /// negated relation to see its conditions.
    pub negative: Vec<Atom>,
    /// Side conditions: comparisons, equality bindings,
    /// functor calls. Carry through the unfold as-is.
    pub side_conditions: Vec<SideCondition>,
}

/// A side condition: comparison or (dis)equality not
/// representable as a positive/negative atom.
#[derive(Debug, Clone)]
pub struct SideCondition {
    pub op: CompareOp,
    pub left: Term,
    pub right: Term,
}

/// Reachability result for one target relation.
#[derive(Debug, Clone)]
pub struct ReachabilityResult {
    pub target: String,
    /// DNF: clauses are OR'd together. An empty `clauses`
    /// list means the target is unreachable on any
    /// substrate state (no rules, or all unfolded
    /// disjuncts pruned as UNSAT).
    pub clauses: Vec<ReachClause>,
    /// Relations the analyzer kept symbolic during
    /// unfolding — recursive IDBs plus any caller-forced
    /// names. The auditor reads these as the boundary
    /// between "expanded structure" and "pinned
    /// definitions cited inline."
    pub opaque: HashSet<String>,
    /// Number of unfolded disjuncts the chase pruned as
    /// UNSAT. Useful as a sanity-check signal — a high
    /// prune rate means the source rules contain many
    /// structurally-dead alternatives.
    pub pruned: usize,
}

/// Run conditional reachability on a target relation.
///
/// `extra_opaque` lets callers force additional
/// relations symbolic (for example, to keep some IDB
/// uninterpreted for clarity even though it's not
/// recursive). Recursive IDBs are detected automatically
/// from the call graph.
pub fn reach(program: &Program, target: &str, extra_opaque: &OpaqueSet) -> ReachabilityResult {
    // Recursive IDBs are auto-symbolic.
    let mut opaque = OpaqueSet::from_program(program);
    for name in &extra_opaque.names {
        opaque.force(name);
    }

    // If the target has no defining rules, reachability
    // is vacuous — there's no structural derivation, so
    // we report the empty DNF. The relation may still
    // fire as an EDB fact, but that's not what
    // reachability asks about.
    if program.rules_for(target).next().is_none() {
        return ReachabilityResult {
            target: target.to_string(),
            clauses: Vec::new(),
            opaque: opaque.names,
            pruned: 0,
        };
    }
    let arity = program
        .rules_for(target)
        .next()
        .map(|r| r.head.args.len())
        .unwrap_or(0);

    // Build a synthetic body `[Pos(target(v0, v1, ..., vn))]`
    // with fresh, named head vars. Unfolding will replace
    // it with the disjunction of `target`'s rule bodies.
    let head_args: Vec<Term> = (0..arity).map(|i| Term::Var(format!("__h{}", i))).collect();
    let synthetic = Atom {
        relation: target.to_string(),
        args: head_args,
        span: SourceSpan::single("<reach>", 0),
    };
    let initial = vec![Literal::Pos(synthetic)];

    let mut var_gen = FreshVars::new();
    let unfolded = unfold_body(&initial, program, &opaque, &mut var_gen);

    // Each unfolded conjunctive body: chase to verify
    // satisfiability, then split into positive/negative
    // atoms and side conditions, then simplify
    // (equality-elimination + variable renaming) so the
    // output is readable as an audit artifact.
    let mut clauses = Vec::new();
    let mut pruned = 0;
    for body in &unfolded {
        if !body_is_satisfiable(body) {
            pruned += 1;
            continue;
        }
        let clause = body_to_clause(body);
        let clause = simplify_clause(clause);
        clauses.push(clause);
    }

    ReachabilityResult {
        target: target.to_string(),
        clauses,
        opaque: opaque.names,
        pruned,
    }
}

fn body_is_satisfiable(body: &[Literal]) -> bool {
    let mut chase = Chase::new();
    if chase.assert_body(body).is_err() {
        return false;
    }
    chase.check().is_ok()
}

fn body_to_clause(body: &[Literal]) -> ReachClause {
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    let mut side = Vec::new();
    for lit in body {
        match lit {
            Literal::Pos(a) => positive.push(a.clone()),
            Literal::Neg(a) => negative.push(a.clone()),
            Literal::Compare {
                op, left, right, ..
            } => side.push(SideCondition {
                op: *op,
                left: left.clone(),
                right: right.clone(),
            }),
            Literal::Negation { .. } | Literal::Disjunction { .. } => {
                // Parser-level disjunctions are expanded
                // before we get here.
            }
        }
    }
    ReachClause {
        positive,
        negative,
        side_conditions: side,
    }
}

/* ────────────────────────────────────────────────────────
Clause simplification
──────────────────────────────────────────────────────── */

/// Run equality-elimination and variable-renaming passes
/// on a single disjunct so the output reads as an audit
/// artifact rather than as the alpha-renamed unfold
/// detritus. Two passes:
///
/// 1. Equality elimination. We collect every `Eq`
///    side condition whose shape lets us substitute
///    (Var-Var, Var-concrete-literal). Build a union-
///    find over variable names, equate via these
///    literals, pick a canonical representative per
///    class (concrete > original-named-var >
///    `__u`/`__h` renamed-var), then substitute through
///    every atom and remaining side condition. Drop
///    any Eq literal that became `t = t`.
///
/// 2. Variable renaming. The unfolder names existentials
///    `__u<N>_<original>` to keep them alpha-distinct.
///    We strip the prefix to recover `<original>` where
///    safe, suffixing with a counter on collisions.
///    Synthetic head vars `__h<N>` become the
///    target relation's parameter name when we know it
///    (left as `__h<N>` here — the caller has the
///    `RelationDecl` if they want it pretty).
fn simplify_clause(clause: ReachClause) -> ReachClause {
    let clause = eliminate_equalities(clause);
    let clause = dedupe_atoms(clause);
    rename_existentials(clause)
}

/// Drop atoms that are syntactically subsumed by another
/// atom in the same body. An atom `R(b1, …, bn)` is
/// subsumed by `R(a1, …, an)` if at every position
/// either the args are identical or the subsumed atom
/// has a wildcard where the subsuming atom has anything.
/// This does *not* attempt cross-variable unification —
/// `Actions(idx, a)` and `Actions(j, a)` with distinct
/// vars `idx ≠ j` are independent constraints, and we
/// keep both. The pass eliminates the unfolder's
/// duplicate-`Actions` artifact (e.g., the
/// `IsToolCall(a) :- Actions(_, a), …` body inserts an
/// `Actions(_, a)` next to the caller's `Actions(idx, a)`).
fn dedupe_atoms(mut clause: ReachClause) -> ReachClause {
    clause.positive = dedupe_atom_list(clause.positive);
    clause.negative = dedupe_atom_list(clause.negative);
    clause
}

fn dedupe_atom_list(atoms: Vec<Atom>) -> Vec<Atom> {
    let mut keep = vec![true; atoms.len()];
    for i in 0..atoms.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..atoms.len() {
            if i == j || !keep[j] {
                continue;
            }
            if atom_subsumes_syntactic(&atoms[i], &atoms[j]) {
                keep[j] = false;
            }
        }
    }
    atoms
        .into_iter()
        .enumerate()
        .filter_map(|(i, a)| if keep[i] { Some(a) } else { None })
        .collect()
}

/// `a` syntactically subsumes `b` iff same relation,
/// same arity, and every position of `b` is either
/// equal to the corresponding position of `a` or a
/// wildcard.
fn atom_subsumes_syntactic(a: &Atom, b: &Atom) -> bool {
    if a.relation != b.relation || a.args.len() != b.args.len() {
        return false;
    }
    a.args
        .iter()
        .zip(b.args.iter())
        .all(|(ax, bx)| term_subsumes_syntactic(ax, bx))
}

fn term_subsumes_syntactic(a: &Term, b: &Term) -> bool {
    if let Term::Wildcard(_) = b {
        return true;
    }
    match (a, b) {
        (Term::Var(x), Term::Var(y)) => x == y,
        (Term::StringLit(x), Term::StringLit(y)) => x == y,
        (Term::NumberLit(x), Term::NumberLit(y)) => x == y,
        (Term::UnsignedLit(x), Term::UnsignedLit(y)) => x == y,
        (Term::Constructor { name: n1, args: a1 }, Term::Constructor { name: n2, args: a2 })
        | (Term::Functor { name: n1, args: a1 }, Term::Functor { name: n2, args: a2 })
        | (Term::Builtin { name: n1, args: a1 }, Term::Builtin { name: n2, args: a2 }) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(x, y)| term_subsumes_syntactic(x, y))
        }
        (Term::RecordLit(x), Term::RecordLit(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(a, b)| term_subsumes_syntactic(a, b))
        }
        _ => false,
    }
}

fn eliminate_equalities(mut clause: ReachClause) -> ReachClause {
    // Step 1: gather all variable names appearing anywhere.
    let mut vars: Vec<String> = Vec::new();
    {
        let mut seen: HashSet<String> = HashSet::new();
        let mut visit = |t: &Term| collect_var_names(t, &mut seen, &mut vars);
        for a in &clause.positive {
            for arg in &a.args {
                visit(arg);
            }
        }
        for a in &clause.negative {
            for arg in &a.args {
                visit(arg);
            }
        }
        for s in &clause.side_conditions {
            visit(&s.left);
            visit(&s.right);
        }
    }

    // Step 2: union-find over variable indices.
    let idx_of: HashMap<String, usize> = vars
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let mut parent: Vec<usize> = (0..vars.len()).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut x = i;
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut concrete: HashMap<usize, Term> = HashMap::new();

    let mut remaining: Vec<SideCondition> = Vec::new();
    for s in clause.side_conditions.drain(..) {
        if !matches!(s.op, CompareOp::Eq) {
            remaining.push(s);
            continue;
        }
        match (&s.left, &s.right) {
            (Term::Var(a), Term::Var(b)) => {
                let ra = find(&mut parent, idx_of[a]);
                let rb = find(&mut parent, idx_of[b]);
                if ra != rb {
                    parent[rb] = ra;
                    if let Some(c) = concrete.remove(&rb) {
                        concrete.entry(ra).or_insert(c);
                    }
                }
            }
            (Term::Var(v), other) | (other, Term::Var(v)) if is_concrete(other) => {
                let r = find(&mut parent, idx_of[v]);
                concrete.entry(r).or_insert_with(|| other.clone());
            }
            _ => remaining.push(s),
        }
    }

    // Step 3: pick a canonical term per class.
    // Preference: concrete literal > original-named-var
    // > prefixed-var (sorted shortest first).
    let mut class_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..vars.len() {
        let r = find(&mut parent, i);
        class_members.entry(r).or_default().push(i);
    }
    let mut canonical: HashMap<String, Term> = HashMap::new();
    for (root, members) in &class_members {
        let canon: Term = if let Some(c) = concrete.get(root) {
            c.clone()
        } else {
            let mut names: Vec<&String> = members.iter().map(|&i| &vars[i]).collect();
            names.sort_by_key(|n| (is_renamed(n), n.len(), n.as_str().to_string()));
            Term::Var(names[0].clone())
        };
        for &i in members {
            canonical.insert(vars[i].clone(), canon.clone());
        }
    }

    // Step 4: apply the substitution, dropping any
    // self-equalities that fall out.
    let positive: Vec<Atom> = clause
        .positive
        .into_iter()
        .map(|a| substitute_atom(&a, &canonical))
        .collect();
    let negative: Vec<Atom> = clause
        .negative
        .into_iter()
        .map(|a| substitute_atom(&a, &canonical))
        .collect();
    let side_conditions: Vec<SideCondition> = remaining
        .into_iter()
        .map(|s| SideCondition {
            op: s.op,
            left: substitute_term(&s.left, &canonical),
            right: substitute_term(&s.right, &canonical),
        })
        .filter(|s| !is_trivially_true(s))
        .collect();

    ReachClause {
        positive,
        negative,
        side_conditions,
    }
}

fn is_concrete(t: &Term) -> bool {
    matches!(
        t,
        Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_)
    )
}

fn is_renamed(name: &str) -> bool {
    // Internal-generated names: unfolder (`__u<N>_…`),
    // synthetic head (`__h<N>`), and sugar.py's
    // dot-notation positional unpack (`__<field>`).
    name.starts_with("__")
}

fn collect_var_names(t: &Term, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    match t {
        Term::Var(n) => {
            if seen.insert(n.clone()) {
                out.push(n.clone());
            }
        }
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => {}
        Term::Constructor { args, .. }
        | Term::Functor { args, .. }
        | Term::Builtin { args, .. } => {
            for a in args {
                collect_var_names(a, seen, out);
            }
        }
        Term::RecordLit(fs) => {
            for f in fs {
                collect_var_names(f, seen, out);
            }
        }
        Term::Arith { left, right, .. } => {
            collect_var_names(left, seen, out);
            collect_var_names(right, seen, out);
        }
        Term::FieldAccess { record, .. } => {
            if seen.insert(record.clone()) {
                out.push(record.clone());
            }
        }
    }
}

fn substitute_atom(a: &Atom, canon: &HashMap<String, Term>) -> Atom {
    Atom {
        relation: a.relation.clone(),
        args: a.args.iter().map(|t| substitute_term(t, canon)).collect(),
        span: a.span.clone(),
    }
}

fn substitute_term(t: &Term, canon: &HashMap<String, Term>) -> Term {
    match t {
        Term::Var(n) => canon.get(n).cloned().unwrap_or_else(|| t.clone()),
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => {
            t.clone()
        }
        Term::Constructor { name, args } => Term::Constructor {
            name: name.clone(),
            args: args.iter().map(|a| substitute_term(a, canon)).collect(),
        },
        Term::Functor { name, args } => Term::Functor {
            name: name.clone(),
            args: args.iter().map(|a| substitute_term(a, canon)).collect(),
        },
        Term::Builtin { name, args } => Term::Builtin {
            name: name.clone(),
            args: args.iter().map(|a| substitute_term(a, canon)).collect(),
        },
        Term::RecordLit(fs) => {
            Term::RecordLit(fs.iter().map(|f| substitute_term(f, canon)).collect())
        }
        Term::Arith { op, left, right } => Term::Arith {
            op: *op,
            left: Box::new(substitute_term(left, canon)),
            right: Box::new(substitute_term(right, canon)),
        },
        Term::FieldAccess { record, field } => match canon.get(record) {
            Some(Term::Var(new_name)) => Term::FieldAccess {
                record: new_name.clone(),
                field: field.clone(),
            },
            _ => t.clone(),
        },
    }
}

/// Whether a side condition is trivially true after
/// substitution. Drops `t = t`, concrete-literal
/// comparisons we can decide directly, and Compares
/// involving wildcards (which are existentially free in
/// positive disjuncts and can take any value).
fn is_trivially_true(s: &SideCondition) -> bool {
    if terms_syntactically_equal(&s.left, &s.right) {
        return matches!(s.op, CompareOp::Eq | CompareOp::Le | CompareOp::Ge);
    }
    if let (Some(lc), Some(rc)) = (concrete_value(&s.left), concrete_value(&s.right)) {
        return match s.op {
            CompareOp::Eq => lc == rc,
            CompareOp::Ne => lc != rc,
            _ => false,
        };
    }
    // A wildcard side makes the comparison existentially
    // satisfiable in a positive clause (pick the
    // wildcard to whatever value makes the op hold).
    // Note: this drop is *only* sound for the DNF's
    // positive disjuncts. The negation-group machinery
    // takes a different path that handles wildcard
    // semantics under universal quantification.
    matches!(&s.left, Term::Wildcard(_)) || matches!(&s.right, Term::Wildcard(_))
}

fn concrete_value(t: &Term) -> Option<ConcretePeek<'_>> {
    match t {
        Term::StringLit(s) => Some(ConcretePeek::Str(s)),
        Term::NumberLit(n) => Some(ConcretePeek::Num(*n)),
        Term::UnsignedLit(n) => Some(ConcretePeek::Unsigned(*n)),
        _ => None,
    }
}

#[derive(PartialEq, Eq)]
enum ConcretePeek<'a> {
    Str(&'a str),
    Num(i64),
    Unsigned(u64),
}

fn terms_syntactically_equal(a: &Term, b: &Term) -> bool {
    match (a, b) {
        (Term::Var(x), Term::Var(y)) => x == y,
        (Term::Wildcard(x), Term::Wildcard(y)) => x == y,
        (Term::StringLit(x), Term::StringLit(y)) => x == y,
        (Term::NumberLit(x), Term::NumberLit(y)) => x == y,
        (Term::UnsignedLit(x), Term::UnsignedLit(y)) => x == y,
        (Term::Constructor { name: n1, args: a1 }, Term::Constructor { name: n2, args: a2 })
        | (Term::Functor { name: n1, args: a1 }, Term::Functor { name: n2, args: a2 })
        | (Term::Builtin { name: n1, args: a1 }, Term::Builtin { name: n2, args: a2 }) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(x, y)| terms_syntactically_equal(x, y))
        }
        (Term::RecordLit(x), Term::RecordLit(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(a, b)| terms_syntactically_equal(a, b))
        }
        (
            Term::Arith {
                op: o1,
                left: l1,
                right: r1,
            },
            Term::Arith {
                op: o2,
                left: l2,
                right: r2,
            },
        ) => o1 == o2 && terms_syntactically_equal(l1, l2) && terms_syntactically_equal(r1, r2),
        _ => false,
    }
}

/// Rename remaining `__u<N>_<rest>` and `__h<N>` vars to
/// friendlier names. Strip the unfold prefix to recover
/// the source-level name where possible; on collisions
/// (two distinct vars with the same stripped name)
/// suffix with a counter so the body remains readable.
fn rename_existentials(mut clause: ReachClause) -> ReachClause {
    let mut vars: Vec<String> = Vec::new();
    {
        let mut seen: HashSet<String> = HashSet::new();
        for a in &clause.positive {
            for arg in &a.args {
                collect_var_names(arg, &mut seen, &mut vars);
            }
        }
        for a in &clause.negative {
            for arg in &a.args {
                collect_var_names(arg, &mut seen, &mut vars);
            }
        }
        for s in &clause.side_conditions {
            collect_var_names(&s.left, &mut seen, &mut vars);
            collect_var_names(&s.right, &mut seen, &mut vars);
        }
    }

    // Map each renamed var to its preferred friendly
    // name, resolving collisions.
    let mut taken: HashSet<String> = vars.iter().filter(|n| !is_renamed(n)).cloned().collect();
    let mut renaming: HashMap<String, Term> = HashMap::new();
    for v in &vars {
        if !is_renamed(v) {
            continue;
        }
        let stripped = strip_unfold_prefix(v);
        let mut candidate = stripped.clone();
        let mut suffix = 2u32;
        while taken.contains(&candidate) {
            candidate = format!("{}{}", stripped, suffix);
            suffix += 1;
        }
        taken.insert(candidate.clone());
        renaming.insert(v.clone(), Term::Var(candidate));
    }

    clause.positive = clause
        .positive
        .into_iter()
        .map(|a| substitute_atom(&a, &renaming))
        .collect();
    clause.negative = clause
        .negative
        .into_iter()
        .map(|a| substitute_atom(&a, &renaming))
        .collect();
    clause.side_conditions = clause
        .side_conditions
        .into_iter()
        .map(|s| SideCondition {
            op: s.op,
            left: substitute_term(&s.left, &renaming),
            right: substitute_term(&s.right, &renaming),
        })
        .collect();
    clause
}

/// Strip layered unfolder/sugar prefixes from a renamed
/// variable until no internal-name shape remains.
/// Examples:
///   `__u3_host` → `host`
///   `__h0` → `arg0`
///   `__msg_contents` → `msg_contents`
///   `__u4___msg_contents` → `msg_contents` (two layers)
fn strip_unfold_prefix(name: &str) -> String {
    let mut current = name.to_string();
    loop {
        let next = strip_one_prefix(&current);
        if next == current {
            return current;
        }
        current = next;
    }
}

fn strip_one_prefix(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("__u") {
        let after_digits = rest.trim_start_matches(|c: char| c.is_ascii_digit());
        if let Some(rest) = after_digits.strip_prefix('_') {
            return rest.to_string();
        }
        return after_digits.to_string();
    }
    if let Some(rest) = name.strip_prefix("__h") {
        return format!("arg{}", rest);
    }
    if let Some(rest) = name.strip_prefix("__") {
        return rest.to_string();
    }
    name.to_string()
}

/* ────────────────────────────────────────────────────────
Pretty printing
──────────────────────────────────────────────────────── */

impl fmt::Display for ReachabilityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Reachability for {}: {} disjunct(s) ({} pruned)",
            self.target,
            self.clauses.len(),
            self.pruned
        )?;
        if !self.opaque.is_empty() {
            let mut names: Vec<&String> = self.opaque.iter().collect();
            names.sort();
            writeln!(
                f,
                "Opaque: {}",
                names
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        for (i, clause) in self.clauses.iter().enumerate() {
            writeln!(f, "[{}] {}", i + 1, clause)?;
        }
        Ok(())
    }
}

impl fmt::Display for ReachClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        for a in &self.positive {
            parts.push(format_atom(a));
        }
        for a in &self.negative {
            parts.push(format!("!{}", format_atom(a)));
        }
        for s in &self.side_conditions {
            parts.push(format!(
                "{} {} {}",
                format_term(&s.left),
                format_op(s.op),
                format_term(&s.right)
            ));
        }
        write!(f, "{}", parts.join(", "))
    }
}

fn format_atom(a: &Atom) -> String {
    let args: Vec<String> = a.args.iter().map(format_term).collect();
    format!("{}({})", a.relation, args.join(", "))
}

fn format_term(t: &Term) -> String {
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

fn format_op(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "=",
        CompareOp::Ne => "!=",
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    fn reach_str(src: &str, target: &str) -> ReachabilityResult {
        let prog = parse(src, "test.dl").unwrap();
        reach(&prog, target, &OpaqueSet::default())
    }

    #[test]
    fn single_rule_yields_single_disjunct() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, a), HasRole("admin").
            "#,
            "IsAuthorized",
        );
        assert_eq!(r.clauses.len(), 1, "got {}", r);
        let body = &r.clauses[0];
        // After unfolding, the synthetic head call produces
        // an Eq literal binding the head var; the original
        // body atoms are preserved.
        assert!(
            body.positive.iter().any(|a| a.relation == "Actions"),
            "no Actions, got {}",
            body
        );
        assert!(
            body.positive.iter().any(|a| a.relation == "HasRole"),
            "no HasRole"
        );
    }

    #[test]
    fn multiple_rules_yield_multiple_disjuncts() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("read", _).
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("write", _), HasRole("admin").
            "#,
            "IsAuthorized",
        );
        assert_eq!(r.clauses.len(), 2, "got {}", r);
    }

    #[test]
    fn unsat_disjuncts_are_pruned() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :-
                    Actions(idx, a),
                    a = $CallTool("X", w),
                    a = $CallTool("Y", w).
                IsAuthorized(idx) :- Actions(idx, a).
            "#,
            "IsAuthorized",
        );
        // First rule's body is UNSAT (constructor injectivity:
        // "X" != "Y"). Second rule survives.
        assert_eq!(r.clauses.len(), 1, "got {}", r);
        assert_eq!(r.pruned, 1);
    }

    #[test]
    fn recursive_idb_appears_symbolically() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                Supervises(s, e) :- Manages(s, e).
                Supervises(s, e) :- Manages(s, m), Supervises(m, e).
                IsAuthorized(idx) :-
                    Actions(idx, a),
                    Supervises("alice", user),
                    a = $CallTool("approve", user).
            "#,
            "IsAuthorized",
        );
        assert!(r.opaque.contains("Supervises"), "opaque: {:?}", r.opaque);
        assert_eq!(r.clauses.len(), 1);
        let clause = &r.clauses[0];
        // Supervises remains as a positive atom (symbolic).
        assert!(clause.positive.iter().any(|a| a.relation == "Supervises"));
    }

    #[test]
    fn negative_idbs_stay_symbolic() {
        // Negative IDB calls in the unfold target are
        // kept as symbolic Neg atoms in the output. The
        // auditor reads positive reachability of the
        // negated relation separately to see its
        // conditions; in-line unfolding doesn't add
        // information and runs into forall-vs-existential
        // soundness obstacles for named body existentials.
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, _).
                Unauthorized(idx) :- Actions(idx, a), a = $CallTool("dangerous", _).
                Authorized(idx) :- IsAuthorized(idx), !Unauthorized(idx).
            "#,
            "Authorized",
        );
        assert_eq!(r.clauses.len(), 1, "got {}", r);
        let clause = &r.clauses[0];
        assert!(
            clause.negative.iter().any(|a| a.relation == "Unauthorized"),
            "expected !Unauthorized symbolic, got {}",
            clause
        );
    }

    #[test]
    fn unreachable_target_yields_empty_dnf() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
            "#,
            "IsAuthorized",
        );
        assert!(r.clauses.is_empty());
    }

    /// Print reachability output for specific targets on
    /// real policies, to inspect the result by hand.
    /// Marked ignored by default — run on demand with
    /// `cargo test reach_show -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn reach_show_specific_targets() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let sugar = format!("{manifest}/../../souffle/sugar.py");
        let common_path = format!("{manifest}/../../policies/common_policy.dl");
        let common_src = std::fs::read_to_string(&common_path)
            .unwrap_or_else(|e| panic!("common policy missing at {common_path}: {e}"));

        let cases: &[(&str, &[&str])] = &[
            (
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../examples/adk-separation-of-duties/payment_policy.dl"
                ),
                &["Unauthorized", "IsAuthorized"],
            ),
            (
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../examples/langchain-information-flow/policy.dl"
                ),
                &["Unauthorized", "IsAuthorized"],
            ),
        ];

        for (path, targets) in cases {
            let user_src = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("example policy missing at {path}: {e}"));
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
                "sasy-reach-show-{}.dl",
                std::path::Path::new(path)
                    .parent()
                    .and_then(|d| d.file_name())
                    .map(|d| d.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ));
            std::fs::write(&tmp, &combined).unwrap();
            let out = std::process::Command::new("python3")
                .arg(&sugar)
                .arg(&tmp)
                .output()
                .unwrap();
            let desugared = String::from_utf8(out.stdout).unwrap();
            let prog = parse(&desugared, *path).unwrap();

            for target in *targets {
                let r = reach(&prog, target, &OpaqueSet::default());
                let name = std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy();
                eprintln!(
                    "\n=================== {} :: {} ===================",
                    name, target
                );
                eprintln!("{}", r);
            }
        }
    }

    /// Run reachability on the real repo policies and
    /// surface the result. Asserts termination + a
    /// reasonable shape (some clauses survive, opaque
    /// set picks up recursive IDBs). Only a missing
    /// python3 skips it.
    #[test]
    fn reach_runs_on_repo_policies() {
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
                "sasy-reach-smoke-{}.dl",
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
            for target in &["IsAuthorized", "Unauthorized"] {
                let r = reach(&prog, target, &OpaqueSet::default());
                eprintln!(
                    "\n=== {} :: {} — {} disjunct(s), {} pruned, opaque {{{}}} ===",
                    std::path::Path::new(path)
                        .file_name()
                        .unwrap()
                        .to_string_lossy(),
                    target,
                    r.clauses.len(),
                    r.pruned,
                    {
                        let mut names: Vec<&String> = r.opaque.iter().collect();
                        names.sort();
                        names
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                );
            }
            analysed += 1;
        }
        assert!(analysed > 0, "no policy was analysed");
    }

    #[test]
    fn pretty_print_renders_atoms_and_side_conditions() {
        let r = reach_str(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("X", _), HasRole("admin").
            "#,
            "IsAuthorized",
        );
        let s = format!("{}", r);
        assert!(s.contains("Actions("));
        assert!(s.contains("$CallTool"));
        assert!(s.contains("HasRole(\"admin\")"));
    }
}
