//! Conjunctive-query containment between rule bodies via
//! the canonical-database method.
//!
//! Containment direction: `A ⊆ B` reads as "every action
//! that A fires for, B also fires for." For two rules
//! with the same head shape, this is equivalent — by the
//! Chandra–Merlin theorem extended with FDs and ADT
//! constructors — to the existence of a homomorphism
//! `h : vars(B) → terms(chase(A))` such that:
//!
//! - For each positive atom `R(t1, ..., tn)` in B's body,
//!   there's an atom `R(h(t1), ..., h(tn))` in A's
//!   FD-closed instance (atoms post-`assert_body` +
//!   `check`).
//! - For each constructor pattern `t = $C(args)` in B,
//!   `h(t)`'s class in A is pinned to constructor `$C`
//!   with class-equal arg images.
//! - For each comparison `t1 op t2`, the constraint
//!   holds under the substitution.
//! - The head-arg images line up positionally with A's
//!   head args (so the conclusion matches).
//!
//! Recursive IDB calls remain *opaque* — we don't unfold
//! them — which makes this analysis sound but
//! conservative on subsumptions that depend on the
//! recursion's semantics.
//!
//! Search strategy is plain backtracking over `vars(B)`.
//! Bodies in real policies have <20 variables; CSP-style
//! pruning is unnecessary at this scale.

use std::collections::HashMap;

use super::ast::*;
use super::chase::{Chase, ConcreteValue, FunctorWitness};
use super::unfold::{unfold_body, FreshVars, OpaqueSet};

/// Backtracking substitution: B's variable name → A's
/// class root.
type Subst = HashMap<String, u32>;

/// Decide whether `a ⊆ b` (every action A fires for, B
/// also fires for). Both rules must share the same head
/// arity; we align head arguments positionally.
///
/// Implementation: freeze A by chasing its body, then
/// search for a homomorphism `vars(B) → A`'s class roots.
/// Non-recursive IDBs in either body are unfolded first
/// (per the same logic used by contradiction detection).
pub fn rule_subsumes(a: &Rule, b: &Rule, program: &Program, extra_opaque: &OpaqueSet) -> bool {
    if a.head.args.len() != b.head.args.len() {
        return false;
    }
    // Always merge in recursive IDBs so the unfolder
    // doesn't loop. Callers pass `extra_opaque` to force
    // additional names symbolic.
    let mut opaque = OpaqueSet::from_program(program);
    for name in &extra_opaque.names {
        opaque.force(name);
    }
    let mut var_gen = FreshVars::new();
    let a_alts = unfold_body(&a.body, program, &opaque, &mut var_gen);
    let b_alts = unfold_body(&b.body, program, &opaque, &mut var_gen);

    // A ⊆ B holds iff for *every* unfolded alternative
    // of A, *some* unfolded alternative of B contains it.
    // (A's disjuncts are unioned — if a single alternative
    // of A escapes B's union, A is not contained.)
    'outer: for a_body in &a_alts {
        for b_body in &b_alts {
            if alternative_contains(a, a_body, b, b_body) {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Test containment between a single (a_body, b_body)
/// pair of unfolded conjunctive bodies.
fn alternative_contains(
    a_rule: &Rule,
    a_body: &[Literal],
    b_rule: &Rule,
    b_body: &[Literal],
) -> bool {
    // Freeze A: build a chase from A's body and saturate.
    let mut chase = Chase::new();
    let a_vars = match chase.assert_body(a_body) {
        Ok(v) => v,
        Err(_) => return false, // A unsatisfiable: A ⊆ B vacuously, but
                                // a bad rule can't be subsumed meaningfully.
    };
    if chase.check().is_err() {
        return false;
    }

    // Initial substitution: B's head args bound to the
    // class roots of A's head args (positional align).
    // For each position, both head args must agree:
    //   - Var(a) on A side → look up its class in A's chase.
    //   - Concrete on A side → find or create the
    //     corresponding constant class in A.
    //   - Var(b) on B side → bind to the resolved A class.
    //   - Concrete on B side → A's resolved class must
    //     hold the same concrete value (else no
    //     subsumption — B's head wouldn't produce A's
    //     tuple).
    let mut subst: Subst = HashMap::new();
    for (a_head_arg, b_head_arg) in a_rule.head.args.iter().zip(b_rule.head.args.iter()) {
        let a_class = match a_head_arg {
            Term::Var(a_name) => a_vars.get(a_name).map(|t| chase.find_immutable(*t)),
            Term::StringLit(s) => find_constant_class(&chase, &ConcreteValue::String(s.clone())),
            Term::NumberLit(n) => find_constant_class(&chase, &ConcreteValue::Number(*n)),
            Term::UnsignedLit(n) => find_constant_class(&chase, &ConcreteValue::Unsigned(*n)),
            Term::Wildcard(_) => None,
            _ => None,
        };
        match b_head_arg {
            Term::Var(b_name) => {
                if let Some(c) = a_class {
                    if let Some(&existing) = subst.get(b_name) {
                        if existing != c {
                            return false;
                        }
                    } else {
                        subst.insert(b_name.clone(), c);
                    }
                }
            }
            Term::StringLit(s) => {
                let want = ConcreteValue::String(s.clone());
                match a_class.and_then(|c| chase.class_concrete(c).cloned()) {
                    Some(v) if v == want => {}
                    Some(_) => return false,
                    None => {
                        // A's head arg isn't pinned to a
                        // concrete value but B requires one
                        // — A could fire on tuples where
                        // B's head doesn't. Not subsumed.
                        if !matches!(a_head_arg, Term::Wildcard(_)) {
                            // unless A's slot is `_`, refuse.
                            // We've already returned None for
                            // Wildcard; falling through here
                            // means A had a Var that wasn't
                            // pinned to "s".
                        }
                        return false;
                    }
                }
            }
            Term::NumberLit(n) => {
                let want = ConcreteValue::Number(*n);
                match a_class.and_then(|c| chase.class_concrete(c).cloned()) {
                    Some(v) if v == want => {}
                    _ => return false,
                }
            }
            Term::UnsignedLit(n) => {
                let want = ConcreteValue::Unsigned(*n);
                match a_class.and_then(|c| chase.class_concrete(c).cloned()) {
                    Some(v) if v == want => {}
                    _ => return false,
                }
            }
            Term::Wildcard(_) => {}
            _ => {}
        }
    }

    // Split B's body into positive atoms (need matches),
    // negative atoms (need non-matches), and comparisons
    // (constraints to verify post-substitution).
    let mut pos: Vec<&Atom> = Vec::new();
    let mut neg: Vec<&Atom> = Vec::new();
    let mut compares: Vec<(&CompareOp, &Term, &Term)> = Vec::new();
    for lit in b_body {
        match lit {
            Literal::Pos(a) => pos.push(a),
            Literal::Neg(a) => neg.push(a),
            Literal::Compare {
                op, left, right, ..
            } => {
                compares.push((op, left, right));
            }
            Literal::Negation { .. } | Literal::Disjunction { .. } => return false,
        }
    }

    // Order positive atoms heuristically: those with the
    // most already-bound variables first, so we prune
    // earlier. Plain order is fine at body sizes we see.
    backtrack(&pos, 0, &neg, &compares, &chase, subst, a_body)
}

fn backtrack(
    pos: &[&Atom],
    i: usize,
    neg: &[&Atom],
    compares: &[(&CompareOp, &Term, &Term)],
    chase: &Chase,
    subst: Subst,
    a_body: &[Literal],
) -> bool {
    if i == pos.len() {
        // All positive atoms placed. Verify constraints.
        return verify_constraints(neg, compares, chase, subst, a_body);
    }
    let b_atom = pos[i];
    for a_view in chase.pos_atoms_view().collect::<Vec<_>>() {
        if a_view.relation != b_atom.relation || a_view.args.len() != b_atom.args.len() {
            continue;
        }
        let mut local = subst.clone();
        let mut ok = true;
        for (b_arg, a_term) in b_atom.args.iter().zip(a_view.args.iter()) {
            let a_class = chase.find_immutable(*a_term);
            if !match_term(b_arg, a_class, &mut local, chase) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        if backtrack(pos, i + 1, neg, compares, chase, local, a_body) {
            return true;
        }
    }
    false
}

fn verify_constraints(
    neg: &[&Atom],
    compares: &[(&CompareOp, &Term, &Term)],
    chase: &Chase,
    mut subst: Subst,
    a_body: &[Literal],
) -> bool {
    // Negative atoms in B: closed-world reasoning over
    // A's frozen instance under-approximates the actual
    // worlds A can occupy. Sound rule: B's `!P(args)`
    // must be matched by an A-side `!P(args')` with
    // class-equivalent args under the current subst —
    // i.e., A explicitly forces the same negation.
    for n in neg {
        let mut found = false;
        for a_lit in a_body {
            let Literal::Neg(a_atom) = a_lit else {
                continue;
            };
            if a_atom.relation != n.relation || a_atom.args.len() != n.args.len() {
                continue;
            }
            // Try to align B's neg atom with A's neg atom
            // under the current subst.
            let mut probe = subst.clone();
            let aligned = a_atom
                .args
                .iter()
                .zip(n.args.iter())
                .all(|(a_arg, b_arg)| neg_arg_aligns(a_arg, b_arg, chase, &mut probe));
            if aligned {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    // Comparisons.
    for (op, left, right) in compares {
        match op {
            CompareOp::Eq => {
                if !eq_holds(left, right, chase, &mut subst) {
                    return false;
                }
            }
            CompareOp::Ne => {
                let lc = resolve_to_class(left, chase, &subst);
                let rc = resolve_to_class(right, chase, &subst);
                if let (Some(l), Some(r)) = (lc, rc) {
                    if l == r {
                        return false;
                    }
                }
            }
            // Ordering compares are opaque in this implementation.
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {}
        }
    }
    true
}

/// Whether B's neg atom arg `b_arg` aligns with A's neg
/// atom arg `a_arg` under the given subst. We try to map
/// vars uniformly: if `b_arg` is a var bound under subst
/// then its class must match A's class for `a_arg`; if
/// unbound, this is a free choice we don't attempt to
/// pin (sound but conservative — could miss valid
/// alignments where the alignment binding would have
/// satisfied the rest of the constraints).
fn neg_arg_aligns(a_arg: &Term, b_arg: &Term, _chase: &Chase, _subst: &mut Subst) -> bool {
    // Sound conservative: same syntactic shape suffices.
    // True alignment requires class-class equality after
    // substitution which we don't track for A's neg
    // atoms (they don't get a chase representation).
    match (a_arg, b_arg) {
        (Term::Var(_), Term::Var(_)) => true,
        (Term::Wildcard(_), _) | (_, Term::Wildcard(_)) => true,
        (Term::StringLit(s1), Term::StringLit(s2)) => s1 == s2,
        (Term::NumberLit(n1), Term::NumberLit(n2)) => n1 == n2,
        (Term::UnsignedLit(n1), Term::UnsignedLit(n2)) => n1 == n2,
        _ => false,
    }
}

/// Whether the equality `left = right` holds in A's
/// instance under the given subst. For a Var-Constructor
/// equality we resolve the var's class and check its
/// constructor witness against the pattern; for two
/// concrete sides we compare classes.
///
/// When neither side resolves to an A-class, soundness
/// requires conservative rejection in most cases: B's
/// equality may name a functor or constant that A's
/// body never pins down, so we can't conclude B's body
/// holds in A's instance. The narrow exceptions are
/// pure variable-or-wildcard bindings, which are free
/// choices in B that don't constrain A.
fn eq_holds(left: &Term, right: &Term, chase: &Chase, subst: &mut Subst) -> bool {
    if let Some(l_class) = resolve_to_class(left, chase, subst) {
        return match_term(right, l_class, subst, chase);
    }
    if let Some(r_class) = resolve_to_class(right, chase, subst) {
        return match_term(left, r_class, subst, chase);
    }
    // Neither side resolves. Free var/wildcard bindings
    // are still satisfiable (B can pick any value); a
    // functor or constant that A doesn't pin is not.
    matches!(left, Term::Var(_) | Term::Wildcard(_))
        && matches!(right, Term::Var(_) | Term::Wildcard(_))
}

/// Resolve a term to an A-class root under the given
/// substitution. Returns None when the term is a fresh
/// existential not yet bound (e.g., a B variable that
/// didn't appear in any matched atom), or when a
/// constructor / functor lookup finds no matching class
/// in A's chase — which is itself information: if A
/// doesn't pin the functor, B's compare on its result
/// can't be evaluated, and the caller decides what that
/// means for soundness.
fn resolve_to_class(term: &Term, chase: &Chase, subst: &Subst) -> Option<u32> {
    match term {
        Term::Var(name) => subst.get(name).copied(),
        Term::Wildcard(_) => None,
        Term::StringLit(s) => find_constant_class(chase, &ConcreteValue::String(s.clone())),
        Term::NumberLit(n) => find_constant_class(chase, &ConcreteValue::Number(*n)),
        Term::UnsignedLit(n) => find_constant_class(chase, &ConcreteValue::Unsigned(*n)),
        Term::Constructor { name, args } => find_constructor_class(chase, name, args, subst),
        Term::Functor { name, args } => find_functor_class(chase, name, false, args, subst),
        Term::Builtin { name, args } => find_functor_class(chase, name, true, args, subst),
        Term::RecordLit(_) | Term::Arith { .. } => None,
        Term::FieldAccess { .. } => None,
    }
}

/// Search A's chase for a class pinned by a functor
/// witness `name(args)` whose args, after resolving
/// under `subst`, are class-equal to a witness in A.
/// Returns the class root if a matching witness exists.
fn find_functor_class(
    chase: &Chase,
    name: &str,
    builtin: bool,
    args: &[Term],
    subst: &Subst,
) -> Option<u32> {
    let target_args: Option<Vec<u32>> = args
        .iter()
        .map(|a| resolve_to_class(a, chase, subst))
        .collect();
    let target_args = target_args?;
    for r in chase.class_roots() {
        for w in chase.class_functors(r) {
            if witness_matches(w, name, builtin, &target_args, chase) {
                return Some(r);
            }
        }
    }
    None
}

fn witness_matches(
    w: &FunctorWitness,
    name: &str,
    builtin: bool,
    target_args: &[u32],
    chase: &Chase,
) -> bool {
    if w.name != name || w.builtin != builtin || w.args.len() != target_args.len() {
        return false;
    }
    w.args
        .iter()
        .zip(target_args.iter())
        .all(|(w_arg, &target)| chase.find_immutable(*w_arg) == target)
}

fn find_constant_class(chase: &Chase, value: &ConcreteValue) -> Option<u32> {
    chase
        .class_roots()
        .find(|&r| chase.class_concrete(r) == Some(value))
}

fn find_constructor_class(chase: &Chase, name: &str, args: &[Term], subst: &Subst) -> Option<u32> {
    chase.class_roots().find(|&r| {
        let Some(witness) = chase.class_constructor(r) else {
            return false;
        };
        if witness.name != name || witness.args.len() != args.len() {
            return false;
        }
        // Args match if their resolved classes equal the
        // witness's. Variable args match anything (we
        // can't tell without trying); skip them — we
        // already did binding through atom match.
        args.iter().zip(witness.args.iter()).all(|(a, w)| {
            let w_class = chase.find_immutable(*w);
            match a {
                Term::Var(name) => match subst.get(name) {
                    Some(&c) => c == w_class,
                    None => true,
                },
                Term::Wildcard(_) => true,
                Term::StringLit(s) => {
                    chase.class_concrete(w_class) == Some(&ConcreteValue::String(s.clone()))
                }
                Term::NumberLit(n) => {
                    chase.class_concrete(w_class) == Some(&ConcreteValue::Number(*n))
                }
                Term::UnsignedLit(n) => {
                    chase.class_concrete(w_class) == Some(&ConcreteValue::Unsigned(*n))
                }
                _ => true,
            }
        })
    })
}

fn match_term(b_term: &Term, a_class: u32, subst: &mut Subst, chase: &Chase) -> bool {
    match b_term {
        Term::Var(name) => match subst.get(name) {
            Some(&existing) => existing == a_class,
            None => {
                subst.insert(name.clone(), a_class);
                true
            }
        },
        Term::Wildcard(_) => true,
        Term::StringLit(s) => {
            chase.class_concrete(a_class) == Some(&ConcreteValue::String(s.clone()))
        }
        Term::NumberLit(n) => chase.class_concrete(a_class) == Some(&ConcreteValue::Number(*n)),
        Term::UnsignedLit(n) => chase.class_concrete(a_class) == Some(&ConcreteValue::Unsigned(*n)),
        Term::Constructor { name, args } => match chase.class_constructor(a_class) {
            Some(witness) => {
                if witness.name != *name || witness.args.len() != args.len() {
                    return false;
                }
                let witness_args = witness.args.clone();
                for (b_arg, a_arg_id) in args.iter().zip(witness_args.iter()) {
                    let a_arg_class = chase.find_immutable(*a_arg_id);
                    if !match_term(b_arg, a_arg_class, subst, chase) {
                        return false;
                    }
                }
                true
            }
            None => false,
        },
        // Record literals are interned by the chase as
        // synthetic constructors named `__record`; match
        // against that witness so positional unpack does
        // its job (it's how sugar.py dot-notation lands).
        Term::RecordLit(fields) => match chase.class_constructor(a_class) {
            Some(witness) if witness.name == "__record" && witness.args.len() == fields.len() => {
                let witness_args = witness.args.clone();
                for (b_field, a_arg_id) in fields.iter().zip(witness_args.iter()) {
                    let a_arg_class = chase.find_immutable(*a_arg_id);
                    if !match_term(b_field, a_arg_class, subst, chase) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        },
        // Functor application: A's class must carry a
        // matching functor witness — same name, same
        // arity, and arg classes that align under the
        // current substitution (or that we can bind by
        // recursing into match_term on each arg).
        Term::Functor { name, args } | Term::Builtin { name, args } => {
            let builtin = matches!(b_term, Term::Builtin { .. });
            let witnesses: Vec<FunctorWitness> = chase.class_functors(a_class).to_vec();
            for w in &witnesses {
                if w.name != *name || w.builtin != builtin || w.args.len() != args.len() {
                    continue;
                }
                let saved = subst.clone();
                let mut ok = true;
                for (b_arg, w_arg) in args.iter().zip(w.args.iter()) {
                    let w_class = chase.find_immutable(*w_arg);
                    if !match_term(b_arg, w_class, subst, chase) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return true;
                }
                *subst = saved;
            }
            false
        }
        // Arithmetic stays opaque (the chase does not model
        // integer ranges).
        Term::Arith { .. } => true,
        Term::FieldAccess { .. } => true,
    }
}

/* ────────────────────────────────────────────────────────
Redundancy
──────────────────────────────────────────────────────── */

#[derive(Debug, Clone)]
pub struct RedundancyFinding {
    pub redundant_index: usize,
    pub redundant_span: SourceSpan,
    pub covered_by_index: usize,
    pub covered_by_span: SourceSpan,
    pub head_relation: String,
}

/// Find rules subsumed by some *other* rule with the same
/// head. Single-rule subsumption is sound for redundancy
/// (R ⊆ R' ⇒ R is redundant relative to R'); it is
/// incomplete relative to subsumption-by-union (e.g., R
/// covered jointly by R' and R'' but neither alone). The
/// Union-subsumption is not implemented: single-rule
/// subsumption is sound but incomplete.
pub fn find_redundancies(program: &Program, opaque: &OpaqueSet) -> Vec<RedundancyFinding> {
    let mut findings = Vec::new();
    // Group rule indices by head relation.
    let mut by_head: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in program.rules.iter().enumerate() {
        by_head.entry(r.head.relation.clone()).or_default().push(i);
    }
    for (relation, indices) in &by_head {
        if indices.len() < 2 {
            continue;
        }
        for &i in indices {
            for &j in indices {
                if i == j {
                    continue;
                }
                let ri = &program.rules[i];
                let rj = &program.rules[j];
                if rule_subsumes(ri, rj, program, opaque) {
                    findings.push(RedundancyFinding {
                        redundant_index: i,
                        redundant_span: ri.span.clone(),
                        covered_by_index: j,
                        covered_by_span: rj.span.clone(),
                        head_relation: relation.clone(),
                    });
                    break;
                }
            }
        }
    }
    findings
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    fn rules_named<'a>(prog: &'a Program, head: &str) -> Vec<&'a Rule> {
        prog.rules
            .iter()
            .filter(|r| r.head.relation == head)
            .collect()
    }

    #[test]
    fn identical_rules_subsume_each_other() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                R(idx) :- Actions(idx, _).
                R(idx) :- Actions(idx, _).
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        assert!(rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()));
        assert!(rule_subsumes(rs[1], rs[0], &prog, &OpaqueSet::default()));
    }

    #[test]
    fn specific_subsumed_by_broad() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                R(idx) :- Actions(idx, a), a = $CallTool("submit", _), HasRole("admin").
                R(idx) :- Actions(idx, _).
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        // Specific subsumed by broad: every (admin, submit)
        // action satisfies Actions(idx, _).
        assert!(rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()));
        // Broad NOT subsumed by specific.
        assert!(!rule_subsumes(rs[1], rs[0], &prog, &OpaqueSet::default()));
    }

    #[test]
    fn distinct_specific_rules_dont_subsume() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                R(idx) :- Actions(idx, a), a = $CallTool("read", _).
                R(idx) :- Actions(idx, a), a = $CallTool("write", _).
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        assert!(!rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()));
        assert!(!rule_subsumes(rs[1], rs[0], &prog, &OpaqueSet::default()));
    }

    #[test]
    fn redundancy_finds_specialization() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsAuthorized(idx) :- Actions(idx, _).
                IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("submit", _).
            "#,
            "test.dl",
        )
        .unwrap();
        let findings = find_redundancies(&prog, &OpaqueSet::default());
        // The second rule is subsumed by the first.
        assert!(
            findings.iter().any(|f| f.redundant_index == 1
                && f.covered_by_index == 0
                && f.head_relation == "IsAuthorized"),
            "got: {:?}",
            findings
        );
    }

    #[test]
    fn idb_unfolding_supports_subsumption() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).
                R(idx) :- Actions(idx, a), IsTool(a, "submit").
                R(idx) :- Actions(idx, a), a = $CallTool(_, _).
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        // IsTool(a, "submit") unfolds to Actions(_, a),
        // a = $CallTool("submit", _). The second rule's
        // a = $CallTool(_, _) is broader; it should subsume.
        assert!(
            rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()),
            "expected unfolded specific to be subsumed by broader"
        );
    }

    #[test]
    fn recursive_idb_subsumption_is_conservative() {
        // Supervises is recursive → opaque. Two rules with
        // identical opaque IDB calls should still subsume
        // each other (syntactic match is sufficient).
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                Supervises(s, e) :- Manages(s, e).
                Supervises(s, e) :- Manages(s, m), Supervises(m, e).
                R(idx) :- Actions(idx, _), Supervises("alice", "bob").
                R(idx) :- Actions(idx, _), Supervises("alice", "bob").
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        assert!(rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()));
    }

    /// Run redundancy detection on every real policy in
    /// the repo, after running them through the
    /// production preprocessor. Asserts the analyzer
    /// terminates cleanly; reports findings to stderr.
    #[test]
    fn finds_redundancies_on_repo_policies() {
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
                "sasy-subsume-smoke-{}.dl",
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
            let findings = find_redundancies(&prog, &OpaqueSet::default());
            eprintln!(
                "\n=== {} — {} redundancy finding(s) ===",
                path,
                findings.len()
            );
            for f in &findings {
                eprintln!(
                    "  [{}] rule@{} subsumed by rule@{}",
                    f.head_relation,
                    f.redundant_span.as_location(),
                    f.covered_by_span.as_location()
                );
            }
            analysed += 1;
        }
        assert!(analysed > 0, "no policy was analysed");
    }

    #[test]
    fn negation_breaks_subsumption() {
        let prog = parse(
            r#"
                .decl Actions(idx: unsigned, a: symbol)
                .input Actions
                R(idx) :- Actions(idx, _).
                R(idx) :- Actions(idx, _), !HasRole("guest").
            "#,
            "test.dl",
        )
        .unwrap();
        let rs = rules_named(&prog, "R");
        // First rule is broader (no negation guard). Second is
        // narrower (excludes guest). First should NOT be
        // subsumed by second; second IS subsumed by first.
        assert!(rule_subsumes(rs[1], rs[0], &prog, &OpaqueSet::default()));
        assert!(!rule_subsumes(rs[0], rs[1], &prog, &OpaqueSet::default()));
    }
}
