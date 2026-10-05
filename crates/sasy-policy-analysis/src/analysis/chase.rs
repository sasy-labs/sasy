//! FD chase + congruence closure for body satisfiability.
//!
//! The chase decides whether a conjunction of body
//! literals — drawn from one or more rule bodies — is
//! satisfiable under the substrate's structural
//! constraints:
//!
//! - **Equalities** (`t1 = t2`) merge term equivalence
//!   classes via union-find.
//! - **Constructor patterns** (`a = $CallTool(name, args)`)
//!   pin a class to a specific ADT branch. Two patterns
//!   on the same class force *injectivity* (recursive
//!   merge of corresponding args) and *disjointness*
//!   (different branches → UNSAT).
//! - **Functional dependencies** declared on built-in
//!   relations (e.g., `Actions(idx, a)` has `idx → a`)
//!   propagate equalities when two atoms agree on the
//!   key columns.
//! - **Functor determinism**: two applications of the
//!   same functor to key-equal arguments produce equal
//!   results (`@url_host(u)` is a function of `u`).
//! - **Disequalities** (`t1 != t2`, concrete-value
//!   conflicts) trigger UNSAT when the disequal terms
//!   merge into one class.
//!
//! Recursive IDB calls are treated as *opaque* relations
//! by default — this is sound for overlap detection but
//! conservative for subsumption. Callers can selectively
//! unfold non-recursive IDBs before feeding bodies in.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use super::ast::*;

/* ────────────────────────────────────────────────────────
Term store and union-find
──────────────────────────────────────────────────────── */

/// Identifier for a term node in the chase. Each
/// syntactic occurrence of a variable, constant,
/// constructor, or functor application gets its own
/// `TermId`; the union-find groups equal terms into
/// equivalence classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(pub u32);

/// A concrete value a class is constrained to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConcreteValue {
    String(String),
    Number(i64),
    Unsigned(u64),
}

impl fmt::Display for ConcreteValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConcreteValue::String(s) => write!(f, "\"{}\"", s),
            ConcreteValue::Number(n) => write!(f, "{}", n),
            ConcreteValue::Unsigned(n) => write!(f, "{}", n),
        }
    }
}

/// A constructor witness pinning a class to a specific
/// ADT branch with specific arg classes.
#[derive(Debug, Clone)]
pub struct ConstructorWitness {
    pub name: String,
    pub args: Vec<TermId>,
}

/// A functor-application witness: the class is the
/// result of applying `name` to `args`.
#[derive(Debug, Clone)]
pub struct FunctorWitness {
    pub name: String,
    /// Builtin calls and same-named `@` user functors are separate functions.
    pub builtin: bool,
    pub args: Vec<TermId>,
}

#[derive(Debug, Clone, Default)]
struct ClassData {
    concrete: Option<ConcreteValue>,
    constructor: Option<ConstructorWitness>,
    /// At most-one functor witness *per functor name*
    /// (multiple distinct names can pin the same class —
    /// e.g., a value derived two different ways). Stored
    /// as a vector since the functor count per class is
    /// tiny in practice.
    functors: Vec<FunctorWitness>,
    /// Atoms (indices into `pos_atoms`) that mention a
    /// term currently in this class. Used to drive FD
    /// propagation when classes merge.
    atom_refs: BTreeSet<usize>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TermNode {
    /// Originating syntactic label, kept for diagnostics
    /// and future pretty-printing of witness models.
    label: TermLabel,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum TermLabel {
    Var {
        scope: u32,
        name: String,
    },
    Wildcard {
        scope: u32,
        id: u32,
    },
    Constant(ConcreteValue),
    Constructor(String),
    Functor(String),
    /// Arithmetic terms are kept opaque — each occurrence
    /// is a fresh class with no concrete value. This is
    /// sound (loses precision on numeric reasoning, gains
    /// nothing in our policy fragment beyond what
    /// concrete-value comparison already gives us).
    Arith,
}

/* ────────────────────────────────────────────────────────
Functional dependencies and substrate facts
──────────────────────────────────────────────────────── */

/// One functional dependency on a relation: argument
/// positions in `keys` determine those in `vals`.
#[derive(Debug, Clone)]
pub struct FunctionalDep {
    pub keys: Vec<usize>,
    pub vals: Vec<usize>,
}

/// FD table — one entry per relation, possibly multiple
/// FDs (rare but supported).
#[derive(Debug, Clone, Default)]
pub struct FdTable {
    fds: HashMap<String, Vec<FunctionalDep>>,
}

impl FdTable {
    /// FDs on the built-in substrate relations declared
    /// in `policies/common_policy.dl`.
    pub fn builtin() -> Self {
        let mut t = Self::default();
        // Actions(idx, a): idx → a (idx is the key)
        t.add("Actions", &[0], &[1]);
        // SentMessage(id, msg): id → msg
        t.add("SentMessage", &[0], &[1]);
        // ToolResult(id, fn_name, args): id → (fn_name, args)
        t.add("ToolResult", &[0], &[1, 2]);
        // MessageMetadata(id, metadata): id → metadata, at most one
        // row per message.
        t.add("MessageMetadata", &[0], &[1]);
        // EdgeData(src, dst, data): (src, dst) → data
        t.add("EdgeData", &[0, 1], &[2]);
        // EdgePrincipal(src, dst, principal): (src, dst) → principal.
        // An edge carries at most one principal row: each recording of
        // the edge replaces it with the recording request's identity.
        t.add("EdgePrincipal", &[0, 1], &[2]);
        // EdgeEntity(src, dst, entity): (src, dst) → entity, one row
        // per edge for the same reason.
        t.add("EdgeEntity", &[0, 1], &[2]);
        // Singletons (empty key forces all tuples equal):
        t.add("Principal", &[], &[0]);
        t.add("Entity", &[], &[0]);
        t.add("TenantId", &[], &[0]);
        t.add("Current", &[], &[0]);
        t
    }

    pub fn add(&mut self, relation: &str, keys: &[usize], vals: &[usize]) {
        self.fds
            .entry(relation.to_string())
            .or_default()
            .push(FunctionalDep {
                keys: keys.to_vec(),
                vals: vals.to_vec(),
            });
    }

    pub fn get(&self, relation: &str) -> Option<&[FunctionalDep]> {
        self.fds.get(relation).map(|v| v.as_slice())
    }
}

/* ────────────────────────────────────────────────────────
Conflicts
──────────────────────────────────────────────────────── */

/// Reason the chase determined the body to be UNSAT.
#[derive(Debug, Clone)]
pub enum Conflict {
    /// Two distinct concrete values forced into one class.
    DistinctConstants { a: ConcreteValue, b: ConcreteValue },
    /// Two different ADT branches assigned to one class.
    DistinctConstructors { a: String, b: String },
    /// A pair recorded as `!=` ended up in the same class.
    DisequalMerged { a: TermId, b: TermId },
    /// Constructor witness with an argument-count
    /// mismatch (e.g., user typo); UNSAT for safety.
    ConstructorArityMismatch {
        name: String,
        got: usize,
        expected: usize,
    },
}

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Conflict::DistinctConstants { a, b } => {
                write!(f, "distinct constants {} vs {}", a, b)
            }
            Conflict::DistinctConstructors { a, b } => {
                write!(f, "distinct constructors ${} vs ${}", a, b)
            }
            Conflict::DisequalMerged { .. } => write!(f, "!= violated by merge"),
            Conflict::ConstructorArityMismatch {
                name,
                got,
                expected,
            } => write!(
                f,
                "${} arity mismatch: got {}, expected {}",
                name, got, expected
            ),
        }
    }
}

impl std::error::Error for Conflict {}

/* ────────────────────────────────────────────────────────
Atoms
──────────────────────────────────────────────────────── */

/// A positive atom asserted into the chase.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AtomInstance {
    relation: String,
    args: Vec<TermId>,
    /// Kept for diagnostic output; not used by the
    /// satisfiability core itself.
    span: SourceSpan,
}

/* ────────────────────────────────────────────────────────
Chase engine
──────────────────────────────────────────────────────── */

/// Map from source variable name to its TermId in a
/// scope.
pub type VarMap = HashMap<String, TermId>;

pub struct Chase {
    fds: FdTable,

    nodes: Vec<TermNode>,
    parent: Vec<u32>,
    rank: Vec<u8>,

    /// Class data keyed by the canonical (root) TermId.
    classes: HashMap<u32, ClassData>,

    pos_atoms: Vec<AtomInstance>,
    /// Negative atoms recorded but not yet checked. We
    /// validate them once at the end of [`check`].
    neg_atoms: Vec<AtomInstance>,

    /// Disequalities: pairs of TermIds that must remain
    /// in distinct classes. Validated incrementally on
    /// every merge.
    disequalities: Vec<(u32, u32)>,

    /// Cached "constant" terms, so two literal `"X"`
    /// occurrences share a class without merging them.
    /// (Conservative — if two distinct literal-X classes
    /// existed they'd still merge into the same value
    /// trivially, but this saves work.)
    const_pool: HashMap<ConcreteValue, u32>,

    next_scope: u32,
}

impl Default for Chase {
    fn default() -> Self {
        Self::with_fds(FdTable::builtin())
    }
}

impl Chase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fds(fds: FdTable) -> Self {
        Self {
            fds,
            nodes: Vec::new(),
            parent: Vec::new(),
            rank: Vec::new(),
            classes: HashMap::new(),
            pos_atoms: Vec::new(),
            neg_atoms: Vec::new(),
            disequalities: Vec::new(),
            const_pool: HashMap::new(),
            next_scope: 0,
        }
    }

    /* ── union-find primitives ──────────────────────────── */

    fn fresh(&mut self, label: TermLabel) -> TermId {
        let id = self.nodes.len() as u32;
        self.nodes.push(TermNode { label });
        self.parent.push(id);
        self.rank.push(0);
        self.classes.insert(id, ClassData::default());
        TermId(id)
    }

    fn find(&mut self, t: TermId) -> u32 {
        let mut x = t.0;
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    pub fn root(&mut self, t: TermId) -> TermId {
        TermId(self.find(t))
    }

    /// Read-only `find` that does not perform path
    /// compression. Used by analyses that need to walk
    /// the chase state without `&mut` access (homomorphism
    /// search, witness rendering).
    pub fn find_immutable(&self, t: TermId) -> u32 {
        let mut x = t.0;
        while self.parent[x as usize] != x {
            x = self.parent[x as usize];
        }
        x
    }

    /// Iterate canonical class roots present in the chase.
    pub fn class_roots(&self) -> impl Iterator<Item = u32> + '_ {
        self.classes.keys().copied()
    }

    /// Concrete value pinning a class, if any.
    pub fn class_concrete(&self, root: u32) -> Option<&ConcreteValue> {
        self.classes.get(&root).and_then(|c| c.concrete.as_ref())
    }

    /// Constructor witness pinning a class, if any.
    pub fn class_constructor(&self, root: u32) -> Option<&ConstructorWitness> {
        self.classes.get(&root).and_then(|c| c.constructor.as_ref())
    }

    /// Functor witnesses pinning a class as the result of
    /// applying a specific functor to specific args. Used
    /// by subsumption to evaluate functor-result
    /// equalities and disequalities in B's body against
    /// A's frozen instance.
    pub fn class_functors(&self, root: u32) -> &[FunctorWitness] {
        self.classes
            .get(&root)
            .map(|c| c.functors.as_slice())
            .unwrap_or(&[])
    }

    /// Borrow asserted positive atoms after closure. Used
    /// by the homomorphism-search containment checker.
    pub fn pos_atoms_view(&self) -> impl Iterator<Item = AtomView<'_>> {
        self.pos_atoms.iter().map(|a| AtomView {
            relation: &a.relation,
            args: &a.args,
        })
    }
}

/// Read-only view of a positive atom in the chase.
#[derive(Debug, Clone, Copy)]
pub struct AtomView<'a> {
    pub relation: &'a str,
    pub args: &'a [TermId],
}

impl Chase {
    /// Equate two terms; saturates FD/constructor/functor
    /// implications transitively.
    pub fn unify(&mut self, a: TermId, b: TermId) -> Result<(), Conflict> {
        let mut work = vec![(a, b)];
        while let Some((x, y)) = work.pop() {
            self.unify_one(x, y, &mut work)?;
        }
        Ok(())
    }

    fn unify_one(
        &mut self,
        a: TermId,
        b: TermId,
        work: &mut Vec<(TermId, TermId)>,
    ) -> Result<(), Conflict> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return Ok(());
        }
        // Union by rank.
        let (root, child) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            (ra, rb)
        } else {
            self.rank[ra as usize] += 1;
            (ra, rb)
        };
        self.parent[child as usize] = root;

        // Merge class data.
        let child_data = self.classes.remove(&child).unwrap_or_default();
        let root_data = self.classes.remove(&root).unwrap_or_default();
        let merged = self.merge_class_data(root_data, child_data, work)?;
        self.classes.insert(root, merged);

        // Disequality check: if any (x, y) in disequalities
        // now maps both to the same root, conflict.
        let diseqs = self.disequalities.clone();
        for (p, q) in diseqs {
            let rp = self.find(TermId(p));
            let rq = self.find(TermId(q));
            if rp == rq {
                return Err(Conflict::DisequalMerged {
                    a: TermId(p),
                    b: TermId(q),
                });
            }
        }

        // After a class merge, atoms involving either
        // class may now share keys with other atoms — run
        // FD propagation against the post-merge state.
        self.propagate_fds(work)?;
        Ok(())
    }

    fn merge_class_data(
        &mut self,
        mut a: ClassData,
        b: ClassData,
        work: &mut Vec<(TermId, TermId)>,
    ) -> Result<ClassData, Conflict> {
        // Concrete value: distinct → conflict.
        match (&a.concrete, &b.concrete) {
            (Some(va), Some(vb)) if va != vb => {
                return Err(Conflict::DistinctConstants {
                    a: va.clone(),
                    b: vb.clone(),
                });
            }
            (None, Some(_)) => a.concrete = b.concrete.clone(),
            _ => {}
        }
        // Constructor: same name → injectivity; distinct → conflict.
        match (a.constructor.clone(), b.constructor.clone()) {
            (Some(ca), Some(cb)) => {
                if ca.name != cb.name {
                    return Err(Conflict::DistinctConstructors {
                        a: ca.name,
                        b: cb.name,
                    });
                }
                if ca.args.len() != cb.args.len() {
                    return Err(Conflict::ConstructorArityMismatch {
                        name: ca.name.clone(),
                        got: cb.args.len(),
                        expected: ca.args.len(),
                    });
                }
                for (x, y) in ca.args.iter().zip(cb.args.iter()) {
                    work.push((*x, *y));
                }
            }
            (None, Some(c)) => a.constructor = Some(c),
            _ => {}
        }
        // Functor witnesses: append, then equate any with
        // the same name + key-equal args.
        for fb in b.functors {
            // For each existing functor witness with the
            // same name, check if args are key-equal; if
            // so, this is just a redundant pinning. (No
            // result-class merge needed: the witnesses
            // pin *this* class, so they're already
            // co-resident.)
            a.functors.push(fb);
        }
        // Combine atom_refs.
        a.atom_refs.extend(b.atom_refs.iter().copied());
        Ok(a)
    }

    /* ── building from AST ──────────────────────────────── */

    /// Allocate a fresh scope id for a body assertion.
    pub fn new_scope(&mut self) -> u32 {
        let s = self.next_scope;
        self.next_scope += 1;
        s
    }

    /// Assert a body conjunction in a fresh scope.
    /// Returns the variable→term map for that scope so
    /// the caller can unify head args with another body.
    pub fn assert_body(&mut self, body: &[Literal]) -> Result<VarMap, Conflict> {
        let scope = self.new_scope();
        let mut vars: VarMap = HashMap::new();
        for lit in body {
            self.assert_literal(lit, scope, &mut vars)?;
        }
        Ok(vars)
    }

    fn assert_literal(
        &mut self,
        lit: &Literal,
        scope: u32,
        vars: &mut VarMap,
    ) -> Result<(), Conflict> {
        match lit {
            Literal::Pos(atom) => {
                let args = atom
                    .args
                    .iter()
                    .map(|a| self.intern_term(a, scope, vars))
                    .collect::<Result<Vec<_>, _>>()?;
                let idx = self.pos_atoms.len();
                let inst = AtomInstance {
                    relation: atom.relation.clone(),
                    args: args.clone(),
                    span: atom.span.clone(),
                };
                self.pos_atoms.push(inst);
                for a in &args {
                    let r = self.find(*a);
                    self.classes.entry(r).or_default().atom_refs.insert(idx);
                }
                let mut work = Vec::new();
                self.propagate_fds_for_atom(idx, &mut work)?;
                while let Some((x, y)) = work.pop() {
                    self.unify_one(x, y, &mut work)?;
                }
                Ok(())
            }
            Literal::Neg(atom) => {
                let args = atom
                    .args
                    .iter()
                    .map(|a| self.intern_term(a, scope, vars))
                    .collect::<Result<Vec<_>, _>>()?;
                self.neg_atoms.push(AtomInstance {
                    relation: atom.relation.clone(),
                    args,
                    span: atom.span.clone(),
                });
                Ok(())
            }
            Literal::Compare {
                op,
                left,
                right,
                span: _,
            } => {
                let l = self.intern_term(left, scope, vars)?;
                let r = self.intern_term(right, scope, vars)?;
                match op {
                    CompareOp::Eq => self.unify(l, r),
                    CompareOp::Ne => {
                        let lr = self.find(l);
                        let rr = self.find(r);
                        if lr == rr {
                            return Err(Conflict::DisequalMerged { a: l, b: r });
                        }
                        self.disequalities.push((l.0, r.0));
                        Ok(())
                    }
                    CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
                        // Treated opaquely: this implementation does not
                        // reason about arithmetic ordering, and the
                        // comparison is recorded for completeness only.
                        Ok(())
                    }
                }
            }
            Literal::Negation { .. } | Literal::Disjunction { .. } => {
                // Should have been expanded by the parser.
                // Keep as a no-op so analyses can compose
                // without paying the expansion price twice.
                Ok(())
            }
        }
    }

    fn intern_term(
        &mut self,
        term: &Term,
        scope: u32,
        vars: &mut VarMap,
    ) -> Result<TermId, Conflict> {
        match term {
            Term::Var(name) => {
                if let Some(&id) = vars.get(name) {
                    return Ok(id);
                }
                let id = self.fresh(TermLabel::Var {
                    scope,
                    name: name.clone(),
                });
                vars.insert(name.clone(), id);
                Ok(id)
            }
            Term::Wildcard(id) => Ok(self.fresh(TermLabel::Wildcard { scope, id: *id })),
            Term::StringLit(s) => Ok(self.const_term(ConcreteValue::String(s.clone()))),
            Term::NumberLit(n) => Ok(self.const_term(ConcreteValue::Number(*n))),
            Term::UnsignedLit(n) => Ok(self.const_term(ConcreteValue::Unsigned(*n))),
            Term::Constructor { name, args } => {
                let arg_ids = args
                    .iter()
                    .map(|a| self.intern_term(a, scope, vars))
                    .collect::<Result<Vec<_>, _>>()?;
                let id = self.fresh(TermLabel::Constructor(name.clone()));
                let r = self.find(id);
                let class = self.classes.entry(r).or_default();
                class.constructor = Some(ConstructorWitness {
                    name: name.clone(),
                    args: arg_ids,
                });
                Ok(id)
            }
            Term::Functor { name, args } | Term::Builtin { name, args } => {
                let builtin = matches!(term, Term::Builtin { .. });
                let arg_ids = args
                    .iter()
                    .map(|a| self.intern_term(a, scope, vars))
                    .collect::<Result<Vec<_>, _>>()?;
                let id = self.fresh(TermLabel::Functor(name.clone()));
                // Check determinism: if any other class
                // pins to (name, key-equal args), equate
                // result classes.
                self.equate_with_existing_functor(id, name, builtin, &arg_ids)?;
                let r = self.find(id);
                let class = self.classes.entry(r).or_default();
                class.functors.push(FunctorWitness {
                    name: name.clone(),
                    builtin,
                    args: arg_ids,
                });
                Ok(id)
            }
            Term::RecordLit(fields) => {
                let arg_ids = fields
                    .iter()
                    .map(|a| self.intern_term(a, scope, vars))
                    .collect::<Result<Vec<_>, _>>()?;
                // Treat record literals like a synthetic
                // constructor `__record` so injectivity
                // and disjointness apply.
                let id = self.fresh(TermLabel::Constructor("__record".into()));
                let r = self.find(id);
                self.classes.entry(r).or_default().constructor = Some(ConstructorWitness {
                    name: "__record".into(),
                    args: arg_ids,
                });
                Ok(id)
            }
            Term::Arith { left, right, .. } => {
                let _ = self.intern_term(left, scope, vars)?;
                let _ = self.intern_term(right, scope, vars)?;
                Ok(self.fresh(TermLabel::Arith))
            }
            Term::FieldAccess { .. } => {
                panic!("FieldAccess should be resolved before chase intern")
            }
        }
    }

    fn const_term(&mut self, v: ConcreteValue) -> TermId {
        if let Some(&id) = self.const_pool.get(&v) {
            return TermId(self.find(TermId(id)));
        }
        let id = self.fresh(TermLabel::Constant(v.clone()));
        let r = self.find(id);
        self.classes.entry(r).or_default().concrete = Some(v.clone());
        self.const_pool.insert(v, r);
        TermId(r)
    }

    fn equate_with_existing_functor(
        &mut self,
        new_id: TermId,
        name: &str,
        builtin: bool,
        args: &[TermId],
    ) -> Result<(), Conflict> {
        // Snapshot: for each class, see if a witness
        // already exists with same name and key-equal
        // args. If yes, the existing class's root and
        // new_id must be merged (functor determinism).
        let mut to_merge: Option<TermId> = None;
        let arg_roots: Vec<u32> = args.iter().map(|a| self.find(*a)).collect();
        for (&root, class) in &self.classes {
            for wit in &class.functors {
                if wit.name != name || wit.builtin != builtin || wit.args.len() != args.len() {
                    continue;
                }
                let same = wit.args.iter().zip(arg_roots.iter()).all(|(a, b)| {
                    let mut x = a.0;
                    while self.parent[x as usize] != x {
                        x = self.parent[x as usize];
                    }
                    x == *b
                });
                if same {
                    to_merge = Some(TermId(root));
                    break;
                }
            }
            if to_merge.is_some() {
                break;
            }
        }
        if let Some(other) = to_merge {
            self.unify(new_id, other)?;
        }
        Ok(())
    }

    /* ── FD propagation ─────────────────────────────────── */

    fn propagate_fds(&mut self, work: &mut Vec<(TermId, TermId)>) -> Result<(), Conflict> {
        for idx in 0..self.pos_atoms.len() {
            self.propagate_fds_for_atom(idx, work)?;
        }
        Ok(())
    }

    fn propagate_fds_for_atom(
        &mut self,
        idx: usize,
        work: &mut Vec<(TermId, TermId)>,
    ) -> Result<(), Conflict> {
        let relation = self.pos_atoms[idx].relation.clone();
        let fds: Vec<FunctionalDep> = match self.fds.get(&relation) {
            Some(v) => v.to_vec(),
            None => return Ok(()),
        };
        for fd in &fds {
            for other in 0..self.pos_atoms.len() {
                if other == idx {
                    continue;
                }
                if self.pos_atoms[other].relation != relation {
                    continue;
                }
                if self.pos_atoms[other].args.len() != self.pos_atoms[idx].args.len() {
                    continue;
                }
                let keys_match = fd.keys.iter().all(|&k| {
                    let a = self.pos_atoms[idx].args[k];
                    let b = self.pos_atoms[other].args[k];
                    self.find(a) == self.find(b)
                });
                if !keys_match {
                    continue;
                }
                for &v in &fd.vals {
                    let a = self.pos_atoms[idx].args[v];
                    let b = self.pos_atoms[other].args[v];
                    if self.find(a) != self.find(b) {
                        work.push((a, b));
                    }
                }
            }
        }
        Ok(())
    }

    /* ── final saturation and verdict ───────────────────── */

    /// Saturate any pending FD/functor implications and
    /// validate negative atoms. Returns Ok(()) on SAT,
    /// Err(Conflict) on UNSAT.
    pub fn check(&mut self) -> Result<(), Conflict> {
        // Iterate to fixpoint: FD propagation may produce
        // merges that require more propagation. We hash
        // the current root assignment per atom-arg and
        // stop once it stabilizes.
        let mut prev = u64::MAX;
        loop {
            let snapshot: Vec<Vec<TermId>> =
                self.pos_atoms.iter().map(|a| a.args.clone()).collect();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::Hasher;
            for atom_args in &snapshot {
                for t in atom_args {
                    let r = self.find(*t);
                    hasher.write_u32(r);
                }
                hasher.write_u32(u32::MAX);
            }
            let key = hasher.finish();
            if key == prev {
                break;
            }
            prev = key;
            let mut work = Vec::new();
            self.propagate_fds(&mut work)?;
            while let Some((x, y)) = work.pop() {
                self.unify_one(x, y, &mut work)?;
            }
        }
        // Negation: a positive atom whose args are all
        // class-equal to the negative atom's args is a
        // contradiction.
        let neg_atoms = self.neg_atoms.clone();
        let pos_atoms = self.pos_atoms.clone();
        for neg in &neg_atoms {
            for pos in &pos_atoms {
                if pos.relation != neg.relation || pos.args.len() != neg.args.len() {
                    continue;
                }
                // Wildcards in a negation atom are
                // universally quantified — `!R(arg0, _)`
                // matches any positive `R(arg0, X)`. We
                // detect a wildcard by its source label
                // and skip the class check for that slot.
                let aligned = pos.args.iter().zip(neg.args.iter()).all(|(p, n)| {
                    if self.is_wildcard(*n) {
                        return true;
                    }
                    self.find(*p) == self.find(*n)
                });
                if aligned {
                    return Err(Conflict::DistinctConstructors {
                        a: format!("!{}", neg.relation),
                        b: pos.relation.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn is_wildcard(&self, t: TermId) -> bool {
        matches!(
            self.nodes.get(t.0 as usize).map(|n| &n.label),
            Some(TermLabel::Wildcard { .. })
        )
    }

    /* ── debug / introspection ──────────────────────────── */

    /// Pretty-print the equivalence classes for tests.
    #[cfg(test)]
    pub fn debug_classes(&mut self) -> String {
        let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
        for i in 0..self.parent.len() {
            let r = self.find(TermId(i as u32));
            groups.entry(r).or_default().push(i as u32);
        }
        let mut out = String::new();
        let mut keys: Vec<_> = groups.keys().copied().collect();
        keys.sort();
        for r in keys {
            let members = &groups[&r];
            let class = self.classes.get(&r);
            out.push_str(&format!("class {}: {:?}", r, members));
            if let Some(c) = class {
                if let Some(v) = &c.concrete {
                    out.push_str(&format!(" const={}", v));
                }
                if let Some(c) = &c.constructor {
                    out.push_str(&format!(" ${}/{}", c.name, c.args.len()));
                }
            }
            out.push('\n');
        }
        out
    }
}

/* ────────────────────────────────────────────────────────
Tests
──────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::super::parser::parse;
    use super::*;

    fn parse_body(src: &str) -> Vec<Literal> {
        let prog = parse(src, "test.dl").expect("parse");
        prog.rules.into_iter().next().expect("rule").body
    }

    #[test]
    fn equality_merges_terms() {
        let body = parse_body(r#"R(x) :- a = "hello", x = a."#);
        let mut chase = Chase::new();
        chase.assert_body(&body).unwrap();
        chase.check().unwrap();
    }

    #[test]
    fn distinct_constants_conflict() {
        let body = parse_body(r#"R(x) :- x = "a", x = "b"."#);
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        assert!(matches!(res, Err(Conflict::DistinctConstants { .. })));
    }

    #[test]
    fn constructor_disjointness() {
        let body = parse_body(r#"R(a) :- a = $CallTool("x", "y"), a = $SendAttempt(m)."#);
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        assert!(
            matches!(res, Err(Conflict::DistinctConstructors { .. })),
            "expected constructor conflict, got {:?}",
            res
        );
    }

    #[test]
    fn constructor_injectivity_propagates_to_args() {
        let body = parse_body(r#"R(a) :- a = $CallTool("submit", x), a = $CallTool("delete", y)."#);
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        // "submit" merged with "delete" via injectivity → distinct constants.
        assert!(
            matches!(res, Err(Conflict::DistinctConstants { .. })),
            "expected injectivity conflict, got {:?}",
            res
        );
    }

    #[test]
    fn fd_on_actions_merges_action_terms() {
        // Actions(idx, a) FD: idx → a. So if Actions(idx, a)
        // and Actions(idx, b) are both asserted, a = b.
        let body = parse_body(
            r#"R(idx) :- Actions(idx, a), Actions(idx, b),
                       a = $CallTool("x", "p"),
                       b = $CallTool("y", "q")."#,
        );
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        // idx → a forces a = b; constructor injectivity then
        // forces "x" = "y" → distinct constants.
        assert!(
            matches!(res, Err(Conflict::DistinctConstants { .. })),
            "expected FD-driven constant conflict, got {:?}",
            res
        );
    }

    #[test]
    fn singleton_principal() {
        let body = parse_body(r#"R() :- Principal("alice"), Principal("bob")."#);
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        assert!(
            matches!(res, Err(Conflict::DistinctConstants { .. })),
            "expected singleton conflict, got {:?}",
            res
        );
    }

    #[test]
    fn disequality_violation() {
        let body = parse_body(r#"R(x) :- x = "a", x != "a"."#);
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        assert!(matches!(res, Err(Conflict::DisequalMerged { .. })));
    }

    #[test]
    fn satisfiable_normal_body() {
        let body = parse_body(
            r#"R(idx) :- Actions(idx, a), a = $CallTool("submit", args), HasRole("admin")."#,
        );
        let mut chase = Chase::new();
        chase.assert_body(&body).unwrap();
        chase.check().unwrap();
    }

    #[test]
    fn functor_determinism_propagates() {
        // @url_host(u) is a function: same input → same output.
        let body = parse_body(
            r#"R(host) :- h1 = @url_host(u), h2 = @url_host(u), h1 = "openai.com", h2 = "openfda.gov"."#,
        );
        let mut chase = Chase::new();
        let res = chase.assert_body(&body);
        assert!(
            matches!(res, Err(Conflict::DistinctConstants { .. })),
            "expected determinism conflict, got {:?}",
            res
        );
    }

    #[test]
    fn cross_body_unification() {
        // body_A: Actions(idx, a), a = $CallTool("submit", _)
        // body_D: Actions(idx, a), a = $CallTool("delete", _)
        // Heads share idx; we unify them and expect UNSAT.
        let prog = parse(
            r#"
                A(idx) :- Actions(idx, a), a = $CallTool("submit", w).
                D(idx) :- Actions(idx, a), a = $CallTool("delete", w).
            "#,
            "test.dl",
        )
        .unwrap();
        let mut chase = Chase::new();
        let va = chase.assert_body(&prog.rules[0].body).unwrap();
        let vd = chase.assert_body(&prog.rules[1].body).unwrap();
        // Unify idx across scopes.
        let res = chase.unify(va["idx"], vd["idx"]);
        if res.is_ok() {
            // The Actions FD then merges a_A and a_D, and
            // constructor injectivity merges "submit" and
            // "delete" → conflict surfaces in `check`.
            let res = chase.check();
            assert!(
                matches!(res, Err(Conflict::DistinctConstants { .. })),
                "expected post-check conflict, got {:?}",
                res
            );
        } else {
            assert!(matches!(res, Err(Conflict::DistinctConstants { .. })));
        }
    }

    #[test]
    fn cross_body_compatible_passes() {
        let prog = parse(
            r#"
                A(idx) :- Actions(idx, a), a = $CallTool("submit", w).
                D(idx) :- Actions(idx, a), HasRole("admin").
            "#,
            "test.dl",
        )
        .unwrap();
        let mut chase = Chase::new();
        let va = chase.assert_body(&prog.rules[0].body).unwrap();
        let vd = chase.assert_body(&prog.rules[1].body).unwrap();
        chase.unify(va["idx"], vd["idx"]).unwrap();
        chase.check().unwrap();
    }
}
