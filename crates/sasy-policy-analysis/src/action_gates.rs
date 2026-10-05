//! Bounded action-demand inference for the two existing ancestry gates.
//!
//! This is a post-sugar, source-preserving optimization, not a policy emitter.
//! Only provenance-marked sugar defaults may be replaced. All ordinary policy
//! text and explicit gate definitions remain byte-for-byte unchanged.
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;

use crate::analysis::{
    ast::{Atom, CompareOp, Literal, Program, Rule, Term},
    lexer::{tokenize, Token},
    parse_with_options, ParseOptions,
};

const GATES: [(&str, &str); 2] = [
    ("CurrentDependsPolicyRelevant", "CurrentDepends"),
    ("ReachableFromPolicyRelevant", "ReachableFrom"),
];
const ROOTS: [&str; 9] = [
    "HasPrincipal",
    "Authorized",
    "IsAuthorized",
    "Unauthorized",
    "DenialReason",
    "ApplyTransform",
    "DenyUnauthorized",
    "AllowPassthrough",
    "SasyAllowRoute",
];
const MARKER: &str = "// SASY_AUTO_GATE_DEFAULT: ";
const USER_MARKER: &str = "// === USER_POLICY_BEGIN ===";
const MAX_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 100_000;
const MAX_DEPTH: usize = 32;
const MAX_WORK: usize = 200_000;
const MAX_PATTERNS: usize = 64;
const MAX_CLAUSE_ATOMS: usize = 32;
const MAX_CONDITION_BYTES: usize = 32 * 1024;
const MAX_ATOM_BYTES: usize = 1024;
const MAX_CONDITION_WORK: usize = 16_000_000;

#[derive(Debug, Clone, Serialize)]
pub struct ActionGateTransform {
    pub source: String,
    pub report: GateReport,
}
#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    pub gates: Vec<GateDecision>,
    pub recursive_groups: Vec<Vec<String>>,
    pub fallback: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct GateDecision {
    pub gate: String,
    pub helper: String,
    /// preserved, unconditional, inferred, or unused.
    pub outcome: String,
    /// Action patterns appearing in conditions (a summary, not their Boolean formula).
    pub action_patterns: Vec<String>,
    /// ORed clauses of conjunctive, independently existential positive atoms.
    pub conditions: Vec<String>,
}

/// `authored_user_source` is either the isolated user policy or the combined
/// post-sugar source with its exact USER_POLICY_BEGIN boundary. Included user
/// outputs must be present. With no boundary, all outputs count as observable.
/// A default is editable only with a lexically active, adjacent compiler-owned
/// SASY_AUTO_GATE_DEFAULT marker. Callers must strip authored marker comments
/// before creating fresh provenance. For a separate isolated user argument,
/// USER_POLICY_BEGIN text has no authority and is not used to trim that source.
/// Missing provenance, authored gates, unsupported syntax, and exhausted budgets
/// never cause an inferred restrictive gate to be emitted.
pub fn infer_action_gates(desugared: &str, authored_user_source: &str) -> ActionGateTransform {
    let mut report = GateReport {
        gates: Vec::new(),
        recursive_groups: Vec::new(),
        fallback: None,
    };
    if desugared.len() > MAX_BYTES || authored_user_source.len() > MAX_BYTES {
        return preserved(desugared, report, "source byte limit");
    }
    let lines: Vec<&str> = desugared.split_inclusive('\n').collect();
    let editable = auto_default_lines(desugared);
    if editable.values().all(Vec::is_empty) {
        return preserved(desugared, report, "no provenance-marked defaults");
    }
    let active = lexical_lines(authored_user_source).0;
    let authored = (authored_user_source == desugared)
        .then(|| {
            authored_user_source
                .lines()
                .enumerate()
                .find(|(index, line)| {
                    active.get(*index) == Some(&true) && line.trim() == USER_MARKER
                })
                .map(|(line, _)| {
                    authored_user_source
                        .split_inclusive('\n')
                        .skip(line + 1)
                        .collect::<String>()
                })
        })
        .flatten();
    let authored = authored.as_deref().unwrap_or(authored_user_source);
    let authored = without_auto_defaults(authored);
    let inference = (|| -> Result<_, &'static str> {
        preflight(desugared)?;
        let authored_tokens = checked_tokens(&authored)?;
        let mut explicit = BTreeSet::new();
        let mut roots: BTreeSet<String> = ROOTS.iter().map(|s| s.to_string()).collect();
        for pair in authored_tokens.windows(2) {
            if matches!(pair[0].token, Token::DotOutput | Token::DotPrintsize) {
                if let Token::Ident(name) = &pair[1].token {
                    roots.insert(name.clone());
                    if GATES.iter().any(|(g, _)| g == name) {
                        explicit.insert(name.clone());
                    }
                }
            }
        }
        // A user may also observe the gate directly inside a rule. Preserve
        // any authored gate use, not only its definitions, to keep that meaning.
        for window in authored_tokens.windows(2) {
            if let Token::Ident(name) = &window[0].token {
                if GATES.iter().any(|(g, _)| g == name) && matches!(window[1].token, Token::LParen)
                {
                    explicit.insert(name.clone());
                }
            }
        }
        let program = parse_with_options(
            desugared,
            "<action-gates>",
            ParseOptions {
                expand_body_disjunctions: false,
                reject_duplicate_relations: true,
                full_rule_spans: true,
                preserve_aggregates: false,
            },
        )
        .map_err(|_| "unsupported policy syntax")?
        .into_program();
        // Gate conditions must depend only on request EDB. A user definition
        // of Actions can depend on the very ancestry being gated and create a
        // new empty recursive cycle even though its action pattern is necessary.
        if program.rules.iter().any(|r| r.head.relation == "Actions")
            || program.facts.iter().any(|f| f.atom.relation == "Actions")
        {
            return Err("authored Actions definitions require conservative fallback");
        }
        let graph = dependency_graph(&program);
        let groups = recursive_groups(&graph);
        let mut engine = Inference::new(&program, &graph, &groups)?;
        let mut demand: BTreeMap<String, Demand> =
            roots.into_iter().map(|r| (r, Demand::Always)).collect();
        let mut queue: Vec<String> = demand.keys().cloned().collect();
        while let Some(head) = queue.pop() {
            engine.tick()?;
            let incoming = demand[&head].clone();
            for rule in engine.rules.get(&head).cloned().unwrap_or_default() {
                let local = engine.body_support(&rule.body, &HashMap::new(), &mut Vec::new())?;
                let need = engine.combine(incoming.clone(), local, true)?;
                for dependency in dependencies(&rule.body) {
                    let old = demand.get(&dependency).cloned().unwrap_or(Demand::Never);
                    let new = engine.combine(old.clone(), need.clone(), false)?;
                    if old != new {
                        demand.insert(dependency.clone(), new);
                        queue.push(dependency);
                    }
                }
            }
        }
        Ok((explicit, demand, groups))
    })();
    let (explicit, demand) = match inference {
        Ok((explicit, demand, groups)) => {
            report.recursive_groups = groups;
            (explicit, demand)
        }
        Err(reason) => return preserved(desugared, report, reason),
    };
    let mut replacements = BTreeMap::new();
    for (gate, helper) in GATES {
        let indices = &editable[gate];
        if indices.is_empty() || explicit.contains(gate) {
            report
                .gates
                .push(decision(gate, helper, "preserved", &Demand::Always));
            continue;
        }
        let need = demand.get(helper).cloned().unwrap_or(Demand::Never);
        let outcome = match need {
            Demand::Always => "unconditional",
            Demand::Never => "unused",
            Demand::Clauses(_) => "inferred",
        };
        report.gates.push(decision(gate, helper, outcome, &need));
        let replacement = match &need {
            Demand::Always => format!("{gate}().\n"),
            Demand::Never => format!("// {gate}: no observable consumer requires this helper.\n"),
            Demand::Clauses(clauses) => clauses
                .iter()
                .map(|clause| {
                    format!(
                        "{gate}() :- {}.\n",
                        clause.iter().cloned().collect::<Vec<_>>().join(", ")
                    )
                })
                .collect(),
        };
        // Duplicate sugar defaults have the same meaning; keep one replacement.
        for (i, line) in indices.iter().enumerate() {
            replacements.insert(
                *line,
                if i == 0 {
                    replacement.clone()
                } else {
                    String::new()
                },
            );
        }
    }
    let source = lines
        .iter()
        .enumerate()
        .map(|(i, line)| replacements.get(&i).map_or(*line, String::as_str))
        .collect();
    ActionGateTransform { source, report }
}

fn preserved(source: &str, mut report: GateReport, reason: &str) -> ActionGateTransform {
    report.fallback = Some(reason.to_string());
    report.gates = GATES
        .iter()
        .map(|(g, h)| decision(g, h, "preserved", &Demand::Always))
        .collect();
    ActionGateTransform {
        source: source.to_string(),
        report,
    }
}
fn decision(gate: &str, helper: &str, outcome: &str, demand: &Demand) -> GateDecision {
    GateDecision {
        gate: gate.into(),
        helper: helper.into(),
        outcome: outcome.into(),
        action_patterns: match demand {
            Demand::Clauses(clauses) => clauses
                .iter()
                .flatten()
                .filter_map(|atom| {
                    atom.strip_prefix("Actions(_, ")
                        .and_then(|s| s.strip_suffix(')'))
                        .map(str::to_string)
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            _ => Vec::new(),
        },
        conditions: match demand {
            Demand::Clauses(clauses) => clauses
                .iter()
                .map(|clause| clause.iter().cloned().collect::<Vec<_>>().join(", "))
                .collect(),
            _ => Vec::new(),
        },
    }
}
// Lexical line starts distinguish actual compiler line comments from marker
// text inside strings/block comments. The caller still supplies provenance:
// authored marker comments must be stripped before sugar creates fresh ones.
fn lexical_lines(source: &str) -> (Vec<bool>, bool) {
    let bytes = source.as_bytes();
    let mut active = vec![true];
    let mut state = 0; // normal, string, block comment, line comment
    let mut faithful = true;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' {
            if state == 3 {
                state = 0;
            }
            active.push(state == 0);
            i += 1;
            continue;
        }
        match state {
            1 if b == b'\\' => {
                let next = bytes.get(i + 1).copied();
                if !matches!(next, Some(b'n' | b't' | b'r' | b'\\' | b'"')) {
                    faithful = false;
                }
                i += if next == Some(b'\n') { 1 } else { 2 };
                continue;
            }
            1 if b == b'"' => state = 0,
            2 if bytes[i..].starts_with(b"*/") => {
                state = 0;
                i += 2;
                continue;
            }
            0 if bytes[i..].starts_with(b"//") => {
                state = 3;
                i += 2;
                continue;
            }
            0 if bytes[i..].starts_with(b"/*") => {
                state = 2;
                i += 2;
                continue;
            }
            0 if b == b'"' => state = 1,
            _ => {}
        }
        i += 1;
    }
    (active, faithful)
}
fn auto_default_lines(source: &str) -> BTreeMap<&'static str, Vec<usize>> {
    let lines: Vec<_> = source.split_inclusive('\n').collect();
    let active = lexical_lines(source).0;
    GATES
        .iter()
        .map(|(gate, _)| {
            let indices = lines
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    (active[i]
                        && active[i + 1]
                        && pair[0].trim() == format!("{MARKER}{gate}")
                        && pair[1].trim() == format!("{gate}()."))
                    .then_some(i + 1)
                })
                .collect();
            (*gate, indices)
        })
        .collect()
}
fn without_auto_defaults(source: &str) -> String {
    let automatic: BTreeSet<usize> = auto_default_lines(source).into_values().flatten().collect();
    source
        .split_inclusive('\n')
        .enumerate()
        .filter_map(|(i, line)| (!automatic.contains(&i)).then_some(line))
        .collect()
}
fn checked_tokens(source: &str) -> Result<Vec<crate::analysis::lexer::Spanned>, &'static str> {
    if source.len() > MAX_BYTES {
        return Err("source byte limit");
    }
    let tokens = tokenize(source).map_err(|_| "unsupported lexical syntax")?;
    if tokens.len() > MAX_TOKENS {
        return Err("token limit");
    }
    Ok(tokens)
}
fn preflight(source: &str) -> Result<(), &'static str> {
    if !lexical_lines(source).1 {
        return Err("non-faithful string escape requires conservative fallback");
    }
    let tokens = checked_tokens(source)?;
    // The parser currently discards these relation qualifiers. In particular,
    // inline/magic can duplicate a reused predicate's stateful functor body,
    // and eqrel adds implicit recursive semantics missing from our graph.
    // Preserve defaults for all erased qualifiers until modeled explicitly.
    for (i, token) in tokens.iter().enumerate() {
        if !matches!(token.token, Token::DotDecl) {
            continue;
        }
        let mut declaration_depth = 0usize;
        for (j, next) in tokens.iter().enumerate().skip(i + 1) {
            match &next.token {
                Token::LParen => declaration_depth += 1,
                Token::RParen => {
                    declaration_depth = declaration_depth.saturating_sub(1);
                    if declaration_depth == 0 {
                        if let Some(crate::analysis::lexer::Spanned {
                            token: Token::Ident(name),
                            ..
                        }) = tokens.get(j + 1)
                        {
                            if [
                                "inline",
                                "no_inline",
                                "brie",
                                "btree",
                                "btree_delete",
                                "eqrel",
                                "override",
                                "magic",
                                "no_magic",
                            ]
                            .contains(&name.as_str())
                            {
                                return Err(
                                    "erased relation qualifier requires conservative fallback",
                                );
                            }
                        }
                        break;
                    }
                }
                // Malformed declarations are rejected by the parser; do not
                // repeatedly scan following declarations during this preflight.
                Token::DotDecl | Token::DotType | Token::Dot => break,
                _ => {}
            }
        }
    }
    let mut depth = 0usize;
    let mut run = 0;
    for token in tokens {
        run += 1;
        match token.token {
            Token::LParen | Token::LBracket | Token::LBrace => {
                depth += 1;
                if depth > MAX_DEPTH {
                    return Err("syntax nesting limit");
                }
            }
            Token::RParen | Token::RBracket | Token::RBrace => {
                depth = depth.saturating_sub(1);
            }
            Token::Dot | Token::DotDecl | Token::DotType => run = 0,
            // Cast erasure and aggregate placeholders are unsuitable proof inputs.
            Token::KwAs => return Err("casts require conservative fallback"),
            Token::Ident(ref name)
                if ["count", "sum", "min", "max", "mean"].contains(&name.as_str()) =>
            {
                return Err("aggregate or reserved syntax requires conservative fallback")
            }
            _ => {}
        }
        if run > 512 {
            return Err("statement token limit");
        }
    }
    Ok(())
}

/// Necessary (not sufficient) condition for any tuple in a relation. Never is
/// used only for absent observable demand, not to prove unknown predicates empty.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Demand {
    Never,
    Always,
    Clauses(BTreeSet<BTreeSet<String>>),
}
impl Demand {
    fn atom(atom: String) -> Self {
        if atom.len() > MAX_ATOM_BYTES {
            return Self::Always;
        }
        Self::Clauses(BTreeSet::from([BTreeSet::from([atom])]))
    }
    // Every atom projects its variables independently to wildcards. Conjoining
    // these existential conditions never identifies witnesses across actions,
    // rule invocations, or helper scopes. Losing joins only widens a gate.
    fn bounded(clauses: BTreeSet<BTreeSet<String>>) -> Self {
        if clauses.iter().any(BTreeSet::is_empty) {
            return Self::Always;
        }
        // A OR (A AND B) = A. Absorb redundant paths before checking the
        // alternative count, otherwise harmless repeated demand widens to true.
        let reduced: BTreeSet<BTreeSet<String>> = clauses
            .iter()
            .filter(|c| {
                !clauses
                    .iter()
                    .any(|other| other.len() < c.len() && other.is_subset(c))
            })
            .cloned()
            .collect();
        if reduced.len() > MAX_PATTERNS
            || reduced.iter().any(|c| c.len() > MAX_CLAUSE_ATOMS)
            || reduced.iter().flatten().map(String::len).sum::<usize>() > MAX_CONDITION_BYTES
        {
            // The intersection of every alternative is still necessary. Keep
            // that bounded common prerequisite when a full DNF is too large.
            let mut common = reduced.iter().next().cloned().unwrap_or_default();
            for clause in &reduced {
                common.retain(|atom| clause.contains(atom));
            }
            let common = common
                .into_iter()
                .take(MAX_CLAUSE_ATOMS)
                .collect::<BTreeSet<_>>();
            return if common.is_empty() {
                Self::Always
            } else {
                Self::Clauses(BTreeSet::from([common]))
            };
        }
        Self::Clauses(reduced)
    }
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Always, _) | (_, Self::Always) => Self::Always,
            (Self::Never, d) | (d, Self::Never) => d,
            (Self::Clauses(mut a), Self::Clauses(b)) => {
                a.extend(b);
                Self::bounded(a)
            }
        }
    }
    fn and_approx(self, other: Self) -> Self {
        match (self, other) {
            (Self::Never, _) | (_, Self::Never) => Self::Never,
            (Self::Always, d) | (d, Self::Always) => d,
            (Self::Clauses(a), Self::Clauses(b)) => {
                if a.len().saturating_mul(b.len()) > MAX_PATTERNS {
                    // A AND B implies A: retain a bounded existing prerequisite
                    // instead of expanding a large Cartesian product.
                    return Self::Clauses(a);
                }
                Self::bounded(
                    a.iter()
                        .flat_map(|left| {
                            b.iter()
                                .map(move |right| left.union(right).cloned().collect())
                        })
                        .collect(),
                )
            }
        }
    }
}

struct Inference<'a> {
    program: &'a Program,
    rules: HashMap<String, Vec<&'a Rule>>,
    factual: BTreeSet<String>,
    independent: BTreeSet<String>,
    recursive: BTreeSet<String>,
    work: usize,
    condition_work: usize,
}
impl<'a> Inference<'a> {
    fn new(
        program: &'a Program,
        graph: &BTreeMap<String, BTreeSet<String>>,
        groups: &[Vec<String>],
    ) -> Result<Self, &'static str> {
        let mut rules: HashMap<String, Vec<&Rule>> = HashMap::new();
        for rule in &program.rules {
            rules
                .entry(rule.head.relation.clone())
                .or_default()
                .push(rule);
        }
        let factual = program
            .facts
            .iter()
            .map(|f| f.atom.relation.clone())
            .collect();
        // Signed dependencies count equally. One unsafe defining alternative
        // makes the entire relation unsafe to reuse. Propagate backwards from
        // BOTH gated closures/gates and every recursive SCC, never just the gate
        // currently being inferred (which could introduce a cross-gate cycle).
        let mut unsafe_relations: BTreeSet<String> = GATES
            .iter()
            .flat_map(|(gate, helper)| [gate.to_string(), helper.to_string()])
            .chain(groups.iter().flatten().cloned())
            .collect();
        let mut reverse: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (head, deps) in graph {
            for dep in deps {
                reverse.entry(dep.clone()).or_default().push(head.clone());
            }
        }
        let mut todo: Vec<_> = unsafe_relations.iter().cloned().collect();
        let mut work = MAX_WORK;
        while let Some(name) = todo.pop() {
            for dependent in reverse.get(&name).into_iter().flatten() {
                work = work.checked_sub(1).ok_or("inference work limit")?;
                if unsafe_relations.insert(dependent.clone()) {
                    todo.push(dependent.clone());
                }
            }
        }
        let independent = program
            .relations
            .keys()
            .filter(|name| !unsafe_relations.contains(*name))
            .cloned()
            .collect();
        Ok(Self {
            program,
            rules,
            factual,
            independent,
            recursive: groups.iter().flatten().cloned().collect(),
            work,
            condition_work: MAX_CONDITION_WORK,
        })
    }
    fn tick(&mut self) -> Result<(), &'static str> {
        self.work = self.work.checked_sub(1).ok_or("inference work limit")?;
        Ok(())
    }
    fn combine(
        &mut self,
        left: Demand,
        right: Demand,
        conjunction: bool,
    ) -> Result<Demand, &'static str> {
        if let (Demand::Clauses(a), Demand::Clauses(b)) = (&left, &right) {
            let candidates = if conjunction {
                a.len().saturating_mul(b.len())
            } else {
                a.len() + b.len()
            };
            // Absorption compares candidate clauses pairwise; each clause has
            // at most twice the atom cap before normalization. Charge that
            // bounded work separately from relation traversal, before allocation.
            let cost = if conjunction && candidates > MAX_PATTERNS {
                1
            } else {
                candidates.saturating_mul(candidates + MAX_CLAUSE_ATOMS * 2)
            };
            self.condition_work = self
                .condition_work
                .checked_sub(cost)
                .ok_or("condition work limit")?;
        }
        Ok(if conjunction {
            left.and_approx(right)
        } else {
            left.or(right)
        })
    }
    fn atom_support(
        &mut self,
        atom: &Atom,
        outer: &HashMap<String, Term>,
        stack: &mut Vec<String>,
    ) -> Result<Demand, &'static str> {
        self.tick()?;
        let args: Vec<_> = atom.args.iter().map(|a| resolve(a, outer, 0)).collect();
        if atom.relation == "Actions" {
            return Ok(args
                .get(1)
                .and_then(action_pattern)
                .map_or(Demand::Always, |pattern| {
                    Demand::atom(format!("Actions(_, {pattern})"))
                }));
        }
        let retained = if self.independent.contains(&atom.relation) {
            // Reuse the materialized relation; never copy its functor-bearing
            // body into generated rules. Unbound variables, records, casts and
            // opaque terms project to independent existential wildcards.
            Demand::atom(format!(
                "{}({})",
                atom.relation,
                args.iter().map(project_term).collect::<Vec<_>>().join(", ")
            ))
        } else {
            Demand::Always
        };
        if retained != Demand::Always {
            return Ok(retained);
        }
        // Never summarize a recursive relation by unrolling it: its consumers
        // are handled by the signed dependency worklist instead.
        if self.recursive.contains(&atom.relation) {
            return Ok(Demand::Always);
        }
        // Unknown/EDB/factual/recursive paths are never assumed action-scoped.
        if self.program.is_edb(&atom.relation)
            || self.factual.contains(&atom.relation)
            || stack.contains(&atom.relation)
            || stack.len() >= 16
        {
            return Ok(retained);
        }
        let rules = self.rules.get(&atom.relation).cloned().unwrap_or_default();
        if rules.is_empty() {
            return Ok(retained);
        }
        stack.push(atom.relation.clone());
        let mut result = Demand::Never;
        for rule in rules {
            let mut bindings = HashMap::new();
            for (formal, actual) in rule.head.args.iter().zip(&args) {
                bind(formal, &erase_variables(actual), &mut bindings);
            }
            let support = self.body_support(&rule.body, &bindings, stack)?;
            result = self.combine(result, support, false)?;
            if result == Demand::Always {
                break;
            }
        }
        stack.pop();
        self.combine(retained, result, true)
    }
    fn body_support(
        &mut self,
        body: &[Literal],
        initial: &HashMap<String, Term>,
        stack: &mut Vec<String>,
    ) -> Result<Demand, &'static str> {
        self.tick()?;
        let mut bindings = initial.clone();
        // Equalities are conjunctive here. Branch equalities get separate maps.
        for literal in body {
            if let Literal::Compare {
                op: CompareOp::Eq,
                left,
                right,
                ..
            } = literal
            {
                bind(left, right, &mut bindings);
            }
        }
        let mut demand = Demand::Always;
        for literal in body {
            self.tick()?;
            let support = match literal {
                Literal::Pos(atom) => self.atom_support(atom, &bindings, stack)?,
                Literal::Disjunction { alternatives, .. } => {
                    let mut alternatives_demand = Demand::Never;
                    for alternative in alternatives {
                        let support = self.body_support(alternative, &bindings, stack)?;
                        alternatives_demand = self.combine(alternatives_demand, support, false)?;
                    }
                    alternatives_demand
                }
                // Absence of a relation cannot establish existence of an action.
                _ => Demand::Always,
            };
            demand = self.combine(demand, support, true)?;
        }
        Ok(demand)
    }
}

fn resolve(term: &Term, bindings: &HashMap<String, Term>, _depth: usize) -> Term {
    fn walk(term: &Term, bindings: &HashMap<String, Term>, depth: usize, left: &mut usize) -> Term {
        if depth > 16 || *left == 0 {
            return Term::Wildcard(0);
        }
        *left -= 1;
        match term {
            Term::Var(v) => bindings
                .get(v)
                .map_or_else(|| term.clone(), |t| walk(t, bindings, depth + 1, left)),
            Term::Constructor { name, args } => Term::Constructor {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|t| walk(t, bindings, depth + 1, left))
                    .collect(),
            },
            _ => term.clone(),
        }
    }
    walk(term, bindings, 0, &mut 128)
}
fn erase_variables(term: &Term) -> Term {
    match term {
        Term::Var(_) => Term::Wildcard(0),
        Term::Constructor { name, args } => Term::Constructor {
            name: name.clone(),
            args: args.iter().map(erase_variables).collect(),
        },
        _ => term.clone(),
    }
}
fn bind(left: &Term, right: &Term, bindings: &mut HashMap<String, Term>) {
    let left = resolve(left, bindings, 0);
    let right = resolve(right, bindings, 0);
    match (&left, &right) {
        (Term::Var(a), Term::Var(b)) if a == b => {}
        (Term::Var(a), b) if !matches!(b, Term::Wildcard(_)) => {
            bindings.insert(a.clone(), b.clone());
        }
        (a, Term::Var(b)) if !matches!(a, Term::Wildcard(_)) => {
            bindings.insert(b.clone(), a.clone());
        }
        (Term::Constructor { name: a, args: aa }, Term::Constructor { name: b, args: bb })
            if a == b && aa.len() == bb.len() =>
        {
            for (a, b) in aa.iter().zip(bb) {
                bind(a, b, bindings);
            }
        }
        _ => {}
    }
}
fn action_pattern(term: &Term) -> Option<String> {
    let Term::Constructor { name, args } = term else {
        return None;
    };
    if !matches!(
        (name.as_str(), args.len()),
        ("CallTool", 2) | ("SendAttempt", 1) | ("HTTPRequest", 3)
    ) {
        return None;
    }
    Some(format!(
        "${name}({})",
        args.iter().map(project_term).collect::<Vec<_>>().join(", ")
    ))
}
fn project_term(arg: &Term) -> String {
    match arg {
        // No new escape spelling, casts, arithmetic or functor evaluation.
        Term::StringLit(s)
            if s.len() <= 128
                && s.bytes()
                    .all(|b| (32..127).contains(&b) && b != b'"' && b != b'\\') =>
        {
            format!("\"{s}\"")
        }
        Term::NumberLit(n) => n.to_string(),
        Term::UnsignedLit(n) => n.to_string(),
        // Only the validated request ADT constructors are reproduced. Other
        // compound terms are deliberately widened instead of emitted.
        Term::Constructor { .. } => action_pattern(arg).unwrap_or_else(|| "_".into()),
        _ => "_".to_string(),
    }
}

fn dependencies(body: &[Literal]) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    let mut todo: Vec<_> = body.iter().collect();
    while let Some(literal) = todo.pop() {
        match literal {
            Literal::Pos(a) | Literal::Neg(a) => {
                result.insert(a.relation.clone());
            }
            Literal::Negation { literal, .. } => todo.push(literal),
            Literal::Disjunction { alternatives, .. } => todo.extend(alternatives.iter().flatten()),
            _ => {}
        }
    }
    result
}
fn dependency_graph(program: &Program) -> BTreeMap<String, BTreeSet<String>> {
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for Rule { head, body, .. } in &program.rules {
        let deps = dependencies(body);
        for dep in &deps {
            graph.entry(dep.clone()).or_default();
        }
        graph.entry(head.relation.clone()).or_default().extend(deps);
    }
    graph
}
/// Iterative Kosaraju; recursion groups are diagnostic, not proof shortcuts.
fn recursive_groups(graph: &BTreeMap<String, BTreeSet<String>>) -> Vec<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for name in graph.keys() {
        let mut stack = vec![(name.clone(), false)];
        while let Some((name, visited)) = stack.pop() {
            if visited {
                order.push(name);
                continue;
            }
            if !seen.insert(name.clone()) {
                continue;
            }
            stack.push((name.clone(), true));
            stack.extend(graph[&name].iter().rev().cloned().map(|n| (n, false)));
        }
    }
    let mut reverse: BTreeMap<String, Vec<String>> =
        graph.keys().map(|k| (k.clone(), Vec::new())).collect();
    for (head, deps) in graph {
        for dep in deps {
            reverse.get_mut(dep).unwrap().push(head.clone());
        }
    }
    seen.clear();
    let mut result = Vec::new();
    for root in order.into_iter().rev() {
        if seen.contains(&root) {
            continue;
        }
        let mut group = Vec::new();
        let mut stack = vec![root];
        while let Some(name) = stack.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            stack.extend(reverse[&name].iter().cloned());
            group.push(name);
        }
        group.sort();
        if group.len() > 1 || graph[&group[0]].contains(&group[0]) {
            result.push(group);
        }
    }
    result.sort();
    result
}
