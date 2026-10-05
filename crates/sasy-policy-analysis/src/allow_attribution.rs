//! Bounded, additive why-not attribution over a completed request snapshot.
//!
//! This is a necessary-condition filter, not reachability or a repair solver.
//! Custom predicates are changeable and are never queried by the projection.
//! Only a small, definition-checked set of request inputs/common helpers can
//! eliminate a route. Unsupported conditions remain unknown. Original policy
//! bytes are preserved; diagnostic rules never feed authorization relations.

use std::collections::HashSet;

use crate::analysis::{
    self,
    lexer::{tokenize, Token},
    Atom, CompareOp, Literal, Term,
};
use crate::rule_metadata::RuleMetadataParser;

pub const RELATION: &str = "SasyAllowRoute";
pub const MAX_ROUTES: usize = 64;
pub const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_STATEMENTS: usize = 8192;
const MAX_RULE_BYTES: usize = 16 * 1024;
const MAX_LITERALS: usize = 32;
const MAX_DEPTH: usize = 32;
const MAX_GENERATED_BYTES: usize = 256 * 1024;
const MAX_HINT_BYTES: usize = 1024;
const MAX_SUGGESTIONS: usize = 4;

#[derive(Debug, Clone)]
pub struct RouteMetadata {
    pub rule_id: String,
    /// Post-sugar location, not a claimed raw-source map.
    pub source_location: String,
    pub message: String,
    pub suggestions: Vec<String>,
    /// True if every condition was understood as fixed or changeable.
    /// A possible projection with unsupported conditions emits `unknown`.
    pub supported: bool,
}

#[derive(Debug, Clone)]
pub struct AttributionTransform {
    pub source: String,
    pub routes: Vec<RouteMetadata>,
    /// All syntactic allow routes were enumerated, not a sufficiency proof.
    pub complete: bool,
    pub fallback_reason: Option<String>,
}

fn fallback(source: &str, reason: &str) -> AttributionTransform {
    AttributionTransform {
        source: source.into(),
        routes: vec![],
        complete: false,
        fallback_reason: Some(reason.into()),
    }
}

#[derive(Debug)]
struct Statement<'a> {
    text: &'a str,
    line: u32,
    tokens: Vec<Token>,
}

// Mask comments without changing byte offsets. A lexical depth cap runs before
// the recursive parser, and semicolon branches are never expanded by it here.
fn masked(source: &str) -> Result<String, &'static str> {
    let mut out = source.as_bytes().to_vec();
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut quoted = false;
    let mut stack = Vec::new();
    while i < bytes.len() {
        if quoted {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                quoted = false;
            }
            i += 1;
            continue;
        }
        if bytes[i] == b'"' {
            quoted = true;
            i += 1;
            continue;
        }
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            out[i] = b' ';
            out[i + 1] = b' ';
            i += 2;
            while i < bytes.len() && !bytes[i..].starts_with(b"*/") {
                if bytes[i] != b'\n' {
                    out[i] = b' ';
                }
                i += 1;
            }
            if i == bytes.len() {
                return Err("unterminated comment");
            }
            out[i] = b' ';
            out[i + 1] = b' ';
            i += 2;
            continue;
        }
        match bytes[i] {
            b'(' | b'[' | b'{' => {
                stack.push(bytes[i]);
                if stack.len() > MAX_DEPTH {
                    return Err("source nesting limit");
                }
            }
            b')' | b']' | b'}' => {
                let expected = match bytes[i] {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                if stack.pop() != Some(expected) {
                    return Err("unbalanced source");
                }
            }
            b'#' => return Err("unresolved preprocessing directive"),
            _ => {}
        }
        i += 1;
    }
    if quoted || !stack.is_empty() {
        return Err("unbalanced source");
    }
    String::from_utf8(out).map_err(|_| "invalid source")
}

fn statements(clean: &str) -> Result<Vec<Statement<'_>>, &'static str> {
    let b = clean.as_bytes();
    let mut result = Vec::new();
    let mut start = 0;
    let mut line = 1;
    while start < b.len() {
        while start < b.len() && b[start].is_ascii_whitespace() {
            if b[start] == b'\n' {
                line += 1;
            }
            start += 1;
        }
        if start == b.len() {
            break;
        }
        let directive = b[start] == b'.';
        let mut end = start;
        let mut depth = 0i32;
        let mut quoted = false;
        while end < b.len() {
            if quoted {
                if b[end] == b'\\' {
                    end += 2;
                    continue;
                }
                if b[end] == b'"' {
                    quoted = false;
                }
            } else {
                match b[end] {
                    b'"' => quoted = true,
                    b'(' | b'[' | b'{' => depth += 1,
                    b')' | b']' | b'}' => depth -= 1,
                    b'.' if !directive && depth == 0 => {
                        end += 1;
                        break;
                    }
                    // Multiline ADT declarations continue with `|`.
                    b'\n'
                        if directive
                            && depth == 0
                            && !clean[end + 1..].trim_start().starts_with('|') =>
                    {
                        break;
                    }
                    _ => {}
                }
            }
            end += 1;
        }
        if end > b.len() {
            return Err("invalid string escape");
        }
        let text = clean[start..end].trim();
        let tokens = tokenize(text)
            .map_err(|_| "unsupported source syntax")?
            .into_iter()
            .filter_map(|t| (t.token != Token::Eof).then_some(t.token))
            .collect::<Vec<_>>();
        if tokens
            .iter()
            .any(|t| matches!(t, Token::Ident(name) if name.starts_with("SasyAllow")))
        {
            return Err("reserved attribution namespace is already used");
        }
        if directive
            && !matches!(
                tokens.first(),
                Some(
                    Token::DotDecl
                        | Token::DotInput
                        | Token::DotOutput
                        | Token::DotType
                        | Token::DotFunctor
                        | Token::DotPrintsize
                )
            )
        {
            return Err("unsupported directive");
        }
        result.push(Statement { text, line, tokens });
        if result.len() > MAX_STATEMENTS {
            return Err("statement limit");
        }
        line += clean[start..end].bytes().filter(|&c| c == b'\n').count() as u32;
        start = end;
    }
    Ok(result)
}

fn relation_head<'a>(statement: &'a Statement<'_>) -> Option<&'a str> {
    match statement.tokens.as_slice() {
        [Token::Ident(name), Token::LParen, ..] => Some(name),
        _ => None,
    }
}

fn signature(statement: &Statement<'_>, name: &str, types: &[&str]) -> bool {
    let Ok(program) = analysis::parse(statement.text, "attribution") else {
        return false;
    };
    let Some(decl) = program.relations.get(name) else {
        return false;
    };
    decl.params.len() == types.len()
        && decl
            .params
            .iter()
            .zip(types)
            .all(|((_, ty), expected)| ty == &analysis::TypeRef::from_str(expected))
}

fn trusted_relations(statements: &[Statement<'_>]) -> HashSet<String> {
    // Request-local inputs only. Graph ancestry and policy-defined projections
    // of metadata are intentionally excluded; they can be gated or changeable.
    let inputs: &[(&str, &[&str])] = &[
        ("Actions", &["unsigned", "Action"]),
        ("Principal", &["symbol"]),
        ("PrincipalRole", &["symbol", "symbol"]),
        ("Entity", &["symbol"]),
        ("TenantId", &["symbol"]),
        ("Current", &["symbol"]),
        (
            "ActionMetadata",
            &["unsigned", "symbol", "symbol", "symbol"],
        ),
        ("PolicyMetadata", &["symbol", "symbol", "symbol"]),
    ];
    let mut trusted = HashSet::new();
    for &(name, types) in inputs {
        let decls = statements.iter().filter(|s| matches!(s.tokens.as_slice(), [Token::DotDecl, Token::Ident(n), ..] if n == name)).collect::<Vec<_>>();
        let input_count = statements.iter().filter(|s| matches!(s.tokens.as_slice(), [Token::DotInput, Token::Ident(n), ..] if n == name)).count();
        if decls.len() == 1
            && signature(decls[0], name, types)
            && input_count == 1
            && !statements.iter().any(|s| relation_head(s) == Some(name))
        {
            trusted.insert(name.to_string());
        }
    }
    // No name-only trust for helpers. Extra facts/rules or changed definitions
    // disable pruning through the helper. No external function is copied.
    let helpers: &[(&str, &[&str], &[&str], &str)] = &[
        (
            "HasPrincipal",
            &[],
            &["Principal"],
            "HasPrincipal() :- Principal(_).",
        ),
        (
            "HasRole",
            &["symbol"],
            &["Principal", "PrincipalRole"],
            "HasRole(role) :- Principal(p), PrincipalRole(p, role).",
        ),
        (
            "IsTool",
            &["Action", "symbol"],
            &["Actions"],
            "IsTool(a, name) :- Actions(_, a), a = $CallTool(name, _).",
        ),
        (
            "IsToolCall",
            &["Action"],
            &["Actions"],
            "IsToolCall(a) :- Actions(_, a), a = $CallTool(_, _).",
        ),
    ];
    for &(name, types, dependencies, canonical) in helpers {
        let rules = statements
            .iter()
            .filter(|s| relation_head(s) == Some(name))
            .collect::<Vec<_>>();
        let decls = statements.iter().filter(|s| matches!(s.tokens.as_slice(), [Token::DotDecl, Token::Ident(n), ..] if n == name)).collect::<Vec<_>>();
        let expected = tokenize(canonical)
            .expect("constant helper")
            .into_iter()
            .filter_map(|t| (t.token != Token::Eof).then_some(t.token))
            .collect::<Vec<_>>();
        if dependencies.iter().all(|d| trusted.contains(*d)) && rules.len() == 1
            && rules[0].tokens == expected && decls.len() == 1 && signature(decls[0], name, types)
            && !statements.iter().any(|s| matches!(s.tokens.as_slice(), [Token::DotInput, Token::Ident(n), ..] if n == name)) {
            trusted.insert(name.into());
        }
    }
    trusted
}

fn vars(term: &Term, out: &mut HashSet<String>) -> bool {
    match term {
        Term::Var(name) => {
            out.insert(name.clone());
            true
        }
        Term::Wildcard(_) | Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => true,
        Term::Constructor { args, .. } | Term::RecordLit(args) => args.iter().all(|t| vars(t, out)),
        _ => false, // Includes builtins: diagnostics never add functor calls.
    }
}
fn term_vars(term: &Term) -> Option<HashSet<String>> {
    let mut result = HashSet::new();
    vars(term, &mut result).then_some(result)
}
fn grounded(term: &Term, bound: &HashSet<String>) -> bool {
    match term {
        Term::Var(name) => bound.contains(name),
        Term::StringLit(_) | Term::NumberLit(_) | Term::UnsignedLit(_) => true,
        Term::Constructor { args, .. } | Term::RecordLit(args) => {
            args.iter().all(|t| grounded(t, bound))
        }
        // A wildcard is a pattern, not a value from which another variable
        // can be grounded after a custom relation has been removed.
        _ => false,
    }
}
fn atom_vars(atom: &Atom) -> Option<HashSet<String>> {
    let mut result = HashSet::new();
    atom.args
        .iter()
        .all(|t| vars(t, &mut result))
        .then_some(result)
}

// Original literal slices, rather than AST pretty-printing, preserve numeric
// spelling and Soufflé string escapes exactly. Only top-level conjunctions are
// supported; an unsupported disjunction gets an unknown route.
fn body_slices(text: &str) -> Option<Vec<&str>> {
    let (_, body) = text.split_once(":-")?;
    let body = body.trim().strip_suffix('.')?;
    let mut start = 0;
    let mut depth = 0;
    let mut quoted = false;
    let mut escaped = false;
    let mut result = Vec::new();
    for (i, c) in body.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                result.push(body[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(body[start..].trim());
    Some(result)
}

// These common relations need graph completeness, gating, or external-function
// evidence that this transform cannot establish. Never treat them as an empty
// immutable relation, and do not imply a completed necessary-condition check.
fn snapshot_unknown(name: &str) -> bool {
    matches!(
        name,
        "Edge"
            | "EdgeData"
            | "EdgePrincipal"
            | "EdgeEntity"
            | "SentMessage"
            | "ToolResult"
            | "MessageMetadata"
            | "CurrentDepends"
            | "ReachableFrom"
            | "CurrentDependsPolicyRelevant"
            | "ReachableFromPolicyRelevant"
            | "ToolResultFieldName"
            | "ToolResultField"
            | "QueriesHost"
    )
}

fn projection(
    statement: &Statement<'_>,
    trusted: &HashSet<String>,
) -> Option<(String, Vec<String>, bool)> {
    if statement.text.len() > MAX_RULE_BYTES
        || statement
            .tokens
            .iter()
            .any(|t| matches!(t, Token::Semicolon | Token::KwAs | Token::KwNil))
    {
        return None;
    }
    let program = analysis::parse(statement.text, "allow-policy").ok()?;
    let (head, body) = if let [rule] = program.rules.as_slice() {
        (&rule.head, rule.body.as_slice())
    } else if let [fact] = program.facts.as_slice() {
        (&fact.atom, [].as_slice())
    } else {
        return None;
    };
    if head.args.len() != 1 || body.len() > MAX_LITERALS {
        return None;
    }
    let index = match &head.args[0] {
        Term::Var(v) => v.clone(),
        _ => return None,
    };
    let raw = if body.is_empty() {
        vec![]
    } else {
        body_slices(statement.text)?
    };
    if raw.len() != body.len() {
        return None;
    }
    let mut bound = HashSet::from([index.clone()]);
    let mut keep = HashSet::new();
    let mut unknown = false;
    for (i, literal) in body.iter().enumerate() {
        if let Literal::Pos(atom) = literal {
            if trusted.contains(&atom.relation) {
                if let Some(v) = atom_vars(atom) {
                    bound.extend(v);
                    keep.insert(i);
                } else {
                    unknown = true;
                }
            }
        }
    }
    // Equality can expose fields of a grounded request ADT/record. Retain all
    // conditions in one joined projection, never per-condition existential
    // booleans that could silently switch witnesses.
    for _ in 0..=body.len() {
        let before = bound.len();
        for (i, literal) in body.iter().enumerate() {
            if let Literal::Compare {
                op: CompareOp::Eq,
                left,
                right,
                ..
            } = literal
            {
                if let (Some(l), Some(r)) = (term_vars(left), term_vars(right)) {
                    if grounded(left, &bound) || grounded(right, &bound) {
                        bound.extend(l);
                        bound.extend(r);
                        keep.insert(i);
                    }
                }
            }
        }
        if bound.len() == before {
            break;
        }
    }
    for (i, literal) in body.iter().enumerate() {
        match literal {
            Literal::Pos(atom) | Literal::Neg(atom) if !trusted.contains(&atom.relation) => {
                // User predicates stay changeable, even if currently empty or
                // recursively defined. Never evaluate their present membership.
                if atom_vars(atom).is_none() || snapshot_unknown(&atom.relation) {
                    unknown = true;
                }
            }
            Literal::Neg(atom) => {
                if atom_vars(atom).is_some_and(|v| v.is_subset(&bound)) {
                    keep.insert(i);
                } else {
                    unknown = true;
                }
            }
            Literal::Compare { left, right, .. } => {
                if keep.contains(&i) || (grounded(left, &bound) && grounded(right, &bound)) {
                    keep.insert(i);
                } else {
                    unknown = true;
                }
            }
            Literal::Pos(_) => {}
            _ => unknown = true,
        }
    }
    let body = raw
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, text)| text.to_string())
        .collect();
    Some((index, body, !unknown))
}

fn quoted(text: &str) -> String {
    // Soufflé accepts these escapes. Control characters are omitted from
    // author hints rather than allowed to inject source or terminal controls.
    let text = text.chars().filter(|c| !c.is_control()).collect::<String>();
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}
/// The form a rule's suggestions take in a route: the first few, each bounded.
/// Grouping later drops the ones on a route ruled out by fixed context and
/// removes repeats, so a caller pairing a route with its rule compares against
/// this list with those two steps applied as well.
pub fn route_suggestions(suggestions: &[String]) -> Vec<String> {
    suggestions
        .iter()
        .take(MAX_SUGGESTIONS)
        .map(|s| bounded_hint(s))
        .collect()
}

/// The form an authored message takes in a route: bounded, and without the
/// control characters the diagnostic relation cannot carry. A caller pairing a
/// route with the rule metadata it came from has to compare against this, not
/// against the original text.
pub fn bounded_hint(text: &str) -> String {
    let mut end = text.len().min(MAX_HINT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].chars().filter(|c| !c.is_control()).collect()
}
fn id(text: &str, ordinal: usize) -> String {
    let hash = text.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    format!("allow-{ordinal:02}-{hash:016x}")
}

/// Detect the reserved compiler namespace before admission, including when
/// attribution itself would fall back. This scan allocates no source copy and
/// does not rely on the subset parser accepting the policy. Callers must reject
/// collisions: silently falling back could let authored rows forge diagnostics.
pub fn reserved_namespace_collision(source: &str) -> bool {
    let b = source.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else if b[i] == b'"' {
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    i = (i + 2).min(b.len());
                } else if b[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
        } else if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            if b[start..i].starts_with(b"SasyAllow") {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Append bounded diagnostic-only rules to post-sugar source.
///
/// `blocked` means that this rule's trusted, fixed necessary conditions have
/// no joint witness in this request snapshot. `possible` means only that the
/// projection has a witness; it is never a promise of a feasible repair.
/// `unknown` retains unsupported routes. Callers must not infer an overall
/// impossibility result from this report, or use it to change a decision.
/// The caller must enforce its existing evaluation deadline/output byte caps.
pub fn transform_desugared(source: &str) -> AttributionTransform {
    if reserved_namespace_collision(source) {
        return fallback(source, "reserved attribution namespace is already used");
    }
    if source.len() > MAX_SOURCE_BYTES {
        return fallback(source, "source size limit");
    }
    let clean = match masked(source) {
        Ok(s) => s,
        Err(e) => return fallback(source, e),
    };
    let statements = match statements(&clean) {
        Ok(s) => s,
        Err(e) => return fallback(source, e),
    };
    let trusted = trusted_relations(&statements);
    if !trusted.contains("Actions") {
        return fallback(source, "Actions is not a trusted request input");
    }
    let allows = statements
        .iter()
        .filter(|s| relation_head(s) == Some("IsAuthorized"))
        .collect::<Vec<_>>();
    if allows.len() > MAX_ROUTES {
        return fallback(source, "allow route limit");
    }
    if allows.is_empty() {
        return AttributionTransform {
            source: source.into(),
            routes: vec![],
            complete: true,
            fallback_reason: None,
        };
    }
    let metadata = RuleMetadataParser::parse_content(source, "allow-policy".into());
    let mut generated = format!("\n// Bounded fixed-request allow-route attribution (diagnostic only).\n.decl {RELATION}(idx: unsigned, rule_id: symbol, status: symbol, details: symbol, suggestion: symbol, source_location: symbol)\n.output {RELATION}(IO=stdout, rfc4180=true)\n");
    let mut routes = Vec::new();
    let ambiguous_lines = allows
        .iter()
        .filter(|s| allows.iter().filter(|other| other.line == s.line).count() > 1)
        .map(|s| s.line)
        .collect::<HashSet<_>>();
    for (ordinal, statement) in allows.into_iter().enumerate() {
        let meta = (!ambiguous_lines.contains(&statement.line))
            .then(|| metadata.get_metadata(statement.line))
            .flatten();
        let route = RouteMetadata {
            rule_id: id(statement.text, ordinal),
            source_location: format!("allow-policy:{}", statement.line),
            message: meta
                .and_then(|m| m.deny_message.as_deref())
                .map(bounded_hint)
                .unwrap_or_default(),
            suggestions: meta
                .map(|m| {
                    m.suggestions
                        .iter()
                        .take(MAX_SUGGESTIONS)
                        .map(|s| bounded_hint(s))
                        .collect()
                })
                .unwrap_or_default(),
            supported: false,
        };
        let mut route = route;
        let projection = projection(statement, &trusted);
        let (index, conditions, supported) =
            projection.unwrap_or_else(|| ("SasyAllowIndex".into(), vec![], false));
        route.supported = supported;
        let helper = format!("SasyAllowFixed{ordinal}");
        generated.push_str(&format!(
            ".decl {helper}(idx: unsigned)\n{helper}({index}) :- Actions({index}, _)"
        ));
        for condition in conditions {
            generated.push_str(", ");
            generated.push_str(&condition);
        }
        generated.push_str(".\n");
        let mut hints = route.suggestions.clone();
        if hints.is_empty() {
            hints.push(String::new());
        }
        for hint in hints {
            for (status, condition) in [
                (
                    if supported { "possible" } else { "unknown" },
                    format!("{helper}(SasyAllowIndex)"),
                ),
                ("blocked", format!("!{helper}(SasyAllowIndex)")),
            ] {
                let details = if route.message.is_empty() {
                    String::new()
                } else {
                    route.message.clone()
                };
                generated.push_str(&format!("{RELATION}(SasyAllowIndex, {}, {}, {}, {}, {}) :- Actions(SasyAllowIndex, _), {condition}.\n",
                    quoted(&route.rule_id), quoted(status), quoted(&details), quoted(&hint), quoted(&route.source_location)));
            }
        }
        routes.push(route);
        if generated.len() > MAX_GENERATED_BYTES {
            return fallback(source, "generated source limit");
        }
    }
    AttributionTransform {
        source: format!("{source}{generated}"),
        routes,
        complete: true,
        fallback_reason: None,
    }
}
