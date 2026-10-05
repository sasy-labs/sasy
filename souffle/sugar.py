#!/usr/bin/env python3
"""
Soufflé syntactic sugar preprocessor.

Adds dot notation for record field access to Soufflé's Datalog dialect
and rewrites annotated Unauthorized rules into DenialReason rules.

Usage:
    python sugar.py input.dl > output.dl
    python sugar.py --check input.dl   # validate only, no output

Supported sugar:
    msg.agent           → positional unpack of record field
    msg.agent == "X"    → unpack + comparison
    f(msg.agent)        → unpack + pass variable to functor
    head(msg.agent)     → unpack in head via body variable

DenialReason rewriting:
    Unauthorized rules with @deny_message / @suggestion annotations are
    rewritten into DenialReason(idx, kind, reason, suggestion) rules, where
    kind is "ask" for a rule carrying // @ask and "block" otherwise.
    Unannotated Unauthorized rules get a default reason.
    common_policy.dl derives Unauthorized(idx) from a "block"-kind
    DenialReason.

Example:
    // Input (with sugar)
    .type Message = [contents: symbol, agent: symbol, role: AgentRole]
    .decl SentMessage(id: symbol, message: Message)
    Result(id) :- SentMessage(id, msg), msg.agent == "billing-agent".

    // Output (valid Soufflé)
    .type Message = [contents: symbol, agent: symbol, role: AgentRole]
    .decl SentMessage(id: symbol, message: Message)
    Result(id) :- SentMessage(id, msg), msg = [_, __msg_agent, _], __msg_agent = "billing-agent".
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Type registry ───────────────────────────────────────────────────

@dataclass
class RecordType:
    """A Soufflé record type with named fields."""
    name: str
    fields: list[tuple[str, str]]  # [(field_name, field_type), ...]

    def field_index(self, name: str) -> int:
        for i, (fn, _) in enumerate(self.fields):
            if fn == name:
                return i
        raise KeyError(f"No field '{name}' in record type '{self.name}'. "
                       f"Fields: {[f[0] for f in self.fields]}")

    @property
    def arity(self) -> int:
        return len(self.fields)


@dataclass
class ADTBranch:
    """A branch of an algebraic data type."""
    name: str
    fields: list[tuple[str, str]]  # [(field_name, field_type), ...]


@dataclass
class ADTType:
    """A Soufflé ADT (union type) with named branches."""
    name: str
    branches: list[ADTBranch]

    def get_branch(self, name: str) -> ADTBranch:
        for b in self.branches:
            if b.name == name:
                return b
        raise KeyError(f"No branch '{name}' in ADT '{self.name}'")


@dataclass
class RelationDecl:
    """A relation declaration with typed parameters."""
    name: str
    params: list[tuple[str, str]]  # [(param_name, param_type), ...]


@dataclass
class TypeRegistry:
    """Tracks all type and relation declarations."""
    records: dict[str, RecordType] = field(default_factory=dict)
    adts: dict[str, ADTType] = field(default_factory=dict)
    relations: dict[str, RelationDecl] = field(default_factory=dict)

    def resolve_type(self, type_name: str) -> RecordType | ADTType | None:
        if type_name in self.records:
            return self.records[type_name]
        if type_name in self.adts:
            return self.adts[type_name]
        return None

    def get_var_type(self, rel_name: str, param_index: int) -> str | None:
        """Get the type of a parameter by position in a relation."""
        rel = self.relations.get(rel_name)
        if rel and param_index < len(rel.params):
            return rel.params[param_index][1]
        return None


# ── Parsing ─────────────────────────────────────────────────────────

def parse_record_type(line: str) -> RecordType | None:
    """Parse: .type Name = [field1: type1, field2: type2, ...]"""
    # Strip all inline comments first
    line_clean = re.sub(r'//[^\n]*', '', line).strip()

    m = re.match(r'\.type\s+(\w+)\s*=\s*\[', line_clean)
    if not m:
        return None
    name = m.group(1)
    # Find matching ] bracket
    start = line_clean.index('[') + 1
    depth = 1
    end = start
    while end < len(line_clean) and depth > 0:
        if line_clean[end] == '[':
            depth += 1
        elif line_clean[end] == ']':
            depth -= 1
        end += 1
    if depth != 0:
        return None
    bracket_content = line_clean[start:end - 1]

    fields = []
    for part in bracket_content.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            fname, ftype = part.split(':', 1)
            fields.append((fname.strip(), ftype.strip()))

    return RecordType(name=name, fields=fields)


def parse_adt_type(line: str) -> ADTType | None:
    """Parse: .type Name = Branch1 { f1: t1 } | Branch2 { f2: t2 } | ..."""
    m = re.match(r'\.type\s+(\w+)\s*=\s*(.+)', line)
    if not m:
        return None
    name = m.group(1)
    rest = m.group(2).strip()

    # Check it's an ADT (has | or { })
    if '[' in rest and '|' not in rest:
        return None  # It's a record type, not ADT

    branches = []
    for part in rest.split('|'):
        part = part.strip()
        bm = re.match(r'(\w+)\s*\{([^}]*)\}', part)
        if bm:
            bname = bm.group(1)
            fields_str = bm.group(2).strip()
            fields = []
            if fields_str:
                for fp in fields_str.split(','):
                    fp = fp.strip()
                    if ':' in fp:
                        fn, ft = fp.split(':', 1)
                        fields.append((fn.strip(), ft.strip()))
            branches.append(ADTBranch(name=bname, fields=fields))

    if not branches:
        return None
    return ADTType(name=name, branches=branches)


def parse_relation_decl(line: str) -> RelationDecl | None:
    """Parse: .decl Name(param1: type1, param2: type2, ...)"""
    m = re.match(r'\.decl\s+(\w+)\s*\(([^)]*)\)', line)
    if not m:
        return None
    name = m.group(1)
    params_str = m.group(2)
    params = []
    for part in params_str.split(','):
        part = part.strip()
        if ':' in part:
            pname, ptype = part.split(':', 1)
            params.append((pname.strip(), ptype.strip()))
    return RelationDecl(name=name, params=params)


# ── Dot notation resolution ────────────────────────────────────────

DOT_PATTERN = re.compile(r'\b([a-z_]\w*)\.([a-z_]\w*)\b')


def find_var_type_in_rule(var_name: str, rule_body: str, registry: TypeRegistry) -> RecordType | None:
    """Find the record type of a variable by looking at relation atoms in the rule."""
    # Find all relation atoms: RelName(..., var_name, ...)
    # Match patterns like: RelName(arg1, var_name, arg3)
    atom_pattern = re.compile(r'(\w+)\s*\(([^)]+)\)')
    for m in atom_pattern.finditer(rule_body):
        rel_name = m.group(1)
        args_str = m.group(2)

        # Skip if this is a functor call (@func) or keyword
        if rel_name.startswith('@') or rel_name in ('not', 'count', 'min', 'max', 'sum'):
            continue

        args = split_args(args_str)
        for i, arg in enumerate(args):
            arg = arg.strip()
            if arg == var_name:
                # Found the variable at position i in relation rel_name
                type_name = registry.get_var_type(rel_name, i)
                if type_name:
                    resolved = registry.resolve_type(type_name)
                    if isinstance(resolved, RecordType):
                        return resolved
    return None


def split_args(s: str) -> list[str]:
    """Split comma-separated arguments, respecting brackets and parens."""
    args = []
    depth = 0
    current = []
    for c in s:
        if c in ('(', '['):
            depth += 1
            current.append(c)
        elif c in (')', ']'):
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            args.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        args.append(''.join(current))
    return args


def resolve_dots_in_rule(rule: str, registry: TypeRegistry) -> str:
    """Resolve all var.field references in a single rule."""
    # Split into head :- body (or just a fact)
    if ':-' in rule:
        head, body = rule.split(':-', 1)
        # Remove trailing period from body
        body = body.rstrip()
        if body.endswith('.'):
            body = body[:-1]
        trailing_dot = '.'
    else:
        return rule  # facts don't need transformation

    full_rule = head + ':-' + body

    # Find all dot accesses
    dots = list(DOT_PATTERN.finditer(full_rule))
    if not dots:
        return rule  # no dots to resolve

    # Group by variable name
    var_fields: dict[str, set[str]] = {}
    for m in dots:
        var_name = m.group(1)
        field_name = m.group(2)
        # Skip if this looks like a directive or keyword
        if var_name in ('std', 'str', 'json', 'url'):
            continue
        var_fields.setdefault(var_name, set()).add(field_name)

    if not var_fields:
        return rule

    # For each variable, find its type and generate unpacking
    replacements: dict[str, str] = {}  # "var.field" -> generated var name
    extra_clauses: list[str] = []

    # Detect simple equality patterns: var.field = "literal" or var.field = other_var
    # These can be inlined directly into the unpacking pattern
    simple_eq_pattern = re.compile(
        r'\b(\w+)\.(\w+)\s*=\s*([^,\n.]+?)(?:\s*[,.]|\s*$)'
    )
    # Also detect: "literal" = var.field and other_var = var.field
    simple_eq_pattern_rev = re.compile(  # noqa: F841 — reserved for future reverse-equality support
        r'([^,\n=]+?)\s*=\s*(\w+)\.(\w+)(?:\s*[,.]|\s*$)'
    )

    # Track which dot refs are inlined (don't generate separate vars for them)
    inlined: dict[str, str] = {}  # "var.field" -> inline value

    all_dot_refs = {f'{v}.{f}' for v, fields in var_fields.items() for f in fields}

    # Find inlinable equalities in body only
    for m in simple_eq_pattern.finditer(body):
        vn, fn, val = m.group(1), m.group(2), m.group(3).strip()
        ref = f'{vn}.{fn}'
        if ref not in all_dot_refs:
            continue
        # Don't inline if the value is itself a dot reference
        if DOT_PATTERN.search(val):
            continue
        # Don't inline if the value is a variable that has dot accesses
        if val in var_fields:
            continue
        # Check if val is a simple literal (not a variable with dot accesses)
        if (val.startswith('"') or val.startswith('$') or re.match(r'^\d+$', val)):
            inlined[ref] = val

    for var_name, fields in var_fields.items():
        # Find the record type
        rec_type = find_var_type_in_rule(var_name, full_rule, registry)
        if rec_type is None:
            rec_type = find_var_type_in_rule(var_name, head, registry)
        if rec_type is None:
            continue

        # Generate unpacking pattern
        unpack_vars = ['_'] * rec_type.arity
        for field_name in fields:
            try:
                idx = rec_type.field_index(field_name)
            except KeyError:
                continue
            ref = f'{var_name}.{field_name}'
            if ref in inlined:
                # Inline the value directly into the unpacking pattern
                unpack_vars[idx] = inlined[ref]
                replacements[ref] = inlined[ref]
            else:
                gen_var = f'__{var_name}_{field_name}'
                unpack_vars[idx] = gen_var
                replacements[ref] = gen_var

        unpack_clause = f'{var_name} = [{", ".join(unpack_vars)}]'
        extra_clauses.append(unpack_clause)

    if not replacements:
        return rule

    # Apply replacements to the full rule text
    new_rule = full_rule
    for dot_ref, gen_var in sorted(replacements.items(), key=lambda x: -len(x[0])):
        new_rule = new_rule.replace(dot_ref, gen_var)

    # Remove inlined equality clauses from body (they're now in the unpack)
    if ':-' in new_rule:
        head_part, body_part = new_rule.split(':-', 1)
        body_part = body_part.strip()

        # Remove clauses that were inlined: "__var_field = value" or "value = __var_field"
        for ref, val in inlined.items():
            # The dot ref was already replaced with the inline value,
            # so the clause looks like: val = val (tautology). Remove it.
            # Pattern: ", val = val" or "val = val,"
            tautology = f'{val} = {val}'
            body_part = body_part.replace(f', {tautology}', '')
            body_part = body_part.replace(f'{tautology}, ', '')
            body_part = body_part.replace(tautology, '')

        # Clean up any double commas or trailing commas
        body_part = re.sub(r',\s*,', ',', body_part)
        body_part = re.sub(r',\s*$', '', body_part)
        body_part = re.sub(r'^\s*,', '', body_part)

        for clause in extra_clauses:
            body_part = body_part.rstrip() + ', ' + clause
        return head_part + ':- ' + body_part.strip() + trailing_dot
    else:
        return new_rule + trailing_dot


# ── Multi-line joining ──────────────────────────────────────────────

def join_multiline(lines: list[str]) -> list[str]:
    """Join multi-line type declarations and rules into single lines."""
    result = []
    current = []
    in_type = False
    bracket_depth = 0

    # Strip inline comments before joining (they break single-line parsing)
    stripped_lines = []
    in_block_comment = False
    for line in lines:
        if in_block_comment:
            if '*/' in line:
                in_block_comment = False
                stripped_lines.append(line[line.index('*/') + 2:])
            else:
                stripped_lines.append('')
            continue
        if '/*' in line and '*/' not in line:
            in_block_comment = True
            stripped_lines.append(line[:line.index('/*')])
            continue
        # Strip inline // comments but preserve standalone comment lines
        s = line.strip()
        if s.startswith('//'):
            stripped_lines.append(line)  # keep full-line comments
        elif '//' in line:
            stripped_lines.append(line[:line.index('//')])
        else:
            stripped_lines.append(line)

    for line in stripped_lines:
        stripped = line.strip()

        # Track bracket depth for type declarations
        if stripped.startswith('.type') and '[' in stripped and ']' not in stripped:
            in_type = True
            bracket_depth = stripped.count('[') - stripped.count(']')
            current.append(stripped)
            continue

        if in_type:
            bracket_depth += stripped.count('[') - stripped.count(']')
            current.append(stripped)
            if bracket_depth <= 0:
                result.append(' '.join(current))
                current = []
                in_type = False
            continue

        # Join rules that span multiple lines (no trailing period)
        if current:
            if stripped:
                current.append(stripped)
            if stripped.endswith('.') or stripped.endswith('}') or not stripped:
                result.append(' '.join(current))
                current = []
            continue

        # Start of a new rule/fact (has :- or relation head but no trailing .)
        if (stripped and not stripped.startswith('//') and not stripped.startswith('/*')
                and not stripped.startswith('.') and not stripped.startswith('#')
                and ':-' not in stripped and not stripped.endswith('.')
                and not stripped.endswith('{') and not stripped.endswith('}')
                and re.match(r'\w+\(', stripped)):
            current.append(stripped)
            continue

        if (stripped and ':-' in stripped and not stripped.endswith('.')):
            current.append(stripped)
            continue

        result.append(line)

    if current:
        result.append(' '.join(current))

    return result


# ── DenialReason rewriting ─────────────────────────────────────────

# Pattern to match Unauthorized rule heads: Unauthorized(var) or Unauthorized(idx)
UNAUTHORIZED_HEAD = re.compile(r'^Unauthorized\(\s*(\w+)\s*\)')

ANNOTATION_KEY = re.compile(r'^//\s*@(\w+):\s*(.*)')
# Flag annotations carry no value, e.g. `// @ask`.
ANNOTATION_FLAG = re.compile(r'^//\s*@(\w+)\s*$')


def _collect_annotations(comment_lines: list[str]) -> dict[str, str]:
    """Extract @key: value annotations (and @flag flags) from comment lines."""
    annotations: dict[str, str] = {}
    for line in comment_lines:
        s = line.strip()
        m = ANNOTATION_KEY.match(s)
        if m:
            key, value = m.group(1), m.group(2).strip()
            if key in ('deny_message', 'message', 'reason'):
                annotations['deny_message'] = value
            elif key in ('suggestion', 'fix', 'hint'):
                annotations['suggestion'] = value
            continue
        f = ANNOTATION_FLAG.match(s)
        if f and f.group(1) == 'ask':
            # `@ask` marks the denial soft: the reference monitor prompts the
            # user instead of hard-denying when an approval channel exists.
            annotations['ask'] = '1'
    return annotations


def _rewrite_unauthorized_to_denial_reason(lines: list[str]) -> list[str]:
    """Rewrite Unauthorized(idx) rules into DenialReason(idx, kind, reason, suggestion).

    Scans lines for Unauthorized rules, collects preceding annotation
    comments, and rewrites the rule head. The original Unauthorized(idx)
    is derived from DenialReason in common_policy.dl.

    Comment lines with @deny_message / @suggestion / @url_pattern / @tool_pattern
    that were consumed by the rewrite are dropped from output (they are now
    encoded in the DenialReason tuple).
    """
    output: list[str] = []
    # Buffer of consecutive comment lines that may contain annotations
    comment_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Accumulate comment lines (potential annotations)
        if stripped.startswith('//'):
            comment_buffer.append(line)
            continue

        # Check if this is an Unauthorized rule to rewrite.
        # Skip the derivation rule from common_policy.dl
        # (Unauthorized(idx) :- DenialReason(idx, "block", _, _).)
        m = UNAUTHORIZED_HEAD.match(stripped)
        if m and ':-' in stripped and 'DenialReason' not in stripped:
            idx_var = m.group(1)
            annotations = _collect_annotations(comment_buffer)
            reason = annotations.get('deny_message', 'Action is denylisted')
            suggestion = annotations.get('suggestion', '')
            # `@ask` makes the denial soft (kind="ask"); default is a hard block.
            kind = 'ask' if annotations.get('ask') else 'block'

            # Escape quotes in annotation values for Soufflé string literals
            reason_escaped = reason.replace('\\', '\\\\').replace('"', '\\"')
            suggestion_escaped = suggestion.replace('\\', '\\\\').replace('"', '\\"')

            # Rewrite head:
            #   Unauthorized(idx) → DenialReason(idx, "block"|"ask", "reason", "suggestion")
            new_head = f'DenialReason({idx_var}, "{kind}", "{reason_escaped}", "{suggestion_escaped}")'
            new_line = UNAUTHORIZED_HEAD.sub(new_head, stripped)

            # Emit non-annotation comments, drop annotation comments
            for cl in comment_buffer:
                cl_stripped = cl.strip()
                if not ANNOTATION_KEY.match(cl_stripped):
                    output.append(cl)

            output.append(new_line)
            comment_buffer = []
            continue

        # Not an Unauthorized rule — flush comment buffer as-is
        output.extend(comment_buffer)
        comment_buffer = []
        output.append(line)

    # Flush remaining comments
    output.extend(comment_buffer)
    return output


# ── Main preprocessor ──────────────────────────────────────────────

def _has_rule_for(lines: list[str], relation: str) -> bool:
    """Return True if any line declares a rule (or fact) whose head is
    ``relation``. Matches both ``Foo(x) :- ...`` and ``Foo(x).`` shapes,
    skips comments and ``.decl`` / ``.output`` directives.
    """
    head_pat = re.compile(rf"^\s*{re.escape(relation)}\s*\(")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("//", "/*", "*", "#", ".")):
            continue
        if head_pat.match(line):
            return True
    return False


USER_POLICY_BEGIN_MARKER = "// === USER_POLICY_BEGIN ==="

# Each gate predicate is paired with the IDBs whose rules in
# common_policy.dl join on it. If the user policy doesn't reference
# any of those IDBs, the gate is irrelevant — we leave it un-set
# (gate empty → IDBs empty without computation) instead of emitting
# a trivially-true default. Policies that *do* reference the IDBs
# but don't write their own gate rule still get the trivially-true
# default so behavior is preserved.
# A helper in common_policy.dl whose rules join on a gated relation is empty
# whenever the gate is, so a policy that uses the helper consumes the gate
# just as much as one that names the gated relation itself. Every such helper
# has to be listed, or the gate stays unset and the helper silently yields
# nothing — a rule that never fires, not an error.
# ``tests/test_sugar_gate_defaults.py`` recomputes this closure from
# common_policy.dl and fails when a new helper is missing from it.
_GATE_RELATIONS = (
    (
        "CurrentDependsPolicyRelevant",
        (
            "CurrentDepends",
            "ToolResultField",
            "UnattributedInput",
            "DependsOnUnattributedInput",
            "UnattributedInputOrigin",
        ),
    ),
    ("ReachableFromPolicyRelevant", ("ReachableFrom",)),
)


def _user_policy_lines(lines: list[str]) -> list[str]:
    """Return only the user-policy portion of the merged source.

    ``_resolve_includes`` injects a marker at the boundary so the
    user policy can be isolated from the prepended common_policy.
    When the marker is absent (e.g., callers that build the source
    themselves without going through ``_resolve_includes``), we
    fall back to the full source — the conservative default is to
    treat all references as user references.
    """
    in_string = in_block = False
    for i, line in enumerate(lines):
        if not in_string and not in_block and line.strip() == USER_POLICY_BEGIN_MARKER:
            return lines[i + 1:]
        index = 0
        while index < len(line):
            if in_string:
                if line[index] == '\\':
                    index += 2
                    continue
                if line[index] == '"':
                    in_string = False
            elif in_block:
                if line.startswith('*/', index):
                    in_block = False
                    index += 2
                    continue
            elif line.startswith('//', index):
                break
            elif line.startswith('/*', index):
                in_block = True
                index += 2
                continue
            elif line[index] == '"':
                in_string = True
            index += 1
    return lines


def _references_relation(lines: list[str], relation: str) -> bool:
    """Return True if any line references ``relation`` as a body
    literal (or fact in any rule). We scan with a simple regex
    since accurate parsing is out of scope here; comments and
    directives are skipped.
    """
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(relation)}\s*\(")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("//", "/*", "*", "#", ".")):
            continue
        if pat.search(line):
            return True
    return False


def _default_rules(lines: list[str]) -> list[str]:
    """Append default rules for opt-in gating relations the policy
    doesn't define — but only when the policy actually consumes the
    gated IDB.

    Three cases per gate predicate:

    1. User policy declares its own rule for the gate → respect it,
       emit nothing.
    2. User policy doesn't declare a rule but does reference the
       gated IDB → emit a trivially-true default so behavior is
       preserved.
    3. User policy neither declares a rule nor references the gated
       IDB → emit nothing. The gate stays empty, the IDB stays
       empty, and Soufflé skips the (empty) recursion entirely.
    """
    user_lines = _user_policy_lines(lines)
    additions: list[str] = []
    for relation, gated_idbs in _GATE_RELATIONS:
        # Scope the rule check to user lines so implication rules in
        # common_policy.dl (e.g. ``CurrentDependsPolicyRelevant :-
        # ReachableFromPolicyRelevant``) don't suppress the
        # behavior-preserving default for policies that use the
        # gated IDB without writing their own gate rule.
        if _has_rule_for(user_lines, relation):
            continue  # case 1
        consumes_idb = any(
            _references_relation(user_lines, idb) for idb in gated_idbs
        )
        if not consumes_idb:
            continue  # case 3 — leave gate empty
        additions.extend([
            "",
            f"// Auto-emitted by sugar.py: no custom rule for "
            f"{relation}, so the gate is trivially",
            "// satisfied and the corresponding helper rules "
            "behave as if ungated.",
            f"// SASY_AUTO_GATE_DEFAULT: {relation}",
            f"{relation}().",
        ])
    return lines + additions


def _clear_authored_gate_markers(source: str, *, clear_user_boundary: bool = False) -> str:
    """Only this preprocessing invocation may mark generated gate defaults.

    Remove provenance spelling in comments, including block comments that look
    like generated lines. Preserve literal strings and ordinary policy bytes.
    """
    parts = []
    start = index = 0
    while index < len(source):
        if source[index] == '"':
            index += 1
            while index < len(source):
                if source[index] == '\\':
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        elif source.startswith('//', index) or source.startswith('/*', index):
            if source.startswith('//', index):
                end = source.find('\n', index)
                if end < 0:
                    end = len(source)
            else:
                end = source.find('*/', index + 2)
                end = len(source) if end < 0 else end + 2
            parts.append(source[start:index])
            comment = source[index:end].replace('SASY_AUTO_GATE_DEFAULT:', 'SASY_AUTHORED_GATE_MARKER:')
            if clear_user_boundary:
                comment = comment.replace('=== USER_POLICY_BEGIN ===', '=== AUTHORED_POLICY_BOUNDARY ===')
            parts.append(comment)
            start = index = end
        else:
            index += 1
    parts.append(source[start:])
    return ''.join(parts)


def preprocess(source: str) -> str:
    """Preprocess Soufflé source with dot notation sugar."""
    lines = _clear_authored_gate_markers(source).splitlines()

    # Phase 1: Join multi-line declarations and rules
    lines = join_multiline(lines)

    # Phase 1b: Inject defaults for opt-in gating predicates that the
    # policy didn't declare itself. Done before
    # ``_rewrite_unauthorized_to_denial_reason`` so any subsequent
    # phases see the synthesized rules in the same shape as
    # author-written ones.
    lines = _default_rules(lines)

    # Phase 2: Rewrite Unauthorized rules into DenialReason
    lines = _rewrite_unauthorized_to_denial_reason(lines)

    # Phase 3: Parse type and relation declarations
    registry = TypeRegistry()

    for line in lines:
        stripped = line.strip()

        # Try record type
        rec = parse_record_type(stripped)
        if rec:
            registry.records[rec.name] = rec
            continue

        # Try ADT type
        adt = parse_adt_type(stripped)
        if adt:
            registry.adts[adt.name] = adt
            # Also register each branch's fields as accessible
            continue

        # Try relation declaration
        rel = parse_relation_decl(stripped)
        if rel:
            registry.relations[rel.name] = rel

    # Phase 4: Transform rules with dot notation
    output = []
    for line in lines:
        stripped = line.strip()

        # Skip comments and directives
        if (not stripped or stripped.startswith('//') or stripped.startswith('/*')
                or stripped.startswith('*') or stripped.startswith('.')
                or stripped.startswith('#')):
            output.append(line)
            continue

        # Check if this line has any dot notation
        if DOT_PATTERN.search(stripped):
            transformed = resolve_dots_in_rule(stripped, registry)
            output.append(transformed)
        else:
            output.append(line)

    return '\n'.join(output)


def _find_common_policy(start: Path) -> Path | None:
    """Locate `policies/common_policy.dl`.

    Lookup order (first match wins):
      1. `COMMON_POLICY_DL` env var (explicit override)
      2. Walking up from `start`'s directory for `policies/common_policy.dl`
      3. Walking up from this sugar.py's own directory

    Matches the sasy Rust compiler's behavior — common_policy.dl is
    prepended automatically and any `#include` for it is stripped,
    so a user-supplied policy doesn't need a correct relative
    path (or any include at all).
    """
    env = os.environ.get("COMMON_POLICY_DL")
    if env:
        p = Path(env)
        if p.exists():
            return p

    roots = []
    here = start.resolve() if start.is_dir() else start.resolve().parent
    roots.append(here)
    roots.extend(here.parents)
    # Fallback: walk up from sugar.py's own directory (repo layout)
    script_dir = Path(__file__).resolve().parent
    roots.append(script_dir)
    roots.extend(script_dir.parents)

    for cand in roots:
        p = cand / "policies" / "common_policy.dl"
        if p.exists():
            return p
    return None


def preprocess_file(path: str, *, include_root: str | None = None) -> str:
    """Preprocess a file, resolving #include directives first.

    If a `policies/common_policy.dl` exists in the repo (found by
    walking up from the input file), it is automatically prepended
    and any existing `#include` for common_policy.dl is stripped.
    This matches the behavior of the Rust compiler's
    compile_souffle_with_assets() pipeline.

    Idempotency: if the input already contains common_policy's content
    (detected via the ``.decl IsAuthorized`` marker), skip the prepend.
    This matters because the Rust compile pipeline prepends common_policy
    itself and then invokes sugar.py on the combined file — without this
    check we'd inline common_policy twice and trigger
    "Redefinition of relation" errors in Soufflé.
    """
    p = Path(path)
    root = Path(include_root) if include_root else None
    source = _resolve_includes(p, strip_common=True, root=root)

    already_has_common = bool(
        re.search(r"^\.decl\s+IsAuthorized\b", source, re.MULTILINE)
    )
    if not already_has_common:
        common = _find_common_policy(p)
        if common is not None:
            # common_policy.dl is found by walking up the tree, so it is
            # rooted at its own directory rather than the caller's.
            common_text = _resolve_includes(common, strip_common=True)
            # Marker so default-rule injection can scan only the
            # user portion when deciding whether to emit gating
            # defaults — see ``_default_rules``.
            source = (
                f"{common_text}\n\n"
                f"{USER_POLICY_BEGIN_MARKER}\n\n"
                f"{source}"
            )

    # join_multiline runs inside preprocess(), so this is correct
    return preprocess(source)


class IncludeError(Exception):
    """An #include that cannot be resolved safely."""


# Deep enough for any real policy; a bound at all is what stops a cycle
# between files that each include the other from recursing forever.
_MAX_INCLUDE_DEPTH = 32


def _within(candidate: Path, root: Path) -> bool:
    """Is `candidate` inside `root`, judged lexically?

    Deliberately lexical: ``os.path.abspath`` normalises ``..`` without
    following symlinks. Resolving them would reject the repository's own
    ``policies/common_policy.dl``, which is a symlink to
    ``souffle/common_policy.dl`` and so points outside the
    directory it lives in.
    """
    c = os.path.abspath(str(candidate))
    r = os.path.abspath(str(root))
    return c == r or c.startswith(r + os.sep)


def _resolve_includes(
    path: Path,
    *,
    strip_common: bool = False,
    root: Path | None = None,
    _stack: tuple[str, ...] = (),
) -> str:
    """Recursively resolve #include directives.

    If `strip_common` is True, lines that #include common_policy.dl
    are dropped rather than expanded — the caller is responsible for
    prepending it once, matching the Rust compiler's behavior.

    Includes are confined to `root`, which defaults to the directory of the
    top-level file. This matters because policy text is attacker-supplied on
    the `SetPolicy` and `ValidatePolicy` endpoints and the desugared result is
    returned to the caller: without confinement, ``#include "/etc/passwd"``
    resolves outright (``Path("/a") / "/etc/passwd"`` is ``/etc/passwd``) and
    ``../`` walks anywhere. The server also runs this under bubblewrap, so
    this is the second of two barriers rather than the only one.
    """
    if root is None:
        root = path.parent

    here = os.path.abspath(str(path))
    if here in _stack:
        chain = " -> ".join([*_stack, here])
        raise IncludeError(f"#include cycle: {chain}")
    if len(_stack) >= _MAX_INCLUDE_DEPTH:
        raise IncludeError(
            f"#include nested more than {_MAX_INCLUDE_DEPTH} deep at {path}"
        )

    if not path.exists():
        return ""
    lines = []
    source = path.read_text()
    if _stack:
        # Only the entry may carry the boundary inserted by the Rust compiler.
        # Included files are authored text, even when the entry already inlined
        # common_policy and no new boundary will be added by preprocess_file.
        source = _clear_authored_gate_markers(source, clear_user_boundary=True)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith('#include'):
            m = re.search(r'"([^"]+)"', stripped)
            if m:
                inc_rel = m.group(1)
                if strip_common and inc_rel.endswith("common_policy.dl"):
                    continue
                if os.path.isabs(inc_rel):
                    raise IncludeError(
                        f"absolute #include is not allowed: {inc_rel!r} in {path}"
                    )
                inc_path = path.parent / inc_rel
                if not _within(inc_path, root):
                    raise IncludeError(
                        f"#include escapes {root}: {inc_rel!r} in {path}"
                    )
                if not inc_path.exists():
                    raise IncludeError(
                        f"#include not found: {inc_rel!r} in {path}"
                    )
                lines.append(
                    _resolve_includes(
                        inc_path,
                        strip_common=strip_common,
                        root=root,
                        _stack=(*_stack, here),
                    )
                )
            continue
        lines.append(line)
    return '\n'.join(lines)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Soufflé syntactic sugar preprocessor (dot notation)')
    parser.add_argument('input', help='Input .dl file')
    parser.add_argument('--check', action='store_true',
                        help='Validate only, no output')
    parser.add_argument('--resolve-includes', action='store_true',
                        help='Resolve #include directives')
    parser.add_argument('--include-root', metavar='DIR',
                        help='Directory #include may read from '
                             '(default: the input file\'s own directory). '
                             'Widen this only for local authoring; the server '
                             'relies on the default to confine policy text it '
                             'was handed.')
    args = parser.parse_args()

    if args.resolve_includes:
        try:
            source = preprocess_file(args.input, include_root=args.include_root)
        except IncludeError as exc:
            # The server maps a non-zero exit to a compile error and shows
            # stderr to the caller, so a message beats a traceback.
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        source = Path(args.input).read_text()
        source = preprocess(source)

    if args.check:
        # Just verify it parses without errors
        print(f"OK: processed {len(source.splitlines())} lines", file=sys.stderr)
    else:
        print(source)
    return 0


if __name__ == '__main__':
    sys.exit(main())
