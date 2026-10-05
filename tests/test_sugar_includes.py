"""Confinement of `#include` resolution in the Soufflé preprocessor.

`sugar.py --resolve-includes` runs on policy text supplied by whoever called
`SetPolicy` or `ValidatePolicy`, and `ValidatePolicy` hands the desugared
result back to that caller. An `#include` that escapes the policy's own
directory would therefore read a file off the server and return it.

The server also runs the preprocessor under bubblewrap
(`compile_souffle_with_assets` in `crates/sasy-policy/src/compiler.rs`), so
these checks are the second of two barriers -- and the only one on a host with
no bubblewrap, which is the case for local development on macOS.

Run with:
    pytest tests/test_sugar_includes.py -q
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUGAR = REPO / "." / "souffle" / "sugar.py"


def _load():
    """Import sugar.py by path; it is a script beside the Souffle assets.

    The module has to be registered in ``sys.modules`` before it executes:
    ``@dataclass`` resolves its field types through
    ``sys.modules[cls.__module__]``, which is not yet populated during
    ``exec_module``.
    """
    spec = importlib.util.spec_from_file_location("sugar_under_test", SUGAR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sugar = _load()


def _policy(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_absolute_include_is_rejected(tmp_path):
    """`Path("/a") / "/etc/passwd"` is `/etc/passwd`, so this escapes outright."""
    entry = _policy(tmp_path, "policy.dl", '#include "/etc/passwd"\n.decl X(a:number)\n')
    with pytest.raises(sugar.IncludeError, match="absolute"):
        sugar.preprocess_file(str(entry))


def test_parent_traversal_is_rejected(tmp_path):
    entry = _policy(tmp_path, "policy.dl",
                    '#include "../../../../etc/passwd"\n.decl X(a:number)\n')
    with pytest.raises(sugar.IncludeError, match="escapes"):
        sugar.preprocess_file(str(entry))


def test_include_cycle_is_rejected(tmp_path):
    """Two files including each other must not recurse forever."""
    _policy(tmp_path, "a.dl", '#include "b.dl"\n.decl A(x:number)\n')
    entry = _policy(tmp_path, "b.dl", '#include "a.dl"\n.decl B(x:number)\n')
    with pytest.raises(sugar.IncludeError, match="cycle"):
        sugar.preprocess_file(str(entry))


def test_self_include_is_rejected(tmp_path):
    entry = _policy(tmp_path, "policy.dl",
                    '#include "policy.dl"\n.decl X(a:number)\n')
    with pytest.raises(sugar.IncludeError, match="cycle"):
        sugar.preprocess_file(str(entry))


def test_missing_include_is_an_error_not_an_empty_expansion(tmp_path):
    """Silently expanding to nothing would drop rules from a security policy."""
    entry = _policy(tmp_path, "policy.dl",
                    '#include "absent.dl"\n.decl X(a:number)\n')
    with pytest.raises(sugar.IncludeError, match="not found"):
        sugar.preprocess_file(str(entry))


def test_nested_include_inside_the_tree_still_works(tmp_path):
    """Confinement must not break the legitimate case."""
    _policy(tmp_path, "sub/helper.dl", ".decl Helper(x:number)\n")
    entry = _policy(tmp_path, "policy.dl",
                    '#include "sub/helper.dl"\n.decl X(a:number)\n')
    out = sugar.preprocess_file(str(entry))
    assert ".decl Helper(x:number)" in out
    assert ".decl X(a:number)" in out


def test_repo_policies_still_preprocess():
    """Every policy checked into the repository must survive the confinement."""
    policies = sorted(
        p for d in ("policies", "examples", "plugins")
        for p in (REPO / d).rglob("*.dl")
    )
    assert policies, "expected to find policies to check"
    failures = []
    for p in policies:
        try:
            sugar.preprocess_file(str(p))
        except sugar.IncludeError as exc:
            failures.append(f"{p.relative_to(REPO)}: {exc}")
    assert not failures, "policies broken by include confinement:\n" + "\n".join(failures)


def test_include_root_widens_confinement_deliberately(tmp_path):
    """Local authoring can opt out; the server never passes this flag."""
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "helpers.dl").write_text(".decl Helper(x:number)\n")
    entry = _policy(tmp_path, "policy/p.dl",
                    '#include "../shared/helpers.dl"\n.decl X(a:number)\n')

    with pytest.raises(sugar.IncludeError, match="escapes"):
        sugar.preprocess_file(str(entry))

    out = sugar.preprocess_file(str(entry), include_root=str(tmp_path))
    assert ".decl Helper(x:number)" in out


def test_absolute_include_rejected_even_with_a_widened_root(tmp_path):
    """Widening the root is not a way to reach outside the filesystem tree."""
    entry = _policy(tmp_path, "policy.dl", '#include "/etc/passwd"\n')
    with pytest.raises(sugar.IncludeError, match="absolute"):
        sugar.preprocess_file(str(entry), include_root=str(tmp_path))


@pytest.mark.parametrize("body,expected", [
    ('#include "/etc/passwd"\n', 2),
    ('#include "../../../etc/passwd"\n', 2),
    ('.decl X(a:number)\n', 0),
])
def test_cli_exit_code(tmp_path, body, expected):
    """The server reads the exit code, so a rejection must be non-zero."""
    entry = _policy(tmp_path, "policy.dl", body)
    proc = subprocess.run(
        [sys.executable, str(SUGAR), "--resolve-includes", str(entry)],
        capture_output=True, text=True,
    )
    assert proc.returncode == expected, proc.stderr
    if expected:
        assert "error:" in proc.stderr
        assert "Traceback" not in proc.stderr


def test_gate_provenance_is_neutralized_only_in_comments():
    source = (
        '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\n'
        '/* authored block\n'
        '// SASY_AUTO_GATE_DEFAULT: ReachableFromPolicyRelevant\n'
        '*/\n'
        'R("SASY_AUTO_GATE_DEFAULT: literal"). '
        '// SASY_AUTO_GATE_DEFAULT: inline\n'
        'R("escaped \\" // SASY_AUTO_GATE_DEFAULT: still a string").\n'
    )
    expected = (
        '// SASY_AUTHORED_GATE_MARKER: CurrentDependsPolicyRelevant\n'
        '/* authored block\n'
        '// SASY_AUTHORED_GATE_MARKER: ReachableFromPolicyRelevant\n'
        '*/\n'
        'R("SASY_AUTO_GATE_DEFAULT: literal"). '
        '// SASY_AUTHORED_GATE_MARKER: inline\n'
        'R("escaped \\" // SASY_AUTO_GATE_DEFAULT: still a string").\n'
    )
    assert sugar._clear_authored_gate_markers(source) == expected


def test_preprocess_only_marks_its_own_fresh_gate_default():
    source = (
        '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\n'
        '/*\n// SASY_AUTO_GATE_DEFAULT: ReachableFromPolicyRelevant\n*/\n'
        '.decl Result(x:symbol)\n'
        'Result(x) :- CurrentDepends(x).\n'
    )
    output = sugar.preprocess(source)
    lines = output.splitlines()
    marked = [i for i, line in enumerate(lines) if line.startswith('// SASY_AUTO_GATE_DEFAULT:')]
    assert len(marked) == 1
    assert lines[marked[0]] == '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant'
    assert lines[marked[0] + 1] == 'CurrentDependsPolicyRelevant().'
    assert '// SASY_AUTHORED_GATE_MARKER: CurrentDependsPolicyRelevant' in output


def test_included_explicit_gate_keeps_authored_provenance(tmp_path):
    _policy(tmp_path, 'helper.dl',
            '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\n'
            'CurrentDependsPolicyRelevant().\n')
    entry = _policy(tmp_path, 'policy.dl',
                    '#include "helper.dl"\n'
                    '.decl Result(x:symbol)\n'
                    'Result(x) :- CurrentDepends(x).\n')
    output = sugar.preprocess_file(str(entry))
    assert '// SASY_AUTHORED_GATE_MARKER: CurrentDependsPolicyRelevant' in output
    assert '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant' not in output
    assert sum(line.strip() == 'CurrentDependsPolicyRelevant().' for line in output.splitlines()) == 1
@pytest.mark.parametrize('prefix', [
    ['R("// === USER_POLICY_BEGIN ===").'],
    ['// ordinary comment // === USER_POLICY_BEGIN ==='],
    ['/*', '// === USER_POLICY_BEGIN ===', '*/'],
    ['R("multiline', '// === USER_POLICY_BEGIN ===', 'literal").'],
])
def test_only_active_exact_entry_boundary_selects_user_lines(prefix):
    original = ['before'] + prefix + ['after']
    assert sugar._user_policy_lines(original) == original
    with_boundary = original + [sugar.USER_POLICY_BEGIN_MARKER, 'actual-user']
    assert sugar._user_policy_lines(with_boundary) == ['actual-user']


def test_included_boundary_cannot_suppress_preceding_graph_consumers(tmp_path):
    common = (REPO / 'souffle/common_policy.dl').read_text()
    _policy(tmp_path, 'child.dl', sugar.USER_POLICY_BEGIN_MARKER + '\n.decl Marker(x:symbol)\n')
    entry = _policy(tmp_path, 'root.dl', common + '\nIsAuthorized(i) :- Actions(i,_), CurrentDepends("danger").\n#include "child.dl"\n')
    output = sugar.preprocess_file(str(entry))
    # In the already-inlined common path, its ordinary gate implication remains
    # authoritative; the Reachable default activates both helpers as before.
    assert '\nReachableFromPolicyRelevant().' in output
    assert sugar.USER_POLICY_BEGIN_MARKER not in output
    assert '// === AUTHORED_POLICY_BOUNDARY ===' in output


def test_include_cleanup_preserves_entry_boundary_and_quoted_data(tmp_path):
    child = _policy(tmp_path, 'nested/child.dl',
        '/*\n' + sugar.USER_POLICY_BEGIN_MARKER + '\n*/\n'
        'R("// === USER_POLICY_BEGIN ===").\n'
        '// SASY_AUTO_GATE_DEFAULT: CurrentDependsPolicyRelevant\n')
    _policy(tmp_path, 'middle.dl', '#include "nested/child.dl"\n')
    entry = _policy(tmp_path, 'root.dl', sugar.USER_POLICY_BEGIN_MARKER + '\n#include "middle.dl"\n')
    resolved = sugar._resolve_includes(entry)
    assert resolved.startswith(sugar.USER_POLICY_BEGIN_MARKER + '\n')
    assert 'R("// === USER_POLICY_BEGIN ===").' in resolved
    assert '/*\n// === AUTHORED_POLICY_BOUNDARY ===\n*/' in resolved
    assert 'SASY_AUTHORED_GATE_MARKER:' in resolved
    assert sugar.USER_POLICY_BEGIN_MARKER in child.read_text()  # no source mutation
