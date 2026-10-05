"""No example or test picks the project's own servers for the caller.

The SDK has no built-in endpoint: `SASY_URL`, `SASY_WRITE_URL` and
`SASY_TRANSLATE_URL` are supplied by whoever runs the code. That is only
worth anything if the layer above does not quietly restore a default, so
this test reads every Python file under `examples/` and `tests/` and
fails on a fallback that names a host the project operates (`fly.dev`).

What the guard covers, exactly: a single call whose method name is
``get``, ``setdefault`` or ``getenv``, whose first argument is one of the
endpoint variables written as a literal, and whose default is a string
literal given either as the second positional argument or as
``default=``. So ``os.environ.get("SASY_URL", "sasy.fly.dev:443")``,
``os.environ.setdefault(...)`` and ``os.getenv("SASY_URL",
default="sasy.fly.dev:443")`` are all caught.

What it does not cover, so nobody reads more into a pass than is there:
files outside `examples/` and `tests/`, files that are not Python, a
variable name built at run time, and a fallback assembled across more
than one statement (``url = os.environ.get("SASY_URL")`` followed by
``if not url: url = "sasy.fly.dev:443"``).

Public examples and tests have no hosted-service exceptions.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("examples", "tests")

ENDPOINT_VARS = {"SASY_URL", "SASY_WRITE_URL", "SASY_TRANSLATE_URL"}
PROJECT_HOST = "fly.dev"

ALLOWED_STEMS: set[str] = set()


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts or "node_modules" in path.parts:
                continue
            files.append(path)
    return files


LOOKUP_METHODS = {"get", "setdefault", "getenv"}


def _fallbacks(tree: ast.AST) -> list[tuple[int, str, str]]:
    """(line, variable, fallback) for every env lookup with a default."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in LOOKUP_METHODS:
            continue
        if not node.args:
            continue
        name = node.args[0]
        if not (isinstance(name, ast.Constant) and name.value in ENDPOINT_VARS):
            continue
        # The default is the second positional argument, or `default=`, which
        # `os.getenv` accepts as a keyword.
        default = node.args[1] if len(node.args) > 1 else None
        if default is None:
            for kw in node.keywords:
                if kw.arg == "default":
                    default = kw.value
                    break
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            found.append((node.lineno, name.value, default.value))
    return found


def test_no_example_or_test_falls_back_to_a_project_host():
    offenders = []
    for path in _python_files():
        if path.stem in ALLOWED_STEMS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, var, fallback in _fallbacks(tree):
            if PROJECT_HOST in fallback:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{line}: {var} falls back to {fallback!r}")
    assert not offenders, "\n".join(offenders)


def test_the_guard_catches_every_shape_its_docstring_claims():
    """Each shape named above is really matched, so a pass means something."""
    source = "\n".join(
        (
            'a = os.environ.get("SASY_URL", "sasy.fly.dev:443")',
            'b = os.environ.setdefault("SASY_WRITE_URL", "write.fly.dev:443")',
            'c = os.getenv("SASY_TRANSLATE_URL", "tr.fly.dev:443")',
            'd = os.getenv("SASY_URL", default="kw.fly.dev:443")',
        )
    )
    caught = {fallback for _, _, fallback in _fallbacks(ast.parse(source))}
    assert caught == {
        "sasy.fly.dev:443",
        "write.fly.dev:443",
        "tr.fly.dev:443",
        "kw.fly.dev:443",
    }, f"the guard missed a shape its docstring claims: {sorted(caught)}"


def test_the_guard_ignores_a_lookup_with_no_default():
    """A bare lookup is the correct pattern and must not be reported."""
    source = 'url = os.environ.get("SASY_URL")\nalso = os.getenv("SASY_URL")'
    assert _fallbacks(ast.parse(source)) == []


def test_the_allowlist_still_describes_real_files():
    stems = {p.stem for p in _python_files()}
    missing = sorted(ALLOWED_STEMS - stems)
    assert not missing, f"allowlisted files no longer exist: {missing}"
