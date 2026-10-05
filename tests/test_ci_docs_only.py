"""CI skips engine lanes only for pull requests that change documentation alone."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify(paths: list[str]) -> int:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci_docs_only.py")],
        input="\n".join(paths), capture_output=True, text=True, timeout=30,
    ).returncode


def test_documentation_changes_take_the_fast_path():
    assert classify([
        "README.md", "CONTRIBUTING.md", "docs-site/src/content/docs/index.mdx",
        "docs-site/astro.config.mjs", "docs/engine-rehearsal.md",
        "examples/message-flow/README.md", "plugins/policy-compiler/skills/write-policy/SKILL.md",
    ]) == 0


def test_any_code_or_package_file_requires_the_full_gate():
    for path in [
        "crates/sasy-binary/src/main.rs", "sdk/python/sasy/cli.py", "sdk/python/README.md",
        "tests/test_engine_cli.py", "Dockerfile", "Makefile", "Cargo.lock",
        ".github/workflows/ci.yml", "examples/message-flow/demo.py", "souffle/sugar.py",
    ]:
        assert classify(["README.md", path]) == 1, path


def test_an_empty_or_unknown_list_fails_closed():
    assert classify([]) == 1
    assert classify(["something-new.txt"]) == 1
