"""The release workflow's tag check agrees with the tree and catches drift."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_release_version.py"
FILES = [
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    "nix/package.nix",
    "sdk/python/pyproject.toml",
    "sdk/typescript/package.json",
    "docs-site/src/content/docs/local-engine.mdx",
]


def check(root: Path, tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), tag, "--root", str(root)],
        capture_output=True, text=True, timeout=30,
    )


def current_version() -> str:
    return tomllib.loads((ROOT / "sdk/python/pyproject.toml").read_text())["project"]["version"]


def copy_tree(tmp_path: Path) -> Path:
    for name in FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, tmp_path / name)
    return tmp_path


def test_the_tree_matches_its_own_version_tag():
    result = check(ROOT, f"sasy-v{current_version()}")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == current_version()


def test_a_different_tag_is_refused_naming_every_location():
    result = check(ROOT, "sasy-v99.0.0")
    assert result.returncode == 1
    for name in FILES:
        assert name in result.stderr


def test_one_stale_location_is_refused(tmp_path):
    root = copy_tree(tmp_path)
    makefile = root / "Makefile"
    makefile.write_text(makefile.read_text().replace(f"sasy:{current_version()}", "sasy:0.0.1"))
    result = check(root, f"sasy-v{current_version()}")
    assert result.returncode == 1
    assert "Makefile: 0.0.1" in result.stderr


def test_malformed_tags_are_refused():
    for tag in ["0.5.0", "v0.5.0", "sasy-v0.5", "sasy-v0.5.0-rc1", "release"]:
        assert check(ROOT, tag).returncode == 1
