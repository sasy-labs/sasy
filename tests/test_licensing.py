"""The repository ships under Apache-2.0, and every published artifact says so.

These checks are file-level: the licence text is present at the repository root
and inside each package directory that gets published (the Python wheel and the
npm tarball), and each package manifest declares the SPDX identifier
``Apache-2.0``. A copy, not a symlink, in the package directories — build
backends resolve `license-files` / `files` relative to the package root and a
symlink out of the package would not survive an sdist.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# sha256 of the canonical Apache License 2.0 text from
# https://www.apache.org/licenses/LICENSE-2.0.txt
APACHE_2_0_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

LICENSE_COPIES = [
    REPO_ROOT / "LICENSE",
    REPO_ROOT / "sdk" / "python" / "LICENSE",
    REPO_ROOT / "sdk" / "typescript" / "LICENSE",
]


@pytest.mark.parametrize("license_path", LICENSE_COPIES, ids=lambda p: str(p.name))
def test_license_file_is_the_canonical_apache_2_0_text(license_path: Path) -> None:
    import hashlib

    assert license_path.is_file(), f"{license_path} is missing"
    assert not license_path.is_symlink(), f"{license_path} must be a copy, not a symlink"
    digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    assert digest == APACHE_2_0_SHA256, (
        f"{license_path} is not the canonical Apache-2.0 text (sha256 {digest})"
    )


def test_rust_workspace_declares_apache_2_0() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "." / "Cargo.toml").read_text(encoding="utf-8")
    )
    assert manifest["workspace"]["package"]["license"] == "Apache-2.0"


def test_every_rust_crate_inherits_the_workspace_licence() -> None:
    crates = sorted((REPO_ROOT / "." / "crates").glob("*/Cargo.toml"))
    assert crates, "no crate manifests found"
    for crate in crates:
        manifest = tomllib.loads(crate.read_text(encoding="utf-8"))
        # `license.workspace = true` parses as the nested table {"workspace": True}.
        declared = manifest["package"].get("license")
        assert declared in ({"workspace": True}, "Apache-2.0"), (
            f"{crate} does not carry the workspace licence (license = {declared!r})"
        )
        for name in ("LICENSE", "NOTICE"):
            packaged = crate.parent / name
            assert packaged.is_file() and not packaged.is_symlink(), packaged
            assert packaged.read_bytes() == (REPO_ROOT / name).read_bytes(), packaged


def test_python_package_declares_apache_2_0_and_ships_the_file() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "sdk" / "python" / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = manifest["project"]
    assert project["license"] == "Apache-2.0"
    assert "LICENSE" in project["license-files"]
    assert any(
        c == "License :: OSI Approved :: Apache Software License"
        for c in project.get("classifiers", [])
    ), "Trove classifier missing"


def test_typescript_package_declares_apache_2_0_and_ships_the_file() -> None:
    manifest = json.loads(
        (REPO_ROOT / "sdk" / "typescript" / "package.json").read_text(encoding="utf-8")
    )
    assert manifest["license"] == "Apache-2.0"
    assert "LICENSE" in manifest["files"], "npm pack would not include LICENSE"
