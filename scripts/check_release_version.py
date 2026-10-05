"""Refuse a release tag that disagrees with any version recorded in the tree.

The SDK's `sasy engine start` pulls `ghcr.io/sasy-labs/sasy:<SDK version>`, so
the image tag, the Python SDK, the engine crates and the documented image must
all name the same version. Prints that version on success.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

IMAGE = "ghcr.io/sasy-labs/sasy:"


def recorded_versions(root: Path) -> dict[str, set[str]]:
    """Return each place a release version is recorded, with what it says.

    Args:
        root: Repository root.

    Returns:
        A map from a location description to the versions found there.
    """
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    python = tomllib.loads((root / "sdk/python/pyproject.toml").read_text())
    typescript = json.loads((root / "sdk/typescript/package.json").read_text())
    nix = (root / "nix/package.nix").read_text()
    internal = {
        dep["version"]
        for dep in cargo["workspace"]["dependencies"].values()
        if isinstance(dep, dict) and "path" in dep and "version" in dep
    }
    lock = tomllib.loads((root / "Cargo.lock").read_text())
    # Packages without a registry source are this workspace's own crates.
    local = {pkg["version"] for pkg in lock["package"] if "source" not in pkg}
    found = {
        "Cargo.toml [workspace.package]": {cargo["workspace"]["package"]["version"]},
        "Cargo.toml internal crate versions": internal,
        "Cargo.lock workspace crates": local,
        "sdk/python/pyproject.toml": {python["project"]["version"]},
        "sdk/typescript/package.json": {typescript["version"]},
        "nix/package.nix": set(re.findall(r'version = "([^"]+)";', nix)),
    }
    image_tag = re.compile(re.escape(IMAGE) + r"([0-9][^\s`'\"]*)")
    for path in ["Makefile", "docs-site/src/content/docs/local-engine.mdx"]:
        found[path] = set(image_tag.findall((root / path).read_text()))
    return found


def main(argv: list[str] | None = None) -> int:
    """Check a tag such as sasy-v0.5.1 against the tree.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, such as sasy-v0.5.1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    match = re.fullmatch(r"sasy-v(\d+\.\d+\.\d+)", args.tag)
    if not match:
        print(f"Release tags look like sasy-v1.2.3, not {args.tag!r}", file=sys.stderr)
        return 1
    version = match[1]
    wrong = {
        where: found
        for where, found in recorded_versions(args.root).items()
        if found != {version}
    }
    for where, found in wrong.items():
        shown = ", ".join(sorted(found)) or "no version"
        print(f"{where}: {shown} (tag says {version})", file=sys.stderr)
    if wrong:
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
