#!/usr/bin/env python3
"""Stage only explicit public build inputs before Nix imports a flake source."""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
PACKAGING_FILES = [
    "flake.nix", "flake.lock", "nix/source-files.json", "nix/toolchain.nix",
    "nix/package.nix", "nix/docker-root.nix", "nix/stage-source.py", "nix/build.sh",
]


def stage(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    source_files = json.loads((ROOT / "nix/source-files.json").read_text())
    for relative in source_files + PACKAGING_FILES:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Build input must be relative to the repository")
        parent = ROOT
        for part in path.parts[:-1]:
            parent /= part
            if parent.is_symlink():
                raise ValueError(f"Symlinked build input directory: {relative}")
        source = ROOT / path
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            raise ValueError(f"Unexpected build input symlink: {relative}")
        elif source.is_file() and source.resolve().is_relative_to(ROOT):
            shutil.copy2(source, target)
        else:
            raise ValueError(f"Build input is not a regular repository file: {relative}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: stage-source.py NEW_OUTPUT_DIRECTORY")
    stage(Path(sys.argv[1]))
