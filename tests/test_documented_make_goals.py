"""Every `make <goal>` a document shows names a goal the Makefile defines.

A quick start whose command has no target is a reader's first impression of the
project, and `make: *** No rule to make target` is where it ends. The scan
covers the fenced shell blocks of every tracked Markdown and MDX file.

What counts as defined is the union of the goals every tracked Makefile in the
repository declares, not only the root one: a document that describes an example
directory shows that directory's goals, and this test is not the place to decide
which Makefile each document means. What it refuses is a goal no Makefile here
defines at all. A block that changes directory first is skipped for the same
reason — the goal after a `cd` belongs somewhere else entirely.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^\s*(```|~~~)")
# `target: prereqs`, one or more names, but not `VAR := value` or `VAR ?= value`.
RULE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._/%-]*(?:[ \t]+[A-Za-z0-9][A-Za-z0-9._/%-]*)*)[ \t]*:(?!=)")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _defined_goals() -> set[str]:
    listed = subprocess.run(
        [shutil.which("git") or "git", "ls-files", "Makefile", "*/Makefile"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    goals: set[str] = set()
    for rel in listed:
        for line in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines():
            if line.startswith("\t") or line.lstrip().startswith("#"):
                continue
            if line.startswith(".PHONY:"):
                goals.update(
                    w for w in line.split(":", 1)[1].split() if not w.startswith("\\")
                )
                continue
            found = RULE.match(line)
            if not found:
                continue
            names = found.group(1).split()
            if names[0].startswith("."):
                continue
            goals.update(names)
    return goals


def _documented_goals() -> list[tuple[str, int, str]]:
    """(file, line, goal) for every `make <goal>` in a fenced block."""
    listed = subprocess.run(
        [shutil.which("git") or "git", "ls-files", "*.md", "*.mdx"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()

    out: list[tuple[str, int, str]] = []
    for rel in listed:
        in_block = False
        block_changed_directory = False
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for n, raw in enumerate(text.splitlines(), 1):
            if FENCE.match(raw):
                in_block = not in_block
                block_changed_directory = False
                continue
            if not in_block:
                continue
            line = raw.strip().lstrip("$").strip()
            segments = re.split(r"&&|\|\||;", line)
            for segment in segments:
                words = segment.split()
                if words and words[0] == "cd":
                    block_changed_directory = True
                    continue
                while words and ASSIGNMENT.match(words[0]):
                    words.pop(0)
                if not words or words[0] != "make" or block_changed_directory:
                    continue
                for word in words[1:]:
                    if word.startswith(("-", "\\")) or "=" in word:
                        continue
                    out.append((rel, n, word))
                    break
    return out


def test_every_documented_make_goal_exists() -> None:
    goals = _defined_goals()
    assert {"serve", "install", "build-rust", "docs"} <= goals, sorted(goals)
    offenders = [
        f"{rel}:{n}: make {goal}"
        for rel, n, goal in _documented_goals()
        if goal not in goals
    ]
    assert not offenders, (
        "a documented command names a target the Makefile does not define:\n"
        + "\n".join(offenders)
    )
