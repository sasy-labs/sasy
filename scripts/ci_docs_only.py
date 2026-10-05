"""Exit 0 when every changed path is documentation, so CI may skip engine lanes.

Reads changed paths, one per line, on stdin. Anything not on the
documentation allowlist, and an empty or unreadable list, means "not docs
only" (exit 1), so the full gate runs whenever there is any doubt.
"""

import sys
from fnmatch import fnmatchcase

# Documentation that no engine, SDK or packaging step reads. Package READMEs
# (sdk/**) are excluded on purpose: they are part of a published package.
DOCS = (
    "docs-site/*",
    "docs/*",
    "examples/*.md",
    "plugins/*.md",
)


def is_docs(path: str) -> bool:
    """Return whether a changed path is documentation only.

    Args:
        path: Repository-relative path, as Git reports it.

    Returns:
        True for top-level Markdown files and paths matching ``DOCS``.
    """
    if "/" not in path and path.endswith(".md"):
        return True
    return any(fnmatchcase(path, pattern) for pattern in DOCS)


def main() -> int:
    """Classify the paths on stdin.

    Returns:
        0 when the list is non-empty and every path is documentation, else 1.
    """
    paths = [line.strip() for line in sys.stdin if line.strip()]
    docs_only = bool(paths) and all(is_docs(path) for path in paths)
    print("docs only" if docs_only else "engine lanes required")
    return 0 if docs_only else 1


if __name__ == "__main__":
    sys.exit(main())
