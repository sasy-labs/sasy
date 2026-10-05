"""``python -m sasy`` — ask the SDK what it would do, without running anything.

Today it answers one question: ``--print-url`` prints the endpoint this SDK
resolves from the process environment and ``.env``. A caller that must decide
whether a run is local or remote asks here rather than reading ``SASY_URL``
itself, so the address it classifies is the address the SDK will dial.

It lives in its own module so ``python -m sasy`` does not re-execute a module
the package has already imported, which is what makes the interpreter warn.
"""

from __future__ import annotations

import sys

from sasy.config import _print_url

if __name__ == "__main__":
    sys.exit(_print_url(sys.argv[1:]))
