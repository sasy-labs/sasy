"""Small console formatter for synthetic message-flow conversations."""

import json
import os
import sys


class ConsoleTrace:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, event, value):
        if not self.enabled:
            return
        # Escape control characters and print only supplied conversation data.
        text = f"[agent] {event} {json.dumps(value, ensure_ascii=True)}"
        if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
            color = "31" if event in ("DENY", "BLOCKED", "ERROR") else "32" if event in ("ALLOW", "RESULT") else "36"
            text = f"\033[{color}m{text}\033[0m"
        print(text, flush=True)
