"""Shared terminal narration for the scripted SASY examples.

A narrated run prints a header per step, a few sentences on what is about to
happen, one line per agent event, and waits for Enter between steps. Pauses
happen only in an interactive terminal, and colors follow NO_COLOR.
"""

import os
import re
import shutil
import sys
import textwrap
import time
from contextlib import ExitStack, contextmanager
from itertools import cycle
from threading import Event, Thread


@contextmanager
def compiling_policy(enabled=True, output=None):
    """Show progress on a terminal while a session binds/compiles its policy."""
    output = output if output is not None else sys.__stderr__
    if not enabled or output is None or not output.isatty():
        yield
        return
    label = "Compiling policy…"
    stop = Event()

    def animate():
        for frame in cycle("|/-\\"):
            if stop.wait(0.1):
                return
            output.write(f"\r  {frame} {label}")
            output.flush()

    output.write(f"  | {label}")
    output.flush()
    worker = Thread(target=animate, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        output.write("\r" + " " * (len(label) + 4) + "\r")
        output.flush()


@contextmanager
def policy_session(factory, *, progress=True, **settings):
    """Stop the spinner before agent events and preserve session cleanup."""
    with ExitStack() as stack:
        with compiling_policy(enabled=progress):
            session = stack.enter_context(factory(**settings))
        yield session


# Datalog tokens: comment, string, built-in, rule arrow, predicate, directive.
DATALOG = re.compile(r'(?P<comment>//.*)|(?P<string>"[^"]*")|(?P<builtin>[@$]\w+)'
                     r'|(?P<arrow>:-)|(?P<pred>\b[A-Z]\w*(?=\())|(?P<decl>\.decl\b)')


def highlight_datalog(line, style):
    """Color one line of Datalog for a terminal; rule heads in bold."""
    if line.lstrip().startswith("//"):
        note = re.match(r"(\s*//\s*)(@\w+:)(.*)", line)
        if note:
            return style("2", note[1]) + style("33", note[2]) + style("2", note[3])
        return style("2", line)
    codes = {"string": "32", "builtin": "33", "arrow": "35", "decl": "35", "pred": "34"}

    def paint(match):
        kind = match.lastgroup
        if kind == "comment":
            return style("2", match[0])
        code = "1;34" if kind == "pred" and match.start() == 0 else codes[kind]
        return style(code, match[0])

    return DATALOG.sub(paint, line)


class Narrator:
    """Stage headers, short explanations and pauses for a terminal walkthrough."""

    def __init__(self, pause=True):
        self.color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        self.pausing = pause and sys.stdin.isatty() and sys.stdout.isatty()
        self.width = min(shutil.get_terminal_size((80, 24)).columns, 80)
        self.after_events = False  # event lines end without a blank line
        self.event_delay = 0.05 if sys.stdout.isatty() else 0

    def style(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def title(self, heading, text):
        print(self.style("1", heading) + "\n")
        self.say(text)

    def stage(self, number, total, heading):
        rule = "─" * self.width
        header = self.style("1;36", f" STEP {number} OF {total} · {heading}")
        print(f"\n{self.style('2', rule)}\n{header}\n{self.style('2', rule)}\n")

    def say(self, text):
        lead = "\n" if self.after_events else ""
        self.after_events = False
        print(lead + textwrap.fill(text, self.width, initial_indent="  ", subsequent_indent="  ") + "\n")

    def bullets(self, items):
        for i, item in enumerate(items, 1):
            print(textwrap.fill(item, self.width, initial_indent=f"  {i}. ", subsequent_indent="     "))
        print()

    def pace(self):
        """Space scripted terminal events; redirected output stays immediate."""
        if self.event_delay:
            time.sleep(self.event_delay)

    def line(self, who, text, code="0"):
        self.pace()
        self.after_events = True
        print(f"  {self.style('36', f'{who:<6}')}  {self.style(code, text)}", flush=True)

    def quote(self, text):
        for part in text.splitlines():
            print(self.style("2", textwrap.fill(part, self.width, initial_indent="          │ ",
                                                 subsequent_indent="          │ ")))

    def block(self, text):
        for part in text.rstrip().splitlines():
            print("    " + highlight_datalog(part, self.style))
        print()

    def pause(self, next_step):
        if not self.pausing:
            return
        try:
            input(self.style("33", f"  ▸ Press Enter to continue: {next_step} "))
        except EOFError:
            self.pausing = False
        except KeyboardInterrupt:
            print()
            raise SystemExit(130) from None
