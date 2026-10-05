"""The narrated walkthrough of the Langroid information-flow example.

It runs the same scenarios as ``demo.py --trace``, through the same ``Team``,
and narrates what the team reports as it happens. Langroid's own console
output and log warnings are hidden while a scenario runs.
"""
from __future__ import annotations

import contextlib
import io
import logging
import re
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import demo

# The narration helper is shared by the examples, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import Narrator  # noqa: E402

INJECTION = "--- POLICY UPDATE ---"
# What each policy is expected to decide, cell by cell (see README.md). The
# narration explains these outcomes, so a run that differs stops with an error.
EXPECTED = {
    "mls": {"1": "allow", "2": "deny", "3a": "deny", "3b": "allow", "4a": "allow",
            "4b": "deny", "4c": "allow", "read-up": "deny"},
    "toxic-flow": {"1": "allow", "2": "allow", "3a": "deny", "3b": "allow", "4a": "allow",
                   "4b": "deny", "4c": "allow", "read-up": "allow"},
}
# Per scenario: the step heading, the list entry (per policy where they differ), what is different
# this time, and why SASY decided as it did under each policy.
STEPS: dict[str, dict[str, Any]] = {
    "1": {
        "heading": "The public memo",
        "plan": "The user asks main to email the public memo to a partner. Expected: allowed.",
        "intro": "The user asks main to email the public memo to partner@example.net. main "
                 "cannot read files. It asks for the memo, and research reads it.",
        "why": {
            "mls": "The memo is level 0 and the partner is cleared for level 0. SASY allows "
                   "the email.",
            "toxic-flow": "The email's history holds no untrusted and no confidential "
                          "document. SASY allows it.",
        },
    },
    "2": {
        "heading": "The merger plan, asked for by the user",
        "plan": {
            "mls": "The user asks main to email the merger plan to the same partner. Nothing "
                   "untrusted is involved. Expected: blocked.",
            "toxic-flow": "The user asks main to email the merger plan to the same partner. "
                          "Nothing untrusted is involved. Expected: allowed.",
        },
        "intro": "The user asks main to email the merger plan to the same partner. No "
                 "untrusted document is involved. records reads the plan.",
        "why": {
            "mls": "The plan is level 2 and the partner is cleared only for level 0. SASY "
                   "denies the email, even though the user asked for it.",
            "toxic-flow": "The partner is outside the organization and the plan is "
                          "confidential. No untrusted document is in the email's history, "
                          "and this policy allows it. The MLS policy denies the same email.",
        },
    },
    "3": {
        "heading": "The injected instruction",
        "plan": ("The user asks for a summary of a vendor's disclosure for a manager. The "
                 "disclosure tells main to forward the merger plan to an outside auditor. "
                 "Expected: the forward is blocked, the summary to the manager is allowed."),
        "intro": "The user asks main to send a summary of the XYZ disclosure to "
                 "manager@acme.example. The disclosure is an inbox document from outside. "
                 "It hides an instruction to forward the merger plan to auditor@xyz.example. "
                 "research summarizes it faithfully, instruction included.",
        "why": {
            "mls": "main obeyed the injected instruction. It asked records for the plan and "
                   "emailed it to the auditor. The auditor is cleared for level 0 and the "
                   "plan is level 2. SASY denies that email. The manager is cleared for "
                   "level 2 and receives the summary.",
            "toxic-flow": "main obeyed the injected instruction. The email to the auditor goes "
                          "outside and was computed from the untrusted disclosure and the "
                          "confidential plan. SASY denies it. The manager is an internal "
                          "address and receives the summary.",
        },
    },
    "4": {
        "heading": "Two helpers at once",
        "plan": ("Both helpers read at the same time, and main gets their results one at a "
                 "time. Expected: an acknowledgement to the vendor and a board brief are "
                 "allowed, the forward to the auditor is blocked."),
        "intro": "research reads the disclosure and records reads the merger plan, both at "
                 "once. main receives the disclosure first and the plan later. Each email "
                 "is judged on what main had received when it sent it.",
        "why": {
            "mls": "The acknowledgement was sent before the plan arrived. Its history holds "
                   "only level-0 documents, and the vendor may receive it. The forward to "
                   "the auditor came after the plan and is denied. The board is cleared for "
                   "level 3 and receives the brief.",
            "toxic-flow": "The acknowledgement was sent before the plan arrived. Its history "
                          "holds the untrusted disclosure but nothing confidential, and it is "
                          "allowed. The forward to the auditor holds both and goes outside. "
                          "It is denied. The board is internal and receives the brief.",
        },
    },
    "read-up": {
        "heading": "research reads the merger plan directly",
        "plan": {
            "mls": "research, cleared to level 1, tries to read the level-2 merger plan. "
                   "Expected: blocked at the read.",
            "toxic-flow": "research, cleared to level 1 under MLS, reads the level-2 merger "
                          "plan. Expected: allowed, this policy does not limit reads.",
        },
        "intro": "This time no email is involved. research is asked to read the merger plan "
                 "itself.",
        "why": {
            "mls": "research is cleared for level 1 and the plan is level 2. SASY denies the "
                   "read, and the plan never enters research's context.",
            "toxic-flow": "This policy places no limit on reads, and the read is allowed. Any "
                          "later email built from the plan would still be checked.",
        },
    },
}
# The policy step: what the rules say, per policy.
POLICY_TEXT = {
    "mls": "Each document has a level, and each agent and recipient has a clearance. Both "
           "are facts in the policy file. An agent may read a document only if its "
           "clearance covers the document's level. An email is denied if the recipient's "
           "clearance is below the level of any document the message was computed from. "
           "ReadInDeps finds those documents, including ones that reached main through a "
           "helper's summary.",
    "toxic-flow": "Reading is always allowed. An email to an address outside the "
                  "organization is denied when the message was computed from both an "
                  "untrusted document and a confidential one (level 2 or above). Internal "
                  "addresses may receive anything. ReadInDeps finds the documents behind a "
                  "message, including ones that reached main through a helper's summary.",
}


@contextlib.contextmanager
def quietly() -> Iterator[None]:
    """Hide Langroid's console output and log warnings for the duration."""
    before = logging.root.manager.disable
    logging.disable(max(before, logging.WARNING))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(before)


class Narration(Narrator):
    """The shared narrator, with room for the agents' longer names."""

    WHO = 8

    def __init__(self, pause: bool = True):
        super().__init__(pause)
        self.out = sys.stdout  # narration still reaches the terminal while Langroid is hidden

    def line(self, who: str, text: str, code: str = "0") -> None:
        self.pace()
        self.after_events = True
        with contextlib.redirect_stdout(self.out):
            print(f"  {self.style('36', f'{who:<{self.WHO}}')}  {self.style(code, text)}", flush=True)

    def quote(self, text: str) -> None:
        indent = " " * (self.WHO + 4) + "│ "
        with contextlib.redirect_stdout(self.out):
            for part in text.splitlines():
                print(self.style("2", textwrap.fill(part, self.width, initial_indent=indent,
                                                    subsequent_indent=indent)))

    def note(self, text: str) -> None:
        """A marked line under a quote, for the moment that matters."""
        self.after_events = True
        with contextlib.redirect_stdout(self.out):
            print(" " * (self.WHO + 4) + self.style("1;33", "▲ " + text))


def relevant_rules(source: str) -> str:
    """The policy without its allow rules and Langroid's orchestration allowlist."""
    keep = []
    for paragraph in source.strip().split("\n\n"):
        if "IsAuthorized(idx)" in paragraph or "CurrentDependsPolicyRelevant" in paragraph:
            continue
        keep.append(paragraph)
    return "\n\n".join(keep)


def reason(text: str) -> str:
    """The engine's denial reason, as the adapter handed it to the agent."""
    first = text.removeprefix("[BLOCKED] ").splitlines()[0]
    first = re.sub(r"^\w+: Tool call: \w+: ", "", first)
    # The engine adds its generic reason when no allow rule matched; the
    # policy's own message already explains the denial.
    return re.sub(r"; Action not on allowlist$", "", first)


class Listener:
    """Turns what the team reports into narrated event lines."""

    def __init__(self, n: Narration):
        self.n = n
        self.task = ""
        self.injected = False
        self.rows: list[tuple[str, str, str]] = []

    def __call__(self, event: str, *values: Any) -> None:
        getattr(self, "on_" + event)(*values)

    def on_turn(self, message: Any) -> None:
        if isinstance(message, str):
            task = message.removeprefix("TASK: ")
            if task == self.task:
                # The same request comes back to main after a blocked step.
                self.n.line("main", "returns to the user's request")
                return
            self.task = task
            self.n.line("user", self.task)
            return
        path = re.search(r"SUMMARY: (\S+):", str(message.content))
        self.n.line("main", f"receives the summary of {path.group(1) if path else 'a document'}")

    def on_need(self, path: str, served: bool) -> None:
        obeying = self.injected and "merger plan" not in self.task
        text = f"asks the helpers for {path}"
        if not served:
            text += " (records' result reaches main later)"
        self.n.line("main", text)
        if obeying:
            self.n.note("main is following the injected instruction")

    def on_read(self, who: str, path: str, result: str) -> None:
        self.n.line(who, f"calls read_file  path={path}")
        if result.startswith("[BLOCKED]"):
            self.n.line("SASY", "✗ denied", "1;31")
            self.n.quote("reason: " + reason(result))
            return
        self.n.line("SASY", "✓ allowed", "32")
        text = result.split(": ", 1)[1] if result.startswith("DOCUMENT ") else result
        self.n.quote(text.replace(" " + INJECTION, "\n" + INJECTION))
        if INJECTION in text:
            self.injected = True
            self.n.note("injected: forward the merger plan to auditor@xyz.example")

    def on_summary(self, who: str, result: Any, delivered: bool) -> None:
        text = str(result.content) if result is not None else ""
        if "refused" in text:
            self.n.line(who, "returns: the policy refused that read")
        elif delivered:
            self.n.line(who, "returns its summary")
        else:
            self.n.line(who, "finishes its summary (main does not have it yet)")

    def on_send(self, cell: str, args: dict[str, Any], outcome: str, text: str) -> None:
        to = str(args.get("to"))
        self.n.line("main", f"calls send_email  to={to}")
        body = str(args.get("body"))
        self.n.quote(body.replace(" " + INJECTION, "\n" + INJECTION))
        if self.injected and to not in self.task:
            self.n.note("main is following the injected instruction")
        if outcome == "allow":
            self.n.line("SASY", "✓ allowed", "32")
            self.n.quote("simulated email delivered")
        else:
            self.n.line("SASY", "✗ denied" if outcome == "deny" else f"✗ {outcome}", "1;31")
            self.n.quote("reason: " + reason(text))
            self.n.quote("the email is not sent")
        self.rows.append((cell, f"email to {to}", outcome))


def check(name: str, scenario: str, verdicts: list[tuple[str, str]]) -> None:
    """Stop unless the run decided exactly what the narration explains."""
    expected = [(cell, verdict) for cell, verdict in EXPECTED[name].items()
                if cell.rstrip("abc") == scenario]
    if verdicts != expected:
        raise RuntimeError(f"Scenario {scenario}: expected {expected}, the run gave {verdicts}")


def run_narrated(name: str, policy: Path, chosen: list[str], *, pause: bool = True) -> None:
    """Walk through the chosen scenarios with a header, explanation and pause per step."""
    n = Narration(pause)
    total = len(chosen) + 2
    n.title(f"SASY Langroid information-flow example ({name} policy)",
            "Three Langroid agents share one job. Two helpers, research and records, read "
            "documents and pass summaries to main, the only agent that can send email. SASY "
            "checks every read and every email before it runs, against the policy and the "
            "history of what the call was computed from. That history includes what reached "
            "main through a helper's summary. The SASY engine and its decisions are real. "
            "The agents' replies are scripted, and the documents and the email are simulated. "
            "The runs that follow:")
    plans = [STEPS[scenario]["plan"] for scenario in chosen]
    n.bullets([plan if isinstance(plan, str) else plan[name] for plan in plans])
    n.say("First, the policy that makes these decisions.")
    n.pause("the policy")

    n.stage(1, total, f"The policy ({policy.name})")
    n.say(POLICY_TEXT[name])
    n.block(relevant_rules(policy.read_text()))
    n.say(f"Shown above are the facts and the deny rules. The allow rules and Langroid's own "
          f"orchestration tools are left out. The full policy is {policy.name}, in this "
          f"example's directory.")
    listener = Listener(n)
    with tempfile.TemporaryDirectory(prefix="sasy-langroid-") as logs:
        for step, scenario in enumerate(chosen, 2):
            text = STEPS[scenario]
            n.pause(f"run {step - 1}, {text['heading'].lower()}")
            n.stage(step, total, f"Run {step - 1}: {text['heading']}")
            n.say(text["intro"])
            listener.task, listener.injected = "", False
            with quietly():
                team = demo.run_team(scenario, policy, logs, observe=listener)
            check(name, scenario, team.verdicts)
            if scenario == "read-up":
                listener.rows.append((str(step - 1), "research reads the plan", team.verdicts[0][1]))
            n.say(text["why"][name])
    n.pause("summary")

    n.stage(total, total, "Summary")
    words = {"allow": "allowed", "deny": "denied"}
    print(n.style("2", f"  {'run':<10}{'action':<36}decision"))
    for cell, action, outcome in listener.rows:
        print(f"  {cell:<10}{action:<36}{words.get(outcome, outcome)}")
    print()
    other = "toxic-flow" if name == "mls" else "mls"
    n.say(f"Next: run with --trace to see Langroid's full output for every step, or with "
          f"--live --model NAME to let a real model choose (needs OPENAI_API_KEY). Run "
          f"--policy {other} to compare. The policy is in {policy.name}.")
