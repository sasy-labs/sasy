"""Narrated walkthrough for the message-flow example.

demo.py holds the integration (tools, recorded messages and the SASY check). This
file only explains a scripted run: it receives the same trace events and prints
a header, the events and a short explanation for each step.
"""

import sys
from pathlib import Path

# The narration helper is shared by the examples, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import Narrator  # noqa: E402


class NarratedTrace:
    """Turn the scripted run's trace events into short narrated lines."""

    def __init__(self, narrator):
        self.n = narrator
        self.tool = None

    def __call__(self, event, value):
        n = self.n
        if event == "USER":
            n.line("user", value)
        elif event == "REQUEST":
            self.tool = value["tool"]
            detail = "  ".join(f"{k}={v}" for k, v in value["arguments"].items() if k != "body")
            n.line("agent", f"calls {self.tool}  {detail}")
        elif event == "ALLOW":
            n.line("SASY", "✓ allowed", "32")
        elif event == "DENY":
            n.line("SASY", "✗ denied", "1;31")
        elif event == "RESULT" and self.tool == "send_summary":
            n.quote("simulated email delivered")
        elif event == "RESULT":
            n.quote(value["content"])
        elif event == "BLOCKED":
            n.quote("reason: " + value["content"].removeprefix("[BLOCKED] "))
            n.quote("the email is not sent")
        elif event == "SCRIPTED OUTPUT":
            n.line("model", "writes the summary from the request and everything the agent read")


INTRO = ("A user asks an agent to summarize a confidential report and email the summary to "
         "their reviewer. SASY checks every tool call before it runs, against a policy and "
         "the history behind the call. The SASY engine and its decisions are real; the "
         "documents and the email are simulated.")

# The two scripted runs: the plan bullet, the step heading, and what to say
# before and after each run.
RUNS = {
    False: {
        "bullet": "Without the external note: the agent reads the report and emails the "
                  "summary to the reviewer. The email depends on the report but not on the "
                  "note, and SASY allows it.",
        "heading": "without the external note",
        "before": "The user asks for a summary of the report, sent to their reviewer.",
        "after": "The send depends on the sensitive report but not on the external note. "
                 "The policy allows it, and the reviewer gets the email.",
    },
    True: {
        "bullet": "With the external note: the agent also reads an untrusted note from an "
                  "outside sender, which says to send the summary to a different address. "
                  "The agent follows it. The email now depends on both the untrusted note "
                  "and the confidential report, and the policy blocks that combination.",
        "heading": "with the external note",
        "before": "Same request. This time the agent also reads a note from an outside "
                  "sender, which tells it to send the summary to a different address.",
        "after": "The agent follows the note and addresses the email to the outsider. The "
                 "send now depends on both documents through the summary. The policy "
                 "denies it before anything is sent. No rule names the attacker's address "
                 "or looks for attack text.",
    },
}


def run_narrated(example, pause=True, only_note=False):
    """Walk through the scripted runs with a header, explanation and pause per step.

    example is the demo.py module; its run_scenario performs the real SASY calls.
    With only_note, only the run with the external note is shown.
    """
    n = Narrator(pause=pause)
    runs = [True] if only_note else [False, True]
    total = len(runs) + 2
    if only_note:
        n.title("SASY message-flow example",
                INTRO + " On the way, the agent reads an untrusted note from an outside "
                "sender, which says to send the summary to a different address.")
        n.say("Expected: the agent follows the note, and the policy blocks the email, "
              "because it depends on both the untrusted note and the confidential report.")
    else:
        n.title("SASY message-flow example", INTRO + " The same request runs twice:")
        n.bullets([RUNS[untrusted]["bullet"] for untrusted in runs])
    n.say("First, the policy that makes these decisions.")
    n.pause("the policy")

    n.stage(1, total, "The policy")
    n.say("The policy allows reading the two documents and sending a summary. It denies a send "
          "whose history includes both the external note (untrusted) and the sensitive report. "
          "The history counts indirect influence too, for example through a summary.")
    n.block(Path(example.__file__).with_name("policy.dl").read_text())

    trace = NarratedTrace(n)
    results = []
    for number, untrusted in enumerate(runs, 1):
        run = RUNS[untrusted]
        label = "the run" if only_note else f"run {number}, {run['heading']}"
        n.pause(label)
        n.stage(number + 1, total, "The run with the external note" if only_note
                else f"Run {number}: {run['heading']}")
        before = run["before"]
        if only_note:
            before = ("The user asks for a summary of the report, sent to their reviewer. "
                      "The agent also reads a note from an outside sender, which tells it to "
                      "send the summary to a different address.")
        n.say(before)
        results.append((number, untrusted, example.run_scenario(untrusted=untrusted, console=trace)))
        n.say(run["after"])
    n.pause("summary")

    n.stage(total, total, "Summary")
    rows = [("run", "recipient", "decision")]
    for number, untrusted, (to, allowed) in results:
        rows.append((f"{number}  {'with' if untrusted else 'without'} the note", to,
                     "allowed" if allowed else "denied"))
    for i, (run, to, decision) in enumerate(rows):
        print(n.style("2" if i == 0 else "0", f"  {run:<26}{to:<26}{decision}"))
    print()
    if only_note:
        n.say("Next: run without --only-note to also see the same request without the note, "
              "which SASY allows. Use --trace to see every recorded event, or --live to let "
              "a real model choose the actions (needs OPENAI_API_KEY). The policy is in "
              "policy.dl.")
    else:
        n.say("Next: run with --trace to see every recorded event, or with --live to let a "
              "real model choose the actions (needs OPENAI_API_KEY). The policy is in "
              "policy.dl.")
