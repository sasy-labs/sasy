"""Narrated walkthrough for the separation-of-duties example.

main.py holds the integration (agents, tools and SASY calls). This file only
explains a scripted run: it listens to the same ADK events and prints a header,
the events and a short explanation for each step.
"""

import asyncio
import json
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

# The narration helper is shared by the examples, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import Narrator  # noqa: E402

STORY = {
    "approved": ("Each agent does its own step for the same payment: invoice-104, "
                 "Example Supplies, 120.00.",
                 "The payer's call has a matching request from the requester and a matching "
                 "approval from the reviewer in its history. ApprovedPayment holds for this "
                 "payment, and the simulated ledger gets one entry."),
    "wrong-payee": ("The requester and the reviewer handle the same invoice. The payer then "
                    "pays Different Supplier instead of Example Supplies.",
                    "The request and the approval are both in the history, but they name "
                    "Example Supplies. No ApprovedPayment fact matches the payee in the call, "
                    "and SASY denies it before the ledger changes."),
    "wrong-amount": ("The payer pays 900.00 instead of the approved 120.00.",
                     "The approval covers 120.00 only. An approval binds the whole payment, "
                     "and a different amount has no matching evidence."),
    "wrong-request": ("The reviewer approves invoice-999 instead of invoice-104. The payer "
                      "then pays invoice-104.",
                      "The approval names invoice-999 and the call names invoice-104. "
                      "ApprovedPayment needs an approval for the exact request ID in the call."),
    "missing-approval": ("The reviewer is left out of the workflow. The payer runs right "
                         "after the requester.",
                         "The history has the request but no approval result. ApprovedPayment "
                         "needs both, and the disbursement is denied."),
    "same-agent": ("The requester submits the payment and then approves it itself. The "
                   "reviewer does not run.",
                   "An approval with the right fields is in the history, but the requester "
                   "produced it. The policy accepts approvals only from the agent labeled "
                   "reviewer. The application assigns these labels; they are not separate "
                   "logins."),
    "denied-approval": ("This run adds one rule to the policy that blocks every "
                        "approve_payment call.",
                        "The approval call is denied before it runs and leaves no approval "
                        "result. The payer's call then has no approval in its history and is "
                        "denied too."),
    "approval-error": ("The approval service fails. SASY allows the approval call, but the "
                       "tool returns an error.",
                       "SASY records no approval result for a failed tool call. The payer's "
                       "call has no approval evidence in its history and is denied."),
}
NARRATED_DEFAULT = ["approved", "wrong-payee", "missing-approval", "same-agent", "approval-error"]
RESULT_TEXT = {"submit_payment": "payment request recorded",
               "approve_payment": "approval recorded",
               "disburse_payment": "simulated payment added to the ledger"}


class NarratedTrace:
    """Turn the completed ADK events of a scripted run into short narrated lines."""

    def __init__(self, narrator):
        self.n = narrator
        self.payment = None

    def __call__(self, event):
        if event.partial:
            return
        n = self.n
        for call in event.get_function_calls():
            args = call.args or {}
            if call.name == "disburse_payment":
                self.payment = args
            fields = ", ".join(str(args.get(k, "")) for k in ("request_id", "payee", "amount"))
            n.line("agent", f"{event.author} calls {call.name}({fields})")
        for response in event.get_function_responses():
            result = response.response
            error = str(result.get("error", "")) if isinstance(result, dict) else ""
            if error.startswith("[BLOCKED]"):
                n.line("SASY", "✗ denied", "1;31")
                reason = error.removeprefix("[BLOCKED] ").removeprefix(f"SASY denied {response.name}: ")
                n.quote("reason: " + reason)
                n.quote("no payment is made" if response.name == "disburse_payment"
                        else "the tool does not run")
            else:
                n.line("SASY", "✓ allowed", "32")
                n.quote("tool error: " + error if error
                        else RESULT_TEXT.get(response.name, json.dumps(result, default=str)))


def run_narrated(example, scenarios, *, pause=True, stream=False):
    """Walk through scripted runs with a header, explanation and pause per step.

    example is the main.py module; its run_scenario performs the real SASY calls.
    """
    n = Narrator(pause=pause)
    total = len(scenarios) + 2
    n.title("SASY separation-of-duties example",
            "Three ADK agents handle one synthetic invoice. A requester submits a payment, a "
            "reviewer approves it, and a payer disburses it. SASY checks every tool call before "
            "it runs, and allows a disbursement only when a matching request and approval are "
            "in the call's history. The ADK agents, the SASY engine and its decisions are real. "
            "The models are scripted and the ledger is simulated; no money moves.")
    n.say("The runs that follow:" if len(scenarios) > 1 else "The run that follows:")
    n.bullets([f"{s}: {STORY[s][0]} Expected: "
               f"{'allowed' if s == 'approved' else 'blocked'}." for s in scenarios])
    n.say("First, the policy that makes these decisions.")
    n.pause("the policy")

    n.stage(1, total, "The policy")
    n.say("Requests and approvals are always allowed. A disbursement is allowed only when "
          "ApprovedPayment holds for its request ID, payee and amount. ApprovedPayment needs a "
          "submit_payment result from the requester and an approve_payment result from the "
          "reviewer in the call's history, both with the same three fields. The host rule lets "
          "live Gemini requests through. A disbursement no rule allows is denied with the "
          "message on its rule.")
    n.block(example.POLICY.read_text())
    rows = []
    for step, scenario in enumerate(scenarios, 2):
        n.pause(f"run {step - 1}, {scenario}")
        n.stage(step, total, f"Run {step - 1}: {scenario}")
        n.say(STORY[scenario][0])
        n.line("user", example.USER_MESSAGE)
        trace = NarratedTrace(n)
        ledger, _ = asyncio.run(example.run_scenario(scenario, stream=stream, on_event=trace))
        if bool(ledger) != (scenario == "approved"):
            raise SystemExit("Unexpected ledger outcome; inspect model/tool events")
        n.say(STORY[scenario][1])
        payment = trace.payment or {}
        rows.append((f"{step - 1}  {scenario}", payment.get("payee", ""),
                     payment.get("amount", ""), "allowed" if ledger else "denied"))
    n.pause("summary")

    n.stage(total, total, "Summary")
    for i, (run, payee, amount, decision) in enumerate([("run", "payee", "amount", "payment")] + rows):
        print(n.style("2" if i == 0 else "0", f"  {run:<24}{payee:<22}{amount:<10}{decision}"))
    print()
    rest = [s for s in example.SCENARIOS if s not in scenarios]
    if rest:
        kind = "scenarios" if "approved" in rest else "denial scenarios"
        n.say(f"Other {kind}: " + ", ".join(rest) + ". Run one with --scenario NAME.")
    n.say("Next: run with --trace to see every ADK event, or with --live to let Gemini choose "
          "the actions (needs GOOGLE_API_KEY). The policy is in payment_policy.dl.")


@contextmanager
def narration_quiet():
    """Hide framework log records and warnings so only the narration shows."""
    adk = logging.getLogger("google_adk")
    level = adk.level
    adk.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        adk.setLevel(level)
