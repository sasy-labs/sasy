"""Three ADK agents, an exact-payment approval policy, and a synthetic ledger.

This file is the integration: the agents, their tools and the SASY calls. A plain
scripted run prints a narrated walkthrough, which lives in narrated.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import sasy
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow
from google.genai import types
from pydantic import PrivateAttr
from sasy.instrumentation.adk import event_ids

# Shared progress display; session creation and cleanup remain owned by SASY.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import policy_session  # noqa: E402

PAYMENT = {"request_id": "invoice-104", "payee": "Example Supplies", "amount": "120.00"}
POLICY = Path(__file__).with_name("payment_policy.dl")
DEFAULT_LIVE_MODEL = "gemini-3.8-flash"
USER_MESSAGE = "Process the synthetic invoice using the assigned roles."
SCENARIOS = ["approved", "wrong-payee", "wrong-amount", "wrong-request", "missing-approval",
             "same-agent", "denied-approval", "approval-error"]
_AGENT_COLORS = {"requester": "34", "reviewer": "35", "payer": "36"}


def _trace_event(event) -> None:
    """Show completed ADK events without changing the agent or its policy inputs."""
    if event.partial:
        return
    agent = event.author
    label = f"[{agent}]"
    color = "NO_COLOR" not in os.environ and sys.stdout.isatty()
    if color:
        label = f"\033[{_AGENT_COLORS.get(agent, '37')}m{label}\033[0m"

    def emit(kind, detail, *, status=None):
        line = f"{kind} {detail}"
        if color and status:
            line = f"\033[{'31' if status == 'error' else '32'}m{line}\033[0m"
        print(f"{label} {line}", flush=True)

    for call in event.get_function_calls():
        emit("REQUEST", f"{call.name}({json.dumps(call.args, ensure_ascii=False, default=str)})")
    for response in event.get_function_responses():
        result = response.response
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        if isinstance(result, dict) and str(result.get("error", "")).startswith("[BLOCKED]"):
            emit("DENIED", f"{response.name}: {rendered}", status="error")
        elif isinstance(result, dict) and "error" in result:
            emit("TOOL ERROR", f"{response.name}: {rendered}", status="error")
        else:
            emit("ALLOWED", f"{response.name}: {rendered}", status="ok")
    if event.content:
        for part in event.content.parts or []:
            if part.text:
                emit("MODEL", json.dumps(part.text, ensure_ascii=False))


class _ScriptedUsageFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == "google_adk.google.adk.telemetry._metrics"
                    and record.msg == "Skipping missing token usage metadata for agent %s and model %s"
                    and isinstance(record.args, tuple) and len(record.args) == 2
                    and record.args[1] == "scripted")


@contextmanager
def _demo_adk_warnings(*, live):
    """Hide only ADK notices caused by this example's supported scripted path."""
    metrics = logging.getLogger("google_adk.google.adk.telemetry._metrics")
    usage_filter = _ScriptedUsageFilter()
    if not live:
        metrics.addFilter(usage_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL is enabled\.",
                                    category=UserWarning)
            yield
    finally:
        if not live:
            metrics.removeFilter(usage_filter)


class ScriptedModel(BaseLlm):
    model: str = "scripted"
    _turns: list = PrivateAttr()

    def __init__(self, tool, args):
        super().__init__()
        self._turns = [LlmResponse(content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(name=tool, args=args, id=tool + "-1"))])),
            LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Step complete.")]))]

    async def generate_content_async(self, llm_request, stream=False):
        if stream:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Working…")]), partial=True)
        yield self._turns.pop(0)


async def run_scenario(scenario="approved", *, live=False, stream=False, trace=False, on_event=None, progress=True):
    """Run a real ADK workflow and SASY policy; return the synthetic ledger/events."""
    sasy.instrument(http=live)
    ledger = []
    requested, approved, dispatched = (dict(PAYMENT) for _ in range(3))
    if scenario == "wrong-payee":
        dispatched["payee"] = "Different Supplier"
    elif scenario == "wrong-amount":
        dispatched["amount"] = "900.00"
    elif scenario == "wrong-request":
        approved["request_id"] = "invoice-999"

    def submit_payment(request_id: str, payee: str, amount: str) -> dict:
        """Record a synthetic payment request."""
        return {"submitted": True, "request_id": request_id, "payee": payee, "amount": amount}

    def approve_payment(request_id: str, payee: str, amount: str) -> dict:
        """Approve this exact synthetic payment tuple."""
        if scenario == "approval-error":
            return {"error": "Synthetic approval service is unavailable"}
        return {"approved": True, "request_id": request_id, "payee": payee, "amount": amount}

    def disburse_payment(request_id: str, payee: str, amount: str) -> dict:
        """Append this payment to an in-memory ledger; no money moves."""
        ledger.append({"request_id": request_id, "payee": payee, "amount": amount})
        return {"disbursed": True}

    http_client = None
    provider = None
    if live:
        import httpx
        from google import genai
        from google.adk.models.google_llm import Gemini
        # Explicit httpx transport keeps model egress on SASY's supported HTTP
        # interception path even when ADK has installed aiohttp.
        http_client = httpx.AsyncClient()
        provider = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                                http_options=types.HttpOptions(httpx_async_client=http_client))
        def model(tool, args):
            return Gemini(model=os.environ.get("SASY_ADK_LIVE_MODEL") or DEFAULT_LIVE_MODEL, client=provider)
    else:
        model = ScriptedModel

    def make(name, tool, args):
        # Literal braces are ADK state templates; use a plain static instruction.
        instruction = f"Call {tool.__name__} exactly once with " + ", ".join(
            f"{k} = {v!r}" for k, v in args.items()) + ". After the result, finish."
        return LlmAgent(name=name, model=model(tool.__name__, args), tools=[tool], instruction=instruction)

    requester = make("requester", submit_payment, requested)
    reviewer = make("reviewer", approve_payment, approved)
    payer = make("payer", disburse_payment, dispatched)
    agents = [requester, payer] if scenario == "missing-approval" else [requester, reviewer, payer]
    if scenario == "same-agent":
        # Both tools belong to the same configured agent; a scripted two-call
        # conversation makes the distinction precise without duplicate names.
        requester.tools = [submit_payment, approve_payment]
        first = ScriptedModel("submit_payment", requested)
        first._turns = [first._turns[0], ScriptedModel("approve_payment", approved)._turns[0], first._turns[1]]
        requester.model = first
        agents = [requester, payer]
    runner = InMemoryRunner(node=Workflow(name="payments", edges=[(START, *agents)]), app_name="payments")
    adk_session = await runner.session_service.create_session(app_name="payments", user_id="demo-user")
    events = []
    try:
        policy_source = POLICY.read_text()
        if scenario == "denied-approval":
            policy_source += '\nUnauthorized(idx) :- Actions(idx, a), IsTool(a, "approve_payment").\n'
        with policy_session(sasy.session, progress=progress, policy=policy_source, backend="souffle"):
            async for event in runner.run_async(user_id="demo-user", session_id=adk_session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=USER_MESSAGE)]),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE if stream else StreamingMode.NONE)):
                events.append(event)
                if trace:
                    _trace_event(event)
                if on_event:
                    on_event(event)
                if not event.partial and event.get_function_calls():
                    from sasy.observability.api import backward_slice
                    for node in event_ids(runner, event.id, user_id="demo-user", session_id=adk_session.id):
                        graph = backward_slice(node)
                        assert node in graph, "Observed tool origin is missing from the real graph"
                        if event.author == "payer" and scenario == "approved":
                            authors = {data.get("agent") for _, data in graph.nodes(data=True)}
                            assert {"requester", "reviewer"} <= authors, "Payment input ancestry is incomplete"
                        if event.author == "payer" and scenario in ("denied-approval", "approval-error"):
                            assert not any(data.get("derived_from", {}).get("name") == "approve_payment"
                                           for _, data in graph.nodes(data=True)), "Failed approval fabricated ToolResult evidence"
    finally:
        if provider:
            await provider.aio.aclose()
            provider.close()
        if http_client:
            await http_client.aclose()
    return ledger, events


HELP_DESCRIPTION = (
    'SASY separation-of-duties example with Google ADK. A requester submits a\n'
    'payment, a reviewer approves it and a payer disburses it. SASY allows a\n'
    'disbursement only when a matching request and approval from the right agents\n'
    'are in its history. The models are scripted and the ledger is simulated.\n'
    '\n'
    'Start an engine first: uv run sasy engine start\n'
)
HELP_EXAMPLES = (
    'examples (from the repository root, with an engine running):\n'
    '  uv run examples/adk-separation-of-duties/main.py\n'
    '  uv run examples/adk-separation-of-duties/main.py --scenario wrong-payee\n'
    '  uv run examples/adk-separation-of-duties/main.py --trace\n'
    '  uv run examples/adk-separation-of-duties/main.py --live  (needs GOOGLE_API_KEY)\n'
)


def main():
    parser = argparse.ArgumentParser(
        description=HELP_DESCRIPTION, epilog=HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=SCENARIOS,
                        help="Run one scenario; the narrated default walks through several, "
                             "and --trace, --quiet and --live default to approved")
    parser.add_argument("--live", action="store_true", help="Use Gemini 3.8 Flash by default; requires GOOGLE_API_KEY")
    parser.add_argument("--stream", action="store_true", help="Use ordinary ADK SSE streaming; only final responses establish provenance")
    parser.add_argument("--quiet", action="store_true", help="Print only the final JSON summary")
    parser.add_argument("--trace", action="store_true", help="Print every completed ADK event instead of the narrated walkthrough")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter between steps")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    sasy.configure(ca_path=os.environ.get("TLS_CA_PATH"))
    if not (args.live or args.quiet or args.trace):
        # The narrated walkthrough lives in narrated.py; it runs run_scenario above.
        from narrated import NARRATED_DEFAULT, narration_quiet, run_narrated

        scenarios = [args.scenario] if args.scenario else NARRATED_DEFAULT
        with _demo_adk_warnings(live=False), narration_quiet():
            run_narrated(sys.modules[__name__], scenarios, pause=not args.no_pause, stream=args.stream)
        return
    scenario = args.scenario or "approved"
    with _demo_adk_warnings(live=args.live):
        ledger, events = asyncio.run(run_scenario(scenario, live=args.live, stream=args.stream,
                                                   trace=not args.quiet, progress=not args.quiet))
    expected = scenario == "approved"
    if bool(ledger) != expected:
        raise SystemExit("Unexpected ledger outcome; inspect model/tool events")
    print(json.dumps({"scenario": scenario, "dispatched": ledger, "events": len(events)}))

if __name__ == "__main__":
    main()
