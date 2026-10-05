"""A supervisor delegates to two agents, then attempts a simulated publication.

This file is the integration: the agents, their tools and the SASY setup. A plain
scripted run prints a narrated walkthrough, which lives in narrated.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import sasy
from console import with_trace
from demo import ScriptedModel
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

# Shared progress display; session creation and cleanup remain owned by SASY.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import policy_session  # noqa: E402


def call(name: str, args: dict[str, Any], identity: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": identity, "type": "tool_call"}


def run(*, confidential: bool = True, destination: str = "external", model: Any = None,
        asynchronous: bool = False, trace: bool = False) -> Any:
    """Run cooperating agents with scripted or live model calls and simulated effects."""
    deliveries = []

    def read_confidential() -> str:
        """Read the synthetic confidential budget."""
        return "Confidential budget: 42 units."

    def read_public() -> str:
        """Read the synthetic public announcement."""
        return "Public announcement: the sample product launches next month."

    def worker(private: bool, name: str):
        source = read_confidential if private else read_public
        worker_model, tools = with_trace(model or ScriptedModel(responses=[
            AIMessage("", tool_calls=[call(source.__name__, {}, name + "-read")]),
            AIMessage(name + " finished its summary."),
        ]), [source], name, trace)
        return create_agent(
            worker_model, tools, name=name,
            system_prompt=(f"Call {source.__name__} once to read your reference, then "
                           "briefly summarize its returned text. Do not guess the reference content."),
        )

    researcher = worker(confidential, "researcher")
    reviewer = worker(False, "reviewer")

    def research() -> str:
        """Ask the research agent for a document summary."""
        return str(researcher.invoke({"messages": [HumanMessage("Read and summarize the reference.")]})["messages"][-1].content)

    def review() -> str:
        """Ask an independent agent to review public information."""
        return str(reviewer.invoke({"messages": [HumanMessage("Review the public announcement.")]})["messages"][-1].content)

    async def aresearch() -> str:
        """Ask the research agent for a document summary."""
        result = await researcher.ainvoke({"messages": [HumanMessage("Read and summarize the reference.")]})
        return str(result["messages"][-1].content)

    async def areview() -> str:
        """Ask an independent agent to review public information."""
        result = await reviewer.ainvoke({"messages": [HumanMessage("Review the public announcement.")]})
        return str(result["messages"][-1].content)

    # The same tool names are presented to the supervisor in both modes.
    aresearch.__name__, areview.__name__ = "research", "review"

    def publish(text: str, destination: str = "internal") -> str:
        """Simulate publishing the combined report."""
        deliveries.append({"text": text, "destination": destination})
        return "Simulated publication succeeded"

    supervisor_model, tools = with_trace(model or ScriptedModel(responses=[
        AIMessage("", tool_calls=[call("research", {}, "research"), call("review", {}, "review")]),
        AIMessage("", tool_calls=[call("publish", {"text": "Combined report", "destination": destination}, "publish")]),
        AIMessage("Finished; inspect the publication decision."),
    ]), [aresearch if asynchronous else research, areview if asynchronous else review, publish], "supervisor", trace)
    supervisor = create_agent(
        supervisor_model, tools, name="supervisor",
        system_prompt=("Call research and review once each. After receiving both summaries, "
                       "call publish once with a combined report and the requested destination. "
                       "Do not retry or change the destination if publication is denied."),
    )
    inputs = {"messages": [HumanMessage(
        f"Prepare a report from both the research and review summaries, then publish it to '{destination}'. "
        "Attempt publication exactly once; the policy decides whether it may run."
    )]}
    if asynchronous:
        async def execute():
            return await supervisor.ainvoke(inputs, {"recursion_limit": 12}), deliveries
        return execute()
    return supervisor.invoke(inputs, {"recursion_limit": 12}), deliveries


HELP_DESCRIPTION = (
    'SASY LangChain supervisor example. A supervisor delegates research and\n'
    'review to two agents, then publishes a combined report. SASY applies the\n'
    'same rule as demo.py across the cooperating agents. The models are scripted\n'
    'and publishing is simulated.\n'
    '\n'
    'Start an engine first: uv run sasy engine start\n'
)
HELP_EXAMPLES = (
    'examples (from the repository root, with an engine running):\n'
    '  uv run examples/langchain-information-flow/supervisor.py\n'
    '  uv run examples/langchain-information-flow/supervisor.py --public\n'
    '  uv run examples/langchain-information-flow/supervisor.py --async\n'
    '  uv run examples/langchain-information-flow/supervisor.py --live  (needs OPENAI_API_KEY)\n'
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=HELP_DESCRIPTION, epilog=HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public", action="store_true", help="Workers read only public synthetic information")
    parser.add_argument("--destination", choices=("internal", "external"), default="external",
                        help="Publication channel (default: external)")
    parser.add_argument("--async", dest="asynchronous", action="store_true",
                        help="Run the agents asynchronously")
    parser.add_argument("--live", action="store_true", help="Use a live model (defaults to gpt-4.1-mini)")
    parser.add_argument("--live-model", help="Use this live model; also enables live mode")
    parser.add_argument("--quiet", action="store_true", help="Print only the final deliveries summary")
    parser.add_argument("--trace", action="store_true",
                        help="Print every model output and tool event instead of the narrated walkthrough")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter between steps")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    live = args.live or args.live_model is not None
    model = None
    if live:
        from langchain_openai import ChatOpenAI
        model_name = args.live_model or os.environ.get("SASY_LIVE_MODEL") or "gpt-4.1-mini"
        model = ChatOpenAI(model=model_name, max_tokens=512, max_retries=0, timeout=45)
    sasy.configure()
    # Live provider requests are mediated alongside the three LangChain agents.
    sasy.instrument(langchain=True, http=live)
    if not (live or args.quiet or args.trace):
        # The narrated walkthrough lives in narrated.py; it calls run() above.
        from narrated import supervisor_walkthrough

        supervisor_walkthrough(sys.modules[__name__], public=args.public,
                               destination=args.destination,
                               asynchronous=args.asynchronous, pause=not args.no_pause)
        return
    with policy_session(sasy.session, progress=not args.quiet, policy=Path(__file__).with_name("policy.dl")):
        pending = run(confidential=not args.public, destination=args.destination, model=model,
                      asynchronous=args.asynchronous, trace=not args.quiet)
        _, deliveries = asyncio.run(pending) if args.asynchronous else pending
    print(json.dumps({"simulated_deliveries": deliveries}, indent=2))


if __name__ == "__main__":
    main()
