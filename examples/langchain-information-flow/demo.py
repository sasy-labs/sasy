"""A real LangChain agent with scripted or live model calls and simulated sends.

This file is the integration: the agent, its tools and the SASY setup. A plain
scripted run prints a narrated walkthrough, which lives in narrated.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import sasy
from console import with_trace
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

# Shared progress display; session creation and cleanup remain owned by SASY.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import policy_session  # noqa: E402


class ScriptedModel(FakeMessagesListChatModel):
    """Exercise the real agent loop without a model provider or credentials."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


def scripted_model(confidential: bool, destination: str) -> ScriptedModel:
    responses = []
    for index, (name, arguments) in enumerate([
        ("read_confidential" if confidential else "read_public", {}),
        ("draft_summary", {"text": "Synthetic quarterly plans"}),
        ("publish", {"text": "A short summary", "destination": destination}),
    ]):
        responses.append(AIMessage(content="", tool_calls=[{
            "name": name, "args": arguments, "id": f"call-{index}", "type": "tool_call",
        }]))
    responses.append(AIMessage(content="Task finished; inspect the publication result."))
    return ScriptedModel(responses=responses)


def run(*, confidential: bool = True, destination: str = "external", model: Any = None, asynchronous: bool = False, trace: bool = False) -> Any:
    """Return an agent run plus its in-memory deliveries; no email is sent.

    Call ``sasy.instrument()`` first: it is what makes LangChain's
    ``create_agent`` build SASY's recorded and authorized agent.
    """
    sent: list[dict[str, str]] = []

    def read_public() -> str:
        """Read the synthetic public product announcement."""
        return "Public announcement: the sample product launches next month."

    def read_confidential() -> str:
        """Read the synthetic confidential product planning document."""
        return "Confidential synthetic plan: the sample product budget is 42 units."

    def draft_summary(text: str) -> str:
        """Prepare a short draft from the supplied text."""
        return "Draft: " + text[:120]

    def publish(text: str, destination: str = "internal") -> str:
        """Simulate publishing text to the internal or external channel."""
        sent.append({"text": text, "destination": destination})
        return "Simulated publication succeeded"

    model, tools = with_trace(model or scripted_model(confidential, destination),
                              [read_public, read_confidential, draft_summary, publish], "agent", trace)
    agent = create_agent(model, tools,
                         system_prompt="Use tools for each requested step. Respect denials; do not retry or change the destination.")
    prompt = (f"Read {'read_confidential' if confidential else 'read_public'}, "
              f"then call draft_summary with a short summary of its result, "
              f"then publish that draft to destination '{destination}'. "
              "Attempt the publication exactly once even if the document is confidential; the policy decides whether it may run.")
    inputs = {"messages": [HumanMessage(prompt)]}
    if asynchronous:
        async def execute() -> tuple[dict[str, Any], list[dict[str, str]]]:
            return await agent.ainvoke(inputs, recursion_limit=12), sent
        return execute()
    return agent.invoke(inputs, recursion_limit=12), sent


HELP_DESCRIPTION = (
    'SASY LangChain information-flow example. An agent reads a document, drafts a\n'
    'summary and publishes it. SASY denies publishing to the external channel\n'
    'when the call depends on the confidential document, even after the summary\n'
    'rewrote it. The model is scripted and publishing is simulated.\n'
    '\n'
    'Start an engine first: uv run sasy engine start\n'
)
HELP_EXAMPLES = (
    'examples (from the repository root, with an engine running):\n'
    '  uv run examples/langchain-information-flow/demo.py\n'
    '  uv run examples/langchain-information-flow/demo.py --destination internal\n'
    '  uv run examples/langchain-information-flow/demo.py --trace\n'
    '  uv run examples/langchain-information-flow/demo.py --live  (needs OPENAI_API_KEY)\n'
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=HELP_DESCRIPTION, epilog=HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public", action="store_true", help="Read only public synthetic information")
    parser.add_argument("--destination", choices=("internal", "external"),
                        help="Publication channel (default: external)")
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
    # HTTP mediation is for the live provider's requests; the scripted model
    # makes none.
    sasy.instrument(http=live)
    if not (live or args.quiet or args.trace):
        # The narrated walkthrough lives in narrated.py; it calls run() above.
        from narrated import demo_walkthrough

        demo_walkthrough(sys.modules[__name__], public=args.public,
                         destination=args.destination, pause=not args.no_pause)
        return
    with policy_session(sasy.session, progress=not args.quiet, policy=Path(__file__).with_name("policy.dl")):
        _, sent = run(confidential=not args.public, destination=args.destination or "external",
                      model=model, trace=not args.quiet)
    print(json.dumps({"simulated_deliveries": sent}, indent=2))


if __name__ == "__main__":
    main()
