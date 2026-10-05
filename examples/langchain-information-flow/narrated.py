"""Narrated event lines for the LangChain examples, from the agents' own callbacks.

``NarratedTrace`` collects one callback handler per agent. Each tool call is
shown once, followed by SASY's decision, the tool's result and, for a
delegation tool, the events of the agent it delegated to. Calls that one model
turn requests together run in parallel; their events are buffered and printed
in the order the model requested them, so the output does not depend on
thread scheduling.
"""
from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage

# The narration helper is shared by the examples, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narration import Narrator, policy_session  # noqa: E402

DENIED_PREFIX = "SASY denied "


@dataclass
class Call:
    """One tool call by the root agent, with everything it caused."""

    line: tuple[str, str]
    decision: list[tuple] = field(default_factory=list)
    children: list[tuple] = field(default_factory=list)
    done: bool = False


def describe(name: str, args: dict[str, Any]) -> str:
    """Render a tool call as ``calls name  key="value" ...``."""
    detail = "  ".join(f'{key}="{value}"' for key, value in args.items())
    return f"calls {name}  {detail}".rstrip()


def denial_lines(message: str) -> list[str]:
    """Split the adapter's denial text into the engine's reason and suggestion."""
    text = message.split(": ", 1)[1] if message.startswith(DENIED_PREFIX) else message
    reason, _, suggestion = text.partition(" Suggested: ")
    lines = [f"reason: {reason.strip()}"]
    if suggestion:
        lines.append(f"suggestion: {suggestion.strip()}")
    return lines


class NarratedTrace:
    """Collect the agents' callbacks and print them as narrated event lines.

    Args:
        narrator: The shared narrator that owns styling and width.
        root: Name of the agent the user talks to.
        delegates: Maps a worker agent's name to the root tool that runs it.
        width: Width of the column that names who acts.
    """

    def __init__(self, narrator: Narrator, root: str,
                 delegates: dict[str, str] | None = None, width: int = 6) -> None:
        self.n = narrator
        self.root = root
        self.delegates = delegates or {}
        self.width = width
        self.lock = RLock()
        self.reset()

    def reset(self) -> None:
        """Start a new run: forget the user prompt, calls and decisions."""
        self.batch: list[Call] = []
        self.calls: dict[str, Call] = {}  # root tool call id -> call
        self.active: dict[str, Call] = {}  # root tool name -> call in progress
        self.runs: dict[Any, tuple[str, str, str | None]] = {}
        self.decisions: list[tuple[str, str, bool]] = []  # (agent, tool, allowed)
        self.user_shown = False

    def __call__(self, agent: str) -> BaseCallbackHandler:
        """Return the callback handler for one agent; used by ``with_trace``."""
        return AgentHandler(self, agent)

    # Rendering -----------------------------------------------------------

    def _print_line(self, who: str, text: str, code: str, depth: int) -> None:
        self.n.pace()
        lead = "  " + "  " * depth
        hang = " " * (len(lead) + self.width + 2)
        body = textwrap.fill(text, self.n.width, initial_indent=hang,
                             subsequent_indent=hang)[len(hang):]
        first, *rest = body.split("\n")
        who_text = self.n.style("36", f"{who:<{self.width}}")
        print(f"{lead}{who_text}  {self.n.style(code, first)}", flush=True)
        for part in rest:
            print(self.n.style(code, part))
        self.n.after_events = True

    def _print_quote(self, text: str, depth: int) -> None:
        bar = " " * (2 + 2 * depth + self.width + 2) + "│ "
        for part in str(text).splitlines() or [""]:
            print(self.n.style("2", textwrap.fill(part, self.n.width, initial_indent=bar,
                                                  subsequent_indent=bar)))

    def _render(self, event: tuple) -> None:
        kind, *rest = event
        if kind == "line":
            self._print_line(*rest)
        else:
            self._print_quote(*rest)

    def _emit(self, events: list[tuple], call: Call | None) -> None:
        """Print events now, or keep them with the root call they belong to."""
        if call is None:
            for event in events:
                self._render(event)
        else:
            call.children.extend(events)

    def _flush(self, force: bool = False) -> None:
        """Print the buffered calls once every call of the turn has finished."""
        if self.batch and (force or all(call.done for call in self.batch)):
            for call in self.batch:
                self._render(("line", call.line[0], call.line[1], "0", 0))
                for event in call.decision + call.children:
                    self._render(event)
            self.batch = []

    def finish(self) -> None:
        """Print anything still buffered at the end of a run."""
        with self.lock:
            self._flush(force=True)

    def decision(self, tool: str) -> bool | None:
        """Return the last decision SASY made for ``tool`` in this run."""
        found = [allowed for _, name, allowed in self.decisions if name == tool]
        return found[-1] if found else None

    # Events from the handlers -------------------------------------------

    def _owner(self, agent: str) -> Call | None:
        """The root call whose output an agent's events belong to, if any."""
        if agent == self.root:
            return None
        return self.active.get(self.delegates.get(agent, ""))

    def user(self, agent: str, messages: Any) -> None:
        with self.lock:
            if agent != self.root or self.user_shown:
                return
            for batch in messages:
                for message in batch:
                    if isinstance(message, HumanMessage) and not self.user_shown:
                        self.user_shown = True
                        self._print_line("user", str(message.content), "0", 0)

    def model(self, agent: str, message: Any) -> None:
        with self.lock:
            calls = list(getattr(message, "tool_calls", []) or [])
            depth = 0 if agent == self.root else 1
            if agent != self.root:
                # A worker's final answer is shown as its delegation tool's result.
                events = [("line", agent, describe(c["name"], c["args"]), "0", depth)
                          for c in calls]
                self._emit(events, self._owner(agent))
                return
            self._flush(force=True)
            for c in calls:
                call = Call(line=(agent, describe(c["name"], c["args"])))
                self.calls[c["id"]] = call
                self.active[c["name"]] = call
                self.batch.append(call)
            text = getattr(message, "content", "")
            if not calls and text:
                self._print_line(agent, f"replies: {text}", "0", 0)

    def start(self, agent: str, name: str, run_id: Any, call_id: str | None) -> None:
        with self.lock:
            self.runs[run_id] = (agent, name, call_id)

    def end(self, run_id: Any, output: Any, error: BaseException | None = None) -> None:
        with self.lock:
            agent, name, call_id = self.runs.pop(run_id, ("", "tool", None))
            if isinstance(output, ToolMessage):
                error_text = output.content if output.status == "error" else None
                output = output.content
            else:
                error_text = None
            depth = 0 if agent == self.root else 1
            if error is not None and type(error).__name__ == "ActionDenied":
                allowed = False
                events = [("line", "SASY", "✗ denied", "1;31", depth)]
                events += [("quote", line, depth) for line in denial_lines(str(error))]
                if name == "publish":
                    events.append(("quote", "nothing is published", depth))
            elif error is not None or error_text is not None:
                allowed = True  # Authorized, but the tool itself failed.
                detail = f"{type(error).__name__}: {error}" if error else str(error_text)
                events = [("line", "SASY", "✓ allowed", "32", depth),
                          ("quote", f"tool error: {detail}", depth)]
            else:
                allowed = True
                events = [("line", "SASY", "✓ allowed", "32", depth),
                          ("quote", str(output), depth)]
            self.decisions.append((agent, name, allowed))
            if agent == self.root and call_id in self.calls:
                call = self.calls.pop(call_id)
                # The decision comes before the delegated agent's events.
                call.decision = events[:1]
                if call.children and allowed and error is None and error_text is None:
                    # A delegation tool returns the worker's answer.
                    events[1] = ("quote", f"returned to {agent}: {output}", depth)
                call.children.extend(events[1:])
                call.done = True
                if self.active.get(name) is call:
                    del self.active[name]
                self._flush()
            else:
                self._emit(events, self._owner(agent))


class AgentHandler(BaseCallbackHandler):
    """Forward one agent's model and tool callbacks to the shared trace."""

    run_inline = True

    def __init__(self, trace: NarratedTrace, agent: str) -> None:
        self.trace = trace
        self.agent = agent

    def __deepcopy__(self, memo: dict) -> AgentHandler:
        # SASY copies StructuredTools; keep one handler per agent.
        return self

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.trace.user(self.agent, messages)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for batch in response.generations:
            for generation in batch:
                self.trace.model(self.agent, getattr(generation, "message", None))

    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any,
                      tool_call_id: str | None = None, **kwargs: Any) -> None:
        self.trace.start(self.agent, serialized.get("name", "tool"), run_id, tool_call_id)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self.trace.end(run_id, output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self.trace.end(run_id, None, error)


# demo.py walkthrough

# Each run: confidential?, destination, title, plan bullet, before, after.
DEMO_RUNS = [
    (True, "external", "Run 1: confidential plan, external channel",
     "The agent reads the confidential plan and publishes a draft to the external "
     "channel. Expected: denied.",
     "The agent reads the confidential plan, drafts a summary and asks to publish it "
     "to the external channel.",
     "The draft text no longer contains the plan, but the publish call still depends on "
     "the confidential read through the conversation. The rule for external publication "
     "matches, and the engine denies the call before the tool runs."),
    (True, "internal", "Run 2: confidential plan, internal channel",
     "Same document, but the draft goes to the internal channel. Expected: allowed.",
     "Same document and steps. This time the agent publishes to the internal channel.",
     "The call still depends on the confidential read, but the deny rule covers only "
     "the external channel. An allow rule matches the internal destination, and the "
     "simulated publication runs."),
    (False, "external", "Run 3: public announcement, external channel",
     "The agent reads the public announcement and publishes to the external channel. "
     "Expected: allowed.",
     "The agent reads the public announcement instead, then publishes to the external "
     "channel.",
     "Nothing in this call's history comes from read_confidential. The deny rule does "
     "not match, and the external publication is allowed."),
]


def _demo_chosen_run(confidential: bool, destination: str) -> tuple:
    """The single narrated run for explicit --public / --destination flags."""
    document = "confidential plan" if confidential else "public announcement"
    expected = "denied" if confidential and destination == "external" else "allowed"
    after = next((entry[5] for entry in DEMO_RUNS if entry[:2] == (confidential, destination)),
                 "Nothing in this call's history comes from read_confidential, and the "
                 "channel is internal. The deny rule does not match, and the publication "
                 "is allowed.")
    return (confidential, destination, f"Run 1: {document}, {destination} channel",
            f"The agent reads the {document} and publishes a draft to the {destination} "
            f"channel. Expected: {expected}.",
            f"The agent reads the {document}, drafts a summary and asks to publish it to "
            f"the {destination} channel.",
            after)


def _run_demo(example: Any, runs: list[tuple], pause: bool = True) -> None:
    """Walk through scripted runs with a header, explanation and pause per step."""
    n = Narrator(pause=pause)
    total = len(runs) + 2
    n.title("SASY LangChain information-flow example",
            "A LangChain agent reads a document, drafts a short summary and asks to publish "
            "it. SASY checks every tool call before it runs, against a policy and the history "
            "of messages and tool results behind the call. The agent loop, the SASY engine and "
            "its decisions are real. The model's replies are scripted, and the documents and "
            "the publication are simulated. "
            + ("The runs that follow:" if len(runs) > 1 else "The run that follows:"))
    n.bullets([entry[3] for entry in runs])
    n.say("First, the policy that makes these decisions.")
    n.pause("the policy")

    n.stage(1, total, "The policy")
    n.say("Allow rules cover reading, drafting, delegation and publishing to either "
          "channel. One deny rule blocks publication to the external channel when the "
          "call's history contains a read_confidential result, even after a drafting "
          "step rewrote the text. The last two rules serve live mode and speed.")
    n.block(Path(__file__).with_name("policy.dl").read_text())
    n.pause(runs[0][2].lower())

    trace = NarratedTrace(n, root="agent")
    rows = []
    for step, (confidential, destination, title, _, before, after) in enumerate(runs, 2):
        n.stage(step, total, title)
        n.say(before)
        trace.reset()
        with policy_session(example.sasy.session, policy=Path(__file__).with_name("policy.dl")):
            _, sent = example.run(confidential=confidential, destination=destination, trace=trace)
        trace.finish()
        allowed = trace.decision("publish")
        expected = not (confidential and destination == "external")
        if allowed is not expected or len(sent) != int(expected):
            raise AssertionError(f"Unexpected publication: allowed={allowed}, sent={len(sent)}")
        n.say(after)
        rows.append(("confidential" if confidential else "public", destination,
                     "allowed" if allowed else "denied"))
        n.pause(runs[step - 1][2].lower() if step - 1 < len(runs) else "summary")

    n.stage(total, total, "Summary")
    header = ("run", "document read", "channel", "publish")
    for i, (label, read, destination, decision) in enumerate(
            [header] + [(str(i), *row) for i, row in enumerate(rows, 1)]):
        print(n.style("2" if i == 0 else "0", f"  {label:<6}{read:<16}{destination:<12}{decision}"))
    print()
    n.say("Next: run with --trace to see every model output and tool event, or with --live "
          "to let a real model choose the actions (needs OPENAI_API_KEY). The policy is in "
          "policy.dl. supervisor.py applies the same rule across cooperating agents.")


def demo_walkthrough(example: Any, *, public: bool, destination: str | None,
                     pause: bool = True) -> None:
    """Narrate demo.py's scripted runs; example is the demo.py module."""
    # Without scenario flags, narrate all three runs; with them, only that run.
    explicit = public or destination is not None
    runs = ([_demo_chosen_run(not public, destination or "external")] if explicit
            else DEMO_RUNS)
    _run_demo(example, runs, pause=pause)


# supervisor.py walkthrough

# Each run: confidential research?, title, plan bullet, before, after (external).
SUPERVISOR_RUNS = [
    (True, "Run 1: the researcher reads the confidential budget",
     "The researcher reads the confidential budget, the reviewer reads the public "
     "announcement, and the supervisor publishes a combined report.",
     "The supervisor delegates to both workers in one turn, and they run in parallel. "
     "Their events are shown grouped under the call that started them.",
     "Neither worker's answer nor the combined report quotes the budget. The publish "
     "call still depends on the researcher's answer, and that answer depends on the "
     "confidential read. The deny rule follows that history across the agent boundary "
     "and blocks the external publication."),
    (False, "Run 2: both workers read public information",
     "Same supervisor and request, but the researcher reads the public announcement.",
     "Same request. This time the researcher reads the public announcement as well.",
     "No agent read confidential data. Nothing in the publish call's history comes "
     "from read_confidential, and the external publication is allowed."),
]

# What to say after each run when the report goes to the internal channel.
SUPERVISOR_INTERNAL_AFTER = {
    True: "The publish call depends on the confidential read through the researcher's "
          "answer, but the deny rule covers only the external channel. The internal "
          "publication is allowed.",
    False: "No agent read confidential data, and the channel is internal. The "
           "publication is allowed.",
}


def _run_supervisor(example: Any, runs: list[tuple], destination: str = "external", asynchronous: bool = False,
                 pause: bool = True) -> None:
    """Walk through scripted runs with a header, explanation and pause per step."""
    n = Narrator(pause=pause)
    total = len(runs) + 2
    n.title("SASY LangChain supervisor example",
            "A supervisor agent asks a researcher and a reviewer for summaries, then asks to "
            f"publish a combined report to the {destination} channel. SASY checks every tool "
            "call of all three agents before it runs, against one policy and the history "
            "behind the call, including messages passed between agents. The LangChain agents, "
            "the SASY engine and its decisions are real. The models' replies are scripted, and "
            "the documents and the publication are simulated. "
            + ("The runs that follow:" if len(runs) > 1 else "The run that follows:"))
    n.bullets([f"{entry[2]} Expected: {'denied' if entry[0] and destination == 'external' else 'allowed'}."
               for entry in runs])
    n.say("First, the policy that makes these decisions.")
    n.pause("the policy")

    n.stage(1, total, "The policy")
    n.say("The policy is the one demo.py uses. Allow rules cover reading, the research "
          "and review delegation tools, and publishing. The deny rule blocks external "
          "publication when the call's history contains a read_confidential result. "
          "These are the relevant rules; the full file is policy.dl.")
    policy = Path(__file__).with_name("policy.dl").read_text()
    n.block(policy[policy.index(".decl"):policy.index("// The live mode")])
    n.pause(runs[0][1].lower())

    trace = NarratedTrace(n, root="supervisor", width=10,
                          delegates={"researcher": "research", "reviewer": "review"})
    rows = []
    for step, (confidential, title, _, before, after) in enumerate(runs, 2):
        n.stage(step, total, title)
        n.say(before)
        trace.reset()
        with policy_session(example.sasy.session, policy=Path(__file__).with_name("policy.dl")):
            pending = example.run(confidential=confidential, destination=destination,
                          asynchronous=asynchronous, trace=trace)
            _, deliveries = example.asyncio.run(pending) if asynchronous else pending
        trace.finish()
        allowed = trace.decision("publish")
        expected = not (confidential and destination == "external")
        if allowed is not expected or len(deliveries) != int(expected):
            raise AssertionError(f"Unexpected publication: allowed={allowed}, sent={len(deliveries)}")
        n.say(after if destination == "external" else SUPERVISOR_INTERNAL_AFTER[confidential])
        rows.append(("confidential" if confidential else "public",
                     "allowed" if allowed else "denied"))
        n.pause(runs[step - 1][1].lower() if step - 1 < len(runs) else "summary")

    n.stage(total, total, "Summary")
    header = ("run", "researcher read", f"publish ({destination})")
    for i, (label, read, decision) in enumerate(
            [header] + [(str(i), *row) for i, row in enumerate(rows, 1)]):
        print(n.style("2" if i == 0 else "0", f"  {label:<6}{read:<18}{decision}"))
    print()
    n.say("Next: run with --trace to see every model output and tool event, or with --live "
          "to let a real model drive all three agents (needs OPENAI_API_KEY). The policy is "
          "in policy.dl.")


def supervisor_walkthrough(example: Any, *, public: bool, destination: str,
                           asynchronous: bool, pause: bool = True) -> None:
    """Narrate supervisor.py's scripted runs; example is the supervisor.py module."""
    # Without --public, narrate both runs; with it, only the public run.
    runs = SUPERVISOR_RUNS
    if public:
        after = SUPERVISOR_RUNS[1][4]
        runs = [(False, "Run 1: both workers read public information",
                 "Both workers read the public announcement, and the supervisor "
                 "publishes a combined report.",
                 "The supervisor delegates to both workers in one turn, and they run in "
                 "parallel. Both read the public announcement.", after)]
    _run_supervisor(example, runs, destination, asynchronous, pause=pause)
