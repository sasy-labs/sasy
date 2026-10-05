"""Exercise native Langroid tasks and explicit edits against an owned engine."""
import subprocess
import sys
from pathlib import Path

import pytest
import sasy

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_task_and_explicit_message_update(engine, asynchronous):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", PROGRAM + f"\nmain({asynchronous!r})"],
                            cwd=engine.root, env=env, capture_output=True, text=True, timeout=240)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output


PROGRAM = r'''
"""Run against an owned configured SASY engine; no live model/provider calls.

Environment: SASY_URL, SASY_API_KEY, TLS_CA_PATH (matching the engine harness).
PYTHONPATH must point at the SDK containing the snapshot protocol and adapter.
"""
import asyncio
import os
import tempfile
from pathlib import Path

import sasy
from langroid import ChatAgent, ChatAgentConfig, Task, TaskConfig, ToolMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.constants import DONE
from sasy.instrumentation.langroid import _version, record_message_update, record_message_update_async
from sasy.observability.api import backward_slice, get_current_input_ids


def main(asynchronous=False):
    calls = []
    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str
    class Agent(ChatAgent):
        def lookup(self, msg: Lookup) -> str:
            calls.append((msg.topic, get_current_input_ids()))
            return "synthetic result"
    def respond(text):
        return DONE + " completed" if "synthetic result" in text else '{"request":"lookup","topic":"public"}'
    sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
    sasy.instrument(http=False, langroid=True)
    with tempfile.TemporaryDirectory(prefix="sasy-langroid-probe-") as scratch:
        with sasy.session(policy='IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "lookup").'):
            agent = Agent(ChatAgentConfig(llm=MockLMConfig(response_fn=respond), use_tools=True,
                                         use_functions_api=False, show_stats=False))
            agent.enable_message(Lookup)
            task = Task(agent, interactive=False, config=TaskConfig(logs_dir=str(Path(scratch)/"logs")))
            result = asyncio.run(task.run_async("lookup public", turns=5)) if asynchronous else task.run("lookup public", turns=5)
            assert result is not None and len(calls) == 1
            assert calls[0][0] == "public" and calls[0][1]
            graph = backward_slice(calls[0][1][0])
            assert len(graph.nodes) >= 3
            old = _version(result).canonical_id
            result.content += " (explicit application edit)"
            new = asyncio.run(record_message_update_async(result, input_ids=[])) if asynchronous else record_message_update(result, input_ids=[])
            assert new != old
            edited = backward_slice(new)
            assert old in edited.nodes
            previous = backward_slice(old)
            assert new not in previous.nodes
            print({"async": asynchronous, "tool_invocations": len(calls),
                   "tool_input_ancestors": len(graph.nodes), "edited_version_ancestors": len(edited.nodes)})


'''


def test_resumed_task_and_sub_task_delegation(engine):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", DELEGATION_PROGRAM + "\nmain()"],
                            cwd=engine.root, env=env, capture_output=True, text=True, timeout=240)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output


DELEGATION_PROGRAM = r'''
"""Continue a task from its own history, and let a sub-task handle a tool the
parent emitted, against an owned configured SASY engine.

Environment: SASY_URL, SASY_API_KEY, TLS_CA_PATH (matching the engine harness).
"""
import os
import tempfile
from pathlib import Path

import sasy
from langroid import ChatAgent, ChatAgentConfig, Task, TaskConfig, ToolMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.constants import DONE
from sasy.instrumentation.langroid import _version
from sasy.observability.api import backward_slice, get_current_input_ids

POLICY = ('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "lookup").\n'
          'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "alpha").')


def resumed_task(logs):
    calls = []

    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str

    class Agent(ChatAgent):
        def lookup(self, msg: Lookup) -> str:
            calls.append(get_current_input_ids())
            return "synthetic result"

    agent = Agent(ChatAgentConfig(use_tools=True, use_functions_api=False, show_stats=False,
                                  llm=MockLMConfig(response_fn=lambda _: '{"request":"lookup","topic":"public"}')))
    agent.enable_message(Lookup)
    answer = agent.llm_response("user asks about a private record")
    recorded = _version(answer).canonical_id
    task = Task(agent, interactive=False, restart=False, config=TaskConfig(logs_dir=logs))
    assert task.init() is answer
    assert _version(answer).canonical_id == recorded, "the resume recorded a new version"
    task.step()
    assert len(calls) == 1 and calls[0], calls
    texts = {data.get("text", "") for _, data in backward_slice(calls[0][0]).nodes(data=True)}
    assert any("private record" in text for text in texts), texts
    return len(calls)


def delegated_task(logs):
    handled = []

    class Alpha(ToolMessage):
        request: str = "alpha"
        purpose: str = "Run alpha"

    class Handler(ChatAgent):
        def alpha(self, msg: Alpha) -> str:
            handled.append(get_current_input_ids())
            return DONE + " alpha ran"

    parent = ChatAgent(ChatAgentConfig(name="Parent", use_tools=True, use_functions_api=False, show_stats=False,
                                       llm=MockLMConfig(response_fn=lambda text: DONE if "alpha ran" in text else '{"request":"alpha"}')))
    parent.enable_message(Alpha, use=True, handle=False)
    sub = Handler(ChatAgentConfig(name="Sub", llm=None, use_tools=True, use_functions_api=False, show_stats=False))
    sub.enable_message(Alpha)
    parent_task = Task(parent, interactive=False, config=TaskConfig(logs_dir=logs))
    parent_task.add_sub_task(Task(sub, interactive=False, single_round=True, config=TaskConfig(logs_dir=logs)))
    parent_task.run("go", turns=6)
    assert len(handled) == 1 and handled[0], handled
    names = {tool.get("name") for _, data in backward_slice(handled[0][0]).nodes(data=True)
             for tool in data.get("tools", [])}
    assert "alpha" in names, names
    return len(handled)


def main():
    sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
    sasy.instrument(http=False, langroid=True)
    with tempfile.TemporaryDirectory(prefix="sasy-langroid-delegation-") as scratch:
        logs = str(Path(scratch) / "logs")
        with sasy.session(policy=POLICY):
            resumed = resumed_task(logs)
        with sasy.session(policy=POLICY):
            delegated = delegated_task(logs)
    print({"resumed_tool_calls": resumed, "delegated_tool_calls": delegated})
'''


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_reply_holding_several_tool_calls_records_each_result(engine, asynchronous):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", MULTI_TOOL_PROGRAM + f"\nmain({asynchronous!r})"],
                            cwd=engine.root, env=env, capture_output=True, text=True, timeout=240)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output


MULTI_TOOL_PROGRAM = r'''
"""One model reply holding several tool calls, against an owned configured engine.

The model chooses how many calls a reply holds, so a rule that reads a tool
result must see each of them. Environment: SASY_URL, SASY_API_KEY, TLS_CA_PATH.
"""
import asyncio
import os

import sasy
from langroid import ChatAgent, ChatAgentConfig, Entity, ToolMessage
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall
from sasy.instrumentation.langroid import record_message, recorded_version_id
from sasy.observability.api import backward_slice

# A send is denied when a read of a secret document is among the tool results
# the message being sent was computed from.
POLICY = (
    'CurrentDependsPolicyRelevant() :- IsTool(_, "send").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_file").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send").\n'
    'Unauthorized(idx) :- Actions(idx, a), IsTool(a, "note").\n'
    'Unauthorized(idx) :- Actions(idx, a), IsTool(a, "send"), CurrentDepends(id),\n'
    '    ToolResult(id, "read_file", args), path = @json_get_str(args, "path"),\n'
    '    @str_contains(path, "secret") = 1.\n'
)


class Read(ToolMessage):
    request: str = "read_file"
    purpose: str = "Read a file"
    path: str


class Note(ToolMessage):
    request: str = "note"
    purpose: str = "Write a note"
    text: str


class Send(ToolMessage):
    request: str = "send"
    purpose: str = "Send a message"
    to: str


class Worker(ChatAgent):
    def read_file(self, msg: Read) -> str:
        return f"contents of {msg.path}"

    def note(self, msg: Note) -> str:
        return f"noted {msg.text}"

    def send(self, msg: Send) -> str:
        return f"sent to {msg.to}"


def worker():
    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_functions_api=True,
                                   use_tools_api=True, show_stats=False))
    agent.enable_message([Read, Note, Send])
    return agent


def reply(agent, calls, inputs):
    """A model reply holding ``calls``, recorded as computed from ``inputs``."""
    document = ChatDocument(content="", metadata=ChatDocMetaData(sender=Entity.LLM))
    document.oai_tool_calls = [
        OpenAIToolCall(id=f"c{i}", type="function",
                       function=LLMFunctionCall(name=name, arguments=arguments))
        for i, (name, arguments) in enumerate(calls)
    ]
    record_message(document, input_ids=list(inputs), agent=agent)
    return document


def respond(agent, document, asynchronous):
    if asynchronous:
        return asyncio.run(agent.agent_response_async(document))
    return agent.agent_response(document)


def results(identifier):
    """The tool results reachable from a recorded message, by tool name."""
    return {data["derived_from"]["name"]: data["derived_from"]["arguments"]
            for _, data in backward_slice(identifier).nodes(data=True)
            if data.get("derived_from")}


def two_reads_then_a_send(asynchronous):
    agent = worker()
    read_both = reply(agent, [("read_file", {"path": "public/memo.txt"}),
                              ("read_file", {"path": "secret/plan.txt"})], [])
    combined = respond(agent, read_both, asynchronous)
    reached = results(recorded_version_id(combined))
    assert sorted(reached) == ["read_file"], reached
    paths = sorted(path for _, data in backward_slice(recorded_version_id(combined)).nodes(data=True)
                   if data.get("derived_from") for path in [data["derived_from"]["arguments"]])
    assert len(paths) == 2 and "public/memo.txt" in paths[0] and "secret/plan.txt" in paths[1], paths

    send = reply(agent, [("send", {"to": "partner"})], [recorded_version_id(combined)])
    answer = respond(agent, send, asynchronous)
    assert "[BLOCKED] send" in answer.content, answer.content
    return paths


def one_allowed_and_one_denied(asynchronous):
    agent = worker()
    both = reply(agent, [("read_file", {"path": "public/memo.txt"}),
                         ("note", {"text": "hi"})], [])
    combined = respond(agent, both, asynchronous)
    assert "[BLOCKED] note" in combined.content, combined.content
    reached = results(recorded_version_id(combined))
    assert sorted(reached) == ["read_file"], reached
    return combined.content


def one_call_on_its_own(asynchronous):
    agent = worker()
    single = reply(agent, [("read_file", {"path": "public/memo.txt"})], [])
    combined = respond(agent, single, asynchronous)
    recorded = recorded_version_id(combined)
    slice_ = backward_slice(recorded)
    # The response itself is the tool-result node, and nothing else in its
    # ancestry carries tool-result provenance.
    assert slice_.nodes[recorded]["derived_from"]["name"] == "read_file", slice_.nodes[recorded]
    assert [node for node, data in slice_.nodes(data=True) if data.get("derived_from")] == [recorded]
    return combined.content


def main(asynchronous=False):
    sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
    sasy.instrument(http=False, langroid=True)
    with sasy.session(policy=POLICY):
        paths = two_reads_then_a_send(asynchronous)
    with sasy.session(policy=POLICY):
        mixed = one_allowed_and_one_denied(asynchronous)
    with sasy.session(policy=POLICY):
        single = one_call_on_its_own(asynchronous)
    print({"async": asynchronous, "reads_recorded": len(paths),
           "mixed_blocked": "[BLOCKED] note" in mixed, "single": single})
'''


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_reply_whose_tools_nest_or_repeat_records_each_result(engine, asynchronous):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", NESTED_TOOL_PROGRAM + f"\nmain({asynchronous!r})"],
                            cwd=engine.root, env=env, capture_output=True, text=True, timeout=240)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output


NESTED_TOOL_PROGRAM = r'''
"""Tool calls that nest or repeat inside one reply, against an owned engine.

A handler may return another tool call, which Langroid dispatches at once, and
a reply may hold the same call twice. Environment: SASY_URL, SASY_API_KEY,
TLS_CA_PATH.
"""
import asyncio
import os
from types import SimpleNamespace

import sasy
from langroid import ChatAgent, ChatAgentConfig, Entity, ToolMessage
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall
from sasy.instrumentation import langroid as adapter
from sasy.instrumentation.langroid import record_message, recorded_version_id
from sasy.observability.api import backward_slice

# A send is denied when a read of a secret document is among the tool results
# the message being sent was computed from.
FLOW_POLICY = (
    'CurrentDependsPolicyRelevant() :- IsTool(_, "send").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_file").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "note").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send").\n'
    'Unauthorized(idx) :- Actions(idx, a), IsTool(a, "send"), CurrentDepends(id),\n'
    '    ToolResult(id, "read_file", args), path = @json_get_str(args, "path"),\n'
    '    @str_contains(path, "secret") = 1.\n'
)

# read_file is allowed and send is not: a call that tries to run the send
# handler under the read's name must not get through.
REDIRECT_POLICY = (
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_file").\n'
)

# A send is allowed only when a completed read is among the tool results it was
# computed from: a policy that asks for a tool to have run.
COMPLETED_READ_POLICY = (
    'CurrentDependsPolicyRelevant() :- IsTool(_, "send").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_file").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "note").\n'
    'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send"), CurrentDepends(id),\n'
    '    ToolResult(id, "read_file", args).\n'
)


class Read(ToolMessage):
    request: str = "read_file"
    purpose: str = "Read a file"
    path: str


class Note(ToolMessage):
    request: str = "note"
    purpose: str = "Write a note"
    text: str


class Send(ToolMessage):
    request: str = "send"
    purpose: str = "Send a body"
    to: str
    body: str = ""


class Worker(ChatAgent):
    """Reads return different data each time; a read of a plan sends it on."""

    reads: int = 0

    def read_file(self, msg: Read):
        self.reads += 1
        if msg.path.endswith("plan.txt"):
            return Send(to="partner", body=f"contents of {msg.path}")
        return f"contents of {msg.path} #{self.reads}"

    def note(self, msg: Note) -> str:
        return f"noted {msg.text}"

    def send(self, msg: Send) -> str:
        return f"delivered to {msg.to}"


def worker():
    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_tools=True,
                                   use_functions_api=False, show_stats=False))
    agent.enable_message([Read, Note, Send])
    return agent


def written_reply(agent, calls, inputs):
    """A reply that writes its calls into its text, as Langroid reads them."""
    import json
    document = ChatDocument(content="\n".join(json.dumps({"request": name, **arguments})
                                              for name, arguments in calls),
                            metadata=ChatDocMetaData(sender=Entity.LLM))
    record_message(document, input_ids=list(inputs), agent=agent)
    return document


def provider_reply(agent, calls, inputs):
    document = ChatDocument(content="", metadata=ChatDocMetaData(sender=Entity.LLM))
    document.oai_tool_calls = [
        OpenAIToolCall(id=f"c{index}", type="function",
                       function=LLMFunctionCall(name=name, arguments=arguments))
        for index, (name, arguments) in enumerate(calls)
    ]
    record_message(document, input_ids=list(inputs), agent=agent)
    return document


def respond(agent, document, asynchronous):
    if asynchronous:
        return asyncio.run(agent.agent_response_async(document))
    return agent.agent_response(document)


def tool_results(identifier):
    """Every node carrying tool-result provenance, as (tool name, text)."""
    return [(data["derived_from"]["name"], data.get("text", ""))
            for _, data in backward_slice(identifier).nodes(data=True)
            if data.get("derived_from")]


def a_handler_that_returns_a_tool(asynchronous, shape):
    """The read's result is the send it returned, and the send is checked on it."""
    reply = written_reply if shape == "text" else provider_reply
    agent = worker()
    secret = reply(agent, [("read_file", {"path": "secret/plan.txt"})], [])
    answer = respond(agent, secret, asynchronous)
    assert "[BLOCKED] send" in answer.content, answer.content
    reached = tool_results(recorded_version_id(answer))
    # The read has a result of its own -- the send it asked for -- and the
    # denied send has none.
    assert [name for name, _ in reached] == ["read_file"], reached
    assert '"request": "send"' in reached[0][1], reached[0][1]
    assert "secret/plan.txt" in reached[0][1], reached[0][1]

    agent = worker()
    public = reply(agent, [("read_file", {"path": "public/plan.txt"})], [])
    allowed = respond(agent, public, asynchronous)
    assert "delivered to partner" in allowed.content, allowed.content
    return reached


def two_identical_reads(asynchronous):
    """Two calls with the same arguments are two operations with two results."""
    agent = worker()
    both = written_reply(agent, [("read_file", {"path": "a.txt"}),
                                 ("read_file", {"path": "a.txt"}),
                                 ("note", {"text": "hi"})], [])
    answer = respond(agent, both, asynchronous)
    reached = tool_results(recorded_version_id(answer))
    reads = sorted(text for name, text in reached if name == "read_file")
    assert len(reads) == 2 and reads[0] != reads[1], reached
    assert [name for name, _ in reached].count("note") == 1, reached
    return reads


def a_refusal_at_the_second_gate(refusal):
    """Asynchronous dispatch delegates to the synchronous handler, so one
    operation crosses two gates. Refused at either, the operation has no
    result, and a policy that asks for a completed read is not satisfied."""
    decided = adapter.rm_check_tool_call

    def refuse(name, arguments, inputs):
        if name != "read_file":
            return decided(name, arguments, inputs)
        if refusal == "error":
            raise RuntimeError("no decision")
        trace = SimpleNamespace(action_description="denied at the second gate",
                                reasons=[], suggested_fixes=[])
        return SimpleNamespace(authorized=False, transform_ids=[], denial_trace=trace)

    agent = worker()
    read = written_reply(agent, [("read_file", {"path": "public/memo.txt"})], [])
    # The synchronous check is reached only as the second gate: the agent has
    # no asynchronous read handler.
    adapter.rm_check_tool_call = refuse
    try:
        answer = respond(agent, read, True)
    finally:
        adapter.rm_check_tool_call = decided
    assert "[BLOCKED] read_file" in answer.content, answer.content
    assert tool_results(recorded_version_id(answer)) == [], answer.content

    send = written_reply(agent, [("send", {"to": "partner"})], [recorded_version_id(answer)])
    denied = respond(agent, send, True)
    assert "[BLOCKED] send" in denied.content, denied.content
    return denied.content


def a_completed_read_lets_the_send_through():
    agent = worker()
    read = written_reply(agent, [("read_file", {"path": "public/memo.txt"})], [])
    answer = respond(agent, read, True)
    send = written_reply(agent, [("send", {"to": "partner"})], [recorded_version_id(answer)])
    allowed = respond(agent, send, True)
    assert "delivered to partner" in allowed.content, allowed.content
    return allowed.content


def a_call_that_names_its_own_handler(asynchronous):
    """A model-supplied "_handler" cannot redirect the call to another method.

    The call is refused before it is checked, so neither the read nor the send
    runs and the reply has no tool result of any kind.
    """
    agent = worker()
    call = written_reply(agent, [("read_file", {"path": "public/memo.txt",
                                                "_handler": "send"})], [])
    answer = respond(agent, call, asynchronous)
    assert "[BLOCKED] read_file" in answer.content, answer.content
    assert "may not name its own handler" in answer.content, answer.content
    assert "delivered to partner" not in answer.content, answer.content
    assert tool_results(recorded_version_id(answer)) == [], answer.content
    return answer.content


def main(asynchronous=False):
    sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
    sasy.instrument(http=False, langroid=True)
    chained = {}
    for shape in ("text", "provider"):
        with sasy.session(policy=FLOW_POLICY):
            chained[shape] = a_handler_that_returns_a_tool(asynchronous, shape)
    with sasy.session(policy=FLOW_POLICY):
        reads = two_identical_reads(asynchronous)
    with sasy.session(policy=REDIRECT_POLICY):
        redirected = a_call_that_names_its_own_handler(asynchronous)
    refused = {}
    if asynchronous:
        for refusal in ("deny", "error"):
            with sasy.session(policy=COMPLETED_READ_POLICY):
                refused[refusal] = "[BLOCKED] send" in a_refusal_at_the_second_gate(refusal)
        with sasy.session(policy=COMPLETED_READ_POLICY):
            a_completed_read_lets_the_send_through()
    print({"async": asynchronous, "chained": sorted(chained), "reads": reads,
           "redirected": "[BLOCKED]" in redirected, "second_gate_refused": refused})
'''
