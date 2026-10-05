"""Qualify the real LangChain loop against an owned SASY engine."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sasy

pytestmark = pytest.mark.integration


def example_directory():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples/langchain-information-flow"
        if candidate.is_dir():
            return candidate
    raise AssertionError("LangChain information-flow example is missing from the release")


def run_program(engine, program, *, live=False):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               SASY_LANGCHAIN_EXAMPLE=str(example_directory()),
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    if live:
        env.update(OPENAI_API_KEY=os.environ["OPENAI_API_KEY"],
                   SASY_LIVE_MODEL=os.environ["SASY_LIVE_MODEL"])
    result = subprocess.run([sys.executable, "-c", program], cwd=engine.root, env=env,
                            capture_output=True, text=True, timeout=240)
    output = result.stdout + result.stderr
    for secret in (engine.tenant_a_key, os.environ.get("OPENAI_API_KEY", "")):
        if secret:
            output = output.replace(secret, "[REDACTED]")
    assert result.returncode == 0, output


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_framework_information_flow_and_stored_graph(engine, asynchronous):
    run_program(engine, r'''
import asyncio
import os
import sys
from pathlib import Path
import sasy
from langchain_core.messages import ToolMessage
from sasy.observability import api
sys.path.insert(0, os.environ["SASY_LANGCHAIN_EXAMPLE"])
from demo import run
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
sasy.instrument(http=False)
record = api.resolve_events
recorded = []
def capture(snapshots):
    ids = record(snapshots)
    for snapshot, node_id in zip(snapshots, ids, strict=True):
        event = type(snapshot.event).FromString(snapshot.event.SerializeToString())
        event.id = node_id
        recorded.append(event)
    return ids
api.resolve_events = capture
for confidential, destination, allowed in [(False, "external", True), (True, "external", False), (True, "internal", True)]:
    recorded.clear()
    with sasy.session(policy=Path(os.environ["SASY_LANGCHAIN_EXAMPLE"]) / "policy.dl"):
        pending = run(confidential=confidential, destination=destination, asynchronous=ASYNC)
        output, deliveries = asyncio.run(pending) if ASYNC else pending
        assert len(deliveries) == int(allowed), (confidential, destination, deliveries)
        publication = [m for m in output["messages"] if isinstance(m, ToolMessage) and m.name == "publish"]
        assert len(publication) == 1
        assert publication[0].status == ("success" if allowed else "error")
        action = next(e for e in recorded if e.tools and e.tools[0].name == "publish")
        graph = api.backward_slice(action.id)
        sources = {data.get("derived_from", {}).get("name") for _, data in graph.nodes(data=True)}
        assert "draft_summary" in sources, sources
        assert ("read_confidential" if confidential else "read_public") in sources, sources
        assert len(graph.nodes) >= 7
print("LangChain real engine cases and transitive graph passed")
'''.replace("ASYNC", repr(asynchronous)))

POLICY = """
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").
.decl ConfidentialInput()
ConfidentialInput() :- CurrentDepends(id), ToolResult(id, "read_confidential", _).
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_confidential").
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read_public").
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "delegate").
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "publish"), !ConfidentialInput().
Unauthorized(idx) :- Actions(idx, a), IsTool(a, "publish"), ConfidentialInput().
"""


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_supervisor_parallel_workers_and_publication(engine, asynchronous):
    run_program(engine, r'''
import asyncio
import os
import sys
from pathlib import Path
import sasy
from langchain_core.messages import ToolMessage
from sasy.observability import api
sys.path.insert(0, os.environ["SASY_LANGCHAIN_EXAMPLE"])
from supervisor import run
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
sasy.instrument(langchain=True)
record = api.resolve_events
recorded = []
def capture(snapshots):
    ids = record(snapshots)
    for snapshot, node_id in zip(snapshots, ids, strict=True):
        event = type(snapshot.event).FromString(snapshot.event.SerializeToString())
        event.id = node_id
        recorded.append(event)
    return ids
api.resolve_events = capture
for confidential, destination, allowed in [(False, "external", True), (True, "external", False), (True, "internal", True)]:
    recorded.clear()
    with sasy.session(policy=Path(os.environ["SASY_LANGCHAIN_EXAMPLE"]) / "policy.dl"):
        pending = run(confidential=confidential, destination=destination, asynchronous=ASYNC)
        output, deliveries = asyncio.run(pending) if ASYNC else pending
        assert len(deliveries) == int(allowed), deliveries
        publication = [m for m in output["messages"] if isinstance(m, ToolMessage) and m.name == "publish"]
        assert len(publication) == 1 and publication[0].status == ("success" if allowed else "error")
        action = next(e for e in recorded if e.tools and e.tools[0].name == "publish")
        ancestry = api.backward_slice(action.id)
        sources = {d.get("derived_from", {}).get("name") for _, d in ancestry.nodes(data=True)}
        assert {"research", "review", "read_public"} <= sources, sources
        assert ("read_confidential" in sources) == confidential, sources
        agents = {d.get("agent") for _, d in ancestry.nodes(data=True)}
        assert {"supervisor", "researcher", "reviewer"} <= agents, agents
        # The public sibling cannot acquire the researcher's confidential read.
        reviewer = next(e for e in recorded if e.agent == "reviewer" and not e.tools)
        sibling = api.backward_slice(reviewer.id)
        assert not any(d.get("derived_from", {}).get("name") == "read_confidential"
                       for _, d in sibling.nodes(data=True))
        successes = {e.id for e in recorded if e.derived_from.name == "publish"}
        assert len(successes) == int(allowed), (allowed, successes)
print("Native supervisor: allow/deny, downstream provenance and sibling isolation passed")
'''.replace("ASYNC", repr(asynchronous)))

SCRIPTED = r'''
import asyncio
import os
import sasy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sasy.instrumentation.langchain import create_agent
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])

class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

def steps(*responses):
    return Scripted(responses=list(responses))

def call(name, args, id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}])

def read_confidential() -> str:
    "Read confidential data."
    return "BUDGET 42"

def read_public() -> str:
    "Read public data."
    return "WEATHER fine"

published = []
def publish(text: str) -> str:
    "Publish text outside the organization."
    published.append(text)
    return "published"

def publication(output):
    return [m for m in output["messages"] if isinstance(m, ToolMessage) and m.name == "publish"][-1].status
'''


def test_message_marks_carry_ancestry_across_turns_on_the_real_engine(engine):
    """A mark is honored only if the engine resolves it, which no test double decides."""
    run_program(engine, SCRIPTED + r'''
for source, carried, expected in [("read_confidential", "messages", "error"), ("read_public", "messages", "success"),
                                  ("read_confidential", "text", "success")]:
    published.clear()
    with sasy.session(policy=POLICY):
        first = create_agent(steps(call(source, {}, "read-1"), AIMessage("the figure")), [read_confidential, read_public]).invoke(
            {"messages": [HumanMessage("look it up")]})
        history = first["messages"] if carried == "messages" else [HumanMessage(str(first["messages"][-1].content))]
        second = create_agent(steps(call("publish", {"text": "the figure"}, "publish-1"), AIMessage("done")), [publish]).invoke(
            {"messages": [*history, HumanMessage("publish it")]})
        assert publication(second) == expected, (source, carried, publication(second))
        assert len(published) == int(expected == "success")
print("LangChain marks verified by the real engine across turns")
'''.replace("POLICY", repr(POLICY)))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sub_agent_answer_taints_the_supervisor_on_the_real_engine(engine, asynchronous):
    run_program(engine, SCRIPTED + r'''
for source, expected in [("read_confidential", "error"), ("read_public", "success")]:
    published.clear()
    if ASYNC:
        async def delegate(task: str) -> str:
            "Hand a task to a sub-agent."
            inner = create_agent(steps(call(source, {}, "inner-1"), AIMessage("the figure")), [read_confidential, read_public])
            return str((await inner.ainvoke({"messages": [HumanMessage(task)]}))["messages"][-1].content)
    else:
        def delegate(task: str) -> str:
            "Hand a task to a sub-agent."
            inner = create_agent(steps(call(source, {}, "inner-1"), AIMessage("the figure")), [read_confidential, read_public])
            return str(inner.invoke({"messages": [HumanMessage(task)]})["messages"][-1].content)
    outer = create_agent(steps(call("delegate", {"task": "look it up"}, "delegate-1"),
                               call("publish", {"text": "the figure"}, "publish-1"), AIMessage("done")), [delegate, publish])
    with sasy.session(policy=POLICY):
        request = {"messages": [HumanMessage("go")]}
        output = asyncio.run(outer.ainvoke(request)) if ASYNC else outer.invoke(request)
        assert publication(output) == expected, (source, publication(output))
        assert len(published) == int(expected == "success")
print("LangChain sub-agent ancestry enforced by the real engine")
'''.replace("POLICY", repr(POLICY)).replace("ASYNC", repr(asynchronous)))


def test_sub_agent_given_only_recorded_history_is_checked_with_the_outer_ancestry(engine):
    """Nothing but the delegating tool call ties the inner run to the outer one."""
    run_program(engine, SCRIPTED + r'''
for source, expected in [("read_confidential", "error"), ("read_public", "success")]:
    published.clear()
    statuses = []
    with sasy.session(policy=POLICY):
        history = create_agent(steps(AIMessage("noted")), []).invoke(
            {"messages": [HumanMessage("remember this")]})["messages"]

        def delegate(task: str) -> str:
            "Continue a recorded conversation in a sub-agent."
            inner = create_agent(steps(call("publish", {"text": "the figure"}, "inner-1"), AIMessage("inner done")), [publish])
            # No fresh user message and no system prompt: every input is a
            # message this session already recorded.
            output = inner.invoke({"messages": list(history)})
            statuses.append(publication(output))
            return str(output["messages"][-1].content)

        outer = create_agent(steps(call(source, {}, "read-1"), call("delegate", {"task": "go"}, "delegate-1"), AIMessage("done")),
                             [read_confidential, read_public, delegate])
        outer.invoke({"messages": [HumanMessage("go")]})
        assert statuses == [expected], (source, statuses)
        assert len(published) == int(expected == "success")
print("LangChain delegated runs of recorded history checked with the outer ancestry")
'''.replace("POLICY", repr(POLICY)))


def test_a_replaced_message_identity_reaches_the_real_engine_as_its_own_input(engine):
    """A second, different message under one identity is recorded, not dropped."""
    run_program(engine, SCRIPTED + r'''
from sasy.observability import api
from sasy.instrumentation.langchain import consume_messages
record = api.resolve_events
recorded = []
def capture(snapshots):
    ids = record(snapshots)
    for snapshot, node_id in zip(snapshots, ids, strict=True):
        event = type(snapshot.event).FromString(snapshot.event.SerializeToString())
        event.id = node_id
        recorded.append(event)
    return ids
api.resolve_events = capture

def delegate(task: str) -> str:
    "Read a stored message whose text has changed since the run recorded it."
    consume_messages([AIMessage("HIDDEN FIGURE", id="shared")])
    return "recalled"

with sasy.session(policy=POLICY):
    output = create_agent(steps(call("delegate", {"task": "recall"}, "delegate-1"),
                                call("publish", {"text": "onward"}, "publish-1"), AIMessage("done")),
                          [delegate, publish]).invoke(
        {"messages": [AIMessage("public summary", id="shared"), HumanMessage("go")]})
    assert publication(output) == "success", publication(output)
    action = next(e for e in recorded if e.tools and e.tools[0].name == "publish")
    texts = {data.get("text", "") for _, data in api.backward_slice(action.id).nodes(data=True)}
    # The text the body consumed is in the ancestry the check saw, as an input
    # of its own, and what was recorded for that identity first is still there.
    assert any("HIDDEN FIGURE" in text for text in texts), texts
    assert any("public summary" in text for text in texts), texts
print("LangChain records a replaced message identity as an input of its own")
'''.replace("POLICY", repr(POLICY)))


def test_provider_fields_added_to_a_marked_message_are_refused_by_the_real_engine(engine):
    """The stored content hash decides whether a mark holds, and no double does."""
    run_program(engine, SCRIPTED + r'''
from sasy.observability import api
record = api.resolve_events
recorded = []
def capture(snapshots):
    ids = record(snapshots)
    for snapshot, node_id in zip(snapshots, ids, strict=True):
        event = type(snapshot.event).FromString(snapshot.event.SerializeToString())
        event.id = node_id
        recorded.append(event)
    return ids
api.resolve_events = capture

# Three edits of a recorded answer. The first is a raw provider tool call the
# client would send. The other two are fields no list of "what a provider reads"
# had named: the keyword that picks a system message's transmitted role, and the
# response id the client reads to decide which messages it sends at all. The
# engine decides on the stored content hash, so all three have to break the mark.
hidden = [{"id": "hidden-1", "type": "function",
           "function": {"name": "publish", "arguments": '{"text": "HIDDEN FIGURE"}'}}]
def tamper_tool_call(message):
    message.additional_kwargs["tool_calls"] = hidden
def tamper_role(message):
    message.additional_kwargs["__openai_role__"] = "developer"
def tamper_response_id(message):
    message.response_metadata["id"] = "resp_HIDDEN FIGURE"
for edit, expected in [(None, "error"), (tamper_tool_call, "success"),
                       (tamper_role, "success"), (tamper_response_id, "success")]:
    published.clear()
    recorded.clear()
    with sasy.session(policy=POLICY):
        first = create_agent(steps(call("read_confidential", {}, "read-1"), AIMessage("the figure")), [read_confidential]).invoke(
            {"messages": [HumanMessage("look it up")]})
        answer = first["messages"][-1].model_copy(deep=True)
        if edit is not None:
            edit(answer)
        second = create_agent(steps(call("publish", {"text": "the figure"}, "publish-1"), AIMessage("done")), [publish]).invoke(
            {"messages": [answer, HumanMessage("publish it")]})
        # Unedited the mark holds and the confidential read is still in the
        # ancestry; edited the message is an input of unknown origin, and
        # whatever it carries is recorded as that input's text.
        assert publication(second) == expected, (edit, publication(second))
        action = next(e for e in recorded if e.tools and e.tools[0].name == "publish")
        texts = {data.get("text", "") for _, data in api.backward_slice(action.id).nodes(data=True)}
        carried = edit in (tamper_tool_call, tamper_response_id)
        assert any("HIDDEN FIGURE" in text for text in texts) == carried, (edit, texts)
print("LangChain messages are verified by the engine as a whole, field by field")
'''.replace("POLICY", repr(POLICY)))


def test_edits_an_earlier_record_would_have_hidden_break_the_mark_on_the_real_engine(engine):
    """Two messages an earlier record ran together are two records to the engine."""
    run_program(engine, SCRIPTED + r'''
# Each edit is one a record made of less than the whole message would have
# hidden. `model_dump(mode="json")` writes the bytes key as the string it
# decodes to, so the first edit leaves that dump unchanged; the id is in the
# dump but was left out of an earlier adapter projection, so the second edit
# left the record unchanged. This scenario does not show a provider acting on
# either one: for an assistant message with string content, as here, the pinned
# client sends the same payload before and after. The unit suite shows the
# shapes where it does not (a system message's transmitted role, a text block's
# item id). What is checked here is the engine's side, on a real engine: either
# edit makes this a different message from the one recorded, so its mark must
# stop verifying and the confidential read must leave the ancestry.
def swap_key_for_bytes(message):
    message.additional_kwargs = {b"__openai_role__": "developer"}
def change_item_id(message):
    message.id = "msg_other"
for edit, expected in [(None, "error"), (swap_key_for_bytes, "success"), (change_item_id, "success")]:
    published.clear()
    with sasy.session(policy=POLICY):
        figure = AIMessage("the figure", id="msg_first")
        figure.additional_kwargs = {"__openai_role__": "developer"}
        first = create_agent(steps(call("read_confidential", {}, "read-1"), figure), [read_confidential]).invoke(
            {"messages": [HumanMessage("look it up")]})
        answer = first["messages"][-1].model_copy(deep=True)
        if edit is not None:
            edit(answer)
        second = create_agent(steps(call("publish", {"text": "the figure"}, "publish-1"), AIMessage("done")), [publish]).invoke(
            {"messages": [answer, HumanMessage("publish it")]})
        assert publication(second) == expected, (edit, publication(second))
        assert len(published) == int(expected == "success"), (edit, published)
print("LangChain records keep apart what a JSON dump would merge")
'''.replace("POLICY", repr(POLICY)))


def test_live_langchain_provider_information_flow(engine):
    if os.environ.get("SASY_RUN_LIVE_FRAMEWORK_TESTS") != "1":
        pytest.skip("Set SASY_RUN_LIVE_FRAMEWORK_TESTS=1 for bounded paid provider integration")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("SASY_LIVE_MODEL"):
        pytest.fail("Live LangChain lane requires OPENAI_API_KEY and explicit SASY_LIVE_MODEL")
    run_program(engine, r'''
import os
import sys
from pathlib import Path
import sasy
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
sys.path.insert(0, os.environ["SASY_LANGCHAIN_EXAMPLE"])
from demo import run
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
sasy.instrument(http=True, langroid=False)
for confidential, allowed in [(False, True), (True, False)]:
    model = ChatOpenAI(model=os.environ["SASY_LIVE_MODEL"], max_tokens=512, max_retries=0, timeout=45)
    with sasy.session(policy=Path(os.environ["SASY_LANGCHAIN_EXAMPLE"]) / "policy.dl"):
        output, deliveries = run(confidential=confidential, model=model)
        results = [m for m in output["messages"] if isinstance(m, ToolMessage)]
        names = [m.name for m in results]
        assert ("read_confidential" if confidential else "read_public") in names, names
        assert "draft_summary" in names and "publish" in names, names
        publications = [m for m in results if m.name == "publish"]
        assert len(publications) == 1, names
        assert publications[0].status == ("success" if allowed else "error")
        assert len(deliveries) == int(allowed), deliveries
print("Live LangChain model and real engine allowed/denied cases passed")
''', live=True)


def test_live_langchain_supervisor_information_flow(engine):
    if os.environ.get("SASY_RUN_LIVE_FRAMEWORK_TESTS") != "1":
        pytest.skip("Set SASY_RUN_LIVE_FRAMEWORK_TESTS=1 for bounded paid provider integration")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("SASY_LIVE_MODEL"):
        pytest.fail("Live LangChain lane requires OPENAI_API_KEY and explicit SASY_LIVE_MODEL")
    run_program(engine, r'''
import os
import sys
from pathlib import Path
import sasy
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
sys.path.insert(0, os.environ["SASY_LANGCHAIN_EXAMPLE"])
from supervisor import run
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
sasy.instrument(http=True, langroid=False)
for confidential, allowed in [(False, True), (True, False)]:
    model = ChatOpenAI(model=os.environ["SASY_LIVE_MODEL"], max_tokens=512, max_retries=0, timeout=45)
    with sasy.session(policy=Path(os.environ["SASY_LANGCHAIN_EXAMPLE"]) / "policy.dl"):
        output, deliveries = run(confidential=confidential, model=model)
        results = [m for m in output["messages"] if isinstance(m, ToolMessage)]
        names = [m.name for m in results]
        assert names.count("research") == names.count("review") == names.count("publish") == 1, names
        publication = next(m for m in results if m.name == "publish")
        assert publication.status == ("success" if allowed else "error"), publication
        assert len(deliveries) == int(allowed), deliveries
print("Live LangChain supervisor allowed/denied cases passed")
''', live=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_denied_model_egress_with_real_provider_client(engine, asynchronous):
    run_program(engine, r'''
import asyncio
import json
import os
import httpx
import sasy
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import PermissionDeniedError
from sasy.instrumentation import http
from sasy.instrumentation.langchain import create_agent
from sasy.observability.api import backward_slice
sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
sasy.instrument(http=True, langroid=False)
requests = []
real_proxy = http.proxy_http
real_async_proxy = http.proxy_http_async
def capture(request, **kwargs):
    requests.append(request)
    yield from real_proxy(request, **kwargs)
async def capture_async(request, **kwargs):
    requests.append(request)
    async for response in real_async_proxy(request, **kwargs):
        yield response
http.proxy_http = capture
http.proxy_http_async = capture_async
sends = []
def unexpected(request):
    sends.append(request)
    raise AssertionError("The local provider transport must not run")
transport = httpx.MockTransport(unexpected)
model = ChatOpenAI(model="synthetic", api_key="synthetic-test-key",
                   base_url="https://model.example.invalid/v1", max_retries=0,
                   http_client=httpx.Client(transport=transport),
                   http_async_client=httpx.AsyncClient(transport=transport))
agent = create_agent(model, [], system_prompt="Synthetic system context")
with sasy.session(policy='IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "allowed_tool").'):
    try:
        inputs = {"messages": [HumanMessage("Synthetic input")]}
        asyncio.run(agent.ainvoke(inputs)) if ASYNC else agent.invoke(inputs)
    except PermissionDeniedError:
        pass
    else:
        raise AssertionError("Provider invocation should have been denied")
    assert sends == [] and len(requests) == 1
    request = requests[0]
    assert "model.example.invalid" in request.request.url
    assert len(request.input_node_ids) == 2
    for node in request.input_node_ids:
        assert len(backward_slice(node).nodes) == 1
print("Real provider client model egress denied before network dispatch")
'''.replace("ASYNC", repr(asynchronous)))


METADATA_POLICY = """
CurrentDependsPolicyRelevant() :- IsTool(_, "publish").
.decl MarkedMessage(id: symbol)
MarkedMessage(id) :-
    Current(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".
MarkedMessage(id) :-
    CurrentDepends(id),
    MessageMetadata(id, md),
    @json_get_str_path(md, "s:additional_kwargs.s:marker") = "exfil".
.decl MarkedInput()
MarkedInput() :- MarkedMessage(_).
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "publish"), !MarkedInput().
Unauthorized(idx) :- Actions(idx, a), IsTool(a, "publish"), MarkedInput().
"""


def test_message_metadata_is_read_by_a_policy_on_the_real_engine(engine):
    """A field outside the text decides the call, through `MessageMetadata`.

    `additional_kwargs` is not in a node's text, so this passes only if the
    adapter's record of the message reaches the evaluator as its own relation.
    """
    run_program(engine, SCRIPTED + r'''
for kwargs, expected in [({"marker": "exfil"}, "error"), ({"marker": "ordinary"}, "success"), ({}, "success")]:
    published.clear()
    with sasy.session(policy=POLICY):
        output = create_agent(steps(call("publish", {"text": "the figure"}, "publish-1"), AIMessage("done")),
                              [publish]).invoke(
            {"messages": [HumanMessage("publish it", additional_kwargs=kwargs)]})
        assert publication(output) == expected, (kwargs, publication(output))
        assert len(published) == int(expected == "success")
print("LangChain message metadata read by a policy on the real engine")
'''.replace("POLICY", repr(METADATA_POLICY)))
