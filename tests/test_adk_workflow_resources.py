"""Resource provenance through native Workflow scheduling boundaries."""
import asyncio
import json

import pytest
from google.adk.workflow import JoinNode, Workflow
from test_adk_instrumentation import (
    AdkInstrumentationError,
    InMemoryRunner,
    LlmAgent,
    ScriptedModel,
    adk,
    calls,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


async def _run(workflow, initial=None):
    runner = InMemoryRunner(node=workflow, app_name="test")
    adk.instrument()
    native_session = await runner.session_service.create_session(
        app_name="test", user_id="u", state=initial)
    with session(end_on_exit=False):
        async for _ in runner.run_async(user_id="u", session_id=native_session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="review")])):
            pass
    return runner, native_session


def _consumer(name="consumer"):
    def consume():
        """Consume the recorded review."""
        return "consumed"
    return LlmAgent(name=name, include_contents="none",
        instruction="State {decision}; report {artifact.report}", tools=[consume],
        model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]))


def _writer(name="writer"):
    async def produce(tool_context):
        """Write the review to state and a text artifact."""
        tool_context.state["decision"] = "approved"
        await tool_context.save_artifact("report", types.Part(text="review evidence"))
        return "saved"
    return LlmAgent(name=name, tools=[produce],
        model=ScriptedModel([calls(("produce", {}, "produce")), text("ready")]))


def _resource_nodes(sink, nodes, kind):
    result = []
    for event in ancestors(sink, nodes):
        try:
            payload = json.loads(event.text)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("adk_resource", [None])[0] == kind:
            result.append(event)
    return result


@pytest.mark.parametrize("shape", ["serial", "prefix", "fork", "suffix"])
def test_sequential_regions_preserve_state_and_artifact_producers(sink, shape):
    writer, consumer = _writer(), _consumer()
    if shape == "serial":
        edges = [("START", writer, consumer)]
    else:
        left = LlmAgent(name="left", model=ScriptedModel([text("left result")]))
        right = LlmAgent(name="right", model=ScriptedModel([text("right result")]))
        merge = JoinNode(name="merge")
        if shape == "suffix":
            edges = [("START", (left, right), merge, writer, consumer)]
        elif shape == "fork":
            edges = [("START", writer, (left, right), merge, consumer)]
        else:
            fork = LlmAgent(name="fork", model=ScriptedModel([text("dispatch")]))
            edges = [("START", writer, fork, (left, right), merge, consumer)]
    runner, native_session = asyncio.run(_run(Workflow(name="resources", edges=edges),
        {"unread": "unread secret"}))
    consumed = next(check for check in sink.checks if check[0] == "consume")
    for kind in ("state", "artifact"):
        resources = _resource_nodes(sink, consumed[2], kind)
        assert resources, f"Missing {kind} dependency"
        assert any("observed production" in event.text for event in resources)
        assert any(event.agent == "writer" for event in ancestors(sink, [e.id for e in resources]))
    assert "approved" in str(consumer.model._requests[0].config.system_instruction)
    assert "review evidence" in str(consumer.model._requests[0].config.system_instruction)
    assert not any("unread secret" in event.text for event in sink.events.values())
    assert runner.session_service.sessions["test"]["u"][native_session.id].state["decision"] == "approved"


@pytest.mark.parametrize("selected", ["approved", "rejected", "held"])
def test_exclusive_branch_resources_reach_only_selected_finalizer(sink, selected):
    def select(tool_context):
        """Choose exactly one review branch."""
        tool_context.actions.route = selected
        return "selected"
    router = LlmAgent(name="router", tools=[select],
        model=ScriptedModel([calls(("select", {}, "route")), text("route")]))
    branches = {name: _writer(name) for name in ("approved", "rejected", "held")}
    consumer = _consumer()
    workflow = Workflow(name="conditional_resources", edges=[
        ("START", router, branches), *((branch, consumer) for branch in branches.values())])
    asyncio.run(_run(workflow))
    check = next(check for check in sink.checks if check[0] == "consume")
    resources = _resource_nodes(sink, check[2], "state") + _resource_nodes(sink, check[2], "artifact")
    assert len(resources) >= 2
    lineage = ancestors(sink, [event.id for event in resources])
    assert any(event.agent == selected for event in lineage)
    for name, branch in branches.items():
        assert bool(branch.model._requests) == (name == selected)
        if name != selected:
            assert not any(event.agent == name for event in sink.events.values())


def test_post_join_same_value_output_keys_keep_distinct_productions(sink):
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]))
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    first = LlmAgent(name="first", output_key="decision", model=ScriptedModel([text("same")]))
    second = LlmAgent(name="second", instruction="Previous {decision}", include_contents="none",
        output_key="decision", model=ScriptedModel([text("same")]))
    final = LlmAgent(name="final", instruction="Current {decision}", include_contents="none",
        model=ScriptedModel([text("finished")]))
    asyncio.run(_run(Workflow(name="overwrites", edges=[
        ("START", (left, right), JoinNode(name="merge"), first, second, final)])))
    output = next(event for event in sink.events.values() if event.text == "finished")
    productions = [event for event in _resource_nodes(sink, [output.id], "state")
                   if '"same"' in event.text and "observed production" in event.text]
    assert {event.agent for event in productions} == {"first", "second"}
    assert len({event.id for event in productions}) == 2
    first_production = next(event for event in productions if event.agent == "first")
    assert not any(event.agent == "second" for event in ancestors(sink, [first_production.id]))
    assert "same" in str(second.model._requests[0].config.system_instruction)
    assert "same" in str(final.model._requests[0].config.system_instruction)


def test_post_join_unobserved_state_mutation_rejected_before_consumer(sink):
    from sasy.instrumentation.adk_state import _frame
    def mutate():
        """Simulate an out-of-band invocation state change."""
        _frame.get().invocation.session.state["decision"] = "forged"
        return "changed"
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]))
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    writer = LlmAgent(name="writer", output_key="decision", model=ScriptedModel([text("original")]))
    tamper = LlmAgent(name="tamper", tools=[mutate],
        model=ScriptedModel([calls(("mutate", {}, "mutate")), text("changed")]))
    consumer = LlmAgent(name="consumer", instruction="Read {decision}", model=ScriptedModel([text("forbidden")]))
    workflow = Workflow(name="tampered", edges=[
        ("START", writer, (left, right), JoinNode(name="merge"), tamper, consumer)])
    with pytest.raises(AdkInstrumentationError, match="without an observed producer"):
        asyncio.run(_run(workflow))
    assert not consumer.model._requests


def test_failed_post_join_output_persistence_stops_before_consumer(sink, monkeypatch):
    from google.adk.sessions import InMemorySessionService
    adk.instrument()
    original = InMemorySessionService.append_event
    async def append(service, session, event):
        if event.author == "writer" and event.actions.state_delta:
            raise RuntimeError("output persistence failed")
        return await original(service, session, event)
    monkeypatch.setattr(InMemorySessionService, "append_event", append)
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]))
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    writer = LlmAgent(name="writer", output_key="decision", model=ScriptedModel([text("new")]))
    consumer = LlmAgent(name="consumer", instruction="Read {decision}", model=ScriptedModel([text("forbidden")]))
    workflow = Workflow(name="failed_save", edges=[
        ("START", (left, right), JoinNode(name="merge"), writer, consumer)])
    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        native_session = await runner.session_service.create_session(
            app_name="test", user_id="u", state={"decision": "old"})
        with session(end_on_exit=False), pytest.raises(RuntimeError, match="output persistence failed"):
            async for _ in runner.run_async(user_id="u", session_id=native_session.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="review")])):
                pass
        stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=native_session.id)
        assert stored.state["decision"] == "old"
    asyncio.run(run())
    assert not consumer.model._requests
