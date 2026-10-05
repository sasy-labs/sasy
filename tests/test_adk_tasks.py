"""Application orchestration explicitly imports only consumed child responses."""
import asyncio
from uuid import uuid4

import grpc
import pytest
from sasy.instrumentation import adk_tasks
from sasy.proto.observability_pb2 import Edge, Event, EventSnapshot, Role
from test_adk_instrumentation import (
    AdkInstrumentationError,
    LlmAgent,
    ScriptedModel,
    adk,
    calls,
    conversation,
    text,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


@pytest.mark.parametrize("sync", [True, False])
def test_explicit_child_response_enters_only_consuming_computation(sink, monkeypatch, sync):
    def resolve_sync(snapshots):
        for item in snapshots:
            stored = Event.FromString(sink.events[item.base_id].SerializeToString())
            stored.id = item.event.id
            assert stored == item.event and item.reuse_dependencies
        return [item.base_id for item in snapshots]
    monkeypatch.setattr(adk.observation, "resolve_events", resolve_sync)
    async def child(value):
        event = Event(id=str(uuid4()), text=value, role=Role.AGENT, agent="application-task")
        node, = await adk.observation.resolve_events_async([EventSnapshot(event=event)])
        return EventSnapshot(event=event, base_id=node, reuse_dependencies=True)

    async def delegate():
        """Consume one of two independently registered child responses."""
        selected, unused = await asyncio.gather(child("selected response"), child("unused response"))
        assert selected.base_id != unused.base_id
        consumed = adk_tasks.consume_events([selected]) if sync else await adk_tasks.consume_events_async([selected])
        assert consumed == [selected.base_id]
        return {"response": selected.event.text}

    agent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("delegate", {}, "delegate")), text("finished")]), tools=[delegate])
    asyncio.run(conversation(agent))
    output = next(event for event in sink.events.values() if event.HasField("derived_from"))
    lineage = ancestors(sink, [output.id])
    assert any(event.text == "selected response" for event in lineage)
    assert not any(event.text == "unused response" for event in lineage)


@pytest.mark.parametrize("invalid", ["new", "dependency", "changed", "unknown"])
def test_unverified_response_cannot_complete_a_tool(sink, invalid):
    async def delegate():
        """Attempt to consume an invalid response receipt."""
        event = Event(id="child", text="registered", role=Role.AGENT)
        node, = await adk.observation.resolve_events_async([EventSnapshot(event=event)])
        reference = EventSnapshot(event=event, base_id=node, reuse_dependencies=True)
        if invalid == "new":
            reference.ClearField("base_id")
            reference.reuse_dependencies = False
        elif invalid == "dependency":
            reference.dependencies.append(Edge(source=node, destination="child"))
        elif invalid == "changed":
            reference.event.text = "changed after registration"
        else:
            reference.base_id = "sasy:mv1:unknown"
        await adk_tasks.consume_events_async([reference])
        return "must not complete"

    agent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("delegate", {}, "delegate")), text("finished")]), tools=[delegate])
    with pytest.raises((AdkInstrumentationError, grpc.RpcError)):
        asyncio.run(conversation(agent))
    assert not any(event.HasField("derived_from") for event in sink.events.values())


def test_inherited_task_cannot_consume_in_parent_scope(sink):
    async def delegate():
        """Attempt to modify the parent's dependency set from a child task."""
        event = Event(id="child", text="registered")
        node, = await adk.observation.resolve_events_async([EventSnapshot(event=event)])
        with pytest.raises(AdkInstrumentationError, match="child task"):
            await asyncio.create_task(adk_tasks.consume_events_async([
                EventSnapshot(event=event, base_id=node, reuse_dependencies=True)]))
        return "no response consumed"

    agent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("delegate", {}, "delegate")), text("finished")]), tools=[delegate])
    asyncio.run(conversation(agent))
    output = next(event for event in sink.events.values() if event.HasField("derived_from"))
    assert not any(event.text == "registered" for event in ancestors(sink, [output.id]))


def test_response_consumption_requires_an_active_scope():
    with pytest.raises(AdkInstrumentationError, match="observed ADK computation"):
        adk_tasks.consume_events([])
