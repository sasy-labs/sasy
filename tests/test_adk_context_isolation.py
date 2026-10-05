"""Each concurrent ADK tool keeps its own initiating-message context."""
import asyncio

import pytest
from sasy.instrumentation.otel import _current_input_ids, get_current_input_ids
from test_adk_instrumentation import ScriptedModel, calls, conversation, text
from test_adk_instrumentation import sink as _sink


@pytest.fixture
def sink(monkeypatch):
    return _sink.__wrapped__(monkeypatch)


def test_parallel_agents_keep_tool_origins_across_await(sink):
    from google.adk.agents import LlmAgent, ParallelAgent

    async def run():
        entered = 0
        ready = asyncio.Event()
        observed = {}

        async def act(marker: str):
            """Inspect context on either side of an interleaved tool invocation."""
            nonlocal entered
            before = list(get_current_input_ids())
            assert before
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
            after = list(get_current_input_ids())
            assert before == after
            assert {sink.events[node].agent for node in before} == {marker}
            assert all(tool.name == "act" for node in before for tool in sink.events[node].tools)
            observed[marker] = before
            return {"marker": marker}

        agents = [LlmAgent(name=marker, model=ScriptedModel([
            calls(("act", {"marker": marker}, "shared-call-id")), text(marker + " finished")]),
            tools=[act]) for marker in ("left", "right")]
        token = _current_input_ids.set(["outer-context"])
        try:
            await conversation(ParallelAgent(name="parallel", sub_agents=agents))
            assert get_current_input_ids() == ["outer-context"]
        finally:
            _current_input_ids.reset(token)
        assert set(observed) == {"left", "right"}
        assert set(observed["left"]).isdisjoint(observed["right"])
        for name, args, ids, _ in sink.checks:
            assert name == "act"
            assert ids == observed[args["marker"]]
    asyncio.run(run())


def test_tool_callbacks_receive_origin_and_result_context(sink):
    from google.adk.agents import LlmAgent

    seen = {}
    def before(tool, args, tool_context):
        seen["before"] = list(get_current_input_ids())
    def consume():
        """Produce an observed tool result."""
        seen["tool"] = list(get_current_input_ids())
        return {"consumed": True}
    def after(tool, args, tool_context, tool_response):
        seen["after"] = list(get_current_input_ids())
        assert tool_response == {"consumed": True}
    agent = LlmAgent(name="worker", model=ScriptedModel([
        calls(("consume", {}, "consume-1")), text("finished")]), tools=[consume],
        before_tool_callback=before, after_tool_callback=after)
    asyncio.run(conversation(agent))
    assert seen["before"] == seen["tool"] == sink.checks[0][2]
    assert set(seen["tool"]) <= set(seen["after"])
    assert any(sink.events[node].HasField("derived_from")
               and sink.events[node].derived_from.name == "consume" for node in seen["after"])
    assert not get_current_input_ids()
