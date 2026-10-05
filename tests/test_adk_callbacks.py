"""Callback provenance on real ADK runners, without provider requests."""
import asyncio

import pytest
from sasy.instrumentation import adk_callbacks
from sasy.instrumentation.otel import get_current_input_ids

from tests.test_adk_instrumentation import (  # noqa: F401
    AdkInstrumentationError,
    LlmAgent,
    LlmResponse,
    ScriptedModel,
    adk,
    calls,
    conversation,
    text,
    types,
)
from tests.test_adk_instrumentation import (
    sink as sink,
)


@pytest.fixture(autouse=True)
def install_callbacks():
    adk.instrument()
    adk_callbacks.install()


def test_model_callbacks_consume_request_then_raw_output_and_restore_scope(sink):
    seen = {}
    async def before(callback_context, llm_request):
        assert adk_callbacks.in_callback()
        seen["before"] = list(get_current_input_ids())
        await asyncio.sleep(0)
        assert get_current_input_ids() == seen["before"]
    async def after(callback_context, llm_response):
        assert adk_callbacks.in_callback()
        seen["after"] = list(get_current_input_ids())
        await asyncio.sleep(0)
        llm_response.content.parts[0].text = "rewritten output"
    model = ScriptedModel([text("raw output")])
    agent = LlmAgent(name="writer", model=model, before_model_callback=before, after_model_callback=after)
    token = adk._current_input_ids.set(["outer"])
    try:
        asyncio.run(conversation(agent))
        assert get_current_input_ids() == ["outer"]
        assert not adk_callbacks.in_callback()
    finally:
        adk._current_input_ids.reset(token)
    assert any(sink.events[node].text == "hello" for node in seen["before"])
    assert [sink.events[node].text for node in seen["after"]] == ["raw output"]
    rewritten = next(node for node, event in sink.events.items() if event.text == "rewritten output")
    assert any(edge.destination == rewritten and edge.source in seen["after"] for edge in sink.edges)
    assert agent.before_model_callback is before and agent.after_model_callback is after


def test_before_model_filter_has_original_inputs_but_provider_has_filtered_inputs(sink):
    seen = []
    def before(callback_context, llm_request):
        seen.extend(get_current_input_ids())
        llm_request.contents.clear()
    model = ScriptedModel([text("done")])
    agent = LlmAgent(name="writer", model=model, instruction="Static instruction", before_model_callback=before)
    asyncio.run(conversation(agent))
    assert any(sink.events[node].text == "hello" for node in seen)
    assert model._requests[0].contents == []
    output = next(node for node, event in sink.events.items() if event.text == "done")
    parents = [sink.events[edge.source].text for edge in sink.edges if edge.destination == output]
    assert "hello" not in parents


def test_no_callbacks_do_not_add_input_resolution(sink, monkeypatch):
    original = adk._State.inputs
    calls_seen = []
    async def inputs(self, request, agent):
        calls_seen.append(agent)
        return await original(self, request, agent)
    monkeypatch.setattr(adk._State, "inputs", inputs)
    asyncio.run(conversation(LlmAgent(name="writer", model=ScriptedModel([text("done")]))))
    assert calls_seen == ["writer"]


def test_before_model_instruction_has_request_provenance(sink):
    def before(callback_context, llm_request):
        llm_request.config.system_instruction = "invented approval"
    model = ScriptedModel([text("done")])
    asyncio.run(conversation(LlmAgent(name="writer", model=model, before_model_callback=before)))
    instruction = next(event for event in sink.events.values() if event.text == "invented approval")
    assert any(edge.destination == instruction.id for edge in sink.edges)
    assert not instruction.HasField("derived_from")
    assert not adk_callbacks.in_callback()


def test_partial_after_model_callback_rejects_before_callback_execution(sink):
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import InMemoryRunner
    from sasy.instrumentation.session import session
    seen = []
    def after(callback_context, llm_response):
        seen.append(True)
    model = ScriptedModel([LlmResponse(content=types.Content(role="model", parts=[types.Part(text="partial")] ), partial=True)])
    runner = InMemoryRunner(agent=LlmAgent(name="writer", model=model, after_model_callback=after), app_name="test")
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            async for _ in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE)):
                pass
    with pytest.raises(AdkInstrumentationError, match="partial responses"):
        asyncio.run(run())
    assert not seen
    assert not any(event.text == "partial" for event in sink.events.values())


def test_parallel_tool_callbacks_receive_own_result_and_origin(sink):
    seen_before = {}
    seen_after = {}
    async def pay(amount: int):
        """Pay a synthetic amount."""
        await asyncio.sleep(0)
        return {"paid": amount}
    async def before(tool, args, tool_context):
        assert adk_callbacks.in_callback()
        seen_before[args["amount"]] = list(get_current_input_ids())
        await asyncio.sleep(0)
        assert get_current_input_ids() == seen_before[args["amount"]]
    async def after(tool, args, tool_context, tool_response):
        assert adk_callbacks.in_callback()
        seen_after[args["amount"]] = list(get_current_input_ids())
        await asyncio.sleep(0)
        assert get_current_input_ids() == seen_after[args["amount"]]
    agent = LlmAgent(name="payer", tools=[pay], before_tool_callback=before, after_tool_callback=after,
        model=ScriptedModel([calls(("pay", {"amount": 1}, "one"), ("pay", {"amount": 2}, "two")), text("done")]))
    asyncio.run(conversation(agent))
    for amount in (1, 2):
        check = next(check for check in sink.checks if check[1] == {"amount": amount})
        assert seen_before[amount] == check[2]
        results = [sink.events[node] for node in seen_after[amount] if sink.events[node].HasField("derived_from")]
        assert len(results) == 1 and results[0].text == '{"paid":' + str(amount) + '}'
        assert set(check[2]) <= set(seen_after[amount])
    assert not adk_callbacks.in_callback()


def test_synthetic_before_tool_result_stops_before_after_callback(sink):
    effects = []
    def pay():
        """Pay."""
        effects.append("tool")
        return {"paid": True}
    def before(tool, args, tool_context):
        return {"paid": True}
    def after(tool, args, tool_context, tool_response):
        effects.append("after callback")
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "pay"))]),
        tools=[pay], before_tool_callback=before, after_tool_callback=after)
    with pytest.raises(AdkInstrumentationError, match="before_tool_callback returned a result"):
        asyncio.run(conversation(agent))
    assert not effects and not sink.checks


def test_callback_cancellation_restores_input_and_callback_scope(sink):
    entered = asyncio.Event()
    async def before(callback_context, llm_request):
        assert adk_callbacks.in_callback()
        entered.set()
        await asyncio.Event().wait()
    async def run():
        token = adk._current_input_ids.set(["outer"])
        try:
            task = asyncio.create_task(conversation(LlmAgent(name="writer", model=ScriptedModel([]), before_model_callback=before)))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert get_current_input_ids() == ["outer"] and not adk_callbacks.in_callback()
        finally:
            adk._current_input_ids.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["model", "tool"])
def test_error_callbacks_have_input_scope_and_callback_guard(sink, kind):
    seen = []
    def error_callback(**kwargs):
        seen.append((adk_callbacks.in_callback(), list(get_current_input_ids())))
    class BrokenModel(ScriptedModel):
        async def generate_content_async(self, llm_request, stream=False):
            raise ValueError("provider failed")
            yield
    def broken():
        """Fail after dispatch."""
        raise ValueError("tool failed")
    if kind == "model":
        agent = LlmAgent(name="writer", model=BrokenModel([]), on_model_error_callback=error_callback)
    else:
        agent = LlmAgent(name="writer", model=ScriptedModel([calls(("broken", {}, "fail"))]),
            tools=[broken], on_tool_error_callback=error_callback)
    with pytest.raises(ValueError, match="failed"):
        asyncio.run(conversation(agent))
    assert seen and all(flag and ids and all(node in sink.events for node in ids) for flag, ids in seen)
    assert not adk_callbacks.in_callback()
