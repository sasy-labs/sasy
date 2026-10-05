"""Native ADK single-turn/task delegation uses observed dispatch and return values."""
import asyncio

import pytest
from test_adk_instrumentation import (
    LlmAgent,
    ScriptedModel,
    calls,
    conversation,
    text,
)
from test_adk_instrumentation import sink as sink
from test_adk_otel import tracing as tracing
from test_adk_state import ancestors


def agents(mode):
    child_script = [text("child answer")] if mode == "single_turn" else [calls(("finish_task", {"result": "child answer"}, "finish"))]
    child = LlmAgent(name="child", mode=mode, model=ScriptedModel(child_script), description="Answer a delegated question.")
    parent = LlmAgent(name="parent", model=ScriptedModel([calls(("child", {"request": "question"}, "delegate")), text("done")]), sub_agents=[child])
    return parent, child


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_native_delegation_tracks_exact_child_input_and_output(sink, mode):
    parent, child = agents(mode)
    asyncio.run(conversation(parent))
    assert sink.checks[0][0] == "child"
    assert sink.checks[0][1] == {"request": "question"}
    assert len(child.model._requests) == 1
    returned = [e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child"]
    assert len(returned) == 1
    lineage = ancestors(sink, [returned[0].id])
    assert any(e.agent == "child" and (e.text == "child answer" or any(t.name == "finish_task" for t in e.tools)) for e in lineage)
    assert any(any(t.name == "child" for t in e.tools) for e in lineage)


@pytest.mark.parametrize("mode", ["single_turn", "task"])
@pytest.mark.parametrize("transform", [False, True])
def test_denied_delegation_never_launches_child(sink, mode, transform):
    sink.allowed = transform
    sink.transforms = ["unsupported"] if transform else []
    parent, child = agents(mode)
    asyncio.run(conversation(parent))
    assert not child.model._requests
    assert [check[0] for check in sink.checks] == ["child"]
    assert not any(event.HasField("derived_from") for event in sink.events.values())


def test_denied_finish_callback_cannot_forge_completed_task(sink, monkeypatch):
    from sasy.instrumentation import adk
    from sasy.instrumentation.adk import AdkInstrumentationError
    original = adk.monitor.check_tool_call_async
    async def check(name, args, ids, **kwargs):
        result = await original(name, args, ids, **kwargs)
        result.authorized = name != "finish_task"
        return result
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    async def after(tool, args, tool_context, tool_response):
        return {"result": "Task completed."}
    parent, child = agents("task")
    child.after_tool_callback = after
    with pytest.raises(AdkInstrumentationError, match="completion response changed or was not authorized"):
        asyncio.run(conversation(parent))
    assert [check[0] for check in sink.checks] == ["child", "finish_task"]
    assert not any(event.HasField("derived_from") for event in sink.events.values())


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_sibling_delegations_do_not_share_return_or_input_ancestry(sink, mode):
    children = []
    for name in ("left", "right"):
        script = [text(name + " response")] if mode == "single_turn" else [calls(("finish_task", {"result": name + " response"}, name + "-finish"))]
        children.append(LlmAgent(name=name, mode=mode, model=ScriptedModel(script), description="Handle the request."))
    parent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("left", {"request": "left input"}, "left-call"), ("right", {"request": "right input"}, "right-call")),
        text("done")]), sub_agents=children)
    asyncio.run(conversation(parent))
    for child in children:
        other = "right" if child.name == "left" else "left"
        returned = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == child.name)
        lineage = ancestors(sink, [returned.id])
        assert not any(e.agent == other and (e.text == other + " response" or e.HasField("derived_from")) for e in lineage)
        request = child.model._requests[0]
        assert not any(other + " response" in (part.text or "") for content in request.contents for part in content.parts)


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_cancelled_child_produces_no_successful_delegation(sink, mode):
    from sasy.instrumentation import adk, adk_tasks_native, dependencies
    from test_adk_otel import Model
    async def run():
        entered = asyncio.Event()
        async def generate(request, stream):
            entered.set()
            await asyncio.Event().wait()
            yield text("unreachable")
        parent, child = agents(mode)
        child.model = Model(generate)
        task = asyncio.create_task(conversation(parent))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adk_tasks_native._delegation.get() is None
        assert dependencies._resolver.get() is None
        assert adk._active.get() is None
        assert not any(event.HasField("derived_from") for event in sink.events.values())
    asyncio.run(run())


def test_invalid_task_completion_can_retry_without_false_tool_result(sink):
    parent, child = agents("task")
    child.model = ScriptedModel([
        calls(("finish_task", {"wrong": "invalid"}, "invalid")),
        calls(("finish_task", {"result": "valid answer"}, "valid")),
    ])
    asyncio.run(conversation(parent))
    completions = [e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "finish_task"]
    assert len(completions) == 1
    assert completions[0].derived_from.arguments == '{"result":"valid answer"}'
    returned = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child")
    assert "valid answer" in returned.text


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_sequential_calls_to_same_child_keep_distinct_completion(sink, mode):
    parent, child = agents(mode)
    child.model = ScriptedModel([text("first"), text("second")] if mode == "single_turn" else [
        calls(("finish_task", {"result": "first"}, "finish-first")),
        calls(("finish_task", {"result": "second"}, "finish-second")),
    ])
    parent.model = ScriptedModel([
        calls(("child", {"request": "first input"}, "first-call")),
        calls(("child", {"request": "second input"}, "second-call")),
        text("done"),
    ])
    asyncio.run(conversation(parent))
    returned = [e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child"]
    assert len(returned) == 2
    assert "first" in returned[0].text and "second" in returned[1].text
    assert len(child.model._requests) == 2
    assert not any("first" in (p.text or "") for c in child.model._requests[1].contents for p in c.parts)


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_child_failure_cannot_become_successful_delegation(sink, mode):
    from test_adk_otel import Model
    async def generate(request, stream):
        raise RuntimeError("synthetic child failure")
        yield text("unreachable")
    parent, child = agents(mode)
    child.model = Model(generate)
    with pytest.raises(Exception):
        asyncio.run(conversation(parent))
    assert not any(event.HasField("derived_from") for event in sink.events.values())


def test_deferred_task_span_has_its_own_return_link(sink, tracing):
    parent, _ = agents("task")
    asyncio.run(conversation(parent))
    spans = tracing.memory.get_finished_spans()
    task = next(span for span in spans if span.name == "execute_task child")
    assert task.attributes["sasy.operation"] == "tool"
    assert task.attributes["sasy.dispatch.outcome"] == "completed"
    assert task.attributes["sasy.rm.authorized"] is True
    assert task.attributes["_output_message_id"] in sink.events
    assert sink.events[task.attributes["_output_message_id"]].derived_from.name == "child"


def test_task_delegation_and_regular_tool_in_one_turn(sink):
    parent, child = agents("task")
    def ordinary():
        """An ordinary function tool."""
        return "ordinary return"
    from google.adk.tools.function_tool import FunctionTool
    parent.tools.append(FunctionTool(ordinary))
    parent.model = ScriptedModel([
        calls(("ordinary", {}, "ordinary"), ("child", {"request": "question"}, "delegate")),
        text("done"),
    ])
    asyncio.run(conversation(parent))
    assert [check[0] for check in sink.checks] == ["ordinary", "child", "finish_task"]
    assert len(child.model._requests) == 1


def test_single_turn_child_input_includes_state_read_by_before_tool_callback(sink):
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from sasy.instrumentation import adk
    from sasy.instrumentation.session import session
    parent, _ = agents("single_turn")
    async def before(tool, args, tool_context):
        args["request"] = tool_context.state["question"]
    parent.before_tool_callback = before
    async def run():
        runner = InMemoryRunner(agent=parent, app_name="test")
        conversation = await runner.session_service.create_session(app_name="test", user_id="u", state={"question": "state-derived input"})
        adk.instrument()
        with session(end_on_exit=False):
            async for _ in runner.run_async(user_id="u", session_id=conversation.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="request")])):
                pass
    asyncio.run(run())
    assert sink.checks[0][1] == {"request": "state-derived input"}
    returned = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child")
    assert any('"state"' in e.text and 'state-derived input' in e.text for e in ancestors(sink, [returned.id]))


@pytest.mark.parametrize("allowed", [True, False])
def test_raw_function_in_mixed_task_turn_is_dispatched_and_authorized(sink, monkeypatch, allowed):
    from sasy.instrumentation import adk
    effects = []
    def ordinary(value: int = 7):
        """An ordinary Python function."""
        effects.append(value)
        return "ordinary return"
    original = adk.monitor.check_tool_call_async
    async def check(name, args, ids, **kwargs):
        result = await original(name, args, ids, **kwargs)
        if name == "ordinary":
            result.authorized = allowed
        return result
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    parent, child = agents("task")
    parent.tools.append(ordinary)
    parent.model = ScriptedModel([
        calls(("ordinary", {}, "ordinary"), ("child", {"request": "question"}, "delegate")),
        text("done"),
    ])
    asyncio.run(conversation(parent))
    assert [c[0] for c in sink.checks] == ["ordinary", "child", "finish_task"]
    assert sink.checks[0][1] == {"value": 7}
    assert effects == ([7] if allowed else [])
    assert len(child.model._requests) == 1
    assert any(e.HasField("derived_from") and e.derived_from.name == "ordinary" for e in sink.events.values()) == allowed


@pytest.mark.parametrize("mode", ["single_turn", "task"])
def test_native_target_mutation_during_authorization_never_launches_child(sink, monkeypatch, mode):
    from sasy.instrumentation import adk, adk_tasks_native
    from sasy.instrumentation.adk import AdkInstrumentationError
    parent, child = agents(mode)
    replacement = LlmAgent(name="replacement", mode=mode, model=ScriptedModel([text("must not run")]))
    original = adk.monitor.check_tool_call_async
    async def check(name, args, ids, **kwargs):
        result = await original(name, args, ids, **kwargs)
        if name == "child":
            parent.tools[0].agent = replacement
        return result
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    with pytest.raises(AdkInstrumentationError, match="exact configured LlmAgent|changed during authorization"):
        asyncio.run(conversation(parent))
    assert not child.model._requests and not replacement.model._requests
    assert adk_tasks_native._delegation.get() is None


def test_finish_target_mutation_during_authorization_cannot_complete(sink, monkeypatch):
    from sasy.instrumentation import adk
    from sasy.instrumentation.adk import AdkInstrumentationError
    parent, child = agents("task")
    original = adk.monitor.check_tool_call_async
    async def check(name, args, ids, **kwargs):
        result = await original(name, args, ids, **kwargs)
        if name == "finish_task":
            child.tools[0]._wrapper_key = "replacement"
        return result
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    with pytest.raises(AdkInstrumentationError, match="completion target changed"):
        asyncio.run(conversation(parent))
    assert not any(e.HasField("derived_from") for e in sink.events.values())


def test_a_delegation_origin_tells_a_tuple_from_the_list_it_dumps_as(sink, monkeypatch):
    """The dispatched call is compared with the observed one on the canonical record.

    A JSON dump writes a tuple and the list it holds alike, so a guard built on
    one would accept a call whose arguments are no longer the observed ones.
    """
    from types import SimpleNamespace

    from google.adk.events import Event as AdkEvent
    from google.genai import types
    from sasy.instrumentation import adk, adk_tasks_native
    from sasy.instrumentation.adk import AdkInstrumentationError

    observed = types.FunctionCall(name="child", id="delegate", args={"k": [1, 2]})
    event = AdkEvent(author="parent", content=types.Content(
        role="model", parts=[types.Part(function_call=observed)]))
    record = SimpleNamespace(event=event, ids=[], key="")
    monkeypatch.setattr(adk, "_state", lambda: SimpleNamespace(records={"e": record}))
    assert adk_tasks_native._origin("parent", observed.model_copy(deep=True))[1] is record
    swapped = observed.model_copy(deep=True)
    swapped.args = {"k": (1, 2)}
    assert swapped.model_dump(mode="json") == observed.model_dump(mode="json")
    with pytest.raises(AdkInstrumentationError, match="exact observed function call"):
        adk_tasks_native._origin("parent", swapped)
