"""Native ADK occurrence identity and exact delegated-response dependencies."""
import asyncio
import os

import pytest

if os.environ.get("SASY_REQUIRE_FRAMEWORKS") == "1":
    import google.adk  # noqa: F401
else:
    pytest.importorskip("google.adk")
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from test_adk_instrumentation import (
    AdkInstrumentationError,
    LlmAgent,
    ParallelAgent,
    ScriptedModel,
    SequentialAgent,
    adk,
    calls,
    conversation,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


def test_repeated_identical_native_messages_keep_distinct_origins(sink):
    def echo():
        """Return the same result."""
        return "same"
    async def run():
        model = ScriptedModel([calls(("echo", {}, "first")), text("same"),
                               calls(("echo", {}, "second")), text("same")])
        runner = InMemoryRunner(agent=LlmAgent(name="worker", model=model, tools=[echo]), app_name="test")
        guarded = adk.instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            events = []
            for _ in range(2):
                events.append([e async for e in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="same")]))])
            assert len(model._requests) == 4
            ids = [guarded.event_ids(turn[-1].id, user_id="u", session_id=sess.id) for turn in events]
            assert ids[0] != ids[1]
    asyncio.run(run())
    assert len(sink.checks) == 2


def test_parallel_fan_in_filters_unconsumed_sibling_approval(sink):
    def approve():
        """Approve."""
        return "approved"
    def consume():
        """Consume."""
        return "used"
    def only_plain(callback_context, llm_request):
        llm_request.contents = [c for c in llm_request.contents
            if any("[plain] said:" in (p.text or "") for p in c.parts or [])]
    approved = LlmAgent(name="approved", model=ScriptedModel([calls(("approve", {}, "a")), text("same")]), tools=[approve])
    plain = LlmAgent(name="plain", model=ScriptedModel([text("same")]))
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "c")), text("answer")]),
        instruction="Read the selected reply.", tools=[consume], before_model_callback=only_plain)
    asyncio.run(conversation(SequentialAgent(name="sequence", sub_agents=[
        ParallelAgent(name="parallel", sub_agents=[approved, plain]), reader])))
    inputs = next(check[2] for check in sink.checks if check[0] == "consume")
    lineage = ancestors(sink, inputs)
    assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in sink.events.values())
    assert not any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage)
    assert any("[plain] said:" in e.text for e in lineage)
    assert not any("[approved]" in e.text for e in lineage)


@pytest.mark.parametrize("selected_approved", [False, True])
def test_agent_tool_result_depends_on_selected_child_response(sink, selected_approved):
    def approve():
        """Approve."""
        return "approved"
    def only_initial_user(callback_context, llm_request):
        llm_request.contents = [c for c in llm_request.contents
            if any(p.text == "child request" for p in c.parts or [])]
    first = LlmAgent(name="first", model=ScriptedModel([calls(("approve", {}, "approve")), text("same")]), tools=[approve])
    last = LlmAgent(name="last", model=ScriptedModel([text("same")]), instruction="Answer.",
        before_model_callback=None if selected_approved else only_initial_user)
    worker = SequentialAgent(name="delegate", sub_agents=[first, last])
    parent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("delegate", {"request": "child request"}, "call")), text("parent answer")]), tools=[AgentTool(worker)])
    asyncio.run(conversation(parent))
    result = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "delegate")
    lineage = ancestors(sink, [result.id])
    assert any(e.text == "child request" for e in lineage)
    assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == selected_approved
    last_outputs = [e for e in lineage if e.agent == "last" and e.text == "same"]
    assert len(last_outputs) == 1
    initial = next(e for e in sink.events.values() if e.text == "child request")
    assert any(e.tools and e.tools[0].name == "delegate" for e in ancestors(sink, [initial.id]))


@pytest.mark.parametrize("transform", [False, True])
def test_denied_agent_tool_never_starts_child(sink, transform):
    child_model = ScriptedModel([text("must not run")])
    child = LlmAgent(name="child", model=child_model)
    parent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("child", {"request": "request"}, "call")), text("done")]), tools=[AgentTool(child)])
    sink.allowed = transform
    sink.transforms = ["unsupported"] if transform else []
    asyncio.run(conversation(parent))
    assert not child_model._requests
    assert not any(e.HasField("derived_from") for e in sink.events.values())


def test_parallel_agent_tool_calls_keep_selected_results_separate(sink):
    child = LlmAgent(name="child", model=ScriptedModel([text("same"), text("same")]))
    parent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("child", {"request": "request one"}, "one"), ("child", {"request": "request two"}, "two")),
        text("done")]), tools=[AgentTool(child)])
    asyncio.run(conversation(parent))
    results = [e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child"]
    assert len(results) == 2
    for result in results:
        lineage = ancestors(sink, [result.id])
        own = "request one" if "request one" in result.derived_from.arguments else "request two"
        other = "request two" if own == "request one" else "request one"
        assert any(e.text == own for e in lineage)
        assert not any(e.text == other for e in lineage)
        assert len([e for e in lineage if e.agent == "child" and e.text == "same"]) == 1


def test_agent_tool_copied_unread_state_is_not_an_input(sink):
    async def run():
        child_model = ScriptedModel([text("child result")])
        agent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "request"}, "call")), text("done")]),
            tools=[AgentTool(LlmAgent(name="child", model=child_model))])
        runner = InMemoryRunner(agent=agent, app_name="test")
        adk.instrument()
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"hidden": "unread approval"})
        with session(end_on_exit=False):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        assert len(child_model._requests) == 1
    asyncio.run(run())
    assert not any("unread approval" in event.text for event in sink.events.values())


def test_unqualified_agent_tool_schema_rejects_at_setup(sink):
    from pydantic import BaseModel
    class Answer(BaseModel):
        answer: str
    child = LlmAgent(name="child", model=ScriptedModel([]), output_schema=Answer)
    with pytest.raises(AdkInstrumentationError, match="text schemas"):
        adk.instrument_adk(InMemoryRunner(agent=LlmAgent(name="parent", model=ScriptedModel([]),
            tools=[AgentTool(child)]), app_name="test"))
    assert not sink.events


def test_agent_tool_interruption_without_response_has_no_success(sink):
    from sasy.instrumentation import adk_agents, adk_state
    class Cancelled(ScriptedModel):
        async def generate_content_async(self, llm_request, stream=False):
            raise asyncio.CancelledError()
            yield  # pragma: no cover
    child = LlmAgent(name="child", model=Cancelled([]))
    parent = LlmAgent(name="parent", model=ScriptedModel([
        calls(("child", {"request": "request"}, "call"))]), tools=[AgentTool(child)])
    # ADK treats an internally cancelled child as an empty completed run.
    # The adapter must not turn that into a successful delegation result.
    with pytest.raises(AdkInstrumentationError, match="without an observed child response"):
        asyncio.run(conversation(parent))
    assert adk_agents._delegation.get() is None
    assert adk_state._frame.get() is None
    assert not any(e.HasField("derived_from") and e.derived_from.name == "child" for e in sink.events.values())


def test_agent_tool_reads_forwarded_artifact_without_parent_read_union(sink):
    async def run():
        child = LlmAgent(name="child", model=ScriptedModel([text("child answer")]),
            instruction="Read {artifact.selected}")
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "read the artifact"}, "call")), text("parent answer")]), tools=[AgentTool(child)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        for name, value in [("selected", "selected source"), ("unread", "unread source")]:
            await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename=name, artifact=types.Part(text=value))
        adk.instrument()
        with session(end_on_exit=False):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        result = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "child")
        assert any("selected source" in e.text for e in ancestors(sink, [result.id]))
        assert not any("unread source" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_agent_tool_native_load_preserves_selected_forwarded_body(sink):
    from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
    async def run():
        child_model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["selected"]}, "load")), text("child answer")])
        child = LlmAgent(name="child", model=child_model, tools=[LoadArtifactsTool()])
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "read selected"}, "call")), text("parent answer")]), tools=[AgentTool(child)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="selected", artifact=types.Part(text="forwarded body"))
        adk.instrument()
        with session(end_on_exit=False):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        assert [c[0] for c in sink.checks] == ["child", "load_artifacts"]
        assert any("forwarded body" in (p.text or "") for c in child_model._requests[1].contents for p in c.parts or [])
        child_output = next(e for e in sink.events.values() if e.text == "child answer")
        assert any("forwarded body" in e.text for e in ancestors(sink, [child_output.id]))
    asyncio.run(run())


def test_nested_agent_tools_forward_artifacts_through_exact_parent_chain(sink):
    async def run():
        leaf = LlmAgent(name="leaf", model=ScriptedModel([text("leaf answer")]), instruction="Read {artifact.selected}")
        middle = LlmAgent(name="middle", model=ScriptedModel([
            calls(("leaf", {"request": "leaf request"}, "inner")), text("middle answer")]), tools=[AgentTool(leaf)])
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("middle", {"request": "middle request"}, "outer")), text("parent answer")]), tools=[AgentTool(middle)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="selected", artifact=types.Part(text="forwarded twice"))
        adk.instrument()
        with session(end_on_exit=False):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        final = next(e for e in sink.events.values() if e.text == "parent answer")
        assert any("forwarded twice" in e.text for e in ancestors(sink, [final.id]))
    asyncio.run(run())


def test_agent_tool_forwarded_write_rejects_before_storage(sink):
    from google.adk.tools.tool_context import ToolContext
    async def tool(tool_context: ToolContext):
        """Attempt a forwarded write."""
        await tool_context.save_artifact("forbidden", types.Part(text="unobserved write"))
        return "done"
    async def run():
        child_model = ScriptedModel([calls(("tool", {}, "write"))])
        child = LlmAgent(name="child", model=child_model, tools=[tool])
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "request"}, "call"))]), tools=[AgentTool(child)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="through the forwarding service"):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        assert len(child_model._requests) == 1
        assert await runner.artifact_service.load_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="forbidden") is None
    asyncio.run(run())


def test_agent_tool_forwarding_override_rejects_before_unobserved_prompt(sink):
    async def run():
        def ping():
            """Continue."""
            return "ok"
        async def shadow(tool, args, tool_context):
            async def hidden(**kwargs):
                return types.Part(text="unobserved replacement")
            tool_context._invocation_context.artifact_service.load_artifact = hidden
        child_model = ScriptedModel([calls(("ping", {}, "ping")), text("must not run")])
        child = LlmAgent(name="child", model=child_model, instruction="{artifact.source}", tools=[ping], before_tool_callback=shadow)
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "request"}, "call"))]), tools=[AgentTool(child)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="source", artifact=types.Part(text="observed original"))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="forwarding overrides"):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        assert len(child_model._requests) == 1
        assert not any("unobserved replacement" in e.text for e in sink.events.values())
    asyncio.run(run())


@pytest.mark.parametrize("selected", ["approved", "other"])
def test_forwarded_artifact_keeps_only_its_actual_producer(sink, selected):
    async def run():
        runner = None
        sess = None
        def approve():
            """Approve."""
            return "approved"
        async def save_approved():
            """Save the approved artifact."""
            return await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename="approved", artifact=types.Part(text="saved approval"))
        def only_request(callback_context, llm_request):
            llm_request.contents = [c for c in llm_request.contents if any(p.text == "request" for p in c.parts or [])]
        writer = LlmAgent(name="writer", model=ScriptedModel([
            calls(("approve", {}, "approve")), calls(("save_approved", {}, "save")), text("saved")]),
            tools=[approve, save_approved])
        child = LlmAgent(name="child", model=ScriptedModel([text("child result")]),
            instruction="Read {artifact." + selected + "}")
        caller = LlmAgent(name="caller", model=ScriptedModel([
            calls(("child", {"request": "child request"}, "call")), text("caller result")]),
            tools=[AgentTool(child)], before_model_callback=only_request)
        runner = InMemoryRunner(agent=SequentialAgent(name="sequence", sub_agents=[writer, caller]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        for filename in ("approved", "other"):
            await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename=filename, artifact=types.Part(text="initial"))
        adk.instrument()
        with session(end_on_exit=False):
            _ = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        child_output = next(e for e in sink.events.values() if e.text == "child result")
        lineage = ancestors(sink, [child_output.id])
        assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == (selected == "approved")
    asyncio.run(run())


def test_native_response_merge_tracks_only_surviving_occurrences(sink):
    from google.adk.events import Event
    from google.adk.flows.llm_flows import _tool_call_rearranger
    from sasy.instrumentation import adk_agents
    adk.instrument()
    def response(call_id, value):
        return Event(author="parent", content=types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(id=call_id, name="child", response={"result": value}))]))
    stale, other, latest = response("one", "same"), response("two", "same"), response("one", "same")
    build = adk_agents.Build(None, "parent")
    for event, node in ((stale, "old"), (other, "sibling"), (latest, "new")):
        build.add_source(event.content, [node])
    token = adk_agents._build.set(build)
    try:
        with session(end_on_exit=False):
            merged = _tool_call_rearranger.merge_function_response_events([stale, other, latest])
        assert build.source(merged.content) == ["new", "sibling"]
        assert [part.function_response.id for part in merged.content.parts] == ["one", "two"]
    finally:
        adk_agents._build.reset(token)


@pytest.mark.parametrize("scope", ["", "app:", "user:"])
@pytest.mark.parametrize("selected", ["approved", "other"])
@pytest.mark.parametrize("read_kind", ["template", "context"])
def test_copied_state_keeps_only_actual_key_producer(sink, scope, selected, read_kind):
    from google.adk.tools.tool_context import ToolContext
    def approve():
        """Approve the writer's output."""
        return "approved"
    def only_initial(callback_context, llm_request):
        llm_request.contents = [content for content in llm_request.contents
            if any(part.text == "start" for part in content.parts or [])]
    def read(tool_context: ToolContext):
        """Read the selected copied key."""
        return tool_context.state[scope + selected]
    async def run():
        writer = LlmAgent(name="writer", model=ScriptedModel([calls(("approve", {}, "approve")), text("same value")]),
            tools=[approve], output_key=scope + "approved")
        child_model = ScriptedModel([text("reply")] if read_kind == "template" else [calls(("read", {}, "read")), text("reply")])
        child = LlmAgent(name="child", model=child_model,
            instruction="{" + scope + selected + "}" if read_kind == "template" else "Read the selected key.",
            tools=[read] if read_kind == "context" else [])
        caller = LlmAgent(name="caller", model=ScriptedModel([
            calls(("child", {"request": "request"}, "delegate")), text("done")]),
            tools=[AgentTool(child)], before_model_callback=only_initial)
        runner = InMemoryRunner(agent=SequentialAgent(name="sequence", sub_agents=[writer, caller]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u",
            state={scope + "approved": "initial", scope + "other": "same value"})
        adk.instrument()
        with session(end_on_exit=False):
            _ = [event async for event in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="start")]))]
        if read_kind == "template":
            assert "same value" in child_model._requests[0].config.system_instruction
    asyncio.run(run())
    result = next(event for event in sink.events.values() if event.HasField("derived_from") and event.derived_from.name == "child")
    lineage = ancestors(sink, [result.id])
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage) == (selected == "approved")


def test_copied_state_write_reaches_the_parent_only_when_the_child_finishes(sink):
    from google.adk.tools.tool_context import ToolContext
    def write(tool_context: ToolContext):
        """Write copied child state."""
        tool_context.state["key"] = "changed"
        return "written"
    async def run():
        child = LlmAgent(name="child", model=ScriptedModel([
            calls(("write", {}, "write")), text("child answer")]), tools=[write])
        parent = LlmAgent(name="parent", model=ScriptedModel([
            calls(("child", {"request": "request"}, "delegate")), text("done")]), tools=[AgentTool(child)])
        runner = InMemoryRunner(agent=parent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"key": "original"})
        adk.instrument()
        with session(end_on_exit=False):
            events = [event async for event in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="start")]))]
        # The child's own events never change the parent session; the parent's
        # completed delegation carries the whole forwarded change.
        assert [event.actions.state_delta for event in events if event.actions.state_delta] == [{"key": "changed"}]
        stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id)
        assert stored.state["key"] == "changed"
    asyncio.run(run())


def test_native_agent_tool_does_not_copy_temp_state(sink):
    async def run():
        # Native session creation discards temp state when constructing the
        # isolated child session, even though AgentTool transports the dictionary.
        writer = LlmAgent(name="writer", model=ScriptedModel([text("temporary")]), output_key="temp:key")
        model = ScriptedModel([text("reply")])
        child = LlmAgent(name="child", model=model, instruction="value={temp:key?}")
        caller = LlmAgent(name="caller", model=ScriptedModel([
            calls(("child", {"request": "request"}, "delegate")), text("done")]), tools=[AgentTool(child)])
        await conversation(SequentialAgent(name="sequence", sub_agents=[writer, caller]))
        assert "value=" in model._requests[0].config.system_instruction
        assert "temporary" not in model._requests[0].config.system_instruction
    asyncio.run(run())


def test_nested_copy_transports_unread_key_producer(sink):
    def approve():
        """Approve the produced state."""
        return "approved"
    def only_initial(callback_context, llm_request):
        llm_request.contents = [content for content in llm_request.contents
            if any(part.text == "start" for part in content.parts or [])]
    writer = LlmAgent(name="writer", model=ScriptedModel([
        calls(("approve", {}, "approve")), text("copied value")]), tools=[approve], output_key="key")
    leaf = LlmAgent(name="leaf", model=ScriptedModel([text("answer")]), instruction="{key}")
    middle = LlmAgent(name="middle", model=ScriptedModel([
        calls(("leaf", {"request": "leaf request"}, "leaf-call")), text("middle answer")]), tools=[AgentTool(leaf)])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("middle", {"request": "middle request"}, "middle-call")), text("done")]),
        tools=[AgentTool(middle)], before_model_callback=only_initial)
    asyncio.run(conversation(SequentialAgent(name="sequence", sub_agents=[writer, caller])))
    result = next(event for event in sink.events.values() if event.HasField("derived_from") and event.derived_from.name == "leaf")
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in ancestors(sink, [result.id]))


def test_explicit_callback_derivation_replaces_native_occurrence(sink):
    def rewrite(callback_context, llm_request):
        llm_request.contents[0].parts[0].text += " " + callback_context.state["suffix"]
    async def run():
        model = ScriptedModel([text("reply")])
        runner = InMemoryRunner(agent=LlmAgent(name="agent", model=model, before_model_callback=rewrite), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"suffix": "observed edit"})
        adk.instrument()
        with session(end_on_exit=False):
            _ = [event async for event in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
        assert model._requests[0].contents[0].parts[0].text == "request observed edit"
    asyncio.run(run())
    result = next(event for event in sink.events.values() if event.text == "reply")
    lineage = ancestors(sink, [result.id])
    assert any(event.text == "request" for event in lineage)
    assert any('"suffix"' in event.text and '"observed edit"' in event.text for event in lineage)


def isolated(callback_context, llm_request):
    llm_request.contents = []


def consume():
    """Consume the decision."""
    return "consumed"


def reader_agent(instruction="Decision {decision}"):
    return LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction=instruction, before_model_callback=isolated)


@pytest.mark.parametrize("success", [True, False])
def test_forwarded_child_write_keeps_the_child_producer(sink, success):
    def approve(tool_context):
        """Record a decision."""
        tool_context.state["decision"] = "approved"
        return "approved" if success else {"error": "approval failed"}
    child = LlmAgent(name="child", model=ScriptedModel([
        calls(("approve", {}, "approve")), text("child answer")]), tools=[approve])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("child", {"request": "request"}, "delegate")), text("done")]), tools=[AgentTool(child)])
    _, _, sess = asyncio.run(conversation(SequentialAgent(name="pipeline", sub_agents=[caller, reader_agent()])))
    check = next(check for check in sink.checks if check[0] == "consume")
    lineage = ancestors(sink, check[2])
    assert any(sess.id in event.text and '"decision"' in event.text and "approved" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve"
               for event in lineage) == success
    # The delegation's own result is not a producer of what the child wrote.
    assert not any(event.HasField("derived_from") and event.derived_from.name == "child" for event in lineage)


def test_unread_forwarded_key_is_not_a_parent_dependency(sink):
    def approve(tool_context):
        """Record a decision."""
        tool_context.state["decision"] = "approved"
        return "approved"
    def finish():
        """Finish the turn."""
        return "finished"
    child = LlmAgent(name="child", model=ScriptedModel([
        calls(("approve", {}, "approve")), text("child answer")]), tools=[approve])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("child", {"request": "request"}, "delegate")),
        calls(("finish", {}, "finish")), text("done")]), tools=[AgentTool(child), finish])
    _, _, sess = asyncio.run(conversation(caller))
    check = next(check for check in sink.checks if check[0] == "finish")
    forwarded = [event for event in sink.events.values() if sess.id in event.text and '"decision"' in event.text]
    lineage = {event.id for event in ancestors(sink, check[2])}
    assert any("approved" in event.text for event in forwarded)
    assert not any(event.id in lineage for event in forwarded)


def test_nested_forwarded_output_key_reaches_a_later_reader(sink):
    def approve():
        """Approve the decision."""
        return "approved"
    leaf = LlmAgent(name="leaf", model=ScriptedModel([
        calls(("approve", {}, "approve")), text("leaf decision")]), tools=[approve], output_key="decision")
    middle = LlmAgent(name="middle", model=ScriptedModel([
        calls(("leaf", {"request": "leaf request"}, "leaf-call")), text("middle answer")]), tools=[AgentTool(leaf)])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("middle", {"request": "middle request"}, "middle-call")), text("done")]), tools=[AgentTool(middle)])
    asyncio.run(conversation(SequentialAgent(name="pipeline", sub_agents=[caller, reader_agent()])))
    check = next(check for check in sink.checks if check[0] == "consume")
    lineage = ancestors(sink, check[2])
    assert any('"decision"' in event.text and "leaf decision" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)


def test_forwarded_delta_must_match_the_observed_child_write(sink):
    from sasy.instrumentation import adk_agents
    def approve(tool_context):
        """Record a decision."""
        tool_context.state["decision"] = "approved"
        return "approved"
    def inspect(tool_context):
        """Check what the forwarded delta may claim."""
        delegation = adk_agents._delegation.get()
        with pytest.raises(AdkInstrumentationError, match="does not match the observed child write"):
            adk_agents._forwarded_write(delegation, "decision", "tampered")
        with pytest.raises(AdkInstrumentationError, match="no observed child write"):
            adk_agents._forwarded_write(delegation, "invented", "approved")
        with pytest.raises(AdkInstrumentationError, match="temporary or internal"):
            adk_agents._forwarded_write(delegation, "temp:decision", "approved")
        assert adk_agents._forwarded_write(delegation, "decision", "approved")[1]
        return "checked"
    child = LlmAgent(name="child", model=ScriptedModel([
        calls(("approve", {}, "approve")), calls(("inspect", {}, "inspect")), text("child answer")]),
        tools=[approve, inspect])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("child", {"request": "request"}, "delegate")), text("done")]), tools=[AgentTool(child)])
    asyncio.run(conversation(caller))
    assert any(check[0] == "inspect" for check in sink.checks)


def test_forwarded_internal_adk_state_is_rejected(sink):
    def note(tool_context):
        """Write an ADK internal key."""
        tool_context.state["_adk_note"] = "value"
        return "written"
    child = LlmAgent(name="child", model=ScriptedModel([
        calls(("note", {}, "note")), text("child answer")]), tools=[note])
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("child", {"request": "request"}, "delegate")), text("done")]), tools=[AgentTool(child)])
    with pytest.raises(AdkInstrumentationError, match="temporary or internal"):
        asyncio.run(conversation(caller))


def _writing_child(name, key, value, tool):
    def approve(tool_context):
        """Record a decision."""
        tool_context.state[key] = value
        return "approved"
    approve.__name__ = tool
    return LlmAgent(name=name, model=ScriptedModel([
        calls((tool, {}, tool)), text(name + " answer")]), tools=[approve])


def test_sibling_agent_tool_writes_keep_separate_producers(sink):
    one = _writing_child("one", "decision", "approved one", "approve_one")
    two = _writing_child("two", "other", "approved two", "approve_two")
    caller = LlmAgent(name="caller", model=ScriptedModel([
        calls(("one", {"request": "first"}, "call-one")),
        calls(("two", {"request": "second"}, "call-two")), text("done")]),
        tools=[AgentTool(one), AgentTool(two)])
    _, _, sess = asyncio.run(conversation(SequentialAgent(name="pipeline", sub_agents=[caller, reader_agent()])))
    check = next(check for check in sink.checks if check[0] == "consume")
    lineage = ancestors(sink, check[2])
    assert any(sess.id in event.text and '"decision"' in event.text and "approved one" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve_one" for event in lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "approve_two" for event in lineage)


def test_parallel_agent_tool_batch_rejects_forwarded_writes(sink):
    async def run():
        one = _writing_child("one", "decision", "approved one", "approve_one")
        two = LlmAgent(name="two", model=ScriptedModel([text("two answer")]))
        caller = LlmAgent(name="caller", model=ScriptedModel([
            calls(("one", {"request": "first"}, "call-one"), ("two", {"request": "second"}, "call-two")),
            text("done")]), tools=[AgentTool(one), AgentTool(two)])
        runner = InMemoryRunner(agent=caller, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="parallel tool calls"):
            _ = [event async for event in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="start")]))]
        stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id)
        assert "decision" not in stored.state
    asyncio.run(run())
