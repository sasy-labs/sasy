"""Canonical versions reach real Langroid model/tool boundaries without stale ancestry."""
import asyncio
import copy
import importlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langroid import ChatAgent, ChatAgentConfig, ChatDocument, Entity
from langroid.agent.chat_document import ChatDocMetaData
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.mock_lm import MockLMConfig
from opentelemetry.trace import NoOpTracerProvider
from sasy.instrumentation import langroid as adapter
from sasy.instrumentation.dependencies import ComputationScopeError
from sasy.instrumentation.otel.context import _current_input_ids
from sasy.instrumentation.session import _session_id_var, session
from sasy.proto.observability_pb2 import Event, Tool


@pytest.fixture
def boundary(monkeypatch):
    state = SimpleNamespace(events={}, parents={}, calls=[], failures=False)
    def resolve(snapshots):
        if state.failures:
            raise RuntimeError("observation unavailable")
        state.calls.append(copy.deepcopy(snapshots))
        aliases = {}
        ids = []
        for snapshot in snapshots:
            event = Event()
            event.CopyFrom(snapshot.event)
            if snapshot.reuse_dependencies:
                assert snapshot.base_id in state.events
                before = state.events[snapshot.base_id]
                if before != event:
                    raise adapter.MessageProvenanceError("unexplained changed snapshot")
                identifier = snapshot.base_id
            else:
                parents = tuple(aliases.get(edge.source, edge.source) for edge in snapshot.dependencies)
                assert all(parent in state.events for parent in parents), parents
                matches = [key for key, old in state.events.items() if old == event and state.parents[key] == parents]
                identifier = matches[0] if matches else f"canonical-{len(state.events)}"
                state.events[identifier] = event
                state.parents[identifier] = parents
            aliases[event.id] = identifier
            ids.append(identifier)
        return ids
    async def resolve_async(snapshots):
        await asyncio.sleep(0)
        return resolve(snapshots)
    monkeypatch.setattr(adapter, "resolve_events", resolve)
    monkeypatch.setattr(adapter, "resolve_events_async", resolve_async)
    with session("snapshot-tests", end_on_exit=False):
        yield state


@pytest.fixture
def wrappers(monkeypatch):
    registered = {}
    def register(module, name, wrapper):
        registered[name] = wrapper
    monkeypatch.setattr(adapter, "wrap_function_wrapper", register)
    monkeypatch.setattr(adapter, "get_tracer", lambda *_: NoOpTracerProvider().get_tracer(__name__))
    adapter.instrument.cache_clear()
    adapter.instrument()
    yield registered
    adapter.instrument.cache_clear()


@pytest.fixture
def real_wrappers(monkeypatch):
    import wrapt
    def register(module, name, wrapper):
        module_object = importlib.import_module(module)
        class_name, method_name = name.split(".")
        owner = getattr(module_object, class_name)
        # Record the descriptor, including staticmethod semantics, for restoration.
        monkeypatch.setattr(owner, method_name, inspect.getattr_static(owner, method_name))
        wrapt.wrap_function_wrapper(module, name, wrapper)
    monkeypatch.setattr(adapter, "wrap_function_wrapper", register)
    monkeypatch.setattr(adapter, "get_tracer", lambda *_: NoOpTracerProvider().get_tracer(__name__))
    adapter.instrument.cache_clear()
    adapter.instrument()
    yield
    adapter.instrument.cache_clear()


def document(text, sender=Entity.USER):
    return ChatDocument(content=text, metadata=ChatDocMetaData(sender=sender))


def ancestry(boundary, ids):
    """The texts of every recorded event reachable from ``ids``."""
    pending, seen = list(ids), set()
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(boundary.parents[node])
    return {boundary.events[node].text for node in seen}


def allow_tools(monkeypatch, checks):
    """Authorize every tool call and record the ancestry each check was sent."""
    verdict = SimpleNamespace(authorized=True, transform_ids=[], denial_trace=None)
    def check(name, arguments, inputs):
        checks.append((name, list(inputs)))
        return verdict
    async def check_async(*args):
        return check(*args)
    monkeypatch.setattr(adapter, "rm_check_tool_call", check)
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", check_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unchanged_full_inputs_batched_and_canonical_ids_reused(boundary, asynchronous):
    messages = [LLMMessage(role=Role.USER, content="one"), LLMMessage(role=Role.SYSTEM, content="two")]
    resolve = adapter._resolve_inputs_async if asynchronous else adapter._resolve_inputs
    first = await resolve(messages) if asynchronous else resolve(messages)
    second = await resolve(messages) if asynchronous else resolve(messages)
    assert first == second
    assert first == ["canonical-0", "canonical-1"]
    assert len(boundary.calls) == 2
    assert all(len(batch) == 2 for batch in boundary.calls)
    assert all(snapshot.reuse_dependencies and snapshot.event.text for snapshot in boundary.calls[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_edit_preserves_old_outputs_and_clears_tool_provenance(boundary, asynchronous):
    source = document("original")
    old = adapter._resolve_inputs([source])[0]
    result = document("tool result", Entity.AGENT)
    tool = Tool(name="approval", arguments='{"action":"one"}')
    result_id = adapter._record_output(result, [old], derived_from=tool)
    later = document("other actual input")
    later_id = adapter._resolve_inputs([later])[0]
    result.content = "edited tool result"
    if asynchronous:
        changed = await adapter.record_message_update_async(result, input_ids=[later_id])
    else:
        changed = adapter.record_message_update(result, input_ids=[later_id])
    assert changed != result_id
    assert boundary.parents[changed] == (result_id, later_id)
    assert not boundary.events[changed].HasField("derived_from")
    assert boundary.events[result_id].text == "tool result"
    assert boundary.events[result_id].derived_from == tool
    assert boundary.parents[result_id] == (old,)
    assert adapter._resolve_inputs([result]) == [changed]


def test_unexplained_content_parent_and_generated_inputs_fail(boundary):
    item = document("first")
    first = adapter._resolve_inputs([item])[0]
    item.content = "unexplained change"
    with pytest.raises(adapter.MessageProvenanceError, match="changed"):
        adapter._resolve_inputs([item])
    assert adapter._version(item).canonical_id == first
    item.content = "first"
    item.metadata.parent_id = "unrelated approval"
    with pytest.raises(adapter.MessageProvenanceError, match="parent"):
        adapter._resolve_inputs([item])
    with pytest.raises(adapter.MessageProvenanceError, match="Generated"):
        adapter._resolve_inputs([document("unrecorded", Entity.LLM)])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_finalized_inputs_replace_prehistory_and_reset_context(boundary, wrappers, asynchronous):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.message_history = [LLMMessage(role=Role.USER, content="stale history")]
    actual = [LLMMessage(role=Role.SYSTEM, content="actual system"), LLMMessage(role=Role.USER, content="new user input")]
    output = document("generated", Entity.LLM)
    seen = []
    def run(messages):
        seen.extend(_current_input_ids.get())
        return output
    async def arun(messages):
        return run(messages)
    token = _current_input_ids.set(["outer"])
    try:
        if asynchronous:
            result = await wrappers["ChatAgent.llm_response_messages_async"](arun, agent, (actual,), {})
        else:
            result = wrappers["ChatAgent.llm_response_messages"](run, agent, (actual,), {})
        assert _current_input_ids.get() == ["outer"]
    finally:
        _current_input_ids.reset(token)
    assert result is output
    assert [boundary.events[key].text for key in seen] == ["actual system", "new user input"]
    assert boundary.parents[adapter._version(output).canonical_id] == tuple(seen)
    assert not any(event.text == "stale history" for event in boundary.events.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_observation_failure_stops_dispatch_without_cache_or_context_leak(boundary, wrappers, asynchronous):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    message = LLMMessage(role=Role.USER, content="input")
    dispatch = Mock(side_effect=AssertionError("must not dispatch"))
    boundary.failures = True
    token = _current_input_ids.set(["outer"])
    try:
        with pytest.raises(RuntimeError, match="observation unavailable"):
            if asynchronous:
                await wrappers["ChatAgent.llm_response_messages_async"](dispatch, agent, ([message],), {})
            else:
                wrappers["ChatAgent.llm_response_messages"](dispatch, agent, ([message],), {})
        assert adapter._version(message) is None
        assert _current_input_ids.get() == ["outer"]
    finally:
        _current_input_ids.reset(token)
    dispatch.assert_not_called()


def test_observed_truncation_is_an_old_version_dependency(boundary, wrappers):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    message = LLMMessage(role=Role.USER, content="long initial content")
    agent.message_history = [message]
    first = adapter._resolve_inputs([message])[0]
    def truncate(index):
        message.content = "long ..."
        return message
    assert wrappers["ChatAgent.truncate_message"](truncate, agent, (0,), {}) is message
    current = adapter._resolve_inputs([message])[0]
    assert current != first and boundary.parents[current] == (first,)
    assert boundary.events[first].text == "long initial content"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_framework_preparation_and_conversion_history(boundary, real_wrappers, asynchronous):
    seen = []
    def respond(text):
        seen.append((_current_input_ids.get().copy(), text))
        return "generated response"
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=respond), system_message="actual system", show_stats=False))
    if asynchronous:
        first = await agent.llm_response_async("new input")
        second = await agent.llm_response_async("follow-up")
    else:
        first = agent.llm_response("new input")
        second = agent.llm_response("follow-up")
    assert first is not None and second is not None
    assert {boundary.events[key].text for key in seen[0][0]} == {"actual system", "new input"}
    second_inputs = seen[1][0]
    assert "generated response" in {boundary.events[key].text for key in second_inputs}
    assert "follow-up" in {boundary.events[key].text for key in second_inputs}
    first_output = adapter._version(first).canonical_id
    views = [key for key in second_inputs if boundary.events[key].text == "generated response"]
    assert any(first_output in boundary.parents[key] for key in views)
    assert all(not identifier.startswith(first.metadata.id) for identifier in second_inputs)


def test_conversion_captures_old_source_and_rejects_unknown_edits(boundary, real_wrappers):
    source = document("original source")
    original_id = adapter._resolve_inputs([source])[0]
    converted = ChatDocument.to_LLMMessage(source)[0]
    source.content = "later source content"
    changed_id = adapter.record_message_update(source, input_ids=[])
    view = adapter._resolve_inputs([converted])[0]
    assert original_id in boundary.parents[view]
    assert changed_id not in boundary.parents[view]
    assert boundary.events[view].text == "original source"
    current = ChatDocument.to_LLMMessage(source)[0]
    current_view = adapter._resolve_inputs([current])[0]
    assert changed_id in boundary.parents[current_view]
    assert adapter._resolve_inputs([converted]) == [view]
    other = ChatDocument.to_LLMMessage(source)[0]
    other.content = "unexplained view change"
    with pytest.raises(adapter.MessageProvenanceError):
        adapter._resolve_inputs([other])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_task_tool_loop_uses_canonical_inputs(boundary, real_wrappers, monkeypatch, tmp_path, asynchronous):
    from langroid import Task, TaskConfig, ToolMessage
    from langroid.utils.constants import DONE
    calls = []
    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str
    class Agent(ChatAgent):
        def lookup(self, msg: Lookup) -> str:
            calls.append((msg.topic, _current_input_ids.get().copy()))
            return "synthetic result"
    def respond(text):
        return DONE + " completed" if "synthetic result" in text else '{"request":"lookup","topic":"public"}'
    agent = Agent(ChatAgentConfig(llm=MockLMConfig(response_fn=respond), use_tools=True, use_functions_api=False, show_stats=False))
    agent.enable_message(Lookup)
    verdict = SimpleNamespace(authorized=True, transform_ids=[])
    monkeypatch.setattr(adapter, "rm_check_tool_call", lambda *args: verdict)
    async def acheck(*args):
        return verdict
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", acheck)
    task = Task(agent, interactive=False, config=TaskConfig(logs_dir=str(tmp_path)))
    result = await task.run_async("lookup public", turns=5) if asynchronous else task.run("lookup public", turns=5)
    assert result is not None
    assert len(calls) == 1 and calls[0][0] == "public"
    assert calls[0][1] and all(value.startswith("canonical-") for value in calls[0][1])
    assert adapter._version(result) is not None


@pytest.mark.asyncio
async def test_async_observation_wait_cannot_change_consumed_message(boundary, wrappers, monkeypatch):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    message = LLMMessage(role=Role.USER, content="checked")
    entered, resume = asyncio.Event(), asyncio.Event()
    original = adapter.resolve_events_async
    async def resolve(snapshots):
        entered.set()
        await resume.wait()
        return await original(snapshots)
    monkeypatch.setattr(adapter, "resolve_events_async", resolve)
    consumed = []
    async def dispatch(messages):
        consumed.append(messages[0].content)
        return document("response", Entity.LLM)
    task = asyncio.create_task(wrappers["ChatAgent.llm_response_messages_async"](dispatch, agent, ([message],), {}))
    await entered.wait()
    message.content = "mutated during RPC"
    resume.set()
    await task
    assert consumed == ["checked"]
    assert adapter._version(message).event.text == "checked"
    with pytest.raises(adapter.MessageProvenanceError):
        adapter._resolve_inputs([message])


def test_message_versions_are_session_scoped(boundary):
    message = document("same object")
    first = adapter._resolve_inputs([message])[0]
    token = _session_id_var.set("another-session")
    try:
        assert adapter._version(message) is None
        adapter._resolve_inputs([message])
        assert not boundary.calls[-1][0].HasField("base_id")
    finally:
        _session_id_var.reset(token)
    assert adapter._version(message).canonical_id == first
    adapter._resolve_inputs([message])
    assert boundary.calls[-1][0].base_id == first
    assert boundary.calls[-1][0].reuse_dependencies


def test_missing_attachment_representation_fails_before_dispatch(boundary, wrappers):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    message = LLMMessage(role=Role.USER, content="see attachment")
    object.__setattr__(message, "files", ["unobserved attachment"])
    dispatch = Mock()
    with pytest.raises(adapter.MessageProvenanceError, match="attachment"):
        wrappers["ChatAgent.llm_response_messages"](dispatch, agent, ([message],), {})
    dispatch.assert_not_called()
    assert boundary.calls == []


def test_denied_tool_output_has_no_successful_tool_provenance(boundary, wrappers):
    from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall
    agent = ChatAgent(ChatAgentConfig(llm=None))
    message = document("", Entity.LLM)
    message.oai_tool_calls = [OpenAIToolCall(id="call", type="function", function=LLMFunctionCall(name="approve", arguments={}))]
    adapter._record_output(message, [])
    output = document("[BLOCKED] approve", Entity.AGENT)
    def respond(msg):
        adapter.add_tool_denial("approve", "denied", "{}", [])
        return output
    wrappers["Agent.agent_response"](respond, agent, (message,), {})
    assert not boundary.events[adapter._version(output).canonical_id].HasField("derived_from")


def test_conflicting_origins_in_one_batch_do_not_dispatch(boundary):
    first = LLMMessage(role=Role.USER, content="first")
    second = LLMMessage(role=Role.USER, content="different")
    object.__setattr__(first, "_observability_id", "shared")
    object.__setattr__(second, "_observability_id", "shared")
    with pytest.raises(adapter.MessageProvenanceError, match="Conflicting"):
        adapter._resolve_inputs([first, second])
    assert boundary.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_context_restored_after_dispatch_error_or_cancellation(boundary, wrappers, cancelled):
    agent = ChatAgent(ChatAgentConfig(llm=None))
    inputs = [LLMMessage(role=Role.USER, content="input")]
    async def dispatch(messages):
        assert _current_input_ids.get()[0].startswith("canonical-")
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("dispatch failed")
    token = _current_input_ids.set(["outer"])
    try:
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await wrappers["ChatAgent.llm_response_messages_async"](dispatch, agent, (inputs,), {})
        assert _current_input_ids.get() == ["outer"]
        assert _session_id_var.get() == "snapshot-tests"
    finally:
        _current_input_ids.reset(token)


def test_provider_metadata_edit_needs_explicit_provenance(boundary):
    message = LLMMessage(role=Role.USER, content="same text")
    original = adapter._resolve_inputs([message])[0]
    message.tool_call_id = "different correlation"
    with pytest.raises(adapter.MessageProvenanceError, match="metadata"):
        adapter._resolve_inputs([message])
    changed = adapter.record_message_update(message, input_ids=[])
    assert boundary.parents[changed] == (original,)
    assert adapter._resolve_inputs([message]) == [changed]


def test_identical_content_with_distinct_versions_cannot_collapse_in_batch(boundary):
    source = document("input")
    source_id = adapter._resolve_inputs([source])[0]
    first = document("same output", Entity.LLM)
    first_id = adapter._record_output(first, [source_id])
    second = copy.deepcopy(first)
    second_id = adapter.record_message_update(second, input_ids=[])
    assert first_id != second_id
    assert boundary.events[first_id] == boundary.events[second_id]
    before = len(boundary.calls)
    with pytest.raises(adapter.MessageProvenanceError, match="Conflicting"):
        adapter._resolve_inputs([first, second])
    assert len(boundary.calls) == before
    assert adapter._version(first).canonical_id == first_id
    assert adapter._version(second).canonical_id == second_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("block", ["deny", "transform", "error"])
async def test_real_task_blocked_approval_never_authorizes_sink_even_when_feedback_cleared(
    boundary, real_wrappers, monkeypatch, tmp_path, asynchronous, block,
):
    from langroid import Task, TaskConfig, ToolMessage
    from langroid.utils.constants import DONE
    from sasy.instrumentation import feedback
    from sasy.instrumentation.config import InstrumentationConfig
    effects, checks = [], []
    responses = iter(['{"request":"approve"}', '{"request":"send"}', DONE + " finished"])
    class Approve(ToolMessage):
        request: str = "approve"
        purpose: str = "Approve the synthetic send"
    class Send(ToolMessage):
        request: str = "send"
        purpose: str = "Send a synthetic value after approval"
    class Agent(ChatAgent):
        def approve(self, msg: Approve) -> str:
            if block == "error":
                raise ValueError("approval handler failed")
            effects.append("approved")
            return "Approved"
        def send(self, msg: Send) -> str:
            effects.append("sent")
            return "Sent"
        def handle_message(self, message):
            try:
                return super().handle_message(message)
            except ValueError:
                return "The approval handler failed"
        async def handle_message_async(self, message):
            try:
                return await super().handle_message_async(message)
            except ValueError:
                return "The approval handler failed"
    def check(name, arguments, inputs):
        checks.append(name)
        if name == "approve":
            return SimpleNamespace(authorized=block != "deny", transform_ids=["required"] if block == "transform" else [], denial_trace=None)
        pending, ancestors = list(inputs), set()
        while pending:
            node = pending.pop()
            if node not in ancestors:
                ancestors.add(node)
                pending.extend(boundary.parents[node])
        approved = any(boundary.events[node].HasField("derived_from") and boundary.events[node].derived_from.name == "approve" for node in ancestors)
        return SimpleNamespace(authorized=approved, transform_ids=[], denial_trace=None)
    async def acheck(*args):
        return check(*args)
    monkeypatch.setattr(adapter, "rm_check_tool_call", check)
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", acheck)
    config = InstrumentationConfig(log_denials=False, feedback_callback=lambda accumulator, *_: accumulator.clear())
    monkeypatch.setattr(adapter, "get_config", lambda: config)
    monkeypatch.setattr(feedback, "get_config", lambda: config)
    agent = Agent(ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: next(responses)), use_tools=True, use_functions_api=False, show_stats=False))
    agent.enable_message([Approve, Send])
    task = Task(agent, interactive=False, config=TaskConfig(logs_dir=str(tmp_path)))
    if asynchronous:
        await task.run_async("approve then send", turns=7)
    else:
        task.run("approve then send", turns=7)
    assert checks == (["approve", "approve", "send"] if asynchronous and block == "error" else ["approve", "send"])
    assert effects == []
    assert adapter._dispatch_outcomes.get() is None
    assert not any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in boundary.events.values())


def test_one_origin_cannot_declare_conflicting_explicit_dependencies(boundary):
    left, right = adapter._resolve_inputs([document("left"), document("right")])
    output = document("same payload", Entity.LLM)
    batch = adapter._Batch()
    batch.add(output, [left])
    before = len(boundary.calls)
    with pytest.raises(adapter.MessageProvenanceError, match="Conflicting"):
        batch.add(copy.deepcopy(output), [right])
    assert len(boundary.calls) == before


def test_identical_resolution_plans_still_deduplicate(boundary):
    message = document("input")
    canonical = adapter._resolve_inputs([message])[0]
    assert adapter._resolve_inputs([message, copy.deepcopy(message)]) == [canonical, canonical]
    assert len(boundary.calls[-1]) == 1


def test_handled_tool_call_is_the_same_message_when_observed_again(boundary, wrappers):
    # Langroid writes ``request=<tool name>`` into a tool call's live
    # arguments when an agent handles it. The message an agent then reads
    # again must still be the version that was recorded, or every multi-step
    # run refuses at its second observation.
    from langroid import ToolMessage
    from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall

    class SubmitAnswer(ToolMessage):
        request: str = "submit_answer"
        purpose: str = "Submit the answer"
        answer: str

        def handle(self) -> str:
            return "recorded"

    agent = ChatAgent(ChatAgentConfig(llm=None, use_functions_api=True, use_tools_api=True))
    agent.enable_message(SubmitAnswer)
    message = document("", Entity.LLM)
    message.oai_tool_calls = [OpenAIToolCall(id="call", type="function",
        function=LLMFunctionCall(name="submit_answer", arguments={"answer": "42"}))]
    adapter._record_output(message, [])
    recorded = boundary.events[adapter._version(message).canonical_id]
    assert [tool.arguments for tool in recorded.tools] == ['{"answer": "42"}']

    # Langroid parses the call in place while handling it.
    assert [tool.answer for tool in agent.get_oai_tool_calls_classes(message)] == ["42"]
    assert message.oai_tool_calls[0].function.arguments == {"answer": "42", "request": "submit_answer"}

    output = document("recorded", Entity.AGENT)
    wrappers["Agent.agent_response"](lambda msg: output, agent, (message,), {})
    reused = [snapshot for call in boundary.calls for snapshot in call if snapshot.reuse_dependencies]
    assert reused and reused[-1].base_id == adapter._version(message).canonical_id
    assert boundary.events[adapter._version(message).canonical_id] == recorded


def test_text_tool_calls_stay_recorded_when_another_agent_reads_the_message(boundary, wrappers):
    # A tool call written into the text (Langroid's JSON form) is read with
    # the observing agent's tool registry. The agent that produced it records
    # it; a critic that only handles a different tool reads the same message.
    from langroid import ToolMessage

    class FinalAnswer(ToolMessage):
        request: str = "final_answer"
        purpose: str = "Submit the answer"
        answer: str

    class Feedback(ToolMessage):
        request: str = "feedback"
        purpose: str = "Review the answer"
        critique: str

    author = ChatAgent(ChatAgentConfig(llm=None, use_functions_api=False, use_tools=True))
    author.enable_message(FinalAnswer)
    critic = ChatAgent(ChatAgentConfig(llm=None, use_functions_api=False, use_tools=True))
    critic.enable_message(Feedback)
    message = document('TOOL: {"request": "final_answer", "answer": "42"}', Entity.LLM)
    adapter._record_output(message, [], author)
    recorded = boundary.events[adapter._version(message).canonical_id]
    assert [tool.name for tool in recorded.tools] == ["final_answer"]

    output = document("looks right", Entity.AGENT)
    wrappers["Agent.agent_response"](lambda msg: output, critic, (message,), {})
    reused = [snapshot for call in boundary.calls for snapshot in call if snapshot.reuse_dependencies]
    assert reused and reused[-1].base_id == adapter._version(message).canonical_id
    assert boundary.events[adapter._version(message).canonical_id] == recorded

    # An edit to the text is still a change.
    message.content = 'TOOL: {"request": "final_answer", "answer": "43"}'
    with pytest.raises(adapter.MessageProvenanceError):
        wrappers["Agent.agent_response"](lambda msg: output, critic, (message,), {})


def test_function_result_copy_is_named_after_the_agent_not_the_function(boundary, real_wrappers):
    # A function-result message carries the function's name as its provider
    # ``name``. The copy the model reads is the agent's own message, so a
    # rule that follows one agent's messages reaches the recorded result.
    from langroid.language_models.base import LLMFunctionCall

    agent = ChatAgent(ChatAgentConfig(llm=None, name="FDAHandler", use_functions_api=True, use_tools_api=False))
    call = document("", Entity.LLM)
    call.function_call = LLMFunctionCall(name="register_fda_usage", arguments={})
    call.metadata.sender_name = "FDAHandler"
    adapter._record_output(call, [], agent)
    result = document("Confirmed.", Entity.AGENT)
    result.metadata.sender_name = "FDAHandler"
    result.metadata.parent_id = call.id()
    adapter._record_output(result, [adapter._version(call).canonical_id], agent,
                           Tool(name="register_fda_usage", arguments="{}"))
    converted = ChatDocument.to_LLMMessage(result)[0]
    assert converted.role == Role.FUNCTION and converted.name == "register_fda_usage"
    view = adapter._resolve_inputs([converted], agent)[0]
    assert boundary.events[view].agent == "FDAHandler"
    assert adapter._version(result).canonical_id in boundary.parents[view]


def test_resumed_task_keeps_what_its_pending_message_was_computed_from(boundary, real_wrappers, monkeypatch, tmp_path):
    # A task started with no message continues from the agent's own history.
    # The message it continues from keeps the inputs it was recorded with, so
    # the first tool call of the continuation is still checked against the
    # user input the model answered.
    from langroid import Task, TaskConfig, ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)

    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str

    class Agent(ChatAgent):
        def lookup(self, msg: Lookup) -> str:
            return "synthetic result"

    agent = Agent(ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: '{"request":"lookup","topic":"public"}'),
                                  use_tools=True, use_functions_api=False, show_stats=False, system_message="sys"))
    agent.enable_message(Lookup)
    answer = agent.llm_response("user asks: secret data here")
    recorded = adapter._version(answer).canonical_id
    assert "user asks: secret data here" in ancestry(boundary, [recorded])

    task = Task(agent, interactive=False, restart=False, config=TaskConfig(logs_dir=str(tmp_path)))
    assert task.init() is answer
    assert adapter._version(answer).canonical_id == recorded

    task.step()
    assert [name for name, _ in checks] == ["lookup"]
    assert "user asks: secret data here" in ancestry(boundary, checks[-1][1])


def test_resuming_a_task_from_unobserved_history_is_refused(boundary, real_wrappers, tmp_path):
    from langroid import Task, TaskConfig
    agent = ChatAgent(ChatAgentConfig(llm=None, show_stats=False, system_message="sys"))
    restored = document("restored answer", Entity.LLM)
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="restored answer", chat_document_id=restored.id()),
    ]
    task = Task(agent, interactive=False, restart=False, config=TaskConfig(logs_dir=str(tmp_path)))
    with pytest.raises(adapter.MessageProvenanceError, match="continues from"):
        task.init()
    assert adapter._version(restored) is None


def test_two_agents_read_the_same_tool_calls_out_of_one_message(boundary):
    # The tools a message carries are a property of the message, not of the
    # agent reading it: an agent that may emit a call and an agent that
    # handles it must record the same thing, or one of them refuses the other's
    # message.
    from langroid import ToolMessage

    class Alpha(ToolMessage):
        request: str = "alpha"
        purpose: str = "Run alpha"

    emitter = ChatAgent(ChatAgentConfig(name="Emitter", llm=None, use_tools=True, use_functions_api=False))
    emitter.enable_message(Alpha, use=True, handle=False)
    handler = ChatAgent(ChatAgentConfig(name="Handler", llm=None, use_tools=True, use_functions_api=False))
    handler.enable_message(Alpha)
    stranger = ChatAgent(ChatAgentConfig(name="Stranger", llm=None, use_tools=True, use_functions_api=False))

    message = document('{"request":"alpha"}', Entity.LLM)
    adapter._record_output(message, [], emitter)
    recorded = boundary.events[adapter._version(message).canonical_id]
    assert [tool.name for tool in recorded.tools] == ["alpha"]
    # Every reader derives them again and agrees, so re-observation is a reuse.
    for reader in (emitter, handler, stranger):
        assert adapter._resolve_inputs([message], reader) == [adapter._version(message).canonical_id]
    assert boundary.events[adapter._version(message).canonical_id] == recorded


def test_sub_task_handles_a_tool_its_parent_emitted(boundary, real_wrappers, monkeypatch, tmp_path):
    from langroid import Task, TaskConfig, ToolMessage
    from langroid.utils.constants import DONE
    checks = []
    allow_tools(monkeypatch, checks)

    class Alpha(ToolMessage):
        request: str = "alpha"
        purpose: str = "Run alpha"

    class Handler(ChatAgent):
        def alpha(self, msg: Alpha) -> str:
            return DONE + " alpha ran"

    parent = ChatAgent(ChatAgentConfig(
        name="Parent", use_tools=True, use_functions_api=False, show_stats=False,
        llm=MockLMConfig(response_fn=lambda text: DONE if "alpha ran" in text else '{"request":"alpha"}')))
    parent.enable_message(Alpha, use=True, handle=False)
    sub = Handler(ChatAgentConfig(name="Sub", llm=None, use_tools=True, use_functions_api=False, show_stats=False))
    sub.enable_message(Alpha)

    parent_task = Task(parent, interactive=False, config=TaskConfig(logs_dir=str(tmp_path)))
    parent_task.add_sub_task(Task(sub, interactive=False, single_round=True, config=TaskConfig(logs_dir=str(tmp_path))))
    parent_task.run("go", turns=6)

    assert [name for name, _ in checks] == ["alpha"]
    assert '{"request":"alpha"}' in ancestry(boundary, checks[-1][1])


def test_removing_a_recorded_tool_call_from_an_unchanged_message_is_refused(boundary):
    # Dropping a call from a message the model already sent is an edit, even
    # though the text around it did not move. Its recorded facts must not
    # survive as if the message still made the call.
    from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall

    def call(name):
        return OpenAIToolCall(id=f"call-{name}", type="function",
                              function=LLMFunctionCall(name=name, arguments={}))

    message = document("here are two calls", Entity.LLM)
    message.oai_tool_calls = [call("alpha"), call("beta")]
    adapter._record_output(message, [])
    assert [tool.name for tool in boundary.events[adapter._version(message).canonical_id].tools] == ["alpha", "beta"]

    message.oai_tool_calls = [call("alpha")]
    with pytest.raises(adapter.MessageProvenanceError):
        adapter._resolve_inputs([message])
    message.oai_tool_calls = []
    with pytest.raises(adapter.MessageProvenanceError):
        adapter._resolve_inputs([message])


def test_a_malformed_tool_call_is_recorded_and_left_to_the_framework(boundary, real_wrappers, monkeypatch, tmp_path):
    # Models routinely emit a known tool with unusable fields. Recording the
    # attempt must not end the run: Langroid's own validation feedback tells
    # the model what to fix.
    from langroid import Task, TaskConfig, ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)

    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str

    class Agent(ChatAgent):
        def lookup(self, msg: Lookup) -> str:
            return "synthetic result"

    malformed = '{"request":"lookup","topic":{"not":"a string"}}'
    agent = Agent(ChatAgentConfig(llm=None, use_tools=True, use_functions_api=False, show_stats=False))
    agent.enable_message(Lookup)
    message = document(malformed, Entity.LLM)
    adapter._record_output(message, [], agent)
    recorded = boundary.events[adapter._version(message).canonical_id]
    assert [tool.name for tool in recorded.tools] == ["lookup"]
    assert json.loads(recorded.tools[0].arguments) == {"request": "lookup", "topic": {"not": "a string"}}

    responses = iter([malformed, "DONE recovered"])
    running = Agent(ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: next(responses)),
                                    use_tools=True, use_functions_api=False, show_stats=False))
    running.enable_message(Lookup)
    task = Task(running, interactive=False, config=TaskConfig(logs_dir=str(tmp_path)))
    assert task.run("go", turns=4) is not None
    assert checks == []


def test_observing_a_message_leaves_the_agents_tool_error_alone(boundary):
    # The adapter observes; it does not touch the framework state Langroid's
    # own recovery from a malformed tool call depends on.
    from langroid import ToolMessage

    class Lookup(ToolMessage):
        request: str = "lookup"
        purpose: str = "Look up a synthetic value"
        topic: str

    agent = ChatAgent(ChatAgentConfig(llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message(Lookup)
    agent.tool_error = True
    adapter._resolve_inputs([document("an unrelated message")], agent)
    assert agent.tool_error is True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_tool_dispatched_outside_a_responder_is_refused(boundary, real_wrappers, monkeypatch, asynchronous):
    # Outside a responder there is no record of what the call was computed
    # from, so the check would be sent with no ancestry at all.
    from langroid import ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)
    ran = []

    class Alpha(ToolMessage):
        request: str = "alpha"
        purpose: str = "Run alpha"

    class Agent(ChatAgent):
        def alpha(self, msg: Alpha) -> str:
            ran.append("alpha")
            return "alpha ran"

    agent = Agent(ChatAgentConfig(llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message(Alpha)
    message = document('{"request":"alpha"}', Entity.LLM)
    adapter._record_output(message, [], agent)

    with pytest.raises(ComputationScopeError, match="responder"):
        if asynchronous:
            await agent.handle_message_async(message)
        else:
            agent.handle_message(message)
    assert checks == [] and ran == []

    # The same call through the responder is checked with its ancestry. The
    # async path reaches Langroid's sync handler, which is checked again.
    if asynchronous:
        await agent.agent_response_async(message)
    else:
        agent.agent_response(message)
    assert {name for name, _ in checks} == {"alpha"} and ran == ["alpha"]
    assert '{"request":"alpha"}' in ancestry(boundary, checks[-1][1])


def test_instrument_warns_when_a_tool_may_run_without_a_decision(monkeypatch, caplog):
    from sasy.instrumentation.config import InstrumentationConfig
    monkeypatch.setattr(adapter, "wrap_function_wrapper", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "get_tracer", lambda *_: NoOpTracerProvider().get_tracer(__name__))

    def install(fail_closed):
        monkeypatch.setattr(adapter, "get_config", lambda: InstrumentationConfig(tool_policy_fail_closed=fail_closed))
        adapter.instrument.cache_clear()
        caplog.clear()
        with caplog.at_level("WARNING", logger=adapter.logger.name):
            adapter.instrument()
        adapter.instrument.cache_clear()
        return caplog.text

    assert "tool_policy_fail_closed is off" in install(False)
    assert "tool_policy_fail_closed" not in install(True)


def test_a_tool_dispatched_from_a_worker_thread_needs_the_context_carried(boundary, real_wrappers, monkeypatch):
    # A handler that fans work out to a thread pool starts each worker from
    # the context defaults: the worker knows neither the session nor what the
    # work was computed from. Raw executor work is outside instrumentation;
    # carrying the context opts that work into the protected computation.
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from langroid import ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)
    observed = {}

    class Alpha(ToolMessage):
        request: str = "alpha"
        purpose: str = "Run alpha"

    class Beta(ToolMessage):
        request: str = "beta"
        purpose: str = "Run beta"

    class Agent(ChatAgent):
        def alpha(self, msg: Alpha) -> str:
            def dispatch():
                return self.handle_message(second)

            with ThreadPoolExecutor(1) as pool:
                observed["lost"] = pool.submit(dispatch).exception()
                observed["carried"] = pool.submit(contextvars.copy_context().run, dispatch).result()
            return "alpha ran"

        def beta(self, msg: Beta) -> str:
            return "beta ran"

    agent = Agent(ChatAgentConfig(llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Alpha, Beta])
    first = document('{"request":"alpha"}', Entity.LLM)
    second = document('{"request":"beta"}', Entity.LLM)
    adapter._record_output(first, [], agent)
    adapter._record_output(second, [], agent)

    agent.agent_response(first)
    assert observed["lost"] is None
    assert "beta ran" in str(observed["carried"])
    assert [name for name, _ in checks] == ["alpha", "beta"]
    # The carried context keeps the responder's inputs, so the worker's call is
    # checked against what the agent was answering.
    assert checks[-1][1] == checks[0][1]
    assert '{"request":"alpha"}' in ancestry(boundary, checks[-1][1])


def two_tool_agent(name="worker"):
    """An agent that handles two tools, for replies that hold more than one call."""
    from langroid import ToolMessage

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Note(ToolMessage):
        request: str = "note"
        purpose: str = "Write a note"
        text: str

    class Worker(ChatAgent):
        def read_file(self, msg: Read) -> str:
            return f"contents of {msg.path}"

        def note(self, msg: Note) -> str:
            return f"noted {msg.text}"

    agent = Worker(ChatAgentConfig(name=name, llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Note])
    return agent


def tool_result_nodes(boundary):
    """Every recorded event that carries tool-result provenance, by tool name."""
    return {event.derived_from.name: (identifier, event)
            for identifier, event in boundary.events.items() if event.HasField("derived_from")}


def deny_tool(monkeypatch, denied, checks):
    """Authorize every call except ``denied``, recording what was checked."""
    def check(name, arguments, inputs):
        checks.append((name, list(inputs)))
        if name == denied:
            trace = SimpleNamespace(action_description="not allowed", reasons=[], suggested_fixes=[])
            return SimpleNamespace(authorized=False, transform_ids=[], denial_trace=trace)
        return SimpleNamespace(authorized=True, transform_ids=[], denial_trace=None)
    async def check_async(*args):
        return check(*args)
    monkeypatch.setattr(adapter, "rm_check_tool_call", check)
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", check_async)


def test_a_reply_that_completed_two_tools_records_each_result_on_its_own(boundary, real_wrappers, monkeypatch):
    # The model chooses how many calls a reply holds. The combined result names
    # no single tool, so each tool's own result is its own node and a rule that
    # looks for a read through its tool result still reaches it.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = two_tool_agent()
    call = document('{"request":"read_file","path":"secret.txt"}\n{"request":"note","text":"hi"}', Entity.LLM)
    adapter._record_output(call, [], agent)
    requested = adapter._version(call).canonical_id

    output = agent.agent_response(call)
    combined = adapter._version(output).canonical_id
    results = tool_result_nodes(boundary)

    assert sorted(results) == ["note", "read_file"]
    assert boundary.events[results["read_file"][0]].text == "contents of secret.txt"
    assert boundary.events[results["note"][0]].text == "noted hi"
    assert json.loads(results["read_file"][1].derived_from.arguments)["path"] == "secret.txt"
    # Each result was computed from what its own call was checked against.
    for name, (identifier, _) in results.items():
        assert boundary.parents[identifier] == (requested,), name
    # The combined response depends on both, so CurrentDepends reaches each.
    assert set(boundary.parents[combined]) == {requested, results["read_file"][0], results["note"][0]}
    assert not boundary.events[combined].HasField("derived_from")
    # The per-tool nodes read as the same agent's messages as the response.
    for identifier, _ in results.values():
        assert boundary.events[identifier].role == boundary.events[combined].role
        assert boundary.events[identifier].agent == boundary.events[combined].agent == "worker"


def test_a_denied_call_beside_an_allowed_one_records_only_the_allowed_result(boundary, real_wrappers, monkeypatch):
    checks = []
    deny_tool(monkeypatch, "note", checks)
    agent = two_tool_agent()
    call = document('{"request":"read_file","path":"secret.txt"}\n{"request":"note","text":"hi"}', Entity.LLM)
    adapter._record_output(call, [], agent)

    output = agent.agent_response(call)
    results = tool_result_nodes(boundary)

    assert [name for name, _ in checks] == ["read_file", "note"]
    assert sorted(results) == ["read_file"]
    assert results["read_file"][0] in boundary.parents[adapter._version(output).canonical_id]
    assert "[BLOCKED] note" in output.content


def test_a_refused_redirect_beside_a_completed_read_leaves_the_read_its_own_result(boundary, real_wrappers, monkeypatch):
    # A call that names its own handler is refused before any check. It is
    # still a dispatch of this reply: if it were not counted, a reply holding it
    # and one completed read would look like a one-tool reply, and the combined
    # text, refusal included, would be recorded as the read's result.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = two_tool_agent()
    call = document('{"request":"read_file","path":"secret.txt"}\n'
                    '{"request":"note","text":"hi","_handler":"read_file"}', Entity.LLM)
    adapter._record_output(call, [], agent)

    output = agent.agent_response(call)
    results = tool_result_nodes(boundary)

    assert [name for name, _ in checks] == ["read_file"], checks        # the redirect was never checked
    assert sorted(results) == ["read_file"]
    read_id, read_event = results["read_file"]
    assert read_event.text == "contents of secret.txt"                 # the read's own result, nothing else
    assert "may not name its own handler" in output.content
    assert "may not name its own handler" not in read_event.text
    response = adapter._version(output).canonical_id
    assert read_id in boundary.parents[response] and read_id != response


def test_a_nested_handler_that_raises_leaves_the_callers_inputs_in_place(boundary, real_wrappers, monkeypatch):
    # A nested call's handler runs on the nested call's inputs, which differ from
    # its caller's (a top-level handler's do not, so it could not show this).
    # They are put back when the handler raises, so code that catches the
    # exception and carries on is not left reading the failed handler's inputs.
    from langroid import ToolMessage
    from sasy.instrumentation.otel.context import get_current_input_ids
    checks = []
    allow_tools(monkeypatch, checks)
    seen = {}

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Send(ToolMessage):
        request: str = "send"
        purpose: str = "Send a body"
        to: str

    class Worker(ChatAgent):
        def read_file(self, msg: Read) -> ToolMessage:
            seen["read_file"] = list(get_current_input_ids())
            return Send(to="partner")

        def send(self, msg: Send) -> str:
            seen["send"] = list(get_current_input_ids())
            raise RuntimeError("send failed")

    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Send])
    request = document('{"request":"read_file","path":"secret/plan.txt"}', Entity.LLM)
    adapter._record_output(request, [], agent)
    caller = [adapter._version(request).canonical_id]
    token = adapter._current_input_ids.set(caller)
    outcomes_token = adapter._dispatch_outcomes.set([])
    try:
        with pytest.raises(RuntimeError, match="send failed"):
            agent.handle_tool_message(Read(path="secret/plan.txt"))
        assert seen["read_file"] == caller
        assert seen["send"] != caller, "the nested handler did not run on its own inputs"
        assert list(get_current_input_ids()) == caller
        assert adapter._current_dispatch.get() is None
    finally:
        adapter._dispatch_outcomes.reset(outcomes_token)
        adapter._current_input_ids.reset(token)


def test_a_handler_that_handled_nothing_gets_no_tool_result_node(boundary, real_wrappers, monkeypatch):
    # Langroid returns None when no handler accepted the call. Nothing ran, so
    # there is no result to record beside the tools that did run.
    from langroid import ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)

    class Skip(ToolMessage):
        request: str = "skip"
        purpose: str = "Handled by nobody"

    agent = two_tool_agent()
    agent.enable_message(Skip, use=True, handle=True)
    call = document('{"request":"read_file","path":"secret.txt"}\n{"request":"skip"}', Entity.LLM)
    adapter._record_output(call, [], agent)

    agent.agent_response(call)
    assert [name for name, _ in checks] == ["read_file", "skip"]
    assert sorted(tool_result_nodes(boundary)) == ["read_file"]


@pytest.mark.asyncio
async def test_one_operation_crossing_both_gates_records_one_result_node(boundary, real_wrappers, monkeypatch):
    # Langroid's async dispatch delegates to the synchronous handler, so one
    # operation passes both gates. It is one call with one result, not two.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = two_tool_agent()
    call = document('{"request":"read_file","path":"secret.txt"}\n{"request":"note","text":"hi"}', Entity.LLM)
    await adapter._record_output_async(call, [], agent)

    output = await agent.agent_response_async(call)
    results = tool_result_nodes(boundary)

    assert [name for name, _ in checks] == ["read_file", "read_file", "note", "note"]
    assert sorted(results) == ["note", "read_file"]
    assert len(set(boundary.parents[adapter._version(output).canonical_id])) == 3


def test_a_single_tool_reply_records_exactly_what_it_recorded_before(boundary, real_wrappers, monkeypatch):
    # The single-call shape is what every recorded run and every deployed
    # policy was built against: one node, the response itself, carrying the
    # tool's provenance and depending only on the message it answers.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = two_tool_agent()
    call = document('{"request":"read_file","path":"secret.txt"}', Entity.LLM)
    adapter._record_output(call, [], agent)
    requested = adapter._version(call).canonical_id

    output = agent.agent_response(call)
    recorded = adapter._version(output).canonical_id

    assert list(boundary.events) == [requested, recorded]
    assert boundary.parents[recorded] == (requested,)
    assert boundary.events[recorded].text == "contents of secret.txt"
    assert boundary.events[recorded].derived_from.name == "read_file"
    assert json.loads(boundary.events[recorded].derived_from.arguments) == {
        "request": "read_file", "purpose": "Read a file", "path": "secret.txt", "id": ""}
    assert boundary.events[recorded].agent == "worker"


def reply(agent, calls, shape):
    """A model reply holding ``calls``, written the way ``shape`` says.

    ``text`` is Langroid's JSON-in-the-reply format, ``provider`` the
    structured tool-call field. A reply may hold several calls in either.
    """
    from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall
    if shape == "text":
        return document("\n".join(json.dumps({"request": name, **arguments})
                                  for name, arguments in calls), Entity.LLM)
    message = document("", Entity.LLM)
    message.oai_tool_calls = [
        OpenAIToolCall(id=f"c{index}", type="function",
                       function=LLMFunctionCall(name=name, arguments=arguments))
        for index, (name, arguments) in enumerate(calls)
    ]
    return message


def reachable(boundary, ids):
    """Every recorded node reachable from ``ids``, including themselves."""
    pending, seen = list(ids), set()
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(boundary.parents[node])
    return seen


def strict_ancestors(boundary, ids):
    """What ``CurrentDepends`` holds for ``ids``: their ancestors, not themselves.

    The relation is strict, so a node a check names is not among the nodes it
    depends on; only what that node was computed from is.
    """
    pending = [parent for node in ids for parent in boundary.parents[node]]
    seen = set()
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(boundary.parents[node])
    return seen


def refuse_second_gate(monkeypatch, checks, refusal):
    """Authorize each call once; refuse the second gate of ``read_file``."""
    def check(name, arguments, inputs):
        checks.append((name, list(inputs)))
        if name == "read_file" and [item for item, _ in checks].count("read_file") > 1:
            if refusal == "error":
                raise RuntimeError("no decision")
            trace = SimpleNamespace(action_description="not allowed", reasons=[], suggested_fixes=[])
            return SimpleNamespace(authorized=False, transform_ids=[], denial_trace=trace)
        return SimpleNamespace(authorized=True, transform_ids=[], denial_trace=None)
    async def check_async(*args):
        return check(*args)
    monkeypatch.setattr(adapter, "rm_check_tool_call", check)
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", check_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "provider"])
@pytest.mark.parametrize("refusal", ["deny", "error"])
async def test_a_refusal_at_either_gate_leaves_the_operation_without_a_result(
    boundary, real_wrappers, monkeypatch, shape, refusal,
):
    # Langroid's asynchronous dispatch delegates to the synchronous handler, so
    # one operation crosses two gates. The handler runs only if both authorize;
    # the "[BLOCKED]" string the refused gate returns is not a result, so the
    # operation has none and nothing may read a completed read_file out of it.
    checks = []
    refuse_second_gate(monkeypatch, checks, refusal)
    monkeypatch.setattr(adapter, "get_config", lambda: SimpleNamespace(
        log_policy_decisions=False, tool_policy_fail_closed=True))
    agent = two_tool_agent()
    call = reply(agent, [("read_file", {"path": "secret.txt"}), ("note", {"text": "hi"})], shape)
    await adapter._record_output_async(call, [], agent)

    output = await agent.agent_response_async(call)
    results = tool_result_nodes(boundary)

    assert [name for name, _ in checks] == ["read_file", "read_file", "note", "note"]
    assert sorted(results) == ["note"]
    assert "[BLOCKED] read_file" in output.content
    # A rule asking for a completed read_file finds nothing in what the reply left.
    recorded = adapter._version(output).canonical_id
    assert not any(boundary.events[node].derived_from.name == "read_file"
                   for node in reachable(boundary, [recorded])
                   if boundary.events[node].HasField("derived_from"))


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["deny", "error"])
async def test_a_lone_call_refused_at_its_second_gate_has_no_tool_provenance(
    boundary, real_wrappers, monkeypatch, refusal,
):
    # The single-call shape records the response itself as the tool's result.
    # A refusal at either gate means there is no result to record it as.
    checks = []
    refuse_second_gate(monkeypatch, checks, refusal)
    monkeypatch.setattr(adapter, "get_config", lambda: SimpleNamespace(
        log_policy_decisions=False, tool_policy_fail_closed=True))
    agent = two_tool_agent()
    call = reply(agent, [("read_file", {"path": "secret.txt"})], "text")
    await adapter._record_output_async(call, [], agent)

    output = await agent.agent_response_async(call)
    recorded = adapter._version(output).canonical_id
    assert not boundary.events[recorded].HasField("derived_from")
    assert tool_result_nodes(boundary) == {}


def chain_agent(name="worker"):
    """An agent whose read handler returns a send call for Langroid to dispatch."""
    from langroid import ToolMessage

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Send(ToolMessage):
        request: str = "send"
        purpose: str = "Send a body"
        to: str
        body: str

    class Worker(ChatAgent):
        def read_file(self, msg: Read) -> ToolMessage:
            return Send(to="partner", body=f"contents of {msg.path}")

        def send(self, msg: Send) -> str:
            return f"delivered to {msg.to}"

    agent = Worker(ChatAgentConfig(name=name, llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Send])
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_handler_that_returned_another_tool_is_recorded_as_that_tools_input(
    boundary, real_wrappers, monkeypatch, asynchronous,
):
    # A handler may return another tool message, which Langroid dispatches at
    # once. The returned message is the first tool's result, so it is recorded
    # before the second call is checked, and the second call is checked against
    # it: what it was computed from includes the read it came out of.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = chain_agent()
    call = reply(agent, [("read_file", {"path": "secret/plan.txt"})], "text")
    if asynchronous:
        await adapter._record_output_async(call, [], agent)
        output = await agent.agent_response_async(call)
    else:
        adapter._record_output(call, [], agent)
        output = agent.agent_response(call)
    requested = adapter._version(call).canonical_id
    results = tool_result_nodes(boundary)

    assert sorted(results) == ["read_file", "send"]
    read_node, send_node = results["read_file"][0], results["send"][0]
    # The read's result is the send call it returned, as Langroid serializes it.
    assert json.loads(boundary.events[read_node].text) == {
        "request": "send", "to": "partner", "body": "contents of secret/plan.txt"}
    assert boundary.parents[read_node] == (requested,)
    # Asking for the send is the agent's next message, computed from that
    # result: the two nodes an ordinary reply has, so a rule reading what the
    # send depends on finds the read's result one hop back.
    asked = [node for node, event in boundary.events.items()
             if boundary.parents[node] == (read_node,) and not event.HasField("derived_from")]
    assert len(asked) == 1, asked
    request_node = asked[0]
    assert boundary.events[request_node].text == boundary.events[read_node].text
    # The send was checked against it, and is recorded as computed from it.
    # Neither result node carries the other's text. The read has no
    # asynchronous handler, so on that path its one operation crosses two gates.
    gates = 2 if asynchronous else 1
    assert [(name, list(inputs)) for name, inputs in checks] == (
        [("read_file", [requested])] * gates + [("send", [request_node])])
    assert boundary.parents[send_node] == (request_node,)
    assert boundary.events[send_node].text == "delivered to partner"
    assert "delivered" not in boundary.events[read_node].text
    # The reply reads the leaf of the chain, which reaches the read through it.
    combined = adapter._version(output).canonical_id
    assert boundary.parents[combined] == (requested, send_node)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_rule_about_the_reads_result_denies_the_send_it_produced(
    boundary, real_wrappers, monkeypatch, asynchronous,
):
    # The rule a deployed policy states: a send is denied when a read of a
    # secret document is among the tool results it was computed from. The
    # ancestry is read the way CurrentDepends holds it -- strictly, so a
    # result node the check itself names does not count as a dependency.
    checks = []
    def check(name, arguments, inputs):
        checks.append((name, list(inputs)))
        secret = any(boundary.events[node].derived_from.name == "read_file"
                     and "secret" in boundary.events[node].derived_from.arguments
                     for node in strict_ancestors(boundary, inputs)
                     if boundary.events[node].HasField("derived_from"))
        if name == "send" and secret:
            trace = SimpleNamespace(action_description="not allowed", reasons=[], suggested_fixes=[])
            return SimpleNamespace(authorized=False, transform_ids=[], denial_trace=trace)
        return SimpleNamespace(authorized=True, transform_ids=[], denial_trace=None)
    async def check_async(*args):
        return check(*args)
    monkeypatch.setattr(adapter, "rm_check_tool_call", check)
    monkeypatch.setattr(adapter, "rm_check_tool_call_async", check_async)

    agent = chain_agent()
    secret = reply(agent, [("read_file", {"path": "secret/plan.txt"})], "text")
    public = reply(agent, [("read_file", {"path": "public/memo.txt"})], "text")
    if asynchronous:
        await adapter._record_output_async(secret, [], agent)
        await adapter._record_output_async(public, [], agent)
        denied = await agent.agent_response_async(secret)
        allowed = await agent.agent_response_async(public)
    else:
        adapter._record_output(secret, [], agent)
        adapter._record_output(public, [], agent)
        denied = agent.agent_response(secret)
        allowed = agent.agent_response(public)

    assert "[BLOCKED] send" in denied.content
    assert "delivered to partner" in allowed.content
    sends = [event for event in boundary.events.values()
             if event.HasField("derived_from") and event.derived_from.name == "send"]
    # Only the allowed send has a result of its own.
    assert [event.text for event in sends] == ["delivered to partner"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "provider"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_two_identical_calls_in_one_reply_each_get_their_own_result(
    boundary, real_wrappers, monkeypatch, shape, asynchronous,
):
    # Two calls with the same name and the same arguments are two operations.
    # They may return different data, so each has a result of its own.
    from langroid import ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Note(ToolMessage):
        request: str = "note"
        purpose: str = "Write a note"
        text: str

    class Worker(ChatAgent):
        reads: int = 0

        def read_file(self, msg: Read) -> str:
            self.reads += 1
            return f"reading {msg.path} #{self.reads}"

        def note(self, msg: Note) -> str:
            return f"noted {msg.text}"

    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Note])
    call = reply(agent, [("read_file", {"path": "a.txt"}), ("read_file", {"path": "a.txt"}),
                         ("note", {"text": "hi"})], shape)
    if asynchronous:
        await adapter._record_output_async(call, [], agent)
        output = await agent.agent_response_async(call)
    else:
        adapter._record_output(call, [], agent)
        output = agent.agent_response(call)

    nodes = [(event.derived_from.name, event.text) for event in boundary.events.values()
             if event.HasField("derived_from")]
    assert nodes == [("read_file", "reading a.txt #1"), ("read_file", "reading a.txt #2"),
                     ("note", "noted hi")]
    combined = adapter._version(output).canonical_id
    assert len(boundary.parents[combined]) == 4


@pytest.mark.asyncio
async def test_native_async_handlers_record_each_result(boundary, real_wrappers, monkeypatch):
    # Langroid dispatches ``<tool>_async`` when the agent defines it, and the
    # operation then crosses the asynchronous gate alone.
    from langroid import ToolMessage
    checks = []
    allow_tools(monkeypatch, checks)

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Note(ToolMessage):
        request: str = "note"
        purpose: str = "Write a note"
        text: str

    class Worker(ChatAgent):
        async def read_file_async(self, msg: Read) -> str:
            await asyncio.sleep(0)
            return f"contents of {msg.path}"

        async def note_async(self, msg: Note) -> str:
            await asyncio.sleep(0)
            return f"noted {msg.text}"

    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Note])
    call = reply(agent, [("read_file", {"path": "secret.txt"}), ("note", {"text": "hi"})], "text")
    await adapter._record_output_async(call, [], agent)

    output = await agent.agent_response_async(call)
    results = tool_result_nodes(boundary)

    assert [name for name, _ in checks] == ["read_file", "note"]
    assert sorted(results) == ["note", "read_file"]
    assert boundary.events[results["read_file"][0]].text == "contents of secret.txt"
    assert boundary.events[results["note"][0]].text == "noted hi"
    combined = adapter._version(output).canonical_id
    assert set(boundary.parents[combined]) == {
        adapter._version(call).canonical_id, results["read_file"][0], results["note"][0]}


def redirect_agent(name="worker"):
    """An agent whose two tools have distinct handlers, one of them declared.

    ``Read`` and ``Send`` are handled by the methods their ``request`` names;
    ``Custom`` declares a handler of its own, which is the legitimate use of
    ``_handler`` and must keep working.
    """
    from langroid import ToolMessage

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Send(ToolMessage):
        request: str = "send"
        purpose: str = "Send a body"
        to: str = "partner"
        body: str = ""

    class Custom(ToolMessage):
        request: str = "custom"
        purpose: str = "A tool whose class declares its handler"
        text: str = ""
        _handler: str = "run_custom"

    class Worker(ChatAgent):
        def read_file(self, msg: Read) -> str:
            return f"contents of {msg.path}"

        def send(self, msg) -> str:
            return f"delivered to {getattr(msg, 'to', 'partner')}"

        def run_custom(self, msg: Custom) -> str:
            return f"custom ran on {msg.text}"

    agent = Worker(ChatAgentConfig(name=name, llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Send, Custom])
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "provider"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_call_that_names_its_own_handler_is_refused_before_any_check(
    boundary, real_wrappers, monkeypatch, shape, asynchronous,
):
    # A tool message accepts extra fields, so a model can put "_handler" on the
    # call. Langroid up to 0.67.0 ran the method named there while the call was
    # checked and recorded under its "request" name. The adapter refuses such a
    # call outright: neither handler runs, nothing is checked, nothing recorded.
    checks = []
    allow_tools(monkeypatch, checks)
    denials = []
    monkeypatch.setattr(adapter, "add_tool_denial",
                        lambda name, message, arguments, suggestions: denials.append((name, message)))
    agent = redirect_agent()
    call = reply(agent, [("read_file", {"path": "secret.txt", "_handler": "send"})], shape)
    if asynchronous:
        await adapter._record_output_async(call, [], agent)
        output = await agent.agent_response_async(call)
    else:
        adapter._record_output(call, [], agent)
        output = agent.agent_response(call)

    assert "[BLOCKED] read_file" in output.content
    assert "may not name its own handler" in output.content
    assert "delivered" not in output.content and "contents of" not in output.content
    assert checks == []
    assert [name for name, _ in denials] == ["read_file"]
    assert tool_result_nodes(boundary) == {}
    assert not boundary.events[adapter._version(output).canonical_id].HasField("derived_from")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_tool_class_that_declares_its_handler_still_runs(
    boundary, real_wrappers, monkeypatch, asynchronous,
):
    # The class-level declaration is the supported use of "_handler": the tool
    # is checked and recorded as usual, and its declared method runs.
    checks = []
    allow_tools(monkeypatch, checks)
    agent = redirect_agent()
    call = reply(agent, [("custom", {"text": "a note"})], "text")
    if asynchronous:
        await adapter._record_output_async(call, [], agent)
        output = await agent.agent_response_async(call)
    else:
        adapter._record_output(call, [], agent)
        output = agent.agent_response(call)

    assert "custom ran on a note" in output.content
    assert [name for name, _ in checks][0] == "custom"
    assert boundary.events[adapter._version(output).canonical_id].derived_from.name == "custom"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_nested_handler_runs_on_what_its_own_call_was_checked_against(
    boundary, real_wrappers, monkeypatch, asynchronous,
):
    # The context a handler runs in is what an HTTP request or an explicit
    # handle_message inside it is checked against. A nested call's handler must
    # run on the nested call's inputs -- its parent tool's result -- and the
    # sibling call that follows must see the reply's inputs again.
    from langroid import ToolMessage
    from sasy.instrumentation.otel.context import get_current_input_ids
    checks = []
    allow_tools(monkeypatch, checks)
    seen = {}

    class Read(ToolMessage):
        request: str = "read_file"
        purpose: str = "Read a file"
        path: str

    class Send(ToolMessage):
        request: str = "send"
        purpose: str = "Send a body"
        to: str
        body: str

    class Note(ToolMessage):
        request: str = "note"
        purpose: str = "Write a note"
        text: str

    class Worker(ChatAgent):
        def read_file(self, msg: Read) -> ToolMessage:
            seen["read_file"] = list(get_current_input_ids())
            return Send(to="partner", body=f"contents of {msg.path}")

        def send(self, msg: Send) -> str:
            seen["send"] = list(get_current_input_ids())
            return f"delivered to {msg.to}"

        def note(self, msg: Note) -> str:
            seen["note"] = list(get_current_input_ids())
            return f"noted {msg.text}"

    agent = Worker(ChatAgentConfig(name="worker", llm=None, use_tools=True, use_functions_api=False))
    agent.enable_message([Read, Send, Note])
    call = reply(agent, [("read_file", {"path": "secret/plan.txt"}), ("note", {"text": "hi"})], "text")
    before = list(_current_input_ids.get())
    if asynchronous:
        await adapter._record_output_async(call, [], agent)
        await agent.agent_response_async(call)
    else:
        adapter._record_output(call, [], agent)
        agent.agent_response(call)

    requested = adapter._version(call).canonical_id
    send_inputs = [inputs for name, inputs in checks if name == "send"][0]
    assert seen["read_file"] == [requested]
    # The send's handler runs on the read's result, which is what the send was
    # checked against, and not on the reply that asked for the read.
    assert seen["send"] == send_inputs != [requested]
    # The sibling call that follows the nested one has the reply's inputs back.
    assert seen["note"] == [requested]
    assert list(_current_input_ids.get()) == before
