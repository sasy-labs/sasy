"""Bounded state production and exact template consumption on native ADK."""
import asyncio

import pytest
from sasy.instrumentation.adk_state import EXTERNAL_SESSION_ORIGIN
from test_adk_instrumentation import (  # noqa: F401
    AdkInstrumentationError,
    InMemoryRunner,
    LlmAgent,
    ParallelAgent,
    ScriptedModel,
    SequentialAgent,
    adk,
    calls,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink


def ancestors(sink, nodes):
    result = set(nodes)
    pending = list(nodes)
    while pending:
        node = pending.pop()
        for edge in sink.edges:
            if edge.destination == node and edge.source not in result:
                result.add(edge.source)
                pending.append(edge.source)
    # Batch-local dependency sources are named by their pre-resolution event id.
    return [sink.events[node] for node in result if node in sink.events]


async def run_agent(agent, initial=None, *, runner=None, user="u"):
    runner = runner or InMemoryRunner(agent=agent, app_name="test")
    adk.instrument()
    sess = await runner.session_service.create_session(app_name="test", user_id=user, state=initial)
    with session(end_on_exit=False):
        result = [event async for event in runner.run_async(user_id=user, session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text="request")]))]
    return runner, sess, result


@pytest.mark.parametrize("approved", [True, False])
def test_output_key_template_preserves_real_producer(sink, approved):
    def approve():
        """Approve."""
        return "approval"
    def consume():
        """Consume."""
        return "consumed"
    writer_script = [calls(("approve", {}, "approve")), text("approved")] if approved else [text("approved")]
    writer = LlmAgent(name="writer", model=ScriptedModel(writer_script), tools=[approve], output_key="decision")
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]),
        instruction="Decision: {decision}", include_contents="none", tools=[consume])
    asyncio.run(run_agent(SequentialAgent(name="pipeline", sub_agents=[writer, reader]),
        {"unread": "unrelated secret"}))
    check = next(check for check in sink.checks if check[0] == "consume")
    lineage = ancestors(sink, check[2])
    assert any("observed production" in event.text and "approved" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage) == approved
    assert not any("unrelated secret" in event.text for event in sink.events.values())


@pytest.mark.parametrize("key", ["decision", "app:decision", "user:decision"])
def test_only_referenced_initial_key_is_unattributed_input(sink, key):
    agent = LlmAgent(name="reader", model=ScriptedModel([text("done")]), instruction="Value: {" + key + "}")
    asyncio.run(run_agent(agent, {key: {"nested": ["value"]}, "unread": "unrelated"}))
    nodes = [event for event in sink.events.values() if "adk_resource" in event.text]
    assert len(nodes) == 1
    assert "unattributed external input" in nodes[0].text
    assert not nodes[0].HasField("derived_from")
    assert not any("unrelated" in event.text for event in sink.events.values())


def test_callback_session_access_fails_before_model(sink):
    def callback(callback_context, llm_request):
        llm_request.config.system_instruction = callback_context.session.state["approval"]
    model = ScriptedModel([text("must not run")])
    agent = LlmAgent(name="reader", model=model, before_model_callback=callback)
    with pytest.raises(AdkInstrumentationError, match="context state/session"):
        asyncio.run(run_agent(agent, {"approval": "hidden"}))
    assert not model._requests
    assert not any("hidden" in event.text for event in sink.events.values())


def test_callback_generated_instruction_derives_from_observed_request(sink):
    def callback(callback_context, llm_request):
        llm_request.config.system_instruction = "application instruction"
    model = ScriptedModel([text("done")])
    asyncio.run(run_agent(LlmAgent(name="reader", model=model, before_model_callback=callback)))
    instruction = next(event for event in sink.events.values() if event.text == "application instruction")
    assert any(edge.destination == instruction.id for edge in sink.edges)


def test_duplicate_agent_names_fail_before_observation(sink):
    root = ParallelAgent(name="root", sub_agents=[LlmAgent(name="same", model=ScriptedModel([])),
        LlmAgent(name="same", model=ScriptedModel([]))])
    with pytest.raises(AdkInstrumentationError, match="names must be unique"):
        asyncio.run(run_agent(root))
    assert not sink.events


@pytest.mark.parametrize("field", ["cache_config", "cache_metadata", "previous_interaction_id"])
def test_finalized_native_cache_request_is_rejected(sink, field):
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.models.cache_metadata import CacheMetadata
    def callback(callback_context, llm_request):
        values = {"cache_config": ContextCacheConfig(), "cache_metadata": CacheMetadata(fingerprint="cache", contents_count=1),
                  "previous_interaction_id": "interaction"}
        setattr(llm_request, field, values[field])
    model = ScriptedModel([text("must not run")])
    with pytest.raises(AdkInstrumentationError, match="context caches"):
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, before_model_callback=callback)))
    assert not model._requests


def test_native_app_cache_is_rejected_before_observation(sink):
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    runner = Runner(app=App(name="test", root_agent=LlmAgent(name="reader", model=ScriptedModel([])),
        context_cache_config=ContextCacheConfig()), session_service=InMemorySessionService())
    with pytest.raises(AdkInstrumentationError, match="context caches"):
        adk.instrument_adk(runner)
    assert not sink.events


@pytest.mark.parametrize("key", ["decision", "app:decision", "user:decision", "temp:decision"])
def test_output_key_scopes_propagate_within_invocation(sink, key):
    writer = LlmAgent(name="writer", model=ScriptedModel([text("value")]), output_key=key)
    reader = LlmAgent(name="reader", model=ScriptedModel([text("read")]), instruction="Value: {" + key + "}", include_contents="none")
    asyncio.run(run_agent(SequentialAgent(name="sequence", sub_agents=[writer, reader])))
    output = next(event for event in sink.events.values() if event.text == "read")
    assert any(event.text == "value" for event in ancestors(sink, [output.id]))


def test_nested_external_state_mutation_is_rejected_and_claim_released(sink):
    async def run():
        model = ScriptedModel([text("first"), text("third")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, instruction="Value {item}"), app_name="test")
        adk.instrument()
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"item": {"nested": [1]}})
        with session(end_on_exit=False):
            async def turn(message):
                return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=message)]))]
            await turn("first")
            stored = runner.session_service.sessions["test"]["u"][sess.id]
            stored.state["item"]["nested"].append(2)
            with pytest.raises(AdkInstrumentationError, match="without an observed producer"):
                await turn("second")
            assert len(model._requests) == 1
            stored.state["item"]["nested"].pop()
            await turn("third")
            assert len(model._requests) == 2
    asyncio.run(run())


@pytest.mark.parametrize("same_graph", [True, False])
def test_shared_app_state_preserves_graph_scope(sink, same_graph):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    service = InMemorySessionService()
    writer = Runner(agent=LlmAgent(name="writer", model=ScriptedModel([text("produced")]), output_key="app:value"),
        app_name="test", session_service=service)
    reader_model = ScriptedModel([text("read")])
    reader = Runner(agent=LlmAgent(name="reader", model=reader_model, instruction="Value {app:value}", include_contents="none"),
        app_name="test", session_service=service)
    adk.instrument()
    async def turn(runner, name):
        sess = await service.create_session(app_name="test", user_id="u", session_id=name)
        return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text=name)]))]
    async def run():
        with session(end_on_exit=False):
            await turn(writer, "writer")
            written = set(sink.events)
            if same_graph:
                await turn(reader, "reader")
        if not same_graph:
            with session(end_on_exit=False):
                await turn(reader, "reader")
        assert len(reader_model._requests) == 1
        lineage = ancestors(sink, [next(e for e in sink.events.values() if e.text == "read").id])
        assert any(e.text == "produced" for e in lineage) is same_graph
        external = [e for e in lineage if EXTERNAL_SESSION_ORIGIN in e.text]
        if same_graph:
            assert not external
            return
        assert len(external) == 1
        assert "unattributed external input" in external[0].text and '"produced"' in external[0].text
        assert not any(e.id in written for e in lineage)
    asyncio.run(run())


def test_cross_graph_approval_does_not_authorize_the_reading_graph(sink):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    def approve():
        """Approve."""
        return "approval"
    def consume():
        """Consume."""
        return "consumed"
    service = InMemorySessionService()
    writer = Runner(app_name="test", session_service=service, agent=LlmAgent(name="writer", output_key="app:decision",
        model=ScriptedModel([calls(("approve", {}, "approve")), text("approved")]), tools=[approve]))
    reader = Runner(app_name="test", session_service=service, agent=LlmAgent(name="reader", tools=[consume],
        model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]),
        instruction="Decision: {app:decision}", include_contents="none"))
    adk.instrument()
    async def turn(runner, name):
        sess = await service.create_session(app_name="test", user_id="u", session_id=name)
        return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text=name)]))]
    written = set()
    async def run():
        with session(end_on_exit=False):
            await turn(writer, "writer")
        written.update(sink.events)
        with session(end_on_exit=False):
            await turn(reader, "reader")
    asyncio.run(run())
    lineage = ancestors(sink, next(check for check in sink.checks if check[0] == "consume")[2])
    assert any(EXTERNAL_SESSION_ORIGIN in event.text and "approved" in event.text for event in lineage)
    assert not any(event.HasField("derived_from") for event in lineage)
    assert not any(event.id in written for event in lineage)


def test_cross_graph_value_read_twice_is_one_node(sink):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    service = InMemorySessionService()
    writer = Runner(agent=LlmAgent(name="writer", model=ScriptedModel([text("produced")]), output_key="app:value"),
        app_name="test", session_service=service)
    reader = Runner(agent=LlmAgent(name="reader", model=ScriptedModel([text("first"), text("second")]),
        instruction="Value {app:value}", include_contents="none"), app_name="test", session_service=service)
    adk.instrument()
    async def turn(runner, name):
        sess = await service.create_session(app_name="test", user_id="u", session_id=name)
        return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text=name)]))]
    async def run():
        with session(end_on_exit=False):
            await turn(writer, "writer")
        with session(end_on_exit=False):
            await turn(reader, "one")
            await turn(reader, "two")
    asyncio.run(run())
    external = [event for event in sink.events.values() if EXTERNAL_SESSION_ORIGIN in event.text]
    assert len(external) == 1
    for name in ("first", "second"):
        result = next(event for event in sink.events.values() if event.text == name)
        assert external[0].id in {event.id for event in ancestors(sink, [result.id])}


def test_parallel_state_operations_reject_without_poisoning_service(sink):
    root = ParallelAgent(name="parallel", sub_agents=[
        LlmAgent(name="a", model=ScriptedModel([text("a")]), instruction="Read {key}"),
        LlmAgent(name="b", model=ScriptedModel([text("b")]))])
    with pytest.raises(AdkInstrumentationError, match="concurrency qualification"):
        asyncio.run(run_agent(root, {"key": "value"}))


def test_deferred_resource_context_is_revoked(sink):
    from sasy.instrumentation import adk_state
    async def run():
        released = asyncio.Event()
        tasks = []
        async def schedule():
            """Schedule a delayed read."""
            frame = adk_state._frame.get()
            async def delayed():
                await released.wait()
                with pytest.raises(AdkInstrumentationError, match="outlived"):
                    frame.resources.claim(frame.resources.runner.session_service)
            tasks.append(asyncio.create_task(delayed()))
            return "scheduled"
        agent = LlmAgent(name="writer", model=ScriptedModel([calls(("schedule", {}, "schedule")), text("done")]), tools=[schedule])
        await run_agent(agent)
        released.set()
        await asyncio.gather(*tasks)
    asyncio.run(run())


def test_late_tool_instruction_renderer_keeps_state_dependencies(sink):
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.utils import instructions_utils
    class LateInstructionTool(FunctionTool):
        async def process_llm_request(self, *, tool_context, llm_request):
            await super().process_llm_request(tool_context=tool_context, llm_request=llm_request)
            value = await instructions_utils.inject_session_state("Late {decision}", tool_context)
            llm_request.append_instructions([value])
    def consume():
        """Consume."""
        return "done"
    agent = LlmAgent(name="reader", model=ScriptedModel([text("result")]), tools=[LateInstructionTool(consume)])
    asyncio.run(run_agent(agent, {"decision": "observed value"}))
    output = next(event for event in sink.events.values() if event.text == "result")
    assert any("adk_resource" in event.text and "observed value" in event.text for event in ancestors(sink, [output.id]))


def test_gemini_static_continuation_prompt_is_observed_before_provider(sink):
    from google import genai
    from google.adk.models.google_llm import Gemini
    invoked = []
    class NoNetworkGemini(Gemini):
        async def generate_content_async(self, llm_request, stream=False):
            invoked.extend(llm_request.contents)
            from sasy.instrumentation.otel import get_current_input_ids
            ids = get_current_input_ids()
            assert any("Handle the requests" in sink.events[node].text for node in ids)
            before = len(llm_request.contents)
            self._maybe_append_user_content(llm_request)
            assert len(llm_request.contents) == before
            yield text("done")
    client = genai.Client(api_key="synthetic-test-key")
    try:
        model = NoNetworkGemini(client=client)
        def omit_messages(callback_context, llm_request):
            llm_request.contents = []
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, instruction="Respond.",
            include_contents="none", before_model_callback=omit_messages)))
        assert len(invoked) == 1
    finally:
        client.close()


@pytest.mark.parametrize("agent_type", [SequentialAgent, ParallelAgent])
def test_agent_callback_supplied_content_is_rejected(sink, agent_type):
    def callback(callback_context):
        return types.Content(role="model", parts=[types.Part(text="fabricated")])
    model = ScriptedModel([text("must not run")])
    agent = agent_type(name="root", sub_agents=[LlmAgent(name="child", model=model)], before_agent_callback=callback)
    with pytest.raises(AdkInstrumentationError, match="agent_callback returned content"):
        asyncio.run(run_agent(agent))
    assert not model._requests
    assert not any("fabricated" in event.text for event in sink.events.values())


def test_failed_output_persistence_keeps_previous_state_version(sink, monkeypatch):
    from google.adk.sessions import InMemorySessionService
    adk.instrument()
    original = InMemorySessionService.append_event
    failed = False
    async def append(service, session, event):
        nonlocal failed
        if event.actions.state_delta and not failed:
            failed = True
            raise RuntimeError("transient append failure")
        return await original(service, session, event)
    monkeypatch.setattr(InMemorySessionService, "append_event", append)
    async def run():
        model = ScriptedModel([text("new"), text("committed")])
        runner = InMemoryRunner(agent=LlmAgent(name="writer", model=model,
            instruction="Previous {value}", output_key="value"), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"value": "old"})
        with session(end_on_exit=False):
            async def turn(message):
                return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=message)]))]
            with pytest.raises(RuntimeError, match="transient append failure"):
                await turn("first")
            stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id)
            assert stored.state["value"] == "old"
            await turn("retry")
            assert len(model._requests) == 2
            stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id)
            assert stored.state["value"] == "committed"
    asyncio.run(run())


def test_tool_request_processor_cannot_read_public_session_state(sink):
    from google.adk.tools.function_tool import FunctionTool
    def harmless():
        """Do nothing."""
        return "done"
    class StatefulTool(FunctionTool):
        async def process_llm_request(self, *, tool_context, llm_request):
            hidden = tool_context.session.state["approval"]
            llm_request.append_instructions([hidden])
            await super().process_llm_request(tool_context=tool_context, llm_request=llm_request)
    model = ScriptedModel([text("must not run")])
    with pytest.raises(AdkInstrumentationError, match="context state/session"):
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, tools=[StatefulTool(harmless)]),
            {"approval": "hidden approval"}))
    assert not model._requests
    assert not any("hidden approval" in event.text for event in sink.events.values())


def approve(tool_context):
    """Approve."""
    tool_context.state["approval"] = "granted"
    return "approved"


def consume():
    """Consume."""
    return "consumed"


async def run_provider_turns(provider, initial=None, *, global_instruction=""):
    """Approve in a first turn, then read that approval only through the provider."""
    model = ScriptedModel([calls(("approve", {}, "approve")), text("noted"),
                           calls(("consume", {}, "consume")), text("done")])
    agent = LlmAgent(name="reader", model=model, tools=[approve, consume], include_contents="none",
        instruction=provider, global_instruction=global_instruction)
    runner = InMemoryRunner(agent=agent, app_name="test")
    adk.instrument()
    sess = await runner.session_service.create_session(app_name="test", user_id="u", state=initial)
    with session(end_on_exit=False):
        for message in ("approve this", "continue"):
            async for _ in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=message)])):
                pass
    return model


def consume_lineage(sink):
    check = next(check for check in sink.checks if check[0] == "consume")
    return ancestors(sink, check[2])


def approval_evidence(lineage):
    return any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)


def test_instruction_provider_depends_only_on_read_keys(sink):
    def provider(context):
        return "Approval: " + context.state.get("approval", "pending")
    asyncio.run(run_provider_turns(provider, {"unread": "unrelated secret"}))
    assert approval_evidence(consume_lineage(sink))
    assert not any("unrelated secret" in event.text for event in sink.events.values())


def test_unread_key_approval_does_not_reach_a_later_tool(sink):
    def provider(context):
        return "Static guidance"
    asyncio.run(run_provider_turns(provider))
    assert not approval_evidence(consume_lineage(sink))
    assert not any("adk_resource" in event.text for event in consume_lineage(sink))


def test_instruction_provider_presence_check_carries_the_producer(sink):
    def provider(context):
        return "Decided" if "approval" in context.state else "Undecided"
    model = asyncio.run(run_provider_turns(provider))
    assert "Decided" in str(model._requests[-1].config.system_instruction)
    assert approval_evidence(consume_lineage(sink))


def test_instruction_provider_presence_check_does_not_read_the_value(sink):
    def provider(context):
        return "Decided" if "approval" in context.state else "Undecided"
    asyncio.run(run_agent(LlmAgent(name="reader", model=ScriptedModel([text("done")]), instruction=provider),
        {"approval": "granted secret"}))
    nodes = [event for event in sink.events.values() if "adk_resource" in event.text]
    assert len(nodes) == 1
    assert "state-presence" in nodes[0].text
    assert not any("granted secret" in event.text for event in sink.events.values())


def test_async_instruction_provider_records_its_reads(sink):
    async def provider(context):
        await asyncio.sleep(0)
        return "Approval: " + context.state.get("approval", "pending")
    asyncio.run(run_provider_turns(provider))
    assert approval_evidence(consume_lineage(sink))


def test_global_instruction_provider_records_its_reads(sink):
    def provider(context):
        return "Approval: " + context.state.get("approval", "pending")
    asyncio.run(run_provider_turns("", global_instruction=provider))
    assert approval_evidence(consume_lineage(sink))


@pytest.mark.parametrize("attribute", ["session", "user_content", "custom_metadata",
                                       "_invocation_context", "_frame"])
def test_instruction_provider_untracked_context_access_is_blocked(sink, attribute):
    def provider(context):
        return str(getattr(context, attribute))
    model = ScriptedModel([text("must not run")])
    with pytest.raises(AdkInstrumentationError, match="requires additional instrumentation"):
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, instruction=provider),
            {"approval": "hidden approval"}))
    assert not model._requests
    assert not any("hidden approval" in event.text for event in sink.events.values())


def test_instruction_provider_cannot_read_state_through_the_raw_invocation(sink):
    def provider(context):
        return "Approval: " + context._invocation_context.session.state["approval"]
    model = ScriptedModel([text("must not run")])
    with pytest.raises(AdkInstrumentationError, match="requires additional instrumentation"):
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, instruction=provider),
            {"approval": "hidden approval"}))
    assert not model._requests
    assert not any("hidden approval" in event.text for event in sink.events.values())


def test_instruction_provider_must_return_text(sink):
    model = ScriptedModel([text("must not run")])
    with pytest.raises(AdkInstrumentationError, match="must return text"):
        asyncio.run(run_agent(LlmAgent(name="reader", model=model, instruction=lambda context: ["not text"])))
    assert not model._requests
