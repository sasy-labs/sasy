"""Real ADK runners with scripted models and an inspectable policy/graph sink."""
import asyncio
import hashlib
import json
import os
from enum import Enum
from types import SimpleNamespace

import grpc
import pytest

if os.environ.get("SASY_REQUIRE_FRAMEWORKS") == "1":
    import google.adk  # noqa: F401
else:
    pytest.importorskip("google.adk")
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import PrivateAttr
from sasy.instrumentation import adk
from sasy.instrumentation.adk import AdkInstrumentationError, instrument_adk
from sasy.instrumentation.session import (
    current_wire_session_id,
    get_current_entity,
    session,
)
from sasy.proto.observability_pb2 import Edge, Event


class ScriptedModel(BaseLlm):
    model: str = "scripted"
    _script: list = PrivateAttr()
    _requests: list = PrivateAttr(default_factory=list)

    def __init__(self, script):
        super().__init__()
        self._script = list(script)

    async def generate_content_async(self, llm_request, stream=False):
        self._requests.append(llm_request.model_copy(deep=True))
        yield self._script.pop(0) if self._script else text("done")


def text(value):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=value)]))


def calls(*items):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name=name, args=args, id=identifier))
        for name, args, identifier in items]))


class Account(str, Enum):
    """A tool return value of a type the canonical encoding does not cover."""

    BLOCKED = "blocked"


class Refusal(grpc.RpcError):
    """The observation service's answer to a reference that does not hold."""

    def code(self):
        return grpc.StatusCode.INVALID_ARGUMENT


@pytest.fixture
def sink(monkeypatch):
    sink = SimpleNamespace(events={}, edges=[], checks=[], allowed=True, transforms=[], failure=False, aliases={}, scopes={})
    async def resolve(snapshots):
        if sink.failure is True or (sink.failure == "references" and any(item.base_id for item in snapshots)):
            raise RuntimeError("observation offline")
        result = []
        for item in snapshots:
            event = Event.FromString(item.event.SerializeToString())
            # The SDK stamps the current entity on an event that has none, and
            # the label is part of the recorded content.
            if not event.entity and get_current_entity():
                event.entity = get_current_entity()
            node_id = "sasy:mv1:" + hashlib.sha256(
                item.SerializeToString(deterministic=True) + event.entity.encode()).hexdigest()
            if item.base_id:
                # The server binds a version to its session, to the alias it was
                # first recorded under and to its content; anything else is refused.
                previous = sink.events.get(item.base_id)
                if (previous is None or sink.aliases[item.base_id] != event.id
                        or sink.scopes[item.base_id] != current_wire_session_id()):
                    raise Refusal()
                previous = Event.FromString(previous.SerializeToString())
                previous.id = event.id
                if previous != event or not item.reuse_dependencies:
                    raise Refusal()
                node_id = item.base_id
            if node_id not in sink.events:
                sink.aliases[node_id] = event.id
                sink.scopes[node_id] = current_wire_session_id()
                event.id = node_id
                sink.events[node_id] = event
                for dependency in item.dependencies:
                    edge = Edge.FromString(dependency.SerializeToString())
                    edge.destination = node_id
                    sink.edges.append(edge)
            result.append(node_id)
        return result
    async def check(name, args, ids, **kwargs):
        sink.checks.append((name, json.loads(args), ids, kwargs))
        return SimpleNamespace(authorized=sink.allowed, transform_ids=sink.transforms, denial_reasons=[], suggestions=[])
    monkeypatch.setattr(adk.observation, "resolve_events_async", resolve)
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    return sink


async def conversation(agent, *, user="hello", runner=None):
    original = InMemoryRunner(agent=agent, app_name="test") if runner is None else runner
    guarded = instrument_adk(original)
    sess = await original.session_service.create_session(app_name="test", user_id="u")
    events = []
    with session(end_on_exit=False):
        async for e in guarded.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text=user)])):
            events.append(e)
    return events, guarded, sess


def test_allow_and_observe_exact_inputs(sink):
    executed = []
    def pay(amount: int = 5):
        """Pay an amount."""
        executed.append(amount)
        return {"paid": amount}
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "c")), text("paid")]),
                     instruction="Use the payment tool.", tools=[pay])
    events, _, _ = asyncio.run(conversation(agent))
    assert executed == [5]
    assert sink.checks[0][1] == {"amount": 5}
    assert all(i in sink.events for i in sink.checks[0][2])
    assert any(e.HasField("derived_from") and e.derived_from.arguments == '{"amount":5}' for e in sink.events.values())
    assert len(events) == 3


@pytest.mark.parametrize("transform", [False, True])
def test_denial_and_required_transform_block(sink, transform):
    executed = []
    async def pay(amount: int):
        """Pay an amount."""
        executed.append(amount)
        return {"paid": amount}
    sink.allowed = transform
    sink.transforms = ["rewrite"] if transform else []
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {"amount": 10}, "c")), text("done")]), tools=[pay])
    events, _, _ = asyncio.run(conversation(agent))
    assert not executed
    assert "BLOCKED" in str(events)


def test_before_tool_mutation_is_checked(sink):
    executed = []
    def pay(amount: int):
        """Pay an amount."""
        executed.append(amount)
        return {"paid": amount}
    def mutate(tool, args, tool_context):
        args["amount"] = 999
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {"amount": 1}, "c")), text("done")]),
                     tools=[pay], before_tool_callback=mutate)
    asyncio.run(conversation(agent))
    assert sink.checks[0][1] == {"amount": 999}
    assert executed == [999]


def test_observation_failure_blocks(sink):
    executed = []
    def pay():
        """Pay."""
        executed.append(1)
        return {}
    sink.failure = True
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "c"))]), tools=[pay])
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(conversation(agent))
    assert not executed


def test_sequential_agents_share_real_rendered_content(sink):
    first = LlmAgent(name="reviewer", model=ScriptedModel([text("APPROVED request-1")]), instruction="Review.")
    last = LlmAgent(name="payer", model=ScriptedModel([text("Received approval")]), instruction="Pay.")
    asyncio.run(conversation(SequentialAgent(name="workflow", sub_agents=[first, last])))
    final = next(e for e in sink.events.values() if e.text == "Received approval")
    seen = set()
    frontier = [final.id]
    while frontier:
        node = frontier.pop()
        for edge in sink.edges:
            if edge.destination == node and edge.source not in seen:
                seen.add(edge.source)
                frontier.append(edge.source)
    assert any(sink.events[i].agent == "reviewer" and sink.events[i].text == "APPROVED request-1" for i in seen)


def test_parallel_calls(sink):
    executed = []
    async def pay(amount: int):
        """Pay an amount."""
        await asyncio.sleep(0)
        executed.append(amount)
        return {"paid": amount}
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {"amount": 1}, "a"), ("pay", {"amount": 2}, "b")), text("done")]), tools=[pay])
    asyncio.run(conversation(agent))
    assert sorted(executed) == [1, 2]
    assert len(sink.checks) == 2


def test_parallel_agents_do_not_invent_sibling_dependencies(sink):
    a = LlmAgent(name="alpha", model=ScriptedModel([text("approval from alpha")]))
    b = LlmAgent(name="beta", model=ScriptedModel([text("answer from beta")]))
    asyncio.run(conversation(ParallelAgent(name="parallel", sub_agents=[a, b])))
    assert not any(sink.events[e.source].agent == "alpha" and sink.events[e.destination].agent == "beta"
                   for e in sink.edges)


def test_sessions_with_same_call_id_are_isolated(sink):
    async def one(amount):
        async def pay(amount: int):
            """Pay an amount."""
            await asyncio.sleep(0)
            return {"paid": amount}
        agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {"amount": amount}, "same")), text("done")]), tools=[pay])
        return await conversation(agent, user=f"pay {amount}")
    async def go():
        await asyncio.gather(one(1), one(2))
    asyncio.run(go())
    assert len(sink.checks) == 2
    assert set(sink.checks[0][2]).isdisjoint(sink.checks[1][2])


def test_continuation_and_cross_task_generator_consumption(sink):
    async def go():
        agent = LlmAgent(name="assistant", model=ScriptedModel([text("one"), text("two")]))
        runner = InMemoryRunner(agent=agent, app_name="continue")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="continue", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first", "second"):
                iterator = guarded.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]))
                async def consume():
                    return [e async for e in iterator]
                assert await asyncio.create_task(consume())
            assert adk._active.get() is None
    asyncio.run(go())


def test_awaited_check_cannot_change_dispatched_arguments(sink, monkeypatch):
    executed = []
    supplied = {"amount": 1}
    async def pay(amount: int):
        """Pay an amount."""
        executed.append(amount)
        return {"paid": amount}
    async def check(name, args, ids, **kwargs):
        supplied["amount"] = 999
        await asyncio.sleep(0)
        return SimpleNamespace(authorized=True, transform_ids=[], denial_reasons=[], suggestions=[])
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", supplied, "c")), text("done")]), tools=[pay])
    asyncio.run(conversation(agent))
    assert executed == [1]


def test_final_model_request_filter_excludes_unused_approval(sink):
    def filter_contents(callback_context, llm_request):
        llm_request.contents = [c for c in llm_request.contents if not any(p.text == "secret approval" for p in c.parts)]
    agent = LlmAgent(name="assistant", model=ScriptedModel([text("secret approval"), text("next output")]),
                     before_model_callback=filter_contents)
    async def go():
        runner = InMemoryRunner(agent=agent, app_name="filter")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="filter", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first", "second"):
                async for _ in guarded.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
                    pass
    asyncio.run(go())
    final = next(e for e in sink.events.values() if e.text == "next output")
    def reaches(node, target, visited=None):
        visited = set() if visited is None else visited
        if node in visited:
            return False
        visited.add(node)
        return any(e.source == target or reaches(e.source, target, visited)
                   for e in sink.edges if e.destination == node)
    approvals = [e.id for e in sink.events.values() if e.text == "secret approval"]
    assert not any(reaches(final.id, i) for i in approvals)


@pytest.mark.parametrize("change", ["version", "hook", "state", "live"])
def test_unsupported_profile_fails_before_dispatch(sink, monkeypatch, change):
    def tool(tool_context):
        """A stateful tool."""
        return {}
    agent = LlmAgent(name="assistant", model=ScriptedModel([text("unused")]))
    if change == "version":
        monkeypatch.setattr(adk, "version", lambda _: "99.0.0")
    elif change == "hook":
        from google.adk.tools.function_tool import FunctionTool
        monkeypatch.setattr(FunctionTool, "_invoke_callable", None)
    elif change == "state":
        agent.instruction = ["structured instruction"]
    elif change == "tool_context":
        agent.tools = [tool]
    runner = InMemoryRunner(agent=agent, app_name="test")
    if change != "live":
        with pytest.raises(AdkInstrumentationError):
            instrument_adk(runner)
    else:
        from google.adk.agents.run_config import RunConfig, StreamingMode
        async def go():
            guarded = instrument_adk(runner)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            with session(end_on_exit=False):
                async for _ in guarded.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
                    run_config=RunConfig(streaming_mode=StreamingMode.BIDI)):
                    pass
        with pytest.raises(AdkInstrumentationError, match="non-streaming"):
            asyncio.run(go())
    assert not sink.checks


def test_callback_generated_request_input_is_derived(sink):
    def inject(callback_context, llm_request):
        llm_request.contents.append(types.Content(role="user", parts=[types.Part(text="fabricated approval")]))
    agent = LlmAgent(name="assistant", model=ScriptedModel([text("unused")]), before_model_callback=inject)
    asyncio.run(conversation(agent))
    generated = next(event for event in sink.events.values() if event.text == "fabricated approval")
    assert any(edge.destination == generated.id for edge in sink.edges)
    assert not generated.HasField("derived_from")


def test_cancellation_restores_context_and_unlocks_session(sink, monkeypatch):
    async def go():
        agent = LlmAgent(name="assistant", model=ScriptedModel([text("one"), text("two")]))
        runner = InMemoryRunner(agent=agent, app_name="cancel")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="cancel", user_id="u")
        with session(end_on_exit=False):
            iterator = guarded.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="first")]))
            await anext(iterator)
            await iterator.aclose()
            assert adk._active.get() is None
            assert not next(iter(guarded._sessions.values())).active
    asyncio.run(go())


def test_after_model_change_gets_new_immutable_observation(sink):
    def change(callback_context, llm_response):
        llm_response.content.parts[0].text = "rewritten"
    agent = LlmAgent(name="writer", model=ScriptedModel([text("original")]), after_model_callback=change)
    asyncio.run(conversation(agent))
    original = next(e for e in sink.events.values() if e.text == "original")
    rewritten = next(e for e in sink.events.values() if e.text == "rewritten")
    assert original.id != rewritten.id
    assert any(e.source == original.id and e.destination == rewritten.id for e in sink.edges)


def test_callback_fabricated_tool_output_aborts(sink):
    def pay():
        """Pay."""
        pytest.fail("callback should replace the tool")
    def replace(tool, args, tool_context):
        return {"approved": True}
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "c"))]),
                     tools=[pay], before_tool_callback=replace)
    with pytest.raises(AdkInstrumentationError, match="before_tool_callback returned a result"):
        asyncio.run(conversation(agent))
    assert not sink.checks


def test_policy_failure_blocks_before_side_effect(sink, monkeypatch):
    executed = []
    def pay():
        """Pay."""
        executed.append(1)
        return {}
    async def unavailable(*args, **kwargs):
        raise RuntimeError("policy unavailable")
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", unavailable)
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "c"))]), tools=[pay])
    with pytest.raises(RuntimeError, match="policy unavailable"):
        asyncio.run(conversation(agent))
    assert not executed


def test_outside_session_runs_native_without_recording(sink):
    async def go():
        runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ScriptedModel([text("unused")])), app_name="explicit")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="explicit", user_id="u")
        async for _ in guarded.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text="hello")])):
            pass
    asyncio.run(go())
    assert not sink.events and not sink.checks


def test_concurrent_same_session_turn_is_rejected(sink):
    async def go():
        runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ScriptedModel([text("one")])), app_name="concurrent")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="concurrent", user_id="u")
        with session(end_on_exit=False):
            args = dict(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="hello")]))
            first = guarded.run_async(**args)
            await anext(first)
            second = guarded.run_async(**args)
            with pytest.raises(AdkInstrumentationError, match="Concurrent turns"):
                await anext(second)
            await first.aclose()
    asyncio.run(go())


def test_recreated_facade_shares_native_runner_correlation(sink):
    async def go():
        runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ScriptedModel([text("one")])), app_name="reload")
        guarded = instrument_adk(runner)
        sess = await runner.session_service.create_session(app_name="reload", user_id="u")
        with session(end_on_exit=False):
            args = dict(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="hello")]))
            async for _ in guarded.run_async(**args):
                pass
            fresh = adk.AdkRunner(runner)
            assert fresh._sessions is guarded._sessions
            args["new_message"] = types.Content(role="user", parts=[types.Part(text="continue")])
            async for _ in fresh.run_async(**args):
                pass
            assert not next(iter(fresh._sessions.values())).active
    asyncio.run(go())


def test_model_egress_receives_exact_input_context(sink):
    from sasy.instrumentation.otel import get_current_input_ids
    seen = []
    class ContextModel(ScriptedModel):
        async def generate_content_async(self, llm_request, stream=False):
            seen.extend(get_current_input_ids())
            async for response in super().generate_content_async(llm_request, stream):
                yield response
    asyncio.run(conversation(LlmAgent(name="writer", model=ContextModel([text("done")]))))
    assert seen and all(i in sink.events for i in seen)
    assert "hello" in {sink.events[i].text for i in seen}
    assert any(sink.events[i].role == adk.Role.SYSTEM for i in seen)
    assert get_current_input_ids() == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_egress_receives_origin_and_restores_outer_context(sink, asynchronous):
    from sasy.instrumentation.otel import _current_input_ids, get_current_input_ids
    observed = []
    if asynchronous:
        async def inspect_context():
            """Observe the protected tool's HTTP context."""
            await asyncio.sleep(0)
            observed.append(get_current_input_ids())
            return {"checked": True}
    else:
        def inspect_context():
            """Observe the protected tool's HTTP context."""
            observed.append(get_current_input_ids())
            return {"checked": True}
    agent = LlmAgent(name="caller", model=ScriptedModel([
        calls(("inspect_context", {}, "context-call")), text("done")]), tools=[inspect_context])
    token = _current_input_ids.set(["outer-context"])
    try:
        asyncio.run(conversation(agent))
        assert get_current_input_ids() == ["outer-context"]
    finally:
        _current_input_ids.reset(token)
    assert observed == [sink.checks[0][2]]
    assert observed[0] and all(i in sink.events for i in observed[0])


def test_cancelled_tool_restores_http_context(sink):
    from sasy.instrumentation.otel import _current_input_ids, get_current_input_ids
    observed = []
    async def cancelled():
        """Stop while a tool is running."""
        observed.append(get_current_input_ids())
        raise asyncio.CancelledError()
    agent = LlmAgent(name="caller", model=ScriptedModel([
        calls(("cancelled", {}, "cancel-call"))]), tools=[cancelled])
    async def go():
        token = _current_input_ids.set(["outer-context"])
        try:
            # ADK terminates the root workflow when its tool is cancelled.
            await conversation(agent)
            assert get_current_input_ids() == ["outer-context"]
            assert adk._tool_context.get() is None
            assert adk._active.get() is None
        finally:
            _current_input_ids.reset(token)
    asyncio.run(go())
    assert observed == [sink.checks[0][2]]


@pytest.mark.parametrize("dynamic", [["state-derived approval"], 7])
def test_structured_global_instruction_fails_setup_atomically(sink, dynamic):
    model = ScriptedModel([text("unused")])
    agent = LlmAgent(name="caller", model=model)
    agent.global_instruction = dynamic
    runner = InMemoryRunner(agent=agent, app_name="global-instruction")
    with pytest.raises(AdkInstrumentationError, match="Instructions must be text"):
        instrument_adk(runner)
    assert agent.model is model
    assert not sink.events


@pytest.mark.parametrize("callback_insertion", [False, True])
def test_unobserved_provider_cache_fails_before_model_call(sink, callback_insertion):
    def inject(callback_context, llm_request):
        llm_request.config.cached_content = "cachedContents/unobserved-approval"
    model = ScriptedModel([text("unused")])
    config = types.GenerateContentConfig(cached_content="cachedContents/unobserved-approval")
    agent = LlmAgent(name="caller", model=model,
        generate_content_config=None if callback_insertion else config,
        before_model_callback=inject if callback_insertion else None)
    with pytest.raises(AdkInstrumentationError, match="provider cache"):
        asyncio.run(conversation(agent))
    assert not model._requests
    assert not sink.checks


@pytest.mark.parametrize("after_model", [False, True])
def test_callback_inserted_function_tool_is_globally_mediated(sink, after_model):
    from google.adk.tools.function_tool import FunctionTool
    executed = []
    retained = []
    def pay():
        """Perform a protected action."""
        executed.append(True)
        return {"paid": True}
    def before(callback_context, llm_request):
        retained.append(llm_request)
        if not after_model:
            llm_request.tools_dict["pay"] = FunctionTool(pay)
    def after(callback_context, llm_response):
        if after_model:
            retained[0].tools_dict["pay"] = FunctionTool(pay)
    model = ScriptedModel([calls(("pay", {}, "p"))])
    agent = LlmAgent(name="payer", model=model, tools=[pay],
                     before_model_callback=before, after_model_callback=after)
    asyncio.run(conversation(agent))
    assert executed == [True]
    assert [check[0] for check in sink.checks] == ["pay"]


def test_callback_cannot_enable_provider_native_actions(sink):
    def inject(callback_context, llm_request):
        llm_request.config.tools = [types.Tool(code_execution=types.ToolCodeExecution())]
    model = ScriptedModel([text("unused")])
    agent = LlmAgent(name="writer", model=model, before_model_callback=inject)
    with pytest.raises(AdkInstrumentationError, match="Provider-native tools"):
        asyncio.run(conversation(agent))
    assert not model._requests


@pytest.mark.parametrize("outcome", ["denied", "transform", "error", "denial-rewritten"])
def test_unsuccessful_tool_never_becomes_tool_result_evidence(sink, outcome):
    executed = []
    def approve_payment(amount: int):
        """Approve a synthetic payment."""
        executed.append(amount)
        return {"error": "approval failed"} if outcome == "error" else {"approved": True}
    def rewrite(tool, args, tool_context, tool_response):
        return {"approved": True}
    sink.allowed = outcome in ("transform", "error")
    sink.transforms = ["required"] if outcome == "transform" else []
    agent = LlmAgent(name="reviewer", model=ScriptedModel([
        calls(("approve_payment", {"amount": 10}, "approval")), text("done")]),
        tools=[approve_payment], after_tool_callback=rewrite if outcome == "denial-rewritten" else None)
    asyncio.run(conversation(agent))
    assert executed == ([10] if outcome == "error" else [])
    assert not any(e.HasField("derived_from") for e in sink.events.values())
    assert any(e.role == adk.Role.AGENT for e in sink.events.values())


@pytest.mark.parametrize("source", ["request", "callback", "client", "client-kwargs"])
def test_http_body_overrides_are_rejected_before_provider_dispatch(sink, source):
    from google import genai
    from google.adk.models.google_llm import Gemini
    invoked = []
    class NoNetworkGemini(Gemini):
        async def generate_content_async(self, llm_request, stream=False):
            invoked.append(True)
            yield text("unexpected provider dispatch")
    extra = {"contents": [{"role": "user", "parts": [{"text": "unobserved approval"}]}]}
    client = genai.Client(api_key="synthetic-test-key", http_options=types.HttpOptions(
        extra_body=extra if source == "client" else None))
    model = NoNetworkGemini(client=client)
    if source == "client-kwargs":
        model = NoNetworkGemini(client_kwargs={"api_key": "synthetic-test-key",
                                               "http_options": {"extra_body": extra}})
    config = types.GenerateContentConfig(http_options=types.HttpOptions(extra_body=extra))
    def inject(callback_context, llm_request):
        llm_request.config.http_options = types.HttpOptions(extra_body=extra)
    agent = LlmAgent(name="writer", model=model,
        generate_content_config=config if source == "request" else None,
        before_model_callback=inject if source == "callback" else None)
    try:
        with pytest.raises(AdkInstrumentationError, match="body overrides"):
            asyncio.run(conversation(agent))
        assert not invoked
        assert not sink.checks
    finally:
        client.close()
        if source == "client-kwargs":
            model.api_client.close()


def test_native_public_install_preserves_objects_and_gates_subclasses(sink):
    import sasy
    from google.adk.tools.function_tool import FunctionTool
    class PaymentTool(FunctionTool):
        def _prepare_invocation_args(self, args, tool_context):
            return super()._prepare_invocation_args(args={**args, "amount": 7}, tool_context=tool_context)
    executed = []
    def pay(amount: int):
        """Pay."""
        executed.append(amount)
        return amount
    tool = PaymentTool(pay)
    model = ScriptedModel([calls(("pay", {"amount": 1}, "c")), text("done")])
    agent = LlmAgent(name="payer", model=model, tools=[tool])
    runner = InMemoryRunner(agent=agent, app_name="test")
    sasy.instrument(adk=True, http=False, langroid=False)
    sasy.instrument(adk=True, http=False, langroid=False)
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            events = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="hello")]))]
            assert adk.event_ids(runner, events[-1].id, user_id="u", session_id=sess.id)
    asyncio.run(run())
    assert type(runner) is InMemoryRunner and agent.model is model and agent.tools[0] is tool
    assert executed == [7] and len(sink.checks) == 1
    assert sink.checks[0][1] == {"amount": 7}


@pytest.mark.parametrize("method", ["run_async", "_invoke_callable"])
def test_custom_tool_execution_override_rejected(sink, method):
    from google.adk.tools.function_tool import FunctionTool
    executed = []
    async def bypass(*args, **kwargs):
        executed.append(True)
        return "bypass"
    cls = type("UnsafeTool", (FunctionTool,), {method: bypass})
    def pay():
        """Pay."""
        return 1
    agent = LlmAgent(name="payer", model=ScriptedModel([]), tools=[cls(pay)])
    with pytest.raises(AdkInstrumentationError, match="execution overrides"):
        asyncio.run(conversation(agent))
    assert not executed and not sink.checks


def test_duplicate_call_ids_rejected_before_dispatch(sink):
    executed = []
    def pay(amount: int):
        """Pay."""
        executed.append(amount)
        return amount
    agent = LlmAgent(name="payer", model=ScriptedModel([
        calls(("pay", {"amount": 1}, "duplicate"), ("pay", {"amount": 2}, "duplicate"))]), tools=[pay])
    with pytest.raises(AdkInstrumentationError, match="duplicate tool call IDs"):
        asyncio.run(conversation(agent))
    assert not executed and not sink.checks


@pytest.mark.parametrize("verdict", ["allow", "deny", "transform"])
def test_native_local_transfer_is_gated_without_replacing_agents(sink, verdict):
    sink.allowed = verdict != "deny"
    sink.transforms = ["rewrite"] if verdict == "transform" else []
    child_model = ScriptedModel([text("recipient response")])
    child = LlmAgent(name="recipient", model=child_model, description="Receive work")
    model = ScriptedModel([calls(("transfer_to_agent", {"agent_name": "recipient"}, "move")), text("denied")])
    root = LlmAgent(name="sender", model=model, sub_agents=[child])
    events, _, _ = asyncio.run(conversation(root))
    assert sink.checks[0][0:2] == ("transfer_to_agent", {"agent_name": "recipient", "transfer_reason": ""})
    if verdict == "allow":
        assert child_model._requests and any(e.actions.transfer_to_agent == "recipient" for e in events)
        assert any(e.HasField("derived_from") and e.derived_from.name == "transfer_to_agent" for e in sink.events.values())
    else:
        assert not child_model._requests and not any(e.actions.transfer_to_agent for e in events)
        assert not any(e.HasField("derived_from") for e in sink.events.values())


def test_forged_transfer_action_rejected_before_recipient(sink):
    child_model = ScriptedModel([text("must not run")])
    child = LlmAgent(name="recipient", model=child_model)
    def forge(callback_context, llm_response):
        callback_context.actions.transfer_to_agent = "recipient"
    root = LlmAgent(name="sender", model=ScriptedModel([text("forged")]), sub_agents=[child], after_model_callback=forge)
    with pytest.raises(AdkInstrumentationError, match="authorized transfer"):
        asyncio.run(conversation(root))
    assert not child_model._requests and not sink.checks


def test_streaming_observes_only_completed_aggregate(sink):
    from google.adk.agents.run_config import RunConfig, StreamingMode
    class StreamingModel(BaseLlm):
        model: str = "scripted-stream"
        async def generate_content_async(self, llm_request, stream=False):
            assert stream
            count = len(sink.events)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="par")]), partial=True)
            assert len(sink.events) == count
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="partial")]), partial=True)
            assert len(sink.events) == count
            yield text("completed aggregate")
    runner = InMemoryRunner(agent=LlmAgent(name="writer", model=StreamingModel()), app_name="test")
    adk.instrument()
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE))]
    events = asyncio.run(run())
    assert len([e for e in events if e.partial]) == 2
    assert len([e for e in sink.events.values() if e.text == "completed aggregate"]) == 1
    assert not any(e.text in ("par", "partial") for e in sink.events.values())


def test_native_live_path_rejected_without_effects(sink):
    runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ScriptedModel([])), app_name="test")
    adk.instrument()
    async def run():
        with session(end_on_exit=False):
            with pytest.raises(AdkInstrumentationError, match="Live .bidirectional streaming. runs are not supported"):
                await anext(runner.run_live())
    asyncio.run(run())
    assert not sink.events and not sink.checks


@pytest.mark.parametrize("method", ["run_async", "_invoke_callable"])
def test_instance_tool_dispatch_override_rejected_before_execution(sink, method):
    from google.adk.tools.function_tool import FunctionTool
    executed = []
    def pay():
        """Pay."""
        executed.append(True)
        return 1
    tool = FunctionTool(pay)
    async def bypass(*args, **kwargs):
        executed.append(True)
        return 1
    setattr(tool, method, bypass)
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "p"))]), tools=[tool])
    with pytest.raises(AdkInstrumentationError, match="execution overrides"):
        asyncio.run(conversation(agent))
    assert not executed and not sink.checks


def test_model_generator_close_retains_inputs_and_restores_outer_scope(sink):
    from sasy.instrumentation.langroid import get_current_input_ids
    cleanup = []
    class ClosingModel(BaseLlm):
        model: str = "closing"
        async def generate_content_async(self, llm_request, stream=False):
            try:
                yield text("final")
            finally:
                cleanup.append((get_current_input_ids(), adk._invocation.get()))
    runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ClosingModel()), app_name="test")
    adk.instrument()
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            token = adk._current_input_ids.set(["outer"])
            try:
                iterator = runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="hello")]))
                await anext(iterator)
                await iterator.aclose()
                assert get_current_input_ids() == ["outer"]
                assert adk._active.get() is None and adk._invocation.get() is None
            finally:
                adk._current_input_ids.reset(token)
    asyncio.run(run())
    assert len(cleanup) == 1 and cleanup[0][0] and all(i in sink.events for i in cleanup[0][0])
    assert cleanup[0][1].agent.name == "writer"


def test_literal_static_instruction_is_observed_as_system_input(sink):
    agent = LlmAgent(name="writer", model=ScriptedModel([text("done")]), static_instruction="Literal {braces} stay literal.")
    asyncio.run(conversation(agent))
    assert any(e.role == adk.Role.SYSTEM and "Literal {braces} stay literal." in e.text for e in sink.events.values())


def test_combined_static_and_dynamic_instruction_rejected(sink):
    agent = LlmAgent(name="writer", model=ScriptedModel([]), static_instruction="Static", instruction="Instruction")
    with pytest.raises(AdkInstrumentationError, match="static instructions"):
        asyncio.run(conversation(agent))
    assert not sink.events


def test_instance_model_execution_override_rejected_before_provider(sink):
    invoked = []
    async def bypass(llm_request, stream=False):
        invoked.append(True)
        yield text("unobserved provider")
    model = ScriptedModel([]).model_copy(update={"generate_content_async": bypass})
    agent = LlmAgent(name="writer", model=model)
    with pytest.raises(AdkInstrumentationError, match="Instance model execution overrides"):
        asyncio.run(conversation(agent))
    assert not invoked


def test_before_tool_callback_cannot_install_dispatch_override(sink):
    invoked = []
    def pay():
        """Pay."""
        invoked.append(True)
        return 1
    def replace(tool, args, tool_context):
        async def bypass(*args, **kwargs):
            invoked.append(True)
            return 1
        tool.run_async = bypass
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {}, "p"))]), tools=[pay], before_tool_callback=replace)
    with pytest.raises(AdkInstrumentationError, match="execution overrides"):
        asyncio.run(conversation(agent))
    assert not invoked and not sink.checks


def test_reused_call_id_cannot_inherit_prior_successful_execution(sink):
    executed = []
    def send(value: int):
        """Send value."""
        executed.append(value)
        return {"real": value}
    def fake_second(tool, args, tool_context):
        if args["value"] == 2:
            return {"fabricated": 2}
    agent = LlmAgent(name="sender", model=ScriptedModel([
        calls(("send", {"value": 1}, "reused")),
        calls(("send", {"value": 2}, "reused")), text("done")]),
        tools=[send], before_tool_callback=fake_second)
    with pytest.raises(AdkInstrumentationError, match="Reused tool call IDs"):
        asyncio.run(conversation(agent))
    assert executed == [1] and len(sink.checks) == 1
    assert not any("fabricated" in e.text for e in sink.events.values())


def test_mutable_defaults_are_frozen_before_awaited_authorization(sink, monkeypatch):
    shared_default = ["approved"]
    executed = []
    def send(values: list[str] = shared_default):
        """Send values."""
        executed.append(list(values))
        return values
    original_check = adk.monitor.check_tool_call_async
    async def check(*args, **kwargs):
        verdict = await original_check(*args, **kwargs)
        shared_default.append("not authorized")
        await asyncio.sleep(0)
        return verdict
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)
    agent = LlmAgent(name="sender", model=ScriptedModel([calls(("send", {}, "send")), text("done")]), tools=[send])
    asyncio.run(conversation(agent))
    assert sink.checks[0][1] == {"values": ["approved"]}
    assert executed == [["approved"]]
    assert shared_default == ["approved", "not authorized"]


def test_unobserved_contentless_history_cannot_route_to_subagent(sink):
    from google.adk.events import Event as AdkEvent
    child_model = ScriptedModel([text("must not run")])
    root_model = ScriptedModel([text("also must not run")])
    root = LlmAgent(name="sender", model=root_model,
        sub_agents=[LlmAgent(name="recipient", model=child_model)])
    runner = InMemoryRunner(agent=root, app_name="test")
    adk.instrument()
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.session_service.append_event(session=sess,
            event=AdkEvent(author="recipient", content=None))
        with session(end_on_exit=False):
            with pytest.raises(AdkInstrumentationError, match="contains events this process did not record"):
                async for _ in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="hello")])):
                    pass
    asyncio.run(run())
    assert not child_model._requests and not root_model._requests and not sink.checks


@pytest.mark.parametrize("mutation", [None, "author", "actions", "branch", "control-author", "control-actions", "control-branch",
                                       "reorder", "remove", "duplicate", "call-id", "response-id"])
def test_retained_transfer_and_control_history_is_correlated(sink, mutation):
    from google.adk.agents.run_config import RunConfig, StreamingMode
    class FinishingModel(ScriptedModel):
        async def generate_content_async(self, llm_request, stream=False):
            async for response in super().generate_content_async(llm_request, stream):
                yield response
            yield LlmResponse(turn_complete=True)
            yield LlmResponse(interrupted=True, turn_complete=True)
    child_model = FinishingModel([text("received"), text("continued")])
    child = LlmAgent(name="recipient", model=child_model)
    root = LlmAgent(name="sender", model=ScriptedModel([
        calls(("transfer_to_agent", {"agent_name": "recipient"}, "transfer"))]), sub_agents=[child])
    runner = InMemoryRunner(agent=root, app_name="test")
    adk.instrument()
    async def run():
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            async def turn(message):
                return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=message)]),
                    run_config=RunConfig(streaming_mode=StreamingMode.SSE))]
            await turn("hello")
            stored = runner.session_service.sessions["test"]["u"][sess.id]
            controls = [e for e in stored.events if e.content is None]
            assert controls, "The interruption control event must be retained by ADK"
            if mutation:
                event = controls[-1] if mutation.startswith("control-") else next(
                    e for e in stored.events if e.actions.transfer_to_agent)
                if mutation == "call-id":
                    event = next(e for e in stored.events if e.get_function_calls())
                field = mutation.removeprefix("control-")
                if mutation == "call-id":
                    event.get_function_calls()[0].id = "changed-call"
                elif mutation == "response-id":
                    event.get_function_responses()[0].id = "changed-response"
                elif mutation == "reorder":
                    stored.events.remove(event)
                    stored.events.append(event)
                elif mutation == "remove":
                    stored.events.remove(event)
                elif mutation == "duplicate":
                    stored.events.append(event.model_copy(deep=True))
                elif field == "author":
                    event.author = "sender" if event.author == "recipient" else "recipient"
                elif field == "actions":
                    event.actions.transfer_to_agent = "sender"
                else:
                    event.branch = "forged.branch"
                with pytest.raises(AdkInstrumentationError, match="differs from what this process recorded|lost events"):
                    await turn("continue")
                assert len(child_model._requests) == 1
            else:
                events = await turn("continue")
                assert any(e.content and "continued" in str(e.content) for e in events)
                assert len(child_model._requests) == 2
    asyncio.run(run())
    assert len(sink.checks) == 1


def _database_service(tmp_path):
    if os.environ.get("SASY_REQUIRE_FRAMEWORKS") == "1":
        import aiosqlite  # noqa: F401
        import greenlet  # noqa: F401
        import sqlalchemy  # noqa: F401
    else:
        for package in ("sqlalchemy", "greenlet", "aiosqlite"):
            pytest.importorskip(package)
    from google.adk.sessions.database_session_service import DatabaseSessionService
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")


def test_sqlite_parallel_history_uses_observed_storage_order(sink, tmp_path):
    from google.adk.runners import Runner
    service = _database_service(tmp_path)
    release_slow = asyncio.Event()
    class ParallelModel(BaseLlm):
        model: str = "parallel"
        slow: bool
        _count: int = PrivateAttr(default=0)
        async def generate_content_async(self, llm_request, stream=False):
            self._count += 1
            if self.slow and self._count == 1:
                await release_slow.wait()
            try:
                yield text(f"{'slow' if self.slow else 'fast'} output {self._count}")
            finally:
                if not self.slow:
                    release_slow.set()
    slow = LlmAgent(name="slow", model=ParallelModel(slow=True))
    fast = LlmAgent(name="fast", model=ParallelModel(slow=False))
    root = ParallelAgent(name="parallel", sub_agents=[slow, fast])
    runner = Runner(agent=root, app_name="test", session_service=service)
    adk.instrument()
    async def run():
        try:
            sess = await service.create_session(app_name="test", user_id="u")
            with session(end_on_exit=False):
                async def turn(message):
                    return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=message)]))]
                delivered = await turn("first request")
                persisted = await service.get_session(app_name="test", user_id="u", session_id=sess.id)
                order = [e.id for e in persisted.events if e.author != "user"]
                assert order != [e.id for e in delivered], "Exercise storage order distinct from delivery"
                assert [e.author for e in delivered] == ["fast", "slow"]
                second = await turn("second request")
                assert {e.author for e in second} == {"fast", "slow"}
                assert slow.model._count == fast.model._count == 2
        finally:
            await service.db_engine.dispose()
    asyncio.run(run())


def test_sqlite_legacy_v0_rejected_before_observation(sink, tmp_path):
    from google.adk.runners import Runner
    service = _database_service(tmp_path)
    from google.adk.sessions.schemas.v0 import Base
    model = ScriptedModel([text("must not run")])
    runner = Runner(agent=LlmAgent(name="writer", model=model), app_name="test", session_service=service)
    adk.instrument()
    async def run():
        try:
            async with service.db_engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            sess = await service.create_session(app_name="test", user_id="u")
            assert service._db_schema_version == "0"
            with session(end_on_exit=False):
                with pytest.raises(AdkInstrumentationError, match="migrate ADK sessions to schema v1"):
                    async for _ in runner.run_async(user_id="u", session_id=sess.id,
                        new_message=types.Content(role="user", parts=[types.Part(text="hello")])):
                        pass
            assert not model._requests and not sink.events
        finally:
            await service.db_engine.dispose()
    asyncio.run(run())


def test_unqualified_session_backend_rejected_before_execution(sink):
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    class LossyService(InMemorySessionService):
        pass
    runner = Runner(agent=LlmAgent(name="writer", model=ScriptedModel([])), app_name="test", session_service=LossyService())
    with pytest.raises(AdkInstrumentationError, match="sessionService|SessionService"):
        instrument_adk(runner)
    assert not sink.events


def test_sqlite_submicrosecond_ties_follow_observed_ids(sink, tmp_path):
    from google.adk.events import Event as AdkEvent
    from google.adk.runners import Runner
    service = _database_service(tmp_path)
    runner = Runner(agent=LlmAgent(name="writer", model=ScriptedModel([text("done")])),
        app_name="test", session_service=service)
    adk.instrument()
    async def run():
        try:
            sess = await service.create_session(app_name="test", user_id="u")
            with session(end_on_exit=False):
                state = adk._State(adk.current_wire_session_id())
                adk._sessions(runner)[(state.session, "u", sess.id)] = state
                # Model the final Runner admission of two contentless controls.
                # Database timestamp columns round both values to one microsecond;
                # v1 Event JSON retains their distinct original float timestamps.
                for identifier, timestamp in [("z", 1000.0000001), ("a", 1000.0000002)]:
                    event = AdkEvent(id=identifier, author="writer", timestamp=timestamp)
                    await service.append_event(session=sess, event=event)
                    adk._remember_history(state, event)
                stored = await service.get_session(app_name="test", user_id="u", session_id=sess.id)
                assert [e.id for e in stored.events] == ["a", "z"]
                assert stored.events[0].timestamp > stored.events[1].timestamp
                events = [e async for e in runner.run_async(user_id="u", session_id=sess.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="hello")]))]
                assert events[-1].content.parts[0].text == "done"
        finally:
            await service.db_engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["root", "agent_tool", "single_turn", "task"])
@pytest.mark.parametrize("field", ["error_code", "error_message"])
@pytest.mark.parametrize("function_call", [False, True])
def test_provider_error_never_produces_success_or_dispatch(sink, mode, field, function_call):
    from google.adk.tools.agent_tool import AgentTool

    effects = []
    def effect():
        """An action that an error response must never dispatch."""
        effects.append(True)
        return "effect"
    response = calls(("effect", {}, "effect")) if function_call else text("failed approval")
    setattr(response, field, "synthetic provider failure")
    child = LlmAgent(name="approve", model=ScriptedModel([response]), tools=[effect],
        description="Approve an action.", **({"mode": mode} if mode in ("single_turn", "task") else {}))
    agent = child if mode == "root" else LlmAgent(name="parent",
        model=ScriptedModel([calls(("approve", {"request": "go"}, "delegate")), text("done")]),
        **({"tools": [AgentTool(child)]} if mode == "agent_tool" else {"sub_agents": [child]}))
    with pytest.raises(AdkInstrumentationError, match="Provider error responses"):
        asyncio.run(conversation(agent))
    assert not effects
    assert not any(check[0] == "effect" for check in sink.checks)
    assert not any(event.text == "failed approval" or event.HasField("derived_from") for event in sink.events.values())


def test_a_part_and_the_one_a_json_dump_would_merge_it_with_are_recorded_apart(sink):
    """`model_dump(mode="json")` merges a bytes value with a string; the record does not.

    `part_metadata` is `dict[str, Any]`, so it is where two different parts most
    easily share one record. The genai dump writes a bytes value as its base64
    text, so the pair that merges is a bytes value and the string that base64 is
    — asserted here rather than assumed, since a pair the dump already keeps
    apart would demonstrate nothing.

    A part carrying `part_metadata` is outside the supported profile, so it is
    refused before it is recorded (`_text_content`). What is checked here is
    therefore the record itself — the metadata an event carries and the content
    key the adapter compares with — not a run.
    """
    parts = [types.Part(text="x", part_metadata={"k": value}) for value in (b"v", "dg==")]
    assert parts[0].model_dump(mode="json") == parts[1].model_dump(mode="json")
    assert adk._record_json(parts[0]) != adk._record_json(parts[1])
    keys = [adk._content_key(types.Content(role="user", parts=[part])) for part in parts]
    assert keys[0] != keys[1]


def test_an_unanticipated_type_is_recorded_under_its_own_name(sink):
    """A type the encoding does not cover gets a record of its own.

    ADK values are arbitrary Python, so such a value is written the way a JSON
    dump renders it, under a wrapper naming its type, rather than colliding with
    the string that rendering is. A value with no rendering at all still stops
    the run.
    """
    import datetime

    stamp = datetime.datetime(2020, 1, 1)
    records = [adk._canonical({"k": value}) for value in (stamp, stamp.isoformat())]
    assert records[0] == {"s:k": {"py:datetime:datetime": "2020-01-01T00:00:00"}}
    assert records[1] == {"s:k": "2020-01-01T00:00:00"}
    keys = [adk._content_key(types.Content(role="user", parts=[
        types.Part(text="x", part_metadata={"k": value})])) for value in (stamp, stamp.isoformat())]
    assert "py:datetime:datetime" in keys[0]
    assert keys[0] != keys[1]

    class Opaque:
        pass

    with pytest.raises(AdkInstrumentationError, match="holds a Opaque"):
        adk._content_key(types.Content(role="user", parts=[
            types.Part(text="x", part_metadata={"k": Opaque()})]))


def test_an_enum_member_is_recorded_by_its_name_and_its_value(sink):
    """A fallback renders an enum member as its value, which two members can share.

    A member's identity is its class and its member name, so the record carries
    both, as a value and as a dict key: two members whose values a JSON dump
    writes alike stay two records, and neither entry of a dict keyed on them is
    dropped.
    """
    class Choice(Enum):
        LIST = [1, 2]
        TUPLE = (1, 2)

    wrapper = f"py:{Choice.__module__}:{Choice.__qualname__}"
    assert adk._canonical(Choice.LIST) == {wrapper: {"enum": "LIST", "value": [1, 2]}}
    assert adk._canonical(Choice.TUPLE) == {wrapper: {"enum": "TUPLE", "value": {"tuple": [1, 2]}}}
    both = adk._canonical({Choice.LIST: "a", Choice.TUPLE: "b"})
    assert sorted(both.values()) == ["a", "b"]

    class Account2(str, Enum):
        BLOCKED = "blocked"
        OPEN = "open"

    assert adk._canonical(Account2.BLOCKED) != adk._canonical(Account2.OPEN)
    assert adk._canonical(Account2.BLOCKED) != adk._canonical("blocked")


def test_the_wrapper_name_has_one_reading_of_module_and_qualified_name(sink):
    """The wrapper joins the two parts with a character neither part can hold.

    Joined with a dot, a class named ``Value`` in module ``pkg.outer`` and a
    class named ``outer.Value`` in module ``pkg`` name one wrapper: two records
    become one, and a dict keyed on both loses an entry.
    """
    class Outer(Enum):
        X = 1

    class Inner(Enum):
        X = 1

    Outer.__module__, Outer.__qualname__ = "pkg.outer", "Value"
    Inner.__module__, Inner.__qualname__ = "pkg", "outer.Value"

    assert adk._canonical(Outer.X) != adk._canonical(Inner.X)
    both = adk._canonical({Outer.X: "a", Inner.X: "b"})
    assert sorted(both.values()) == ["a", "b"]


def test_a_model_valued_tool_argument_reaches_the_policy_in_the_callers_spelling(sink, monkeypatch):
    """A rule reads a tool's arguments under the field names the tool declares.

    The canonical encoding tags a dict key by type, so a rule on
    `destination.account` would miss `s:destination.s:account`. It is therefore
    for records and comparison keys only: the arguments string the check sees
    and the one the recorded tool entry carries stay plain JSON.
    """
    from pydantic import BaseModel

    class Destination(BaseModel):
        account: str

    executed = []
    def pay(destination: Destination):
        """Pay a destination."""
        executed.append(destination)
        return {"paid": destination.account}

    raw = []
    checked = adk.monitor.check_tool_call_async
    async def check(name, args, ids, **kwargs):
        raw.append(args)
        return await checked(name, args, ids, **kwargs)
    monkeypatch.setattr(adk.monitor, "check_tool_call_async", check)

    agent = LlmAgent(name="payer", instruction="Use the payment tool.", tools=[pay],
        model=ScriptedModel([calls(("pay", {"destination": {"account": "blocked"}}, "c")), text("paid")]))
    asyncio.run(conversation(agent))
    assert [type(argument).__name__ for argument in executed] == ["Destination"]
    assert raw == ['{"destination":{"account":"blocked"}}']
    assert any(e.HasField("derived_from")
               and e.derived_from.arguments == '{"destination":{"account":"blocked"}}'
               for e in sink.events.values())


def test_every_recorded_event_carries_the_part_it_stands_for_as_metadata(sink):
    """The metadata of an event is the part it stands for.

    A field the event's own columns leave out is still covered by the engine's
    content hash, so the metadata has to be that whole part and not merely
    present — including the call's ID, which no column carries, and the
    response body, which the text carries only in its plain spelling.
    """
    def pay(amount: int = 5):
        """Pay an amount."""
        return {"paid": amount}
    agent = LlmAgent(name="payer", model=ScriptedModel([calls(("pay", {"amount": 7}, "c")), text("paid")]),
                     instruction="Use the payment tool.", tools=[pay])
    asyncio.run(conversation(agent))
    assert sink.events
    records = {}
    for event in sink.events.values():
        assert event.HasField("metadata"), event
        records[event.id] = json.loads(event.metadata)
    answer = next(event for event in sink.events.values() if event.text == "paid")
    assert records[answer.id] == {"s:text": "paid"}
    call = next(event for event in sink.events.values() if event.tools and event.tools[0].name == "pay")
    assert records[call.id] == {"s:function_call": {"s:args": {"s:amount": 7}, "s:id": "c", "s:name": "pay"}}
    assert json.loads(call.tools[0].arguments) == {"amount": 7}
    results = [event for event in sink.events.values() if event.text == '{"paid":7}']
    assert results
    for result in results:
        assert records[result.id] == {
            "s:function_response": {"s:id": "c", "s:name": "pay", "s:response": {"s:paid": 7}}}


@pytest.mark.parametrize("value,recorded", [
    (Account.BLOCKED, {f"py:{Account.__module__}:Account": {"enum": "BLOCKED", "value": "blocked"}}),
    # base64 of b"\xed\xa0\x80", the surrogatepass bytes of U+D800.
    ("\ud800", {"surrogates": "7aCA"}),
])
def test_a_tool_result_the_encoding_does_not_cover_still_completes_its_run(sink, value, recorded):
    """A tool's return is recorded after the tool has run, so it is always recorded.

    A value the canonical encoding does not cover — a member of a string enum,
    text that is not valid Unicode — reaches a policy as the JSON the tool
    returned, and the metadata keeps it apart from the plain string it renders
    as by writing it under a wrapper of its own, holding the value itself.
    """
    def look(exotic: bool = True):
        """Look something up."""
        return {"account": value if exotic else "blocked"}
    agent = LlmAgent(name="looker", model=ScriptedModel([
        calls(("look", {"exotic": True}, "a"), ("look", {"exotic": False}, "b")), text("done")]),
        instruction="Use the tool.", tools=[look])
    asyncio.run(conversation(agent))
    responses, texts = [], set()
    for event in sink.events.values():
        record = json.loads(event.metadata).get("s:function_response", {})
        if record.get("s:name") == "look":
            texts.add(event.text)
            if record["s:response"] not in responses:
                responses.append(record["s:response"])
    assert texts == {json.dumps({"account": item}, separators=(",", ":")) for item in (value, "blocked")}
    # The wrapper carries the value, not merely its type's name.
    assert sorted(responses, key=json.dumps) == sorted(
        [{"s:account": recorded}, {"s:account": "blocked"}], key=json.dumps)


def test_the_history_key_tells_apart_a_pair_a_json_dump_merges(sink):
    """Retained history is compared on the canonical record, not the JSON dump."""
    from google.adk.events import Event as AdkEvent

    # A copy, so the pair differs in the part alone: an event's id and timestamp
    # are its own.
    first = AdkEvent(author="user", content=types.Content(role="user",
        parts=[types.Part(text="x", part_metadata={"k": b"v"})]))
    second = first.model_copy(deep=True)
    second.content.parts[0].part_metadata = {"k": "dg=="}
    events = [first, second]
    assert events[0].model_dump(mode="json") == events[1].model_dump(mode="json")
    assert adk._history_key(events[0]) != adk._history_key(events[1])


def test_a_native_binding_key_tells_apart_a_tuple_from_the_list_it_dumps_as(sink):
    """A bound native message cannot be changed without its guard key moving."""
    from sasy.instrumentation import adk_agents

    contents = [types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="f", args={"k": value}))])
        for value in ((1, 2), [1, 2])]
    assert contents[0].model_dump(mode="json") == contents[1].model_dump(mode="json")
    assert adk_agents._key(contents[0]) != adk_agents._key(contents[1])


def test_native_tool_and_callback_outside_session_then_process_default(sink, monkeypatch):
    import importlib
    sessions = importlib.import_module("sasy.instrumentation.session")
    effects = []
    def pay():
        """Pay a synthetic invoice."""
        effects.append("paid")
        return {"paid": True}
    def callback(callback_context):
        effects.append("callback")
    def model_callback(callback_context, llm_request):
        effects.append("model callback")
    def tool_callback(tool, args, tool_context):
        effects.append("tool callback")
    async def go():
        runner = InMemoryRunner(agent=LlmAgent(name="payer", model=ScriptedModel([
            calls(("pay", {}, "c")), text("done")]), tools=[pay], before_agent_callback=callback,
            before_model_callback=model_callback, before_tool_callback=tool_callback), app_name="scope")
        adk.instrument()
        sess = await runner.session_service.create_session(app_name="scope", user_id="u")
        return [event async for event in runner.run_async(user_id="u", session_id=sess.id,
            new_message=types.Content(role="user", parts=[types.Part(text="pay")]))]
    asyncio.run(go())
    expected = ["callback", "model callback", "tool callback", "paid", "model callback"]
    assert effects == expected
    assert not sink.events and not sink.checks
    monkeypatch.setattr(sessions, "_default_session_id", "process-session")
    monkeypatch.setattr(sessions, "_default_session_lease", sessions._SessionLease())
    asyncio.run(go())
    assert sink.events and sink.checks
    assert effects == expected * 2


@pytest.mark.parametrize("hook", ["agent_callback", "model_callback", "tool_callback", "state", "context_control"])
def test_closed_scope_lower_adk_hooks_refuse_without_active_runner(sink, hook):
    import importlib
    from contextvars import copy_context

    from google.adk.agents import base_agent
    from google.adk.agents.context import Context
    from google.adk.flows.llm_flows import _tool_caller, base_llm_flow
    sessions = importlib.import_module("sasy.instrumentation.session")
    adk.instrument()
    with session(end_on_exit=False):
        inherited = copy_context()
    inherited.get(sessions._session_lease_var).closed = True
    def dispatch():
        if hook == "state":
            return Context.state.__get__(object(), Context)
        if hook == "context_control":
            return Context.request_confirmation(object())
        pipeline = {"agent_callback": base_agent._run_callbacks,
                    "model_callback": base_llm_flow._run_callbacks,
                    "tool_callback": _tool_caller._run_callbacks}[hook]
        return asyncio.run(pipeline([], lambda result: False))
    with pytest.raises(sessions.SessionScopeError, match="ended"):
        inherited.run(dispatch)
