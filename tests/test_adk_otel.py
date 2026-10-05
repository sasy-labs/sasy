"""Native ADK spans retain SASY correlation without becoming enforcement state."""
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import PrivateAttr
from sasy.instrumentation import adk, adk_otel
from sasy.instrumentation.config import get_config
from sasy.instrumentation.otel import exporter
from sasy.instrumentation.otel.context import _current_input_ids
from sasy.instrumentation.session import (
    GLOBAL_SESSION,
    current_wire_session_id,
    get_current_entity,
    session,
)
from test_adk_instrumentation import calls, text
from test_adk_instrumentation import sink as _sink_fixture

sink = _sink_fixture


@pytest.fixture
def tracing(monkeypatch):
    from google.adk.flows.llm_flows import base_llm_flow
    from google.adk.telemetry import node_tracing
    from google.adk.telemetry import tracing as native

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(exporter.SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    for module in (native, node_tracing, base_llm_flow):
        monkeypatch.setattr(module, "tracer", provider.get_tracer("gcp.vertex.agent"))
    monkeypatch.setattr(get_config(), "otel_enabled", True)
    yield SimpleNamespace(provider=provider, memory=memory, tracer=provider.get_tracer("test"))
    provider.shutdown()


class Model(BaseLlm):
    model: str = "trace-scripted"
    _generate: object = PrivateAttr()

    def __init__(self, generate):
        super().__init__()
        self._generate = generate

    async def generate_content_async(self, llm_request, stream=False):
        async for result in self._generate(llm_request, stream):
            yield result


async def run(agent, *, stream=False):
    adk.instrument()
    runner = InMemoryRunner(agent=agent, app_name="trace_test")
    conversation = await runner.session_service.create_session(app_name="trace_test", user_id="user")
    return [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
        new_message=types.Content(role="user", parts=[types.Part(text="synthetic input")]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE if stream else StreamingMode.NONE))]


def test_native_model_tool_spans_keep_graph_ids_and_trace_parentage(sink, tracing):
    entered = []
    def consume(value: int):
        """Consume a synthetic value."""
        entered.append((trace.get_current_span().get_span_context(), list(_current_input_ids.get())))
        return {"value": value}
    async def generate(request, stream):
        yield text("done") if any(p.function_response for c in request.contents for p in c.parts) else calls(("consume", {"value": 7}, "call-7"))
    with session("trace-session", entity="actor", end_on_exit=False), tracing.tracer.start_as_current_span("parent") as parent:
        asyncio.run(run(LlmAgent(name="worker", model=Model(generate), tools=[consume])))
        assert trace.get_current_span() is parent
        assert not _current_input_ids.get() and adk._active.get() is None
    spans = tracing.memory.get_finished_spans()
    operations = [s for s in spans if s.attributes.get("sasy.operation")]
    assert len(operations) == 3
    assert sum(s.name.startswith("generate_content") for s in operations) == 2
    assert sum(s.name == "execute_tool consume" for s in operations) == 1
    assert {s.context.trace_id for s in spans} == {parent.get_span_context().trace_id}
    assert all(s.attributes["sasy.session_id"] == "trace-session" for s in spans)
    for span in operations:
        assert set(span.attributes["_input_message_ids"].split(",")) <= sink.events.keys()
        assert span.attributes["_output_message_id"] in sink.events
        assert span.attributes["sasy.dispatch.outcome"] == "completed"
    tool = next(s for s in operations if s.name.startswith("execute_tool"))
    assert tool.attributes["sasy.adk.function_call_id"] == "call-7"
    assert tool.attributes["sasy.rm.authorized"] is True
    assert tool.attributes["sasy.rm.transform_ids"] == ()
    assert entered[0][0].span_id == tool.context.span_id
    assert entered[0][1] == tool.attributes["_input_message_ids"].split(",")


@pytest.mark.parametrize("outcome", ["denied", "transform_required", "error_result", "error"])
def test_tool_dispatch_outcomes_do_not_turn_failures_into_success(sink, tracing, outcome):
    sink.allowed = outcome != "denied"
    sink.transforms = ["rewrite"] if outcome == "transform_required" else []
    executed = []
    def consume():
        """Consume a synthetic value."""
        executed.append(True)
        if outcome == "error":
            raise ValueError("synthetic error")
        return {"error": "synthetic result"}
    async def generate(request, stream):
        yield text("done") if any(p.function_response for c in request.contents for p in c.parts) else calls(("consume", {}, "action"))
    with session("outcome", end_on_exit=False):
        if outcome == "error":
            with pytest.raises(ValueError, match="synthetic"):
                asyncio.run(run(LlmAgent(name="worker", model=Model(generate), tools=[consume])))
        else:
            asyncio.run(run(LlmAgent(name="worker", model=Model(generate), tools=[consume])))
    tool = next(s for s in tracing.memory.get_finished_spans() if s.attributes.get("sasy.operation") == "tool")
    assert tool.attributes["sasy.dispatch.outcome"] == outcome
    assert bool(executed) == (outcome in ("error_result", "error"))
    assert not any(e.HasField("derived_from") for e in sink.events.values())
    if outcome == "error":
        assert tool.attributes["sasy.error.type"] == "ValueError"
        assert "_output_message_id" not in tool.attributes


def test_after_tool_callback_error_preserves_completed_dispatch_and_native_error(sink, tracing):
    executed = []

    def consume():
        """Consume a synthetic value."""
        executed.append(True)
        return {"value": 7}

    async def after(tool, args, tool_context, tool_response):
        raise RuntimeError("synthetic callback failure")

    async def generate(request, stream):
        yield calls(("consume", {}, "callback-failure"))

    with session("callback-error", end_on_exit=False):
        with pytest.raises(RuntimeError, match="synthetic callback failure"):
            asyncio.run(run(LlmAgent(name="worker", model=Model(generate), tools=[consume],
                after_tool_callback=after)))
        assert not _current_input_ids.get() and adk._active.get() is None
    tool = next(s for s in tracing.memory.get_finished_spans() if s.name == "execute_tool consume")
    assert executed == [True]
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.attributes["sasy.dispatch.outcome"] == "completed"
    assert "sasy.execution.outcome" not in tool.attributes
    assert tool.attributes["sasy.rm.authorized"] is True
    assert tool.attributes["_output_message_id"] in sink.events


def test_streaming_ignores_chunks_and_keeps_each_final_output_link(sink, tracing):
    counts = []
    async def generate(request, stream):
        assert stream
        counts.append(len(sink.events))
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="partial")]), partial=True)
        assert len(sink.events) == counts[0]
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="first"), types.Part(text="second")]))
    with session("multipart", end_on_exit=False):
        asyncio.run(run(LlmAgent(name="worker", model=Model(generate)), stream=True))
    spans = tracing.memory.get_finished_spans()
    model = next(s for s in spans if s.attributes.get("sasy.operation") == "model")
    children = [s for s in spans if s.name == "adk.output"]
    assert len(children) == 2
    assert set(model.attributes["_output_message_ids"]) == {s.attributes["_output_message_id"] for s in children}
    assert all(s.parent.span_id == model.context.span_id for s in children)
    assert all(s.attributes["sasy.session_id"] == "multipart" for s in children)
    converted = [exporter.ObservabilitySpanExporter()._span_to_computation(s) for s in children]
    assert {c.output_message_id for c in converted} == set(model.attributes["_output_message_ids"])
    assert not any(e.text == "partial" for e in sink.events.values())


def test_cancelled_stream_ends_native_span_and_resets_context(sink, tracing):
    async def go():
        entered = asyncio.Event()
        async def generate(request, stream):
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="partial")]), partial=True)
            entered.set()
            await asyncio.Event().wait()
        with session("cancelled", end_on_exit=False):
            task = asyncio.create_task(run(LlmAgent(name="worker", model=Model(generate)), stream=True))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not _current_input_ids.get() and adk._active.get() is None
    asyncio.run(go())
    model = next(s for s in tracing.memory.get_finished_spans() if s.attributes.get("sasy.operation") == "model")
    assert model.attributes["sasy.dispatch.outcome"] == "cancelled"
    assert "_output_message_ids" not in model.attributes


def test_concurrent_tools_and_sessions_have_separate_spans(sink, tracing):
    seen = []
    async def consume(value: int):
        """Consume concurrently."""
        before = trace.get_current_span().get_span_context()
        await asyncio.sleep(0)
        assert trace.get_current_span().get_span_context() == before
        seen.append((current_wire_session_id(), before.span_id, list(_current_input_ids.get())))
        return {"value": value}
    async def generate(request, stream):
        yield text("done") if any(p.function_response for c in request.contents for p in c.parts) else calls(("consume", {"value": 1}, "one"), ("consume", {"value": 2}, "two"))
    async def one(sid):
        with session(sid, entity=sid+"-actor", end_on_exit=False):
            await run(LlmAgent(name="worker", model=Model(generate), tools=[consume]))
    async def go():
        await asyncio.gather(one("first-session"), one("second-session"))
    asyncio.run(go())
    tools = [s for s in tracing.memory.get_finished_spans() if s.attributes.get("sasy.operation") == "tool"]
    assert len(tools) == len(seen) == 4
    assert len({s.context.span_id for s in tools}) == 4
    for sid, span_id, inputs in seen:
        span = next(s for s in tools if s.context.span_id == span_id)
        assert span.attributes["sasy.session_id"] == sid
        assert span.attributes["sasy.entity"] == sid+"-actor"
        assert span.attributes["_input_message_ids"].split(",") == inputs


def test_disabled_enrichment_and_broken_span_do_not_change_actions(sink, tracing, monkeypatch):
    monkeypatch.setattr(get_config(), "otel_enabled", False)
    executed = []
    def consume():
        """Consume without SASY trace enrichment."""
        executed.append(True)
        return {}
    async def generate(request, stream):
        yield text("done") if any(p.function_response for c in request.contents for p in c.parts) else calls(("consume", {}, "disabled"))
    with session("disabled", end_on_exit=False):
        asyncio.run(run(LlmAgent(name="worker", model=Model(generate), tools=[consume])))
    assert executed == [True]
    assert not any(s.attributes.get("sasy.operation") for s in tracing.memory.get_finished_spans())
    with tracing.tracer.start_as_current_span("disabled"):
        with adk_otel.operation("tool", "agent", ["input"]) as operation:
            operation.decision(SimpleNamespace(authorized=True, transform_ids=[]))
            operation.produced(["output"])
    assert "sasy.operation" not in tracing.memory.get_finished_spans()[-1].attributes
    monkeypatch.setattr(get_config(), "otel_enabled", True)
    class Broken:
        def is_recording(self):
            return True
        def set_attributes(self, attributes):
            raise RuntimeError("exporter unavailable")
    monkeypatch.setattr(trace, "get_current_span", lambda: Broken())
    with adk_otel.operation("tool", "agent", ["input"]) as operation:
        operation.decision(SimpleNamespace(authorized=False, transform_ids=[]))
        operation.produced(["output"])


def test_additional_consumed_ids_and_repeated_outputs_keep_all_links(tracing):
    with session("additional-input", end_on_exit=False), tracing.tracer.start_as_current_span("native"):
        with adk_otel.operation("tool", "agent", ["origin"]) as operation:
            operation.consumed(["artifact", "origin"])
            operation.produced(["first"])
            operation.produced(["second", "third"])
            operation.produced(["first"])
    spans = tracing.memory.get_finished_spans()
    native = next(s for s in spans if s.name == "native")
    assert native.attributes["_input_message_ids"] == "origin,artifact"
    assert native.attributes["_output_message_ids"] == ("first", "second", "third")
    assert {s.attributes["_output_message_id"] for s in spans} == {"first", "second", "third"}
    assert all(s.attributes["_input_message_ids"] == "origin,artifact" for s in spans)


@pytest.mark.parametrize("logfire", [False, True])
def test_configure_otel_installs_scope_capture_before_batching(monkeypatch, logfire):
    from sasy.instrumentation.otel import config
    captured = []
    monkeypatch.setattr(config, "_configured", False)
    monkeypatch.setattr(config, "_config", None)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda provider: captured.extend(provider._active_span_processor._span_processors))
    fake_logfire = SimpleNamespace(configure=lambda **kwargs: captured.extend(kwargs["additional_span_processors"]))
    monkeypatch.setitem(sys.modules, "logfire", fake_logfire)
    config.configure_otel(config.OTelConfig(enabled=True, use_logfire=logfire, capture_logs=False))
    assert isinstance(captured[0], exporter.SessionSpanProcessor)
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    assert isinstance(captured[1], BatchSpanProcessor)
    captured[1].shutdown()


def test_exporter_groups_captured_scope_off_thread_and_restores_context(tracing, monkeypatch):
    with session("one", entity="actor-one", end_on_exit=False), tracing.tracer.start_as_current_span("one"):
        pass
    with session("two", entity="", end_on_exit=False), tracing.tracer.start_as_current_span("two"):
        pass
    with session(GLOBAL_SESSION, entity="", end_on_exit=False), tracing.tracer.start_as_current_span("global"):
        pass
    calls = []
    def register(computations):
        calls.append((current_wire_session_id(), get_current_entity(), computations))
    monkeypatch.setattr(exporter, "register_computations", register)
    def off_thread():
        with session("export-thread", entity="wrong-actor", end_on_exit=False):
            result = exporter.ObservabilitySpanExporter().export(tracing.memory.get_finished_spans())
            assert current_wire_session_id() == "export-thread" and get_current_entity() == "wrong-actor"
            return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(off_thread).result() == SpanExportResult.SUCCESS
    assert [(sid, actor) for sid, actor, _ in calls] == [("one", "actor-one"), ("two", ""), (None, "")]
    assert calls[0][2][0].entity == "actor-one"
    assert not calls[1][2][0].HasField("entity")


def test_export_failure_restores_scope_and_untagged_spans_are_skipped(tracing, monkeypatch):
    with session("source", entity="source-actor", end_on_exit=False), tracing.tracer.start_as_current_span("tagged"):
        pass
    def fail(computations):
        assert current_wire_session_id() == "source" and get_current_entity() == "source-actor"
        raise RuntimeError("export unavailable")
    monkeypatch.setattr(exporter, "register_computations", fail)
    with session("caller", entity="caller-actor", end_on_exit=False):
        assert exporter.ObservabilitySpanExporter().export(tracing.memory.get_finished_spans()) == SpanExportResult.FAILURE
        assert current_wire_session_id() == "caller" and get_current_entity() == "caller-actor"
        plain = TracerProvider()
        capture = InMemorySpanExporter()
        plain.add_span_processor(SimpleSpanProcessor(capture))
        with plain.get_tracer("legacy").start_as_current_span("untagged"):
            pass
        received = []
        monkeypatch.setattr(exporter, "register_computations", lambda values: received.append((current_wire_session_id(), get_current_entity())))
        assert exporter.ObservabilitySpanExporter().export(capture.get_finished_spans()) == SpanExportResult.SUCCESS
        assert received == []
        plain.shutdown()
