"""Span routing survives OTel attribute eviction and exporter thread changes."""
import gc
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sasy.instrumentation.otel import exporter
from sasy.instrumentation.session import (
    GLOBAL_SESSION,
    _entity_var,
    _session_id_var,
    current_wire_session_id,
    get_current_entity,
)
from sasy.instrumentation.session import (
    session as session_context,
)


def setup_tracing(limit=128):
    provider = TracerProvider(span_limits=SpanLimits(max_attributes=limit), shutdown_on_exit=False)
    processor = exporter.SessionSpanProcessor()
    memory = InMemorySpanExporter()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    return provider, processor, memory, provider.get_tracer("session-scope-test")


def start_scoped(tracer, session, entity):
    with session_context(session, entity=entity, end_on_exit=False):
        return tracer.start_span("scoped")


@pytest.mark.parametrize("limit", [128, 0])
def test_evicted_scope_stays_with_span_across_export_threads(monkeypatch, limit):
    provider, processor, memory, tracer = setup_tracing(limit)
    try:
        for session, entity in [("source", "actor"), (GLOBAL_SESSION, "")]:
            span = start_scoped(tracer, session, entity)
            for index in range(140):
                span.set_attribute(f"attribute.{index}", index)
            span.end()
        spans = memory.get_finished_spans()
        assert all("sasy.session_id" not in span.attributes for span in spans)
        assert not processor._pending
        calls = []
        def register(computations):
            calls.append((current_wire_session_id(), get_current_entity(), computations))
        monkeypatch.setattr(exporter, "register_computations", register)
        def export():
            session_token = _session_id_var.set("wrong-worker-session")
            entity_token = _entity_var.set("wrong-worker-actor")
            try:
                assert exporter.ObservabilitySpanExporter().export(spans) == SpanExportResult.SUCCESS
                assert current_wire_session_id() == "wrong-worker-session"
                assert get_current_entity() == "wrong-worker-actor"
            finally:
                _entity_var.reset(entity_token)
                _session_id_var.reset(session_token)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(export).result()
        assert [(sid, actor) for sid, actor, _ in calls] == [("source", "actor"), (None, "")]
        assert calls[0][2][0].entity == "actor"
        assert not calls[1][2][0].HasField("entity")
        for (_, _, values), expected in zip(calls, [("source", "actor"), ("", "")], strict=True):
            attributes = json.loads(values[0].attributes_json)
            assert (attributes["sasy.session_id"], attributes["sasy.entity"]) == expected
    finally:
        provider.shutdown()


def test_late_attribute_changes_do_not_rewrite_captured_scope(monkeypatch):
    provider, processor, memory, tracer = setup_tracing()
    try:
        span = start_scoped(tracer, "source", "actor")
        span.set_attribute("sasy.session_id", "replacement")
        span.set_attribute("sasy.entity", "replacement")
        span.end()
        calls = []
        monkeypatch.setattr(exporter, "register_computations", lambda values: calls.append(
            (current_wire_session_id(), get_current_entity(), values[0].entity, json.loads(values[0].attributes_json))))
        assert exporter.ObservabilitySpanExporter().export(memory.get_finished_spans()) == SpanExportResult.SUCCESS
        assert calls == [("source", "actor", "actor", {"sasy.session_id": "source", "sasy.entity": "actor"})]
        assert not processor._pending
    finally:
        provider.shutdown()


def test_abandoned_and_shutdown_spans_release_pending_state(monkeypatch):
    provider, processor, memory, tracer = setup_tracing(0)
    try:
        span = start_scoped(tracer, "abandoned", "actor")
        assert len(processor._pending) == 1
        del span
        gc.collect()
        assert not processor._pending
        span = start_scoped(tracer, "unfinished", "actor")
        processor.shutdown()
        assert not processor._pending
        span.end()
        calls = []
        monkeypatch.setattr(exporter, "register_computations", calls.append)
        assert exporter.ObservabilitySpanExporter().export(memory.get_finished_spans()) == SpanExportResult.FAILURE
        assert not calls
    finally:
        provider.shutdown()


def test_export_error_keeps_retry_scope_and_cleans_context(monkeypatch):
    provider, processor, memory, tracer = setup_tracing(0)
    try:
        start_scoped(tracer, "source", "actor").end()
        assert not processor._pending
        calls = []
        def register(values):
            calls.append((current_wire_session_id(), get_current_entity()))
            if len(calls) == 1:
                raise RuntimeError("offline")
        monkeypatch.setattr(exporter, "register_computations", register)
        session_token = _session_id_var.set("worker")
        entity_token = _entity_var.set("worker-actor")
        try:
            target = exporter.ObservabilitySpanExporter()
            assert target.export(memory.get_finished_spans()) == SpanExportResult.FAILURE
            assert current_wire_session_id() == "worker"
            assert get_current_entity() == "worker-actor"
            assert target.export(memory.get_finished_spans()) == SpanExportResult.SUCCESS
            assert calls == [("source", "actor"), ("source", "actor")]
        finally:
            _entity_var.reset(entity_token)
            _session_id_var.reset(session_token)
    finally:
        provider.shutdown()


def test_inactive_and_attribute_only_spans_are_not_exported(monkeypatch):
    import importlib
    scope = importlib.import_module("sasy.instrumentation.session")
    monkeypatch.setattr(scope, "_default_session_id", None)
    monkeypatch.setattr(scope, "_default_session_lease", None)
    provider, processor, memory, tracer = setup_tracing()
    try:
        with tracer.start_as_current_span("outside") as span:
            # User attributes cannot opt an unprotected span into recording.
            span.set_attribute("sasy.session_id", "forged")
            span.set_attribute("sasy.entity", "actor")
        unbound = TracerProvider(shutdown_on_exit=False)
        unbound.add_span_processor(SimpleSpanProcessor(memory))
        with unbound.get_tracer("raw").start_as_current_span("untagged"):
            pass
        calls = []
        monkeypatch.setattr(exporter, "register_computations", calls.append)
        assert exporter.ObservabilitySpanExporter().export(memory.get_finished_spans()) == SpanExportResult.SUCCESS
        assert calls == []
        unbound.shutdown()
    finally:
        provider.shutdown()


def test_completed_spans_flush_before_close_and_late_spans_cannot_revive_scope(monkeypatch):
    import importlib
    scope = importlib.import_module("sasy.instrumentation.session")
    provider, processor, memory, tracer = setup_tracing()
    calls = []
    monkeypatch.setattr(exporter, "register_computations", lambda values: calls.append(current_wire_session_id()))
    monkeypatch.setattr("sasy.policy.api.end_session", lambda *_: None)
    target = exporter.ObservabilitySpanExporter()
    flush_results = []
    def flush():
        flush_results.append(target.export(memory.get_finished_spans()))
        memory.clear()
    monkeypatch.setattr(scope, "_session_flush_callbacks", [flush])
    try:
        with session_context("closing"):
            tracer.start_span("completed").end()
            late = tracer.start_span("still-running")
        assert flush_results == [SpanExportResult.SUCCESS]
        assert calls == ["closing"]
        late.end()
        assert target.export(memory.get_finished_spans()) == SpanExportResult.FAILURE
        assert calls == ["closing"]
    finally:
        provider.shutdown()


def test_closed_scope_in_mixed_batch_does_not_drop_active_scope(monkeypatch):
    provider, processor, memory, tracer = setup_tracing()
    monkeypatch.setattr("sasy.policy.api.end_session", lambda *_: None)
    calls = []
    monkeypatch.setattr(exporter, "register_computations", lambda values: calls.append(current_wire_session_id()))
    try:
        with session_context("closed"):
            tracer.start_span("completed-before-close").end()
        with session_context("active", end_on_exit=False):
            tracer.start_span("active").end()
            assert exporter.ObservabilitySpanExporter().export(memory.get_finished_spans()) == SpanExportResult.FAILURE
            assert calls == ["active"]
            assert current_wire_session_id() == "active"
    finally:
        provider.shutdown()
