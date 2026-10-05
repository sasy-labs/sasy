"""
OpenTelemetry Span Exporter for the Observability Server.

This module implements a custom SpanExporter that sends OTel spans
to the observability server via gRPC, which then stores them as
Computation nodes in Neo4j.
"""

import json
import weakref
from collections.abc import Sequence
from threading import RLock

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from sasy.capture import capture_logger
from sasy.instrumentation.session import (
    CapturedSessionScope,
    bind_session_scope,
    capture_session_scope,
    current_wire_session_id,
    get_current_entity,
    is_session_active,
)
from sasy.observability.api import register_computations
from sasy.proto.observability_pb2 import Computation, SpanStatusCode

logger = capture_logger(__name__)


def _span_scope(span):
    if hasattr(span, "_sasy_captured_scope"):
        scope = span._sasy_captured_scope
        if not isinstance(scope, tuple) or len(scope) != 2 or not all(isinstance(value, str) for value in scope):
            raise ValueError("Missing or invalid captured span scope")
        return scope
    attributes = span.attributes or {}
    return attributes.get("sasy.session_id"), attributes.get("sasy.entity")


class SessionSpanProcessor(SpanProcessor):
    """Capture task-local scope before a batch exporter moves to another thread.

    Empty session means explicit tenant-global. Inactive spans are skipped.
    Captured leases preserve scope across exporter threads and prohibit late
    exports after closure. Labels do not establish authenticated identity.
    """
    def __init__(self):
        self._lock = RLock()
        self._pending = {}
        self._closed = False

    def on_start(self, span, parent_context=None):
        active = is_session_active()
        binding = capture_session_scope() if active else None
        scope = (current_wire_session_id() or "", get_current_entity() or "") if active else None
        key = (span.context.trace_id, span.context.span_id)

        def abandoned(reference):
            with self._lock:
                entry = self._pending.get(key)
                if entry is not None and entry[0] is reference:
                    self._pending.pop(key)

        # SDK Span.end constructs a separate ReadableSpan. Keep only a weak
        # reference until on_end transfers scope independently of attributes.
        with self._lock:
            if not self._closed:
                self._pending[key] = (weakref.ref(span, abandoned), scope, binding)
        if scope is not None:
            span.set_attribute("sasy.session_id", scope[0])
            span.set_attribute("sasy.entity", scope[1])

    def on_end(self, span):
        key = (span.context.trace_id, span.context.span_id)
        with self._lock:
            entry = self._pending.pop(key, None)
        # Missing processor state must fail closed, never become a legacy span
        # routed using the exporter thread's session.
        span._sasy_captured_scope = entry[1] if entry is not None else None
        span._sasy_session_binding = entry[2] if entry is not None else None
        span._sasy_inactive = entry is not None and entry[1] is None

    def shutdown(self):
        with self._lock:
            self._closed = True
            self._pending.clear()

    def force_flush(self, timeout_millis=30000):
        return True


class ObservabilitySpanExporter(SpanExporter):
    """
    Custom SpanExporter that sends OTel spans to the observability server.

    The exporter converts OTel spans to Computation protobuf messages and
    sends them via the RegisterComputations gRPC endpoint. The server then
    creates:
    - Computation nodes for each span with trace/span IDs, timing, status, etc.
    - CHILD_OF relationships between parent and child spans
    - PRODUCES relationships from spans to output messages (via _output_message_id attr)
    - CONSUMES relationships from spans to input messages (via _input_message_ids attr)

    Usage:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from sasy.instrumentation.otel import ObservabilitySpanExporter

        exporter = ObservabilitySpanExporter()
        processor = BatchSpanProcessor(exporter)
        trace_provider.add_span_processor(processor)
    """

    def __init__(self) -> None:
        """Initialize the observability span exporter."""
        self._closed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to the observability server.

        Args:
            spans: Sequence of ReadableSpan objects to export

        Returns:
            SpanExportResult.SUCCESS or SpanExportResult.FAILURE
        """
        if self._closed:
            return SpanExportResult.FAILURE

        groups: dict[
            tuple[tuple[str, str], int, str | None],
            tuple[CapturedSessionScope, list[Computation]],
        ] = {}
        result = SpanExportResult.SUCCESS
        for span in spans:
            try:
                # Attribute-only spans do not carry trustworthy capture context.
                if not hasattr(span, "_sasy_captured_scope") or getattr(span, "_sasy_inactive", False):
                    continue
                scope = _span_scope(span)
                binding = getattr(span, "_sasy_session_binding", None)
                if binding is None:
                    raise ValueError("Missing captured span session binding")
                key = (scope, id(binding.lease), binding.policy_id)
                if key not in groups:
                    groups[key] = (binding, [])
                groups[key][1].append(self._span_to_computation(span))
            except Exception as e:
                logger.error(f"Failed to prepare span export: {e}")
                result = SpanExportResult.FAILURE
        for binding, computations in groups.values():
            if not computations:
                continue
            try:
                with bind_session_scope(binding):
                    register_computations(computations)
            except Exception as e:
                # One closed session or failed RPC must not discard unrelated
                # active sessions collected by the same batch processor.
                logger.error(f"Failed to export spans: {e}")
                result = SpanExportResult.FAILURE
        return result

    def _span_to_computation(self, span: ReadableSpan) -> Computation:
        """Convert an OTel ReadableSpan to a Computation protobuf message."""
        # Extract span identifiers
        trace_id = format(span.context.trace_id, "032x")
        span_id = format(span.context.span_id, "016x")
        parent_span_id = None
        if span.parent is not None:
            parent_span_id = format(span.parent.span_id, "016x")

        # Extract timing
        start_time_ns = span.start_time or 0
        end_time_ns = span.end_time or 0

        # Extract status - map OTel StatusCode to our SpanStatusCode enum
        status_code: SpanStatusCode = SpanStatusCode.STATUS_UNSET
        status_message: str | None = None
        if span.status:
            otel_code = span.status.status_code.value
            if otel_code == 1:  # OK
                status_code = SpanStatusCode.STATUS_OK
            elif otel_code == 2:  # ERROR
                status_code = SpanStatusCode.STATUS_ERROR
            status_message = span.status.description

        # Convert attributes to dict, separating special link attributes
        attributes = dict(span.attributes) if span.attributes else {}
        if hasattr(span, "_sasy_captured_scope"):
            session, entity = _span_scope(span)
            attributes["sasy.session_id"] = session
            attributes["sasy.entity"] = entity
        input_message_ids_raw = attributes.pop("_input_message_ids", None)
        input_message_ids: str | list[str] | None = (
            str(input_message_ids_raw) if isinstance(input_message_ids_raw, str)
            else list(input_message_ids_raw) if isinstance(input_message_ids_raw, (list, tuple))
            else None
        )
        output_message_id_raw = attributes.pop("_output_message_id", None)
        output_message_id: str | None = str(output_message_id_raw) if output_message_id_raw else None

        # Convert events to JSON array
        events_list = []
        for event in span.events or []:
            event_attrs = dict(event.attributes) if event.attributes else {}
            events_list.append({
                "name": event.name,
                "timestamp_ns": event.timestamp or 0,
                "attributes": event_attrs,
            })

        # Extract resource info
        service_name: str | None = None
        service_version: str | None = None
        if span.resource:
            sn = span.resource.attributes.get("service.name")
            service_name = str(sn) if sn else None
            sv = span.resource.attributes.get("service.version")
            service_version = str(sv) if sv else None

        # Build Computation message
        computation = Computation(
            trace_id=trace_id,
            span_id=span_id,
            name=span.name,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            status_code=status_code,
            attributes_json=json.dumps(attributes),
            events_json=json.dumps(events_list),
        )

        # Set optional fields
        if parent_span_id:
            computation.parent_span_id = parent_span_id
        if status_message:
            computation.status_message = status_message
        if service_name:
            computation.service_name = service_name
        if service_version:
            computation.service_version = service_version
        if input_message_ids:
            # input_message_ids is stored as comma-separated string in span attributes
            if isinstance(input_message_ids, str):
                computation.input_message_ids.extend(input_message_ids.split(","))
            else:
                computation.input_message_ids.extend(input_message_ids)
        if output_message_id:
            computation.output_message_id = output_message_id
        _, entity = _span_scope(span)
        if isinstance(entity, str) and entity:
            computation.entity = entity

        return computation

    def shutdown(self) -> None:
        """Clean up resources."""
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any pending exports.

        Since we write synchronously, this is a no-op.

        Returns:
            True (always successful)
        """
        return True
