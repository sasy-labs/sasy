"""
OpenTelemetry configuration, span exporter, and context utilities.

This package provides:
- OTelConfig: Configuration for OTel/Logfire integration
- configure_otel(): Setup function for OTel with observability server backend
- ObservabilitySpanExporter: Custom SpanExporter that sends spans to the observability server
- Context utilities for propagating input IDs through the call stack

Usage:
    from sasy.instrumentation.otel import configure_otel, OTelConfig, get_current_input_ids

    # Configure with environment variables
    configure_otel()

    # Or explicit configuration
    configure_otel(OTelConfig(service_name="my-app"))

    # Get input IDs in nested calls (e.g., HTTP interceptors)
    input_ids = get_current_input_ids()
"""

from .context import (
    _current_input_ids,
    get_current_input_ids,
    get_current_message_id,
    get_current_span,
    set_current_input_ids,
)

try:
    from .config import OTelConfig, configure_otel, get_tracer, is_configured
    from .exporter import ObservabilitySpanExporter
except ImportError:
    # opentelemetry-sdk not installed — provide no-op stubs
    OTelConfig = None  # type: ignore[assignment,misc]
    ObservabilitySpanExporter = None  # type: ignore[assignment,misc]

    def configure_otel(*args, **kwargs):  # type: ignore[misc]
        """No-op: opentelemetry-sdk not installed."""
        pass

    def is_configured() -> bool:  # type: ignore[misc]
        return False

    class _NoOpSpan:
        """Dummy span that does nothing."""
        def set_attribute(self, *a, **kw): pass
        def set_status(self, *a, **kw): pass
        def record_exception(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _NoOpTracer:
        """Dummy tracer that returns no-op spans."""
        def start_as_current_span(self, *a, **kw):
            return _NoOpSpan()

    _noop_tracer = _NoOpTracer()

    def get_tracer(name: str = "instrumentation"):  # type: ignore[misc]
        """No-op tracer: opentelemetry-sdk not installed."""
        return _noop_tracer

__all__ = [
    "OTelConfig",
    "configure_otel",
    "get_tracer",
    "is_configured",
    "ObservabilitySpanExporter",
    "_current_input_ids",
    "get_current_input_ids",
    "get_current_message_id",
    "get_current_span",
    "set_current_input_ids",
]
