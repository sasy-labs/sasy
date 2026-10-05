"""
Configuration for OpenTelemetry integration.

This module provides configuration and setup for OTel/Logfire
with the observability server span exporter.

Key design decisions:
- Uses the global TracerProvider so client tracers automatically export spans
- Spans are sent to the observability server via gRPC
- Supports log capture via OpenTelemetry logging handler
- Compatible with logfire and other OTel instrumentation libraries
"""

import logging
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OTelConfig(BaseSettings):
    """
    Configuration for OpenTelemetry integration.

    Can be configured via environment variables with OTEL_ prefix:
        OTEL_ENABLED=true
        OTEL_SERVICE_NAME=my-agent-system
        OTEL_USE_LOGFIRE=true
        OTEL_SEND_TO_LOGFIRE=false
        OTEL_AUTO_TRACE_MODULES=langroid,myapp.agents

    Spans are exported to the observability server via gRPC. The observability
    server connection is configured separately via SASY_URL and TLS_*
    environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="OTEL_",
        env_file=".env",
        extra="ignore",
    )

    # Enable/disable OTel integration
    enabled: bool = Field(
        default=True,
        description="Whether OTel integration is enabled",
    )

    # Service identification
    service_name: str = Field(
        default="observability-server",
        description="Name of the service for OTel resource",
    )
    service_version: str | None = Field(
        default=None,
        description="Version of the service",
    )

    # Logfire settings
    use_logfire: bool = Field(
        default=False,
        description="Use logfire for OTel configuration. If False, configure OTel directly.",
    )
    send_to_logfire: bool = Field(
        default=False,
        description="Whether to also send data to Logfire cloud",
    )

    # Auto-tracing configuration
    auto_trace_modules: str = Field(
        default="",
        description="Comma-separated list of modules to auto-trace",
    )
    auto_trace_min_duration: float = Field(
        default=0.0,
        description="Minimum duration (seconds) for auto-traced spans",
    )

    # Logging integration settings
    capture_logs: bool = Field(
        default=True,
        description="Whether to capture Python logs as span events",
    )
    log_level: str = Field(
        default="INFO",
        description="Minimum log level to capture (DEBUG, INFO, WARNING, ERROR)",
    )
    attach_logs_to_spans: bool = Field(
        default=True,
        description="Attach logs as events to the current span (vs separate log records)",
    )

    def get_auto_trace_modules(self) -> list[str]:
        """Get list of modules to auto-trace."""
        if not self.auto_trace_modules:
            return []
        return [m.strip() for m in self.auto_trace_modules.split(",") if m.strip()]


# Global configuration instance
_config: OTelConfig | None = None
_configured: bool = False


def configure_otel(config: OTelConfig | None = None) -> None:
    """
    Configure OpenTelemetry with the observability server span exporter.

    This sets up the global TracerProvider with our ObservabilitySpanExporter, so that:
    - Any tracer obtained via `trace.get_tracer()` will export to the observability server
    - Client code using logfire, opentelemetry, or other OTel libraries will
      automatically have their spans captured
    - Python logs can be attached to spans as events

    The exporter sends spans to the observability server via gRPC. Configure the
    server connection using SASY_URL and TLS_* environment variables.

    Args:
        config: Optional OTelConfig instance. If None, creates from env vars.

    Example:
        # Use environment variables
        configure_otel()

        # Or explicit configuration
        configure_otel(OTelConfig(
            enabled=True,
            service_name="my-app",
            auto_trace_modules="langroid,myapp",
            capture_logs=True,
        ))

        # Client code can then use any OTel-compatible library:
        import logfire
        with logfire.span("my-operation"):  # This span exports to observability server
            ...

        # Or standard OpenTelemetry:
        from opentelemetry import trace
        tracer = trace.get_tracer("my-app")
        with tracer.start_as_current_span("my-span"):  # Also exports
            ...
    """
    global _config, _configured

    if _configured:
        logging.getLogger(__name__).debug("OTel already configured, skipping")
        return

    config = config or OTelConfig()
    _config = config

    if not config.enabled:
        _configured = True
        return

    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from .exporter import ObservabilitySpanExporter, SessionSpanProcessor

    # Create our custom exporter that sends spans to the observability server
    exporter = ObservabilitySpanExporter()

    # Wrap in BatchSpanProcessor for efficiency
    span_processor = BatchSpanProcessor(exporter)
    from ..session import register_session_flush
    def flush_session_spans() -> None:
        span_processor.force_flush()
    register_session_flush(flush_session_spans)

    if config.use_logfire:
        # Use logfire for configuration - this sets up the global TracerProvider
        # and provides nice integrations with popular libraries
        import logfire

        logfire.configure(
            send_to_logfire=config.send_to_logfire,
            service_name=config.service_name,
            service_version=config.service_version,
            additional_span_processors=[SessionSpanProcessor(), span_processor],
        )

        # Optional: Enable auto-tracing for specified modules
        auto_trace_modules = config.get_auto_trace_modules()
        if auto_trace_modules:
            logfire.install_auto_tracing(
                modules=auto_trace_modules,
                min_duration=config.auto_trace_min_duration,
            )
    else:
        # Configure OpenTelemetry directly without logfire
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create({
            "service.name": config.service_name,
            "service.version": config.service_version or "",
        })

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SessionSpanProcessor())
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

    # Set up log capture if enabled
    if config.capture_logs:
        _setup_log_capture(config)

    _configured = True


def _setup_log_capture(config: OTelConfig) -> None:
    """
    Set up Python logging integration with OpenTelemetry.

    Logs are attached as events to the current active span.
    """
    from opentelemetry import trace

    class OTelLoggingHandler(logging.Handler):
        """Logging handler that attaches logs to the current span as events."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                from ..session import is_session_active
                if not is_session_active():
                    return
                span = trace.get_current_span()
                if span is None or not span.is_recording():
                    return

                # Add log as span event with attributes
                attributes = {
                    "log.level": record.levelname,
                    "log.logger": record.name,
                }

                # Include exception info if present
                if record.exc_info:
                    import traceback
                    attributes["exception.type"] = str(record.exc_info[0].__name__) if record.exc_info[0] else ""
                    attributes["exception.message"] = str(record.exc_info[1]) if record.exc_info[1] else ""
                    attributes["exception.stacktrace"] = "".join(
                        traceback.format_exception(*record.exc_info)
                    )

                span.add_event(
                    name=f"log.{record.levelname.lower()}",
                    attributes={
                        **attributes,
                        "message": record.getMessage(),
                    },
                )
            except Exception:
                pass  # Don't break logging if OTel fails

    # Add handler to root logger
    handler = OTelLoggingHandler()
    handler.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Only add if not already present
    root_logger = logging.getLogger()
    if not any(isinstance(h, OTelLoggingHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)


def get_config() -> OTelConfig | None:
    """Get the current OTel configuration."""
    return _config


def is_configured() -> bool:
    """Check if OTel has been configured."""
    return _configured


# Tracer cache
_tracers: dict[str, Any] = {}


def get_tracer(name: str = "instrumentation") -> Any:
    """
    Get or create a tracer for the given name.

    This ensures OTel is configured before returning a tracer.
    Tracers are cached by name.

    Args:
        name: Name for the tracer (default: "instrumentation")

    Returns:
        An OpenTelemetry Tracer instance
    """
    from opentelemetry import trace

    # Ensure OTel is configured
    if not _configured:
        from ..config import get_config
        config = get_config()
        configure_otel(OTelConfig(
            enabled=config.otel_enabled,
            service_name=config.otel_service_name,
        ))

    # Return cached tracer or create new one
    if name not in _tracers:
        _tracers[name] = trace.get_tracer(name)
    return _tracers[name]
