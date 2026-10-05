"""
Context utilities for OpenTelemetry span context propagation.

This module provides functions for propagating context (like input message IDs)
through the call stack using ContextVars, independent of Langroid.
"""

from contextvars import ContextVar

# ContextVar to propagate input message IDs through nested calls
# This ensures child spans (e.g., HTTP calls within a responder) can access
# the input IDs from their parent span
_current_input_ids: ContextVar[list[str]] = ContextVar("current_input_ids", default=[])


def get_current_span():
    """
    Get the current OTel span if tracing is enabled.

    Returns the current span if it is recording, None otherwise.
    """
    from opentelemetry import trace

    span = trace.get_current_span()
    if span.is_recording():
        return span
    return None


def get_current_input_ids() -> list[str]:
    """
    Get the input message IDs from the current context.

    This function handles nested spans (e.g., HTTP calls made within a responder)
    by using a ContextVar that propagates input IDs through the call stack.
    Child spans automatically inherit the input IDs from their parent.

    Returns:
        List of input message IDs if available, empty list otherwise.
    """
    # First, check the ContextVar (propagated from parent spans)
    ctx_ids = _current_input_ids.get()
    if ctx_ids:
        return ctx_ids

    # Fall back to checking the current span's attributes directly
    span = get_current_span()
    if span is None:
        return []

    try:
        # Try to get the attribute from span context
        # Note: ReadableSpan attributes are available after span ends,
        # but we store them early for access during execution
        attrs = getattr(span, "_attributes", {}) or {}
        input_ids_str = attrs.get("_input_message_ids", "")
        if input_ids_str:
            return [id.strip() for id in input_ids_str.split(",") if id.strip()]
    except Exception:
        pass
    return []


def set_current_input_ids(input_ids: list[str]) -> None:
    """
    Set the current input message IDs in the context.

    This is called by instrumentation when entering a new scope (e.g., responder)
    to propagate input IDs to child operations.

    Args:
        input_ids: List of input message IDs to set in context
    """
    _current_input_ids.set(input_ids)


def get_current_message_id() -> str | None:
    """
    Get the most recent input message ID from the current context.

    Convenience function that returns the last input ID, representing the
    last message that was processed before the current operation.

    Returns:
        The most recent input message ID, or None if not available.
    """
    input_ids = get_current_input_ids()
    return input_ids[-1] if input_ids else None
