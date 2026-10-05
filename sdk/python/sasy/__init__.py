"""
sasy -- Policy enforcement for LLM-based agent systems.

Usage::

    import os
    import sasy
    from sasy.auth.hooks import APIKeyAuthHook

    sasy.configure(
        sasy_url=os.environ["SASY_URL"],
        auth_hook=APIKeyAuthHook(api_key=os.environ["SASY_API_KEY"]),
    )
    sasy.instrument()

The SDK reads ``SASY_URL`` (host:port) from the environment or ``.env``, or accepts
``sasy.configure(sasy_url=...)``. For a local engine, ``sasy engine start``
creates and selects a connection profile under ``~/.sasy`` that works across
project directories. An explicit endpoint takes precedence over that profile.
Without a configured endpoint or a selected profile, any call that would open
a connection raises ``SasyEndpointNotConfigured``.

Provision an independent full API key for each principal. APIKeyAuthHook
sends it unchanged. Setting ``SASY_API_KEY`` also configures this hook
automatically when no other auth hook is supplied. The SDK does not compose
credentials from entity names, suffixes or other environment settings.
"""

from .config import SasyEndpointNotConfigured
from .instrumentation import (
    check_tool_call,
    configure,
    get_config,
    instrument,
)
from .instrumentation.feedback import (
    AuthorizationFeedback,
    FeedbackAccumulator,
    get_current_feedback,
)
from .instrumentation.session import (
    GLOBAL_SESSION,
    SessionScopeError,
    configure_default_session,
    get_current_session_id,
    global_session,
    session,
)
from .observability.record import record, record_async

__all__ = [
    # Setup
    "configure",
    "get_config",
    "instrument",
    "SasyEndpointNotConfigured",
    # Tool checks
    "check_tool_call",
    # Recording messages from your own agent loop
    "record",
    "record_async",
    # Feedback
    "AuthorizationFeedback",
    "FeedbackAccumulator",
    "get_current_feedback",
    # Session scoping
    "session",
    "global_session",
    "GLOBAL_SESSION",
    "SessionScopeError",
    "configure_default_session",
    "get_current_session_id",
]
