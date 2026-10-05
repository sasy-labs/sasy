"""
HTTP library instrumentation for routing requests through the reference monitor.

Patches requests and httpx libraries to intercept HTTP requests and route them
through the reference monitor for policy enforcement.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any
from urllib.parse import urlparse

import httpx
import requests
import requests.adapters
import wrapt
from httpx import AsyncByteStream, SyncByteStream

from sasy.capture import capture_text
from sasy.config import get_config as get_sasy_config
from sasy.proto.reference_monitor_pb2 import HTTPRequest, HTTPResponse
from sasy.reference_monitor import proxy_http, proxy_http_async

from .config import get_config as get_inst_config
from .feedback import add_denial
from .http_utils import (
    from_httpx_request,
    from_requests_request,
    to_httpx_response,
    to_requests_response,
)
from .otel import get_current_input_ids
from .session import current_wire_session_id, get_current_entity, is_session_active

_TRANSFORMS_HEADER = b"X-Sasy-Transforms"

# ANSI codes for policy decision output
_GREEN_BOLD = "\033[92m\033[1m"
_RED_BOLD = "\033[91m\033[1m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


def _extract_transforms(base_response: HTTPResponse) -> list[str]:
    """Extract transform IDs from the first response chunk headers."""
    if not base_response.message:
        return []
    for header in base_response.message.headers:
        if header.key == _TRANSFORMS_HEADER:
            raw = header.value.decode("utf-8", errors="replace")
            return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _log_policy_decision(
    method: str,
    url: str,
    denied: bool,
    transforms: list[str],
) -> None:
    """Print colored [POLICY] and [TRANSFORM] lines.

    Only called when ``log_policy_decisions`` is enabled.
    Respects ``log_policy_decisions_transforms_only`` (skips
    AUTHORIZED when no transforms) and ``log_transforms``.
    """
    config = get_inst_config()
    url = capture_text(url)
    if denied:
        print(
            f"{_RED_BOLD}[POLICY] DENIED {method} {url}{_RESET}"
        )
    else:
        if config.log_policy_decisions_transforms_only and not transforms:
            return
        print(
            f"{_GREEN_BOLD}[POLICY] AUTHORIZED {method} {url}{_RESET}"
        )
    if config.log_transforms and transforms:
        for xf in transforms:
            print(f"{_CYAN}[TRANSFORM] {xf}{_RESET}")


def _record_denial_feedback(
    method: str,
    url: str,
    response: HTTPResponse,
) -> None:
    """Record auth feedback + log the policy decision for a proxied response.

    On a 401/403 denial, parses the optional ``\\nSuggestion: `` tail out of
    the response body into structured suggestions and records a denial for
    later retrieval; always logs the decision when ``log_policy_decisions``
    is enabled. Shared by the httpx (sync/async) and requests wrappers.
    """
    denied = response.status in (401, 403)
    if denied:
        message = ""
        suggestions: list[str] = []
        if response.message and response.message.content:
            raw_message = response.message.content.decode("utf-8", errors="replace")
            # Suggestions may be embedded as a "...\nSuggestion: ...\nSuggestion: ..." tail.
            if "\nSuggestion: " in raw_message:
                parts = raw_message.split("\nSuggestion: ")
                message = parts[0]
                suggestions = [s.strip() for s in parts[1:] if s.strip()]
            else:
                message = raw_message
        add_denial(
            url=capture_text(url),
            message=capture_text(message) or f"HTTP {response.status}",
            method=method,
            suggestions=[capture_text(s) for s in suggestions],
        )

    if get_inst_config().log_policy_decisions:
        transforms = _extract_transforms(response)
        _log_policy_decision(method, url, denied, transforms)


class Stream(SyncByteStream):
    """Content stream extracted from the gRPC response stream."""

    def __init__(self, response_stream: Iterator[HTTPResponse]):
        super().__init__()
        self.response_stream = response_stream

    def __iter__(self) -> Iterator[bytes]:
        for msg in self.response_stream:
            yield msg.message.content


class AsyncStream(AsyncByteStream):
    """Async content stream extracted from the gRPC response stream."""

    def __init__(self, response_stream: AsyncIterator[HTTPResponse]):
        super().__init__()
        self.response_stream = response_stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for msg in self.response_stream:
            yield msg.message.content


_instrumented = False


def instrument() -> None:
    """
    Instrument HTTP libraries to route through the reference monitor.

    Connection settings come from ``sasy.configure()`` or the
    ``SASY_URL`` env var.
    """
    global _instrumented

    if _instrumented:
        return
    _instrumented = True
    # Hosts that are neither authorized nor recorded: a request to one of them
    # is invisible to the policy. The match is on the exact name written in the
    # URL, so other spellings of the local machine (127.0.0.2, 0.0.0.0, a
    # decimal address) go to the reference monitor like any other host.
    bypass_hosts = {"localhost", "127.0.0.1", "::1"}

    def wrapper_httpx_send(
        wrapped,
        instance: httpx._client.BaseClient,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ):
        """
        Replace local httpx requests with the results
        from the reference monitor.
        """
        if not is_session_active():
            return wrapped(*args, **kwargs)
        request = args[0] if args else kwargs.get("request", None)

        # Invalid args, allow original method to raise error
        if request is None:
            return wrapped(*args, **kwargs)

        # Bypass local requests
        if request.url.host in bypass_hosts:
            return wrapped(*args, **kwargs)

        base_request = from_httpx_request(request)
        input_node_ids = get_current_input_ids()
        metadata = get_sasy_config().auth_hook.get_metadata()

        response_iter = proxy_http(
            HTTPRequest(
                request=base_request,
                input_node_ids=input_node_ids,
                session_id=current_wire_session_id(),
                entity=get_current_entity(),
            ),
            metadata=metadata if metadata else None,
        )
        base_response = next(response_iter)

        _record_denial_feedback(str(request.method), str(request.url), base_response)

        response = to_httpx_response(
            base_response,
            Stream(response_iter),
            base_request=request,
        )

        instance.cookies.extract_cookies(response)
        response.default_encoding = instance._default_encoding
        return response

    async def wrapper_httpx_send_async(
        wrapped,
        instance: httpx._client.AsyncClient,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ):
        """
        Replace local httpx async requests with the results
        from the reference monitor.
        """
        if not is_session_active():
            return await wrapped(*args, **kwargs)
        request = args[0] if args else kwargs.get("request", None)

        # Invalid args, allow original method to raise error
        if request is None:
            return await wrapped(*args, **kwargs)

        # Bypass local requests
        if request.url.host in bypass_hosts:
            return await wrapped(*args, **kwargs)

        base_request = from_httpx_request(request)
        input_node_ids = get_current_input_ids()
        metadata = get_sasy_config().auth_hook.get_metadata()

        response_iter = proxy_http_async(
            HTTPRequest(
                request=base_request,
                input_node_ids=input_node_ids,
                session_id=current_wire_session_id(),
                entity=get_current_entity(),
            ),
            metadata=metadata if metadata else None,
        )
        base_response = await response_iter.__anext__()

        _record_denial_feedback(str(request.method), str(request.url), base_response)

        response = to_httpx_response(
            base_response,
            AsyncStream(response_iter),
            base_request=request,
        )

        instance.cookies.extract_cookies(response)
        response.default_encoding = instance._default_encoding
        return response

    # Apply sync wrapper for sync Client
    wrapt.wrap_function_wrapper(
        "httpx._client",
        "Client._send_single_request",
        wrapper_httpx_send,
    )

    # Apply async wrapper for AsyncClient
    wrapt.wrap_function_wrapper(
        "httpx._client",
        "AsyncClient._send_single_request",
        wrapper_httpx_send_async,
    )

    def wrapper_requests_send(
        wrapped,
        instance: requests.adapters.HTTPAdapter,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ):
        """
        Replace local requests with the results
        from the reference monitor.
        """
        if not is_session_active():
            return wrapped(*args, **kwargs)
        request = args[0] if args else kwargs.get("request", None)

        # Invalid args, allow original method to raise error
        if request is None:
            return wrapped(*args, **kwargs)

        # Bypass local requests
        parsed = urlparse(request.url)
        if parsed.hostname in bypass_hosts:
            return wrapped(*args, **kwargs)

        base_request = from_requests_request(request)
        input_node_ids = get_current_input_ids()
        metadata = get_sasy_config().auth_hook.get_metadata()

        response = proxy_http(
            HTTPRequest(
                request=base_request,
                input_node_ids=input_node_ids,
                session_id=current_wire_session_id(),
                entity=get_current_entity(),
            ),
            metadata=metadata if metadata else None,
        )

        # Process the request as would have been done by HTTPAdapter.build_response()
        httpx_response = next(response)

        _record_denial_feedback(request.method or "GET", request.url, httpx_response)

        requests_response = to_requests_response(
            httpx_response,
            request,
            Stream(response),
        )
        requests_response.connection = instance

        return requests_response

    wrapt.wrap_function_wrapper(
        "requests.adapters",
        "HTTPAdapter.send",
        wrapper_requests_send,
    )
