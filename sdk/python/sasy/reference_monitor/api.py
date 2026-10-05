"""
Reference monitor gRPC client — HTTP proxying and tool call authorization.

Uses the shared ``sasy.config`` channel for all connections.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, Generator

import grpc

from sasy.capture import capture_logger
from sasy.config import get_async_stub, get_config, get_stub
from sasy.instrumentation import dependencies
from sasy.instrumentation.session import (
    current_wire_session_id,
    get_current_entity,
)
from sasy.proto import policy_engine_pb2
from sasy.proto import reference_monitor_pb2_grpc as rm_grpc
from sasy.proto.reference_monitor_pb2 import (
    HTTPRequest,
    HTTPResponse,
    ToolCallRequest,
)

from .capture import capture_decision
from .result import ToolCallResult

#: Type of one per-action metadata fact: ``(rel, a, b)``, projected by the
#: engine into ``ActionMetadata(idx, rel, a, b)``.
ActionMetadataFact = tuple[str, str, str]


logger = capture_logger(__name__)


def _metadata() -> list[tuple[str, str]]:
    return get_config().get_metadata()


# ── Per-call latency log (opt-in) ─────────────────────────
#
# When ``SASY_LATENCY_LOG_FILE`` is set, every CheckToolCall
# call appends a JSON line with the client-side wall RTT in
# microseconds, plus fn_name, the authorized verdict, and the
# session id. Intended for latency benchmarks; off by default so
# it costs nothing in production.
#
# The fd is opened once and reused; per-record writes go via a
# single ``os.write`` to an O_APPEND fd, which POSIX guarantees is
# atomic for <PIPE_BUF (≥4 KiB) so multiple threads can interleave
# without an in-process lock.

_log_fd: int | None = None
_log_path: str | None = None


def _log_latency(
    fn_name: str,
    authorized: bool,
    rtt_us: float,
    session_id: str | None,
    error: str | None = None,
) -> None:
    global _log_fd, _log_path
    path = os.environ.get("SASY_LATENCY_LOG_FILE")
    if not path:
        return
    if _log_fd is None or _log_path != path:
        # Cache the fd on first use (and re-open if the env var
        # was changed between calls — uncommon but supported).
        try:
            _log_fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            _log_path = path
        except OSError as e:
            logger.debug("latency log open failed (%s): %s", path, e)
            return
    record = {
        "ts": time.time(),
        "fn_name": fn_name,
        "authorized": authorized,
        "rtt_us": float(rtt_us),
        "session_id": session_id or "",
    }
    if error:
        record["error"] = error
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    try:
        os.write(_log_fd, line)
    except OSError as e:
        logger.debug("latency log write failed (%s): %s", path, e)


# ── Sync API ──────────────────────────────────────────────

def _copy_http_request(request: HTTPRequest, binding: dependencies.Binding) -> HTTPRequest:
    frozen = HTTPRequest()
    frozen.CopyFrom(request)
    if (frozen.session_id or None) != binding.session:
        raise dependencies.ComputationScopeError(
            "This HTTP request names a different SASY session than the active scope. "
            "Send the request using the active session."
        )
    return frozen


def proxy_http(
    request: HTTPRequest,
    metadata: list[tuple[str, str]] | None = None,
) -> Generator[HTTPResponse, None, None]:
    """Proxy an HTTP request, resolving pending reads before authorization."""
    binding = dependencies.bind()
    frozen = _copy_http_request(request, binding)
    headers = list(metadata) if metadata is not None else None

    def responses():
        ids = dependencies.resolve_inputs(list(frozen.input_node_ids), binding=binding)
        del frozen.input_node_ids[:]
        frozen.input_node_ids.extend(ids)
        stub = get_stub(rm_grpc.RMProxyStub)  # type: ignore
        stream = stub.ProxyHTTP(iter([frozen]), metadata=headers or _metadata(), timeout=180)
        try:
            # Each response arrives after a wait, and the caller may do
            # anything while it holds one. Re-check on both sides of the
            # yield: the tool body this request belongs to may have ended, or
            # the session or entity changed, in between.
            for response in stream:
                binding.check()
                yield response
                binding.check()
        finally:
            cancel = getattr(stream, "cancel", None)
            if cancel is not None:
                cancel()
    return responses()


def _build_tool_call_request(
    fn_name: str,
    args: str,
    input_node_ids: list[str] | None,
    metadata: list[ActionMetadataFact] | None = None,
) -> tuple[ToolCallRequest, str | None]:
    """Build a CheckToolCall request from the current session/entity context.

    Returns the request plus the resolved wire ``session_id`` (``None`` = the
    per-tenant global session, sent as an unset session_id on the wire; the
    server reads unset as global), which the caller threads into latency logs.
    Shared by the sync and async ``check_tool_call`` entry points.

    Args:
        fn_name: Tool function name.
        args: Tool arguments (JSON string).
        input_node_ids: Node IDs from observability graph context.
        metadata: Optional per-action metadata facts ``[(rel, a, b), ...]``,
            forwarded as ``ToolCallRequest.metadata`` and projected by the
            engine into ``ActionMetadata(idx, rel, a, b)``. Empty/``None`` sends
            no facts (the engine's ``ActionMetadata`` relation stays empty).
    """
    session_id = current_wire_session_id()
    request = ToolCallRequest(
        fn_name=fn_name,
        args=args,
        input_node_ids=input_node_ids or [],
        session_id=session_id,
        entity=get_current_entity(),
        metadata=[
            policy_engine_pb2.PolicyMetadataFact(rel=rel, a=a, b=b)
            for (rel, a, b) in (metadata or [])
        ],
    )
    return request, session_id


def check_tool_call(
    fn_name: str,
    args: str,
    input_node_ids: list[str] | None = None,
    max_retries: int = 3,
    metadata: list[ActionMetadataFact] | None = None,
) -> ToolCallResult:
    """Ask the engine whether a tool call may run.

    Call it immediately before dispatching the tool, with the arguments that
    will actually be used. The decision is made by the policy bound to the
    current :func:`sasy.session`, over the ancestry of ``input_node_ids``:
    every recorded message those messages were computed from.

    Args:
        fn_name: Tool name, matched by ``IsTool(a, name)`` in the policy.
        args: Tool arguments as a JSON string.
        input_node_ids: Immutable version IDs of the messages this call was
            computed from, normally the model message that requested it.
            Inside an ADK or LangChain tool body the body's own inputs are
            added automatically. With no IDs the policy sees a call with no
            ancestry.
        max_retries: Attempts for ``UNAVAILABLE`` and ``CANCELLED``, with
            back-off doubling from 0.1 s. Must be at least 1.
        metadata: Optional per-action metadata facts ``[(rel, a, b), ...]``,
            projected by the engine into ``ActionMetadata(idx, rel, a, b)`` for
            this action (e.g. a package-registry verdict the caller looked up
            for an install command). Defaults to no facts.

    Returns:
        :class:`ToolCallResult`. **A denial is a result, not an exception**:
        read ``result.authorized``. When it is false,
        ``result.denial_reasons`` and ``result.suggestions`` say why. When it
        is true and ``result.transform_ids`` is not empty, the call is
        authorized only if those transforms are applied; a caller that cannot
        apply them must treat the call as denied. Every other attribute
        (``denial_trace`` and any other proto field) is forwarded to the
        underlying ``ToolCallResponse`` unchanged.

    Raises:
        grpc.RpcError: No decision was obtained after ``max_retries``
            attempts. Treat this as denied; the adapters fail closed.
        ValueError: ``max_retries`` is less than 1, which would skip the
            call altogether and so never reach a decision.
        SasyEndpointNotConfigured: No endpoint is configured.
        ComputationScopeError: Called from a different task, thread or session
            than the tool body it belongs to.
    """
    if max_retries < 1:
        raise ValueError(
            f"max_retries must be at least 1, got {max_retries}: fewer "
            "attempts than one would send no request and so decide nothing, "
            "and a call that cannot be decided must be treated as denied."
        )
    binding = dependencies.bind()
    input_node_ids = dependencies.resolve_inputs(input_node_ids, binding=binding)
    request, session_id = _build_tool_call_request(
        fn_name, args, input_node_ids, metadata
    )

    last_error = None
    t_start = time.monotonic()
    for attempt in range(max_retries):
        binding.check()
        try:
            stub = get_stub(rm_grpc.RMProxyStub)  # type: ignore
            response = stub.CheckToolCall(  # type: ignore
                request, metadata=_metadata() or None,
            )
            binding.check()
            _log_latency(
                fn_name,
                bool(response.authorized),
                (time.monotonic() - t_start) * 1_000_000,
                session_id,
            )
            result = ToolCallResult(response)
            capture_decision(
                fn_name, args, input_node_ids, session_id,
                request.entity, result=result, metadata=metadata,
                elapsed_s=time.monotonic() - t_start,
            )
            return result
        except grpc.RpcError as e:
            last_error = e
            if e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
            _log_latency(
                fn_name,
                False,
                (time.monotonic() - t_start) * 1_000_000,
                session_id,
                error=str(e),
            )
            capture_decision(
                fn_name, args, input_node_ids, session_id,
                request.entity, rpc_error=str(e), metadata=metadata,
                elapsed_s=time.monotonic() - t_start,
            )
            raise

    _log_latency(
        fn_name,
        False,
        (time.monotonic() - t_start) * 1_000_000,
        session_id,
        error=str(last_error),
    )
    capture_decision(
        fn_name, args, input_node_ids, session_id,
        request.entity, rpc_error=str(last_error), metadata=metadata,
        elapsed_s=time.monotonic() - t_start,
    )
    raise last_error  # type: ignore


# ── Async API ─────────────────────────────────────────────



def proxy_http_async(
    request: HTTPRequest,
    metadata: list[tuple[str, str]] | None = None,
) -> AsyncGenerator[HTTPResponse, None]:
    """Proxy HTTP asynchronously, bound to the computation at call creation."""
    binding = dependencies.bind()
    frozen = _copy_http_request(request, binding)
    headers = list(metadata) if metadata is not None else None

    async def responses():
        ids = await dependencies.resolve_inputs_async(list(frozen.input_node_ids), binding=binding)
        del frozen.input_node_ids[:]
        frozen.input_node_ids.extend(ids)
        stub = get_async_stub(rm_grpc.RMProxyStub)  # type: ignore

        async def single():
            yield frozen

        stream = stub.ProxyHTTP(single(), metadata=headers or _metadata(), timeout=180)
        try:
            # Re-checked on both sides of the yield, for the reason given in
            # the synchronous proxy_http above.
            async for response in stream:
                binding.check()
                yield response
                binding.check()
        finally:
            cancel = getattr(stream, "cancel", None)
            if cancel is not None:
                cancel()
    return responses()


async def check_tool_call_async(
    fn_name: str,
    args: str,
    input_node_ids: list[str] | None = None,
    max_retries: int = 3,
    metadata: list[ActionMetadataFact] | None = None,
) -> ToolCallResult:
    """Ask the engine whether a tool call may run, from async code.

    See :func:`check_tool_call` for the arguments, the result and what is
    raised. As there, a denial is a result and not an exception.
    """
    if max_retries < 1:
        raise ValueError(
            f"max_retries must be at least 1, got {max_retries}: fewer "
            "attempts than one would send no request and so decide nothing, "
            "and a call that cannot be decided must be treated as denied."
        )
    metadata = list(metadata) if metadata is not None else None
    binding = dependencies.bind()
    input_node_ids = await dependencies.resolve_inputs_async(input_node_ids, binding=binding)
    request, session_id = _build_tool_call_request(
        fn_name, args, input_node_ids, metadata
    )

    last_error = None
    t_start = time.monotonic()
    for attempt in range(max_retries):
        binding.check()
        try:
            stub = get_async_stub(rm_grpc.RMProxyStub)  # type: ignore
            response = await stub.CheckToolCall(  # type: ignore
                request, metadata=_metadata() or None,
            )
            binding.check()
            _log_latency(
                fn_name,
                bool(response.authorized),
                (time.monotonic() - t_start) * 1_000_000,
                session_id,
            )
            result = ToolCallResult(response)
            # Sync slice query on the event loop — fine for
            # opt-in capture runs (see capture.py docstring).
            capture_decision(
                fn_name, args, input_node_ids, session_id,
                request.entity, result=result, metadata=metadata,
                elapsed_s=time.monotonic() - t_start,
            )
            return result
        except grpc.RpcError as e:
            last_error = e
            if e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                    continue
            _log_latency(
                fn_name,
                False,
                (time.monotonic() - t_start) * 1_000_000,
                session_id,
                error=str(e),
            )
            capture_decision(
                fn_name, args, input_node_ids, session_id,
                request.entity, rpc_error=str(e), metadata=metadata,
                elapsed_s=time.monotonic() - t_start,
            )
            raise

    _log_latency(
        fn_name,
        False,
        (time.monotonic() - t_start) * 1_000_000,
        session_id,
        error=str(last_error),
    )
    capture_decision(
        fn_name, args, input_node_ids, session_id,
        request.entity, rpc_error=str(last_error), metadata=metadata,
        elapsed_s=time.monotonic() - t_start,
    )
    raise last_error  # type: ignore
