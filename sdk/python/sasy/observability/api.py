"""Observability API — event recording and graph queries.

Uses the shared ``sasy.config`` for endpoint, TLS, and auth settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sasy.capture import capture_computations, capture_events, capture_logger
from sasy.config import get_async_stub, get_config, get_stub
from sasy.instrumentation.session import (
    current_wire_session_id,
    get_current_entity,
    require_active_session,
)

if TYPE_CHECKING:
    import networkx as nwx
from sasy.proto import observability_pb2_grpc as observability_grpc
from sasy.proto.observability_pb2 import (
    Computation,
    Computations,
    Dependencies,
    Edge,
    Event,
    Events,
    EventSnapshot,
    EventSnapshots,
    EventsWithDependencies,
    SliceRequest,
    SpanRequest,
    TraceGraph,
    TraceRequest,
)

from ._snapshots import CaptureDigestCache, content_digest, raw_fingerprint
from .utils import to_digraph

logger = capture_logger(__name__)
_snapshot_cache = CaptureDigestCache()


def _metadata() -> list[tuple[str, str]]:
    return get_config().get_metadata()


def _current_session_id() -> str | None:
    """Active session id from the context-local var.

    A string names a concrete session; ``None`` is the per-tenant
    **global** session (sent as an unset ``session_id`` on the wire —
    admin-only on write paths)."""
    return current_wire_session_id()


def _stamp_entity(events: list[Event], edges: list[Edge]) -> None:
    """Attach the user-supplied `entity` (caller-domain actor label)
    to events and edges that don't already carry one. Mutates in place.

    `session_id` is not set per item; it travels on the request
    envelope (Events / Dependencies / EventsWithDependencies /
    Computations). Callers pass it via ``_current_session_id()`` when
    building the envelope. Distinct from `principal`, which is
    server-stamped at the gRPC boundary from the auth context and
    cannot be set client-side. Distinct from `agent`, the
    conversation-role label that callers manage per-message.
    """
    entity = get_current_entity()
    if not entity:
        return
    for ev in events:
        if not ev.entity:
            ev.entity = entity
    for ed in edges:
        if not ed.entity:
            ed.entity = entity


def get_current_input_ids() -> list[str]:
    """Get the input message IDs from the current OTel span."""
    try:
        from sasy.instrumentation.otel import get_current_input_ids as _get_input_ids
        return _get_input_ids()
    except ImportError:
        return []


def get_current_message_id() -> str | None:
    """Get the most recent input message ID from the current OTel span."""
    input_ids = get_current_input_ids()
    return input_ids[-1] if input_ids else None


def configure(
    url: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
    ca_path: str | None = None,
    auth_hook=None,
) -> None:
    """Set the connection settings, exactly as :func:`sasy.configure` does.

    This is a second spelling of the same call, kept for code that only
    imports :mod:`sasy.observability`.
    """
    from sasy.config import configure as _configure
    _configure(url=url, cert_path=cert_path, key_path=key_path,
               ca_path=ca_path, auth_hook=auth_hook)


# ── Sync API ──────────────────────────────────────────────

def _snapshot_request(
    snapshots: list[EventSnapshot], *, compact: bool,
) -> tuple[EventSnapshots, dict[bytes, bytes], int]:
    """Freeze, capture, and optionally compact caller-owned full snapshots."""
    request = EventSnapshots(snapshots=snapshots, session_id=_current_session_id())
    # Stamp once on the frozen batch, before fingerprinting or capture. Entity
    # affects content equality and must never be inferred from a cache entry.
    _stamp_entity([item.event for item in request.snapshots],
                  [edge for item in request.snapshots for edge in item.dependencies])
    pending: dict[bytes, bytes] = {}
    epoch = _snapshot_cache.epoch()
    session_id = request.session_id if request.HasField("session_id") else None
    for item in request.snapshots:
        if not item.HasField("event") or not item.event.id:
            raise ValueError("Each snapshot requires a stable event.id origin")
        if item.HasField("content_hash"):
            raise ValueError("resolve_events requires full Event objects, not caller-supplied content hashes")
        if item.reuse_dependencies and (not item.base_id or item.dependencies):
            raise ValueError("reuse_dependencies requires base_id and no dependencies")
        if compact and item.reuse_dependencies:
            key = raw_fingerprint(item.event, session_id, item.base_id, capture_events)
            digest = _snapshot_cache.get(key)
            if digest is None:
                digest = content_digest(capture_events([item.event])[0])
                pending[key] = digest
            origin = item.event.id
            item.event.CopyFrom(Event(id=origin))
            item.content_hash = digest
        else:
            item.event.CopyFrom(capture_events([item.event])[0])
    return request, pending, epoch


def _resolved_ids(response, request: EventSnapshots) -> list[str]:
    ids = list(response.ids)
    if len(ids) != len(request.snapshots) or any(not identifier for identifier in ids):
        raise RuntimeError("Snapshot resolution returned invalid canonical IDs")
    if any(item.reuse_dependencies and identifier != item.base_id
           for item, identifier in zip(request.snapshots, ids)):
        raise RuntimeError("Snapshot reference resolution changed the requested base ID")
    return ids


def resolve_events(snapshots: list[EventSnapshot], *, compact: bool = True) -> list[str]:
    """Record messages as immutable versions and return their IDs, in input order.

    This is the recording call adapters should use. Each snapshot is either

    * a new or edited message: ``event`` plus every incoming dependency
      (``Edge(source=<version ID>, destination=event.id)``), or
    * an unchanged message being read again: ``event``, ``base_id=<its version
      ID>``, ``reuse_dependencies=True`` and no dependencies. The engine
      verifies that it holds exactly this content under that ID in this
      session; if not, the call fails and nothing is created.

    ``event.id`` is the message's *origin*, a stable client-side name. It is
    not what you authorize with: pass the **returned** IDs as
    ``input_node_ids`` to :func:`sasy.check_tool_call` and as ``Edge.source``.

    Args:
        snapshots: Full ``EventSnapshot`` objects; never set ``content_hash``.
        compact: ``True`` (the default) sends a content digest instead of the
            full text for unchanged references. The local cache only avoids
            re-hashing; the engine still verifies every reference, and a cache
            hit is never treated as a resolution.

    Raises:
        ValueError: A snapshot has no ``event.id``, sets ``content_hash``, or
            combines ``reuse_dependencies`` with dependencies or no ``base_id``.
        RuntimeError: The engine returned a different number of IDs, or a
            different ID for a reference.
        grpc.RpcError: Transport or validation failure.
    """
    require_active_session()
    request, pending, epoch = _snapshot_request(snapshots, compact=compact)
    if not request.snapshots:
        return []
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    method = stub.ResolveSnapshots if compact else stub.ResolveEvents
    response = method(request, metadata=_metadata())
    ids = _resolved_ids(response, request)
    if compact:
        _snapshot_cache.commit(epoch, pending)
    return ids

def record_events(events: list[Event]) -> list[str]:
    """Record messages and return their IDs, in the active session.

    Re-recording an id updates the message stored under it field by field:
    a field the new message sets replaces the stored one, and a field it
    leaves unset keeps the value already stored. A re-record therefore
    cannot clear ``derived_from``, tools, role, agent, content or entity.
    Prefer :func:`resolve_events`, which records immutable versions instead
    and returns IDs you can authorize with.
    """
    require_active_session()
    events = capture_events(events)
    _stamp_entity(events, [])
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = stub.RegisterEvents(
        Events(events=events, session_id=_current_session_id()),
        metadata=_metadata(),
    )
    return list(response.ids)


def record_dependencies(edges: list[Edge]) -> None:
    """Record dependencies between messages in the active session.

    Each edge may be an ``Edge`` or a dict of its fields. ``source`` names the
    message that was consumed and ``destination`` the one that was produced.
    """
    require_active_session()
    edges = [Edge(**edge) if isinstance(edge, dict) else Edge.FromString(edge.SerializeToString()) for edge in edges]
    _stamp_entity([], edges)
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    stub.RegisterDependencies(
        Dependencies(edges=edges, session_id=_current_session_id()),
        metadata=_metadata(),
    )


def record_events_with_dependencies(
    events: list[Event], edges: list[Edge]
) -> list[str]:
    """Record events and dependencies in a single atomic transaction."""
    require_active_session()
    events = capture_events(events)
    edges = [Edge(**edge) if isinstance(edge, dict) else Edge.FromString(edge.SerializeToString()) for edge in edges]
    _stamp_entity(events, edges)
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = stub.RegisterEventsWithDependencies(
        EventsWithDependencies(
            events=events, edges=edges, session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return list(response.ids)


def backward_slice(
    event_id: str, max_depth: int | None = None,
) -> nwx.DiGraph:
    """Get the backward slice (ancestors) of an event."""
    require_active_session()
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    graph = stub.BackwardSlice(
        SliceRequest(
            event_id=event_id,
            max_depth=max_depth,
            session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return to_digraph(graph)


def forward_slice(
    event_id: str, max_depth: int | None = None,
) -> nwx.DiGraph:
    """Get the forward slice (descendants) of an event."""
    require_active_session()
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    graph = stub.ForwardSlice(
        SliceRequest(
            event_id=event_id,
            max_depth=max_depth,
            session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return to_digraph(graph)


def register_computations(computations: list[Computation]) -> list[str]:
    """Record OTel spans in the graph."""
    require_active_session()
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = stub.RegisterComputations(
        Computations(
            computations=capture_computations(computations), session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return list(response.ids)


def get_trace(
    trace_id: str,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> TraceGraph:
    """Retrieve a full trace with spans, messages, and relationships.

    A trace that spans multiple sessions (the same ``trace_id`` written
    under different ``sasy.session(...)`` contexts) is returned
    flattened — spans/messages from every session you own are merged
    without per-item session attribution. You only ever see sessions
    you own; you just can't tell from the result which session a given
    span came from.
    """
    require_active_session()
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    return stub.GetTrace(
        TraceRequest(trace_id=trace_id, start_time_ns=start_time_ns, end_time_ns=end_time_ns),
        metadata=_metadata(),
    )


def get_span(
    span_id: str,
    include_children: bool = False,
    include_messages: bool = False,
) -> Computation:
    """Retrieve a single span by ID."""
    require_active_session()
    stub = get_stub(observability_grpc.ObservabilityStub)  # type: ignore
    return stub.GetSpan(
        SpanRequest(
            span_id=span_id,
            include_children=include_children,
            include_messages=include_messages,
            session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )


# ── Async API ─────────────────────────────────────────────

async def resolve_events_async(snapshots: list[EventSnapshot], *, compact: bool = True) -> list[str]:
    """Async equivalent of :func:`resolve_events`, including compact validation."""
    require_active_session()
    request, pending, epoch = _snapshot_request(snapshots, compact=compact)
    if not request.snapshots:
        return []
    stub = get_async_stub(observability_grpc.ObservabilityStub)  # type: ignore
    method = stub.ResolveSnapshots if compact else stub.ResolveEvents
    response = await method(request, metadata=_metadata())
    ids = _resolved_ids(response, request)
    if compact:
        _snapshot_cache.commit(epoch, pending)
    return ids



async def record_events_async(events: list[Event]) -> list[str]:
    """Record events asynchronously."""
    require_active_session()
    events = capture_events(events)
    _stamp_entity(events, [])
    stub = get_async_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = await stub.RegisterEvents(
        Events(events=events, session_id=_current_session_id()),
        metadata=_metadata(),
    )
    return list(response.ids)


async def record_dependencies_async(edges: list[Edge]) -> None:
    """Record dependencies asynchronously."""
    require_active_session()
    edges = [Edge(**edge) if isinstance(edge, dict) else Edge.FromString(edge.SerializeToString()) for edge in edges]
    _stamp_entity([], edges)
    stub = get_async_stub(observability_grpc.ObservabilityStub)  # type: ignore
    await stub.RegisterDependencies(
        Dependencies(edges=edges, session_id=_current_session_id()),
        metadata=_metadata(),
    )


async def record_events_with_dependencies_async(
    events: list[Event], edges: list[Edge]
) -> list[str]:
    """Record events and dependencies atomically (async)."""
    require_active_session()
    events = capture_events(events)
    edges = [Edge(**edge) if isinstance(edge, dict) else Edge.FromString(edge.SerializeToString()) for edge in edges]
    _stamp_entity(events, edges)
    stub = get_async_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = await stub.RegisterEventsWithDependencies(
        EventsWithDependencies(
            events=events, edges=edges, session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return list(response.ids)


async def register_computations_async(computations: list[Computation]) -> list[str]:
    """Record OTel spans asynchronously."""
    require_active_session()
    stub = get_async_stub(observability_grpc.ObservabilityStub)  # type: ignore
    response = await stub.RegisterComputations(
        Computations(
            computations=capture_computations(computations), session_id=_current_session_id(),
        ),
        metadata=_metadata(),
    )
    return list(response.ids)
