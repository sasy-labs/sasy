"""Explicit consumption of responses from application-managed child tasks."""
from __future__ import annotations

from sasy.observability import api as observation
from sasy.observability._snapshots import VERSION_ID_PREFIX
from sasy.proto.observability_pb2 import EventSnapshot

from . import adk_state, dependencies


def _references(snapshots: list[EventSnapshot]) -> list[EventSnapshot]:
    from .adk import AdkInstrumentationError

    references = [EventSnapshot.FromString(item.SerializeToString()) for item in snapshots]
    if any(not item.base_id.startswith(VERSION_ID_PREFIX) or not item.reuse_dependencies
           or item.dependencies for item in references):
        raise AdkInstrumentationError("Child responses require unchanged, registered snapshot references")
    return references


def _scope():
    from .adk import AdkInstrumentationError

    frame = adk_state._frame.get()
    if frame is None:
        raise AdkInstrumentationError("Consume child responses inside an observed ADK computation")
    frame.resources.check()
    return dependencies.bind()


def consume_events(snapshots: list[EventSnapshot]) -> list[str]:
    """Tell the ADK adapter that this tool or callback read responses produced
    outside ADK, for example by child tasks your application runs itself.

    Use it when a tool body obtains a result the adapter cannot see arriving
    (from your own executor, queue or cache). Record the child's work first
    with :func:`sasy.observability.resolve_events`, then pass one reference per
    response you actually used. From then on the enclosing tool result, state
    writes and later authorization checks depend on those responses.

    Args:
        snapshots: One ``EventSnapshot`` per response, with ``event`` set to
            the full original ``Event``, ``base_id`` set to the immutable
            version ID that ``resolve_events`` returned,
            ``reuse_dependencies=True`` and no ``dependencies``.

    Returns:
        The verified immutable version IDs, in input order.

    Raises:
        AdkInstrumentationError: Called outside an instrumented ADK tool or
            callback, or a snapshot is not an unchanged reference.
        grpc.RpcError: ``INVALID_ARGUMENT`` if the engine does not hold that
            exact content under that ID in this session.
    """
    bound = _scope()
    ids = observation.resolve_events(_references(snapshots))
    # The call above blocks. Re-check afterwards: the computation these
    # responses are being consumed into may have ended, or the session or
    # entity changed, while it ran.
    bound.check()
    adk_state.consume(ids)
    return ids


async def consume_events_async(snapshots: list[EventSnapshot]) -> list[str]:
    """Async counterpart of :func:`consume_events`."""
    bound = _scope()
    ids = await observation.resolve_events_async(_references(snapshots))
    # Awaiting suspends this task; re-check for the same reason as in
    # consume_events.
    bound.check()
    adk_state.consume(ids)
    return ids
