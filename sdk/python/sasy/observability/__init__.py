"""Observability API for SASY — event recording and graph queries."""

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
    Role,
    SliceRequest,
    SpanRequest,
    Tool,
    TraceGraph,
    TraceRequest,
)

from .api import (
    backward_slice,
    configure,
    record_dependencies,
    record_events,
    record_events_with_dependencies,
    resolve_events,
    resolve_events_async,
)
from .record import record, record_async

__all__ = [
    # Proto types
    "Computation",
    "Computations",
    "Dependencies",
    "Edge",
    "Event",
    "EventSnapshot",
    "EventSnapshots",
    "Events",
    "EventsWithDependencies",
    "Role",
    "SliceRequest",
    "SpanRequest",
    "Tool",
    "TraceGraph",
    "TraceRequest",
    # API
    "backward_slice",
    "configure",
    "record_dependencies",
    "record_events",
    "record_events_with_dependencies",
    "resolve_events",
    "resolve_events_async",
    "record",
    "record_async",
]
