"""Record one message at a time, for agent code you orchestrate yourself.

:func:`record` is a thin layer over :func:`sasy.observability.resolve_events`:
each call records one new message, links it to the messages it was computed
from, and returns its immutable version ID. Pass that ID as an input of later
messages and in ``input_node_ids`` of :func:`sasy.check_tool_call`.

The framework adapters enabled by :func:`sasy.instrument` record the same
graph for you; use this when your agent loop is your own code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal
from uuid import uuid4

from sasy.proto.observability_pb2 import Edge, Event, EventSnapshot, Role, Tool

from .api import resolve_events, resolve_events_async

RoleName = Literal["system", "user", "llm", "agent"]
ToolCall = tuple[str, Any]

_ROLES = {
    "system": Role.SYSTEM,
    "user": Role.USER,
    "llm": Role.LLM,
    "agent": Role.AGENT,
}


def _arguments(arguments: Any) -> str:
    """Return tool arguments as the JSON string the engine expects."""
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def _snapshot(
    text: str,
    role: RoleName,
    inputs: Sequence[str],
    agent: str | None,
    tool_calls: Sequence[ToolCall],
    result_of: ToolCall | None,
) -> EventSnapshot:
    """Build the snapshot for one new message.

    Raises:
        ValueError: ``role`` is not one of the supported names.
    """
    if role not in _ROLES:
        raise ValueError(f"role must be one of {sorted(_ROLES)}, not {role!r}")
    event = Event(id=f"record-{uuid4()}", text=text, role=_ROLES[role])
    if agent:
        event.agent = agent
    for name, arguments in tool_calls:
        event.tools.append(Tool(name=name, arguments=_arguments(arguments)))
    if result_of is not None:
        name, arguments = result_of
        event.derived_from.CopyFrom(Tool(name=name, arguments=_arguments(arguments)))
    dependencies = [Edge(source=source, destination=event.id) for source in inputs]
    return EventSnapshot(event=event, dependencies=dependencies)


def record(
    text: str,
    *,
    role: RoleName = "agent",
    inputs: Sequence[str] = (),
    agent: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
    result_of: ToolCall | None = None,
) -> str:
    """Record one new message in the active session and return its ID.

    Args:
        text: The message content.
        role: Who produced it: ``"system"``, ``"user"``, ``"llm"`` (model
            output) or ``"agent"`` (another agent or a tool).
        inputs: IDs returned by earlier calls for the messages this one was
            computed from. These become its dependencies in the graph.
        agent: Optional name of the producing agent or component.
        tool_calls: Tool calls this message requests, as ``(name, arguments)``;
            arguments are a JSON string or a JSON-serializable value.
        result_of: For a tool result, the ``(name, arguments)`` of the call it
            came from, so policies can match it with ``ToolResult``.

    Returns:
        The message's immutable version ID.

    Raises:
        ValueError: ``role`` is not a supported name.
        sasy.SessionScopeError: No :func:`sasy.session` is active.
    """
    snapshot = _snapshot(text, role, inputs, agent, tool_calls, result_of)
    [version] = resolve_events([snapshot])
    return version


async def record_async(
    text: str,
    *,
    role: RoleName = "agent",
    inputs: Sequence[str] = (),
    agent: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
    result_of: ToolCall | None = None,
) -> str:
    """Async equivalent of :func:`record`."""
    snapshot = _snapshot(text, role, inputs, agent, tool_calls, result_of)
    [version] = await resolve_events_async([snapshot])
    return version
