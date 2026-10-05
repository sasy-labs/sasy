import json
from typing import Any

try:
    import networkx as nwx
except ImportError:
    nwx = None  # type: ignore[assignment]
from sasy.proto.observability_pb2 import (
    Computation,
    ComputationEdge,
    ComputationMessageEdge,
    ComputationMessageEdgeType,
    Edge,
    Event,
    Graph,
    Role,
    SpanStatusCode,
    Tool,
)


def to_digraph(graph_msg: Graph) -> "nwx.DiGraph":
    """Converts a directed graph in Graph form to the user-facing NetworkX form."""
    if nwx is None:
        raise ImportError(
            "networkx is required for graph conversion. "
            "Install it with: pip install networkx"
        )
    nodes = graph_msg.nodes
    edges = graph_msg.edges

    graph: nwx.DiGraph = nwx.DiGraph()
    for node in nodes:
        graph.add_node(node.id, **dict_from_event(node, keep_id=False))

    for edge in edges:
        graph.add_edge(
            edge.source, edge.destination, **dict_from_edge(edge, keep_ids=False)
        )

    return graph


def event_from_dict(d: dict[str, Any]) -> Event:
    """Converts node data into an Event."""
    match d.get("role", None):
        case "system":
            role: Role | None = Role.SYSTEM
        case "user":
            role = Role.USER
        case "llm":
            role = Role.LLM
        case "agent":
            role = Role.AGENT
        case _:
            role = None

    tools_str = d.get("tools") or "[]"

    # Handle derived_from field for tool results
    derived_from = None
    derived_from_data = d.get("derived_from")
    if derived_from_data:
        # Can be a JSON string or already a dict
        if isinstance(derived_from_data, str):
            try:
                derived_from_data = json.loads(derived_from_data)
            except json.JSONDecodeError:
                derived_from_data = None
        if derived_from_data and isinstance(derived_from_data, dict):
            derived_from = Tool(
                name=derived_from_data.get("name"),
                arguments=(
                    json.dumps(derived_from_data["arguments"])
                    if "arguments" in derived_from_data
                    else None
                ),
            )

    return Event(
        text=d.get("content", ""),
        role=role,
        agent=d.get("agent", None),
        id=d.get("id", None),
        tools=process_tools(tools_str),
        derived_from=derived_from,
    )


def process_tools(tools_json: str) -> list[Tool]:
    """Converts the tools data in JSON string form into a list of Tool messages."""
    return [
        Tool(
            name=tool.get("name", None),
            arguments=json.dumps(tool["arguments"]) if "arguments" in tool else None,
        )
        for tool in json.loads(tools_json)
    ]


def tool_to_dict(t: Tool) -> dict[str, Any]:
    """Converts a single Tool message into a dict representation."""
    result = {}
    if t.HasField("name"):
        result["name"] = t.name
    if t.HasField("arguments") and t.arguments:
        try:
            result["arguments"] = json.loads(t.arguments)
        except json.JSONDecodeError:
            # If arguments is not valid JSON, store as string
            result["arguments"] = t.arguments
    return result


def tool_to_json(tool: Tool) -> str:
    """Converts a single Tool message into a JSON string."""
    return json.dumps(tool_to_dict(tool))


def tools_to_json(tools: list[Tool]) -> str:
    """Converts a list of Tool messages into a JSON string."""
    return json.dumps(tools_to_dicts(tools))


def tools_to_dicts(tools: list[Tool]) -> list[dict[str, Any]]:
    """Unpacks the Tool messages into the representation for NetworkX node data."""
    return [tool_to_dict(t) for t in tools]


def dict_from_event(event: Event, keep_id: bool = True) -> dict[str, Any]:
    """Converts an Event message into the format for NetworkX node data."""
    out: dict[str, Any] = {}
    fields = ["text", "role", "agent", "tools"]
    if keep_id:
        fields.append("id")

    for field in fields:
        present = field == "tools" or event.HasField(field)
        if present:
            value = getattr(event, field)
            if field == "role":
                out[field] = Role.Name(value)
            elif field == "tools":
                out[field] = tools_to_dicts(value)
            else:
                out[field] = value

    # ``derived_from`` carries tool-result provenance that the policy
    # engine consumes as ToolResult facts; surfacing it on the client
    # side lets graph-walking tests and diagnostic dumps see whether
    # tool-result provenance is actually being recorded for cross-agent
    # dependency edges.
    if event.HasField("derived_from"):
        out["derived_from"] = {
            "name": event.derived_from.name,
            "arguments": event.derived_from.arguments,
        }

    return out


def edge_from_dict(d: dict[str, Any]) -> Edge:
    """Converts edge data from neo4j into an Edge message."""
    args = {}
    for field, type in [
        ("source", str),
        ("destination", str),
        ("proximal", bool),
        ("message_index", int),
    ]:
        if field in d:
            args[field] = type(d[field])

    return Edge(**args)


def dict_from_edge(edge: Edge, keep_ids: bool = True) -> dict[str, Any]:
    """Converts an Edge message into edge data for NetworkX."""
    out = {}
    fields = ["message_index", "proximal"]
    if keep_ids:
        fields += ["source", "destination"]

    for field in fields:
        present = edge.HasField(field)
        if present:
            out[field] = getattr(edge, field)

    return out


def computation_from_dict(d: dict[str, Any]) -> Computation:
    """Converts Neo4j node data into a Computation message."""
    # Map status code string to enum
    status_str = d.get("status_code", "STATUS_UNSET")
    if status_str == "OK":
        status_code = SpanStatusCode.STATUS_OK
    elif status_str == "ERROR":
        status_code = SpanStatusCode.STATUS_ERROR
    else:
        status_code = SpanStatusCode.STATUS_UNSET

    return Computation(
        trace_id=d.get("trace_id", ""),
        span_id=d.get("span_id", ""),
        parent_span_id=d.get("parent_span_id"),
        name=d.get("name", ""),
        start_time_ns=d.get("start_time_ns", 0),
        end_time_ns=d.get("end_time_ns", 0),
        duration_ns=d.get("duration_ns", 0),
        status_code=status_code,
        status_message=d.get("status_message"),
        attributes_json=d.get("attributes", "{}"),
        events_json=d.get("events", "[]"),
        service_name=d.get("service_name"),
        service_version=d.get("service_version"),
    )


def computation_edge_from_dict(d: dict[str, Any]) -> ComputationEdge:
    """Converts edge data into a ComputationEdge message."""
    return ComputationEdge(
        parent_span_id=d.get("parent_span_id", ""),
        child_span_id=d.get("child_span_id", ""),
    )


def computation_message_edge_from_dict(d: dict[str, Any]) -> ComputationMessageEdge:
    """Converts edge data into a ComputationMessageEdge message."""
    edge_type_str = d.get("edge_type", "PRODUCES")
    if edge_type_str == "CONSUMES":
        edge_type = ComputationMessageEdgeType.CONSUMES
    else:
        edge_type = ComputationMessageEdgeType.PRODUCES

    return ComputationMessageEdge(
        span_id=d.get("span_id", ""),
        message_id=d.get("message_id", ""),
        edge_type=edge_type,
        message_index=d.get("message_index"),
    )
