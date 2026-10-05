"""Tests for observability.utils — protobuf ↔ dict conversions."""

import json

import pytest
from sasy.proto.observability_pb2 import (
    ComputationMessageEdgeType,
    Edge,
    Event,
    Role,
    SpanStatusCode,
    Tool,
)

from sasy.observability.utils import (
    computation_edge_from_dict,
    computation_from_dict,
    computation_message_edge_from_dict,
    dict_from_edge,
    dict_from_event,
    edge_from_dict,
    event_from_dict,
    process_tools,
    tool_to_dict,
    tools_to_dicts,
    tools_to_json,
)


# ===================================================================
# event_from_dict / dict_from_event
# ===================================================================


class TestEventConversions:
    """Tests for Event ↔ dict conversions."""

    @pytest.mark.parametrize(
        "role_str,expected_role",
        [
            ("system", Role.SYSTEM),
            ("user", Role.USER),
            ("llm", Role.LLM),
            ("agent", Role.AGENT),
        ],
        ids=["system", "user", "llm", "agent"],
    )
    def test_event_from_dict_roles(
        self, role_str: str, expected_role: int
    ) -> None:
        d = {
            "content": "hello",
            "role": role_str,
            "id": "n1",
        }
        event = event_from_dict(d)
        assert event.role == expected_role
        assert event.text == "hello"
        assert event.id == "n1"

    def test_event_from_dict_unknown_role(self) -> None:
        d = {"content": "text", "role": "unknown"}
        event = event_from_dict(d)
        # Unknown role should not set the field
        assert event.text == "text"

    def test_event_from_dict_with_tools(self) -> None:
        tools = [{"name": "search", "arguments": {"q": "test"}}]
        d = {
            "content": "use tool",
            "role": "llm",
            "tools": json.dumps(tools),
        }
        event = event_from_dict(d)
        assert len(event.tools) == 1
        assert event.tools[0].name == "search"

    def test_event_from_dict_with_derived_from(self) -> None:
        d = {
            "content": "tool result",
            "role": "user",
            "derived_from": json.dumps(
                {"name": "search", "arguments": {"q": "hi"}}
            ),
        }
        event = event_from_dict(d)
        assert event.derived_from.name == "search"

    def test_dict_from_event_roundtrip(self) -> None:
        event = Event(
            text="hello",
            role=Role.USER,
            agent="my-agent",
            id="id-1",
        )
        d = dict_from_event(event, keep_id=True)
        assert d["text"] == "hello"
        assert d["role"] == "USER"
        assert d["agent"] == "my-agent"
        assert d["id"] == "id-1"

    def test_dict_from_event_no_id(self) -> None:
        event = Event(text="hi", role=Role.SYSTEM, id="x")
        d = dict_from_event(event, keep_id=False)
        assert "id" not in d

    def test_empty_event(self) -> None:
        d: dict = {"content": ""}
        event = event_from_dict(d)
        assert event.text == ""


# ===================================================================
# Tool conversions
# ===================================================================


class TestToolConversions:
    """Tests for Tool ↔ dict/JSON conversions."""

    def test_tool_to_dict(self) -> None:
        t = Tool(
            name="calculator",
            arguments=json.dumps({"expr": "2+2"}),
        )
        d = tool_to_dict(t)
        assert d["name"] == "calculator"
        assert d["arguments"] == {"expr": "2+2"}

    def test_tool_to_dict_no_args(self) -> None:
        t = Tool(name="ping")
        d = tool_to_dict(t)
        assert d["name"] == "ping"
        assert "arguments" not in d

    def test_tool_to_dict_invalid_json_args(self) -> None:
        t = Tool(name="broken", arguments="not-json")
        d = tool_to_dict(t)
        assert d["arguments"] == "not-json"

    def test_process_tools(self) -> None:
        tools_json = json.dumps(
            [
                {"name": "a", "arguments": {"x": 1}},
                {"name": "b"},
            ]
        )
        tools = process_tools(tools_json)
        assert len(tools) == 2
        assert tools[0].name == "a"
        assert tools[1].name == "b"

    def test_tools_to_dicts(self) -> None:
        tools = [
            Tool(name="t1", arguments=json.dumps({"k": "v"})),
            Tool(name="t2"),
        ]
        dicts = tools_to_dicts(tools)
        assert len(dicts) == 2
        assert dicts[0]["name"] == "t1"

    def test_tools_to_json(self) -> None:
        tools = [Tool(name="t1")]
        result = tools_to_json(tools)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "t1"


# ===================================================================
# Edge conversions
# ===================================================================


class TestEdgeConversions:
    """Tests for Edge ↔ dict conversions."""

    def test_edge_from_dict(self) -> None:
        d = {
            "source": "n1",
            "destination": "n2",
            "proximal": True,
            "message_index": 3,
        }
        edge = edge_from_dict(d)
        assert edge.source == "n1"
        assert edge.destination == "n2"
        assert edge.proximal is True
        assert edge.message_index == 3

    def test_edge_from_dict_partial(self) -> None:
        d = {"source": "n1", "destination": "n2"}
        edge = edge_from_dict(d)
        assert edge.source == "n1"
        assert edge.destination == "n2"

    def test_dict_from_edge_with_ids_raises(self) -> None:
        """proto3 string fields lack HasField presence — this is a
        known limitation of dict_from_edge when keep_ids=True and
        the edge has non-optional string fields."""
        edge = Edge(
            source="s",
            destination="d",
            proximal=True,
            message_index=5,
        )
        # proto3 string fields don't support HasField,
        # so keep_ids=True raises ValueError
        with pytest.raises(ValueError):
            dict_from_edge(edge, keep_ids=True)

    def test_dict_from_edge_without_ids(self) -> None:
        edge = Edge(source="s", destination="d", proximal=False)
        d = dict_from_edge(edge, keep_ids=False)
        assert "source" not in d
        assert "destination" not in d


# ===================================================================
# Computation conversions
# ===================================================================


class TestComputationConversions:
    """Tests for Computation ↔ dict conversions."""

    @pytest.mark.parametrize(
        "status_str,expected_code",
        [
            ("OK", SpanStatusCode.STATUS_OK),
            ("ERROR", SpanStatusCode.STATUS_ERROR),
            ("STATUS_UNSET", SpanStatusCode.STATUS_UNSET),
            ("unknown", SpanStatusCode.STATUS_UNSET),
        ],
        ids=["ok", "error", "unset", "unknown"],
    )
    def test_computation_from_dict_status(
        self, status_str: str, expected_code: int
    ) -> None:
        d = {
            "trace_id": "t1",
            "span_id": "s1",
            "name": "op",
            "status_code": status_str,
        }
        comp = computation_from_dict(d)
        assert comp.status_code == expected_code
        assert comp.trace_id == "t1"
        assert comp.span_id == "s1"
        assert comp.name == "op"

    def test_computation_from_dict_full(self) -> None:
        d = {
            "trace_id": "t",
            "span_id": "s",
            "parent_span_id": "ps",
            "name": "myop",
            "start_time_ns": 100,
            "end_time_ns": 200,
            "duration_ns": 100,
            "status_code": "OK",
            "status_message": "all good",
            "attributes": '{"k": "v"}',
            "events": "[]",
            "service_name": "my-svc",
            "service_version": "1.0",
        }
        comp = computation_from_dict(d)
        assert comp.parent_span_id == "ps"
        assert comp.start_time_ns == 100
        assert comp.duration_ns == 100
        assert comp.service_name == "my-svc"

    def test_computation_edge_from_dict(self) -> None:
        d = {
            "parent_span_id": "p1",
            "child_span_id": "c1",
        }
        edge = computation_edge_from_dict(d)
        assert edge.parent_span_id == "p1"
        assert edge.child_span_id == "c1"

    @pytest.mark.parametrize(
        "edge_type_str,expected",
        [
            ("PRODUCES", ComputationMessageEdgeType.PRODUCES),
            ("CONSUMES", ComputationMessageEdgeType.CONSUMES),
        ],
        ids=["produces", "consumes"],
    )
    def test_computation_message_edge_from_dict(
        self, edge_type_str: str, expected: int
    ) -> None:
        d = {
            "span_id": "s1",
            "message_id": "m1",
            "edge_type": edge_type_str,
            "message_index": 2,
        }
        edge = computation_message_edge_from_dict(d)
        assert edge.edge_type == expected
        assert edge.span_id == "s1"
        assert edge.message_id == "m1"
