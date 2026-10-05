"""`sasy.record` builds one immutable message snapshot from plain values."""

import json

import pytest
import sasy
from sasy.observability.record import _snapshot
from sasy.proto.observability_pb2 import Role


def test_a_message_links_its_inputs_and_gets_a_fresh_origin():
    first = _snapshot("hello", "user", ["v1", "v2"], None, (), None)
    second = _snapshot("hello", "user", [], None, (), None)
    event = first.event
    assert event.text == "hello" and event.role == Role.USER
    assert event.id.startswith("record-") and event.id != second.event.id
    assert [(e.source, e.destination) for e in first.dependencies] == [
        ("v1", event.id), ("v2", event.id),
    ]
    assert not event.HasField("agent") and not event.HasField("derived_from")
    assert not first.HasField("base_id") and not first.reuse_dependencies


def test_tool_calls_and_results_carry_json_arguments():
    request = _snapshot(
        "", "llm", ["u"], "planner",
        [("send", {"to": "a@example.com"}), ("read", '{"name": "x"}')], None,
    ).event
    assert request.role == Role.LLM and request.agent == "planner"
    assert [(t.name, json.loads(t.arguments)) for t in request.tools] == [
        ("send", {"to": "a@example.com"}), ("read", {"name": "x"}),
    ]
    result = _snapshot("secret", "agent", [], None, (), ("read", {"name": "x"})).event
    assert result.derived_from.name == "read"
    assert json.loads(result.derived_from.arguments) == {"name": "x"}


def test_unknown_roles_are_rejected():
    with pytest.raises(ValueError, match="role must be one of"):
        _snapshot("x", "assistant", [], None, (), None)


def test_recording_needs_a_session():
    with pytest.raises(sasy.SessionScopeError):
        sasy.record("outside any session")
