"""Unit tests for the ToolCallResult SDK wrapper.

`check_tool_call` / `check_tool_call_async` return a `ToolCallResult`
that wraps the raw `ToolCallResponse` proto and adds the `denial_reasons`
+ `suggestions` flat convenience properties without breaking access to
`authorized`, `denial_trace`, `transform_ids`, etc.
"""

from sasy.proto.policy_engine_pb2 import DENYLISTED, DenialReason, DenialTrace
from sasy.proto.reference_monitor_pb2 import ToolCallResponse
from sasy.reference_monitor import ToolCallResult


def _authorized_response() -> ToolCallResponse:
    return ToolCallResponse(
        authorized=True,
        transform_ids=["inject_api_key"],
    )


def _denied_response() -> ToolCallResponse:
    return ToolCallResponse(
        authorized=False,
        denial_trace=DenialTrace(
            action_description="cancel_reservation with no insurance",
            reasons=[
                DenialReason(
                    reason_type=DENYLISTED,
                    details="Only gold members or members with travel insurance can cancel reservations",
                ),
            ],
            suggested_fixes=[
                "Purchase travel insurance or upgrade to gold membership",
            ],
        ),
    )


def test_authorized_forwards_core_fields():
    result = ToolCallResult(_authorized_response())
    assert result.authorized is True
    assert list(result.transform_ids) == ["inject_api_key"]


def test_authorized_flat_props_are_empty():
    result = ToolCallResult(_authorized_response())
    assert result.denial_reasons == []
    assert result.suggestions == []


def test_denied_flat_denial_reasons():
    result = ToolCallResult(_denied_response())
    assert result.authorized is False
    assert result.denial_reasons == [
        "Only gold members or members with travel insurance can cancel reservations",
    ]


def test_denied_flat_suggestions():
    result = ToolCallResult(_denied_response())
    assert result.suggestions == [
        "Purchase travel insurance or upgrade to gold membership",
    ]


def test_denied_nested_path_still_works():
    """Backwards compatibility: existing `denial_trace.reasons[*].details`
    access must keep working — the wrapper forwards the full proto."""
    result = ToolCallResult(_denied_response())
    nested = [r.details for r in result.denial_trace.reasons]
    assert nested == result.denial_reasons
    assert list(result.denial_trace.suggested_fixes) == result.suggestions


def test_repr_shows_underlying_proto():
    result = ToolCallResult(_authorized_response())
    assert "ToolCallResult" in repr(result)
    assert "authorized: true" in repr(result)
