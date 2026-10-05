"""Ergonomic wrapper around the raw ``ToolCallResponse`` proto."""

from __future__ import annotations

from typing import Any

from sasy.proto.reference_monitor_pb2 import ToolCallResponse


class ToolCallResult:
    """SDK wrapper around ``ToolCallResponse`` with flat convenience properties.

    Every other attribute is forwarded to the underlying proto::

        result.authorized          # bool
        result.transform_ids       # RepeatedScalarContainer[str]
        result.denial_trace        # DenialTrace (full proto)
        result.denial_trace.reasons        # RepeatedComposite[DenialReason]
        result.denial_trace.suggested_fixes

    and adds two flat properties for the common denial-handling pattern::

        result.denial_reasons      # list[str] of DenialReason.details
        result.suggestions         # list[str] of suggested fixes

    Both flat properties return ``[]`` when the call was authorized.
    """

    __slots__ = ("_proto",)

    def __init__(self, proto: ToolCallResponse) -> None:
        self._proto = proto

    @property
    def denial_reasons(self) -> list[str]:
        """Human-readable reasons for denial, flattened from
        ``denial_trace.reasons[].details``. Empty list when authorized."""
        return [r.details for r in self._proto.denial_trace.reasons]

    @property
    def suggestions(self) -> list[str]:
        """Suggested remediation steps, copied out of
        ``denial_trace.suggested_fixes``. Empty list when authorized."""
        return list(self._proto.denial_trace.suggested_fixes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._proto, name)

    def __repr__(self) -> str:
        return f"ToolCallResult({self._proto!r})"
