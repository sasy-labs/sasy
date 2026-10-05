"""SASY identities on ADK's native spans; no provider setup or enforcement.

Applications opt in by configuring OpenTelemetry/Logfire (or configure_otel).
The shared otel_enabled switch disables this enrichment. Only identifiers and
bounded outcomes are added, not prompts, tool arguments, or exception messages.
``sasy.dispatch.outcome`` describes the observed model/tool dispatch boundary.
The native span status also covers the surrounding ADK callback pipeline.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sasy.instrumentation.config import get_config
from sasy.instrumentation.session import current_wire_session_id, get_current_entity


class Operation:
    def __init__(self, kind: str, agent: str, inputs: list[str], call_id: str | None, tool: str | None):
        self.span: Any = None
        self.inputs = tuple(inputs)
        self.outputs: list[str] = []
        self.attributes: dict[str, Any] = {}
        self.outcome = "completed"
        try:
            if not get_config().otel_enabled:
                return
            from opentelemetry import trace
            span = trace.get_current_span()
            if not span.is_recording():
                return
            self.span = span
            self.attributes = {
                "sasy.session_id": current_wire_session_id() or "",
                "sasy.entity": get_current_entity() or "",
                "sasy.framework": "adk", "sasy.agent": agent,
                "sasy.operation": kind, "_input_message_ids": ",".join(inputs),
            }
            if call_id is not None:
                # This is the ADK function-call ID, not a server-issued RM ID.
                self.attributes["sasy.adk.function_call_id"] = call_id
            if tool is not None:
                self.attributes["sasy.tool.name"] = tool
            span.set_attributes(self.attributes)
        except Exception:
            self.span = None

    def produced(self, ids: list[str]) -> None:
        if self.span is None:
            return
        try:
            from opentelemetry import trace
            fresh = [node for node in ids if node not in self.outputs]
            if not self.outputs and len(fresh) == 1:
                self.span.set_attribute("_output_message_id", fresh[0])
            else:
                # Computation has one output field. Preserve every other output
                # through a child computation, without creating message nodes.
                for node in fresh:
                    with trace.get_tracer("instrumentation.adk").start_span(
                        "adk.output", context=trace.set_span_in_context(self.span),
                        attributes={**self.attributes, "sasy.operation": "output", "_output_message_id": node},
                    ):
                        pass
            self.outputs.extend(fresh)
            self.span.set_attribute("_output_message_ids", tuple(self.outputs))
        except Exception:
            pass

    def consumed(self, ids: list[str]) -> None:
        """Include additional inputs observed during an action, such as artifact reads."""
        self.inputs = tuple(dict.fromkeys((*self.inputs, *ids)))
        if self.span is not None:
            try:
                self.attributes["_input_message_ids"] = ",".join(self.inputs)
                self.span.set_attribute("_input_message_ids", self.attributes["_input_message_ids"])
            except Exception:
                pass

    def decision(self, verdict: Any) -> None:
        transforms = tuple(verdict.transform_ids)
        self.outcome = "denied" if not verdict.authorized else "transform_required" if transforms else "completed"
        if self.span is not None:
            try:
                self.span.set_attribute("sasy.rm.authorized", bool(verdict.authorized))
                self.span.set_attribute("sasy.rm.transform_ids", transforms)
            except Exception:
                pass

    def failed_result(self) -> None:
        if self.outcome == "completed":
            self.outcome = "error_result"

    def finish(self, error: BaseException | None = None) -> None:
        if self.span is not None:
            try:
                if error is not None:
                    self.outcome = "cancelled" if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else "error"
                    self.span.set_attribute("sasy.error.type", type(error).__name__)
                self.span.set_attribute("sasy.dispatch.outcome", self.outcome)
            except Exception:
                pass


@contextmanager
def operation(kind: str, agent: str, inputs: list[str], *, call_id: str | None = None,
              tool: str | None = None) -> Iterator[Operation]:
    """Enrich the current native span without changing trace or task context."""
    current = Operation(kind, agent, inputs, call_id, tool)
    try:
        yield current
    except BaseException as error:
        current.finish(error)
        raise
    else:
        current.finish()
