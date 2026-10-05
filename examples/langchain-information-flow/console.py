"""Local console callbacks; no changes to the agent's graph or policy checks."""
from __future__ import annotations

import inspect
import json
import os
import sys
from threading import RLock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

_LOCK = RLock()
_COLORS = {"agent": "36", "supervisor": "35", "researcher": "34", "reviewer": "33"}


def _json(value: Any) -> str:
    # JSON keeps multiline content readable and escapes terminal control codes.
    return json.dumps(value, ensure_ascii=False, default=str)


class ConsoleTrace(BaseCallbackHandler):
    """One local handler per agent, shared by its model and tools only."""

    run_inline = True

    def __init__(self, agent: str):
        self.agent = agent
        self._calls: dict[Any, tuple[str, str | None]] = {}
        self._seen: set[str] = set()

    def __deepcopy__(self, memo: dict) -> ConsoleTrace:
        # SASY copies StructuredTools. Keep the model/tool deduplication shared.
        return self

    def _write(self, *lines: str) -> None:
        with _LOCK:
            label = f"[{self.agent}]"
            if "NO_COLOR" not in os.environ and sys.stdout.isatty():
                label = f"\033[{_COLORS.get(self.agent, '36')}m{label}\033[0m"
                lines = tuple(
                    f"\033[31m{line}\033[0m" if line.startswith(("TOOL ERROR", "MODEL ERROR", "INVALID REQUEST"))
                    else f"\033[32m{line}\033[0m" if line.startswith("TOOL RESULT") else line
                    for line in lines
                )
            print("\n".join(f"{label} {line}" for line in lines), flush=True)

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        # Some errors (such as an unknown tool) never enter BaseTool callbacks.
        # Report their returned messages when the model receives them.
        with _LOCK:
            for batch in messages:
                for message in batch:
                    if isinstance(message, ToolMessage) and message.tool_call_id not in self._seen:
                        self._seen.add(message.tool_call_id)
                        status = "ERROR" if message.status == "error" else "RESULT"
                        self._write(f"TOOL {status} {message.name or message.tool_call_id}: {_json(message.content)}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                content = message.content if message is not None else generation.text
                lines = [f"MODEL {_json(content)}"]
                for call in getattr(message, "tool_calls", []):
                    lines.append(f"REQUEST {call['name']}({_json(call['args'])})")
                for call in getattr(message, "invalid_tool_calls", []):
                    lines.append(f"INVALID REQUEST {_json(call)}")
                self._write(*lines)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._write(f"MODEL ERROR {type(error).__name__}: {_json(str(error))}")

    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any,
                      inputs: Any = None, tool_call_id: str | None = None, **kwargs: Any) -> None:
        with _LOCK:
            name = serialized.get("name", "tool")
            self._calls[run_id] = (name, tool_call_id)
            # This callback runs before authorization, so it is an attempt.
            self._write(f"TOOL ATTEMPT {name}({_json(inputs if inputs is not None else input_str)})")

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        with _LOCK:
            name, identity = self._calls.pop(run_id, ("tool", None))
            if isinstance(output, ToolMessage):
                identity = output.tool_call_id
                content = output.content
                status = "ERROR" if output.status == "error" else "RESULT"
            else:
                content, status = output, "RESULT"
            if identity is not None:
                self._seen.add(identity)
            self._write(f"TOOL {status} {name}: {_json(content)}")

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        with _LOCK:
            name, identity = self._calls.pop(run_id, ("tool", None))
            if identity is not None:
                self._seen.add(identity)
            self._write(f"TOOL ERROR {name}: {type(error).__name__}: {_json(str(error))}")


def with_trace(model: Any, tools: list[Any], agent: str, enabled: Any) -> tuple[Any, list[Any]]:
    """Attach local callbacks, which are not inherited by delegated agents.

    ``enabled`` is a flag for the console trace, or a callable that returns
    the callback handler for ``agent`` (the narrated walkthrough uses this).
    """
    if not enabled:
        return model, tools
    trace = enabled(agent) if callable(enabled) else ConsoleTrace(agent)
    existing = model.callbacks
    if existing is None or isinstance(existing, list):
        callbacks = [*(existing or []), trace]
    else:
        callbacks = existing.copy()
        callbacks.add_handler(trace, inherit=False)
    model = model.model_copy(update={"callbacks": callbacks})
    traced = [StructuredTool.from_function(
        **({"coroutine": tool} if inspect.iscoroutinefunction(tool) else {"func": tool}),
        callbacks=[trace],
    ) for tool in tools]
    return model, traced
