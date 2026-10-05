"""Explicit observation and enforcement for Google ADK's text/function-tool runner.

``sasy.instrument()`` enables it when google-adk is installed (``adk=True``
requires it); then consume an ordinary ``Runner.run_async`` inside a SASY
session.

Two words are used throughout this adapter and its sibling modules.
*Native* means an execution path implemented by google-adk itself: the
Runner, LlmAgent, AgentTool, task tools and the services they call.
*Qualified* means the adapter has tests for that path at the pinned ADK
version, so it can record what was read and produced and authorize what
runs. Anything else raises :class:`AdkInstrumentationError` rather than
running unobserved. The supported set is listed in the Google ADK
integration guide.
"""
from __future__ import annotations

import copy
import inspect
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from importlib.metadata import version
from threading import RLock
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from pydantic_core import to_jsonable_python

from sasy.instrumentation import adk_agents, adk_otel, adk_state
from sasy.instrumentation._canonical import encoder as _canonical_encoder
from sasy.instrumentation.otel import _current_input_ids
from sasy.instrumentation.session import current_wire_session_id, is_session_active
from sasy.observability import api as observation
from sasy.observability._snapshots import VERSION_ID_PREFIX
from sasy.proto.observability_pb2 import Edge, Event, EventSnapshot, Role, Tool
from sasy.reference_monitor import api as monitor

# google-adk is pinned exactly because instrument() patches names that are
# private to it (BaseLlmFlow._BaseLlmFlow__get_llm, _tool_caller._call_tool_async,
# FunctionTool._invoke_callable). Another version could move or rename one, and
# a patch that silently did not apply would be a missed authorization check,
# not an error; the `required` list in instrument() fails closed if any of them
# is missing.
SUPPORTED_ADK_VERSION = "2.9.1"


class AdkInstrumentationError(RuntimeError):
    """The invocation cannot be observed or mediated under the supported profile."""


# The canonical encoding lives in a module of its own, shared with the LangChain
# adapter; it raises this adapter's error type for a value it cannot write down.
# A tool's arguments and its return are arbitrary Python, and a result is
# recorded only after the tool has run, so a type the encoding does not cover is
# written under a wrapper naming that type, over the rendering pydantic's JSON
# mode gives it, rather than stopping a run that has already had its effect.
_plain_json, _canonical, _canonical_key = _canonical_encoder(
    AdkInstrumentationError, fallback=to_jsonable_python)


def _blocked(verdict: Any, name: str) -> dict:
    """The error result an unauthorized call returns to the model.

    A denial and an authorization that requires a transform are different
    outcomes, so they read differently, and a denial carries the policy's own
    reasons and suggestions.
    """
    if not verdict.authorized:
        message = f"[BLOCKED] SASY denied {name}: " + ("; ".join(verdict.denial_reasons) or "no allow rule matched")
        if verdict.suggestions:
            message += " Suggested: " + "; ".join(verdict.suggestions)
    else:
        message = (f"[BLOCKED] The policy authorized {name} only with transform(s) "
                   f"{', '.join(verdict.transform_ids)}, which the ADK adapter cannot apply. "
                   "Remove the ApplyTransform rule for tool calls, or make the call over "
                   "HTTP routed by sasy.instrument(http=True).")
    return {"error": message}


def _canonical_json(value: Any) -> str:
    """A record of *value* that only *value* can produce.

    ``model_dump(mode="json")`` is lossy, and every loss is two messages sharing
    one record: it writes ``b"v"`` and ``"v"`` alike, and a genai part carries a
    ``part_metadata`` dict whose values are anything at all. So a record and a
    comparison key are taken over the Python-mode dump, through the canonical
    encoding this package shares with the LangChain adapter.
    """
    return _plain_json(_canonical(value))


def _record_json(value: Any) -> str:
    """The canonical record of *value*, a model as the record of itself.

    ``Event.metadata`` and this adapter's local comparison keys are written with
    this function. A rule that reads the metadata reads this encoding — a part's
    text under ``s:text``, a call's name under ``s:function_call.s:name`` — and
    not the field names the caller gave; the plain spelling a rule matches
    against is :func:`_json`, in the event's text and tool arguments.

    The record covers every field of what it records, and any two values that
    differ as JSON data get two records. Two values equal as JSON data and
    differing only in their Python type are told apart where the encoding covers
    the type — the list is in :func:`sasy.instrumentation._canonical.encoder` —
    and are otherwise outside the guarantee, because under this project's trust
    model only the trusted program can produce such a pair: untrusted data and
    model output arrive as text or JSON, which has no tuples, no bytes, no enum
    members and no model instances. The two places that follow from this are a
    saved artifact's provenance, matched on the plain JSON text of the version,
    and a Pydantic model or dataclass passed as a tool argument, recorded as the
    dict it dumps to.
    """
    if hasattr(value, "model_dump"):
        return _canonical_json(value.model_dump(exclude_none=True))
    return _canonical_json(value)


def _json(value: Any) -> str:
    """The spelling a policy reads: plain JSON, with a model as its JSON dump.

    A tool call's arguments, a tool result's text and the arguments handed to
    the authorization check are written with this function, so a rule reads the
    fields under the names the caller gave them. It is deliberately not the
    canonical encoding: that encoding tags dict keys by type, and a rule
    matching ``destination.account`` would miss ``s:destination.s:account``.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return _plain_json(value)


def _content_key(content: Any) -> str:
    # ADK can remove client-generated call IDs when building provider requests.
    data = content.model_dump(exclude_none=True)
    for part in data.get("parts", []):
        for key in ("function_call", "function_response"):
            if key in part:
                part[key].pop("id", None)
    return _canonical_json(data)


def _text_metadata(text: str) -> str:
    """The metadata :meth:`_State.record` writes for a node that is text alone.

    A replayed claim rebuilds the event its producer recorded, and the engine
    decides on the whole event, so a rebuild has to carry the same metadata.
    """
    from google.genai import types
    return _record_json(types.Part(text=text))


def _text_content(content: Any) -> None:
    if content is None:
        raise AdkInstrumentationError("Contentless model/tool events are unsupported")
    for part in content.parts or []:
        data = part.model_dump(exclude_none=True)
        if set(data) - {"text", "function_call", "function_response", "thought", "thought_signature"}:
            raise AdkInstrumentationError("Only text and ordinary function calls/results are supported")


def _history_key(event: Any) -> str:
    # Capture after the native Runner finalizes branch/workflow annotations.
    # Provider metrics and opaque response metadata do not control routing.
    return _canonical_json(event.model_dump(include={
        "author", "actions", "branch", "isolation_scope", "node_info", "invocation_id",
        "content", "timestamp", "cache_metadata", "interaction_id",
    }))


@dataclass
class _Record:
    """One observed ADK event.

    ``event`` is a deep copy of the event as it was observed, ``ids`` the
    immutable version IDs recorded for its parts, and ``key`` the content
    key it had at that moment, so a later sighting can be compared with it.
    """

    event: Any
    ids: list[str]
    key: str


@dataclass
class _State:
    """Everything observed for one conversation: one SASY session, one ADK
    user and one ADK session.

    Attributes:
        session: The SASY session this conversation is recorded in.
        records: ADK event id -> :class:`_Record` for every observed event.
        history, history_order, history_timestamps: The events, their order
            and their timestamps that the ADK session service is expected to
            return on the next turn; a difference means events this process
            did not record.
        outputs: Agent name -> (content key, version IDs) of a model output
            that has not yet been matched to an emitted event.
        results: (agent, function-call id) -> version IDs of that tool result.
        request_validators: Agent name -> the check that the model request
            this adapter observed is the one being sent.
        transfers: (agent, function-call id) -> the authorized
            ``transfer_to_agent`` execution it stands for.
        active: Whether a turn of this conversation is running.
        snapshots: Version ID -> the event recorded under it.
        presentations: Cache of derived renderings, keyed by how they were
            derived, so re-presenting the same content records it once.
        resources: The state and artifact tracking for the running turn.
        artifact_loads: (agent, function-call id) -> the artifact load result
            pinned when that call was dispatched.
    """

    session: str
    records: dict[str, _Record] = field(default_factory=dict)
    history: dict[str, str] = field(default_factory=dict)
    history_order: list[str] = field(default_factory=list)
    history_timestamps: dict[str, datetime] = field(default_factory=dict)
    outputs: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    results: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    request_validators: dict[str, Any] = field(default_factory=dict)
    transfers: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    active: bool = False
    snapshots: dict[str, Event] = field(default_factory=dict)
    presentations: dict[tuple, list[str]] = field(default_factory=dict)
    resources: Any = None
    artifact_loads: dict[tuple[str, str], str] = field(default_factory=dict)

    async def record(self, content: Any, agent: str, inputs: list[str], *,
                     role: Role | None = None, derived: Tool | None = None,
                     cache_key: tuple | None = None) -> list[str]:
        if cache_key is not None and cache_key in self.presentations:
            return self.presentations[cache_key]
        _text_content(content)
        inputs = list(dict.fromkeys(inputs))
        events = []
        edges: list[Edge] = []
        # A separate result node per part preserves each tool's identity.
        for part in content.parts or []:
            # The metadata is the whole part this node stands for, so a field
            # the node's text, role and tool entries leave out cannot be edited
            # without changing the node: the engine's content hash covers it.
            node = Event(id=str(uuid4()), agent=agent, metadata=_record_json(part),
                         role=role if role is not None else (Role.LLM if content.role == "model" else Role.USER))
            if part.text is not None:
                node.text = part.text
            if part.function_call:
                node.tools.append(Tool(name=part.function_call.name, arguments=_json(part.function_call.args or {})))
            if part.function_response:
                node.text = _json(part.function_response.response)
                node.role = Role.AGENT
                # FunctionResponse also represents denials/errors. Only the
                # guarded callable's successful return may establish ToolResult.
                if derived is not None and "error" not in (part.function_response.response or {}):
                    node.derived_from.CopyFrom(derived)
            events.append(node)
            # The last input is the proximal one: the message this output
            # directly answers. Policies use it to tell the request being
            # answered from the rest of the context.
            edges.extend(Edge(source=i, destination=node.id, message_index=n, proximal=n == len(inputs)-1)
                         for n, i in enumerate(dict.fromkeys(inputs)))
        if not events:
            raise AdkInstrumentationError("Empty content cannot establish action provenance")
        pending = [EventSnapshot(event=event, dependencies=[e for e in edges if e.destination == event.id])
                   for event in events]
        returned = await observation.resolve_events_async(pending)
        if len(returned) != len(events) or any(not node.startswith(VERSION_ID_PREFIX) for node in returned):
            raise AdkInstrumentationError("Observation returned invalid immutable identities")
        self.snapshots.update(zip(returned, events, strict=True))
        if cache_key is not None:
            self.presentations[cache_key] = returned
        return returned

    async def observe_event(self, event: Any) -> None:
        _validate_actions(event.actions, event)
        if event.partial:
            if event.actions and event.actions.transfer_to_agent:
                raise AdkInstrumentationError("Partial events cannot authorize a transfer")
            return
        if event.actions and event.actions.transfer_to_agent:
            expected = (event.actions.transfer_to_agent, event.actions.transfer_reason or "")
            if not any(self.transfers.get((event.author, response.id)) == expected
                       for response in event.get_function_responses()):
                raise AdkInstrumentationError("Transfer action has no authorized transfer execution")
        if not event.content:
            return
        if event.author in self.request_validators:
            self.request_validators[event.author]()
        key = _content_key(event.content)
        old = self.records.get(event.id)
        if old and old.key == key:
            return
        calls = event.get_function_calls()
        responses = event.get_function_responses()
        if calls and (any(not c.id for c in calls) or len({c.id for c in calls}) != len(calls)):
            raise AdkInstrumentationError("Missing or duplicate tool call IDs in one model response")
        if calls:
            current_ids = {call.id for call in calls}
            if any(record.event.author == event.author and record.event.id != event.id
                   and any(call.id in current_ids for call in record.event.get_function_calls())
                   for record in self.records.values()):
                raise AdkInstrumentationError("Reused tool call IDs across model responses are unsupported")
        if responses:
            inputs = []
            for response in responses:
                if response.name == "load_artifacts" and self.artifact_loads.get((event.author, response.id)) != _record_json(response.response):
                    raise AdkInstrumentationError("Artifact load result changed after authorized dispatch")
                origin = self.results.get((event.author, response.id))
                if not origin:
                    raise AdkInstrumentationError("Tool response has no observed execution")
                inputs.extend(origin)
        elif event.author == "user":
            # A supplied user message is a root, not an inferred reply to every
            # prior event in the session.
            inputs = []
        else:
            output = self.outputs.pop(event.author, None)
            if output is None:
                raise AdkInstrumentationError("Model event has no observed model invocation")
            output_key, inputs = output
            if output_key == key:
                self.records[event.id] = _Record(event.model_copy(deep=True), inputs, key)
                return
        if responses:
            from google.genai import types
            ids = []
            for part in event.content.parts:
                if not part.function_response:
                    raise AdkInstrumentationError("Mixed result/text events need explicit provenance")
                source = self.results[(event.author, part.function_response.id)]
                ids.extend(await self.record(types.Content(role=event.content.role, parts=[part]),
                                             event.author, source))
        else:
            ids = await self.record(event.content, event.author, inputs)
        self.records[event.id] = _Record(event.model_copy(deep=True), ids, key)

    async def inputs(self, request: Any, agent: str) -> list[str]:
        from google.adk.flows.llm_flows._fencing import _present_other_agent_message
        from google.adk.flows.llm_flows.contents import _copy_content_for_request

        result = []
        for index, content in enumerate(request.contents or []):
            _text_content(content)
            key = _content_key(content)
            native = adk_agents.input_ids(self, agent, content)
            if native is not None:
                result.extend(native)
                continue
            rendered = adk_state.rendered_inputs(self, agent, index, content)
            if rendered is not None:
                result.extend(rendered)
                continue
            candidates = []
            for record in self.records.values():
                event = record.event
                variants = [event.content] if event.author in (agent, "user") else []
                if event.author not in (agent, "user"):
                    rendered = _present_other_agent_message(event)
                    if rendered and rendered.content:
                        variants.append(rendered.content)
                if any(_content_key(_copy_content_for_request(c, strip_client_function_call_ids=True)) == key
                       for c in variants):
                    candidates.append(record)
            if len(candidates) != 1:
                raise AdkInstrumentationError(
                    f"A message in the model request for agent {agent} does not match "
                    "exactly one recorded event. This happens when application code "
                    "adds or edits request contents outside a before_model_callback."
                )
            record = candidates[0]
            # Record the exact presentation as a derivation, including ADK's
            # rendering of messages produced by another configured agent.
            if record.key == key:
                result.extend(record.ids)
            else:
                result.extend(await self.record(content, agent, record.ids,
                    cache_key=("presentation", agent, tuple(record.ids), key)))
        instruction = request.config.system_instruction if request.config else None
        if instruction:
            from google.genai import types
            content = types.Content(parts=[types.Part(text=instruction)]) if isinstance(instruction, str) else instruction
            sources = adk_state.prompt_inputs(self, agent)
            result.extend(await self.record(content, agent, sources, role=Role.SYSTEM,
                cache_key=("system", agent, _content_key(content), tuple(sources))))
        result.extend(self.resources.callback_dependencies.get(agent, []) if self.resources else [])
        if not result:
            raise AdkInstrumentationError("Model invocation has no observed inputs")
        result = list(dict.fromkeys(result))
        # Resend exactly the consumed representations as a single batch. Stored
        # derivations stay immutable; unrelated session messages are not sent.
        resolved = await observation.resolve_events_async([
            EventSnapshot(event=self.snapshots[node], base_id=node, reuse_dependencies=True)
            for node in result
        ])
        if resolved != result:
            raise AdkInstrumentationError("Consumed snapshot identities changed unexpectedly")
        return resolved


_active: ContextVar[_State | None] = ContextVar("sasy_adk_invocation", default=None)
_tool_context: ContextVar[Any] = ContextVar("sasy_adk_tool_context", default=None)


def _state() -> _State:
    state = _active.get()
    if state is None or current_wire_session_id() != state.session:
        raise AdkInstrumentationError(
            "This ADK call is outside the SASY session its run was started in. Call "
            "runner.run_async(...) inside `with sasy.session(policy=...):` and consume "
            "the whole generator inside that block."
        )
    return state


_invocation: ContextVar[Any] = ContextVar("sasy_adk_model_invocation", default=None)
_model_request: ContextVar[Any] = ContextVar("sasy_adk_model_request", default=None)
_runner_states: WeakKeyDictionary = WeakKeyDictionary()
_patch_lock = RLock()
_installed = False


def _validate_actions(actions: Any, event: Any = None) -> None:
    adk_state.validate_actions(actions, event)


def _reject_body_override(options: Any) -> None:
    extra = options.get("extra_body") if isinstance(options, dict) else getattr(options, "extra_body", None)
    if extra:
        raise AdkInstrumentationError("HTTP body overrides can replace observed model inputs")


def _validate_provider(model: Any) -> None:
    from google.adk.models.google_llm import Gemini
    if getattr(model, "use_interactions_api", False):
        raise AdkInstrumentationError("Provider-retained interaction state is unsupported")
    if isinstance(model, Gemini):
        transport = getattr(model.api_client, "_api_client", None)
        options = getattr(transport, "_http_options", None)
        if options is None:
            raise AdkInstrumentationError("Cannot verify Gemini client HTTP defaults")
        _reject_body_override(options)


def _validate_tool(tool: Any) -> None:
    from . import adk_tasks_native
    if adk_tasks_native.validate_tool(tool):
        return
    if adk_agents.validate_tool(tool):
        return
    from . import adk_artifacts
    if adk_artifacts.validate_tool(tool):
        return
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.tools.transfer_to_agent_tool import (
        TransferToAgentTool,
        transfer_to_agent,
    )
    if not isinstance(tool, FunctionTool):
        raise AdkInstrumentationError("Only ordinary function tools and local agent transfers are supported")
    if (getattr(tool.run_async, "__func__", None) is not FunctionTool.run_async
            or getattr(tool._invoke_callable, "__func__", None) is not FunctionTool._invoke_callable):
        raise AdkInstrumentationError("Custom function-tool execution overrides need explicit instrumentation")
    if tool._require_confirmation:
        raise AdkInstrumentationError("ADK confirmation/resumption is unsupported")
    if type(tool) is TransferToAgentTool:
        if tool.func is not transfer_to_agent:
            raise AdkInstrumentationError("Transfer tool must use ADK's trusted local transfer function")
        return
    from .adk_context import context_parameter
    context_parameter(tool)


def _validate_request(request: Any) -> None:
    if request.cache_config is not None or request.cache_metadata is not None or request.previous_interaction_id:
        raise AdkInstrumentationError("Provider-retained context caches and interactions are unsupported")
    if request.config:
        _reject_body_override(request.config.http_options)
        if request.config.cached_content:
            raise AdkInstrumentationError("External provider cache contents have no observed provenance")
    for name, tool in request.tools_dict.items():
        _validate_tool(tool)
        if name != tool.name:
            raise AdkInstrumentationError("Model request contains an unguarded tool dispatch")
    declared: list[str] = []
    for tool in request.config.tools or [] if request.config else []:
        if set(tool.model_dump(exclude_none=True)) != {"function_declarations"}:
            raise AdkInstrumentationError("Provider-native tools are unsupported")
        declared.extend(d.name for d in tool.function_declarations or [])
    if set(declared) != set(request.tools_dict) or len(declared) != len(set(declared)):
        raise AdkInstrumentationError("Function declarations do not match guarded tools")


def _session_backend(service: Any) -> str:
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    if type(service) is InMemorySessionService:
        return "memory"
    try:
        from google.adk.sessions.database_session_service import DatabaseSessionService
    except ImportError:
        raise AdkInstrumentationError("Only InMemorySessionService and v1 DatabaseSessionService are qualified") from None
    if type(service) is DatabaseSessionService:
        return "database"
    raise AdkInstrumentationError("Only InMemorySessionService and v1 DatabaseSessionService are qualified")


def _remember_history(state: _State, event: Any) -> None:
    key = _history_key(event)
    if event.id in state.history and state.history[event.id] != key:
        raise AdkInstrumentationError("Recorded routing history changed without explicit derivation")
    if event.id not in state.history:
        state.history_order.append(event.id)
    state.history[event.id] = key
    # ADK's PreciseTimestamp storage key rounds to datetime microseconds. The
    # original epoch remains in its v1 Event JSON, so raw float sorting differs.
    state.history_timestamps[event.id] = datetime.fromtimestamp(event.timestamp, UTC)


def _validate_runner(runner: Any) -> None:
    from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
    from google.adk.tools.function_tool import FunctionTool
    _session_backend(runner.session_service)
    if runner.app and (runner.app.events_compaction_config or runner.app.resumability_config):
        raise AdkInstrumentationError("Compaction and resumable apps are unsupported")
    if runner.app and runner.app.context_cache_config is not None:
        raise AdkInstrumentationError("Provider-retained context caches are unsupported")
    if runner.plugin_manager.plugins:
        raise AdkInstrumentationError(
            "Runner plugins are not supported by the SASY ADK adapter (found: "
            f"{', '.join(type(p).__name__ for p in runner.plugin_manager.plugins)}). "
            "Remove plugins=[...] from the Runner or App."
        )
    from . import adk_artifacts
    adk_artifacts.validate_service(runner.artifact_service)
    seen = set()
    def visit(agent):
        if agent.name in seen:
            raise AdkInstrumentationError("ADK agent names must be unique throughout the configured tree")
        seen.add(agent.name)
        if type(agent) not in (LlmAgent, LoopAgent, ParallelAgent, SequentialAgent):
            raise AdkInstrumentationError("Only standard LlmAgent/SequentialAgent/ParallelAgent/LoopAgent are supported")
        if isinstance(agent, LlmAgent):
            instructions = (agent.instruction, agent.global_instruction)
            if any(not isinstance(value, str) and not callable(value) for value in instructions):
                raise AdkInstrumentationError("Instructions must be text or a callable provider")
            if agent.code_executor:
                raise AdkInstrumentationError("Code execution is unsupported")
            if agent.static_instruction:
                if agent.instruction or not isinstance(agent.static_instruction, str):
                    raise AdkInstrumentationError("Combined or structured static instructions require additional provenance qualification")
            if agent.generate_content_config and agent.generate_content_config.cached_content:
                raise AdkInstrumentationError("External provider cache contents have no observed provenance")
            if not isinstance(agent.model, str) and getattr(agent.model, "use_interactions_api", False):
                raise AdkInstrumentationError("Provider-retained interaction state is unsupported")
            for tool in agent.tools:
                _validate_tool(FunctionTool(tool) if inspect.isfunction(tool) else tool)
        for child in agent.sub_agents:
            visit(child)
    from . import adk_workflow
    if adk_workflow.is_workflow(runner.agent):
        for agent in adk_workflow.validate(runner.agent):
            visit(agent)
    else:
        visit(runner.agent)


def _sessions(runner: Any) -> dict:
    with _patch_lock:
        return _runner_states.setdefault(runner, {})


def _origin(context: Any) -> tuple[_State, str, _Record]:
    state = _state()
    agent = context._invocation_context.agent.name
    matches = [event for event in context._invocation_context.session.events
               if event.author == agent and any(c.id == context.function_call_id for c in event.get_function_calls())]
    if len(matches) != 1:
        raise AdkInstrumentationError(
            f"The call {context.function_call_id} of agent {agent} does not match "
            "exactly one recorded event, so the tool call has no observed origin. "
            "This happens when application code adds or edits session events."
        )
    origin = state.records.get(matches[0].id)
    if origin is None or origin.key != _content_key(matches[0].content):
        raise AdkInstrumentationError("Tool origin is missing or changed after observation")
    return state, agent, origin


async def _record_tool(tool: Any, context: Any, result: Any, derived: Tool | None = None) -> None:
    from google.genai import types
    state, agent, origin = _origin(context)
    await adk_state.flush_reads_async()
    payload = result if isinstance(result, dict) else {"result": result}
    content = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=context.function_call_id, name=tool.name, response=payload))])
    ids = await state.record(content, agent, adk_state.tool_inputs(origin.ids), derived=derived)
    state.results[(agent, context.function_call_id)] = ids
    from .adk_context import complete_tool
    await complete_tool(context, ids, derived is not None and "error" not in payload)


def _patch_model(model: Any) -> None:
    cls = type(model)
    with _patch_lock:
        if getattr(model.generate_content_async, "__func__", None) is not cls.generate_content_async:
            raise AdkInstrumentationError("Instance model execution overrides need explicit instrumentation")
        original = cls.generate_content_async
        if not callable(original):
            raise AdkInstrumentationError("Required model execution hook is missing")
        if getattr(original, "_sasy_adk_model", False):
            return
        @wraps(original)
        async def observed(self, llm_request, stream=False):
            if not is_session_active():
                iterator = original(self, llm_request, stream=stream)
                try:
                    async for value in iterator:
                        yield value
                finally:
                    await iterator.aclose()
                return
            nested = _model_request.get()
            if nested is not None:
                if nested is not llm_request:
                    raise AdkInstrumentationError("Nested model calls with different inputs require explicit instrumentation")
                iterator = original(self, llm_request, stream=stream)
                try:
                    async for response in iterator:
                        yield response
                finally:
                    await iterator.aclose()
                return
            state = _state()
            context = _invocation.get()
            if context is None:
                raise AdkInstrumentationError("Model call lacks its ADK invocation context")
            agent = context.agent.name
            _validate_request(llm_request)
            _validate_provider(self)
            state.request_validators[agent] = lambda: _validate_request(llm_request)
            request = llm_request.model_copy(deep=True)
            adk_state.copy_renderings(state, agent, llm_request, request)
            adk_agents.copy_inputs(state, agent, llm_request, request)
            from google.adk.models.base_llm import BaseLlm
            from google.adk.models.google_llm import Gemini
            if isinstance(self, Gemini):
                if getattr(self._maybe_append_user_content, "__func__", None) is not BaseLlm._maybe_append_user_content:
                    raise AdkInstrumentationError("Custom Gemini prompt preparation needs explicit provenance")
                start = len(request.contents)
                BaseLlm._maybe_append_user_content(self, request)
                for index in range(start, len(request.contents)):
                    content = request.contents[index]
                    ids = await state.record(content, agent, [], role=Role.SYSTEM,
                        cache_key=("gemini-static", agent, _content_key(content)))
                    adk_state.register_rendering(state, agent, content, ids)
            await adk_state.flush_reads_async()
            inputs = await state.inputs(request, agent)
            with adk_otel.operation("model", agent, inputs) as telemetry:
                iterator = original(self, request, stream=stream)
                try:
                    # An async generator's body runs in the context of whoever
                    # resumes it, and a ContextVar set across a yield would
                    # leak into the consumer's context (where reset() would
                    # then fail). So these variables are set only while ADK's
                    # own iterator is being advanced or closed, and reset
                    # before control returns to the caller. The same shape
                    # appears around every other patched iterator below.
                    while True:
                        token = _current_input_ids.set(inputs)
                        depth_token = _model_request.set(request)
                        try:
                            # Re-validated before each chunk: the client's
                            # options are mutable and could be changed
                            # between chunks.
                            _validate_provider(self)
                            response = await anext(iterator)
                            if response.error_code is not None or response.error_message is not None:
                                raise AdkInstrumentationError("Provider error responses cannot produce completed model output")
                            if response.cache_metadata is not None or response.interaction_id:
                                raise AdkInstrumentationError("Provider-retained context caches and interactions are unsupported")
                            if not response.partial:
                                if not response.content:
                                    # A finish-only streaming marker carries no new output.
                                    if not stream:
                                        raise AdkInstrumentationError("Model must return complete content")
                                else:
                                    ids = await state.record(response.content, agent, inputs)
                                    telemetry.produced(ids)
                                    state.outputs[agent] = (_content_key(response.content), ids)
                        except StopAsyncIteration:
                            break
                        finally:
                            _model_request.reset(depth_token)
                            _current_input_ids.reset(token)
                        yield response
                finally:
                    token = _current_input_ids.set(inputs)
                    depth_token = _model_request.set(request)
                    try:
                        await iterator.aclose()
                    finally:
                        _model_request.reset(depth_token)
                        _current_input_ids.reset(token)
        setattr(observed, "_sasy_adk_model", True)
        cls.generate_content_async = observed


async def _guarded_run(original, runner, *, user_id, session_id, new_message=None,
                       invocation_id=None, state_delta=None, run_config=None, yield_user_message=False):
    from google.adk.agents.run_config import StreamingMode
    from google.adk.events import Event as AdkEvent
    _validate_runner(runner)
    backend = _session_backend(runner.session_service)
    if backend == "database":
        await runner.session_service.prepare_tables()
        if runner.session_service._db_schema_version != "1":
            raise AdkInstrumentationError("Database schema v0 is lossy; migrate ADK sessions to schema v1 before instrumentation")
    sid = current_wire_session_id()
    if not sid:
        raise AdkInstrumentationError(
            "An explicit SASY session is required: call runner.run_async(...) inside "
            "`with sasy.session(policy=...):` and consume the whole generator inside "
            "that block."
        )
    if new_message is None:
        raise AdkInstrumentationError("run_async(new_message=...) is required.")
    if state_delta:
        raise AdkInstrumentationError(
            "run_async(state_delta=...) is not supported by the SASY ADK adapter. "
            "Write state from a tool through ToolContext.state."
        )
    if invocation_id:
        raise AdkInstrumentationError(
            "run_async(invocation_id=...), which resumes an invocation, is not "
            "supported by the SASY ADK adapter. Start a new invocation."
        )
    if run_config and (run_config.streaming_mode not in (StreamingMode.NONE, StreamingMode.SSE) or run_config.support_cfc):
        raise AdkInstrumentationError("Only non-streaming or SSE text run_async is supported")
    states = _sessions(runner)
    with _patch_lock:
        state = states.setdefault((sid, user_id, session_id), _State(sid))
        if state.active:
            raise AdkInstrumentationError("Concurrent turns in one ADK session are unsupported")
        state.active = True
    iterator = None
    user = None
    history_start = len(state.history_order)
    try:
        persisted = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
        if persisted is None:
            raise AdkInstrumentationError("ADK session does not exist")
        for event in persisted.events:
            if event.id not in state.history:
                raise AdkInstrumentationError(
                    f"ADK session {session_id} contains events this process did not "
                    "record (for example after a restart, or a second Runner on the "
                    "same session service). The adapter cannot restore their "
                    "ancestry; start a new ADK session."
                )
            previous = state.records.get(event.id)
            if (state.history[event.id] != _history_key(event)
                    or (previous and (not event.content or previous.key != _content_key(event.content)))
                    or (event.content is not None and previous is None)):
                raise AdkInstrumentationError(
                    f"An event in ADK session {session_id} differs from what this "
                    "process recorded. Events must not be edited outside the "
                    "instrumented run; start a new ADK session."
                )
        expected_order = state.history_order
        if backend == "database":
            expected_order = sorted(expected_order, key=lambda node: (state.history_timestamps[node], node))
        if [event.id for event in persisted.events] != expected_order:
            raise AdkInstrumentationError(
                f"ADK session {session_id} lost events, or returns them in a different "
                "order than this process recorded them; start a new ADK session."
            )
        state.resources = adk_state.new_run(state, runner, user_id, session_id)
        message = new_message.model_copy(deep=True)
        user = AdkEvent(author="user", content=message)
        if not await adk_agents.prepare_user(runner, state, user):
            await state.observe_event(user)
        from . import adk_workflow
        if adk_workflow.is_workflow(runner.agent):
            adk_workflow.prepare_run(state, user)
        iterator = original(runner, user_id=user_id, session_id=session_id, new_message=message.model_copy(deep=True),
                            run_config=run_config, yield_user_message=yield_user_message)
        while True:
            token = _active.set(state)
            try:
                event = await anext(iterator)
                state.resources.persisted(event)
                if event.author == "user" and event.content and _content_key(event.content) == _content_key(user.content):
                    record = state.records.pop(user.id, None)
                    if record:
                        state.records[event.id] = _Record(event.model_copy(deep=True), record.ids, record.key)
                await state.observe_event(event)
                adk_agents.child_event(runner, state, event)
                if not event.partial:
                    _remember_history(state, event)
            except StopAsyncIteration:
                break
            finally:
                _active.reset(token)
            yield event
    finally:
        token = _active.set(state)
        try:
            if iterator is not None:
                await iterator.aclose()
            if user is not None and user.id in state.records:
                persisted = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
                candidates = [event for event in persisted.events if event.author == "user" and event.content
                              and event.id not in state.records and _content_key(event.content) == _content_key(user.content)] if persisted else []
                if len(candidates) == 1:
                    event = candidates[0]
                    record = state.records.pop(user.id)
                    state.records[event.id] = _Record(event.model_copy(deep=True), record.ids, record.key)
                    _remember_history(state, event)
                    state.history_order.remove(event.id)
                    state.history_order.insert(history_start, event.id)
        finally:
            _active.reset(token)
            state.active = False
            if state.resources is not None:
                state.resources.close()


def check_supported() -> None:
    """Raise unless the installed google-adk is the exact version supported."""
    if version("google-adk") != SUPPORTED_ADK_VERSION:
        raise AdkInstrumentationError(
            f"google-adk {version('google-adk')} is installed; this adapter supports "
            f"exactly {SUPPORTED_ADK_VERSION}, because it patches that version's "
            "internals. Run: pip install 'sasy[adk]'"
        )


def instrument() -> None:
    """Install idempotent ADK patches without replacing runners, models or tools."""
    global _installed
    check_supported()
    from google.adk.flows.llm_flows import _tool_caller
    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
    from google.adk.runners import Runner
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
    from google.adk.workflow import Workflow
    from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
    from google.adk.workflow._node_runner import NodeRunner
    required = [(Runner, "run_async"), (BaseLlmFlow, "_call_llm_async"),
                (BaseLlmFlow, "_BaseLlmFlow__get_llm"), (BaseLlmFlow, "_postprocess_async"),
                (FunctionTool, "run_async"), (FunctionTool, "_invoke_callable"),
                (_tool_caller, "_call_tool_async"),
                (Workflow, "_start_node_task"),
                (DynamicNodeScheduler, "__call__"),
                (NodeRunner, "_track_event_in_context")]
    if any(not callable(getattr(cls, name, None)) for cls, name in required):
        raise AdkInstrumentationError("Required ADK dispatch hook is missing")
    with _patch_lock:
        if _installed:
            return
        run = Runner.run_async
        @wraps(run)
        async def run_async(self, **kwargs):
            iterator = _guarded_run(run, self, **kwargs) if is_session_active() else run(self, **kwargs)
            try:
                async for event in iterator:
                    yield event
            finally:
                await iterator.aclose()
        setattr(Runner, "run_async", run_async)

        live = Runner.run_live
        async def run_live(self, *args, **kwargs):
            if not is_session_active():
                iterator = live(self, *args, **kwargs)
                try:
                    async for event in iterator:
                        yield event
                finally:
                    await iterator.aclose()
                return
            raise AdkInstrumentationError(
                "Live (bidirectional streaming) runs are not supported by the SASY "
                "ADK adapter. Use Runner.run_async."
            )
            # The unreachable yield makes this an async generator, matching
            # Runner.run_live's type, so the error surfaces on first iteration.
            yield  # pragma: no cover
        setattr(Runner, "run_live", run_live)

        resolve_model = getattr(BaseLlmFlow, "_BaseLlmFlow__get_llm")
        @wraps(resolve_model)
        async def get_model(self, context):
            model = await resolve_model(self, context)
            if is_session_active():
                _patch_model(model)
            return model
        setattr(BaseLlmFlow, "_BaseLlmFlow__get_llm", get_model)

        flow_call = BaseLlmFlow._call_llm_async
        @wraps(flow_call)
        async def call(self, invocation_context, *args, **kwargs):
            if not is_session_active():
                iterator = flow_call(self, invocation_context, *args, **kwargs)
                try:
                    async for value in iterator:
                        yield value
                finally:
                    await iterator.aclose()
                return
            state = _state()
            state.outputs.pop(invocation_context.agent.name, None)
            iterator = flow_call(self, invocation_context, *args, **kwargs)
            try:
                while True:
                    token = _invocation.set(invocation_context)
                    try:
                        response = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    finally:
                        _invocation.reset(token)
                    yield response
            finally:
                token = _invocation.set(invocation_context)
                try:
                    await iterator.aclose()
                finally:
                    _invocation.reset(token)
        setattr(BaseLlmFlow, "_call_llm_async", call)

        postprocess = BaseLlmFlow._postprocess_async
        @wraps(postprocess)
        async def post(self, *args, **kwargs):
            if not is_session_active():
                iterator = postprocess(self, *args, **kwargs)
                try:
                    async for value in iterator:
                        yield value
                finally:
                    await iterator.aclose()
                return
            iterator = postprocess(self, *args, **kwargs)
            try:
                async for event in iterator:
                    await _state().observe_event(event)
                    yield event
            finally:
                await iterator.aclose()
        setattr(BaseLlmFlow, "_postprocess_async", post)

        call_tool = _tool_caller._call_tool_async
        @wraps(call_tool)
        async def validated_call_tool(tool, args, tool_context):
            if not is_session_active():
                return await call_tool(tool, args, tool_context)
            # Before-tool callbacks run after request validation and can replace
            # instance methods. Validate the actual dispatch target again here.
            _validate_tool(tool)
            return await call_tool(tool, args, tool_context)
        setattr(_tool_caller, "_call_tool_async", validated_call_tool)

        tool_run = FunctionTool.run_async
        @wraps(tool_run)
        async def run_tool(self, *, args, tool_context):
            if not is_session_active():
                return await tool_run(self, args=args, tool_context=tool_context)
            _validate_tool(self)
            _validate_actions(tool_context.actions)
            if tool_context.actions.transfer_to_agent:
                raise AdkInstrumentationError("Transfer action was set before authorized dispatch")
            token = _tool_context.set(tool_context)
            try:
                result = await tool_run(self, args=copy.deepcopy(args), tool_context=tool_context)
                # The tool awaited. Anything its context accumulated in the
                # meantime (a transfer, a state delta) was not part of what
                # the adapter observed going in, so validate again.
                _validate_actions(tool_context.actions)
                state, agent, _ = _origin(tool_context)
                if (agent, tool_context.function_call_id) not in state.results:
                    await _record_tool(self, tool_context, result)
                return result
            finally:
                _tool_context.reset(token)
        setattr(FunctionTool, "run_async", run_tool)

        invoke = FunctionTool._invoke_callable
        @wraps(invoke)
        async def dispatch(self, target, args_to_call):
            if not is_session_active():
                return await invoke(self, target, args_to_call)
            _validate_tool(self)
            context = _tool_context.get()
            if context is None or target is not self.func:
                raise AdkInstrumentationError("Unsupported function dispatch path")
            state, agent, origin = _origin(context)
            transfer = type(self) is TransferToAgentTool
            from .adk_context import context_parameter, validate_injected
            context_name = context_parameter(self)
            if context_name:
                if args_to_call.get(context_name) is not context:
                    raise AdkInstrumentationError("ToolContext was not supplied by ADK")
                validate_injected(context)
            values = {k: copy.deepcopy(v) for k, v in args_to_call.items() if k != context_name}
            bound = inspect.signature(target).bind(**values, **({context_name: context} if context_name else {}))
            bound.apply_defaults()
            # Defaults are inserted by apply_defaults(), so freeze them too
            # before the awaited authorization check. Context is the one trusted
            # ADK control object that must retain identity for local transfers.
            arguments = copy.deepcopy({k: v for k, v in bound.arguments.items()
                                       if k != context_name})
            if context_name:
                arguments[context_name] = context
            public_args = {k: v for k, v in arguments.items() if k != context_name}
            if transfer:
                from google.adk.flows.llm_flows.agent_transfer import (
                    _get_transfer_targets,
                )
                active_agent = context._invocation_context.agent
                target_name = public_args["agent_name"]
                allowed = {a.name for a in _get_transfer_targets(active_agent)}
                if target_name not in allowed or target_name not in self._agent_names or active_agent.root_agent.find_agent(target_name) is None:
                    raise AdkInstrumentationError("Transfer target is not an allowed local agent")
            serialized = _json({k: json.loads(_json(v)) for k, v in public_args.items()})
            with adk_otel.operation("tool", agent, adk_state.tool_inputs(origin.ids), call_id=context.function_call_id, tool=self.name) as telemetry:
                verdict = await monitor.check_tool_call_async(self.name, serialized, adk_state.tool_inputs(origin.ids),
                    metadata=[("adk_agent", agent, ""), ("framework", "adk", "")])
                telemetry.decision(verdict)
                # The authorization check awaits. Anything the tool context
                # accumulated while it was in flight was not part of what was
                # authorized, so validate again before dispatching.
                _validate_actions(context.actions)
                if context.actions.transfer_to_agent:
                    raise AdkInstrumentationError("Transfer action was set before authorized dispatch")
                successful = verdict.authorized and not verdict.transform_ids
                if not successful:
                    result = _blocked(verdict, self.name)
                else:
                    token = _current_input_ids.set(adk_state.tool_inputs(origin.ids))
                    try:
                        result = await invoke(self, target, arguments)
                    finally:
                        _current_input_ids.reset(token)
                    if transfer:
                        state.transfers[(agent, context.function_call_id)] = (public_args["agent_name"], public_args["transfer_reason"])
                payload = result if isinstance(result, dict) else {"result": result}
                await _record_tool(self, context, result,
                                   Tool(name=self.name, arguments=serialized) if successful and "error" not in payload else None)
                telemetry.consumed(adk_state.tool_inputs(origin.ids))
                telemetry.produced(state.results[(agent, context.function_call_id)])
                if "error" in payload:
                    telemetry.failed_result()
                return result
        setattr(FunctionTool, "_invoke_callable", dispatch)
        from . import adk_artifacts, adk_callbacks
        adk_artifacts.install()
        adk_callbacks.install()
        adk_state.install()
        adk_agents.install()
        from . import adk_tasks_native
        adk_tasks_native.install()
        from . import adk_workflow
        adk_workflow.install()
        _installed = True


class AdkRunner:
    """Optional wrapper around an instrumented ``Runner``.

    It adds :meth:`event_ids` and :meth:`forget`; ``run_async`` is the
    runner's own patched method, so wrapping changes nothing about how a
    turn runs.
    """

    def __init__(self, runner: Any):
        self.runner = runner
        self._sessions = _sessions(runner)

    async def run_async(self, **kwargs):
        iterator = self.runner.run_async(**kwargs)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()

    def event_ids(self, event_id: str, *, user_id: str, session_id: str) -> list[str]:
        """The immutable version IDs recorded for one observed ADK event.

        Pass them as ``input_node_ids`` to your own
        :func:`sasy.check_tool_call`.

        Raises:
            AdkInstrumentationError: This process did not record that event in
                this conversation.
        """
        state = self._sessions.get((current_wire_session_id(), user_id, session_id))
        if state is None or event_id not in state.records:
            raise AdkInstrumentationError("Event is not observed in this conversation")
        return list(state.records[event_id].ids)

    def forget(self, *, sasy_session_id: str, user_id: str, session_id: str) -> None:
        """Drop what this process remembers about a finished conversation.

        This is the only way to release the per-conversation memory the
        adapter holds. Afterwards that ADK session cannot be continued under
        instrumentation, because its recorded history is gone.

        Raises:
            AdkInstrumentationError: A turn of that conversation is running.
        """
        key = (sasy_session_id, user_id, session_id)
        if key in self._sessions and self._sessions[key].active:
            raise AdkInstrumentationError("Cannot forget an active invocation")
        self._sessions.pop(key, None)


def instrument_adk(runner: Any = None) -> AdkRunner | None:
    """Install the ADK patches, as ``sasy.instrument(adk=True)`` does.

    If *runner* is given, its agents, tools, services and plugins are checked
    for support now instead of at the first ``run_async``, and an
    :class:`AdkRunner` for it is returned. Without one, ``None`` is returned.

    Raises:
        AdkInstrumentationError: Unsupported google-adk version, or a runner
            configuration this adapter does not support; the message names
            what it found.
    """
    instrument()
    if runner is None:
        return None
    _validate_runner(runner)
    return AdkRunner(runner)


def event_ids(runner: Any, event_id: str, *, user_id: str, session_id: str) -> list[str]:
    """Return the immutable IDs for an event observed on this native runner."""
    return AdkRunner(runner).event_ids(event_id, user_id=user_id, session_id=session_id)
