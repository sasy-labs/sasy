"""Langroid observation using immutable server-resolved message versions.

Finalized model inputs are resolved before dispatch, rather than inferred from
pre-call history. Supported responders, task handoffs and framework truncation
record their actual computations. Arbitrary application edits require an explicit
``record_message_update`` call; unobserved generated history and unexplained
parent/content changes stop execution instead of inventing dependency edges.
Custom responders and hidden application reads still need instrumentation.
"""
from __future__ import annotations

import copy
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from opentelemetry.trace import StatusCode
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from sasy.capture import capture_logger, capture_text
from sasy.instrumentation.session import current_wire_session_id, is_session_active
from sasy.observability.api import resolve_events, resolve_events_async
from sasy.proto.observability_pb2 import Edge, Event, EventSnapshot, Role, Tool
from sasy.reference_monitor import check_tool_call as rm_check_tool_call
from sasy.reference_monitor import check_tool_call_async as rm_check_tool_call_async

from .config import get_config
from .dependencies import ComputationScopeError
from .feedback import FeedbackScope, add_tool_denial
from .otel import get_tracer
from .otel.context import _current_input_ids, get_current_input_ids

if TYPE_CHECKING:
    from langroid import ChatDocument
    from langroid.language_models import LLMMessage

logger = capture_logger(__name__)


class MessageProvenanceError(RuntimeError):
    """A consumed Langroid message has no defensible observed provenance."""


@dataclass
class _Version:
    """What was last resolved for one message object in one SASY session.

    ``canonical_id`` is the server's immutable version ID, ``event`` the event
    that was sent for it, ``parent_id`` the parent pointer it had, and
    ``presentation`` the provider-visible fields at that time.
    """

    canonical_id: str
    event: Event
    parent_id: str
    presentation: dict[str, Any] | None


def _presentation(message: Any) -> dict[str, Any] | None:
    from langroid.language_models import LLMMessage
    # Fields such as tool_call_id affect provider interpretation even though
    # the current Event schema has no corresponding policy fact.
    return message.model_dump(mode="json") if isinstance(message, LLMMessage) else None


def _untagged(dump: Any) -> Any:
    """A message dump without Langroid's ``request`` tag on tool-call arguments.

    Handling a call writes ``request=<tool name>`` into its live arguments dict,
    which the converted LLMMessage shares with the ChatDocument. The tag says
    nothing the call's name does not, so the comparison of a converted message
    with its conversion-time dump leaves it out on both sides (as
    :func:`get_tools` does for the recorded arguments).
    """
    def call(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        arguments = value.get("arguments")
        if isinstance(arguments, dict) and arguments.get("request") == value.get("name"):
            return {**value, "arguments": {key: item for key, item in arguments.items() if key != "request"}}
        return value

    if not isinstance(dump, dict):
        return dump
    dump = dict(dump)
    if "function_call" in dump:
        dump["function_call"] = call(dump["function_call"])
    if isinstance(dump.get("tool_calls"), list):
        dump["tool_calls"] = [{**item, "function": call(item.get("function"))} if isinstance(item, dict) else item
                              for item in dump["tool_calls"]]
    return dump


def _origin(message: Any) -> str:
    """The name every version of this message is bound to.

    A ChatDocument has an id of its own. An LLMMessage does not, so the origin
    is derived from the ChatDocument it was converted from plus the role and
    tool_call_id, because one document converts into several LLMMessages. The
    server binds every immutable version to the origin it was first recorded
    under, so the same view must produce the same origin each time it is
    observed again.
    """
    from langroid import ChatDocument
    if isinstance(message, ChatDocument):
        return message.metadata.id
    if not getattr(message, "_observability_id", None):
        document = getattr(message, "chat_document_id", "")
        if document:
            identity = f"{document}:llm:{message.role}:{getattr(message, 'tool_call_id', '')}"
        else:
            identity = str(uuid4())
        object.__setattr__(message, "_observability_id", identity)
    return str(message._observability_id)


def _document(message: Any) -> Any:
    from langroid import ChatDocument
    if isinstance(message, ChatDocument):
        return message
    identifier = getattr(message, "chat_document_id", "")
    return ChatDocument.from_id(identifier) if identifier else None


def _versions(message: Any) -> dict[tuple[str | None, str], _Version]:
    # Keep version state on the exact framework object. Two views with the
    # same logical origin can legitimately retain different historical versions.
    versions = getattr(message, "_sasy_versions", None)
    if versions is None:
        versions = {}
        object.__setattr__(message, "_sasy_versions", versions)
    return versions


def _version(message: Any) -> _Version | None:
    return _versions(message).get((current_wire_session_id(), _origin(message)))


def _parent(message: Any) -> str:
    return str(getattr(getattr(message, "metadata", None), "parent_id", ""))


def _role(value: Any) -> Role:
    label = str(getattr(value, "value", value)).lower()
    return {"system": Role.SYSTEM, "user": Role.USER, "assistant": Role.LLM,
            "llm": Role.LLM}.get(label, Role.AGENT)


def _event(message: Any, agent: Any = None, *, preserve_provenance: bool = True) -> Event:
    from langroid import ChatDocument
    version = _version(message)
    if getattr(message, "files", None):
        raise MessageProvenanceError("File-bearing messages require explicit attachment observation before Langroid use")
    if isinstance(message, ChatDocument):
        sender = message.metadata.sender
        name = message.metadata.sender_name
    else:
        sender = message.role
        # On a function- or tool-result message the provider's ``name`` is
        # the function's, not the agent's: the copy the model reads belongs
        # to the agent whose history it is in, like every other message
        # there, so a rule that follows one agent's own messages reaches
        # the result it is a copy of.
        result = str(getattr(sender, "value", sender)).lower() in ("function", "tool")
        name = "" if result else getattr(message, "name", "") or ""
    fallback = version.event.agent if version else getattr(getattr(agent, "config", None), "name", "langroid")
    # The tool calls are a function of the message alone, so every reader of
    # an unchanged message derives the same ones and an edit to them shows up.
    event = Event(id=_origin(message), role=_role(sender), agent=name or fallback, tools=get_tools(message))
    if message.content is not None:
        event.text = message.content
    if preserve_provenance and version and version.event.HasField("derived_from"):
        event.derived_from.CopyFrom(version.event.derived_from)
    return event


class _Batch:
    """Plan a whole consumed message list, including new conversion sources."""
    def __init__(self, agent: Any = None):
        self.agent = agent
        self.session_id = current_wire_session_id()
        self.snapshots: list[EventSnapshot] = []
        self.messages: list[Any] = []
        self.presentations: list[dict[str, Any] | None] = []
        self.parents: list[str] = []
        self.indices: dict[str, int] = {}

    def add(self, message: Any, dependencies: list[str] | None = None,
            *, derived_from: Tool | None = None) -> int:
        from langroid import ChatDocument
        origin = _origin(message)
        previous = _version(message)
        event = _event(message, self.agent, preserve_provenance=dependencies is None)
        if dependencies is None and previous is not None:
            if previous.presentation != _presentation(message):
                raise MessageProvenanceError("Unexplained message content or metadata change; record its actual edit")
            if _parent(message) != previous.parent_id:
                raise MessageProvenanceError("Unexplained ChatDocument parent change; record the actual computation explicitly")
            snapshot = EventSnapshot(event=event, base_id=previous.canonical_id, reuse_dependencies=True)
        else:
            if dependencies is None:
                source = getattr(message, "_sasy_conversion_source", None)
                if source is not None and not isinstance(message, ChatDocument):
                    expected = _untagged(getattr(message, "_sasy_conversion_fingerprint", None))
                    actual = _untagged(message.model_dump(mode="json"))
                    if actual != expected:
                        # Langroid canonicalizes identified whitespace-only tool
                        # results in preparation immediately after conversion.
                        normalized = dict(expected or {})
                        if isinstance(normalized.get("content"), str) and not normalized["content"].strip() and getattr(message, "tool_call_id", None):
                            normalized["content"] = ""
                        if actual != normalized:
                            raise MessageProvenanceError("Converted message changed before observation; record its actual edit")
                    self.add(source)
                    dependencies = [_origin(source)]
                elif (isinstance(message, ChatDocument) or not getattr(message, "chat_document_id", "")) and event.role in (Role.USER, Role.SYSTEM) and not _parent(message):
                    dependencies = []
                else:
                    raise MessageProvenanceError("Generated message has not been observed; register its actual inputs before use")
            if derived_from is not None:
                event.derived_from.CopyFrom(derived_from)
            # The last dependency is the proximal one: the message this output
            # directly answers. Policies use it to tell the request being
            # answered from the rest of the context.
            snapshot = EventSnapshot(event=event, dependencies=[
                Edge(source=source, destination=origin, message_index=i,
                     proximal=i == len(dependencies) - 1)
                for i, source in enumerate(dependencies)
            ])
            if previous is not None:
                snapshot.base_id = previous.canonical_id
        presentation, parent = _presentation(message), _parent(message)
        if origin in self.indices:
            index = self.indices[origin]
            if (self.snapshots[index] != snapshot or self.presentations[index] != presentation
                    or self.parents[index] != parent):
                raise MessageProvenanceError("Conflicting versions or derivations use the same message origin")
            return index
        index = len(self.snapshots)
        self.indices[origin] = index
        self.snapshots.append(snapshot)
        self.messages.append(message)
        self.presentations.append(presentation)
        self.parents.append(parent)
        return index

    def publish(self, ids: list[str]) -> None:
        if current_wire_session_id() != self.session_id:
            raise MessageProvenanceError("Snapshot resolution lost its SASY session")
        if len(ids) != len(self.snapshots) or any(not identifier for identifier in ids):
            raise MessageProvenanceError("Snapshot resolution returned invalid canonical IDs")
        for message, snapshot, identifier, presentation, parent in zip(self.messages, self.snapshots, ids, self.presentations, self.parents):
            event = Event()
            event.CopyFrom(snapshot.event)
            _versions(message)[(current_wire_session_id(), event.id)] = _Version(identifier, event, parent, presentation)

    def resolve(self) -> list[str]:
        ids = resolve_events(self.snapshots)
        self.publish(ids)
        return ids

    async def resolve_async(self) -> list[str]:
        ids = await resolve_events_async(self.snapshots)
        self.publish(ids)
        return ids


def _resolve_inputs(messages: list[Any], agent: Any = None) -> list[str]:
    batch = _Batch(agent)
    indices = [batch.add(message) for message in messages]
    if not indices:
        return []
    ids = batch.resolve()
    return [ids[index] for index in indices]


async def _resolve_inputs_async(messages: list[Any], agent: Any = None) -> list[str]:
    batch = _Batch(agent)
    indices = [batch.add(message) for message in messages]
    if not indices:
        return []
    ids = await batch.resolve_async()
    return [ids[index] for index in indices]


def _record_output(message: Any, inputs: list[str], agent: Any = None,
                   derived_from: Tool | None = None) -> str:
    batch = _Batch(agent)
    index = batch.add(message, inputs, derived_from=derived_from)
    return batch.resolve()[index]


async def _record_output_async(message: Any, inputs: list[str], agent: Any = None,
                               derived_from: Tool | None = None) -> str:
    batch = _Batch(agent)
    index = batch.add(message, inputs, derived_from=derived_from)
    return (await batch.resolve_async())[index]


def record_message_update(message: Any, *, input_ids: list[str], agent: Any = None,
                          derived_from: Tool | None = None) -> str:
    """Record an application edit that consumed the old version and these inputs.

    Call after the edit, before passing the message to another instrumented
    method. ``input_ids`` must be canonical IDs for all additional actual inputs.
    The old version is included automatically; parent pointers are not evidence
    of a computation. Tool-result provenance is cleared unless explicitly supplied.
    To declare a replacement that did not read the old version, use the lower-level
    ``resolve_events`` API with its exact complete dependency set instead.
    """
    previous = _version(message)
    if previous is None:
        raise MessageProvenanceError("An edit requires a previously resolved message")
    return _record_output(message, [previous.canonical_id, *input_ids], agent, derived_from)


async def record_message_update_async(message: Any, *, input_ids: list[str], agent: Any = None,
                                      derived_from: Tool | None = None) -> str:
    """Async equivalent of :func:`record_message_update`."""
    previous = _version(message)
    if previous is None:
        raise MessageProvenanceError("An edit requires a previously resolved message")
    return await _record_output_async(message, [previous.canonical_id, *input_ids], agent, derived_from)


def recorded_version_id(message: Any) -> str | None:
    """The version ID this session holds for ``message``, or ``None``.

    The ID names the message as it was last recorded; pass it as an input ID
    to :func:`record_message` or :func:`record_message_update`.
    """
    version = _version(message)
    return None if version is None else version.canonical_id


def record_message(message: Any, *, input_ids: list[str], agent: Any = None,
                   derived_from: Tool | None = None) -> str:
    """Record a message your application composed from recorded inputs.

    For a message no instrumented responder produced, such as a ``ChatDocument``
    your code builds from earlier task results before handing it to a task.
    ``input_ids`` are the version IDs of everything the message was computed
    from (see :func:`recorded_version_id`); the last one is the message it
    directly answers. A message that already has a version is an edit: use
    :func:`record_message_update`.
    """
    if _version(message) is not None:
        raise MessageProvenanceError("The message is already recorded; record an edit with record_message_update")
    return _record_output(message, list(input_ids), agent, derived_from)


def record_task_result(task: Any, result: Any) -> str:
    """Record a task's result document as computed from its pending message.

    The adapter records the result of ``Task.result`` itself. A ``Task``
    subclass that overrides ``result`` replaces that method, so the override
    calls this with the document it built, before returning it.
    """
    from langroid import ChatDocument
    source = task.pending_message
    inputs = _resolve_inputs([source], task.agent) if isinstance(source, ChatDocument) else []
    return _record_output(result, inputs, task.agent)


@dataclass
class _DispatchOutcome:
    """One tool dispatch, and where it sits in the reply's dispatch tree.

    Dispatches nest. Langroid's async handler delegates to the synchronous one
    for a tool with no ``_async`` handler, so one operation crosses two gates;
    a handler may also return another tool message, which Langroid dispatches
    inside the first tool's dispatch. ``gates`` holds the extra gates of this
    same operation, ``children`` the nested dispatches of other tools.
    """
    tool: Tool
    completed: bool = False
    # Whether this gate authorized the call. A denial, a required transform or
    # a failure to reach a decision leaves it false and the handler unrun.
    authorized: bool = False
    # What this tool's own handler returned, and the message versions the call
    # was checked against. A response that completed several tools is recorded
    # as one node per tool from these.
    result: Any = None
    input_ids: tuple[str, ...] = ()
    parent: _DispatchOutcome | None = None
    # The operation this gate belongs to: itself, unless it is the second gate
    # of one operation, in which case the dispatch that opened it.
    root: _DispatchOutcome | None = None
    gates: list[_DispatchOutcome] = field(default_factory=list)
    children: list[_DispatchOutcome] = field(default_factory=list)
    # Set when this tool's result was recorded while the dispatch was still
    # running, because the result is another tool message being dispatched:
    # the result's node, and the node asking for the nested call.
    result_id: str | None = None
    request_id: str | None = None


_dispatch_outcomes: ContextVar[list[_DispatchOutcome] | None] = ContextVar("sasy_langroid_dispatch_outcomes", default=None)
_current_dispatch: ContextVar[_DispatchOutcome | None] = ContextVar("sasy_langroid_current_dispatch", default=None)
# The value Langroid is converting into a ChatDocument: the handler's own
# return value. A tool message dispatched while it is that value is the chain
# Langroid runs for a handler that returned another tool.
_handler_result: ContextVar[Any] = ContextVar("sasy_langroid_handler_result", default=None)


def _declared_handler(tool_message: Any) -> str:
    """The agent method the tool's *class* declares as its handler.

    A tool class may route itself to a method whose name differs from its
    ``request`` value by declaring ``_handler``. Pydantic represents a
    class-level underscore attribute as a private-attribute object, so the
    declared string is unwrapped from it.
    """
    declared: Any = getattr(type(tool_message), "_handler", None)
    if not isinstance(declared, str):
        declared = getattr(declared, "default", None)
    if isinstance(declared, str) and declared:
        return declared
    request = getattr(type(tool_message), "default_value", None)
    return request("request") if callable(request) else str(getattr(tool_message, "request", ""))


def _redirected_handler(tool_message: Any) -> str | None:
    """A handler name carried by the tool *instance* that its class did not declare.

    Langroid's tool messages accept extra fields, so a model's tool JSON can
    put ``_handler`` on the instance; Langroid up to 0.67.0 read the handler
    name from there, which let a call run a method other than the one being
    authorized. Extra fields live in the instance dictionary or, under
    pydantic v2, in its extras mapping.
    """
    declared = _declared_handler(tool_message)
    for store in (getattr(tool_message, "__dict__", None),
                  getattr(tool_message, "model_extra", None),
                  getattr(tool_message, "__pydantic_extra__", None)):
        if isinstance(store, dict) and "_handler" in store:
            named = store["_handler"]
            if named is not None and str(named) != declared:
                return str(named)
    return None


def _begin_dispatch(name: str, arguments: str, tool_message: Any = None) -> _DispatchOutcome:
    """Place this dispatch in the reply's dispatch tree.

    A dispatch of the same serialized tool inside an authorized dispatch is the
    same operation crossing a second gate, and is merged into it. A dispatch of
    the tool message the enclosing handler returned is that handler's result
    being handled, so it becomes the enclosing dispatch's child. Everything
    else is a top-level call of this reply -- including a call a handler makes
    itself -- and two identical top-level calls are two operations with two
    results.
    """
    outcome = _DispatchOutcome(Tool(name=name, arguments=arguments))
    current = _dispatch_outcomes.get()
    if current is None:
        # Outside a responder there is no record of what the call was
        # computed from, so the check would ask the policy about a call with
        # no ancestry. That is a different question from the one the run is
        # actually asking, and answering it would be worse than refusing.
        raise ComputationScopeError(
            f"The tool '{name}' was dispatched outside any instrumented responder, so what it "
            "was computed from is unknown. Dispatch it through the agent's responder "
            "(agent_response / llm_response / task.run); in a worker thread, carry the "
            "context with contextvars.copy_context().run or asyncio.to_thread."
        )
    outcome.root = outcome
    enclosing = _current_dispatch.get()
    if enclosing is None:
        current.append(outcome)
        return outcome
    operation = enclosing.root or enclosing
    if outcome.tool.SerializeToString() == operation.tool.SerializeToString():
        outcome.root = operation
        outcome.parent = operation.parent
        operation.gates.append(outcome)
    elif tool_message is not None and tool_message is _handler_result.get():
        outcome.parent = operation
        operation.children.append(outcome)
    else:
        current.append(outcome)
    return outcome


def _operation_completed(outcome: _DispatchOutcome) -> bool:
    """Whether the operation ran to a result of its own.

    Every gate of the operation must have authorized it: a denial, a required
    transform or a failure to decide at any one of them means the handler did
    not run there, and a user-facing ``[BLOCKED]`` string is not a result. A
    handler that returned ``None`` handled nothing.
    """
    return all(gate.authorized and gate.completed and gate.result is not None
               for gate in (outcome, *outcome.gates))


def _result_text(result: Any) -> str:
    """A handler's return value as the combined response states it.

    Langroid's own combination (``Agent._handle_message_final``) takes a
    ``ChatDocument`` result's content and leaves anything else as it is; a
    handler that returned another tool message was already converted to a
    ``ChatDocument`` by the dispatch this wraps.
    """
    from langroid import ChatDocument
    return result.content if isinstance(result, ChatDocument) else str(result)


def _tool_result_message(text: str, agent: Any) -> Any:
    """A message carrying one tool's own result, in the responding agent's name.

    Built the way the agent builds its own responses, so the node reads with
    the same role and agent as the single-tool result node it stands beside.
    """
    from langroid import ChatDocument, Entity
    from langroid.agent.chat_document import ChatDocMetaData
    if agent is not None and hasattr(agent, "response_template"):
        return agent.response_template(Entity.AGENT, content=text)
    return ChatDocument(content=text, metadata=ChatDocMetaData(sender=Entity.AGENT))


def _tool_message_text(tool_message: Any) -> str:
    """A tool message a handler returned, as Langroid serializes it."""
    try:
        return str(tool_message.to_json())
    except Exception:
        return str(tool_message)


def _handled_plan(outcome: _DispatchOutcome, tool_message: Any, agent: Any) -> tuple[_Batch, int, int]:
    """Plan the two nodes of a handler that returned another tool message.

    A handler may return another tool message, which Langroid dispatches at
    once, inside the first tool's dispatch. Two things happened: the first tool
    produced that message as its result, and the agent then asked for the tool
    written in it. They are recorded as the two nodes an ordinary reply has --
    the tool's result, and the message asking for the next call, computed from
    it -- so the next call is checked with the first tool's ``ToolResult`` among
    what it depends on, which is where every rule about a send computed from a
    secret read looks.
    """
    text = _tool_message_text(tool_message)
    result, request = _tool_result_message(text, agent), _tool_result_message(text, agent)
    batch = _Batch(agent)
    return (batch,
            batch.add(result, list(outcome.input_ids), derived_from=outcome.tool),
            batch.add(request, [_origin(result)]))


def _handled_result_id(outcome: _DispatchOutcome, tool_message: Any, agent: Any) -> str:
    """Record those two nodes, and name the one the nested call is checked on.

    Recorded synchronously, because the dispatch Langroid makes for a
    handler-returned tool message is synchronous even on the asynchronous path.
    """
    if outcome.result_id is None:
        batch, result, request = _handled_plan(outcome, tool_message, agent)
        identifiers = batch.resolve()
        outcome.result_id, outcome.request_id = identifiers[result], identifiers[request]
    return str(outcome.request_id)


def _dispatch_input_ids(outcome: _DispatchOutcome, tool_message: Any, agent: Any) -> list[str]:
    """What this dispatch is checked against, and recorded as computed from.

    A second gate of one operation is checked against what the first was. A
    nested call of another tool is checked against the result of the dispatch
    that produced it, not against the reply that asked for the first tool.
    """
    if outcome.root is not None and outcome.root is not outcome:
        return list(outcome.root.input_ids)
    if outcome.parent is not None:
        return [_handled_result_id(outcome.parent, tool_message, agent)]
    return list(get_current_input_ids())


async def _dispatch_input_ids_async(outcome: _DispatchOutcome, tool_message: Any, agent: Any) -> list[str]:
    """Asynchronous equivalent of :func:`_dispatch_input_ids`."""
    parent = outcome.parent
    if (outcome.root is not None and outcome.root is outcome and parent is not None
            and parent.result_id is None):
        batch, result, request = _handled_plan(parent, tool_message, agent)
        identifiers = await batch.resolve_async()
        parent.result_id, parent.request_id = identifiers[result], identifiers[request]
    return _dispatch_input_ids(outcome, tool_message, agent)


def _plan_dispatch(outcome: _DispatchOutcome, batch: _Batch, agent: Any) -> list[str]:
    """Plan the nodes behind one top-level call, and return what the reply reads.

    The result of a dispatch whose handler returned another tool message is
    already recorded, and what the reply reads from that branch is the nested
    call's own result. Everything else contributes one node per operation that
    ran: its text is that tool's result, its provenance that tool, and its
    dependencies what the call was checked against.
    """
    reachable: list[str] = []
    for child in outcome.children:
        reachable.extend(_plan_dispatch(child, batch, agent))
    if reachable:
        return reachable
    if outcome.request_id is not None:
        return [outcome.request_id]
    if not _operation_completed(outcome):
        return []
    node = _tool_result_message(_result_text(outcome.result), agent)
    batch.add(node, list(outcome.input_ids), derived_from=outcome.tool)
    return [_origin(node)]


def _plan_response(message: Any, input_ids: list[str], agent: Any,
                   outcomes: list[_DispatchOutcome]) -> tuple[_Batch, int]:
    """Plan a responder's output together with the tool results behind it.

    One call, handled straight through: the response itself is the tool-result
    node, the single-call shape every recorded run has. Otherwise the combined
    response names no single tool, so each call's own result becomes a node of
    its own with that call's provenance, and the response is recorded as
    computed from them, so every ``ToolResult`` stays reachable from whatever
    reads the response.
    """
    batch = _Batch(agent)
    dependencies = list(input_ids)
    single = (len(outcomes) == 1 and not outcomes[0].children
              and outcomes[0].result_id is None and _operation_completed(outcomes[0]))
    if not single:
        for outcome in outcomes:
            # Appended after the consumed history, so the inputs keep the
            # positions they were read in and the last tool result is the
            # proximal one: the response is the join of those results.
            dependencies.extend(_plan_dispatch(outcome, batch, agent))
    return batch, batch.add(message, dependencies,
                            derived_from=outcomes[0].tool if single else None)


def _record_response(message: Any, input_ids: list[str], agent: Any,
                     outcomes: list[_DispatchOutcome]) -> str:
    batch, index = _plan_response(message, input_ids, agent, outcomes)
    return batch.resolve()[index]


async def _record_response_async(message: Any, input_ids: list[str], agent: Any,
                                 outcomes: list[_DispatchOutcome]) -> str:
    batch, index = _plan_response(message, input_ids, agent, outcomes)
    return (await batch.resolve_async())[index]


def _text_tool_calls(content: str) -> list[dict[str, Any]]:
    """The tool calls written into a message's text, read from the text alone.

    Langroid lets a model call a tool by writing it into its reply, as a JSON
    object or as an XML ``<tool>`` block. Both forms are found here the way
    Langroid finds them, but without validating the call against any agent's
    tool registry, so the same text yields the same calls for every reader.
    Text that does not parse is not a call: a model that writes a broken one
    gets Langroid's own validation feedback, and the recorder stays out of it.
    """
    from langroid.agent.xml_tool_message import XMLToolMessage
    from langroid.parsing.parse_json import extract_top_level_json

    xml = XMLToolMessage.find_candidates(content)
    calls = []
    for candidate in xml or extract_top_level_json(content):
        try:
            value = XMLToolMessage.extract_field_values(candidate) if xml else json.loads(candidate)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        # Some models wrap the call in a JSON-schema envelope. Langroid
        # dispatches what is under "properties", so that is the call.
        inner = value.get("properties")
        calls.append(inner if isinstance(inner, dict) else value)
    return calls


def get_tools(msg: LLMMessage | ChatDocument) -> list[Tool]:
    """The tool calls a message carries, as the message itself states them.

    Structured calls come from the provider fields; calls written into the
    text are read from the text (see :func:`_text_tool_calls`). Nothing here
    depends on the agent doing the reading, so the agent that emitted a call
    and the agent that handles it record the same message.
    """
    from langroid.language_models import LLMMessage
    tools = []

    def arguments(name: str | None, value: Any) -> str:
        # When an agent handles a function or tool call, Langroid writes
        # ``request=<tool name>`` into the call's live arguments dict
        # (Agent.get_function_call_class / get_oai_tool_calls_classes). The
        # same message then reads differently before and after handling
        # while nothing was said that ``Tool.name`` does not already carry,
        # so the tag is left out of the recorded arguments.
        if isinstance(value, dict) and value.get("request") == name:
            value = {key: item for key, item in value.items() if key != "request"}
        return json.dumps(value)

    if msg.function_call is not None:
        tools.append(
            Tool(
                name=msg.function_call.name,
                arguments=arguments(msg.function_call.name, msg.function_call.arguments),
            )
        )

    if isinstance(msg, LLMMessage):
        oai_tool_calls = msg.tool_calls
    else:
        oai_tool_calls = msg.oai_tool_calls

    if oai_tool_calls is not None:
        for oai_tool in oai_tool_calls:
            if oai_tool.function is not None:
                tools.append(
                    Tool(
                        name=oai_tool.function.name,
                        arguments=arguments(oai_tool.function.name, oai_tool.function.arguments),
                    )
                )

    if msg.content:
        for call in _text_tool_calls(msg.content):
            name = call.get("request")
            if isinstance(name, str) and name:
                tools.append(Tool(name=name, arguments=json.dumps(call)))

    return tools


def is_instrumented() -> bool:
    """Whether :func:`instrument` has patched Langroid in this process.

    Code that records on the adapter's behalf, such as a ``Task.result``
    override calling :func:`record_task_result`, records only when the
    adapter is in place; without it nothing else is recorded either.
    """
    return instrument.cache_info().currsize > 0


# lru_cache makes instrument() idempotent: the wrappers stack, so installing
# twice would check and record every call twice.
@lru_cache
def instrument() -> None:
    """Install observation at finalized model inputs and supported dispatch sites."""
    get_tracer("instrumentation.langroid")
    if not get_config().tool_policy_fail_closed:
        # The gate is the only setting that lets unauthorized work through, so
        # a run that turned it off says so once, where the run starts.
        logger.warning(
            "tool_policy_fail_closed is off: a tool call whose authorization decision cannot be "
            "obtained runs anyway. This is for local experiments; enforcement needs it on."
        )
    try:
        from langroid import ChatDocument

        def responder_wrapper(responder: str, asynchronous: bool, *, finalized: bool = False):
            def prepare(args, kwargs):
                key = "messages" if finalized else "message"
                value = args[0] if args else kwargs.get(key, kwargs.get("msg"))
                original: list[Any] = list(value or []) if finalized else [value] if isinstance(value, ChatDocument) else []
                for message in original:
                    _origin(message)
                detached: list[Any] = copy.deepcopy(original)
                supplied: Any
                if finalized:
                    supplied = detached
                elif isinstance(value, ChatDocument):
                    supplied = detached[0]
                else:
                    supplied = value
                    detached = ChatDocument.to_LLMMessage(value) if isinstance(value, str) else []
                if args:
                    dispatch_args, dispatch_kwargs = (supplied, *args[1:]), kwargs
                else:
                    dispatch_args, dispatch_kwargs = args, dict(kwargs)
                    if key in kwargs:
                        dispatch_kwargs[key] = supplied
                    elif "msg" in kwargs:
                        dispatch_kwargs["msg"] = supplied
                return detached, supplied, original, dispatch_args, dispatch_kwargs

            def publish_inputs(original, detached):
                # Publish only the resolved input version back to the caller's
                # objects. Subsequent output edits belong to the detached object.
                for source, consumed in zip(original, detached):
                    version = _version(consumed)
                    if version is not None:
                        _versions(source)[(current_wire_session_id(), _origin(source))] = copy.deepcopy(version)

            if asynchronous:
                async def async_wrapper(wrapped, instance, args, kwargs):
                    if not is_session_active():
                        return await wrapped(*args, **kwargs)
                    messages, value, original, dispatch_args, dispatch_kwargs = prepare(args, kwargs)
                    input_ids = await _resolve_inputs_async(messages, instance)
                    publish_inputs(original, messages)
                    token = _current_input_ids.set(input_ids)
                    outcomes: list[_DispatchOutcome] = []
                    outcome_token = _dispatch_outcomes.set(outcomes)
                    # A responder reached from inside a handler (a sub-task)
                    # opens its own reply: its calls are that reply's, not the
                    # enclosing handler's result.
                    dispatch_token = _current_dispatch.set(None)
                    agent_name = getattr(instance.config, "name", "langroid")
                    with get_tracer("instrumentation.langroid").start_as_current_span(f"langroid.{responder}_response") as span:
                        span.set_attribute("_input_message_ids", ",".join(input_ids))
                        try:
                            with FeedbackScope(agent_name, responder) as feedback:
                                output = await wrapped(*dispatch_args, **dispatch_kwargs)
                            feedback.process(output, instance)
                            if output is not None:
                                identifier = await _record_response_async(output, input_ids, instance, outcomes if responder == "agent" else [])
                                span.set_attribute("_output_message_id", identifier)
                            span.set_status(StatusCode.OK)
                            return output
                        except Exception as error:
                            span.record_exception(error)
                            span.set_status(StatusCode.ERROR, str(error))
                            raise
                        finally:
                            _dispatch_outcomes.reset(outcome_token)
                            _current_dispatch.reset(dispatch_token)
                            _current_input_ids.reset(token)
                return async_wrapper

            def wrapper(wrapped, instance, args, kwargs):
                if not is_session_active():
                    return wrapped(*args, **kwargs)
                messages, value, original, dispatch_args, dispatch_kwargs = prepare(args, kwargs)
                input_ids = _resolve_inputs(messages, instance)
                publish_inputs(original, messages)
                token = _current_input_ids.set(input_ids)
                outcomes: list[_DispatchOutcome] = []
                outcome_token = _dispatch_outcomes.set(outcomes)
                dispatch_token = _current_dispatch.set(None)
                agent_name = getattr(instance.config, "name", "langroid")
                with get_tracer("instrumentation.langroid").start_as_current_span(f"langroid.{responder}_response") as span:
                    span.set_attribute("_input_message_ids", ",".join(input_ids))
                    try:
                        with FeedbackScope(agent_name, responder) as feedback:
                            output = wrapped(*dispatch_args, **dispatch_kwargs)
                        feedback.process(output, instance)
                        if output is not None:
                            identifier = _record_response(output, input_ids, instance, outcomes if responder == "agent" else [])
                            span.set_attribute("_output_message_id", identifier)
                        span.set_status(StatusCode.OK)
                        return output
                    except Exception as error:
                        span.record_exception(error)
                        span.set_status(StatusCode.ERROR, str(error))
                        raise
                    finally:
                        _dispatch_outcomes.reset(outcome_token)
                        _current_dispatch.reset(dispatch_token)
                        _current_input_ids.reset(token)
            return wrapper

        def conversion(wrapped, instance, args, kwargs):
            if not is_session_active():
                return wrapped(*args, **kwargs)
            source = args[0] if args else kwargs.get("message")
            # Keep the exact source consumed by conversion. A later edit of the
            # original ChatDocument must not change this view's historical input.
            snapshot = copy.deepcopy(source) if isinstance(source, ChatDocument) else None
            output = wrapped(*args, **kwargs)
            if snapshot is None and isinstance(source, str) and output:
                created = _document(output[0])
                snapshot = copy.deepcopy(created) if created is not None else None
            if snapshot is not None:
                for message in output:
                    object.__setattr__(message, "_sasy_conversion_source", snapshot)
                    object.__setattr__(message, "_sasy_conversion_fingerprint", message.model_dump(mode="json"))
            return output

        wrap_function_wrapper("langroid.agent.chat_document", "ChatDocument.to_LLMMessage", conversion)

        def system_message(wrapped, instance, args, kwargs):
            if not is_session_active():
                return wrapped(*args, **kwargs)
            output = wrapped(*args, **kwargs)
            origin = getattr(instance, "_sasy_system_origin", None)
            if origin is None:
                origin = str(uuid4())
                object.__setattr__(instance, "_sasy_system_origin", origin)
            object.__setattr__(output, "_observability_id", origin)
            return output

        wrap_function_wrapper("langroid.agent.chat_agent", "ChatAgent._create_system_and_tools_message", system_message)

        # Preparation may add current user input, replace system instructions, or
        # truncate history. The finalized list is the model's actual input set.
        for asynchronous in (False, True):
            suffix = "_async" if asynchronous else ""
            wrap_function_wrapper("langroid.agent.chat_agent", f"ChatAgent.llm_response_messages{suffix}",
                                  responder_wrapper("llm", asynchronous, finalized=True))
            for responder in ("llm", "agent", "user"):
                wrap_function_wrapper("langroid.agent.base", f"Agent.{responder}_response{suffix}",
                                      responder_wrapper(responder, asynchronous))

        def truncate(wrapped, instance, args, kwargs):
            if not is_session_active():
                return wrapped(*args, **kwargs)
            index = args[0] if args else kwargs["idx"]
            source = instance.message_history[index]
            inputs = _resolve_inputs([source], instance)
            output = wrapped(*args, **kwargs)
            _record_output(output, inputs, instance)
            return output

        wrap_function_wrapper("langroid.agent.chat_agent", "ChatAgent.truncate_message", truncate)

        def task_handoff(result: bool):
            def wrapper(wrapped, instance, args, kwargs):
                if not is_session_active():
                    return wrapped(*args, **kwargs)
                source = instance.pending_message if result else args[0] if args else kwargs.get("msg")
                if isinstance(source, ChatDocument):
                    inputs = _resolve_inputs([source], instance.agent)
                else:
                    inputs = []
                token = _current_input_ids.set(inputs)
                try:
                    output = wrapped(*args, **kwargs)
                    pending = output if result else output or instance.pending_message
                    if isinstance(pending, ChatDocument):
                        if pending is source:
                            _resolve_inputs([pending], instance.agent)
                        elif not result and source is None:
                            # A task started without a message continues from
                            # the agent's own history: the pending message is
                            # one the run already produced, and it keeps the
                            # dependencies it was recorded with. Recording it
                            # again here would replace them with none and hide
                            # everything the next tool call was computed from.
                            if _version(pending) is None:
                                raise MessageProvenanceError(
                                    "The message this task continues from has not been observed in this "
                                    "session; record its actual inputs with record_message, or start the "
                                    "task with the message it should continue from"
                                )
                            _resolve_inputs([pending], instance.agent)
                        else:
                            _record_output(pending, inputs, instance.agent)
                    return output
                finally:
                    _current_input_ids.reset(token)
            return wrapper

        wrap_function_wrapper("langroid.agent.task", "Task.init", task_handoff(False))
        wrap_function_wrapper("langroid.agent.task", "Task.result", task_handoff(True))

        def attach_parent(wrapped, instance, args, kwargs):
            if not is_session_active():
                return wrapped(*args, **kwargs)
            result = args[2] if len(args) > 2 else kwargs.get("result")
            previous = _version(result) if result is not None else None
            if previous is not None and _parent(result) != previous.parent_id:
                raise MessageProvenanceError("Unexplained ChatDocument parent change before task bookkeeping")
            output = wrapped(*args, **kwargs)
            if previous is not None:
                # Task bookkeeping attaches a parent after the responder's actual
                # input edges were recorded. It must not rewrite those edges.
                previous.parent_id = _parent(result)
            return output

        wrap_function_wrapper("langroid.agent.task", "Task._process_valid_responder_result", attach_parent)

        def _extract_denial_message(response) -> str:
            """Extract denial message from ToolCallResponse, including all reasons."""
            if response.denial_trace:
                msg = response.denial_trace.action_description
                if response.denial_trace.reasons:
                    reason_parts = [
                        r.details if r.details else str(r.reason_type)
                        for r in response.denial_trace.reasons
                    ]
                    msg += ": " + "; ".join(reason_parts)
                return msg
            return "Tool call not authorized"

        def _extract_suggestions(response) -> list[str]:
            """Extract all suggestions from ToolCallResponse."""
            if response.denial_trace and response.denial_trace.suggested_fixes:
                return list(response.denial_trace.suggested_fixes)
            return []

        def _log_tool_denial(fn_name: str, message: str, suggestions: list[str]) -> None:
            """Print a colored denial line with message and suggestion."""
            # Strip redundant "Tool call: <name>: " prefix from message
            prefix = f"Tool call: {fn_name}: "
            reason = message[len(prefix):] if message.startswith(prefix) else message
            line = f"\033[91m\033[1m[POLICY] DENIED {fn_name} - {reason}\033[0m"
            print(capture_text(line))
            for s in suggestions:
                print(f"\033[93m[POLICY]   Suggestion: {capture_text(s)}\033[0m")

        def _format_denial(fn_name: str, message: str, suggestions: list[str]) -> str:
            """Format a denial message string."""
            denial_msg = f"[BLOCKED] {fn_name}: {message}"
            if suggestions:
                if len(suggestions) == 1:
                    denial_msg += f"\nRequired action: {suggestions[0]}"
                else:
                    denial_msg += "\nRequired actions:\n" + "\n".join(f"  - {s}" for s in suggestions)
            return denial_msg

        def _refuse_self_named_handler(fn_name: str, fn_args: str, named: str) -> str:
            """Refuse a tool call that names its own handler, before anything runs.

            The call would be checked and recorded under its ``request`` name
            while another agent method ran, so there is no decision to ask for:
            nothing runs and the agent is told why.
            """
            message = (f"a tool call may not name its own handler (it asked for '{named}'); "
                       "the handler is declared by the tool class")
            add_tool_denial(fn_name, message, fn_args, [])
            if get_config().log_policy_decisions:
                _log_tool_denial(fn_name, message, [])
            logger.warning(f"Tool call refused: {fn_name} - {message}")
            return _format_denial(fn_name, message, [])

        def wrap_handle_tool_message(wrapped, instance, args, kwargs):
            """Authorize the tool call before Langroid runs its handler.

            A denial, a required transform, or (when ``tool_policy_fail_closed``
            is true, the default) any failure to reach a decision returns a
            ``[BLOCKED] ...`` string to the agent instead of running the
            handler. A dispatch with no responder around it is refused: see
            :func:`_begin_dispatch`.
            """
            if not is_session_active():
                return wrapped(*args, **kwargs)
            tool = args[0] if args else kwargs.get("tool")
            if tool is None:
                return wrapped(*args, **kwargs)

            fn_name = tool.default_value("request")
            try:
                fn_args = json.dumps(tool.model_dump())
            except Exception:
                fn_args = str(tool)

            outcome = _begin_dispatch(fn_name, fn_args, tool)
            input_ids = _dispatch_input_ids(outcome, tool, instance)
            outcome.input_ids = tuple(input_ids)

            # Refused like a denial, and placed like one: the dispatch is part
            # of the reply's tree, unauthorized, so a sibling that did run keeps
            # its own result and a parent that returned this call keeps its own.
            # No check is made and the handler does not run.
            named = _redirected_handler(tool)
            if named is not None:
                return _refuse_self_named_handler(fn_name, fn_args, named)
            config = get_config()

            try:
                response = rm_check_tool_call(fn_name, fn_args, input_ids)
                if response.authorized and response.transform_ids:
                    message = "Required tool transforms are not supported by the Langroid adapter"
                    add_tool_denial(fn_name, message, fn_args, [])
                    return _format_denial(fn_name, message, [])
                if not response.authorized:
                    message = _extract_denial_message(response)
                    suggestions = _extract_suggestions(response)
                    add_tool_denial(fn_name, message, fn_args, suggestions)
                    if config.log_policy_decisions:
                        _log_tool_denial(fn_name, message, suggestions)
                    logger.warning(f"Tool call denied: {fn_name} - {message}")
                    return _format_denial(fn_name, message, suggestions)
                elif config.log_policy_decisions:
                    print(
                        f"\033[92m\033[1m[POLICY] AUTHORIZED {fn_name}\033[0m"
                    )

            except Exception as e:
                if config.tool_policy_fail_closed:
                    logger.warning(f"Tool call denied (RM unavailable, fail_closed=True): {fn_name} - {e}")
                    # The model sees only the failure's type. The message
                    # can name the endpoint or other server detail, so it
                    # stays in the log line above.
                    reason = (f"could not get an authorization decision "
                              f"({type(e).__name__})")
                    add_tool_denial(fn_name, reason, fn_args, [])
                    return (f"[BLOCKED] {fn_name}: {reason}. Tool calls fail closed; "
                            "check SASY_URL, the credentials and that the engine is "
                            "running, or set tool_policy_fail_closed=False for local "
                            "experiments.")
                else:
                    logger.warning(f"Tool call ran without an authorization decision "
                                   f"(tool_policy_fail_closed=False): {fn_name} - {e}")

            outcome.authorized = True
            token = _current_dispatch.set(outcome)
            # The handler runs on what this call was checked against, which for
            # a nested call is its parent tool's result and not the reply that
            # asked for the first tool. An HTTP request or an explicit
            # handle_message inside the handler reads the same context.
            inputs_token = _current_input_ids.set(input_ids)
            try:
                result = wrapped(*args, **kwargs)
            finally:
                _current_input_ids.reset(inputs_token)
                _current_dispatch.reset(token)
            # Langroid returns None when no handler accepted the tool message.
            # Only a handled dispatch may establish ToolResult provenance.
            outcome.completed = result is not None
            outcome.result = result
            return result

        async def wrap_handle_tool_message_async(wrapped, instance, args, kwargs):
            """Async version of tool call authorization wrapper."""
            if not is_session_active():
                return await wrapped(*args, **kwargs)
            tool = args[0] if args else kwargs.get("tool")
            if tool is None:
                return await wrapped(*args, **kwargs)

            fn_name = tool.default_value("request")
            try:
                fn_args = json.dumps(tool.model_dump())
            except Exception:
                fn_args = str(tool)

            outcome = _begin_dispatch(fn_name, fn_args, tool)
            input_ids = await _dispatch_input_ids_async(outcome, tool, instance)
            outcome.input_ids = tuple(input_ids)

            # Refused like a denial, and placed like one (see the sync wrapper).
            named = _redirected_handler(tool)
            if named is not None:
                return _refuse_self_named_handler(fn_name, fn_args, named)
            config = get_config()

            try:
                response = await rm_check_tool_call_async(fn_name, fn_args, input_ids)
                if response.authorized and response.transform_ids:
                    message = "Required tool transforms are not supported by the Langroid adapter"
                    add_tool_denial(fn_name, message, fn_args, [])
                    return _format_denial(fn_name, message, [])
                if not response.authorized:
                    message = _extract_denial_message(response)
                    suggestions = _extract_suggestions(response)
                    add_tool_denial(fn_name, message, fn_args, suggestions)
                    if config.log_policy_decisions:
                        _log_tool_denial(fn_name, message, suggestions)
                    logger.warning(f"Tool call denied: {fn_name} - {message}")
                    return _format_denial(fn_name, message, suggestions)
                elif config.log_policy_decisions:
                    print(
                        f"\033[92m\033[1m[POLICY] AUTHORIZED {fn_name}\033[0m"
                    )

            except Exception as e:
                if config.tool_policy_fail_closed:
                    logger.warning(f"Tool call denied (RM unavailable, fail_closed=True): {fn_name} - {e}")
                    # The model sees only the failure's type. The message
                    # can name the endpoint or other server detail, so it
                    # stays in the log line above.
                    reason = (f"could not get an authorization decision "
                              f"({type(e).__name__})")
                    add_tool_denial(fn_name, reason, fn_args, [])
                    return (f"[BLOCKED] {fn_name}: {reason}. Tool calls fail closed; "
                            "check SASY_URL, the credentials and that the engine is "
                            "running, or set tool_policy_fail_closed=False for local "
                            "experiments.")
                else:
                    logger.warning(f"Tool call ran without an authorization decision "
                                   f"(tool_policy_fail_closed=False): {fn_name} - {e}")

            outcome.authorized = True
            token = _current_dispatch.set(outcome)
            # See the synchronous wrapper: the handler runs on what this call
            # was checked against.
            inputs_token = _current_input_ids.set(input_ids)
            try:
                result = await wrapped(*args, **kwargs)
            finally:
                _current_input_ids.reset(inputs_token)
                _current_dispatch.reset(token)
            # None means no handler accepted the tool message; see the
            # synchronous wrapper above.
            outcome.completed = result is not None
            outcome.result = result
            return result

        def wrap_to_chat_document(wrapped, instance, args, kwargs):
            """Mark the value Langroid is converting while it converts it.

            A handler that returned another tool message has it dispatched from
            here, inside the first tool's dispatch. The dispatch tree tells that
            chain from a call a handler makes itself by this value's identity.
            """
            if not is_session_active():
                return wrapped(*args, **kwargs)
            value = args[0] if args else kwargs.get("msg")
            token = _handler_result.set(value)
            try:
                return wrapped(*args, **kwargs)
            finally:
                _handler_result.reset(token)

        wrap_function_wrapper(
            "langroid.agent.base",
            name="Agent.to_ChatDocument",
            wrapper=wrap_to_chat_document,
        )

        wrap_function_wrapper(
            "langroid.agent.base",
            name="Agent.handle_tool_message",
            wrapper=wrap_handle_tool_message,
        )

        wrap_function_wrapper(
            "langroid.agent.base",
            name="Agent.handle_tool_message_async",
            wrapper=wrap_handle_tool_message_async,
        )

    except ImportError as exc:
        raise RuntimeError(
            "Langroid is installed but SASY's Langroid adapter cannot load with it; install "
            "sasy[langroid], or, if this application does not use Langroid, pass langroid=False "
            "to sasy.instrument()."
        ) from exc
