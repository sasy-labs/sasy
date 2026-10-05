"""LangChain agent adapter: records the messages an agent consumes and
produces, and authorizes each tool call immediately before it runs.

Only the agent shape built by :func:`create_agent` below is supported: tools
must consume their explicit arguments alone, and hidden state, custom tool
implementations, checkpoint resumption and custom middleware are not supported
by this adapter.

:func:`instrument_langchain`, which ``sasy.instrument()`` calls, makes
``langchain.agents.create_agent`` build that agent too, and refuses, inside a
SASY session, every tool call of a LangGraph ``ToolNode`` that SASY did not
build.

*Supported mode* below means the one this adapter is tested in, in which every
invocation sends its whole conversation and the provider retains nothing
between calls.
"""
from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any, get_args
from uuid import uuid4
from weakref import WeakSet

import grpc

# Imports of the optional framework happen only when this module is requested.
from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import _is_injected_arg_type
from pydantic import BaseModel

from sasy.instrumentation import _rebind_module_globals, dependencies
from sasy.instrumentation._canonical import encoder as _canonical_encoder
from sasy.instrumentation.otel.context import _current_input_ids
from sasy.instrumentation.session import (
    current_wire_session_id,
    is_session_active,
    require_active_session,
)
from sasy.observability import api as observation
from sasy.observability._snapshots import VERSION_ID_PREFIX
from sasy.proto.observability_pb2 import (
    AGENT,
    LLM,
    SYSTEM,
    USER,
    Edge,
    Event,
    EventSnapshot,
    Tool,
)
from sasy.reference_monitor import api as reference_monitor


class InstrumentationError(RuntimeError):
    """Required evidence or a supported execution boundary is unavailable."""


class ActionDenied(InstrumentationError):
    """SASY rejected the actual dispatch, including unsupported transforms."""


# Exact pins, checked in create_agent() and instrument_langchain(): the adapter
# relies on the order the middleware hooks run in, on a private helper
# (_is_injected_arg_type) and, for the backstop, on ToolNode's private per-call
# methods.
# A different version could skip a hook without failing, and a skipped hook is
# a missed authorization check rather than an error.
_SUPPORTED = {
    "langchain": "1.4.0", "langchain-core": "1.6.3",
    "langgraph": "1.2.11", "langgraph-prebuilt": "1.1.0",
}

# The tool functions _gate() produced. A tool is protected when every function
# it can run is one of these; the ToolNode backstop's second line of defence
# reads this set.
_GATED: WeakSet[Callable[..., Any]] = WeakSet()

# The LangChain guide lists what the adapter supports and why.
_GUIDE = "https://docs.sasy.ai/integrations/langchain/"


# A guarded agent may be invoked inside a guarded tool body. The cap fails
# closed instead of exhausting the interpreter stack on unbounded delegation.
_MAXIMUM_NESTING = 8


# The adapter marks each message it resolved with the immutable node that
# message is, so a caller can hand the same objects back on a later invocation.
# The key is stripped from the copy handed to any model, which is what keeps it
# away from a provider: response_metadata is not inert — the pinned
# langchain-openai reads response_metadata["id"] on its responses path to decide
# which messages it sends — so the strip, not the field, is the guarantee.
_MARK_KEY = "sasy"
_MARK_VERSION = 1

# ResolveSnapshots answers INVALID_ARGUMENT when it has decided that a claimed
# reference is not a version of that origin in this session for this principal.
# That is the only status that is a decision about the claim. Every other one
# (UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL, PERMISSION_DENIED and the rest)
# means the question was not answered and stays fatal: downgrading on an outage
# would silently drop real ancestry.
_REFUSAL = grpc.StatusCode.INVALID_ARGUMENT


# The canonical encoding lives in a module of its own, shared with the ADK
# adapter; it refuses what it cannot write down with this adapter's error type.
_json, _canonical, _canonical_key = _canonical_encoder(InstrumentationError)


def _unmarked(message: BaseMessage) -> dict[str, Any]:
    """The message as the graph sees it: the adapter's own mark is not content.

    The dump keeps Python types (``model_dump()``, not ``mode="json"``) because
    :func:`_canonical`, not pydantic, decides how they are written down.
    """
    data = message.model_dump()
    metadata = data.get("response_metadata")
    if isinstance(metadata, dict) and _MARK_KEY in metadata:
        data["response_metadata"] = {key: value for key, value in metadata.items() if key != _MARK_KEY}
    return data


def _projection(message: BaseMessage) -> dict[str, Any]:
    """The whole message as the graph records it, minus the adapter's own mark.

    This is what an event's metadata carries, and the reason it carries all of
    it: the event's other fields are a projection for policies to read, and not
    one of them is injective. Two message classes share a role, an absent name
    and the literal name ``langchain`` give one agent, the text is the content
    alone, and the recorded tool list is the single source the client sends
    rather than every source a call can hide in. A field left out of the event
    is therefore a field a message can be edited in without changing its
    recorded event, which is the same as saying its mark would still verify
    after the edit. The engine's content hash is taken over the whole event,
    metadata included, so carrying the message there is what closes that gap.

    Naming the fields a provider reads and recording only those was an earlier
    design, and it cannot be made to hold: in the pinned ``langchain-openai``
    alone the request path reads ``name``, ``tool_calls``, ``function_call``,
    ``audio`` and ``__openai_role__`` out of ``additional_kwargs`` on the
    chat-completions path, half a dozen more on the responses path, and
    ``response_metadata["id"]`` to decide which messages are sent at all. So the
    record is the message, and the list never has to be kept up to date.

    The LangChain id is in it too. It is not only a name the caller chose: on the
    responses path the pinned client stamps an id beginning ``msg_`` onto the
    text block it sends, and reads the same id to decide whether to run the
    conversion that reorders reasoning and text. Identity is still checked
    separately — a version belongs to its origin alias — but the id is content
    as well, so it is recorded as content.
    """
    return _unmarked(message)


def _snapshot(message: BaseMessage) -> str:
    """The canonical encoding of *message*, which is also its recorded metadata.

    Distinct messages have distinct snapshots, so one recorded event can never
    stand for two different messages, whatever a provider would make of them.
    """
    return _json(_canonical(_projection(message)))


def _text(message: BaseMessage) -> str:
    """The message's content, as a policy reads it.

    A policy matches on the text of a node, so the text is what the message
    says: string content as it stands, and structured content — a list of
    blocks — as its canonical JSON, since there is no other way to write it as
    one string. Two messages can therefore share a text; what keeps their
    records apart is :func:`_snapshot`, in the event's metadata.
    """
    content = message.content
    # Exact type, as everywhere in this encoding: a str subclass is a type the
    # adapter has not anticipated, and _canonical refuses it by name.
    if type(content) is str:
        return content
    return _json(_canonical(content))


def _snapshot_ignoring_id(message: BaseMessage) -> str:
    """The same encoding, without the LangChain id.

    Admission uses this one, because the question it asks — is this the message
    already recorded under the identity it claims? — is asked *about* an
    identity, and a replacement is recorded under an identity the run mints, so
    comparing the id as well would make one replacement two records.
    """
    data = _projection(message)
    data.pop("id", None)
    return _json(_canonical(data))


def _auxiliary_snapshot(message: BaseMessage) -> str:
    # The fields outside the graph's role/name/content/tool-call projection, and
    # outside the identity. The snapshot above covers them as well, so a change
    # here is caught either way; comparing them separately names the problem
    # before the round trip.
    excluded = ("id", "content", "name", "tool_calls")
    return _json(_canonical({key: value for key, value in _projection(message).items()
                             if key not in excluded}))


def _requested_tools(message: AIMessage) -> list[Tool]:
    """The tool calls the pinned provider client puts in the request for *message*.

    ``langchain-openai``'s ``_convert_message_to_dict`` picks one source and
    ignores the rest: the parsed and the unparseable tool calls together when
    the message carries any, otherwise the raw ``additional_kwargs["tool_calls"]``,
    otherwise the legacy ``function_call``. Recording the union of the sources
    instead would name a tool the provider is never asked to call. The fields
    a client ignores stay in the recorded content, where a mark cannot survive
    their being added; they are just not requests.

    An ``additional_kwargs`` entry with no function description is no call the
    client can make, so it too stays in the content alone.
    """
    arguments = message.additional_kwargs if isinstance(message.additional_kwargs, dict) else {}
    if message.tool_calls or message.invalid_tool_calls:
        return ([Tool(name=call["name"], arguments=_json(call["args"])) for call in message.tool_calls]
                + [Tool(name=str(call.get("name") or ""), arguments=str(call.get("args") or ""))
                   for call in message.invalid_tool_calls])
    if "tool_calls" in arguments:
        entries = arguments["tool_calls"] if isinstance(arguments["tool_calls"], list) else []
        functions = [entry.get("function") for entry in entries if isinstance(entry, dict)]
        return [Tool(name=str(function.get("name") or ""), arguments=str(function.get("arguments") or ""))
                for function in functions if isinstance(function, dict)]
    function_call = arguments.get("function_call")
    if "function_call" in arguments and isinstance(function_call, dict):
        return [Tool(name=str(function_call.get("name") or ""), arguments=str(function_call.get("arguments") or ""))]
    return []


def _mark(message: BaseMessage) -> dict[str, Any] | None:
    """The adapter's mark on a supplied message, if it has a usable shape."""
    metadata = getattr(message, "response_metadata", None)
    mark = metadata.get(_MARK_KEY) if isinstance(metadata, dict) else None
    if not isinstance(mark, dict) or mark.get("version") != _MARK_VERSION:
        return None
    if not isinstance(mark.get("node"), str) or not isinstance(mark.get("origin"), str):
        return None
    return mark


def _strip(messages: Sequence[BaseMessage]) -> None:
    for message in messages:
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, dict):
            metadata.pop(_MARK_KEY, None)


def _envelope(message: BaseMessage) -> str:
    """The recorded text of an input whose origin the adapter cannot establish.

    The whole message is under ``langchain_message``, for the reason
    :func:`_projection` gives, with its content repeated under ``value`` so a
    policy can read what the run consumed without unpacking the message. The
    provenance key is what a policy reads to tell this text from the content of
    a recorded message; the two cannot produce one event in any case, because
    the event's metadata is the message and these events carry a different text
    for the same message.
    """
    data = _projection(message)
    return _json(_canonical({
        "langchain_message": data,
        "provenance": "unattributed external input",
        "value": data.get("content"),
    }))


def _refused(error: Exception) -> bool:
    """Whether a failed resolution is a decision that the mark does not hold.

    Only a refusal downgrades a message to an input of unknown origin. An
    observation that could not be carried out — an unreachable, timed-out or
    erroring server — decides nothing, and treating it as a refusal would drop
    the ancestry of a message whose mark is in fact good, so it stays fatal.
    """
    if isinstance(error, InstrumentationError):
        return True
    return isinstance(error, grpc.RpcError) and error.code() == _REFUSAL


@dataclass
class _ToolScope:
    """One tool body's computation, shared by the pool thread or task running it.

    A body may delegate to guarded agents. Everything it has already consumed —
    the dispatch that started it and every inner answer returned to it so far —
    is an input of the next inner run it starts and of its own result.
    """

    session_id: str
    depth: int
    dispatch_inputs: tuple[str, ...]
    run: _Run
    answers: list[str] = field(default_factory=list)
    foreign: list[str] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock)
    closed: bool = False

    def consumed(self) -> tuple[str, ...]:
        self._admit_foreign()
        with self.lock:
            return tuple(dict.fromkeys(self.dispatch_inputs + tuple(self.answers)))

    def record(self, answers: Sequence[str]) -> None:
        with self.lock:
            self.answers.extend(answers)

    # The shared authorization boundary asks the current computation for its
    # inputs, so a check made inside the body after a delegation sees the
    # answers already returned to it.
    def open(self) -> None:
        if self.closed:
            raise InstrumentationError("An action outlived the tool body that started it")
        if self.run.caller is not None:
            self.run.caller.open()

    def check(self) -> None:
        self.run.verify()
        self.open()

    def inputs_sync(self) -> list[str]:
        return list(self.consumed())

    async def inputs_async(self) -> list[str]:
        return list(self.consumed())

    def defer(self, envelope: str) -> None:
        """Hold an answer returned from another session until it can be recorded.

        Message identities are not shared across sessions, so the answer cannot
        be a dependency of this body as itself. It is recorded in this body's own
        session as an input of unknown origin, which only this session can do.
        """
        with self.lock:
            self.foreign.append(envelope)

    def _admit_foreign(self) -> None:
        with self.lock:
            while self.foreign:
                identity = str(uuid4())
                message = HumanMessage(content=self.foreign[0], id=identity)
                self.run.external.add(identity)
                self.answers.append(self.run.observe(message, ()))
                self.foreign.pop(0)


@dataclass
class _Run:
    """One ``invoke``/``ainvoke`` of a guarded agent, and everything it has
    recorded so far.

    Messages are tracked by *identity*, the LangChain message id, and each
    identity is recorded under a stable *origin* so its successive versions
    stay one message to the engine.

    Attributes:
        session_id: The SASY session this run must stay in.
        depth: How many guarded agents are nested at this point.
        entry_parents: Version IDs the caller handed in, used as the inputs of
            the first messages this run records.
        caller: The tool body this run was started inside, if any.
        latest_answers: Version IDs of the messages the last model call
            produced.
        system_identity, origin_prefix: Per-run names, so two runs never share
            an origin.
        observations: (message identity, content snapshot) -> version ID.
        versions: Message identity -> its latest version ID.
        snapshots: Message identity -> the content snapshot this run last
            accepted for it, which decides whether a second claim on that
            identity is the same message or another one.
        replacements: (claimed identity, message) -> the minted identity the
            run recorded that message under, so one replacement consumed twice
            is one input.
        derivations: Message identity -> the inputs it was recorded with.
        auxiliary_snapshots: Message identity -> the fields outside the
            role/name/content/tool-call projection last recorded with it.
        calls, call_owners, message_calls: Tool-call bookkeeping: the call by
            id, which message requested it, and the calls each message carries.
        dispatches: Tool-call id -> the arguments actually dispatched.
        admitted: Identities restored from a mark this adapter can verify.
        external: Identities recorded as inputs of unknown origin. The envelope
            is written from the message each time, never cached against the
            identity: a cached one is a second thing the identity has to agree
            with, and the record has to be what the message is.
        origins: Identity -> the origin carried by its mark.
        evidence: Identity -> (tool name, arguments) of the successful tool
            result it reports.
        lock: Serializes recording within the run.
    """

    session_id: str
    depth: int = 0
    entry_parents: tuple[str, ...] = ()
    # The tool body that started this run, if any. A delegated run lives inside
    # that body: once the body has ended, the run may not observe or act.
    caller: _ToolScope | None = None
    latest_answers: tuple[str, ...] = ()
    system_identity: str = field(default_factory=lambda: str(uuid4()))
    origin_prefix: str = field(default_factory=lambda: str(uuid4()))
    observations: dict[tuple[str, str], str] = field(default_factory=dict)
    derivations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    snapshots: dict[str, str] = field(default_factory=dict)
    replacements: dict[tuple[str, str], str] = field(default_factory=dict)
    auxiliary_snapshots: dict[str, str] = field(default_factory=dict)
    calls: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    call_owners: dict[str, str] = field(default_factory=dict)
    message_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    dispatches: dict[str, dict[str, Any]] = field(default_factory=dict)
    admitted: set[str] = field(default_factory=set)
    external: set[str] = field(default_factory=set)
    origins: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, tuple[str, str]] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)

    def verify(self) -> None:
        if current_wire_session_id() != self.session_id:
            raise InstrumentationError("LangChain execution lost its SASY session")
        if self.caller is not None:
            self.caller.open()

    def origin(self, identity: str) -> str:
        """The client-side alias a message's versions are minted under.

        A message admitted from an earlier invocation keeps the alias it was
        first recorded under; its immutable versions are bound to that alias.
        """
        return self.origins.get(identity, f"{self.origin_prefix}:{identity}")

    def _event(self, message: BaseMessage, identity: str) -> Event:
        """The recorded event for *message*, which is a function of *message*.

        The metadata is the whole message (see :func:`_projection`), so no edit
        to it — content, its type, a raw provider keyword argument, the role a
        client would transmit — leaves the event, and therefore the mark that
        was issued for it, unchanged. The text, role, agent and tool entries
        beside it are the projection policies match on.
        """
        if identity in self.external:
            # An input of unknown origin carries no tool entries and no
            # derivation: it is visible to a policy, never evidence.
            return Event(id=self.origin(identity), text=_envelope(message), role=USER,
                         agent=message.name or "langchain", metadata=_snapshot(message))
        role = USER if isinstance(message, HumanMessage) else SYSTEM if isinstance(message, SystemMessage) else LLM if isinstance(message, AIMessage) else AGENT
        event = Event(id=self.origin(identity), text=_text(message), role=role,
                      agent=message.name or "langchain", metadata=_snapshot(message))
        if isinstance(message, AIMessage):
            # Only an assistant message asks a provider to call a tool, and the
            # recorded requests are the ones the client actually sends.
            event.tools.extend(_requested_tools(message))
        # A failed tool result says so in its own recorded status; it is never
        # evidence that the tool ran, so it gets no derivation.
        if isinstance(message, ToolMessage) and message.status != "error":
            if identity in self.evidence:
                event.derived_from.CopyFrom(Tool(name=self.evidence[identity][0], arguments=self.evidence[identity][1]))
            else:
                origin = self.calls.get(message.tool_call_id)
                if origin is None:
                    raise InstrumentationError("Tool result lacks an observed tool call")
                event.derived_from.CopyFrom(Tool(name=origin[1]["name"], arguments=_json(self.dispatches.get(message.tool_call_id, origin[1]["args"]))))
        return event

    def _untracked(self, message: BaseMessage, identity: str) -> None:
        self.admitted.discard(identity)
        self.versions.pop(identity, None)
        self.origins.pop(identity, None)
        self.evidence.pop(identity, None)
        self.external.add(identity)

    def admit(self, messages: Sequence[BaseMessage]) -> None:
        """Resolve the marks carried by supplied messages before the run starts.

        A verified mark restores the message's recorded node, with the ancestry
        and tool evidence that node already has. A message generated elsewhere,
        an unverifiable mark and an edited message are recorded as untracked
        external inputs instead of being refused, so a policy decides. An
        identity this run has already recorded keeps what it has as long as the
        message is the same one; this is also reached mid-run through
        :func:`consume_messages`.
        """
        self.verify()
        candidates: list[tuple[str, BaseMessage]] = []
        for message in messages:
            if not message.id:
                message.id = str(uuid4())
            identity = message.id
            snapshot = _snapshot_ignoring_id(message)
            if self.snapshots.get(identity, snapshot) != snapshot:
                # One identity, two different messages. What the run recorded
                # stands for the first message and says nothing about this
                # text, so this one is read below as the input it is, under an
                # identity of its own: an assistant or tool message as an input
                # of unknown origin, a user or system message as a fresh input,
                # and a mark that no longer describes its message as one the
                # engine refuses. Whatever the body consumed is recorded and
                # depended on, and nothing the first message established is
                # inherited.
                #
                # The identity is minted, not derived from the message: a
                # derived name is one a caller can work out and occupy in
                # advance, and the comparison would then skip past it and drop
                # the text. It is remembered per (claimed identity, message), so
                # one replacement consumed twice is one recorded input; and a
                # minted name is taken only while it holds this very message, so
                # the comparison guards it exactly as it guards a claimed one.
                minted = self.replacements.get((identity, snapshot))
                while minted is None or self.snapshots.get(minted, snapshot) != snapshot:
                    minted = self.replacements[(identity, snapshot)] = str(uuid4())
                identity = message.id = minted
            if identity in self.snapshots:
                # The same message claimed a second time. It keeps the node it
                # was recorded under, so a stored copy that lost its mark
                # cannot downgrade what the run already established.
                continue
            self.snapshots[identity] = snapshot
            mark = _mark(message)
            if mark is None:
                if not isinstance(message, (HumanMessage, SystemMessage)):
                    self._untracked(message, identity)
                continue
            self.admitted.add(identity)
            self.origins[identity] = mark["origin"]
            self.versions[identity] = mark["node"]
            tool = mark.get("tool")
            if isinstance(message, ToolMessage) and message.status != "error":
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not isinstance(tool.get("arguments"), str):
                    self._untracked(message, identity)
                    continue
                self.evidence[identity] = (tool["name"], tool["arguments"])
            candidates.append((identity, message))
        if not candidates:
            return
        try:
            self._resolve_marks(candidates)
            return
        except Exception as error:
            # A refused batch creates no version, so each mark can be retried
            # on its own to find the ones that do not hold.
            if not _refused(error):
                raise
        for identity, message in candidates:
            try:
                self._resolve_marks([(identity, message)])
            except Exception as error:
                if not _refused(error):
                    raise
                self._untracked(message, identity)

    def _resolve_marks(self, candidates: Sequence[tuple[str, BaseMessage]]) -> None:
        """Resolve each mark as a compact reference to the node it claims.

        The server accepts it only if the message hashes to the content of an
        immutable version of that alias in this session for this principal, so
        a copied, invented or stale mark cannot buy ancestry or tool evidence.
        """
        snapshots = [EventSnapshot(event=self._event(message, identity), base_id=self.versions[identity], reuse_dependencies=True)
                     for identity, message in candidates]
        if observation.resolve_events(snapshots) != [self.versions[identity] for identity, _ in candidates]:
            raise InstrumentationError("Marked message did not resolve to its recorded version")

    def observe(self, message: BaseMessage, parents: Sequence[str] | None = None) -> str:
        return self.observe_many([message], parents)[0]

    def observe_many(self, messages: Sequence[BaseMessage], parents: Sequence[str] | None = None) -> list[str]:
        """Resolve all consumed snapshots in one RPC before publishing their IDs."""
        self.verify()
        if parents is not None:
            parents = tuple(dict.fromkeys(parents))
        with self.lock:
            pending: dict[str, tuple[EventSnapshot, str, list[dict[str, Any]], tuple[str, ...], tuple[str, str] | None]] = {}
            auxiliary: dict[str, str] = {}
            unnamed: dict[str, str] = {}
            ordered = []
            for message in messages:
                if not message.id:
                    message.id = str(uuid4())
                identity = message.id
                ordered.append(identity)
                snapshot = _snapshot(message)
                unnamed[identity] = _snapshot_ignoring_id(message)
                extra = _auxiliary_snapshot(message)
                previous_extra = auxiliary.get(identity, self.auxiliary_snapshots.get(identity))
                if previous_extra is not None and previous_extra != extra:
                    raise InstrumentationError("Auxiliary message fields cannot change after observation")
                auxiliary[identity] = extra
                if parents is not None and identity in self.derivations and self.derivations[identity] != tuple(parents):
                    raise InstrumentationError("A message identity was reused for a new computation")
                # Tool calls carried on a message admitted from elsewhere are
                # history, never a dispatch this run may act on.
                historical = identity in self.admitted or identity in self.external
                calls: list[dict[str, Any]] = [copy.deepcopy(dict(call)) for call in message.tool_calls] if isinstance(message, AIMessage) and not historical else []
                seen: set[str] = set()
                for call in calls:
                    call_id = call.get("id")
                    if not call_id or call_id in seen or self.call_owners.get(call_id, identity) != identity:
                        raise InstrumentationError("Missing or reused tool-call ID")
                    seen.add(call_id)
                if identity in self.message_calls and self.message_calls[identity] != calls:
                    raise InstrumentationError("Tool-call specifications cannot change after observation")
                base = self.versions.get(identity)
                if parents is not None:
                    inputs = tuple(parents)
                elif identity in self.derivations:
                    inputs = self.derivations[identity]
                elif historical:
                    # An admitted message keeps the ancestry its node already
                    # has; an external input has none.
                    inputs = ()
                elif isinstance(message, (HumanMessage, SystemMessage)):
                    # A fresh input of a delegated run depends on what the tool
                    # body that started the run had consumed at that point.
                    inputs = self.entry_parents
                else:
                    raise InstrumentationError("Unobserved generated message: supply only user/system inputs")
                event = self._event(message, identity)
                # A generated message re-observed as input must still represent
                # its recorded computation. An unexplained edit cannot inherit
                # successful tool evidence or a guessed set of dependencies.
                reuse = base is not None and parents is None and (identity in self.admitted or not isinstance(message, (HumanMessage, SystemMessage)))
                provenance = (event.derived_from.name, event.derived_from.arguments) if event.HasField("derived_from") else None
                edges = [] if reuse else [Edge(source=p, destination=event.id, message_index=i) for i, p in enumerate(inputs)]
                item = EventSnapshot(event=event, base_id=base, dependencies=edges, reuse_dependencies=reuse)
                if identity in pending and pending[identity][0] != item:
                    raise InstrumentationError("Conflicting snapshots for one message identity")
                pending[identity] = (item, snapshot, calls, inputs, provenance)
            if not pending:
                return []
            ids = observation.resolve_events([item[0] for item in pending.values()])
            if len(ids) != len(pending) or any(not item.startswith(VERSION_ID_PREFIX) for item in ids):
                raise InstrumentationError("Observation did not confirm immutable identities")
            resolved = dict(zip(pending, ids, strict=True))
            for identity, (_, snapshot, calls, inputs, provenance) in pending.items():
                node_id = resolved[identity]
                self.observations[(identity, snapshot)] = node_id
                self.versions[identity] = node_id
                self.snapshots[identity] = unnamed[identity]
                self.auxiliary_snapshots[identity] = auxiliary[identity]
                self.derivations[identity] = inputs
                if provenance is not None:
                    self.evidence[identity] = provenance
                if calls or identity in self.message_calls:
                    self.message_calls[identity] = calls
                for call in calls:
                    self.call_owners[call["id"]] = identity
                    self.calls[call["id"]] = (node_id, call)
            return [resolved[identity] for identity in ordered]


_run: ContextVar[_Run | None] = ContextVar("sasy_langchain_run", default=None)
_tool_inputs: ContextVar[tuple[str, tuple[str, ...], str] | None] = ContextVar("sasy_langchain_tool", default=None)
_tool_scope: ContextVar[_ToolScope | None] = ContextVar("sasy_langchain_scope", default=None)


def _active() -> _Run:
    run = _run.get()
    if run is None:
        raise InstrumentationError("Use the SASY agent's invoke/ainvoke entry point")
    run.verify()
    return run


@contextmanager
def _inputs(ids: Sequence[str]) -> Iterator[None]:
    token = _current_input_ids.set(list(ids))
    try:
        yield
    finally:
        _current_input_ids.reset(token)


def _decision(result: Any, tool_name: str) -> None:
    """Turn a policy decision into a denial the model can read, or return."""
    if not result.authorized:
        # The model reads the wording of the denial: the reasons and
        # suggestions the engine wrote for the model (the policy author's
        # text). The rule identifiers and analysis verdicts of the allow
        # rules are in ``denial_trace.allow_routes`` for the application.
        reasons = [reason.details.strip() for reason in result.denial_trace.reasons]
        hints = [text.strip() for text in result.suggestions]
        message = f"SASY denied {tool_name}: " + ("; ".join(dict.fromkeys(filter(None, reasons))) or "no allow rule matched")
        if any(hints):
            message += " Suggested: " + "; ".join(dict.fromkeys(filter(None, hints)))
        raise ActionDenied(message)
    if result.transform_ids:
        raise ActionDenied(
            f"The policy authorized {tool_name} only with transform(s) "
            f"{', '.join(result.transform_ids)}, which this adapter cannot apply. "
            "Remove the ApplyTransform rule for tool calls, or make the call over "
            "HTTP routed by sasy.instrument(http=True)."
        )


def _bound(function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments:
    arguments = inspect.signature(function).bind(*args, **kwargs)
    arguments.apply_defaults()
    # A detached object is both checked and dispatched. A caller cannot mutate
    # the original argument container while an async decision is in flight.
    arguments.arguments = copy.deepcopy(arguments.arguments)
    return arguments


def _annotation_parts(annotation: Any) -> Iterator[Any]:
    """An annotation and everything inside it.

    ``Optional[Annotated[int, InjectedState("k")]]`` hides the marker one level
    down, and a container hides a model the same way, so both are looked for on
    every part rather than on the outermost type alone.
    """
    yield annotation
    for part in get_args(annotation):
        yield from _annotation_parts(part)


def _gate(tool: StructuredTool) -> StructuredTool:
    if type(tool) is not StructuredTool or tool.response_format != "content":
        raise ValueError("Only ordinary StructuredTool tools with content results are supported")
    if tool.return_direct:
        # The run ends on such a tool's result, so the run has no final model
        # answer. The adapter records a run's answer as the messages its last
        # model call produced, and that call happened before the tool ran, so
        # the text a delegating tool body reads back would not depend on the
        # tool result it actually came from.
        raise ValueError(
            f"Tool {tool.name} is declared return_direct, which is unsupported: the run "
            "would end on the tool result instead of a model answer, and what a caller "
            "reads back would not depend on that result. Drop return_direct and let the "
            "model answer after the tool."
        )
    schema = tool.get_input_schema()
    if not issubclass(schema, BaseModel):
        raise ValueError("Only Pydantic v2 tool schemas are supported")
    for name, entry in schema.model_fields.items():
        parts = list(_annotation_parts(entry.rebuild_annotation()))
        if any(_is_injected_arg_type(part) for part in parts):
            # Optional[Annotated[..., InjectedState(...)]] reaches the model as
            # an ordinary parameter and fails validation at every call, so the
            # body never runs and no check is ever made. Refusing here says so
            # at setup rather than once per call.
            raise ValueError(f"Injected tool parameter is unsupported: {name}")
        model = next((part for part in parts
                      if isinstance(part, type) and issubclass(part, BaseModel)), None)
        if model is not None:
            # The check sees the arguments as JSON, and a model instance has no
            # JSON form there: the call would stop the run when it is
            # serialized. A policy reading such an argument would need the
            # nested shape anyway, which is a separate decision.
            raise ValueError(
                f"Tool {tool.name} takes {name} as the model {model.__name__}, which is "
                "unsupported: the policy check serializes a call's arguments to JSON and "
                "a model instance has no JSON form. Take the fields as arguments of their "
                "own, or a JSON string the tool parses."
            )
    wrapped = tool.model_copy(deep=True)
    for attribute in ("func", "coroutine"):
        function = getattr(tool, attribute)
        if function is None:
            continue
        signature = inspect.signature(function)
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
            raise ValueError("Tool signatures must declare their arguments explicitly")
        if any(name in signature.parameters for name in ("config", "runtime", "store", "callbacks", "run_manager")):
            raise ValueError("Injected state, runtime, config and callbacks are unsupported")

        def context(name: str) -> tuple[str, ...]:
            _active()
            invocation = _tool_inputs.get()
            if invocation is None or invocation[0] != name:
                raise InstrumentationError("Tool execution lacks its exact dispatch context")
            return invocation[1]

        def sync_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            def checked(*args: Any, **kwargs: Any) -> Any:
                inputs = context(tool.name)
                bound = _bound(fn, args, kwargs)
                invocation = _tool_inputs.get()
                assert invocation is not None
                # Stored before the check, so the tool message's derived_from
                # carries the arguments that were actually authorized and
                # dispatched, after binding and defaults, not the model's raw
                # call.
                _active().dispatches[invocation[2]] = copy.deepcopy(dict(bound.arguments))
                with _inputs(inputs):
                    _decision(reference_monitor.check_tool_call(tool.name, _json(bound.arguments), input_node_ids=list(inputs)), tool.name)
                    # The check can take time; the run must still be allowed to act.
                    _active()
                    return fn(*bound.args, **bound.kwargs)
            return checked

        def async_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            async def checked(*args: Any, **kwargs: Any) -> Any:
                inputs = context(tool.name)
                bound = _bound(fn, args, kwargs)
                invocation = _tool_inputs.get()
                assert invocation is not None
                # Stored before the check, so the tool message's derived_from
                # carries the arguments that were actually authorized and
                # dispatched, after binding and defaults, not the model's raw
                # call.
                _active().dispatches[invocation[2]] = copy.deepcopy(dict(bound.arguments))
                with _inputs(inputs):
                    _decision(await reference_monitor.check_tool_call_async(tool.name, _json(bound.arguments), input_node_ids=list(inputs)), tool.name)
                    # The await can take time; the run must still be allowed to act.
                    _active()
                    return await fn(*bound.args, **bound.kwargs)
            return checked

        gated = async_wrapper(function) if attribute == "coroutine" else sync_wrapper(function)
        _GATED.add(gated)
        setattr(wrapped, attribute, gated)
    return wrapped


def _protected(tool: Any) -> bool:
    """Whether *tool* is one :func:`_gate` produced, so every run is checked."""
    if type(tool) is not StructuredTool:
        return False
    functions = [function for function in (tool.func, tool.coroutine) if function is not None]
    # WeakSet membership is False for an object it cannot reference weakly.
    return bool(functions) and all(function in _GATED for function in functions)


def _validate_model_context(model: Any, settings: Any = None) -> None:
    """Reject provider-side inputs that cannot be resolved into observed messages.

    Provider implementations remain trusted code. For the client this adapter
    is tested against, retained history, remote prompts and raw body overrides
    are outside the supported mode, in which every invocation sends its whole
    conversation; configuration changed after setup is rejected as well.
    """
    hidden_inputs = {
        "previous_response_id", "conversation", "prompt", "input", "messages",
        "instructions", "system", "tools", "extra_body",
    }
    if getattr(model, "use_previous_response_id", False):
        raise InstrumentationError("Provider-retained history is unsupported")
    for name in ("previous_response_id", "conversation", "extra_body"):
        if getattr(model, name, None):
            raise InstrumentationError("Provider-retained context and body overrides are unsupported")
    for parameters in (getattr(model, "model_kwargs", None), settings):
        if parameters and any(name in parameters for name in hidden_inputs):
            raise InstrumentationError("Provider-retained context and input overrides are unsupported")


class _Middleware(AgentMiddleware):
    def _model_inputs(self, request: Any) -> tuple[_Run, Any, list[str]]:
        run = _active()
        _validate_model_context(request.model, request.model_settings)
        # The provider receives the same detached message snapshots we record.
        request = request.override(messages=copy.deepcopy(request.messages), system_message=copy.deepcopy(request.system_message))
        # The framework builds the system message anew for every model call. One
        # identity per run lets an unchanged prompt resolve to the version it
        # already has instead of being registered again at each step.
        if request.system_message is not None and not request.system_message.id:
            request.system_message.id = run.system_identity
        messages = ([request.system_message] if request.system_message is not None else []) + request.messages
        # Marks are the adapter's own bookkeeping. No provider ever receives one.
        _strip(messages)
        # A delegated run is computed from the tool call that started it,
        # whatever its messages are. Restored history and text of unknown
        # origin carry no edge back to the caller on their own, so the dispatch
        # is named as an input of every model call of the run.
        return run, request, list(dict.fromkeys([*run.entry_parents, *run.observe_many(messages)]))

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        run, request, inputs = self._model_inputs(request)
        with _inputs(inputs):
            response = handler(request)
        run.latest_answers = tuple(run.observe(message, inputs) for message in response.result)
        return response

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        run, request, inputs = self._model_inputs(request)
        with _inputs(inputs):
            response = await handler(request)
        run.latest_answers = tuple(run.observe(message, inputs) for message in response.result)
        return response

    @contextmanager
    def _tool(self, request: Any) -> Iterator[tuple[_Run, _ToolScope]]:
        run = _active()
        call = request.tool_call
        origin = run.calls.get(call["id"])
        # A tool that does not exist is not a mismatch: the dispatch is the
        # call the model made, and the framework has nothing to run for it.
        # That case is reported to the model in ``_missing`` below.
        if origin is None or origin[1] != call or (request.tool is not None and request.tool.name != call["name"]):
            raise InstrumentationError("Tool dispatch does not match its observed model output")
        inputs = (origin[0],)
        # A fresh scope per dispatch: sibling calls run in copied contexts, so
        # one call's delegated answers never reach another call's scope.
        scope = _ToolScope(run.session_id, run.depth, inputs, run)
        tokens = (_tool_inputs.set((call["name"], inputs, call["id"])), _tool_scope.set(scope))
        resolver = dependencies.set_resolver(scope)
        try:
            with _inputs(inputs):
                yield run, scope
        finally:
            # Work the body left running keeps a copy of this context; it must
            # not be authorized as part of a computation that has ended.
            scope.closed = True
            dependencies.reset_resolver(resolver)
            _tool_scope.reset(tokens[1])
            _tool_inputs.reset(tokens[0])

    def _result(self, request: Any, run: _Run, scope: _ToolScope, result: Any) -> ToolMessage:
        name = request.tool_call["name"]
        if not isinstance(result, ToolMessage):
            raise InstrumentationError(
                f"Tool {name} returned {type(result).__name__}; tools must return plain "
                "content. A LangGraph Command, which mutates graph state, is not "
                "supported by this adapter."
            )
        if result.tool_call_id != request.tool_call["id"]:
            raise InstrumentationError(
                f"Tool {name} returned a ToolMessage for a different tool_call_id."
            )
        run.observe(result, scope.consumed())
        return result

    def _failed(self, request: Any, text: str) -> ToolMessage:
        """The result of a call that did not run, as the model reads it."""
        call = request.tool_call
        return ToolMessage(content=text, tool_call_id=call["id"], name=call["name"], status="error")

    def _missing(self, request: Any) -> str | None:
        """Why this call cannot run at all, when the named tool does not exist.

        The model naming a tool it was not given is ordinary model behaviour,
        not a broken dispatch. Nothing runs and nothing is checked, so the
        model is told and the run goes on, as it would without the adapter.
        """
        if request.tool is not None:
            return None
        return f"Error: no tool named {request.tool_call['name']} is available."

    def wrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        with self._tool(request) as (run, scope):
            missing = self._missing(request)
            if missing is not None:
                result = self._failed(request, missing)
            else:
                try:
                    result = handler(request)
                except ActionDenied as error:
                    result = self._failed(request, str(error))
            return self._result(request, run, scope, result)

    async def awrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        with self._tool(request) as (run, scope):
            missing = self._missing(request)
            if missing is not None:
                result = self._failed(request, missing)
            else:
                try:
                    result = await handler(request)
                except ActionDenied as error:
                    result = self._failed(request, str(error))
            return self._result(request, run, scope, result)


def _stamp(run: _Run, output: Any) -> None:
    """Mark the returned messages with the node each one is.

    Only messages this run resolved are marked, so a message whose origin the
    run could not establish stays unmarked and is read again as an external
    input if it comes back. Marking happens after the run, and the mark is
    excluded from the auxiliary-field comparison, so it changes nothing observed.
    """
    messages = output.get("messages") if isinstance(output, dict) else None
    for message in messages or []:
        identity = getattr(message, "id", None)
        if not isinstance(message, BaseMessage) or not identity or identity in run.external:
            continue
        node = run.versions.get(identity)
        if node is None:
            continue
        mark: dict[str, Any] = {"version": _MARK_VERSION, "node": node, "origin": run.origin(identity)}
        if isinstance(message, ToolMessage) and identity in run.evidence:
            mark["tool"] = {"name": run.evidence[identity][0], "arguments": run.evidence[identity][1]}
        message.response_metadata[_MARK_KEY] = mark


def _foreign_answer(run: _Run, output: Any) -> str:
    """The recorded text of an answer a run in another session returned.

    The calling body consumed the answer, so its result must depend on it. The
    answer's own nodes live in the other session's graph and cannot be named
    here, so what the body consumed is recorded as an input of unknown origin.
    """
    answers = set(run.latest_answers)
    messages = output.get("messages") if isinstance(output, dict) else None
    return _json({
        "langchain_delegation": {"session": run.session_id},
        "provenance": "unattributed external input",
        "value": [message.content for message in messages or []
                  if isinstance(message, BaseMessage) and message.id and run.versions.get(message.id) in answers],
    })


class SasyAgent:
    """A message-loop agent built by :func:`create_agent`.

    Each invocation is independent: there is no stored conversation to resume,
    so a continued conversation is passed back in as messages.
    """

    def __init__(self, graph: Any):
        self._graph = graph

    @contextmanager
    def _invocation(self, inputs: dict[str, Any]) -> Iterator[tuple[_Run, _ToolScope | None, _ToolScope | None, dict[str, Any]]]:
        require_active_session()
        session = current_wire_session_id()
        if not isinstance(session, str):
            raise InstrumentationError("LangChain execution requires a concrete SASY session")
        caller = self._delegation()
        # Identities are not shared across sessions: a run the caller opened a
        # different session for is independent work, not a delegated one. Its
        # answer still reaches the caller, so it is carried back as an input of
        # unknown origin instead of being dropped.
        scope = caller if caller is not None and caller.session_id == session else None
        if set(inputs) != {"messages"}:
            raise ValueError("Only the messages input is supported")
        messages = copy.deepcopy(convert_to_messages(inputs["messages"]))
        run = _Run(session, depth=caller.depth + 1 if caller else 0, entry_parents=scope.consumed() if scope else (), caller=caller)
        run.admit(messages)
        # The delegating body's dispatch context belongs to the outer run; only
        # this run's own tool dispatches are its execution context. That
        # includes the dependency resolver the shared authorization boundary
        # asks for its inputs: a check made inside this run — the model's own
        # HTTP request, say — must resolve against this run, not against the
        # body that started it. The run's own tool dispatches install theirs.
        tokens = (_run.set(run), _tool_inputs.set(None), _tool_scope.set(None))
        resolver = dependencies.set_resolver(None)
        try:
            yield run, scope, caller if scope is None else None, {"messages": messages}
        finally:
            dependencies.reset_resolver(resolver)
            _tool_scope.reset(tokens[2])
            _tool_inputs.reset(tokens[1])
            _run.reset(tokens[0])

    def _delegation(self) -> _ToolScope | None:
        """The tool body this run was started from, if it was started inside one.

        A guarded agent may be invoked from a guarded tool body of a running
        agent, and nowhere else inside a run: elsewhere there is no dispatch to
        record as the cause of the inner input.
        """
        if _run.get() is None:
            return None
        scope = _tool_scope.get()
        if scope is None:
            raise InstrumentationError("A guarded agent may be invoked only from a guarded tool body of a running "
                "agent; elsewhere inside a run the inner input has no observed cause")
        # A run in another session is a hand-off, so only liveness is required.
        scope.open()
        if scope.depth + 1 > _MAXIMUM_NESTING:
            raise InstrumentationError("Delegation to guarded agents exceeded the supported nesting depth")
        return scope

    def invoke(self, inputs: dict[str, Any], *, recursion_limit: int = 50) -> dict[str, Any]:
        """Run one agent invocation inside the caller's ``sasy.session``.

        Args:
            inputs: ``{"messages": [...]}`` and nothing else. To continue a
                conversation, pass back the message objects a previous
                invocation returned: each carries a mark in
                ``response_metadata["sasy"]`` that restores its recorded
                ancestry. Messages without a verifiable mark are recorded as
                inputs of unknown origin.
            recursion_limit: LangGraph step limit for this invocation.

        Returns:
            The agent state, ``{"messages": [...]}``. Every message whose
            origin this run established carries a mark; a message recorded
            as an input of unknown origin stays unmarked, and is read as an
            unknown input again if it is passed back.

        Raises:
            InstrumentationError: No explicit session, an unsupported feature,
                or a message that cannot be recorded faithfully. The run stops.
            grpc.RpcError: The engine could not be reached; nothing is
                downgraded.

        A denied tool does not raise: the model receives an error
        ``ToolMessage`` and the run continues.
        """
        with self._invocation(inputs) as (run, scope, foreign, prepared):
            output = self._graph.invoke(prepared, {"recursion_limit": recursion_limit})
            run.verify()
            _stamp(run, output)
            if scope is not None:
                scope.record(run.latest_answers)
            elif foreign is not None:
                foreign.defer(_foreign_answer(run, output))
            return output

    async def ainvoke(self, inputs: dict[str, Any], *, recursion_limit: int = 50) -> dict[str, Any]:
        """Async equivalent of :meth:`invoke`; concurrent runs keep separate
        state. Same inputs, result and errors as :meth:`invoke`."""
        with self._invocation(inputs) as (run, scope, foreign, prepared):
            output = await self._graph.ainvoke(prepared, {"recursion_limit": recursion_limit})
            run.verify()
            _stamp(run, output)
            if scope is not None:
                scope.record(run.latest_answers)
            elif foreign is not None:
                foreign.defer(_foreign_answer(run, output))
            return output


def consume_messages(messages: Sequence[BaseMessage]) -> None:
    """Register messages a guarded tool body obtained outside its arguments.

    A body that reads an earlier answer from a variable, a file or a store has
    consumed it without the adapter seeing how. Calling this inside the body
    adds those messages to the body's inputs: verified marks keep their recorded
    ancestry, anything else becomes an input of unknown origin. It only adds
    dependencies, and only to the calling tool body.

    It is not needed for an answer a guarded sub-agent returned to the same
    tool body; that is tracked already.

    Raises:
        InstrumentationError: Called outside a guarded tool body, or after that
            body returned.
    """
    scope = _tool_scope.get()
    if scope is None:
        raise InstrumentationError("consume_messages must be called inside a guarded tool body")
    scope.check()
    copies = copy.deepcopy(convert_to_messages(list(messages)))
    scope.run.admit(copies)
    scope.record(scope.run.observe_many(copies))


def create_agent(model: Any, tools: Sequence[StructuredTool | Callable[..., Any]], *,
                 system_prompt: str | SystemMessage | None = None, name: str | None = None) -> SasyAgent:
    """Build a LangChain agent whose messages are recorded and whose tools are
    authorized by SASY immediately before they run.

    The instrumented ``langchain.agents.create_agent`` uses this factory
    internally and returns its native compiled graph with guarded entry points.
    Calling this function directly works without ``sasy.instrument()`` and returns a
    :class:`SasyAgent` with ``invoke``/``ainvoke`` only; it is deliberately not
    a LangGraph graph, so middleware, checkpointers, stores, callbacks and
    streaming cannot be attached.

    Args:
        model: A ``BaseChatModel`` instance, or a model name such as
            ``"openai:gpt-5"``, which is resolved as LangChain's
            ``create_agent`` resolves it, with
            ``langchain.chat_models.init_chat_model``. A model configured with
            provider-retained history, ``extra_body`` or input overrides is
            rejected, because the provider would then read inputs the adapter
            cannot record.
        tools: Plain functions or ``StructuredTool`` instances with unique
            names. Each must declare its arguments explicitly (no ``*args`` or
            ``**kwargs``, and no injected config, runtime, store or state) and
            return content.
        system_prompt: Optional system prompt, a string or a
            ``SystemMessage``, recorded as an input of every model call. A
            string is the ``SystemMessage`` with that content, as in LangChain.
            A ``SystemMessage`` is copied when the agent is built, so like a
            string it is fixed from then on. A string and a ``SystemMessage``
            with the same content and no other fields are recorded
            identically; any other field of the message is recorded as given.
        name: The name of the LangGraph graph LangChain compiles, as in
            LangChain's ``create_agent``. LangChain also sets it as the name of
            every message the model produces, so it is the agent those
            messages are recorded under (in place of ``"langchain"``) and
            policies that match on agent names see it.

    Returns:
        SasyAgent: call it inside ``with sasy.session(policy=...):``.

    Raises:
        RuntimeError: An installed langchain or langgraph package is not the
            exact version this adapter supports; install ``sasy[langchain]``.
        ValueError: Unsupported model or tool shape; the message names it.

    HTTP calls made by the model client are mediated separately, by
    ``sasy.instrument(http=True)``; HTTP routing is off by default.
    """
    check_supported()
    from langchain_core.language_models import BaseChatModel
    if isinstance(model, str):
        # What LangChain's create_agent does with a model name. Looked up at
        # call time, so the resolver is LangChain's current one.
        import langchain.chat_models
        model = langchain.chat_models.init_chat_model(model)
    if not isinstance(model, BaseChatModel):
        raise ValueError("Supply a concrete BaseChatModel instance or a model name such as 'openai:gpt-5'")
    if system_prompt is not None and not isinstance(system_prompt, (str, SystemMessage)):
        raise ValueError("system_prompt must be a string or a SystemMessage")
    if isinstance(system_prompt, SystemMessage):
        # LangChain sends this very object on every model call; a copy keeps a
        # later change to the caller's message out of the agent, as for a string.
        system_prompt = copy.deepcopy(system_prompt)
    _validate_model_context(model)
    protected = []
    names = set()
    for item in tools:
        if isinstance(item, StructuredTool):
            selected = item
        elif callable(item):
            selected = StructuredTool.from_function(coroutine=item) if inspect.iscoroutinefunction(item) else StructuredTool.from_function(item)
        else:
            raise ValueError("Tools must be ordinary functions or StructuredTool instances")
        if selected.name in names:
            raise ValueError("Tool names must be unique")
        names.add(selected.name)
        protected.append(_gate(selected))
    graph = _create_agent(model, tools=protected, system_prompt=system_prompt, middleware=[_Middleware()], name=name)
    _mark_tool_nodes(graph, expected=bool(protected))
    return SasyAgent(graph)


# The ToolNodes SASY's own agents run their tools through. Inside a session the
# backstop refuses every call of any other ToolNode; these are left to their
# tool-call wrapper, SASY's middleware, which checks each call and reports a
# tool name the node does not know.
_sasy_tool_nodes: WeakSet[Any] = WeakSet()


def _mark_tool_nodes(graph: Any, *, expected: bool) -> None:
    """Record the ``ToolNode`` that LangChain's ``create_agent`` built for a
    SASY agent. LangChain adds it to the compiled graph as a node whose
    runnable (``PregelNode.bound``) is the ``ToolNode`` itself."""
    from langgraph.prebuilt.tool_node import ToolNode

    nodes = [getattr(node, "bound", None) for node in getattr(graph, "nodes", {}).values()]
    found = [node for node in nodes if isinstance(node, ToolNode)]
    if expected and not found:
        raise InstrumentationError("LangChain built the agent without a ToolNode for its tools")
    _sasy_tool_nodes.update(found)


def check_supported() -> None:
    """Raise unless every LangChain package the adapter relies on is installed
    at the exact version it supports."""
    for package, expected in _SUPPORTED.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            found = f"{package} {installed} is installed" if installed else f"{package} is not installed"
            raise RuntimeError(
                f"{found}; this adapter supports exactly {expected}, because it "
                "depends on that version's internals. Run: pip install 'sasy[langchain]'"
            )


# The parameters of LangChain's create_agent other than the four SASY's takes,
# with their defaults. Passing one at its default changes nothing; any other
# value is a feature the adapter cannot record.
_LANGCHAIN_OPTIONS = {
    name: parameter.default
    for name, parameter in inspect.signature(_create_agent).parameters.items()
    if name not in ("model", "tools", "system_prompt", "name")
}


def _unset(value: Any, default: Any) -> bool:
    if value is default:
        return True
    # An empty list says the same as the empty tuple LangChain defaults to.
    return type(default) is tuple and not default and type(value) in (list, tuple) and not value


def _langchain_create_agent(model: Any, tools: Sequence[Any] | None = None, *,
                            system_prompt: str | SystemMessage | None = None, name: str | None = None,
                            **options: Any) -> Any:
    """``langchain.agents.create_agent``, as installed by :func:`instrument_langchain`.

    Returns a native compiled graph. Outside a session it has native behavior;
    inside a session invoke/ainvoke and batch/abatch qualify the factory inputs
    and run with SASY observation and authorization. Only ``model`` (an instance or a model name),
    ``tools``, ``system_prompt`` (a string or a ``SystemMessage``) and
    ``name`` are supported. Direct streaming and custom graph configuration
    are refused until their dependency semantics are supported.

    Raises:
        ValueError: Any other ``create_agent`` argument (``middleware``,
            ``checkpointer``, ``response_format``, ...), which the adapter
            cannot record; the message names it. Also every error
            :func:`create_agent` raises for an unsupported model or tool.
        TypeError: An argument LangChain's ``create_agent`` does not take.
    """
    from .langchain_native import register_lazy

    # Capture the construction inputs once: mutating the caller's prompt or
    # tool list later must not change the graph when it enters a session.
    tools = tuple(tools) if tools is not None else ()
    system_prompt = copy.deepcopy(system_prompt)
    options = {key: copy.copy(value) if isinstance(value, (list, dict, set)) else value
               for key, value in options.items()}
    if isinstance(model, str):
        import langchain.chat_models
        model = langchain.chat_models.init_chat_model(model)
    protected_graph = None
    construction_lock = RLock()

    def protected():
        nonlocal protected_graph
        with construction_lock:
            if protected_graph is not None:
                return protected_graph
            for option, value in options.items():
                if option not in _LANGCHAIN_OPTIONS:
                    raise TypeError(f"create_agent() got an unexpected keyword argument {option!r}")
                if not _unset(value, _LANGCHAIN_OPTIONS[option]):
                    raise ValueError(
                        f"create_agent(..., {option}=...) is not supported under SASY: the LangChain "
                        f"adapter cannot record what {option} does, so the agent could act unchecked. "
                        "Only model, tools, system_prompt and name are supported; see the LangChain "
                        f"integration guide: {_GUIDE}"
                    )
            protected_graph = create_agent(model, tools, system_prompt=system_prompt, name=name)._graph
            return protected_graph

    if is_session_active():
        protected()  # Fail unsupported construction promptly within a protected scope.
    graph = _create_agent(model, tools=tools, system_prompt=system_prompt, name=name, **options)
    return register_lazy(graph, protected)



_langchain_create_agent.__name__ = _langchain_create_agent.__qualname__ = "create_agent"


def _refuse_unprotected(tool: Any) -> None:
    """Stop a tool SASY does not protect from running inside a SASY session.

    The second line of defence: every tool a LangGraph ``ToolNode`` is about to
    execute passes here, including one a middleware substituted. Any
    explicitly declared session counts, ``sasy.global_session`` included: a
    tenant-wide policy is bound there too. Outside every session nothing is
    checked.
    """
    if tool is None or not is_session_active() or _protected(tool):
        return
    raise InstrumentationError(
        f"Tool {getattr(tool, 'name', tool)!r} is not protected by SASY, so it cannot run "
        "inside a SASY session. Call sasy.instrument() before building the agent, or "
        "build it with sasy.instrumentation.langchain.create_agent."
    )


def _refuse_foreign_node(node: Any, call: Any) -> None:
    """Stop every call of a ``ToolNode`` that SASY did not build, inside a
    SASY session, before any tool-call wrapper or tool body runs.

    Such a node's tool-call wrapper is arbitrary code that could run any tool
    and return its result without reaching a guarded tool, so no call on it is
    accepted: known name, unknown name, wrapper or not. SASY's own nodes are
    left to SASY's middleware, which checks each call (and reports a name the
    model invented). Outside every session nothing is checked.
    """
    if not is_session_active() or node in _sasy_tool_nodes:
        return
    name = call.get("name") if isinstance(call, dict) else None
    raise InstrumentationError(
        f"Tool call {name!r} comes from a LangGraph ToolNode that SASY did not build. "
        "Inside a SASY session, tools run only in agents SASY builds: call "
        "sasy.instrument() before building the agent with langchain.agents.create_agent, "
        "or use sasy.instrumentation.langchain.create_agent. Custom LangGraph graphs are "
        "not supported inside a session yet."
    )


_patch_lock = RLock()
_installed = False


def _patch_tool_node() -> None:
    """Refuse, inside a SASY session, every tool call of a ``ToolNode`` that
    SASY did not build, at LangGraph's per-call ``ToolNode`` methods.

    ``_run_one``/``_arun_one`` refuse the call before the original method, and
    so outside its handling of tool errors and before any tool-call wrapper: a
    refusal is never turned into a tool message, and no wrapper or tool body
    runs. (The node has already parsed its input and read graph state by then;
    neither runs a tool.) As a second
    line of defence, the ``_execute_tool_*`` methods refuse a tool SASY does
    not protect when a node is about to execute it.
    """
    from langgraph.prebuilt.tool_node import ToolNode

    names = ("_run_one", "_arun_one", "_execute_tool_sync", "_execute_tool_async")
    if any(not callable(getattr(ToolNode, name, None)) for name in names):
        raise InstrumentationError("Required LangGraph ToolNode hook is missing")
    run_one, arun_one = ToolNode._run_one, ToolNode._arun_one
    execute, aexecute = ToolNode._execute_tool_sync, ToolNode._execute_tool_async

    @wraps(run_one)
    def _run_one(self: Any, call: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_foreign_node(self, call)
        return run_one(self, call, *args, **kwargs)

    @wraps(arun_one)
    async def _arun_one(self: Any, call: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_foreign_node(self, call)
        return await arun_one(self, call, *args, **kwargs)

    @wraps(execute)
    def _execute_tool_sync(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_unprotected(getattr(request, "tool", None))
        return execute(self, request, *args, **kwargs)

    @wraps(aexecute)
    async def _execute_tool_async(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_unprotected(getattr(request, "tool", None))
        return await aexecute(self, request, *args, **kwargs)

    ToolNode._run_one = _run_one  # type: ignore[method-assign]
    ToolNode._arun_one = _arun_one  # type: ignore[method-assign]
    ToolNode._execute_tool_sync = _execute_tool_sync  # type: ignore[method-assign]
    ToolNode._execute_tool_async = _execute_tool_async  # type: ignore[method-assign]


def _rebind() -> None:
    """Point module-level names bound to LangChain's original ``create_agent``
    at the SASY version, so an import made before :func:`instrument_langchain`
    gets it too. Nothing but a module global that is exactly that function is
    touched, and this module keeps the original it builds with."""
    _rebind_module_globals(_create_agent, _langchain_create_agent, keep=(__name__,))


def instrument_langchain() -> None:
    """Install the LangChain adapter, as ``sasy.instrument()`` does when
    LangChain or LangGraph is installed (or with ``langchain=True``).

    ``langchain.agents.create_agent`` (and ``langchain.agents.factory``'s) then
    builds a native compiled graph through :func:`create_agent`, and so does any module that
    had already imported it. As a backstop, inside a ``sasy.session`` or
    ``sasy.global_session``, a LangGraph ``ToolNode`` that SASY did not build
    refuses every tool call before any wrapper or tool runs: an agent built another way
    raises :class:`InstrumentationError` at its first tool call instead of
    acting unchecked. A tool that graph code calls directly, not through a
    ``ToolNode``, is not checked.

    The patches cannot be removed; calling this again is harmless.

    Raises:
        RuntimeError: A LangChain package is not the exact version supported.
        InstrumentationError: A LangGraph hook the backstop needs is missing.
    """
    global _installed
    check_supported()
    import langchain.agents
    import langchain.agents.factory

    with _patch_lock:
        if not _installed:
            _patch_tool_node()
            _installed = True
        langchain.agents.create_agent = _langchain_create_agent
        langchain.agents.factory.create_agent = _langchain_create_agent
        _rebind()
