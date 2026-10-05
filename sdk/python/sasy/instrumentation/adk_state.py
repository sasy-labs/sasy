"""Versioned ADK template/output-key state and immutable text artifact transport.

JSON context reads and explicit assignments carry computation-local dependencies.
Callable instruction providers read through the same tracked mapping.
Mutable container aliases remain outside this profile. Graph identities are never reused across sessions.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import re
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import Any, NoReturn

from sasy.observability._snapshots import VERSION_ID_PREFIX

from .session import get_current_entity, is_session_active

# Values carried over from another SASY session are labelled by origin kind only.
# The producing session's identifier stays out of this graph: it may belong to a
# different tenant or entity, and naming it would carry that identity across.
EXTERNAL_SESSION_ORIGIN = "another SASY session"


def _adk():
    from . import adk
    return adk


def _fail(message) -> NoReturn:
    raise _adk().AdkInstrumentationError(message)


def _value(value):
    if value is None or type(value) in (str, bool, int, float):
        _adk()._json(value)
        return copy.deepcopy(value)
    if type(value) is list:
        return [_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _value(item) for key, item in value.items()}
    _fail("ADK observed state supports JSON values only")


@dataclass
class Version:
    """The last observed value of one state key or one artifact version.

    ``fingerprint`` is its canonical record, which keeps apart values a plain
    JSON dump writes alike, ``value`` is the value that record was taken over,
    ``nodes`` maps each SASY session that has read or written it to the
    immutable version IDs representing it there, ``producer_graph`` is the
    session that produced it (``None`` when the origin is unknown) and
    ``snapshots`` holds the event recorded under each of those IDs.

    The value is carried alongside because the fingerprint is not the value's
    own JSON: a caller that needs the observed value back reads ``value``.
    """

    fingerprint: str
    nodes: dict[str, list[str]] = field(default_factory=dict)
    producer_graph: str | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)
    value: Any = None


def _foreign(version, session):
    return version is not None and version.producer_graph not in (None, session)


def _session_snapshots(version, session):
    """The recorded events of this graph's own nodes for that value."""
    return {node: version.snapshots[node] for node in version.nodes.get(session, []) if node in version.snapshots}


@dataclass
class Ledger:
    """What one service instance holds: a session service's state keys, or an
    artifact service's artifact versions.

    ``versions`` maps each key to its last observed :class:`Version`.
    ``owner`` is the single running turn allowed to use the ledger; a second
    concurrent owner is refused.
    """

    versions: dict[tuple, Version] = field(default_factory=dict)
    owner: object | None = None


_ledgers: dict[int, tuple[Any, Ledger]] = {}
_lock = RLock()
_frame: ContextVar[Any] = ContextVar("sasy_adk_resources", default=None)
_installed = False


def _ledger(service):
    identity = id(service)
    with _lock:
        entry = _ledgers.get(identity)
        if entry is None or entry[0]() is not service:
            entry = (weakref.ref(service, lambda _: _ledgers.pop(identity, None)), Ledger())
            _ledgers[identity] = entry
        return entry[1]


@dataclass
class Frame:
    """One running computation: an agent step, a tool call, a callback or an
    instruction provider.

    It accumulates the reads that computation has made (``reads``, ``origin``)
    and remembers the task and entity it must stay in, so work that leaves
    them is refused rather than authorized against the wrong inputs.
    """

    resources: Any
    invocation: Any
    kind: str = "agent"
    reads: list[str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)
    live: bool = True
    task: Any = field(default_factory=asyncio.current_task)
    entity: Any = field(default_factory=get_current_entity)
    context: Any = None
    call_id: str | None = None
    artifact_writes: list[Any] = field(default_factory=list)


class Resources:
    """State and artifact tracking for one ``run_async`` turn.

    It is closed when the turn ends, after which any further use of the state
    or artifacts it tracked fails. It holds the ledgers this turn has claimed,
    the reads each callback and prompt made, the renderings and state deltas
    observed, the receipts staged by context writes, the versions this turn
    produced, and the ids of the events the session service has already stored.
    """

    def __init__(self, state, runner, user_id, session_id):
        self.state = state
        self.closed = False
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.workflow_resources_guarded = False
        self.claims: list[Ledger] = []
        self.callback_dependencies: dict[str, list[str]] = {}
        self.prompt_ids: dict[str, list[str]] = {}
        self.renderings: dict[tuple, tuple[Any, str, list[str]]] = {}
        self.deltas: dict[str, dict] = {}
        self.appended: set[str] = set()
        self.readonly_import = False
        self.imported_state: dict[str, Any] = {}
        self.context_receipts: dict[tuple, Any] = {}
        self.context_writers: dict[tuple, Any] = {}
        self.pending_receipts: dict[str, list[Any]] = {}
        self.pending_output: dict[str, list[tuple[Ledger, tuple, Version]]] = {}
        self.produced: dict[tuple, Version] = {}

    def check(self):
        frame = _frame.get()
        if self.closed or not self.state.active or frame is None or not frame.live or frame.resources is not self:
            _fail("ADK resource operation outlived its observed consumer")
        if frame.task is not asyncio.current_task():
            _fail("ADK resource access from a child task requires explicit dependency propagation")
        if frame.entity != get_current_entity():
            _fail("ADK resource operation changed its entity")
        if _adk().current_wire_session_id() != self.state.session:
            _fail("This ADK state or artifact operation is running in a different SASY "
                  "session than the turn that started it.")

    def claim(self, service, *, artifact=False):
        self.check()
        if self.workflow_resources_guarded:
            from .adk_workflow import check_resource_access
            check_resource_access(self, artifact=artifact)
        from google.adk.agents import ParallelAgent
        frame = _frame.get()
        parent = frame.invocation.agent.parent_agent if frame else None
        while parent is not None:
            if isinstance(parent, ParallelAgent):
                _fail("State/artifact operations in parallel ADK agents require additional concurrency qualification")
            parent = parent.parent_agent
        ledger = _ledger(service)
        with _lock:
            if ledger.owner is not None and ledger.owner is not self:
                _fail("Concurrent ADK state/artifact consumers sharing a service are unsupported")
            ledger.owner = self
            if not any(held is ledger for held in self.claims):
                self.claims.append(ledger)
        return ledger

    def close(self):
        for receipt in self.context_receipts.values():
            receipt.rollback()
        self.closed = True
        with _lock:
            for ledger in self.claims:
                if ledger.owner is self:
                    ledger.owner = None
            self.claims.clear()

    def state_key(self, key, invocation):
        app = self.runner.app_name
        if key.startswith("app:"):
            return ("state", app, "app", key[4:])
        if key.startswith("user:"):
            return ("state", app, "user", self.user_id, key[5:])
        if key.startswith("temp:"):
            return ("state", app, "temp", self.user_id, self.session_id, invocation.invocation_id, key[5:])
        return ("state", app, "session", self.user_id, self.session_id, key)

    async def snapshot(self, ledger, key, value, *, inputs=None, agent="adk.resource", pending_event=None):
        from google.genai import types
        sdk = _adk()
        self.check()
        fingerprint = sdk._record_json(value)
        version = ledger.versions.get(key)
        foreign = _foreign(version, self.state.session)
        if inputs is None:
            if version is not None and version.fingerprint != fingerprint:
                if not foreign:
                    _fail("ADK state/artifact changed without an observed producer")
                # A value another session changed is simply a new external input here.
                version = Version(fingerprint, producer_graph=version.producer_graph, value=value)
            if version is None:
                version = Version(fingerprint, value=value)
            if self.state.session in version.nodes:
                self.state.snapshots.update(_session_snapshots(version, self.state.session))
                return version.nodes[self.state.session]
            provenance = "unattributed external input"
            parents = []
        else:
            version = Version(fingerprint, producer_graph=self.state.session, value=value)
            provenance = "observed production"
            parents = list(dict.fromkeys(inputs))
            foreign = False
        payload = {"adk_resource": list(key), "provenance": provenance, "value": value}
        if foreign:
            payload["origin"] = EXTERNAL_SESSION_ORIGIN
        content = types.Content(role="user", parts=[types.Part(text=sdk._json(payload))])
        ids = await self.state.record(content, agent, parents,
            role=sdk.Role.AGENT if inputs is not None else sdk.Role.USER)
        self.check()
        version.nodes[self.state.session] = ids
        version.snapshots.update({node: self.state.snapshots[node] for node in ids})
        if pending_event is None:
            ledger.versions[key] = version
        else:
            self.pending_output.setdefault(pending_event, []).append((ledger, key, version))
        return ids

    def persisted(self, event):
        from .adk_context import stage_receipts
        if not event.partial:
            self.appended.add(event.id)
        stage_receipts(self, event)
        for receipt in self.pending_receipts.pop(event.id, []):
            receipt.persisted = True
        for ledger, key, version in self.pending_output.pop(event.id, []):
            ledger.versions[key] = version
            self.produced[key] = version

    async def read_state(self, invocation, key, captured):
        writer = self.context_writers.get(self.state_key(key, invocation))
        if writer is not None and not writer.persisted and writer.computation.frame is not _frame.get():
            _fail("State read raced an uncommitted producer")
        ledger = self.claim(self.runner.session_service)
        value = {"present": key in captured}
        if key in captured:
            value["value"] = _value(captured[key])
        from .adk_context import apply_state_import
        apply_state_import(self, ledger, key, invocation, value)
        return await self.snapshot(ledger, self.state_key(key, invocation), value)

    async def output(self, agent, invocation, event, accumulator, producers):
        from .adk_context import context_state, validate_deltas
        if event.get_function_responses() and validate_deltas(event.actions, event):
            return accumulator, producers
        if not agent.output_key or event.author != agent.name:
            return accumulator, producers
        # Callbacks on this model step write through the same event actions.
        written = context_state(event.actions)
        expected = event.model_copy(deep=True)
        expected.actions.state_delta = {}
        agent._LlmAgent__maybe_save_output_to_state(expected)
        previous = accumulator
        accumulator = agent._LlmAgent__maybe_accumulate_streaming_output(expected, accumulator)
        if event.actions.state_delta != {**written, **expected.actions.state_delta}:
            _fail("Only observed output_key writes may change ADK state")
        if not expected.actions.state_delta:
            return accumulator, producers
        record = self.state.records.get(event.id)
        if record is None:
            _fail("ADK output_key has no observed model production")
        if accumulator != previous:
            producers = list(dict.fromkeys([*producers, *record.ids]))
        else:
            producers = list(record.ids)
        ledger = self.claim(self.runner.session_service)
        value = _value(expected.actions.state_delta[agent.output_key])
        state_key = self.state_key(agent.output_key, invocation)
        await self.snapshot(ledger, state_key,
            {"present": True, "value": value}, inputs=producers, agent=agent.name, pending_event=event.id)
        if agent.output_key not in invocation.session.state:
            await self.snapshot(ledger, ("state-presence", *state_key[1:]),
                {"present": True}, inputs=producers, agent=agent.name, pending_event=event.id)
        self.deltas[event.id] = {**written, **expected.actions.state_delta}
        return accumulator, producers


def new_run(state, runner, user_id, session_id):
    return Resources(state, runner, user_id, session_id)


def prompt_inputs(state, agent):
    frame = _frame.get()
    if frame is not None and frame.resources is state.resources and frame.invocation.agent.name == agent:
        frame.resources.check()
        return list(frame.reads)
    return state.resources.prompt_ids.get(agent, []) if state.resources else []


def register_rendering(state, agent, content, ids):
    from .adk_agents import forget_input
    forget_input(state, agent, content)
    # Keyed by id() for the reason given in adk_agents; the entry keeps the
    # object and its content key so both are re-checked on lookup.
    state.resources.renderings[(agent, id(content))] = (content, _adk()._content_key(content), list(ids))


def rendered_inputs(state, agent, index, content):
    if state.resources is None:
        return None
    rendering = state.resources.renderings.get((agent, id(content)))
    key = _adk()._content_key(content)
    if rendering is not None:
        original, expected, ids = rendering
        if original is not content or key != expected:
            _fail("Observed resource presentation changed without explicit derivation")
        return ids
    if any(a == agent and entry[1] == key for (a, _), entry in state.resources.renderings.items()):
        _fail("Copied resource presentation has ambiguous observation provenance")
    return None


def copy_renderings(state, agent, original, copied):
    # This SDK-owned deep copy preserves the exact occurrence mapping. User
    # callback filtering/reordering retains objects; arbitrary clones do not.
    for index, (source, target) in enumerate(zip(original.contents, copied.contents, strict=True)):
        ids = rendered_inputs(state, agent, index, source)
        if ids is not None:
            register_rendering(state, agent, target, ids)


def tool_inputs(origin):
    frame = _frame.get()
    return list(dict.fromkeys([*origin, *frame.reads])) if frame and frame.kind == "tool" else list(origin)


def stored_delta(event):
    """Whether the session service has already stored this event.

    ADK strips ``temp:`` keys from an event's ``state_delta`` when it appends
    it, so a delta read back after that point no longer names the temp keys
    the writing context recorded.
    """
    state = _adk()._active.get()
    resources = state.resources if state is not None else None
    return event is not None and resources is not None and event.id in resources.appended


def persistent_keys(values):
    """The part of a state delta ADK stores: temp keys never reach storage."""
    return {key: value for key, value in values.items() if not key.startswith("temp:")}


def validate_actions(actions, event=None):
    sdk = _adk()
    state = sdk._active.get()
    resources = state.resources if state is not None else None
    if actions and any(getattr(actions, name, None) for name in ("requested_auth_configs", "requested_tool_confirmations",
            "render_ui_widgets", "rewind_before_invocation_id", "agent_state", "escalate")):
        _fail("Unimplemented credential or control actions are unsupported")
    if actions and actions.route is not None:
        from . import adk_workflow
        adk_workflow.validate_route_action(state, actions, event)
    if actions and actions.compaction:
        _fail("Compaction requires additional instrumentation")
    from .adk_context import validate_deltas
    context_delta = bool(actions and validate_deltas(actions, event))
    if actions and actions.state_delta and not context_delta:
        recorded = resources.deltas.get(event.id) if resources and event is not None else None
        if recorded is not None and stored_delta(event):
            recorded = persistent_keys(recorded)
        if recorded is None or sdk._record_json(recorded) != sdk._record_json(actions.state_delta):
            _fail("Only observed output_key writes may change ADK state")
    if actions and actions.artifact_delta and not context_delta:
        _fail("Context artifact writes require additional instrumentation")


def _add_reads(ids):
    frame = _frame.get()
    if frame is None:
        _fail("ADK resource read lacks a qualified consumer")
    frame.resources.check()
    frame.reads[:] = list(dict.fromkeys([*frame.reads, *ids]))
    if frame.kind in ("tool", "callback"):
        sdk = _adk()
        sdk._current_input_ids.set(list(dict.fromkeys([*frame.origin, *frame.reads])))


class _BlockedState:
    """Retain ADK's schema/delta checks without exposing untracked values."""
    def __init__(self, state):
        self._schema = state._schema
        self._has_delta = state.has_delta()

    def has_delta(self):
        return self._has_delta

    _ADVICE = ("Read and write state through the context's `state` mapping, with "
               "string keys and JSON values.")

    def __getattr__(self, name):
        _fail(f"state.{name} is not supported in an instrumented run. {self._ADVICE}")

    def __getitem__(self, key):
        _fail(f"Reading state[{key!r}] here is not supported in an instrumented run. "
              f"{self._ADVICE}")

    def __setitem__(self, key, value):
        _fail(f"Writing state[{key!r}] here is not supported in an instrumented run. "
              f"{self._ADVICE}")

    def __iter__(self):
        _fail(f"Iterating over state is not supported in an instrumented run. {self._ADVICE}")

    def __contains__(self, key):
        _fail(f"Testing state for {key!r} is not supported in an instrumented run. "
              f"{self._ADVICE}")


async def _provided(agent, context, provider, original):
    """Resolve a callable instruction provider over tracked state only."""
    from .adk_callbacks import in_callback
    from .adk_context import ProviderContext
    if (not is_session_active() or _adk()._active.get() is None) or isinstance(provider, str):
        return await original(agent, context)
    if in_callback():
        _fail("Instruction providers in callbacks require additional instrumentation")
    frame = _frame.get()
    if frame is None or frame.invocation is not context._invocation_context:
        _fail("Instruction provider lacks its observed invocation")
    instruction = provider(ProviderContext(frame))
    if inspect.isawaitable(instruction):
        instruction = await instruction
    if not isinstance(instruction, str):
        _fail("Instruction providers must return text")
    await flush_reads_async()
    return instruction, True


def install():
    global _installed
    if _installed:
        return
    from google.adk.agents.context import Context
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.flows.llm_flows import _tool_caller, instructions
    from google.adk.utils import instructions_utils

    from .adk_callbacks import in_callback

    for cls, name in ((ReadonlyContext, "state"), (ReadonlyContext, "session"), (Context, "state"), (Context, "session")):
        descriptor = getattr(cls, name)
        def getter(self, descriptor=descriptor, cls=cls, name=name):
            if (is_session_active() and _adk()._active.get() is not None):
                if cls is Context and name == "state":
                    raw = descriptor.__get__(self, type(self))
                    if _frame.get() is not None:
                        from .adk_context import state_view
                        return state_view(self, raw)
                    return _BlockedState(raw)
                if in_callback() or _frame.get() is not None:
                    _fail("Arbitrary ADK context state/session access requires additional instrumentation")
            return descriptor.__get__(self, type(self))
        setattr(cls, name, property(getter))

    run_agent = LlmAgent._run_async_impl
    @wraps(run_agent)
    async def observed_agent(self, ctx):
        if not is_session_active():
            iterator = run_agent(self, ctx)
            try:
                async for value in iterator:
                    yield value
            finally:
                await iterator.aclose()
            return
        sdk = _adk()
        state = sdk._state()
        frame = Frame(state.resources, ctx)
        iterator = run_agent(self, ctx)
        accumulator = ""
        producers: list[str] = []
        try:
            while True:
                token = _frame.set(frame)
                resolver_token = enter_resolver(frame)
                try:
                    event = await anext(iterator)
                    await flush_reads_async()
                    accumulator, producers = await state.resources.output(self, ctx, event, accumulator, producers)
                except StopAsyncIteration:
                    break
                finally:
                    leave_resolver(resolver_token)
                    _frame.reset(token)
                yield event
        finally:
            token = _frame.set(frame)
            resolver_token = enter_resolver(frame)
            try:
                await iterator.aclose()
            finally:
                frame.live = False
                leave_resolver(resolver_token)
                _frame.reset(token)
    setattr(LlmAgent, "_run_async_impl", observed_agent)

    inject = instructions_utils.inject_session_state
    @wraps(inject)
    async def render(template, readonly_context, use_jinja2=False):
        sdk = _adk()
        if (not is_session_active() or sdk._active.get() is None):
            return await inject(template, readonly_context, use_jinja2=use_jinja2)
        if in_callback():
            _fail("State/artifact rendering in callbacks requires additional instrumentation")
        if use_jinja2:
            _fail("Jinja state providers require additional instrumentation")
        frame = _frame.get()
        if frame is None:
            _fail("Instruction rendering lacks its observed invocation")
        invocation = readonly_context._invocation_context
        captured = {}
        keys = []
        for match in re.finditer(instructions_utils._TEMPLATE_VAR_PATTERN, template):
            key = match.group().lstrip("{").rstrip("}").strip().removesuffix("?")
            if key.startswith("artifact."):
                continue
            if instructions_utils._is_valid_state_name(key) and key not in keys:
                keys.append(key)
                if key in invocation.session.state:
                    captured[key] = _value(invocation.session.state[key])
        for key in keys:
            _add_reads(await frame.resources.read_state(invocation, key, captured))
        frozen = invocation.model_copy(update={"session": invocation.session.model_copy(update={"state": captured})})
        return await inject(template, ReadonlyContext(frozen), use_jinja2=False)
    instructions_utils.inject_session_state = render

    build = instructions._build_instructions
    @wraps(build)
    async def build_instructions(invocation_context, llm_request):
        if not is_session_active():
            return await build(invocation_context, llm_request)
        from .adk_artifacts import validate_service
        validate_service(invocation_context.artifact_service)
        frame = _frame.get()
        frame.reads.clear()
        frame.resources.callback_dependencies.pop(invocation_context.agent.name, None)
        frame.resources.renderings = {key: value for key, value in frame.resources.renderings.items()
                                     if key[0] != invocation_context.agent.name}
        await build(invocation_context, llm_request)
        frame.resources.prompt_ids[invocation_context.agent.name] = list(frame.reads)
    instructions._build_instructions = build_instructions

    for name in ("canonical_instruction", "canonical_global_instruction"):
        original = getattr(LlmAgent, name)
        field = name.removeprefix("canonical_")
        @wraps(original)
        async def canonical(self, ctx, _original=original, _field=field):
            return await _provided(self, ctx, getattr(self, _field), _original)
        setattr(LlmAgent, name, canonical)

    execute = _tool_caller._execute_single_prepared_call
    @wraps(execute)
    async def tool_scope(invocation_context, prepared_call, agent, *, tool_runner):
        if not is_session_active():
            return await execute(invocation_context, prepared_call, agent, tool_runner=tool_runner)
        sdk = _adk()
        state, _, origin = sdk._origin(prepared_call.tool_context)
        frame = Frame(state.resources, invocation_context, "tool", origin=list(origin.ids),
            context=prepared_call.tool_context, call_id=prepared_call.tool_context.function_call_id)
        token = _frame.set(frame)
        resolver_token = enter_resolver(frame)
        input_token = sdk._current_input_ids.set(list(origin.ids))
        try:
            event = await execute(invocation_context, prepared_call, agent, tool_runner=tool_runner)
            await flush_reads_async()
            if event is not None:
                from .adk_context import stage_event
                stage_event(state.resources, event)
            result_key = (agent.name, prepared_call.tool_context.function_call_id)
            if event is not None and event.content and frame.reads and result_key in state.results:
                state.results[result_key] = await state.record(event.content, agent.name,
                    list(dict.fromkeys([*state.results[result_key], *frame.reads])))
            return event
        finally:
            frame.live = False
            sdk._current_input_ids.reset(input_token)
            leave_resolver(resolver_token)
            _frame.reset(token)
    _tool_caller._execute_single_prepared_call = tool_scope

    from . import adk_context
    adk_context.install()
    _installed = True


def enter_resolver(frame):
    from .adk_context import computation
    from .dependencies import set_resolver
    return set_resolver(computation(frame))


def leave_resolver(token):
    from .dependencies import reset_resolver
    reset_resolver(token)


def flush_reads():
    from .adk_context import computation
    frame = _frame.get()
    if frame is not None:
        computation(frame).flush()


async def flush_reads_async():
    from .adk_context import computation
    frame = _frame.get()
    if frame is not None:
        await computation(frame).flush_async()


def consume(ids):
    if any(not isinstance(node, str) or not node.startswith(VERSION_ID_PREFIX) for node in ids):
        _fail("Consumed response must have immutable canonical IDs")
    _add_reads(ids)
