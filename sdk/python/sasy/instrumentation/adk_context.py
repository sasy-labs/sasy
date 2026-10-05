"""Captured JSON state operations in an observed ADK computation.

State access is synchronous: readers capture values immediately and action
boundaries flush their pending immutable observations through the SDK.
"""
from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from sasy.observability._snapshots import VERSION_ID_PREFIX

from . import adk_state as resources
from .session import is_session_active


class FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        resources._fail("Nested state mutation requires copying and reassigning the state key")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self):
        return {key: thaw(value) for key, value in self.items()}

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        return self.__copy__()


class FrozenList(list):
    def _immutable(self, *args, **kwargs):
        resources._fail("Nested state mutation requires copying and reassigning the state key")
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __copy__(self):
        return [thaw(value) for value in self]

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        return self.__copy__()


def freeze(value):
    if type(value) is dict:
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if type(value) is list:
        return FrozenList(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, dict):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, list):
        return [thaw(item) for item in value]
    return value


@dataclass(eq=False)
class Snapshot:
    """One observed value of one state key or artifact, on its way into the
    graph.

    It names the ledger and key it belongs to, the :class:`~.adk_state.Version`
    it came from, the event recorded for it and that event's parents, and the
    immutable version IDs once it has been recorded. The flags say whether it
    has been consumed, whether it is still to be published, whether it came
    from a callback, and whether it was forwarded from a child agent.
    """

    ledger: Any
    key: tuple
    version: Any
    event: Any
    parents: list[Any] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    consumed: bool = False
    publish: bool = True
    callback: bool = False
    forwarded: bool = False


@dataclass
class Receipt:
    """The state writes made through one observed ToolContext or
    CallbackContext.

    They are staged here until ADK emits the event whose ``state_delta``
    carries them, and then committed to the ledger; if the turn ends first,
    :meth:`rollback` puts the state back as it was.
    """

    context: Any
    computation: Any
    call_id: str
    state: Any
    original: dict[str, tuple[bool, Any]] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    versions: dict[tuple, Snapshot] = field(default_factory=dict)
    artifacts: dict[str, int] = field(default_factory=dict)
    persisted: bool = False

    def check(self):
        frame = self.computation.frame
        if (self.context is not frame.context or self.context.function_call_id != self.call_id
                or self.context._invocation_context is not frame.invocation or self.context._state is not self.state):
            resources._fail("ToolContext identity changed during observed execution")

    def rollback(self):
        if self.persisted:
            return
        for key, (present, value) in self.original.items():
            if self.state._value.get(key) == self.values[key]:
                if present:
                    self.state._value[key] = copy.deepcopy(value)
                else:
                    self.state._value.pop(key, None)
            if self.state._delta.get(key) == self.values[key]:
                self.state._delta.pop(key, None)


class Computation:
    """The state and artifact activity of one running computation: what it has
    read, what it has staged to write, and the receipts of contexts it handed
    out.

    Reads become dependencies of whatever the computation produces; writes are
    published only once ADK emits the event that carries them.
    """

    def __init__(self, frame):
        self.frame = frame
        self.pending: list[Snapshot] = []
        self.busy = False
        self.receipts: dict[int, Any] = {}
        self.local: dict[tuple, Snapshot] = {}

    def check(self):
        self.frame.resources.check()
        if resources._frame.get() is not self.frame:
            resources._fail("State computation context changed")

    def _read(self, key, value):
        self.check()
        owner = self.frame.resources
        sdk = resources._adk()
        ledger = owner.claim(owner.runner.session_service)
        fingerprint = sdk._record_json(value)
        state_key = ("state", *key[1:]) if key[0] == "state-presence" else key
        own = self.own_receipt()
        writer = owner.context_writers.get(state_key)
        if writer is not None and not writer.persisted and writer.computation is not self and writer is not own:
            resources._fail("State read raced an uncommitted producer")
        local = self.local.get(key)
        if local is not None:
            if local.version.fingerprint != fingerprint:
                resources._fail("State changed without an observed assignment")
            local.consumed = True
            return local
        if own is not None and key in own.versions:
            # A write made in this tool call's callback is readable by the tool
            # itself once its own observation has resolved.
            shared = own.versions[key]
            if shared.version.fingerprint != fingerprint:
                resources._fail("State changed without an observed assignment")
            if not shared.ids:
                resources._fail("State read raced an uncommitted producer")
            resources._add_reads(shared.ids)
            return None
        version = ledger.versions.get(key)
        foreign = resources._foreign(version, owner.state.session)
        if version is not None:
            if version.fingerprint != fingerprint:
                if not foreign:
                    resources._fail("State changed without an observed producer")
                # A value another session changed is simply a new external input here.
                version = resources.Version(fingerprint, producer_graph=version.producer_graph, value=value)
                ledger.versions[key] = version
            elif owner.state.session in version.nodes:
                owner.state.snapshots.update(resources._session_snapshots(version, owner.state.session))
                resources._add_reads(version.nodes[owner.state.session])
                return None
        for pending in self.pending:
            if pending.key == key and pending.version.fingerprint == fingerprint and pending.publish:
                pending.consumed = True
                return pending
        snapshot = self._snapshot(ledger, key, value, external=version if foreign else None)
        snapshot.consumed = True
        return snapshot

    def _snapshot(self, ledger, key, value, parents=None, *, publish=True, external=None):
        sdk = resources._adk()
        produced = parents is not None
        payload = {"adk_resource": list(key),
            "provenance": "observed production" if produced else "unattributed external input", "value": value}
        if external is not None:
            payload["origin"] = resources.EXTERNAL_SESSION_ORIGIN
        event = sdk.Event(id=str(uuid4()), agent=self.frame.invocation.agent.name,
            role=sdk.Role.AGENT if produced else sdk.Role.USER, text=sdk._json(payload))
        version = external if external is not None else resources.Version(sdk._record_json(value),
            producer_graph=self.frame.resources.state.session if produced else None, value=value)
        snapshot = Snapshot(ledger, key, version, event, list(parents or []), publish=publish)
        self.pending.append(snapshot)
        return snapshot

    def read(self, context, state, key, *, presence=False):
        self.check()
        if context_invocation(context) is not self.frame.invocation:
            resources._fail("State context does not belong to this observed computation")
        if type(key) is not str:
            resources._fail("ADK state keys must be strings")
        present = key in state
        value = {"present": present}
        if present and not presence:
            value["value"] = resources._value(state[key])
        owner = self.frame.resources
        ledger = owner.claim(owner.runner.session_service)
        apply_state_import(owner, ledger, key, self.frame.invocation, value, presence=presence)
        state_key = owner.state_key(key, self.frame.invocation)
        if presence:
            state_key = ("state-presence", *state_key[1:])
        self._read(state_key, value)
        return present, freeze(value.get("value"))

    def own_receipt(self):
        frame = self.frame
        receipt = frame.resources.context_receipts.get(receipt_key(frame))
        return receipt if receipt is not None and receipt.context is frame.context else None

    def receipt(self, context):
        self.check()
        frame = self.frame
        if frame.kind not in ("tool", "callback") or context is not frame.context:
            resources._fail("State writes require the exact injected ToolContext")
        if frame.kind == "tool" and not frame.call_id:
            resources._fail("State writes require the exact injected ToolContext")
        owner = frame.resources
        key = receipt_key(frame)
        receipt = owner.context_receipts.get(key)
        if receipt is None:
            if (context.actions.state_delta or context.actions.artifact_delta) and not validate_deltas(context.actions):
                resources._fail("ToolContext contains unobserved resource deltas")
            receipt = Receipt(context, self, frame.call_id, context._state)
            owner.context_receipts[key] = receipt
        elif receipt.computation is not self:
            # A tool and its before/after callbacks share one context, one
            # emitted delta and therefore one receipt.
            if receipt.context is not context or receipt.state is not context._state or receipt.persisted:
                resources._fail("ToolContext identity changed during observed execution")
            receipt.computation = self
        receipt.check()
        return receipt

    def write(self, context, state, key, value, *, producers=None, presence_producers=None):
        """Assign a state key. `producers` forwards a child's own observed writers."""
        self.check()
        if type(key) is not str:
            resources._fail("ADK state keys must be strings")
        receipt = self.receipt(context)
        if receipt.state is not state:
            resources._fail("State handle changed during observed computation")
        owner = self.frame.resources
        forwarded = producers is not None
        ledger = owner.claim(owner.runner.session_service)
        state_key = owner.state_key(key, self.frame.invocation)
        competing = owner.context_writers.get(state_key)
        if competing is not None and competing is not receipt and not competing.persisted:
            resources._fail("Concurrent state writers require explicit ordering")
        frozen = resources._value(thaw(value))
        existed = key in state
        if forwarded:
            # The forwarded value keeps the child's producers; the parent call
            # consumes neither the previous value nor its presence.
            parents = list(producers)
            presence_parents = list(presence_producers or producers)
        else:
            self.read(context, state, key, presence=True)
            parents = list(dict.fromkeys([*resources._adk()._current_input_ids.get(), *self.frame.origin, *self.frame.reads]))
            parents.extend(item for item in self.pending if item.consumed)
            presence_parents = parents
        previous = (existed, resources._value(state[key]) if existed else None)
        state[key] = copy.deepcopy(frozen)
        receipt.original.setdefault(key, previous)
        receipt.values[key] = frozen
        owner.context_writers[state_key] = receipt
        callback = self.frame.kind == "callback"
        snapshot = self._snapshot(ledger, state_key, {"present": True, "value": frozen}, parents, publish=False)
        snapshot.callback = callback
        snapshot.forwarded = forwarded
        self.local[state_key] = snapshot
        receipt.versions[state_key] = snapshot
        if not existed:
            presence_key = ("state-presence", *state_key[1:])
            presence = self._snapshot(ledger, presence_key, {"present": True}, presence_parents, publish=False)
            presence.callback = callback
            presence.forwarded = forwarded
            self.local[presence_key] = presence
            receipt.versions[presence_key] = presence

    def _request(self):
        sdk = resources._adk()
        result = []
        for item in self.pending:
            if item.ids:
                continue
            sources: list[str] = []
            for parent in item.parents:
                sources.extend(parent.ids or [parent.event.id] if isinstance(parent, Snapshot) else [parent])
            sources = list(dict.fromkeys(sources))
            result.append(sdk.EventSnapshot(event=item.event, dependencies=[sdk.Edge(
                source=source, destination=item.event.id, message_index=i, proximal=i == len(sources)-1)
                for i, source in enumerate(sources)]))
        return result

    def _complete(self, returned):
        self.check()
        unresolved = [item for item in self.pending if not item.ids]
        if len(returned) != len(unresolved) or any(not node.startswith(VERSION_ID_PREFIX) for node in returned):
            resources._fail("State observation returned invalid immutable identities")
        owner = self.frame.resources
        for item, node in zip(unresolved, returned, strict=True):
            item.ids = [node]
            item.version.nodes[owner.state.session] = [node]
            item.version.snapshots[node] = item.event
            owner.state.snapshots[node] = item.event
            if item.publish and item.key not in item.ledger.versions:
                item.ledger.versions[item.key] = item.version
        for item in self.pending:
            if item.consumed:
                resources._add_reads(item.ids)
                item.consumed = False

    def flush(self):
        self.check()
        if self.busy:
            resources._fail("Concurrent state observation is unsupported")
        self.busy = True
        try:
            request = self._request()
            self._complete(resources._adk().observation.resolve_events(request) if request else [])
        finally:
            self.busy = False

    async def flush_async(self):
        self.check()
        if self.busy:
            resources._fail("Concurrent state observation is unsupported")
        self.busy = True
        try:
            request = self._request()
            self._complete(await resources._adk().observation.resolve_events_async(request) if request else [])
        finally:
            self.busy = False

    def inputs_sync(self):
        self.flush()
        return list(dict.fromkeys([*self.frame.origin, *self.frame.reads]))

    async def inputs_async(self):
        await self.flush_async()
        return list(dict.fromkeys([*self.frame.origin, *self.frame.reads]))


class ReadonlyStateView(Mapping):
    def __init__(self, context, state, computation):
        self._context = context
        self._state = state
        self._computation = computation

    def __getitem__(self, key):
        present, value = self._computation.read(self._context, self._state, key)
        if not present:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self._computation.read(self._context, self._state, key, presence=True)[0]

    def _keys(self):
        return list(self._state)

    def __iter__(self):
        self._computation.check()
        keys = self._keys()
        for key in keys:
            self._computation.read(self._context, self._state, key, presence=True)
        operation = self._computation
        owner = operation.frame.resources
        ledger = owner.claim(owner.runner.session_service)
        parents = [*operation.frame.reads, *(item for item in operation.pending if item.consumed)]
        projection = operation._snapshot(ledger,
            ("state-keys", owner.runner.app_name, owner.user_id, owner.session_id,
             operation.frame.invocation.invocation_id), keys, parents, publish=False)
        projection.consumed = True
        return iter(keys)

    def __len__(self):
        return len(list(iter(self)))

    def to_dict(self):
        return {key: self[key] for key in self}


class StateView(ReadonlyStateView):
    def __init__(self, context, state, computation):
        super().__init__(context, state, computation)
        self._schema = state._schema

    def _keys(self):
        return list(self._state.to_dict())

    def has_delta(self):
        return self._state.has_delta()

    def __setitem__(self, key, value):
        self._computation.write(self._context, self._state, key, value)

    def update(self, values):
        values = resources._value(thaw(dict(values)))
        for key, value in values.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return self[key]


def computation(frame=None):
    frame = frame or resources._frame.get()
    if frame is None:
        resources._fail("State access lacks an observed computation")
    result = getattr(frame, "computation", None)
    if result is None:
        result = Computation(frame)
        frame.computation = result
    result.check()
    return result


def state_view(context, state):
    frame = resources._frame.get()
    if frame is None:
        resources._fail("State context does not belong to this observed computation")
    return StateView(context, state, computation(frame))


# The invocation a provider context stands for, held outside the object so
# that no attribute on it reaches the untracked session.
_provider_frames: WeakKeyDictionary = WeakKeyDictionary()


class ProviderContext:
    """The context an instruction provider receives: tracked state only.

    Session, invocation content and credential access stay blocked because
    those resources carry no observed provenance of their own.
    """
    def __init__(self, frame):
        _provider_frames[self] = frame

    @property
    def state(self):
        frame = provider_frame(self, current=True)
        return ReadonlyStateView(self, frame.invocation.session.state, computation(frame))

    @property
    def invocation_id(self):
        return provider_frame(self).invocation.invocation_id

    @property
    def agent_name(self):
        return provider_frame(self).invocation.agent.name

    @property
    def user_id(self):
        return provider_frame(self).invocation.user_id

    def __getattr__(self, name):
        resources._fail("This instruction provider context resource requires additional instrumentation")


def provider_frame(context, current=False):
    frame: Any = _provider_frames.get(context)
    if frame is None or (current and resources._frame.get() is not frame):
        resources._fail("State context does not belong to this observed computation")
    return frame


def context_invocation(context):
    """The invocation a state context belongs to, for contexts that hold none."""
    if isinstance(context, ProviderContext):
        return provider_frame(context).invocation
    return context._invocation_context


def receipt_key(frame):
    """A tool call's receipt covers its callbacks; other callbacks own theirs."""
    return (frame.invocation.agent.name, frame.call_id or id(frame.context))


def event_receipts(owner, actions, event=None):
    receipts = [receipt for receipt in owner.context_receipts.values() if receipt.context.actions is actions]
    for response in event.get_function_responses() if event is not None else []:
        receipt = owner.context_receipts.get((event.author, response.id))
        if receipt is not None and not any(held is receipt for held in receipts):
            receipts.append(receipt)
    return receipts


def context_deltas(actions, event=None):
    state = resources._adk()._active.get()
    owner = state.resources if state else None
    if owner is None:
        return None
    receipts = event_receipts(owner, actions, event)
    if not receipts:
        return None
    values = {}
    artifacts = {}
    for receipt in receipts:
        receipt.check()
        values.update(receipt.values)
        artifacts.update(receipt.artifacts)
    return values, artifacts


def context_state(actions, event=None):
    """The state keys written through observed contexts sharing these actions."""
    deltas = context_deltas(actions, event)
    return deltas[0] if deltas else {}


def validate_deltas(actions, event=None):
    deltas = context_deltas(actions, event)
    if deltas is None:
        return False
    values, artifacts = deltas
    if resources.stored_delta(event):
        # The stored delta has lost its temp: keys; the writes behind them are
        # still the ones this turn observed.
        values = resources.persistent_keys(values)
    # Both deltas are compared as data, not with Python equality: ``True`` and
    # the artifact version ``1`` are equal to Python and are two different
    # values here, and only the second one was observed.
    sdk = resources._adk()
    return (sdk._record_json(actions.state_delta) == sdk._record_json(values)
            and sdk._record_json(actions.artifact_delta) == sdk._record_json(artifacts))


def stage_receipts(owner, event):
    """Publish callback writes once ADK emits the event that carries them.

    When the model asked for several tools at once, ADK merges the per-call
    response events into a new event with new actions, so the receipts of the
    calls in the batch are reached through the function responses the merged
    event carries rather than through the actions object they wrote.
    """
    if event.partial or event.actions is None:
        return
    # ADK applies the agent's output_key after the callback writes this event
    # carries, so a key both of them wrote keeps the output_key value.
    overwritten = {(id(ledger), key) for ledger, key, _ in owner.pending_output.get(event.id, [])}
    for receipt in event_receipts(owner, event.actions, event):
        if receipt.persisted or any(held is receipt for held in owner.pending_receipts.get(event.id, [])):
            continue
        receipt.check()
        for item in receipt.versions.values():
            if not item.ids:
                resources._fail("State write was not observed before its event")
            if (id(item.ledger), item.key) in overwritten:
                continue
            owner.pending_output.setdefault(event.id, []).append((item.ledger, item.key, item.version))
        owner.pending_receipts.setdefault(event.id, []).append(receipt)


def stage_event(owner, event):
    receipts = [owner.context_receipts[(event.author, response.id)] for response in event.get_function_responses()
                if (event.author, response.id) in owner.context_receipts]
    if receipts and not validate_deltas(event.actions, event):
        resources._fail("Tool event contains unobserved resource deltas")
    for receipt in receipts:
        for item in receipt.versions.values():
            if not item.ids:
                resources._fail("State write was not observed before tool output")
            owner.pending_output.setdefault(event.id, []).append((item.ledger, item.key, item.version))
        owner.pending_receipts.setdefault(event.id, []).append(receipt)


def context_parameter(tool):
    signature = inspect.signature(tool.func)
    if "input_stream" in signature.parameters:
        resources._fail("Streaming tool context requires additional instrumentation")
    name = tool._context_param_name
    return name if name in signature.parameters else None


def validate_injected(context):
    from google.adk.agents.context import Context
    frame = resources._frame.get()
    if type(context) is not Context or frame is None or context is not frame.context:
        resources._fail("Only the exact ADK-injected ToolContext is supported")
    computation(frame).check()
    if context.function_call_id != frame.call_id or context._invocation_context is not frame.invocation:
        resources._fail("ToolContext execution identity changed")


_installed = False


def install():
    global _installed
    if _installed:
        return
    from functools import wraps

    from google.adk.agents.context import Context
    save = Context.save_artifact
    @wraps(save)
    async def observed_save(self, filename, artifact, custom_metadata=None):
        if (not is_session_active() or resources._adk()._active.get() is None):
            return await save(self, filename, artifact, custom_metadata)
        validate_injected(self)
        operation = computation()
        if operation.frame.kind != "tool":
            resources._fail("Callback artifact writes require additional instrumentation")
        receipt = operation.receipt(self)
        await operation.flush_async()
        version = await save(self, filename, artifact, custom_metadata)
        operation.check()
        receipt.artifacts[filename] = version
        return version
    setattr(Context, "save_artifact", observed_save)

    for name in ("save_credential", "load_credential", "get_auth_response", "request_credential", "request_confirmation",
                 "add_session_to_memory", "add_events_to_memory", "add_memory", "search_memory", "render_ui_widget", "run_node"):
        original = getattr(Context, name)
        if inspect.iscoroutinefunction(original):
            @wraps(original)
            async def guarded(self, *args, _original=original, _name=name, **kwargs):
                if (is_session_active() and resources._adk()._active.get() is not None) and not (_name == "run_node" and native_node_allowed()):
                    resources._fail("This ToolContext control or credential API requires additional instrumentation")
                return await _original(self, *args, **kwargs)
        else:
            @wraps(original)
            def guarded(self, *args, _original=original, **kwargs):
                if (is_session_active() and resources._adk()._active.get() is not None):
                    resources._fail("This ToolContext control or credential API requires additional instrumentation")
                return _original(self, *args, **kwargs)
        setattr(Context, name, guarded)
    _installed = True


async def complete_tool(context, ids, successful):
    frame = resources._frame.get()
    if frame is None or frame.kind != "tool":
        return
    operation = computation(frame)
    receipt = frame.resources.context_receipts.get((frame.invocation.agent.name, frame.call_id))
    if not successful:
        return
    from . import adk_artifacts
    await adk_artifacts.complete_tool(frame, ids)
    if receipt is None:
        return
    receipt.check()
    await operation.flush_async()
    for key, previous in list(receipt.versions.items()):
        if previous.callback or previous.forwarded:
            # Neither a callback write nor a child's forwarded write is evidence
            # that the surrounding tool call produced it.
            continue
        produced = operation._snapshot(previous.ledger, key, previous.version.value,
            [*previous.ids, *ids], publish=False)
        receipt.versions[key] = produced
        operation.local[key] = produced
    await operation.flush_async()



_native_node: ContextVar[Any] = ContextVar("sasy_adk_native_node_setup", default=None)


@contextmanager
def native_run_node():
    """Permit the qualified native child setup while parent scopes are suspended."""
    import asyncio

    from . import dependencies
    from .adk_callbacks import in_callback
    sdk = resources._adk()
    if resources._frame.get() is not None or dependencies.bind().resolver is not None or sdk._tool_context.get() is not None or in_callback():
        resources._fail("Native child setup requires suspended application scopes")
    token = _native_node.set((asyncio.current_task(), sdk.current_wire_session_id(), resources.get_current_entity()))
    try:
        yield
    finally:
        _native_node.reset(token)


def native_node_allowed():
    import asyncio

    from . import dependencies
    from .adk_callbacks import in_callback
    sdk = resources._adk()
    return (_native_node.get() == (asyncio.current_task(), sdk.current_wire_session_id(), resources.get_current_entity())
            and resources._frame.get() is None and dependencies.bind().resolver is None
            and sdk._tool_context.get() is None and not in_callback())


@asynccontextmanager
async def callback_scope(context, inputs, *, root=False):
    parent = resources._frame.get()
    if parent is None and not root:
        resources._fail("Callback has no observed enclosing computation")
    owner = parent.resources if parent is not None else resources._adk()._state().resources
    if owner is None:
        resources._fail("Callback has no observed enclosing computation")
    frame = resources.Frame(owner, context._invocation_context, "callback", origin=list(inputs),
        context=context, call_id=getattr(context, "function_call_id", None))
    token = resources._frame.set(frame)
    resolver_token = resources.enter_resolver(frame)
    try:
        yield frame
        await computation(frame).flush_async()
    finally:
        frame.live = False
        resources.leave_resolver(resolver_token)
        resources._frame.reset(token)
    if parent is not None:
        resources.consume(frame.reads)


@dataclass(frozen=True)
class StateTransfer:
    """A qualified native copy; values become inputs only when actually read."""
    graph: str
    entity: Any
    values: dict[str, Any]
    versions: dict[str, tuple[Any, Any]]


def capture_state(context):
    """Capture native AgentTool state transport without declaring all keys read."""
    validate_injected(context)
    operation = computation()
    owner = operation.frame.resources
    ledger = owner.claim(owner.runner.session_service)
    values = resources._value(context._state.to_dict())
    versions = {}
    sdk = resources._adk()
    for key, value in values.items():
        state_key = owner.state_key(key, operation.frame.invocation)
        writer = owner.context_writers.get(state_key)
        if writer is not None and not writer.persisted:
            resources._fail("Cannot transport state with an uncommitted writer")
        payloads = ({"present": True, "value": value}, {"present": True})
        apply_state_import(owner, ledger, key, operation.frame.invocation, payloads[0])
        apply_state_import(owner, ledger, key, operation.frame.invocation, payloads[1], presence=True)
        keys = (state_key, ("state-presence", *state_key[1:]))
        captured = []
        for source_key, payload in zip(keys, payloads, strict=True):
            fingerprint = sdk._record_json(payload)
            version = ledger.versions.get(source_key)
            if version is None:
                version = resources.Version(fingerprint, value=payload)
            if version.fingerprint != fingerprint:
                if not resources._foreign(version, owner.state.session):
                    resources._fail("Copied state changed without an observed producer")
                version = resources.Version(fingerprint, producer_graph=version.producer_graph, value=payload)
                ledger.versions[source_key] = version
            captured.append(version)
        versions[key] = tuple(captured)
    return StateTransfer(owner.state.session, resources.get_current_entity(), values, versions)


def attach_state_import(owner, transfer, keys=None):
    """Attach per-key identities to an isolated native child resource scope."""
    if owner.state.session != transfer.graph or resources.get_current_entity() != transfer.entity:
        resources._fail(
            "Child state was carried into a different SASY session or entity than the "
            "one it was read in."
        )
    if owner.readonly_import:
        resources._fail("Child state transport was already attached")
    selected = set(transfer.versions) if keys is None else set(keys)
    if not selected.issubset(transfer.versions):
        resources._fail("Child imported state contains uncaptured keys")
    owner.imported_state = {key: transfer.versions[key] for key in selected}
    owner.readonly_import = True


def apply_state_import(owner, ledger, key, invocation, value, *, presence=False):
    imported = owner.imported_state.get(key)
    if imported is None:
        return
    version = imported[1 if presence else 0]
    if version.fingerprint != resources._adk()._record_json(value):
        resources._fail("Copied state changed before its observed read")
    state_key = owner.state_key(key, invocation)
    if presence:
        state_key = ("state-presence", *state_key[1:])
    existing = ledger.versions.get(state_key)
    if existing is not None and existing is not version:
        if existing.fingerprint != version.fingerprint or existing.producer_graph != version.producer_graph or existing.nodes != version.nodes:
            resources._fail("State import conflicts with an existing child value")
        return
    ledger.versions[state_key] = version
