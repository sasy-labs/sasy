"""Exact native ADK message handoffs and resource-aware AgentTool delegation."""
from __future__ import annotations

import copy
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from .session import is_session_active

_installed = False
_build: ContextVar[Any] = ContextVar("sasy_adk_content_build", default=None)
_delegation: ContextVar[Any] = ContextVar("sasy_adk_delegation", default=None)


def _sdk():
    from . import adk
    return adk


def _key(content):
    # The same fields as before, through the canonical encoding: a JSON-mode
    # dump writes a tuple and the list it becomes alike, so a bound message
    # could be changed without the key that guards the binding moving.
    return _sdk()._record_json(content.model_dump(exclude_none=True))


# The tables below are keyed by id() because ADK content objects are
# unhashable and compare by value; each entry keeps the object itself, so the
# id cannot be recycled behind our back, and identity and the content key are
# both re-checked on lookup.
def _bindings(state):
    if not hasattr(state.resources, "agent_bindings"):
        state.resources.agent_bindings = {}
    return state.resources.agent_bindings


def input_ids(state, agent, content):
    entry = _bindings(state).get((agent, id(content)))
    if entry is None:
        return None
    original, key, ids = entry
    if original is not content or _key(content) != key:
        raise _sdk().AdkInstrumentationError("Native message presentation changed without explicit derivation")
    return list(ids)


def forget_input(state, agent, content):
    """Retire a native binding when a new observed derivation replaces it."""
    binding = _bindings(state).get((agent, id(content)))
    if binding is not None and binding[0] is content:
        del _bindings(state)[(agent, id(content))]


def copy_inputs(state, agent, original, copied):
    for source, target in zip(original.contents, copied.contents, strict=True):
        ids = input_ids(state, agent, source)
        if ids is not None:
            _bindings(state)[(agent, id(target))] = (target, _key(target), ids)


def capture_input(content, ids):
    """Bind a native generated input to its exact initiating message IDs.

    Native task dispatch uses this while ADK builds the child model request.
    Observation occurs after the builder completes, before application callbacks.
    """
    capture = _build.get() if is_session_active() else None
    if capture is None or not ids:
        raise _sdk().AdkInstrumentationError("Generated child input lacks its native build scope")
    capture.copies[id(content)] = (content, _key(content), list(ids))


@dataclass
class Build:
    """One native build of a child agent's input, while it is being built.

    ``sources`` are the observed events the build reads, ``copies`` the
    contents it produced from them, both keyed by object identity.
    """

    state: Any
    agent: str
    sources: dict = field(default_factory=dict)
    copies: dict = field(default_factory=dict)

    def add_source(self, content, ids):
        previous = self.sources.get(id(content))
        if previous is not None and previous[2] != ids:
            raise _sdk().AdkInstrumentationError("Aliased native events require distinct content occurrences")
        self.sources[id(content)] = (content, _key(content), list(ids))

    def source(self, content):
        entry = self.sources.get(id(content))
        if entry is None:
            return None
        if entry[0] is not content or entry[1] != _key(content):
            raise _sdk().AdkInstrumentationError("Native history content changed before consumption")
        return entry[2]


def _record_for_event(state, event):
    sdk = _sdk()
    record = state.records.get(event.id)
    if record is None and event.author == "user":
        # Runner persists the new user event before model preprocessing. Its
        # native ID replaces the temporary SDK event, even for repeated text.
        pending = [(key, value) for key, value in state.records.items()
                   if key not in state.history and value.event.author == "user"
                   and _key(value.event.content) == _key(event.content)]
        if len(pending) == 1:
            key, record = pending[0]
            state.records.pop(key)
            record = sdk._Record(event.model_copy(deep=True), record.ids, record.key)
            state.records[event.id] = record
            sdk._remember_history(state, event)
    if record is None:
        raise sdk.AdkInstrumentationError("Native message has no observed source event")
    if _key(record.event.content) != _key(event.content):
        raise sdk.AdkInstrumentationError("Native source event changed after observation")
    return record


@dataclass
class Delegation:
    """One AgentTool delegation: a parent agent calling a child agent as a tool.

    (``adk_tasks_native.Delegation`` tracks ADK's native task delegation, which
    is a different mechanism.)
    """

    tool: Any
    context: Any
    parent_state: Any
    inputs: list[str]
    previous: Any = None
    runner: Any = None
    child_state: Any = None
    selected: list[str] = field(default_factory=list)
    state_transfer: Any = None
    forwarded: dict = field(default_factory=dict)


def _forwarded_write(delegation, key, value):
    """Bind one forwarded delta entry to the child's own observed write."""
    sdk = _sdk()
    from . import adk_state
    child = delegation.child_state
    if child is None or child.resources is None or child.resources.closed:
        raise sdk.AdkInstrumentationError("Forwarded AgentTool state lacks its observed child run")
    owner = child.resources
    if owner.state.session != delegation.parent_state.session:
        raise sdk.AdkInstrumentationError(
            "The AgentTool child ran in a different SASY session than its parent, so "
            "the state it forwarded cannot be attributed here."
        )
    if type(key) is not str or key.startswith("temp:") or key.startswith("_adk"):
        raise sdk.AdkInstrumentationError("Forwarded temporary or internal AgentTool state requires explicit provenance")
    state_key = owner.state_key(key, None)
    version = owner.produced.get(state_key)
    if version is None or version.producer_graph != owner.state.session:
        raise sdk.AdkInstrumentationError("Forwarded AgentTool state has no observed child write")
    if version.fingerprint != sdk._record_json({"present": True, "value": adk_state._value(value)}):
        raise sdk.AdkInstrumentationError("Forwarded AgentTool state does not match the observed child write")
    presence = owner.produced.get(("state-presence", *state_key[1:]))
    nodes = version.nodes.get(owner.state.session)
    if not nodes:
        raise sdk.AdkInstrumentationError("Forwarded AgentTool state has no observed child write")
    return (value, list(nodes), list(presence.nodes.get(owner.state.session, [])) if presence is not None else None)


async def _forward_state(context, delegation, origin):
    """Apply the child's completed writes to the parent once the child finishes."""
    if not delegation.forwarded:
        return
    if len(origin.event.get_function_calls()) > 1:
        # Native merging of a parallel batch replaces the response event that
        # would commit these writes, leaving the parent value unattributed.
        raise _sdk().AdkInstrumentationError("AgentTool state forwarding beside parallel tool calls requires explicit ordering")
    from .adk_context import computation
    operation = computation()
    for key, (value, producers, presence) in delegation.forwarded.items():
        if key not in context._state and presence is None:
            raise _sdk().AdkInstrumentationError("Forwarded AgentTool state has no observed child write")
        operation.write(context, context._state, key, value, producers=producers, presence_producers=presence)
    await operation.flush_async()


class _CopiedState:
    def __init__(self, delegation, values):
        self._delegation = delegation
        self._values = values

    def to_dict(self):
        return copy.deepcopy(self._values)

    def update(self, values):
        for key, value in (values or {}).items():
            self._delegation.forwarded[key] = _forwarded_write(self._delegation, key, value)

    def __setitem__(self, key, value):
        raise _sdk().AdkInstrumentationError("AgentTool state forwarding requires explicit provenance")


class _DelegatedContext:
    def __init__(self, context, delegation, transfer):
        self._context = context
        self.state = _CopiedState(delegation, transfer.values)

    def __getattr__(self, name):
        return getattr(self._context, name)


def _validate_child(agent):
    from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
    if type(agent) not in (LlmAgent, SequentialAgent, ParallelAgent, LoopAgent):
        raise _sdk().AdkInstrumentationError("AgentTool child must use qualified native agents")
    if isinstance(agent, LlmAgent):
        if agent.input_schema or agent.output_schema:
            raise _sdk().AdkInstrumentationError("AgentTool currently supports text schemas only")
    for child in agent.sub_agents:
        _validate_child(child)


def validate_tool(tool):
    from google.adk.tools.agent_tool import AgentTool
    if not isinstance(tool, AgentTool):
        return False
    if type(tool) is not AgentTool:
        raise _sdk().AdkInstrumentationError("Task and single-turn agent tools require separate node instrumentation")
    for name in ("run_async", "process_llm_request", "_get_declaration"):
        if getattr(getattr(tool, name), "__func__", None) is not getattr(AgentTool, name):
            raise _sdk().AdkInstrumentationError("AgentTool overrides require explicit instrumentation")
    if tool.propagate_grounding_metadata:
        raise _sdk().AdkInstrumentationError("AgentTool grounding state requires explicit provenance")
    if tool.name != tool.agent.name:
        raise _sdk().AdkInstrumentationError("AgentTool name must identify its actual child")
    _validate_child(tool.agent)
    return True


def _forwarding_delegation(service):
    from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
    if type(service) is not ForwardingArtifactService or not isinstance(service.tool_context, _DelegatedContext):
        return None
    delegation = _delegation.get()
    while delegation is not None:
        if service.tool_context._context is delegation.context:
            for name in ("load_artifact", "list_artifact_keys", "save_artifact", "delete_artifact",
                         "list_versions", "list_artifact_versions", "get_artifact_version"):
                if getattr(getattr(service, name), "__func__", None) is not getattr(ForwardingArtifactService, name):
                    raise _sdk().AdkInstrumentationError("Artifact forwarding overrides require explicit instrumentation")
            if service._invocation_context is not delegation.context._invocation_context:
                raise _sdk().AdkInstrumentationError("Artifact forwarding context changed after delegation")
            return delegation
        delegation = delegation.previous
    return None


def permits_forwarder(service):
    return _forwarding_delegation(service) is not None


async def prepare_user(runner, state, event):
    delegation = _delegation.get()
    if delegation is None or runner.agent is not delegation.tool.agent:
        return False
    if delegation.runner is not None:
        raise _sdk().AdkInstrumentationError("AgentTool attempted multiple nested invocations")
    delegation.runner = runner
    delegation.child_state = state
    from .adk_context import attach_state_import
    attach_state_import(state.resources, delegation.state_transfer, keys=[
        key for key in delegation.state_transfer.values if not key.startswith(("_adk", "temp:"))
    ])
    ids = await state.record(event.content, "user", delegation.inputs)
    state.records[event.id] = _sdk()._Record(event.model_copy(deep=True), ids, _sdk()._content_key(event.content))
    return True


def child_event(runner, state, event):
    delegation = _delegation.get()
    if delegation is None or delegation.runner is not runner:
        return
    if event.actions.artifact_delta:
        raise _sdk().AdkInstrumentationError("AgentTool artifact forwarding requires explicit provenance")
    if event.content:
        record = state.records.get(event.id)
        if record is None or _key(record.event.content) != _key(event.content):
            raise _sdk().AdkInstrumentationError("AgentTool response lacks exact observed provenance")
        delegation.selected = list(record.ids)


def install():
    global _installed
    if _installed:
        return
    from google.adk.flows.llm_flows import contents
    from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
    from google.adk.tools.agent_tool import AgentTool

    from . import adk_artifacts, adk_otel, adk_state

    process = contents._ContentLlmRequestProcessor.run_async
    @wraps(process)
    async def observed_contents(self, invocation_context, llm_request):
        if not is_session_active():
            iterator = process(self, invocation_context, llm_request)
            try:
                async for value in iterator:
                    yield value
            finally:
                await iterator.aclose()
            return
        sdk = _sdk()
        state = sdk._state()
        agent = invocation_context.agent.name
        capture = Build(state, agent)
        for event in invocation_context.session.events:
            if event.content:
                record = _record_for_event(state, event)
                capture.add_source(event.content, record.ids)
        bindings = _bindings(state)
        for key in [key for key in bindings if key[0] == agent]:
            del bindings[key]
        token = _build.set(capture)
        iterator = process(self, invocation_context, llm_request)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()
            _build.reset(token)
        for content in llm_request.contents:
            entry = capture.copies.get(id(content))
            if entry is None:
                continue
            original, expected, ids = entry
            if original is not content or expected != _key(content):
                raise sdk.AdkInstrumentationError("Native request presentation changed before observation")
            # Exact unchanged events keep their identities; fenced child replies
            # receive their own presentation nodes linked only to that reply.
            source = next((record for record in state.records.values()
                           if record.ids == ids and _key(record.event.content) == expected), None)
            if source is None:
                ids = await state.record(content, agent, ids,
                    cache_key=("native-handoff", agent, tuple(ids), expected))
            bindings[(agent, id(content))] = (content, expected, list(ids))
    setattr(contents._ContentLlmRequestProcessor, "run_async", observed_contents)

    present = contents._present_other_agent_message
    @wraps(present)
    def observed_fence(event, **kwargs):
        result = present(event, **kwargs)
        capture = _build.get() if is_session_active() else None
        if capture is not None and result and result.content and event.content:
            ids = capture.source(event.content)
            if ids is not None:
                capture.add_source(result.content, ids)
        return result
    contents._present_other_agent_message = observed_fence

    copy_content = contents._copy_content_for_request
    @wraps(copy_content)
    def observed_copy(content, **kwargs):
        result = copy_content(content, **kwargs)
        capture = _build.get() if is_session_active() else None
        if capture is not None:
            ids = capture.source(content)
            if ids is not None:
                capture.copies[id(result)] = (result, _key(result), ids)
        return result
    contents._copy_content_for_request = observed_copy


    from google.adk.flows.llm_flows import _tool_call_rearranger
    merge = _tool_call_rearranger.merge_function_response_events
    @wraps(merge)
    def observed_merge(events):
        result = merge(events)
        capture = _build.get() if is_session_active() else None
        if capture is not None:
            # Native rearrangement replaces prior responses by call ID. Follow
            # that selection exactly, retaining only the surviving occurrences.
            selected: list[list[str]] = []
            positions: dict[str | None, int] = {}
            for event in events:
                ids = capture.source(event.content)
                if ids is None:
                    raise _sdk().AdkInstrumentationError("Merged tool response lacks its native source")
                for index, part in enumerate(event.content.parts or []):
                    parents = [ids[index]] if len(ids) == len(event.content.parts) else ids
                    response = part.function_response
                    if response and response.id in positions:
                        selected[positions[response.id]] = parents
                    else:
                        if response:
                            positions[response.id] = len(selected)
                        selected.append(parents)
            parents = list(dict.fromkeys(node for ids in selected for node in ids))
            capture.add_source(result.content, parents)
        return result
    setattr(_tool_call_rearranger, "merge_function_response_events", observed_merge)
    setattr(_tool_call_rearranger, "_merge_function_response_events", observed_merge)

    validate_service = adk_artifacts.validate_service
    @wraps(validate_service)
    def forwarding_profile(service):
        if permits_forwarder(service):
            return
        return validate_service(service)
    adk_artifacts.validate_service = forwarding_profile

    def forwarded_read(original):
        @wraps(original)
        async def read(self, **kwargs):
            if (not is_session_active() or _sdk()._active.get() is None):
                return await original(self, **kwargs)
            delegation = _forwarding_delegation(self)
            child_frame = adk_state._frame.get()
            if delegation is None or child_frame is None:
                raise _sdk().AdkInstrumentationError("Artifact forwarding lacks its native delegation scope")
            child_frame.resources.check()
            owner = child_frame.resources
            if (kwargs.get("app_name"), kwargs.get("user_id"), kwargs.get("session_id")) != (
                owner.runner.app_name, owner.user_id, owner.session_id
            ):
                raise _sdk().AdkInstrumentationError("Forwarded artifacts require the current child scope")
            source = adk_state.Frame(delegation.parent_state.resources,
                delegation.context._invocation_context, "tool")
            source.context = delegation.context
            token = adk_state._frame.set(source)
            input_token = _sdk()._current_input_ids.set([])
            resolver = adk_state.enter_resolver(source)
            try:
                value = await original(self, **kwargs)
                source.resources.check()
                ids = list(source.reads)
                presentations = adk_artifacts._presentations.get()
                if presentations is not None:
                    entry = presentations.get(id(value)) if value is not None else presentations.get("missing")
                    if entry is not None:
                        ids.extend(entry[1])
                for node in ids:
                    owner.state.snapshots[node] = source.resources.state.snapshots[node]
            finally:
                source.live = False
                adk_state.leave_resolver(resolver)
                _sdk()._current_input_ids.reset(input_token)
                adk_state._frame.reset(token)
            # Body-presentation reads stay attached to that body, while native
            # template/catalog reads are consumed by this child's current frame.
            if source.reads:
                adk_state.consume(source.reads)
            return value
        return read

    for name in ("load_artifact", "list_artifact_keys"):
        setattr(ForwardingArtifactService, name, forwarded_read(getattr(ForwardingArtifactService, name)))
    for name in ("save_artifact", "delete_artifact", "list_versions", "get_artifact_version", "list_artifact_versions"):
        original = getattr(ForwardingArtifactService, name)
        @wraps(original)
        async def no_forwarded_write(self, *args, _original=original, **kwargs):
            if (is_session_active() and _sdk()._active.get() is not None):
                raise _sdk().AdkInstrumentationError(
                    "An AgentTool child wrote an artifact, or read its metadata, "
                    "through the forwarding service, which the adapter does not "
                    "observe. Save artifacts from a tool through ToolContext inside "
                    "the child."
                )
            return await _original(self, *args, **kwargs)
        setattr(ForwardingArtifactService, name, no_forwarded_write)

    run: Any = AgentTool.run_async
    @wraps(run)
    async def guarded_agent(self, *, args, tool_context):
        if not is_session_active():
            return await run(self, args=args, tool_context=tool_context)
        sdk = _sdk()
        validate_tool(self)
        state, agent, origin = sdk._origin(tool_context)
        sdk._validate_actions(tool_context.actions)
        if tool_context.actions.transfer_to_agent:
            raise sdk.AdkInstrumentationError("Transfer action preceded AgentTool dispatch")
        from .adk_context import capture_state
        state_transfer = capture_state(tool_context)
        arguments = copy.deepcopy(args)
        if "request" in arguments and not isinstance(arguments["request"], str):
            raise sdk.AdkInstrumentationError("AgentTool request must be text")
        await adk_state.flush_reads_async()
        inputs = adk_state.tool_inputs(origin.ids)
        child = self.agent
        name = self.name
        serialized = sdk._json(arguments)
        with adk_otel.operation("tool", agent, inputs, call_id=tool_context.function_call_id, tool=self.name) as telemetry:
            verdict = await sdk.monitor.check_tool_call_async(self.name, serialized, inputs,
                metadata=[("adk_agent", agent, ""), ("framework", "adk", "")])
            telemetry.decision(verdict)
            successful = verdict.authorized and not verdict.transform_ids
            if successful:
                # The check awaited. What is dispatched must still be the tool,
                # child agent and state that were authorized, and nothing may
                # have set a transfer in the meantime.
                validate_tool(self)
                checked_transfer = capture_state(tool_context)
                if sdk._record_json(checked_transfer.values) != sdk._record_json(state_transfer.values):
                    raise sdk.AdkInstrumentationError("Copied AgentTool state changed during authorization")
                state_transfer = checked_transfer
                sdk._validate_actions(tool_context.actions)
                if self.agent is not child or self.name != name or tool_context.actions.transfer_to_agent:
                    raise sdk.AdkInstrumentationError("AgentTool target changed during authorization")
                delegation = Delegation(self, tool_context, state, inputs, previous=_delegation.get(),
                    state_transfer=state_transfer)
                token = _delegation.set(delegation)
                input_token = sdk._current_input_ids.set(inputs)
                frame_token = adk_state._frame.set(None)
                try:
                    result = await run(self, args=arguments,
                        tool_context=_DelegatedContext(tool_context, delegation, state_transfer))
                finally:
                    adk_state._frame.reset(frame_token)
                    sdk._current_input_ids.reset(input_token)
                    _delegation.reset(token)
                if not delegation.selected:
                    raise sdk.AdkInstrumentationError("AgentTool returned without an observed child response")
                adk_state._add_reads(delegation.selected)
                await _forward_state(tool_context, delegation, origin)
            else:
                result = sdk._blocked(verdict, self.name)
            await sdk._record_tool(self, tool_context, result,
                sdk.Tool(name=self.name, arguments=serialized) if successful else None)
            telemetry.consumed(adk_state.tool_inputs(origin.ids))
            telemetry.produced(state.results[(agent, tool_context.function_call_id)])
            if not successful:
                telemetry.failed_result()
            return result
    setattr(AgentTool, "run_async", guarded_agent)
    _installed = True
