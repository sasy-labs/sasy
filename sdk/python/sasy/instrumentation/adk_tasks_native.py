"""Google ADK's own task tools: one agent handing work to another through
ADK's native task mechanism. Arbitrary Workflow nodes are not supported."""
from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, NoReturn

from .session import is_session_active

_installed = False
_delegation: ContextVar[Any] = ContextVar("sasy_adk_native_delegation", default=None)


def _sdk():
    from . import adk
    return adk


def _fail(message: str) -> NoReturn:
    raise _sdk().AdkInstrumentationError(message)


@dataclass
class Delegation:
    """One native task delegation: work handed to a child agent through ADK's
    own task tools, and what was observed of its run.

    (``adk_agents.Delegation`` tracks AgentTool delegation, which is a
    different mechanism.)
    """

    agent: Any
    state: Any
    inputs: list[str]
    call_id: str
    pending: list[Any] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    output: Any = None
    completed: bool = False
    failed: bool = False
    error: BaseException | None = None
    clones: list[Any] = field(default_factory=list)
    finishes: dict[str, tuple[str, bool]] = field(default_factory=dict)

    def owns(self, agent):
        return agent is self.agent or any(agent is clone for clone in self.clones)


def validate_tool(tool):
    from google.adk.agents import LlmAgent
    from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool
    from google.adk.tools.agent_tool import _SingleTurnAgentTool, _TaskAgentTool
    if type(tool) not in (_SingleTurnAgentTool, _TaskAgentTool, FinishTaskTool):
        return False
    for name in ("run_async", "process_llm_request", "_get_declaration"):
        if getattr(getattr(tool, name), "__func__", None) is not getattr(type(tool), name):
            _fail("Native delegation tool overrides require explicit instrumentation")
    if type(tool) is not FinishTaskTool:
        if type(tool.agent) is not LlmAgent or tool.name != tool.agent.name:
            _fail("Native delegation requires an exact configured LlmAgent")
        if tool.agent.input_schema or tool.agent.output_schema:
            _fail("Native delegation currently supports plain text schemas only")
        if tool.agent.parallel_worker or tool.agent.retry_config:
            _fail("Parallel or retrying native task nodes require additional qualification")
        if tool.propagate_grounding_metadata:
            _fail("Delegation grounding metadata requires explicit state provenance")
    return True


def _origin(agent, fc):
    sdk = _sdk()
    state = sdk._state()
    found = [(record, call) for record in state.records.values() if record.event.author == agent
             for call in record.event.get_function_calls() if call.id == fc.id]
    if len(found) != 1 or found[0][1].name != fc.name or sdk._record_json(found[0][1].args) != sdk._record_json(fc.args):
        _fail("Native delegation lacks its exact observed function call")
    return state, found[0][0]


@contextmanager
def _native_setup():
    from . import adk_context, adk_state, dependencies
    sdk = _sdk()
    frame = adk_state._frame.set(None)
    resolver = dependencies.set_resolver(None)
    tool = sdk._tool_context.set(None)
    try:
        with adk_context.native_run_node():
            yield
    finally:
        sdk._tool_context.reset(tool)
        dependencies.reset_resolver(resolver)
        adk_state._frame.reset(frame)


@contextmanager
def _task_span(name):
    # Deferred task dispatch bypasses ADK's ordinary execute_tool span.
    from .config import get_config
    if not get_config().otel_enabled:
        yield
        return
    from opentelemetry import trace
    with trace.get_tracer("instrumentation.adk").start_as_current_span(
            f"execute_task {name}", record_exception=False, set_status_on_exception=False) as span:
        try:
            yield
        except BaseException:
            span.set_status(trace.StatusCode.ERROR)
            raise


async def _result(state, parent, call_id, name, arguments, value, inputs, successful):
    from google.genai import types
    sdk = _sdk()
    response = value if isinstance(value, dict) else {"result": value}
    content = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=call_id, name=name, response=response))])
    derived = sdk.Tool(name=name, arguments=arguments) if successful and "error" not in response else None
    ids = await state.record(content, parent, inputs, derived=derived)
    state.results[(parent, call_id)] = ids
    return ids


async def _delegate(agent, state, parent, fc, inputs, invoke, validate):
    from . import adk_otel, dependencies
    sdk = _sdk()
    inputs = await dependencies.resolve_inputs_async(inputs)
    frozen = fc.model_copy(deep=True)
    arguments = sdk._json(frozen.args or {})
    with adk_otel.operation("tool", parent, inputs, call_id=frozen.id, tool=frozen.name) as telemetry:
        verdict = await sdk.monitor.check_tool_call_async(frozen.name, arguments, inputs,
            metadata=[("adk_agent", parent, ""), ("framework", "adk", "")])
        telemetry.decision(verdict)
        success = verdict.authorized and not verdict.transform_ids
        selected = []
        if not success:
            result = sdk._blocked(verdict, frozen.name)
        else:
            validate()
            current = Delegation(agent, state, list(inputs), frozen.id)
            token = _delegation.set(current)
            try:
                with _native_setup():
                    result = await invoke(frozen)
            finally:
                _delegation.reset(token)
            if isinstance(current.error, _sdk().AdkInstrumentationError):
                raise current.error
            if current.failed or not current.completed or not current.selected:
                _fail("Native child returned without an observed completed output")
            if sdk._record_json(result) != sdk._record_json(current.output):
                _fail("Native child return changed after its observed output")
            selected = current.selected
        ids = await _result(state, parent, frozen.id, frozen.name, arguments, result,
            list(dict.fromkeys([*inputs, *selected])), success)
        telemetry.consumed(selected)
        telemetry.produced(ids)
        if not success or (isinstance(result, dict) and "error" in result):
            telemetry.failed_result()
        return result


def install():
    global _installed
    if _installed:
        return
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool
    from google.adk.flows.llm_flows import contents
    from google.adk.tools.agent_tool import _SingleTurnAgentTool, _TaskAgentTool
    from google.adk.workflow import _llm_agent_wrapper as nodes
    from google.genai import types

    from . import adk_agents, adk_otel, adk_state
    clone = BaseAgent.clone
    @wraps(clone)
    def clone_agent(self, update=None):
        result = clone(self, update=update)
        current = _delegation.get() if is_session_active() else None
        if current is not None and current.owns(self):
            current.clones.append(result)
        return result
    setattr(BaseAgent, "clone", clone_agent)

    single = _SingleTurnAgentTool.run_async
    @wraps(single)
    async def single_turn(self, *, args, tool_context):
        if not is_session_active():
            return await single(self, args=args, tool_context=tool_context)
        sdk = _sdk()
        validate_tool(self)
        state, parent, origin = sdk._origin(tool_context)
        sdk._validate_actions(tool_context.actions)
        active = tool_context._invocation_context.agent
        if not any(c.name == self.agent.name and c.mode == self.agent.mode for c in active.sub_agents) or self.agent.mode != "single_turn":
            _fail("Single-turn target is not a configured native child")
        fc = types.FunctionCall(name=self.name, id=tool_context.function_call_id, args=copy.deepcopy(args))
        target = self.agent
        root = active.root_agent
        configured = root.find_agent(self.name)
        # Called again after the authorization check, which awaits: what is
        # dispatched must still be the agent, tool and call that were
        # authorized, and nothing may have set a transfer in the meantime.
        def validate():
            validate_tool(self)
            sdk._validate_runner(state.resources.runner)
            sdk._validate_actions(tool_context.actions)
            if (self.agent is not target or self.name != fc.name or target.mode != "single_turn"
                    or tool_context._invocation_context.agent is not active
                    or active.root_agent is not root or root.find_agent(fc.name) is not configured
                    or tool_context.actions.transfer_to_agent):
                _fail("Single-turn dispatch target changed during authorization")
        async def invoke(frozen):
            return await single(self, args=frozen.args, tool_context=tool_context)
        return await _delegate(target, state, parent, fc, adk_state.tool_inputs(origin.ids), invoke, validate)
    setattr(_SingleTurnAgentTool, "run_async", single_turn)

    canonical_tools = nodes._safe_canonical_tools_dict
    @wraps(canonical_tools)
    def task_tools(agent):
        if not is_session_active():
            return canonical_tools(agent)
        from google.adk.tools.function_tool import FunctionTool
        result = canonical_tools(agent)
        # ADK's deferred-task loop decides whether to drain ordinary tool calls
        # from this map. Match its normal FunctionTool normalization so a raw
        # Python callable is not silently skipped in a mixed delegation turn.
        for tool in agent.tools:
            if inspect.isfunction(tool):
                result = {name: value for name, value in result.items() if value is not tool}
                normalized = FunctionTool(tool)
                _sdk()._validate_tool(normalized)
                result[normalized.name] = normalized
        return result
    nodes._safe_canonical_tools_dict = task_tools

    dispatch = nodes._dispatch_task_fc
    @wraps(dispatch)
    async def task_dispatch(parent_agent, fc, ctx):
        if not is_session_active():
            return await dispatch(parent_agent, fc, ctx)
        sdk = _sdk()
        state, origin = _origin(parent_agent.name, fc)
        matches = [t for t in parent_agent.tools if type(t) is _TaskAgentTool and t.name == fc.name]
        if len(matches) != 1:
            _fail("Task delegation target is not a configured native task tool")
        tool = matches[0]
        validate_tool(tool)
        if (getattr(tool.agent, "mode", None) != "task" or not any(c.name == tool.agent.name
                and getattr(c, "mode", None) == "task" for c in parent_agent.sub_agents)):
            _fail("Task target is not a configured native child")
        sdk._validate_actions(ctx.actions)
        async def invoke(frozen):
            return await dispatch(parent_agent, frozen, ctx)
        target = parent_agent.root_agent.find_agent(fc.name)
        if target is None or getattr(target, "mode", None) != "task":
            _fail("Native task target changed before dispatch")
        root = parent_agent.root_agent
        def validate():
            validate_tool(tool)
            sdk._validate_runner(state.resources.runner)
            sdk._validate_actions(ctx.actions)
            if (not any(item is tool for item in parent_agent.tools) or tool.name != fc.name
                    or getattr(tool.agent, "mode", None) != "task"
                    or parent_agent.root_agent is not root or root.find_agent(fc.name) is not target
                    or getattr(target, "mode", None) != "task" or ctx.actions.transfer_to_agent):
                _fail("Task dispatch target changed during authorization")
        with _task_span(fc.name):
            result = await _delegate(target, state, parent_agent.name, fc, list(origin.ids), invoke, validate)
        returns = getattr(state.resources, "native_task_returns", None)
        if returns is None:
            returns = state.resources.native_task_returns = {}
        returns[id(fc)] = (fc, parent_agent.name, sdk._record_json(result), state.results[(parent_agent.name, fc.id)])
        return result
    nodes._dispatch_task_fc = task_dispatch

    synthesize = nodes._synthesize_task_fr_event
    @wraps(synthesize)
    def task_response(fc, output):
        if not is_session_active():
            return synthesize(fc, output)
        sdk = _sdk()
        state = sdk._state()
        returns = getattr(state.resources, "native_task_returns", {})
        entry = returns.pop(id(fc), None)
        if entry is None or entry[0] is not fc or entry[2] != sdk._record_json(output):
            _fail("Task response lacks its observed delegation return")
        event = synthesize(fc, output)
        pending = getattr(state.resources, "native_task_events", None)
        if pending is None:
            pending = state.resources.native_task_events = {}
        pending[event.id] = (sdk._content_key(event.content), entry[1], list(entry[3]))
        return event
    nodes._synthesize_task_fr_event = task_response

    observe = _sdk()._State.observe_event
    @wraps(observe)
    async def observe_task_response(self, event):
        pending = getattr(self.resources, "native_task_events", {})
        entry = pending.get(event.id)
        if entry is not None:
            sdk = _sdk()
            if sdk._content_key(event.content) != entry[0]:
                _fail("Native task response changed before observation")
            sdk._validate_actions(event.actions, event)
            ids = await self.record(event.content, entry[1], entry[2])
            self.records[event.id] = sdk._Record(event.model_copy(deep=True), ids, entry[0])
            del pending[event.id]
            return
        return await observe(self, event)
    _sdk()._State.observe_event = observe_task_response

    finish = FinishTaskTool.run_async
    @wraps(finish)
    async def finish_task(self, *, args, tool_context):
        if not is_session_active():
            return await finish(self, args=args, tool_context=tool_context)
        sdk = _sdk()
        validate_tool(self)
        state, agent, origin = sdk._origin(tool_context)
        current = _delegation.get() if is_session_active() else None
        if current is None or not current.owns(tool_context._invocation_context.agent) or current.agent.mode != "task":
            _fail("Task completion requires an observed native delegation")
        frozen = copy.deepcopy(args)
        profile = (self._adapter, self.output_schema, self._wrapper_key, self._task_agent_name)
        arguments = sdk._json(frozen)
        from . import dependencies
        inputs = await dependencies.resolve_inputs_async(adk_state.tool_inputs(origin.ids))
        with adk_otel.operation("tool", agent, inputs, call_id=tool_context.function_call_id, tool=self.name) as telemetry:
            verdict = await sdk.monitor.check_tool_call_async(self.name, arguments, inputs,
                metadata=[("adk_agent", agent, ""), ("framework", "adk", "")])
            telemetry.decision(verdict)
            if verdict.authorized and not verdict.transform_ids:
                validate_tool(self)
                sdk._validate_actions(tool_context.actions)
                if (self._adapter is not profile[0] or self.output_schema is not profile[1]
                        or (self._wrapper_key, self._task_agent_name) != profile[2:]
                        or tool_context.actions.transfer_to_agent):
                    _fail("Task completion target changed during authorization")
                result = await finish(self, args=frozen, tool_context=tool_context)
                success = not isinstance(result, dict) or "error" not in result
            else:
                result = sdk._blocked(verdict, self.name)
                success = False
            ids = await _result(state, agent, tool_context.function_call_id, self.name, arguments, result, inputs, success)
            current.finishes[tool_context.function_call_id] = (sdk._record_json(result if isinstance(result, dict) else {"result": result}), success)
            telemetry.produced(ids)
            if not success:
                telemetry.failed_result()
            return result
    setattr(FinishTaskTool, "run_async", finish_task)

    prepare = nodes.prepare_llm_agent_input
    @wraps(prepare)
    def prepare_input(agent, ctx, node_input):
        current = _delegation.get() if is_session_active() else None
        before = {event.id for event in ctx._invocation_context.session.events}
        prepare(agent, ctx, node_input)
        if current is not None and current.owns(agent):
            current.pending.extend(event for event in ctx._invocation_context.session.events if event.id not in before)
    nodes.prepare_llm_agent_input = prepare_input

    process = contents._ContentLlmRequestProcessor.run_async
    @wraps(process)
    async def flush_inputs(self, invocation_context, llm_request):
        current = _delegation.get() if is_session_active() else None
        if current is not None and current.owns(invocation_context.agent):
            sdk = _sdk()
            for event in current.pending:
                ids = await current.state.record(event.content, current.agent.name, current.inputs)
                current.state.records[event.id] = sdk._Record(event.model_copy(deep=True), ids, sdk._content_key(event.content))
            current.pending.clear()
        iterator = process(self, invocation_context, llm_request)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()
    setattr(contents._ContentLlmRequestProcessor, "run_async", flush_inputs)

    task_input = contents._build_task_input_user_content
    @wraps(task_input)
    def build_task_input(all_events, isolation_scope, is_single_turn=False, user_content=None):
        result = task_input(all_events, isolation_scope, is_single_turn, user_content)
        current = _delegation.get() if is_session_active() else None
        if current is not None and result is not None:
            if isolation_scope != current.call_id:
                _fail("Native task input has a different delegation scope")
            adk_agents.capture_input(result, current.inputs)
        return result
    contents._build_task_input_user_content = build_task_input

    run = nodes.run_llm_agent_as_node
    @wraps(run)
    async def observed_node(agent, *, ctx, node_input):
        current = _delegation.get() if is_session_active() else None
        tracked = current is not None and current.owns(agent)
        iterator = run(agent, ctx=ctx, node_input=node_input)
        try:
            async for event in iterator:
                if current is not None and tracked and event.output is not None:
                    if current.agent.mode == "task":
                        responses = event.get_function_responses()
                        if len(responses) != 1 or responses[0].name != "finish_task":
                            _fail("Task output lacks an observed completion response")
                        expected = current.finishes.get(responses[0].id)
                        if expected is None or not expected[1] or expected[0] != _sdk()._record_json(responses[0].response):
                            _fail("Task completion response changed or was not authorized")
                    record = current.state.records.get(event.id)
                    if record is None:
                        _fail("Native node output has no observed message")
                    current.output = copy.deepcopy(event.output)
                    current.selected = list(record.ids)
                yield event
            if current is not None and tracked:
                current.completed = True
        except BaseException as error:
            if current is not None and tracked:
                current.failed = True
                current.error = error
            raise
        finally:
            await iterator.aclose()
    nodes.run_llm_agent_as_node = observed_node
    _installed = True
