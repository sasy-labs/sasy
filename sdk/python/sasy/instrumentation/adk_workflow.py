"""Provenance for ADK 2.9.1's static LlmAgent and JoinNode Workflow graphs.

ADK passes a node's output to its successor through an in-memory user event.
That event is not yielded by Runner and has no session-service record. Bind it
at the native scheduling boundary to the exact output event of the predecessor,
then record it before the child's model request reads session history.
"""
from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, NoReturn, cast

_installed = False


def _sdk():
    from . import adk
    return adk


def _fail(message: str) -> NoReturn:
    raise _sdk().AdkInstrumentationError(message)


def _input_key(value: Any) -> str:
    return _sdk()._record_json(value)


def is_workflow(root: Any) -> bool:
    from google.adk.workflow import Workflow
    return isinstance(root, Workflow)


def _graph_signature(workflow: Any) -> tuple:
    graph = workflow.graph
    return (id(graph), tuple((id(node), node.name) for node in graph.nodes),
            tuple((id(edge), id(edge.from_node), id(edge.to_node), edge.route)
                  for edge in graph.edges),
            tuple(sorted(graph._terminal_node_names)),
            tuple((node.name, type(node),
                   tuple((field, id(getattr(node, field))) for field in (
                       "retry_config", "timeout", "input_schema", "output_schema",
                       "state_schema", "wait_for_output")),
                   id(getattr(node, "model", None)), getattr(node, "mode", None),
                   tuple(id(tool) for tool in getattr(node, "tools", ())),
                   getattr(node, "output_key", None), getattr(node, "include_contents", None),
                   tuple(id(getattr(node, field)) for field in (
                       "instruction", "global_instruction", "static_instruction")
                         if hasattr(node, field))) for node in graph.nodes))


def _same_agent_execution(node: Any, configured: Any) -> bool:
    from google.adk.agents import LlmAgent
    if type(node) is not LlmAgent or type(configured) is not LlmAgent:
        return False
    if (node.model is not configured.model or node.mode != configured.mode
            or node.output_key != configured.output_key
            or node.include_contents not in (configured.include_contents, "none")
            or len(node.tools) != len(configured.tools)
            or any(a is not b for a, b in zip(node.tools, configured.tools))):
        return False
    for field in ("instruction", "global_instruction", "static_instruction", "generate_content_config"):
        a, b = getattr(node, field), getattr(configured, field)
        if a is not b and (callable(a) or callable(b) or a != b):
            return False
    return all(getattr(node, field) is getattr(configured, field)
               for field in type(node).model_fields if "callback" in field)


@dataclass
class _Plan:
    agents: list[Any]
    predecessors: dict[str, tuple[str, ...]]
    nodes: dict[str, Any]
    route: _RoutePlan | None = None
    parallel_nodes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _RoutePlan:
    router: str
    targets: dict[str, str]
    branches: dict[str, tuple[str, ...]]
    finalizer: str
    tool: str


def _plan(workflow: Any) -> _Plan:
    """Qualify an acyclic graph with explicit barriers at every merge."""
    from google.adk.agents import LlmAgent
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.workflow import JoinNode, Workflow
    from google.adk.workflow._base_node import START
    from google.adk.workflow._graph import DEFAULT_ROUTE, Edge, Graph

    if type(workflow) is not Workflow or type(workflow.graph) is not Graph:
        _fail("Only a standard static Workflow graph is supported")
    if (workflow.retry_config is not None or workflow.timeout is not None
            or workflow.input_schema is not None or workflow.output_schema is not None
            or workflow.state_schema is not None or workflow.wait_for_output
            or workflow.max_concurrency is not None):
        _fail("Workflow retries, timeouts, schemas and wait-for-output need explicit provenance")
    graph = workflow.graph
    edges = graph.edges
    if not edges or any(type(edge) is not Edge for edge in edges):
        _fail("Workflow requires standard static edges")
    by_name = {node.name: node for node in graph.nodes}
    if len(by_name) != len(graph.nodes) or by_name.get(START.name) is not START:
        _fail("Workflow node names must be unique and include START")
    successors: dict[str, list[str]] = {name: [] for name in by_name}
    predecessors: dict[str, list[str]] = {name: [] for name in by_name}
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        source, target = edge.from_node.name, edge.to_node.name
        if (by_name.get(source) is not edge.from_node or by_name.get(target) is not edge.to_node
                or target == START.name or (source, target) in seen_edges):
            _fail("Workflow requires unique static edges and node identities")
        seen_edges.add((source, target))
        successors[source].append(target)
        predecessors[target].append(source)
    ready = [START.name]
    ordered: list[str] = []
    remaining = {name: len(parents) for name, parents in predecessors.items()}
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for target in successors[current]:
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    if len(ordered) != len(graph.nodes) or any(name != START.name and not predecessors[name]
                                               for name in by_name):
        _fail("Workflow requires a connected acyclic static graph")
    terminals = {name for name, targets in successors.items() if not targets}
    if len(terminals) != 1 or terminals != graph._terminal_node_names:
        _fail("Workflow requires one validated terminal node")
    agents = []
    ordinary_merges = []
    for name in ordered[1:]:
        node = by_name[name]
        parents = predecessors[name]
        if type(node) is JoinNode:
            if (len(parents) < 2 or START.name in parents or node.retry_config is not None
                    or node.timeout is not None or node.input_schema is not None
                    or node.output_schema is not None or node.state_schema is not None
                    or node.wait_for_output):
                _fail("JoinNode requires at least two completed agent or join predecessors")
            continue
        if type(node) is not LlmAgent:
            _fail("Workflow nodes must be standard single-turn LlmAgents or JoinNodes")
        if len(parents) != 1:
            ordinary_merges.append(name)
        agents.append(node)
    for agent in agents:
        if agent.mode != "single_turn" or agent.sub_agents:
            _fail("Workflow nodes must be standard single-turn LlmAgents without sub-agents")
        if (agent.retry_config is not None or agent.timeout is not None or agent.parallel_worker
                or agent.wait_for_output or agent.input_schema is not None
                or agent.output_schema is not None):
            _fail("Workflow node retries, parallel workers and schemas need explicit provenance")
        if any(getattr(agent, field) for field in type(agent).model_fields if "callback" in field):
            _fail("Workflow callbacks can read or mutate an unrecorded handoff")
    forks = [name for name, targets in successors.items() if len(targets) > 1]
    if len(forks) > 1:
        _fail("Workflow requires one qualified fork")
    routed = [edge for edge in edges if edge.route is not None]
    route_plan = None
    if routed:
        if (len(forks) != 1 or len(ordinary_merges) != 1
                or any(type(node) is JoinNode for node in by_name.values())):
            _fail("Conditional Workflow requires one exclusive router and ordinary finalizer")
        router = forks[0]
        if predecessors[router] != [START.name]:
            _fail("Conditional Workflow router must follow START")
        if (type(by_name[router]) is not LlmAgent or len(successors[router]) < 2
                or len(routed) != len(successors[router])
                or any(edge.from_node.name != router for edge in routed)
                or any(type(edge.route) is not str or not edge.route or edge.route == DEFAULT_ROUTE
                       for edge in routed)
                or len({edge.route for edge in routed}) != len(routed)):
            _fail("Conditional Workflow requires at least two distinct string routes")
        targets = {cast(str, edge.route): edge.to_node.name for edge in routed}
        if set(targets.values()) != set(successors[router]):
            _fail("Conditional Workflow routes must select one branch each")
        finalizer = ordinary_merges[0]
        branches = {}
        seen_branch_nodes = set()
        for route, target in targets.items():
            previous, current = router, target
            chain = []
            while current != finalizer:
                if (type(by_name[current]) is not LlmAgent or current in seen_branch_nodes
                        or predecessors[current] != [previous] or len(successors[current]) != 1):
                    _fail("Conditional Workflow branches must be disjoint agent chains")
                seen_branch_nodes.add(current)
                chain.append(current)
                previous, current = current, successors[current][0]
            if not chain:
                _fail("Conditional Workflow route cannot skip its handler")
            branches[route] = tuple(chain)
        if (set(predecessors[finalizer]) != {chain[-1] for chain in branches.values()}
                or len(predecessors[finalizer]) != len(targets)):
            _fail("Conditional Workflow finalizer requires exactly one selected predecessor")
        route_agent = cast(LlmAgent, by_name[router])
        if len(route_agent.tools) != 1:
            _fail("Conditional Workflow router requires one FunctionTool")
        configured_tool = route_agent.tools[0]
        wrapped = FunctionTool(configured_tool) if inspect.isfunction(configured_tool) else configured_tool
        if (type(wrapped) is not FunctionTool
                or wrapped._context_param_name not in inspect.signature(wrapped.func).parameters):
            _fail("Conditional Workflow route requires an injected ToolContext")
        for agent in agents:
            if "include_contents" in agent.model_fields_set and agent.include_contents == "default":
                _fail("Conditional Workflow agents must use current-turn contents")
        route_plan = _RoutePlan(router, targets, branches, finalizer, wrapped.name)
    elif ordinary_merges:
        _fail("Multiple Workflow predecessors require an explicit JoinNode")
    branch_nodes: set[str] = set()
    if forks and not routed:
        fork = forks[0]
        joins = set()
        branch_ends = set()
        for target in successors[fork]:
            if type(by_name[target]) is not LlmAgent:
                _fail("Parallel Workflow branches must begin with standard LlmAgents")
            previous, current = fork, target
            while type(by_name[current]) is LlmAgent:
                if (current in branch_nodes or predecessors[current] != [previous]
                        or len(successors[current]) != 1):
                    _fail("Parallel Workflow branches must be disjoint agent chains")
                branch_nodes.add(current)
                previous, current = current, successors[current][0]
            if type(by_name[current]) is not JoinNode:
                _fail("Parallel Workflow branches require a JoinNode barrier")
            joins.add(current)
            branch_ends.add(previous)
        if len(joins) != 1 or set(predecessors[next(iter(joins))]) != branch_ends:
            _fail("Parallel Workflow JoinNode must wait for every branch")
        if any(type(by_name[name]) is JoinNode and name not in joins for name in by_name):
            _fail("Parallel Workflow requires one complete JoinNode barrier")
    # Only branch chains overlap. Prefix and post-barrier agents use the same
    # ordered resource transport as serial Workflows.
    if forks and not routed:
        from google.adk.utils import instructions_utils

        for agent in agents:
            if agent.name not in branch_nodes:
                continue
            if agent.output_key:
                _fail("Parallel Workflow tools cannot access ADK state or artifacts")
            instructions = (agent.instruction, agent.global_instruction, agent.static_instruction)
            if any(callable(value) or (isinstance(value, str) and any(
                    not match.group().lstrip("{").rstrip("}").strip().removesuffix("?").startswith("artifact.")
                    for match in instructions_utils._TEMPLATE_VAR_PATTERN.finditer(value)))
                   for value in instructions):
                _fail("Parallel Workflow tools cannot access ADK state or artifacts")
            for tool in agent.tools:
                wrapped = FunctionTool(tool) if inspect.isfunction(tool) else tool
                # ToolContext can carry immutable, versioned artifact reads and
                # writes. Mutable state remains guarded at each actual access.
                if type(wrapped) is not FunctionTool:
                    _fail("Parallel Workflow tools cannot access ADK state or artifacts")
    return _Plan(agents, {name: tuple(parents) for name, parents in predecessors.items()},
                 by_name, route_plan, frozenset(branch_nodes))


def validate(workflow: Any) -> list[Any]:
    """Return the qualified LlmAgents for the runner's normal checks."""
    return _plan(workflow).agents


def validate_route_action(state: Any, actions: Any, event: Any) -> None:
    """Bind one native route action to one successful, guarded tool result."""
    if state is None or state.resources is None:
        _fail("Workflow route has no active observed run")
    plan = getattr(state.resources, "workflow_route_plan", None)
    route = actions.route
    if plan is None or type(route) is not str or route not in plan.targets:
        _fail("Workflow route is not a qualified exclusive string decision")
    if event is None:
        context = _sdk()._tool_context.get()
        if (context is None or context.actions is not actions
                or context._invocation_context.agent.name != plan.router
                or not context.function_call_id
                or (state.resources.workflow_route_call is None
                    or state.resources.workflow_route_call[1] != context.function_call_id)):
            _fail("Workflow route has no guarded router tool call")
        _, agent, origin = _sdk()._origin(context)
        if (agent != plan.router
                or origin.event.id != state.resources.workflow_route_call[0]):
            _fail("Workflow route tool origin changed")
        ids = tuple(state.results.get((agent, context.function_call_id), ()))
        if (len(ids) != 1 or ids[0] not in state.snapshots
                or not state.snapshots[ids[0]].HasField("derived_from")
                or state.snapshots[ids[0]].derived_from.name != plan.tool):
            _fail("Workflow route requires one authorized successful tool result")
        candidate = (context.function_call_id, route, ids)
        previous = state.resources.workflow_route_candidate
        if previous is not None and previous != candidate:
            _fail("Workflow router emitted more than one decision")
        state.resources.workflow_route_candidate = candidate
        return
    candidate = state.resources.workflow_route_candidate
    responses = event.get_function_responses()
    if (candidate is None or event.author != plan.router
            or event.actions is not actions or len(responses) != 1
            or responses[0].id != candidate[0] or responses[0].name != plan.tool
            or route != candidate[1] or event.content is None
            or len(event.content.parts or ()) != 1):
        _fail("Workflow route event does not match its authorized tool result")
    # ADK's state adapter may add observed state reads to the function response
    # after FunctionTool.run_async returns. The emitted event uses this final
    # version of the result, which retains the successful tool result as an
    # ancestor and is the exact source recorded by _State.observe_event.
    final_ids = tuple(state.results.get((plan.router, candidate[0]), ()))
    if not final_ids or any(node not in state.snapshots for node in final_ids):
        _fail("Workflow route lost its observed tool result")
    candidate = (candidate[0], candidate[1], final_ids)
    frozen = (event, event.id, _sdk()._content_key(event.content),
              _input_key(actions.model_dump(exclude_none=True)), candidate)
    previous = state.resources.workflow_route_event
    if previous is not None and previous[1:] != frozen[1:]:
        _fail("Workflow router emitted more than one route event")
    state.resources.workflow_route_event = frozen


@dataclass(frozen=True)
class _Source:
    agent: str
    key: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class _RouteControl:
    route: str
    target: str
    ids: tuple[str, ...]
    event: Any
    event_id: str
    content_key: str
    actions_key: str
    call_id: str
    result_ids: tuple[str, ...]


_scheduled: ContextVar[_Source | None] = ContextVar("sasy_adk_workflow_scheduled", default=None)
_resource_execution: ContextVar[tuple[Any, bool] | None] = ContextVar(
    "sasy_adk_workflow_resource_execution", default=None)
_current_child: ContextVar[str | None] = ContextVar("sasy_adk_workflow_child", default=None)
_pending: ContextVar[tuple[Any, _Source, str, str, str] | None] = ContextVar(
    "sasy_adk_workflow_pending", default=None)


def check_resource_access(owner: Any, *, artifact: bool = False) -> None:
    """Bind resource access to the scheduled execution, never mutable agent names."""
    execution = _resource_execution.get()
    if execution is None or execution[0] is not owner:
        _fail("Workflow resource access lacks its scheduled execution")
    assert execution is not None
    if execution[1] and not artifact:
        _fail("Parallel Workflow tools cannot access ADK state or artifacts")


def prepare_run(state: Any, user_event: Any) -> None:
    plan = _plan(state.resources.runner.agent)
    record = state.records.get(user_event.id)
    if record is None or not record.ids or record.key != _sdk()._content_key(user_event.content):
        _fail("Workflow root input has no observed user event")
    state.resources.workflow_root = (_input_key(user_event.content), tuple(record.ids))
    state.resources.workflow_root_content_key = record.key
    state.resources.workflow_root_event = user_event.id
    state.resources.workflow_outputs = {}
    state.resources.workflow_join_events = {}
    state.resources.workflow_join_outputs = {}
    state.resources.workflow_started = set()
    state.resources.workflow_graph_signature = _graph_signature(state.resources.runner.agent)
    state.resources.workflow_resources_guarded = True
    state.resources.workflow_route_plan = plan.route
    state.resources.workflow_route_call = None
    state.resources.workflow_route_candidate = None
    state.resources.workflow_route_event = None
    state.resources.workflow_route_control = None


def observed_output(predecessor: str, loop_state: Any, ctx: Any, state: Any) -> tuple[Any, tuple[str, ...]]:
    """Resolve one completed predecessor by its native run, never by text alone."""
    from google.adk.workflow import JoinNode
    from google.adk.workflow._node_status import NodeStatus

    run = loop_state.nodes.get(predecessor)
    if (run is None or run.status != NodeStatus.COMPLETED or run.run_id != "1"
            or predecessor not in loop_state.node_outputs):
        _fail("Workflow predecessor did not complete its observed run")
    path = f"{ctx.node_path}/{predecessor}@{run.run_id}"
    if type(next(node for node in state.resources.runner.agent.graph.nodes
                 if node.name == predecessor)) is JoinNode:
        output = state.resources.workflow_join_outputs.get(path)
        if output is None:
            _fail("Workflow JoinNode has no observed output")
        event, key, ids, branch = output
        if (_input_key(event.output) != key or
                _input_key(loop_state.node_outputs[predecessor]) != key or not ids
                or loop_state.node_branches.get(predecessor) != (branch or "")):
            _fail("Workflow JoinNode output changed after observation")
        return event.output, ids
    output = state.resources.workflow_outputs.get(path)
    if output is None:
        _fail("Workflow predecessor has no observed output")
    event_id, content, key, ids, content_key, branch = output
    record = state.records.get(event_id)
    if (record is None or record.event.author != predecessor
            or record.key != content_key or tuple(record.ids) != ids
            or _input_key(content) != key
            or _input_key(loop_state.node_outputs[predecessor]) != key
            or loop_state.node_branches.get(predecessor) != (branch or "")):
        _fail("Workflow predecessor output changed after observation")
    return content, ids


def observed_route(state: Any, plan: _RoutePlan) -> _RouteControl:
    control = state.resources.workflow_route_control
    if control is None or control.route not in plan.targets or control.target != plan.targets[control.route]:
        _fail("Workflow selected route has no observed decision")
    event = control.event
    record = state.records.get(control.event_id)
    if (event.id != control.event_id or event.author != plan.router
            or event.content is None or _sdk()._content_key(event.content) != control.content_key
            or event.actions is None or event.actions.route != control.route
            or _input_key(event.actions.model_dump(exclude_none=True)) != control.actions_key
            or record is None or record.key != control.content_key or not record.ids
            or tuple(state.results.get((plan.router, control.call_id), ())) != control.result_ids
            or any(node not in state.snapshots for node in control.ids)):
        _fail("Workflow selected route changed after observation")
    return control


def install() -> None:
    global _installed
    if _installed:
        return
    from google.adk.agents import LlmAgent
    from google.adk.flows.llm_flows import contents
    from google.adk.workflow import JoinNode, Workflow
    from google.adk.workflow import _llm_agent_wrapper as wrapper
    from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
    from google.adk.workflow._node_runner import NodeRunner

    track = NodeRunner._track_event_in_context

    @wraps(track)
    def track_output(self, event, ctx):
        state = _sdk()._active.get() if _sdk().is_session_active() else None
        if state is not None and is_workflow(state.resources.runner.agent):
            route_plan = state.resources.workflow_route_plan
            if (route_plan is not None and type(self._node) is LlmAgent
                    and self._node.name == route_plan.router
                    and event.author == route_plan.router and not event.partial):
                calls = event.get_function_calls()
                if calls and (len(calls) != 1 or calls[0].name != route_plan.tool
                              or state.resources.workflow_route_call is not None
                              or state.resources.workflow_route_candidate is not None):
                    _fail("Workflow router requires one tool call and one route decision")
                if calls:
                    state.resources.workflow_route_call = (event.id, calls[0].id)
        track(self, event, ctx)
        sdk = _sdk()
        if state is None or not is_workflow(state.resources.runner.agent):
            return
        if _graph_signature(state.resources.runner.agent) != state.resources.workflow_graph_signature:
            _fail("Workflow graph changed after qualification")
        if type(self._node) is JoinNode and event.output is not None:
            if (event.content is not None or event.actions and (
                    event.actions.route is not None or event.actions.state_delta
                    or event.actions.artifact_delta)):
                _fail("JoinNode emitted an unqualified event")
            path = ctx.node_path
            if path in state.resources.workflow_join_events:
                _fail("JoinNode emitted more than one output")
            state.resources.workflow_join_events[path] = (event, _input_key(event.output))
            return
        if type(self._node) is not LlmAgent or not event.node_info.message_as_output:
            return
        workflow = state.resources.runner.agent
        configured = next((node for node in workflow.graph.nodes if node.name == self._node.name), None)
        if not _same_agent_execution(self._node, configured):
            _fail("Workflow executed a node outside its validated graph")
        if event.author != self._node.name or event.content is None or event.content.role != "model":
            _fail("Workflow output has no standard model event")
        record = state.records.get(event.id)
        if record is None or record.key != sdk._content_key(event.content) or not record.ids:
            _fail("Workflow output has no exact observed model event")
        expected = "".join(part.text or "" for part in event.content.parts or []
                           if not part.thought)
        if (type(ctx.output) is not str or event.output != ctx.output
                or ctx.output != expected):
            _fail("Workflow output differs from the observed model event")
        state.resources.workflow_outputs[ctx.node_path] = (
            event.id, ctx.output, _input_key(ctx.output), tuple(record.ids), record.key,
            ctx._invocation_context.branch)

    NodeRunner._track_event_in_context = track_output  # type: ignore[method-assign]

    start = Workflow._start_node_task

    @wraps(start)
    def start_node(self, loop_state, ctx, node_name, trigger):
        from google.adk.workflow._base_node import START
        sdk = _sdk()
        state = sdk._active.get() if sdk.is_session_active() else None
        if state is None:
            return start(self, loop_state, ctx, node_name, trigger)
        root = state.resources.runner.agent
        if (self.graph is not root.graph or self.name != root.name
                or "/" in ctx.node_path
                or _graph_signature(root) != state.resources.workflow_graph_signature):
            _fail("Nested or dynamically scheduled Workflows need explicit provenance")
        plan = _plan(self)
        if (node_name not in plan.predecessors or node_name == START.name
                or loop_state.recovered_executions or node_name in state.resources.workflow_started
                or loop_state.nodes[node_name].run_counter != 0):
            _fail("Workflow scheduling is outside the supported fresh static graph")
        route_plan = plan.route
        route_control = None
        if route_plan is not None and node_name != route_plan.router:
            route_control = observed_route(state, route_plan)
            unselected = {name for route, chain in route_plan.branches.items()
                          if route != route_control.route for name in chain}
            if node_name in unselected or unselected & state.resources.workflow_started:
                _fail("Workflow scheduled an unselected conditional branch")
        parents = plan.predecessors[node_name]
        if parents == (START.name,):
            key, ids = state.resources.workflow_root
            expected_branch = None
            use_sub_branch = sum(edge.from_node is START for edge in self.graph.edges) > 1
        elif type(plan.nodes[node_name]) is JoinNode:
            values = {}
            all_ids: list[str] = []
            for predecessor in parents:
                value, observed_ids = observed_output(predecessor, loop_state, ctx, state)
                values[predecessor] = value
                all_ids.extend(observed_ids)
            key, ids = _input_key(values), tuple(dict.fromkeys(all_ids))
            from google.adk.workflow._workflow import get_common_branch_prefix
            expected_branch = get_common_branch_prefix(
                [loop_state.node_branches[parent] for parent in parents])
            use_sub_branch = False
        else:
            if len(parents) > 1:
                if (route_plan is None or node_name != route_plan.finalizer
                        or route_control is None):
                    _fail("Workflow merge has no exclusive selected predecessor")
                predecessor = route_plan.branches[route_control.route][-1]
                if (predecessor not in parents or any(
                        name in state.resources.workflow_started
                        for name in parents if name != predecessor)):
                    _fail("Workflow finalizer has ambiguous branch execution")
            else:
                predecessor = parents[0]
            content, ids = observed_output(predecessor, loop_state, ctx, state)
            key = _input_key(content)
            if route_plan is not None and predecessor == route_plan.router:
                if route_control is None or node_name != route_control.target:
                    _fail("Workflow router selected a different branch")
                ids = tuple(dict.fromkeys((*ids, *route_control.ids)))
            run = loop_state.nodes[predecessor]
            path = f"{ctx.node_path}/{predecessor}@{run.run_id}"
            output = (state.resources.workflow_join_outputs.get(path)
                      if type(plan.nodes[predecessor]) is JoinNode else
                      state.resources.workflow_outputs.get(path))
            expected_branch = output[-1]
            use_sub_branch = (False if route_plan is not None and predecessor == route_plan.router
                              else sum(edge.from_node.name == predecessor for edge in self.graph.edges) > 1)
        if trigger.input is None or _input_key(trigger.input) != key:
            _fail("Workflow trigger input differs from its observed predecessor")
        if (trigger.branch != expected_branch or trigger.use_sub_branch != use_sub_branch
                or trigger.isolation_scope is not None):
            _fail("Workflow trigger branch differs from its observed predecessor")
        source = _Source(node_name, key, tuple(ids))
        state.resources.workflow_started.add(node_name)
        token = _scheduled.set(source)
        resource_token = _resource_execution.set((state.resources, node_name in plan.parallel_nodes))
        try:
            return start(self, loop_state, ctx, node_name, trigger)
        finally:
            _resource_execution.reset(resource_token)
            _scheduled.reset(token)

    Workflow._start_node_task = start_node  # type: ignore[method-assign]

    schedule = DynamicNodeScheduler.__call__

    @wraps(schedule)
    async def schedule_node(self, ctx, node, node_input, **kwargs):
        sdk = _sdk()
        state = sdk._active.get() if sdk.is_session_active() else None
        if state is not None and is_workflow(state.resources.runner.agent):
            root = state.resources.runner.agent
            if _graph_signature(root) != state.resources.workflow_graph_signature:
                _fail("Workflow graph changed after qualification")
            if type(node) is Workflow and not ctx.node_path:
                validate(root)
                if node.graph is not root.graph or node.name != root.name:
                    _fail("Workflow root changed before native dispatch")
            else:
                source = _scheduled.get() if _sdk().is_session_active() else None
                if (source is None or "/" in ctx.node_path or type(node) not in (LlmAgent, JoinNode)
                        or node.name != source.agent or _input_key(node_input) != source.key):
                    _fail("Unqualified dynamic Workflow node dispatch")
                configured = _plan(root).nodes.get(node.name)
                if type(node) is JoinNode:
                    if node is not configured:
                        _fail("Workflow JoinNode configuration changed before dispatch")
                elif not _same_agent_execution(node, configured):
                    _fail("Workflow child configuration changed before dispatch")
        result = await schedule(self, ctx, node, node_input, **kwargs)
        if state is not None and is_workflow(state.resources.runner.agent):
            if _graph_signature(state.resources.runner.agent) != state.resources.workflow_graph_signature:
                _fail("Workflow graph changed during dispatch")
        if state is not None and is_workflow(state.resources.runner.agent) and type(node) is JoinNode:
            source = _scheduled.get() if _sdk().is_session_active() else None
            path = result.node_path
            observed = state.resources.workflow_join_events.get(path)
            if (source is None or observed is None or source.agent != node.name
                    or path in state.resources.workflow_join_outputs):
                _fail("JoinNode has no unique native output")
            event, output_key = observed
            if (event.id is None or _input_key(event.output) != source.key
                    or output_key != source.key or _input_key(result.output) != source.key):
                _fail("JoinNode output differs from its observed predecessor values")
            from google.genai import types

            from sasy.proto.observability_pb2 import Role
            content = types.Content(role="model", parts=[types.Part(text=output_key)])
            ids = tuple(await state.record(content, node.name, list(source.ids), role=Role.AGENT))
            state.resources.workflow_join_outputs[path] = (
                event, output_key, ids, result._invocation_context.branch)
        if state is not None and is_workflow(state.resources.runner.agent):
            route_plan = state.resources.workflow_route_plan
            if route_plan is not None and node.name == route_plan.router:
                observed = state.resources.workflow_route_event
                if (observed is None or state.resources.workflow_route_control is not None
                        or type(node) is not LlmAgent):
                    _fail("Workflow router has no unique native route event")
                event, event_id, content_key, actions_key, candidate = observed
                call_id, route, result_ids = candidate
                record = state.records.get(event_id)
                output = state.resources.workflow_outputs.get(result.node_path)
                if (result.route != route or event.id != event_id or event.author != route_plan.router
                        or event.actions is None or event.actions.route != route
                        or _input_key(event.actions.model_dump(exclude_none=True)) != actions_key
                        or event.content is None or _sdk()._content_key(event.content) != content_key
                        or record is None or record.key != content_key or not record.ids
                        or tuple(state.results.get((route_plan.router, call_id), ())) != result_ids
                        or output is None or output[2] != _input_key(result.output)):
                    _fail("Workflow router decision differs from its observed tool event")
                from google.genai import types

                from sasy.proto.observability_pb2 import Role

                content = types.Content(role="model", parts=[types.Part(text=_input_key({
                    "route": route, "call_id": call_id}))])
                ids = tuple(await state.record(content, route_plan.router,
                                               list(dict.fromkeys((*record.ids, *result_ids))),
                                               role=Role.AGENT))
                state.resources.workflow_route_control = _RouteControl(
                    route, route_plan.targets[route], ids, event, event_id,
                    content_key, actions_key, call_id, result_ids)
        return result

    DynamicNodeScheduler.__call__ = schedule_node  # type: ignore[method-assign]

    prepare = wrapper.prepare_llm_agent_input

    @wraps(prepare)
    def prepare_input(agent, ctx, node_input):
        source = _scheduled.get() if _sdk().is_session_active() else None
        if source is None:
            return prepare(agent, ctx, node_input)
        if agent.name != source.agent or _input_key(node_input) != source.key:
            _fail("Workflow child received a different input than its native trigger")
        state = _sdk()._state()
        temporary = getattr(state.resources, "workflow_root_event", None)
        if temporary is not None:
            native = [event for event in ctx._invocation_context.session.events
                      if event.author == "user" and event.id not in state.records
                      and event.content is not None
                      and _sdk()._content_key(event.content) == state.resources.workflow_root_content_key]
            if len(native) != 1:
                _fail("Workflow root message does not match one native user event")
            record = state.records.pop(temporary, None)
            if record is None:
                _fail("Workflow root message lost its observed source")
            state.records[native[0].id] = _sdk()._Record(
                native[0].model_copy(deep=True), record.ids, record.key)
            _sdk()._remember_history(state, native[0])
            state.resources.workflow_root_event = None
        before = {event.id for event in ctx._invocation_context.session.events}
        prepare(agent, ctx, node_input)
        added = [event for event in ctx._invocation_context.session.events if event.id not in before]
        if (len(added) != 1 or added[0].author != "user" or added[0].content is None
                or _pending.get() is not None):
            _fail("Workflow child did not receive exactly one native input event")
        from google.adk.utils.content_utils import to_user_content
        expected = to_user_content(node_input)
        expected.role = "user"
        content_key = _sdk()._content_key(added[0].content)
        if content_key != _sdk()._content_key(expected):
            _fail("Workflow native handoff differs from its scheduled input")
        if added[0].id in state.records:
            _fail("Workflow native handoff reused an observed event identity")
        # The sibling branch can scan session.events while this branch awaits
        # observation. Bind the exact native event immediately to its source
        # IDs; the async observation below replaces the provisional IDs before
        # this branch builds its model request.
        state.records[added[0].id] = _sdk()._Record(
            added[0].model_copy(deep=True), list(source.ids), content_key)
        _pending.set((added[0], source, added[0].id, content_key,
                      _sdk()._history_key(added[0])))
        _current_child.set(source.agent)
        _scheduled.set(None)

    wrapper.prepare_llm_agent_input = prepare_input

    process = contents._ContentLlmRequestProcessor.run_async

    @wraps(process)
    async def record_input(self, invocation_context, llm_request):
        child = _current_child.get() if _sdk().is_session_active() else None
        if child is not None:
            state = _sdk()._state()
            root = state.resources.runner.agent
            configured = _plan(root).nodes.get(child)
            if (invocation_context.agent.name != child
                    or _graph_signature(root) != state.resources.workflow_graph_signature
                    or not _same_agent_execution(invocation_context.agent, configured)):
                _fail("Workflow child configuration changed before model request")
        pending = _pending.get() if _sdk().is_session_active() else None
        if pending is not None:
            event, source, event_id, content_key, event_key = pending
            state = _sdk()._state()
            if (invocation_context.agent.name != source.agent
                    or not any(item is event for item in invocation_context.session.events)
                    or event.id != event_id or event.content is None
                    or _sdk()._content_key(event.content) != content_key
                    or _sdk()._history_key(event) != event_key):
                _fail("Workflow handoff changed before its intended agent consumed it")
            provisional = state.records.get(event_id)
            if (provisional is None or tuple(provisional.ids) != source.ids
                    or provisional.key != content_key
                    or provisional.event.id != event_id):
                _fail("Workflow handoff lost its identity-bound source")
            try:
                ids = await state.record(event.content, source.agent, list(source.ids))
            except BaseException:
                state.records.pop(event_id, None)
                _pending.set(None)
                raise
            state.records[event.id] = _sdk()._Record(
                event.model_copy(deep=True), ids, _sdk()._content_key(event.content))
            _pending.set(None)
        iterator = process(self, invocation_context, llm_request)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()

    contents._ContentLlmRequestProcessor.run_async = record_input  # type: ignore[method-assign]
    _installed = True
