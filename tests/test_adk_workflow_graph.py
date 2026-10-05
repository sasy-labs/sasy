"""Qualification cases for broader native ADK Workflow graph support."""
import asyncio

import pytest
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import JoinNode, Workflow
from google.adk.workflow._base_node import START
from google.adk.workflow._graph import Edge
from sasy.instrumentation import adk
from test_adk_instrumentation import (  # noqa: F401
    AdkInstrumentationError,
    InMemoryRunner,
    ScriptedModel,
    calls,
    conversation,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


def _run(workflow):
    return asyncio.run(conversation(workflow, runner=InMemoryRunner(node=workflow, app_name="test")))


def _joined_workflow(*, left_tool=None, right_tool=None):
    def left_default():
        """Read left branch."""
        return {"left": "confidential"}

    def right_default():
        """Read right branch."""
        return {"right": "public"}

    left_tool = left_tool or left_default
    right_tool = right_tool or right_default

    def pay():
        """Pay after joined review."""
        return {"paid": True}

    left = LlmAgent(name="left", model=ScriptedModel([
        calls((left_tool.__name__, {}, "left-call")), text("same final text")]), tools=[left_tool])
    right = LlmAgent(name="right", model=ScriptedModel([
        calls((right_tool.__name__, {}, "right-call")), text("same final text")]), tools=[right_tool])
    payer = LlmAgent(name="payer", model=ScriptedModel([
        calls(("pay", {}, "pay-call")), text("paid")]), tools=[pay])
    return Workflow(name="joined", edges=[("START", (left, right), JoinNode(name="merge"), payer)])


def test_joined_payer_inherits_both_branch_tool_results(sink):
    workflow = _joined_workflow()
    _run(workflow)
    check = next(check for check in sink.checks if check[0] == "pay")
    lineage = ancestors(sink, check[2])
    assert {event.derived_from.name for event in lineage if event.HasField("derived_from")} >= {
        "left_default", "right_default"}
    assert any(event.agent == "merge" and "left" in event.text and "right" in event.text
               for event in lineage)


def test_parallel_sibling_tool_cannot_see_other_branch(sink):
    workflow = _joined_workflow()
    _run(workflow)
    right = next(check for check in sink.checks if check[0] == "right_default")
    lineage = ancestors(sink, right[2])
    assert not any(event.HasField("derived_from") and event.derived_from.name == "left_default"
                   for event in lineage)


@pytest.mark.parametrize("capability", ["state", "state_alias"])
def test_parallel_state_access_rejected_at_dispatch(sink, capability):
    def stateful(tool_context):
        """Use ADK session state."""
        return {"value": tool_context.state.get("value")}

    def stateful_alias(ctx: ToolContext):
        """Use ADK session state with an aliased context parameter."""
        return {"value": ctx.state.get("value")}

    tool = {"state": stateful, "state_alias": stateful_alias}[capability]
    left = LlmAgent(name="left", model=ScriptedModel([
        calls((tool.__name__, {}, "read")), text("left")]), tools=[tool])
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    workflow = Workflow(name="resources", edges=[("START", (left, right), JoinNode(name="merge"))])
    with pytest.raises(AdkInstrumentationError, match="Parallel Workflow tools cannot access ADK state or artifacts"):
        _run(workflow)
    assert any(check[0] == tool.__name__ for check in sink.checks)


def test_parallel_artifact_catalog_tool_rejected_before_any_node(sink):
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]), tools=[LoadArtifactsTool()])
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    workflow = Workflow(name="resources", edges=[("START", (left, right), JoinNode(name="merge"))])
    with pytest.raises(AdkInstrumentationError, match="Parallel Workflow tools cannot access ADK state or artifacts"):
        _run(workflow)
    assert not sink.events and not sink.checks


def test_parallel_output_key_rejected_before_any_node(sink):
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]), output_key="shared")
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    workflow = Workflow(name="resources", edges=[("START", (left, right), JoinNode(name="merge"))])
    with pytest.raises(AdkInstrumentationError, match="Parallel Workflow tools cannot access ADK state or artifacts"):
        _run(workflow)
    assert not left.model._requests and not right.model._requests
    assert not sink.events and not sink.checks


def test_parallel_indirect_artifact_write_rejected_at_runtime(sink):
    runner = None
    artifact_session = None

    async def write_artifact():
        """Write through a captured artifact service without ToolContext."""
        await runner.artifact_service.save_artifact(
            app_name="test", user_id="u", session_id=artifact_session.id,
            filename="report.txt", artifact=types.Part(text="report"))
        return {"saved": True}

    left = LlmAgent(name="left", model=ScriptedModel([
        calls(("write_artifact", {}, "write")), text("done")]), tools=[write_artifact])
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    workflow = Workflow(name="indirect", edges=[
        ("START", (left, right), JoinNode(name="merge")),
    ])
    runner = InMemoryRunner(node=workflow, app_name="test")
    guarded = adk.instrument_adk(runner)

    async def run():
        nonlocal artifact_session
        artifact_session = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            events = [event async for event in guarded.run_async(
                    user_id="u", session_id=artifact_session.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="hello")]))]
        return events

    with pytest.raises(AdkInstrumentationError, match="Parallel Workflow tools cannot access ADK state or artifacts"):
        asyncio.run(run())
    assert not runner.artifact_service.artifacts


def test_mutated_join_trigger_rejected_before_payer_dispatch(sink, monkeypatch):
    workflow = _joined_workflow()
    original = Workflow._buffer_barrier_trigger

    def corrupt(self, loop_state, target_name):
        original(self, loop_state, target_name)
        if target_name == "merge" and loop_state.trigger_buffer.get("merge"):
            loop_state.trigger_buffer["merge"][-1].input["left"] = "forged"

    monkeypatch.setattr(Workflow, "_buffer_barrier_trigger", corrupt)
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    agents = {node.name: node for node in workflow.graph.nodes}
    assert len(agents["left"].model._requests) == 2
    assert len(agents["right"].model._requests) == 2
    assert not any(check[0] == "pay" for check in sink.checks)


def test_mutated_join_output_rejected_before_payer_dispatch(sink, monkeypatch):
    workflow = _joined_workflow()
    original = Workflow._buffer_downstream_triggers

    def corrupt(self, loop_state, node_name, output, route, branch=None):
        if node_name == "merge":
            output = {"left": "forged", "right": "same final text"}
        return original(self, loop_state, node_name, output, route, branch)

    monkeypatch.setattr(Workflow, "_buffer_downstream_triggers", corrupt)
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    agents = {node.name: node for node in workflow.graph.nodes}
    assert len(agents["left"].model._requests) == 2
    assert len(agents["right"].model._requests) == 2
    assert not any(check[0] == "pay" for check in sink.checks)


@pytest.mark.parametrize("field,value", [
    ("branch", "forged"),
    ("use_sub_branch", True),
    ("isolation_scope", "forged"),
])
def test_mutated_parallel_trigger_context_rejected_before_child_model(
        sink, monkeypatch, field, value):
    left_source = LlmAgent(name="left_source", model=ScriptedModel([text("left")]))
    right_source = LlmAgent(name="right_source", model=ScriptedModel([text("right")]))
    left_child = LlmAgent(name="left_child", model=ScriptedModel([text("left child")]),
                          include_contents="default")
    right_child = LlmAgent(name="right_child", model=ScriptedModel([text("right child")]))
    merge = JoinNode(name="merge")
    workflow = Workflow(name="trigger_context", edges=[
        ("START", left_source, left_child, merge),
        ("START", right_source, right_child, merge),
    ])
    original = Workflow._buffer_downstream_triggers

    def corrupt(self, loop_state, node_name, output, route, branch=None):
        original(self, loop_state, node_name, output, route, branch)
        if node_name == "left_source":
            setattr(loop_state.trigger_buffer["left_child"][-1], field, value)

    monkeypatch.setattr(Workflow, "_buffer_downstream_triggers", corrupt)
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not left_child.model._requests
    assert not any(check[0] == "pay" for check in sink.checks)


def test_repeated_joined_turns_bind_only_current_branch_outputs(sink):
    paid = []

    def pay():
        """Record a synthetic payment."""
        paid.append(True)
        return {"paid": True}

    left = LlmAgent(name="left", model=ScriptedModel([text("same"), text("same")]))
    right = LlmAgent(name="right", model=ScriptedModel([text("same"), text("same")]))
    payer = LlmAgent(name="payer", model=ScriptedModel([
        calls(("pay", {}, "pay-1")), text("done"),
        calls(("pay", {}, "pay-2")), text("done"),
    ]), tools=[pay])
    workflow = Workflow(name="repeated_join", edges=[
        ("START", (left, right), JoinNode(name="merge"), payer),
    ])

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        session_record = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first request", "second request"):
                events = [event async for event in runner.run_async(
                    user_id="u", session_id=session_record.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]))]
                assert any(event.author == "payer" for event in events)

    asyncio.run(run())
    assert paid == [True, True]
    checks = [check for check in sink.checks if check[0] == "pay"]
    assert len(checks) == 2
    first, second = (ancestors(sink, check[2]) for check in checks)
    assert any(event.text == "first request" for event in first)
    assert not any(event.text == "second request" for event in first)
    assert any(event.text == "second request" for event in second)
    assert not any(event.text == "first request" for event in second)


def test_multi_predecessor_agent_without_join_rejected_before_dispatch(sink):
    left = LlmAgent(name="left", model=ScriptedModel([text("left")]))
    right = LlmAgent(name="right", model=ScriptedModel([text("right")]))
    payer = LlmAgent(name="payer", model=ScriptedModel([text("payer")]))
    workflow = Workflow(name="ambiguous", edges=[("START", (left, right), payer)])
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not any(node.model._requests for node in (left, right, payer))
    assert not sink.events and not sink.checks


def test_parallel_identical_handoffs_keep_exact_source_under_overlap(sink, monkeypatch):
    def read_left():
        """Read the left source."""
        return {"source": "left"}

    def read_right():
        """Read the right source."""
        return {"source": "right"}

    def use_left():
        """Use the left source."""
        return {"used": "left"}

    def use_right():
        """Use the right source."""
        return {"used": "right"}

    left_source = LlmAgent(name="left_source", model=ScriptedModel([
        calls(("read_left", {}, "left-read")), text("same")]), tools=[read_left])
    right_source = LlmAgent(name="right_source", model=ScriptedModel([
        calls(("read_right", {}, "right-read")), text("same")]), tools=[read_right])
    left_child = LlmAgent(name="left_child", model=ScriptedModel([
        calls(("use_left", {}, "left-use")), text("left done")]), tools=[use_left])
    right_child = LlmAgent(name="right_child", model=ScriptedModel([
        calls(("use_right", {}, "right-use")), text("right done")]), tools=[use_right])
    merge = JoinNode(name="merge")
    workflow = Workflow(name="overlap", edges=[
        ("START", left_source, left_child, merge),
        ("START", right_source, right_child, merge),
    ])

    original = adk._State.record
    arrived = 0
    seen = set()
    both = asyncio.Event()

    async def overlap(self, content, agent, inputs, **kwargs):
        nonlocal arrived
        if (agent in {"left_child", "right_child"} and agent not in seen
                and content.role == "user" and content.parts[0].text == "same"
                and not kwargs):
            seen.add(agent)
            arrived += 1
            if arrived == 2:
                both.set()
            await asyncio.wait_for(both.wait(), timeout=5)
        return await original(self, content, agent, inputs, **kwargs)

    monkeypatch.setattr(adk._State, "record", overlap)
    _run(workflow)
    assert arrived == 2
    left = next(check for check in sink.checks if check[0] == "use_left")
    right = next(check for check in sink.checks if check[0] == "use_right")
    left_lineage = ancestors(sink, left[2])
    right_lineage = ancestors(sink, right[2])
    assert any(event.HasField("derived_from") and event.derived_from.name == "read_left"
               for event in left_lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "read_right"
                   for event in left_lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "read_right"
               for event in right_lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "read_left"
                   for event in right_lineage)


@pytest.mark.parametrize("selected", ["approved", "rejected"])
def test_conditional_route_finalizer_inherits_only_selected_execution(sink, selected):
    def select_route(tool_context):
        """Choose one handler while forwarding constant output."""
        tool_context.actions.route = selected
        return {"selected": selected}

    def finalize():
        """Record the final decision."""
        return {"finished": True}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route-call")), text("constant")]),
        tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same handler output")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same handler output")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([
        calls(("finalize", {}, "final-call")), text("done")]), tools=[finalize])
    workflow = Workflow(name="conditional", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer),
        (rejected, finalizer),
    ])
    _run(workflow)

    assert len(approved.model._requests) == (1 if selected == "approved" else 0)
    assert len(rejected.model._requests) == (1 if selected == "rejected" else 0)
    checks = [check for check in sink.checks if check[0] == "finalize"]
    assert len(checks) == 1
    lineage = ancestors(sink, checks[0][2])
    assert any(event.HasField("derived_from") and event.derived_from.name == "select_route"
               for event in lineage)
    assert any(event.agent == selected and event.text == "same handler output"
               for event in lineage)
    assert not any(event.agent != selected and event.agent in {"approved", "rejected"}
                   for event in lineage)


@pytest.mark.parametrize("selected", ["approved", "rejected", "held"])
def test_multi_route_finalizer_inherits_only_selected_chain(sink, selected):
    def select_route(tool_context):
        """Select one of three handlers."""
        tool_context.actions.route = selected
        return {"selected": selected}

    def finalize():
        """Complete the selected route."""
        return {"finished": True}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("same")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    held_first = LlmAgent(name="held_first", model=ScriptedModel([text("same")]))
    held_last = LlmAgent(name="held_last", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([
        calls(("finalize", {}, "final")), text("done")]), tools=[finalize])
    workflow = Workflow(name="three_routes", edges=[
        ("START", router, {"approved": approved, "rejected": rejected, "held": held_first}),
        (approved, finalizer), (rejected, finalizer), (held_first, held_last, finalizer),
    ])
    _run(workflow)

    branch_names = {"approved": {"approved"}, "rejected": {"rejected"},
                    "held": {"held_first", "held_last"}}
    checks = [check for check in sink.checks if check[0] == "finalize"]
    assert len(checks) == 1
    lineage = ancestors(sink, checks[0][2])
    assert any(event.HasField("derived_from") and event.derived_from.name == "select_route"
               for event in lineage)
    assert {event.agent for event in lineage} & set().union(*branch_names.values()) == branch_names[selected]
    for name, agent in (("approved", approved), ("rejected", rejected),
                        ("held_first", held_first), ("held_last", held_last)):
        assert len(agent.model._requests) == int(name in branch_names[selected])


def test_repeated_three_route_turns_keep_only_current_selection(sink):
    selections = ["approved", "rejected", "held"]

    def select_route(tool_context):
        """Select a different handler each turn."""
        selected = selections.pop(0)
        tool_context.actions.route = selected
        return {"selected": selected}

    def finalize():
        """Complete the selected route."""
        return {"finished": True}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route-1")), text("same"),
        calls(("select_route", {}, "route-2")), text("same"),
        calls(("select_route", {}, "route-3")), text("same"),
    ]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    held = LlmAgent(name="held", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([
        calls(("finalize", {}, "final-1")), text("done"),
        calls(("finalize", {}, "final-2")), text("done"),
        calls(("finalize", {}, "final-3")), text("done"),
    ]), tools=[finalize])
    workflow = Workflow(name="three_turns", edges=[
        ("START", router, {"approved": approved, "rejected": rejected, "held": held}),
        (approved, finalizer), (rejected, finalizer), (held, finalizer),
    ])

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        guarded = adk.instrument_adk(runner)
        conversation_record = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first request", "second request", "third request"):
                async for _ in guarded.run_async(
                        user_id="u", session_id=conversation_record.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
                    pass

    asyncio.run(run())
    checks = [check for check in sink.checks if check[0] == "finalize"]
    assert len(checks) == 3
    for index, (prompt, branch) in enumerate(zip(
            ("first request", "second request", "third request"),
            ("approved", "rejected", "held"))):
        lineage = ancestors(sink, checks[index][2])
        assert any(event.text == prompt for event in lineage)
        assert {event.agent for event in lineage} & {"approved", "rejected", "held"} == {branch}
        assert not any(event.text in {"first request", "second request", "third request"} - {prompt}
                       for event in lineage)


def test_multi_route_forged_later_branch_rejected_before_handler(sink, monkeypatch):
    def select_route(tool_context):
        """Select the first configured handler."""
        tool_context.actions.route = "approved"
        return {"selected": "approved"}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("same")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    held = LlmAgent(name="held", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="three_routes_tampered", edges=[
        ("START", router, {"approved": approved, "rejected": rejected, "held": held}),
        (approved, finalizer), (rejected, finalizer), (held, finalizer),
    ])
    original = Workflow._buffer_downstream_triggers

    def corrupt(self, loop_state, node_name, output, route, branch=None):
        if node_name == "router":
            route = "held"
        return original(self, loop_state, node_name, output, route, branch)

    monkeypatch.setattr(Workflow, "_buffer_downstream_triggers", corrupt)
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not approved.model._requests and not rejected.model._requests
    assert not held.model._requests and not finalizer.model._requests


def test_multi_route_duplicate_label_rejected_before_router_dispatch(sink):
    def select_route(tool_context):
        """Select a branch."""
        tool_context.actions.route = "approved"
        return {"selected": "approved"}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("same")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    held = LlmAgent(name="held", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="duplicate_route", edges=[
        Edge(from_node=START, to_node=router),
        Edge(from_node=router, to_node=approved, route="approved"),
        Edge(from_node=router, to_node=rejected, route="approved"),
        Edge(from_node=router, to_node=held, route="held"),
        Edge(from_node=approved, to_node=finalizer),
        Edge(from_node=rejected, to_node=finalizer),
        Edge(from_node=held, to_node=finalizer),
    ])
    with pytest.raises(AdkInstrumentationError, match="distinct string routes"):
        _run(workflow)
    assert not any(agent.model._requests for agent in (router, approved, rejected, held, finalizer))


def test_repeated_conditional_turns_do_not_inherit_prior_alternative(sink):
    routes = ["approved", "rejected"]

    def select_route(tool_context):
        """Select one branch per request."""
        selected = routes.pop(0)
        tool_context.actions.route = selected
        return {"selected": selected}

    def finalize():
        """Record a final action."""
        return {"finished": True}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route-1")), text("same"),
        calls(("select_route", {}, "route-2")), text("same"),
    ]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([
        calls(("finalize", {}, "final-1")), text("done"),
        calls(("finalize", {}, "final-2")), text("done"),
    ]), tools=[finalize])
    workflow = Workflow(name="conditional", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        guarded = adk.instrument_adk(runner)
        session_record = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first request", "second request"):
                async for _ in guarded.run_async(
                        user_id="u", session_id=session_record.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
                    pass

    asyncio.run(run())
    checks = [check for check in sink.checks if check[0] == "finalize"]
    assert len(checks) == 2
    first, second = (ancestors(sink, check[2]) for check in checks)
    assert any(event.text == "first request" for event in first)
    assert any(event.agent == "approved" for event in first)
    assert not any(event.agent == "rejected" or event.text == "second request" for event in first)
    assert any(event.text == "second request" for event in second)
    assert any(event.agent == "rejected" for event in second)
    assert not any(event.agent == "approved" or event.text == "first request" for event in second)


@pytest.mark.parametrize("route", ["unknown", ["approved", "rejected"]])
def test_conditional_route_rejects_unmatched_or_multiple_targets(sink, route):
    def select_route(tool_context):
        """Emit a route that cannot select one configured handler."""
        tool_context.actions.route = route
        return {"selected": str(route)}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("constant")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("approved")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("rejected")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("final")]))
    workflow = Workflow(name="conditional", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not approved.model._requests and not rejected.model._requests
    assert not finalizer.model._requests


@pytest.mark.parametrize("label", ["", False, 0])
def test_conditional_route_rejects_falsey_or_nonstring_edge_label(sink, label):
    router = LlmAgent(name="router", model=ScriptedModel([text("router")]))
    handler = LlmAgent(name="handler", model=ScriptedModel([text("handler")]))
    workflow = Workflow(name="invalid_route", edges=[
        ("START", router, {label: handler}),
    ])
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not router.model._requests and not handler.model._requests


def test_conditional_router_state_read_reaches_finalizer(sink):
    def select_route(tool_context):
        """Choose a branch using ADK state while forwarding constant output."""
        selected = tool_context.state["decision"]
        tool_context.actions.route = selected
        return {"selected": selected}

    def finalize():
        """Record the final action."""
        return {"finished": True}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("constant")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("constant")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("constant")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([
        calls(("finalize", {}, "final")), text("done")]), tools=[finalize])
    workflow = Workflow(name="state_route", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        guarded = adk.instrument_adk(runner)
        session_record = await runner.session_service.create_session(
            app_name="test", user_id="u", state={"decision": "approved"})
        with session(end_on_exit=False):
            return [event async for event in guarded.run_async(
                user_id="u", session_id=session_record.id,
                new_message=types.Content(role="user", parts=[types.Part(text="route it")]))]

    asyncio.run(run())
    check = next(check for check in sink.checks if check[0] == "finalize")
    lineage = ancestors(sink, check[2])
    assert any('"adk_resource"' in event.text and '"decision"' in event.text
               for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "select_route"
               for event in lineage)


def test_conditional_router_must_begin_at_start(sink):
    def select_route(tool_context):
        """Choose an exclusive branch."""
        tool_context.actions.route = "approved"
        return {"selected": "approved"}

    prefix = LlmAgent(name="prefix", model=ScriptedModel([text("prefix")]))
    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("same")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="conditional_prefix", edges=[
        ("START", prefix, router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])
    with pytest.raises(AdkInstrumentationError, match="router must follow START"):
        _run(workflow)
    assert not any(agent.model._requests for agent in (prefix, router, approved, rejected, finalizer))


@pytest.mark.parametrize("tamper", ["observed_event", "opposite_trigger"])
def test_conditional_route_tampering_rejected_before_handler(sink, monkeypatch, tamper):
    def select_route(tool_context):
        """Choose the approved handler."""
        tool_context.actions.route = "approved"
        return {"selected": "approved"}

    router = LlmAgent(name="router", model=ScriptedModel([
        calls(("select_route", {}, "route")), text("same")]), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="conditional_tamper", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])
    if tamper == "observed_event":
        from google.adk.workflow._node_runner import NodeRunner
        original = NodeRunner._track_event_in_context

        def corrupt(self, event, ctx):
            original(self, event, ctx)
            if event.author == "router" and event.actions and event.actions.route == "approved":
                event.actions.route = "rejected"

        monkeypatch.setattr(NodeRunner, "_track_event_in_context", corrupt)
    else:
        original = Workflow._buffer_downstream_triggers

        def corrupt(self, loop_state, node_name, output, route, branch=None):
            if node_name == "router":
                route = "rejected"
            return original(self, loop_state, node_name, output, route, branch)

        monkeypatch.setattr(Workflow, "_buffer_downstream_triggers", corrupt)
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert not approved.model._requests and not rejected.model._requests
    assert not finalizer.model._requests


@pytest.mark.parametrize("second_call", ["same_response", "later_response", "later_without_route"])
def test_conditional_router_rejects_second_tool_call_before_execution(sink, second_call):
    executed = []

    def select_route(tool_context):
        """Choose the approved handler."""
        executed.append(True)
        if second_call != "later_without_route" or len(executed) > 1:
            tool_context.actions.route = "approved"
        return {"selected": "approved"}

    if second_call == "same_response":
        script = [calls(("select_route", {}, "first"), ("select_route", {}, "second")),
                  text("constant")]
    else:
        script = [calls(("select_route", {}, "first")),
                  calls(("select_route", {}, "second")), text("constant")]
    router = LlmAgent(name="router", model=ScriptedModel(script), tools=[select_route])
    approved = LlmAgent(name="approved", model=ScriptedModel([text("same")]))
    rejected = LlmAgent(name="rejected", model=ScriptedModel([text("same")]))
    finalizer = LlmAgent(name="finalizer", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="conditional_extra_call", edges=[
        ("START", router, {"approved": approved, "rejected": rejected}),
        (approved, finalizer), (rejected, finalizer),
    ])
    with pytest.raises(AdkInstrumentationError):
        _run(workflow)
    assert len(executed) == (0 if second_call == "same_response" else 1)
    assert not approved.model._requests and not rejected.model._requests
    assert not finalizer.model._requests


def test_mutated_serial_trigger_rejected_before_model_or_tool(sink, monkeypatch):
    first = LlmAgent(name="first", model=ScriptedModel([text("first")]))
    second = LlmAgent(name="second", model=ScriptedModel([text("second")]))
    workflow = Workflow(name="mutated", edges=[("START", first, second)])
    original = Workflow._seed_start_triggers

    def mutate(self, loop_state, ctx, node_input):
        original(self, loop_state, ctx, node_input)
        loop_state.trigger_buffer["first"][0].input = "forged"

    monkeypatch.setattr(Workflow, "_seed_start_triggers", mutate)
    with pytest.raises(AdkInstrumentationError, match="trigger input differs"):
        _run(workflow)
    assert not sink.checks


@pytest.mark.parametrize("node_type", ["function", "tool"])
def test_effectful_non_agent_node_rejected_before_dispatch(sink, node_type):
    executed = []

    def effect():
        """An effectful graph node."""
        executed.append(True)
        return {"done": True}

    node = effect if node_type == "function" else FunctionTool(effect)
    reader = LlmAgent(name="reader", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="effectful", edges=[("START", node, reader)])
    with pytest.raises(AdkInstrumentationError, match="Workflow nodes must be standard"):
        _run(workflow)
    assert executed == []
    assert not sink.events and not sink.checks
