"""Exact handoffs and fail-closed boundaries of native ADK Workflow runs."""
import asyncio

import pytest

from test_adk_instrumentation import (  # noqa: F401
    AdkInstrumentationError,
    InMemoryRunner,
    LlmAgent,
    ScriptedModel,
    adk,
    calls,
    conversation,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors

from google.adk.workflow import JoinNode, Workflow
from sasy.proto.observability_pb2 import Role


def _run(workflow):
    return asyncio.run(conversation(workflow, runner=InMemoryRunner(node=workflow, app_name="test")))


def test_serial_workflow_preserves_tool_result_ancestry_with_repeated_text(sink):
    executed = []

    def read_invoice():
        """Read an invoice."""
        return {"invoice": "invoice-104"}

    def approve_invoice():
        """Approve an invoice."""
        return {"approved": "invoice-104"}

    def disburse_payment():
        """Disburse a payment."""
        executed.append("payment")
        return {"paid": "invoice-104"}

    workflow = Workflow(name="payments", edges=[(
        "START",
        LlmAgent(name="requester", model=ScriptedModel([
            calls(("read_invoice", {}, "read")), text("Step complete.")]), tools=[read_invoice]),
        LlmAgent(name="reviewer", model=ScriptedModel([
            calls(("approve_invoice", {}, "approve")), text("Step complete.")]), tools=[approve_invoice]),
        LlmAgent(name="payer", model=ScriptedModel([
            calls(("disburse_payment", {}, "pay")), text("Step complete.")]), tools=[disburse_payment]),
    )])
    events, _, _ = _run(workflow)

    assert executed == ["payment"]
    assert [event.author for event in events if event.content] == [
        "requester", "requester", "requester",
        "reviewer", "reviewer", "reviewer",
        "payer", "payer", "payer",
    ]
    pay_check = next(check for check in sink.checks if check[0] == "disburse_payment")
    lineage = ancestors(sink, pay_check[2])
    assert any(event.HasField("derived_from") and event.derived_from.name == "read_invoice"
               for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve_invoice"
               for event in lineage)
    assert any(event.agent == "reviewer" and event.text == "Step complete." for event in lineage)


def test_serial_workflow_denies_before_tool_dispatch(sink):
    executed = []

    def disburse_payment():
        """Disburse a payment."""
        executed.append(True)
        return {"paid": True}

    sink.allowed = False
    workflow = Workflow(name="payments", edges=[(
        "START",
        LlmAgent(name="reviewer", model=ScriptedModel([text("Step complete.")])),
        LlmAgent(name="payer", model=ScriptedModel([
            calls(("disburse_payment", {}, "pay")), text("Step complete.")]), tools=[disburse_payment]),
    )])
    events, _, _ = _run(workflow)

    assert executed == []
    assert len(sink.checks) == 1 and sink.checks[0][0] == "disburse_payment"
    assert any("[BLOCKED]" in str(response.response)
               for event in events for response in event.get_function_responses())


def test_repeated_workflow_turns_bind_current_user_message(sink):
    first = LlmAgent(name="first", model=ScriptedModel([text("same"), text("same")]))
    second = LlmAgent(name="second", model=ScriptedModel([text("same"), text("same")]))
    workflow = Workflow(name="repeat", edges=[("START", first, second)])

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        session_record = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("first request", "second request"):
                events = [event async for event in runner.run_async(
                    user_id="u", session_id=session_record.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]))]
                assert [event.author for event in events if event.content] == ["first", "second"]

    asyncio.run(run())
    final = [event for event in sink.events.values()
             if event.agent == "second" and event.text == "same" and event.role == Role.LLM]
    assert len(final) == 2
    ancestries = [ancestors(sink, [event.id]) for event in final]
    assert any(any(event.text == "first request" for event in lineage)
               and not any(event.text == "second request" for event in lineage)
               for lineage in ancestries)
    assert any(any(event.text == "second request" for event in lineage)
               and not any(event.text == "first request" for event in lineage)
               for lineage in ancestries)


def test_workflow_agent_callback_cannot_read_unrecorded_handoff(sink):
    def mutate(callback_context):
        callback_context._invocation_context.session.events[-1].content.parts[0].text = "PRIOR_SECRET"

    workflow = Workflow(name="callback", edges=[(
        "START", LlmAgent(name="first", model=ScriptedModel([text("safe")])),
        LlmAgent(name="payer", model=ScriptedModel([text("done")]), before_agent_callback=mutate),
    )])
    with pytest.raises(AdkInstrumentationError, match="unrecorded handoff"):
        _run(workflow)
    assert not sink.events and not sink.checks


def test_mutated_synthetic_handoff_fails_before_authorization(sink, monkeypatch):
    from google.adk.flows.llm_flows import contents

    executed = []
    def send_email(content: str):
        """Send email."""
        executed.append(content)
        return {"sent": True}

    first = LlmAgent(name="first", model=ScriptedModel([text("safe 1"), text("safe 2")]))
    payer = LlmAgent(name="payer", model=ScriptedModel([
        text("done 1"), calls(("send_email", {"content": "PRIOR_SECRET"}, "send")), text("done 2")
    ]), tools=[send_email])
    workflow = Workflow(name="handoff", edges=[("START", first, payer)])

    adk.instrument()
    original = contents._ContentLlmRequestProcessor.run_async
    turns = 0
    async def altered(self, invocation_context, llm_request):
        nonlocal turns
        if invocation_context.agent.name == "payer":
            turns += 1
            if turns == 2:
                invocation_context.session.events[-1].content.parts[0].text = "PRIOR_SECRET"
        iterator = original(self, invocation_context, llm_request)
        try:
            async for event in iterator:
                yield event
        finally:
            await iterator.aclose()
    monkeypatch.setattr(contents._ContentLlmRequestProcessor, "run_async", altered)

    async def run():
        runner = InMemoryRunner(node=workflow, app_name="test")
        session_record = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            for prompt in ("PRIOR_SECRET", "CURRENT_SAFE"):
                async for _ in runner.run_async(
                    user_id="u", session_id=session_record.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
                    pass

    with pytest.raises(AdkInstrumentationError, match="handoff changed"):
        asyncio.run(run())
    assert turns == 2
    assert executed == []
    assert not any(check[0] == "send_email" for check in sink.checks)


def test_workflow_fan_out_keeps_sibling_tool_scopes_separate(sink):
    def read_left():
        """Read the left document."""
        return {"left": "confidential"}

    def read_right():
        """Read the right document."""
        return {"right": "public"}

    left = LlmAgent(name="left", model=ScriptedModel([
        calls(("read_left", {}, "left-read")), text("left done")]),
        tools=[read_left], include_contents="default")
    right = LlmAgent(name="right", model=ScriptedModel([
        calls(("read_right", {}, "right-read")), text("right done")]),
        tools=[read_right], include_contents="default")
    workflow = Workflow(name="fanout", edges=[("START", (left, right), JoinNode(name="merge"))])
    _run(workflow)

    left_check = next(check for check in sink.checks if check[0] == "read_left")
    right_check = next(check for check in sink.checks if check[0] == "read_right")
    left_lineage = ancestors(sink, left_check[2])
    right_lineage = ancestors(sink, right_check[2])
    assert any(event.text == "hello" for event in left_lineage)
    assert any(event.text == "hello" for event in right_lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "read_right"
                   for event in left_lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "read_left"
                   for event in right_lineage)


def test_workflow_rejects_effectful_function_node_before_dispatch(sink):
    executed = []

    def unguarded():
        executed.append(True)
        return "unsafe"

    reader = LlmAgent(name="reader", model=ScriptedModel([text("done")]))
    workflow = Workflow(name="functions", edges=[("START", unguarded, reader)])
    with pytest.raises(AdkInstrumentationError, match="single-turn LlmAgents"):
        _run(workflow)
    assert executed == []
    assert not sink.events and not sink.checks
