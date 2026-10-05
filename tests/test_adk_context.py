"""Computation-scoped state reads and native ToolContext writes."""
import asyncio
import copy
import hashlib

import pytest
from sasy.instrumentation import adk_state, dependencies
from sasy.proto.observability_pb2 import Edge, Event
from test_adk_instrumentation import (
    AdkInstrumentationError,
    InMemoryRunner,
    LlmAgent,
    ParallelAgent,
    ScriptedModel,
    SequentialAgent,
    adk,
    calls,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


@pytest.fixture
def observed(sink, monkeypatch):
    sink.batches = []
    def resolve(items):
        sink.batches.append([item.SerializeToString() for item in items])
        if sink.failure:
            raise RuntimeError("observation offline")
        aliases = {}
        ids = []
        for item in items:
            event = Event.FromString(item.event.SerializeToString())
            node = item.base_id or "sasy:mv1:" + hashlib.sha256(item.SerializeToString(deterministic=True)).hexdigest()
            aliases[event.id] = node
            event.id = node
            if node not in sink.events:
                sink.events[node] = event
                for source in item.dependencies:
                    edge = Edge.FromString(source.SerializeToString())
                    edge.source = aliases.get(edge.source, edge.source)
                    edge.destination = node
                    assert edge.source.startswith("sasy:mv1:")
                    sink.edges.append(edge)
            ids.append(node)
        return ids
    async def resolve_async(items):
        return resolve(items)
    monkeypatch.setattr(adk.observation, "resolve_events", resolve)
    monkeypatch.setattr(adk.observation, "resolve_events_async", resolve_async)
    return sink


async def turn(runner, sess, message="request"):
    return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]))]


async def run(agent, state=None):
    runner = InMemoryRunner(agent=agent, app_name="test")
    sess = await runner.session_service.create_session(app_name="test", user_id="u", state=state)
    adk.instrument()
    with session(end_on_exit=False):
        events = await turn(runner, sess)
    return runner, sess, events


@pytest.mark.parametrize("sync", [True, False])
def test_getter_captures_without_rpc_and_action_flushes_exact_read(observed, sync):
    async def inspect_state(tool_context):
        """Read state and use the result."""
        count = len(observed.batches)
        value = tool_context.state["needed"]
        assert value == {"nested": [1]}
        assert len(observed.batches) == count
        ids = dependencies.resolve_inputs() if sync else await dependencies.resolve_inputs_async()
        assert any('"needed"' in event.text for event in ancestors(observed, ids))
        assert not any("UNREAD_SECRET" in event.text for event in observed.events.values())
        return value
    model = ScriptedModel([calls(("inspect_state", {}, "inspect")), text("done")])
    asyncio.run(run(LlmAgent(name="reader", model=model, tools=[inspect_state]),
        {"needed": {"nested": [1]}, "unread": "UNREAD_SECRET"}))
    assert observed.checks[0][1] == {}


@pytest.mark.parametrize("success", [True, False])
def test_assigned_state_preserves_success_only_after_real_completion(observed, success):
    async def approve(tool_context):
        """Write a decision."""
        tool_context.state["decision"] = "approved"
        await adk_state.flush_reads_async()
        assert not any(event.HasField("derived_from") for event in observed.events.values())
        return "approved" if success else {"error": "approval failed"}
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([calls(("approve", {}, "approve")), text("done")]), tools=[approve])
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Decision {decision}",
        before_model_callback=lambda callback_context, llm_request: setattr(llm_request, "contents", []))
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"decision"' in event.text and "approved" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage) == success


def test_nested_alias_rejected_and_copy_reassignment_observed(observed):
    async def edit(tool_context):
        """Edit a copy of state."""
        value = tool_context.state["item"]
        with pytest.raises(AdkInstrumentationError, match="copying and reassigning"):
            value["nested"].append(2)
        changed = copy.deepcopy(value)
        changed["nested"].append(2)
        tool_context.state["item"] = changed
        assert tool_context.state["item"] == {"nested": [1, 2]}
        return "edited"
    runner, sess, _ = asyncio.run(run(LlmAgent(name="editor",
        model=ScriptedModel([calls(("edit", {}, "edit")), text("done")]), tools=[edit]), {"item": {"nested": [1]}}))
    stored = asyncio.run(runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id))
    assert stored.state["item"] == {"nested": [1, 2]}


def test_contains_and_keys_observe_presence_without_reading_values(observed):
    async def inspect_state(tool_context):
        """Inspect key names."""
        assert "secret" in tool_context.state
        assert "missing" not in tool_context.state
        assert list(tool_context.state) == ["secret"]
        await adk_state.flush_reads_async()
        assert not any("SECRET_BODY" in event.text for event in observed.events.values())
        return "present"
    asyncio.run(run(LlmAgent(name="reader", model=ScriptedModel([calls(("inspect_state", {}, "inspect")), text("done")]),
        tools=[inspect_state]), {"secret": "SECRET_BODY"}))
    assert any("state-presence" in event.text for event in observed.events.values())
    assert any("state-keys" in event.text for event in observed.events.values())


def test_failed_flush_retries_identical_snapshot_before_action(observed):
    async def inspect_state(tool_context):
        """Read state."""
        assert tool_context.state.get("item") == "value"
        observed.failure = True
        with pytest.raises(RuntimeError, match="observation offline"):
            await dependencies.resolve_inputs_async()
        failed = observed.batches[-1]
        observed.failure = False
        ids = await dependencies.resolve_inputs_async()
        assert observed.batches[-1] == failed
        assert any('"item"' in event.text for event in ancestors(observed, ids))
        return "read"
    asyncio.run(run(LlmAgent(name="reader", model=ScriptedModel([calls(("inspect_state", {}, "inspect")), text("done")]),
        tools=[inspect_state]), {"item": "value"}))


def test_toolcontext_artifact_save_has_exact_delta_and_consumed_state(observed):
    async def save(tool_context):
        """Save a decision artifact."""
        value = tool_context.state["decision"]
        version = await tool_context.save_artifact("decision", types.Part(text=value))
        assert version == 0
        return "saved"
    writer = LlmAgent(name="writer", model=ScriptedModel([calls(("save", {}, "save")), text("done")]), tools=[save])
    reader = LlmAgent(name="reader", model=ScriptedModel([text("read")]), instruction="Read {artifact.decision}", include_contents="none")
    runner, sess, events = asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader]), {"decision": "source value"}))
    assert any(event.actions.artifact_delta == {"decision": 0} for event in events)
    output = next(event for event in observed.events.values() if event.text == "read")
    assert any('"state"' in event.text and "source value" in event.text for event in ancestors(observed, [output.id]))


def test_forged_delta_and_credential_apis_fail_before_effect(observed):
    async def forbidden(tool_context):
        """Try unsupported APIs."""
        with pytest.raises(AdkInstrumentationError, match="credential API"):
            await tool_context.load_credential("secret")
        tool_context.actions.state_delta["forged"] = "approval"
        return "forged"
    with pytest.raises(AdkInstrumentationError, match="resource deltas|output_key"):
        asyncio.run(run(LlmAgent(name="writer", model=ScriptedModel([calls(("forbidden", {}, "forged"))]), tools=[forbidden])))


def test_unpersisted_state_write_is_rolled_back_after_tool_failure(observed):
    async def run_test():
        async def fail(tool_context):
            """Fail after a write."""
            tool_context.state["value"] = "new"
            raise RuntimeError("tool failed")
        runner = InMemoryRunner(agent=LlmAgent(name="writer", model=ScriptedModel([calls(("fail", {}, "fail"))]), tools=[fail]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"value": "old"})
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(RuntimeError, match="tool failed"):
            await turn(runner, sess)
        stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=sess.id)
        assert stored.state["value"] == "old"
    asyncio.run(run_test())


def test_delta_swapped_for_a_value_a_json_dump_writes_alike_is_refused(observed):
    """A persisted delta is the write this turn observed, compared as a record.

    A JSON dump writes a tuple and the list it holds alike, so a delta swapped
    for one would persist a value that no recorded node describes.
    """
    async def edit(tool_context):
        """Edit."""
        tool_context.state["item"] = [1, 2]
        tool_context.actions.state_delta["item"] = (1, 2)
        return "edited"
    agent = LlmAgent(name="editor", model=ScriptedModel([calls(("edit", {}, "edit")), text("done")]), tools=[edit])
    with pytest.raises(AdkInstrumentationError, match="output_key writes|resource deltas"):
        asyncio.run(run(agent, {}))


def test_artifact_delta_swapped_for_a_value_python_reads_as_equal_is_refused(observed):
    """An artifact delta is the version this turn observed, compared as a record.

    Python reads ``True`` and the version number ``1`` as one value, so an
    equality check would let a boolean stand in for the version of a saved
    artifact.
    """
    async def save(tool_context):
        """Save twice, then substitute the version."""
        assert await tool_context.save_artifact("decision", types.Part(text="a")) == 0
        assert await tool_context.save_artifact("decision", types.Part(text="b")) == 1
        tool_context.actions.artifact_delta["decision"] = True
        return "saved"
    agent = LlmAgent(name="writer", model=ScriptedModel([calls(("save", {}, "save")), text("done")]), tools=[save])
    with pytest.raises(AdkInstrumentationError, match="resource deltas|additional instrumentation"):
        asyncio.run(run(agent, {}))


# One emoji, and the surrogate pair UTF-16 writes it as. They are two different
# Python strings, and `json.dumps` writes both as "😀".
EMOJI = "\U0001f600"
SURROGATE_PAIR = chr(0xD83D) + chr(0xDE00)


def test_state_swapped_for_text_a_json_dump_writes_alike_is_refused(observed):
    """A state read is compared on the canonical record, not the JSON dump.

    Swapping the value for one whose JSON dump is identical would otherwise let
    the second read consume a different string under the first read's node.
    """
    async def swap(tool_context):
        """Read, substitute, read again."""
        assert tool_context.state["s"] == EMOJI
        await dependencies.resolve_inputs_async()
        tool_context._invocation_context.session.state["s"] = SURROGATE_PAIR
        return tool_context.state["s"]

    agent = LlmAgent(name="reader", model=ScriptedModel([calls(("swap", {}, "swap")), text("done")]), tools=[swap])
    with pytest.raises(AdkInstrumentationError, match="changed without an observed producer"):
        asyncio.run(run(agent, {"s": EMOJI}))


def test_callback_state_selection_without_system_instruction_has_ancestry(observed):
    def filter_request(callback_context, llm_request):
        llm_request.config.system_instruction = None
        assert callback_context.state["keep"] == "last"
        llm_request.contents = llm_request.contents[-1:]
    agent = LlmAgent(name="reader", model=ScriptedModel([text("selected")]), before_model_callback=filter_request)
    asyncio.run(run(agent, {"keep": "last", "unread": "DO NOT OBSERVE"}))
    output = next(event for event in observed.events.values() if event.text == "selected")
    lineage = ancestors(observed, [output.id])
    assert any('"keep"' in event.text for event in lineage)
    assert not any("DO NOT OBSERVE" in event.text for event in observed.events.values())


def test_callback_state_instruction_is_an_observed_derivation(observed):
    def instruction(callback_context, llm_request):
        llm_request.config.system_instruction = "State says: " + callback_context.state["value"]
    model = ScriptedModel([text("done")])
    asyncio.run(run(LlmAgent(name="reader", model=model, before_model_callback=instruction), {"value": "actual"}))
    system = next(event for event in observed.events.values() if event.text == "State says: actual")
    lineage = ancestors(observed, [system.id])
    assert any('"state"' in event.text and '"value"' in event.text for event in lineage)
    assert any(event.text == "request" for event in lineage)


def test_native_state_copy_is_lazy_and_preserves_selected_producer(observed):
    from sasy.instrumentation import adk_context

    async def approve(tool_context):
        """Produce a decision."""
        tool_context.state["decision"] = "approved"
        return "approved"

    async def transport(tool_context):
        """Exercise the qualified child-copy boundary."""
        before = len(observed.batches)
        transfer = adk_context.capture_state(tool_context)
        assert len(observed.batches) == before
        assert not any("UNREAD_SECRET" in e.text for e in observed.events.values())
        parent = adk_state._frame.get()
        child_runner = InMemoryRunner(agent=LlmAgent(name="child", model=ScriptedModel([])), app_name="child")
        child = adk_state.Resources(parent.resources.state, child_runner, "u", "child-session")
        adk_context.attach_state_import(child, transfer)
        frame = adk_state.Frame(child, parent.invocation, "tool", context=tool_context, call_id=parent.call_id)
        token = adk_state._frame.set(frame)
        resolver = adk_state.enter_resolver(frame)
        try:
            ids = await child.read_state(parent.invocation, "decision", transfer.values)
            lineage = ancestors(observed, ids)
            assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage)
            assert not any("UNREAD_SECRET" in e.text for e in lineage)
            with pytest.raises(AdkInstrumentationError, match="changed before"):
                await child.read_state(parent.invocation, "decision", {"decision": "tampered"})
            with pytest.raises(AdkInstrumentationError, match="already attached"):
                adk_context.attach_state_import(child, transfer)
        finally:
            child.close()
            frame.live = False
            adk_state.leave_resolver(resolver)
            adk_state._frame.reset(token)
        return "copied"

    writer = LlmAgent(name="writer", model=ScriptedModel([calls(("approve", {}, "approve")), text("done")]), tools=[approve])
    transporter = LlmAgent(name="transporter", model=ScriptedModel([calls(("transport", {}, "transport")), text("done")]), tools=[transport])
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, transporter]), {"unread": "UNREAD_SECRET"}))


def test_copied_state_swapped_for_text_a_json_dump_writes_alike_is_refused(observed):
    """The child's import guard is the same canonical record as the parent's read."""
    from sasy.instrumentation import adk_context

    async def transport(tool_context):
        """Copy the state into a child and read it back, substituted."""
        transfer = adk_context.capture_state(tool_context)
        parent = adk_state._frame.get()
        child_runner = InMemoryRunner(agent=LlmAgent(name="child", model=ScriptedModel([])), app_name="child")
        child = adk_state.Resources(parent.resources.state, child_runner, "u", "child-session")
        adk_context.attach_state_import(child, transfer)
        frame = adk_state.Frame(child, parent.invocation, "tool", context=tool_context, call_id=parent.call_id)
        token = adk_state._frame.set(frame)
        resolver = adk_state.enter_resolver(frame)
        try:
            with pytest.raises(AdkInstrumentationError, match="changed before"):
                await child.read_state(parent.invocation, "s", {"s": SURROGATE_PAIR})
        finally:
            child.close()
            frame.live = False
            adk_state.leave_resolver(resolver)
            adk_state._frame.reset(token)
        return "copied"

    agent = LlmAgent(name="transporter", model=ScriptedModel([
        calls(("transport", {}, "transport")), text("done")]), tools=[transport])
    asyncio.run(run(agent, {"s": EMOJI}))


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("context_api", [True, False])
def test_artifact_producer_completion_is_required_without_history(observed, success, context_api):
    holder = {}
    async def approve(tool_context):
        """Write a decision and then finish approval."""
        part = types.Part(text="approved")
        if context_api:
            await tool_context.save_artifact("decision", part)
        else:
            owner = adk_state._frame.get().resources
            await holder["runner"].artifact_service.save_artifact(app_name="test", user_id="u",
                session_id=owner.session_id, filename="decision", artifact=part)
        assert not any(e.HasField("derived_from") for e in observed.events.values())
        return "approved" if success else {"error": "approval failed"}
    def consume():
        """Consume the decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([calls(("approve", {}, "approve")), text("done")]), tools=[approve])
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], instruction="Decision {artifact.decision}", include_contents="none",
        before_model_callback=lambda callback_context, llm_request: setattr(llm_request, "contents", []))
    async def scenario():
        runner = InMemoryRunner(agent=SequentialAgent(name="pipeline", sub_agents=[writer, reader]), app_name="test")
        holder["runner"] = runner
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await turn(runner, sess)
    asyncio.run(scenario())
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"artifact"' in e.text and "approved" in e.text for e in lineage)
    assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == success


def test_pending_read_is_not_replayed_into_later_model_step(observed):
    from google.adk.tools.function_tool import FunctionTool

    preparations = []
    def step():
        """Advance to the next model step."""
        return "next"
    class PreparingTool(FunctionTool):
        async def process_llm_request(self, *, tool_context, llm_request):
            if not preparations:
                assert tool_context.state["decision"] == "ONLY FIRST STEP"
            preparations.append(True)
            await super().process_llm_request(tool_context=tool_context, llm_request=llm_request)
    agent = LlmAgent(name="reader", model=ScriptedModel([calls(("step", {}, "step")), text("second output")]),
        tools=[PreparingTool(step)], instruction="Process the current input.",
        before_model_callback=lambda callback_context, llm_request: setattr(llm_request, "contents", []))
    asyncio.run(run(agent, {"decision": "ONLY FIRST STEP"}))
    result = next(e for e in observed.events.values() if e.text == "second output")
    assert len(preparations) == 2
    assert not any("ONLY FIRST STEP" in e.text for e in ancestors(observed, [result.id]))


@pytest.mark.parametrize("approved", [True, False])
def test_new_output_key_presence_check_inherits_its_producer(observed, approved):
    def approve():
        """Approve."""
        return "approval"
    def consume(tool_context):
        """Act only on whether a decision exists."""
        return "exists" if "decision" in tool_context.state else "absent"
    script = [calls(("approve", {}, "approve")), text("approved")] if approved else [text("approved")]
    writer = LlmAgent(name="writer", model=ScriptedModel(script), tools=[approve], output_key="decision")
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]),
        include_contents="none", tools=[consume])
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    result = next(e for e in observed.events.values() if e.HasField("derived_from") and e.derived_from.name == "consume")
    lineage = ancestors(observed, [result.id])
    presence = [e for e in lineage if "state-presence" in e.text]
    assert presence and all("observed production" in e.text for e in presence)
    assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == approved


def test_observed_absence_is_superseded_by_new_output_key_presence(observed):
    seen = []
    def probe(tool_context):
        """Report whether a decision exists."""
        seen.append("decision" in tool_context.state)
        return "probed"
    def reader(name):
        return LlmAgent(name=name, model=ScriptedModel([calls(("probe", {}, name)), text("done")]),
            include_contents="none", tools=[probe])
    writer = LlmAgent(name="writer", model=ScriptedModel([text("approved")]), output_key="decision")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[reader("before"), writer, reader("after")])))
    assert seen == [False, True]
    results = [e for e in observed.events.values() if e.HasField("derived_from") and e.derived_from.name == "probe"]
    produced = [any("state-presence" in a.text and "observed production" in a.text for a in ancestors(observed, [e.id]))
        for e in results]
    assert sorted(produced) == [False, True]


def test_artifact_saved_before_tool_exception_keeps_inputs_without_success_evidence(observed):
    async def scenario():
        async def save_then_fail(tool_context):
            """Save an artifact and then fail."""
            await tool_context.save_artifact("note.txt", types.Part(text="SAVED_BEFORE_FAILURE"))
            raise RuntimeError("tool failed")
        def consume():
            """Consume the note."""
            return "consumed"
        runner = InMemoryRunner(agent=LlmAgent(name="agent", tools=[save_then_fail],
            model=ScriptedModel([calls(("save_then_fail", {}, "save"))])), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            with pytest.raises(RuntimeError, match="tool failed"):
                await turn(runner, sess, "FIRST_REQUEST")
            runner.agent = LlmAgent(name="agent", tools=[consume], instruction="Note {artifact.note.txt}",
                model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]), include_contents="none",
                before_model_callback=lambda callback_context, llm_request: setattr(llm_request, "contents", []))
            await turn(runner, sess, "second")
    asyncio.run(scenario())
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"artifact"' in e.text and "SAVED_BEFORE_FAILURE" in e.text for e in lineage)
    assert any("FIRST_REQUEST" in e.text for e in lineage)
    assert not any(e.HasField("derived_from") and e.derived_from.name == "save_then_fail" for e in observed.events.values())


def test_callback_write_is_read_later_with_only_its_own_inputs(observed):
    def approve(callback_context, llm_request):
        callback_context.state["decision"] = "approved for " + callback_context.state["source"]
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([text("done")]), before_model_callback=approve,
        output_key="summary")
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Decision {decision}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader]),
        {"source": "the invoice", "unread": "UNREAD SECRET"}))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"decision"' in event.text and "approved for the invoice" in event.text
               and "observed production" in event.text for event in lineage)
    assert any('"source"' in event.text and "the invoice" in event.text for event in lineage)
    assert any(event.text == "request" for event in lineage)
    assert not any("UNREAD SECRET" in event.text for event in observed.events.values())


def test_agent_callback_write_is_read_by_a_later_agent(observed):
    def stamp(callback_context):
        callback_context.state["decision"] = "approved"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([text("done")]), before_agent_callback=stamp)
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Decision {decision}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader]), {"unread": "UNREAD SECRET"}))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"decision"' in event.text and "approved" in event.text
               and "observed production" in event.text for event in lineage)
    assert not any("UNREAD SECRET" in event.text for event in observed.events.values())


def test_before_tool_callback_write_keeps_no_successful_tool_evidence(observed):
    def log(tool, args, tool_context):
        tool_context.state["logged"] = "call for " + args["subject"]
    async def approve(tool_context, subject):
        """Approve a subject."""
        assert tool_context.state["logged"] == "call for " + subject
        lineage = ancestors(observed, await dependencies.resolve_inputs_async())
        assert any("call for invoice" in event.text and "observed production" in event.text for event in lineage)
        assert not any("call for invoice" in event.text and "unattributed" in event.text
                       for event in observed.events.values())
        tool_context.state["decision"] = "approved " + subject
        return "approved"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", tools=[approve], before_tool_callback=log,
        model=ScriptedModel([calls(("approve", {"subject": "invoice"}, "approve")), text("done")]))
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Logged {logged} decided {decision}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    logged = [event for event in lineage if "adk_resource" in event.text and "call for invoice" in event.text]
    assert logged and not any(event.HasField("derived_from")
                              for node in logged for event in ancestors(observed, [node.id]))
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)


@pytest.mark.parametrize("tamper", ["alongside", "alone"])
def test_callback_delta_outside_the_receipt_is_rejected(observed, tamper):
    def callback(callback_context, llm_request):
        if tamper == "alongside":
            callback_context.state["decision"] = "approved"
        callback_context.actions.state_delta["forged"] = "approval"
    agent = LlmAgent(name="writer", model=ScriptedModel([text("done")]), before_model_callback=callback)
    with pytest.raises(AdkInstrumentationError, match="resource deltas|output_key"):
        asyncio.run(run(agent))


def test_callback_state_write_is_rolled_back_when_the_callback_raises(observed):
    captured = {}
    def callback(callback_context, llm_request):
        captured["session"] = adk_state._frame.get().invocation.session
        callback_context.state["value"] = "new"
        captured["during"] = dict(captured["session"].state)
        raise RuntimeError("callback failed")
    async def run_test():
        runner = InMemoryRunner(app_name="test", agent=LlmAgent(name="writer",
            model=ScriptedModel([text("must not run")]), before_model_callback=callback))
        sess = await runner.session_service.create_session(app_name="test", user_id="u", state={"value": "old"})
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(RuntimeError, match="callback failed"):
            await turn(runner, sess)
    asyncio.run(run_test())
    assert captured["during"]["value"] == "new"
    assert captured["session"].state["value"] == "old"
    assert not any(event.actions.state_delta for event in captured["session"].events)


def test_parallel_agent_callback_writes_are_rejected(observed):
    def writer(value):
        def callback(callback_context, llm_request):
            callback_context.state["decision"] = value
        return callback
    root = ParallelAgent(name="parallel", sub_agents=[
        LlmAgent(name="a", model=ScriptedModel([text("a")]), before_model_callback=writer("a")),
        LlmAgent(name="b", model=ScriptedModel([text("b")]), before_model_callback=writer("b"))])
    with pytest.raises(AdkInstrumentationError, match="concurrency qualification"):
        asyncio.run(run(root))


@pytest.mark.parametrize("scope", ["model", "tool"])
def test_callback_artifact_writes_stay_rejected(observed, scope):
    async def save_in_model(callback_context, llm_request):
        await callback_context.save_artifact("note.txt", types.Part(text="FORGED"))
    async def save_in_tool(tool, args, tool_context):
        await tool_context.save_artifact("note.txt", types.Part(text="FORGED"))
    def approve():
        """Approve."""
        return "approved"
    agent = LlmAgent(name="writer", tools=[approve],
        model=ScriptedModel([calls(("approve", {}, "approve")), text("done")]),
        before_model_callback=save_in_model if scope == "model" else None,
        before_tool_callback=None if scope == "model" else save_in_tool)
    with pytest.raises(AdkInstrumentationError, match="Callback artifact writes"):
        asyncio.run(run(agent))
    assert not any("FORGED" in event.text for event in observed.events.values())


def test_after_agent_and_after_model_callback_writes_are_observed(observed):
    def after_agent(callback_context):
        callback_context.state["stamped"] = "agent saw " + callback_context.state["source"]
    def after_model(callback_context, llm_response):
        callback_context.state["noted"] = "model finished"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([text("done")]),
        after_agent_callback=after_agent, after_model_callback=after_model)
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Stamped {stamped} noted {noted}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader]), {"source": "the invoice"}))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any("agent saw the invoice" in event.text and "observed production" in event.text for event in lineage)
    assert any("model finished" in event.text and "observed production" in event.text for event in lineage)


def test_output_key_supersedes_a_callback_write_of_the_same_key(observed):
    def after_model(callback_context, llm_response):
        callback_context.state["result"] = "callback draft"
    def consume():
        """Consume a result."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([text("model answer")]),
        after_model_callback=after_model, output_key="result")
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Result {result}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    assert "Result model answer" in reader.model._requests[0].config.system_instruction
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"result"' in event.text and "model answer" in event.text
               and "observed production" in event.text for event in lineage)
    assert not any('"result"' in event.text and "callback draft" in event.text for event in lineage)


@pytest.mark.parametrize("parallel", [False, True])
def test_state_written_in_a_batch_of_calls_keeps_its_producer(observed, parallel):
    async def approve(tool_context):
        """Write a decision."""
        tool_context.state["decision"] = "approved"
        return "approved"
    async def note():
        """Do something else."""
        return "noted"
    def consume():
        """Consume a decision."""
        return "consumed"
    response = calls(("approve", {}, "approve"), ("note", {}, "note")) if parallel else calls(("approve", {}, "approve"))
    writer = LlmAgent(name="writer", model=ScriptedModel([response, text("done")]), tools=[approve, note])
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Decision {decision}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"decision"' in event.text and "approved" in event.text
               and "observed production" in event.text for event in lineage)
    assert not any("unattributed external input" in event.text and "approved" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)


def test_state_written_in_a_batch_of_calls_is_produced_for_a_later_adk_session(observed):
    async def approve(tool_context):
        """Write a decision."""
        tool_context.state["user:decision"] = "approved"
        return "approved"
    async def note():
        """Do something else."""
        return "noted"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", tools=[approve, note],
        model=ScriptedModel([calls(("approve", {}, "approve"), ("note", {}, "note")), text("done")]))
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Decision {user:decision}")
    async def run_test():
        runner = InMemoryRunner(agent=writer, app_name="test")
        first = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await turn(runner, first)
            runner.agent = reader
            second = await runner.session_service.create_session(app_name="test", user_id="u")
            await turn(runner, second, "read the decision")
    asyncio.run(run_test())
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"decision"' in event.text and "approved" in event.text
               and "observed production" in event.text for event in lineage)
    assert not any("unattributed external input" in event.text and "approved" in event.text for event in lineage)
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)


@pytest.mark.parametrize("kind", ["before_agent", "after_agent"])
def test_agent_callback_delta_outside_the_receipt_never_reaches_the_service(observed, kind):
    def callback(callback_context):
        callback_context.state["noted"] = "observed"
        callback_context.actions.state_delta["user:forged"] = "approval"
    agent = LlmAgent(name="writer", model=ScriptedModel([text("done")]), **{f"{kind}_callback": callback})
    async def run_test():
        runner = InMemoryRunner(agent=agent, app_name="test")
        first = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            with pytest.raises(AdkInstrumentationError, match="outside its observed context"):
                await turn(runner, first)
            later = await runner.session_service.create_session(app_name="test", user_id="u")
            stored = await runner.session_service.get_session(app_name="test", user_id="u", session_id=first.id)
        return dict(later.state), list(stored.events)
    state, events = asyncio.run(run_test())
    assert "user:forged" not in state and "noted" not in state
    assert not any(event.actions.state_delta for event in events)


def test_temp_and_persistent_writes_in_one_call_stay_observed(observed):
    async def record(tool_context):
        """Write a scratch note and a decision."""
        tool_context.state["temp:note"] = "scratch"
        tool_context.state["user:decision"] = "approved"
        return "written"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([calls(("record", {}, "record")), text("done")]), tools=[record])
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Note {temp:note} decision {user:decision}")
    async def run_test():
        runner = InMemoryRunner(agent=SequentialAgent(name="pipeline", sub_agents=[writer, reader]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await turn(runner, sess)
            runner.agent = LlmAgent(name="later", model=ScriptedModel([text("the next turn ran")]))
            await turn(runner, sess, "again")
    asyncio.run(run_test())
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"scratch"' in event.text and "observed production" in event.text for event in lineage)
    assert any('"approved"' in event.text and "observed production" in event.text for event in lineage)
    assert any(event.text == "the next turn ran" for event in observed.events.values())


def test_temp_callback_write_beside_an_output_key_stays_observed(observed):
    def after_model(callback_context, llm_response):
        callback_context.state["temp:note"] = "scratch"
    def consume():
        """Consume a decision."""
        return "consumed"
    writer = LlmAgent(name="writer", model=ScriptedModel([text("approved")]),
        after_model_callback=after_model, output_key="decision")
    reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("finished")]),
        tools=[consume], include_contents="none", instruction="Note {temp:note} decision {decision}")
    asyncio.run(run(SequentialAgent(name="pipeline", sub_agents=[writer, reader])))
    check = next(check for check in observed.checks if check[0] == "consume")
    lineage = ancestors(observed, check[2])
    assert any('"scratch"' in event.text and "observed production" in event.text for event in lineage)
    assert any('"decision"' in event.text and "approved" in event.text
               and "observed production" in event.text for event in lineage)


def test_cross_graph_tool_state_read_is_an_external_input(observed):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    def approve(tool_context):
        """Write a decision."""
        tool_context.state["app:decision"] = "approved"
        return "approved"
    def consume(tool_context):
        """Read the decision."""
        return tool_context.state["app:decision"]
    service = InMemorySessionService()
    writer = Runner(app_name="test", session_service=service, agent=LlmAgent(name="writer", tools=[approve],
        model=ScriptedModel([calls(("approve", {}, "approve")), text("done")])))
    reader = Runner(app_name="test", session_service=service, agent=LlmAgent(name="reader", tools=[consume],
        model=ScriptedModel([calls(("consume", {}, "one")), calls(("consume", {}, "two")), text("finished")])))
    adk.instrument()
    written = set()
    async def go():
        first = await service.create_session(app_name="test", user_id="u", session_id="writer")
        with session(end_on_exit=False):
            await turn(writer, first)
        written.update(observed.events)
        second = await service.create_session(app_name="test", user_id="u", session_id="reader")
        with session(end_on_exit=False):
            await turn(reader, second)
    asyncio.run(go())
    external = [event for event in observed.events.values() if adk_state.EXTERNAL_SESSION_ORIGIN in event.text]
    assert len(external) == 1
    assert "unattributed external input" in external[0].text and "approved" in external[0].text
    lineage = ancestors(observed, [next(event for event in observed.events.values()
        if event.text == "finished" and event.id not in written).id])
    assert external[0].id in {event.id for event in lineage}
    assert not any(event.id in written for event in lineage)
    assert not any(event.HasField("derived_from") and event.derived_from.name == "approve" for event in lineage)
