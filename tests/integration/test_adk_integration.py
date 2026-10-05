"""Owned-engine ADK examples; real provider calls are an explicit opt-in lane."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _example_path():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "examples/adk-separation-of-duties/main.py"
        if candidate.is_file():
            return candidate
    raise AssertionError("ADK example is missing from the release tree")


def _run(engine, scenario, *, live=False, stream=False):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1")
    # A source checkout's SDK is explicit; installed-wheel runs need no path.
    import sasy
    env["PYTHONPATH"] = str(Path(sasy.__file__).resolve().parent.parent)
    # --trace keeps the per-event lines these assertions read; plain runs narrate.
    command = [sys.executable, str(_example_path()), "--scenario", scenario, "--trace"]
    if stream:
        command.append("--stream")
    if live:
        assert os.environ.get("GOOGLE_API_KEY"), "GOOGLE_API_KEY is required for the requested live lane"
        env["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        assert os.environ.get("SASY_ADK_LIVE_MODEL"), "Select an available live Gemini model explicitly"
        env["SASY_ADK_LIVE_MODEL"] = os.environ["SASY_ADK_LIVE_MODEL"]
        command.append("--live")
    completed = subprocess.run(command, cwd=engine.root, env=env, capture_output=True, text=True, timeout=300)
    output = completed.stdout + completed.stderr
    for value in (engine.tenant_a_key, os.environ.get("GOOGLE_API_KEY", "")):
        if value:
            output = output.replace(value, "[REDACTED]")
    assert completed.returncode == 0, output
    assert '"scenario":' in completed.stdout
    if not live:
        assert "[requester] REQUEST submit_payment(" in completed.stdout
        assert "[payer] REQUEST disburse_payment(" in completed.stdout
        outcome = "ALLOWED" if scenario == "approved" else "DENIED"
        assert f"[payer] {outcome} disburse_payment:" in completed.stdout


@pytest.mark.parametrize("scenario", ["approved", "wrong-payee", "wrong-amount", "wrong-request", "missing-approval", "same-agent", "denied-approval", "approval-error"])
def test_adk_payment_policy(engine, scenario):
    _run(engine, scenario)


@pytest.mark.parametrize("scenario", ["approved", "wrong-payee", "denied-approval"])
def test_adk_streaming_payment_policy(engine, scenario):
    _run(engine, scenario, stream=True)


@pytest.mark.skipif(os.environ.get("SASY_RUN_LIVE_FRAMEWORK_TESTS") != "1", reason="set SASY_RUN_LIVE_FRAMEWORK_TESTS=1 for provider calls")
@pytest.mark.parametrize(("scenario", "stream"), [("approved", False), ("wrong-payee", False), ("approved", True)])
def test_adk_live_gemini(engine, scenario, stream):
    _run(engine, scenario, live=True, stream=stream)


@pytest.mark.parametrize("scenario", ["streaming", "transfer-allowed", "transfer-denied", "concurrent", "cancelled-stream", "state-approval", "state-no-approval", "artifact-approval", "artifact-no-approval", "otel", "agent-tool-approval", "agent-tool-no-approval", "context-state-approval", "context-state-no-approval", "context-artifact-approval", "context-artifact-no-approval", "context-state-error", "context-artifact-error", "native-single-turn-approval", "native-single-turn-no-approval", "native-single-turn-denied", "native-task-approval", "native-task-no-approval", "native-task-denied", "forwarded-state-approval", "forwarded-state-no-approval", "forwarded-artifact-approval", "forwarded-artifact-no-approval", "persisted-artifact-approval", "persisted-artifact-no-approval", "nested-argument-allowed", "nested-argument-denied", "workflow-route-denied", "workflow-route-third-denied", "workflow-state-approval", "workflow-state-no-approval", "workflow-artifact-approval", "workflow-artifact-no-approval", "concurrent-workflow-artifact"])
def test_native_adk_streaming_and_transfers(engine, scenario):
    """Run global patches in a separate process so other framework tests stay isolated."""
    import sasy
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               OTEL_ENABLED="true" if scenario == "otel" else "false",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), scenario],
                               cwd=engine.root, env=env, capture_output=True, text=True, timeout=300)
    output = (completed.stdout + completed.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert completed.returncode == 0, output
    assert '"native_adk":' in completed.stdout


async def _native_probe(scenario):
    import asyncio
    import json
    from uuid import uuid4

    import sasy
    from google.adk.agents import LlmAgent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from pydantic import PrivateAttr
    from sasy.config import get_config, get_stub
    from sasy.instrumentation import adk
    from sasy.instrumentation.otel import _current_input_ids
    from sasy.proto import observability_pb2 as obs
    from sasy.proto import observability_pb2_grpc as rpc

    sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
    sasy.instrument(adk=True)
    observer = get_stub(rpc.ObservabilityUpdatesStub)
    graph_client = get_stub(rpc.ObservabilityStub)
    metadata = get_config().get_metadata()
    policy = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "consume").\n'
    policy += 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "transfer_to_agent").\n'

    def state():
        current = observer.GetState(obs.StateRequest(), metadata=metadata, timeout=15)
        current.events.sort(key=lambda event: event.SerializeToString(deterministic=True))
        current.edges.sort(key=lambda edge: edge.SerializeToString(deterministic=True))
        return current

    def ancestry(node):
        return graph_client.BackwardSlice(obs.SliceRequest(event_id=node,
            session_id=sasy.get_current_session_id()), metadata=metadata, timeout=15)

    def response(text=None, *, call=None, partial=False):
        part = types.Part(text=text) if call is None else types.Part(function_call=types.FunctionCall(
            name=call[0], args=call[1], id="shared-call-id"))
        return LlmResponse(content=types.Content(role="model", parts=[part]), partial=partial)

    def user(text):
        return types.Content(role="user", parts=[types.Part(text=text)])

    class Model(BaseLlm):
        model: str = "owned-engine-scripted"
        _generate: object = PrivateAttr()
        _requests: list = PrivateAttr(default_factory=list)

        def __init__(self, generate):
            super().__init__()
            self._generate = generate

        async def generate_content_async(self, llm_request, stream=False):
            self._requests.append(llm_request.model_copy(deep=True))
            async for item in self._generate(llm_request, stream):
                yield item

    async def make_runner(agent):
        runner = InMemoryRunner(agent=agent, app_name="native_probe")
        conversation = await runner.session_service.create_session(app_name="native_probe", user_id="user")
        return runner, conversation

    def without_history(callback_context, llm_request):
        llm_request.contents = []

    def assert_call_observed(name, arguments):
        inputs = _current_input_ids.get()
        assert inputs, "Dispatch has no observed origin"
        graphs = [ancestry(node) for node in inputs]
        assert any(tool.name == name and json.loads(tool.arguments) == arguments
                   for graph in graphs for node in graph.nodes for tool in node.tools), "Final call was not recorded before dispatch"
        return graphs

    if scenario == "streaming":
        executed = []
        baseline = []
        def consume(value: int):
            """Consume a synthetic value."""
            assert_call_observed("consume", {"value": value})
            executed.append(value)
            return {"value": value}
        async def generate(request, stream):
            assert stream, "Native SSE mode did not reach the provider boundary"
            if not any(part.function_response for content in request.contents for part in content.parts or []):
                baseline.append(state())
                yield response("partial-one", partial=True)
                assert state() == baseline[0], "Partial text caused a graph write"
                yield response(call=("consume", {}), partial=True)
                assert state() == baseline[0], "Partial call caused a graph write"
                assert not executed, "Partial call was executed"
                yield response(call=("consume", {"value": 7}))
            else:
                yield response("completed")
        runner, conversation = await make_runner(LlmAgent(name="streamer", model=Model(generate), tools=[consume]))
        events = []
        with sasy.session(policy=policy, backend="souffle"):
            async for event in runner.run_async(user_id="user", session_id=conversation.id,
                new_message=user("streaming-input"), run_config=RunConfig(streaming_mode=StreamingMode.SSE)):
                events.append(event)
                if event.partial:
                    assert state() == baseline[0], "Consumer saw a graph write for a partial chunk"
            assert executed == [7]
            assert sum(bool(event.partial) for event in events) == 2
            assert not any(event.text == "partial-one" for event in state().events)

    elif scenario.startswith("transfer-"):
        allowed = scenario == "transfer-allowed"
        recipient_inputs, executed = [], []
        def consume(value: int):
            """Consume a transferred synthetic request."""
            graphs = assert_call_observed("consume", {"value": value})
            assert any(tool.name == "transfer_to_agent" for graph in graphs for node in graph.nodes for tool in node.tools)
            actual_texts = [part.text for content in recipient_inputs[0].contents for part in content.parts or [] if part.text]
            observed_texts = {node.text for graph in graphs for node in graph.nodes}
            assert actual_texts and set(actual_texts) <= observed_texts, "Recipient input presentations are absent from ancestry"
            executed.append(value)
            return {"value": value}
        async def sender(request, stream):
            if any(part.function_response for content in request.contents for part in content.parts or []):
                yield response("Transfer was denied; stayed with sender.")
            else:
                yield response(call=("transfer_to_agent", {"agent_name": "recipient"}))
        async def recipient(request, stream):
            recipient_inputs.append(request.model_copy(deep=True))
            if any(part.function_response and part.function_response.name == "consume"
                   for content in request.contents for part in content.parts or []):
                yield response("Recipient complete.")
            else:
                yield response(call=("consume", {"value": 9}))
        receiver = LlmAgent(name="recipient", description="Consumes transferred requests", model=Model(recipient), tools=[consume])
        root = LlmAgent(name="sender", model=Model(sender), sub_agents=[receiver])
        runner, conversation = await make_runner(root)
        selected = policy if allowed else policy + 'Unauthorized(idx) :- Actions(idx, a), IsTool(a, "transfer_to_agent").\n'
        before_ids = {event.id for event in state().events}
        with sasy.session(policy=selected, backend="souffle"):
            events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                              new_message=user("transfer-input"))]
            persisted = await runner.session_service.get_session(app_name="native_probe", user_id="user", session_id=conversation.id)
            if allowed:
                assert executed == [9] and recipient_inputs
                assert any(event.actions.transfer_to_agent == "recipient" for event in events)
            else:
                assert not executed and not recipient_inputs, "Denied transfer entered the recipient"
                assert all(not event.actions.transfer_to_agent and not event.actions.transfer_reason
                           for event in events + persisted.events), "Denied transfer mutated control actions"
                assert not any(event.author == "recipient" for event in events + persisted.events)
                assert not any(event.HasField("derived_from") and event.derived_from.name == "transfer_to_agent"
                               for event in state().events if event.id not in before_ids), "Denied transfer fabricated ToolResult evidence"

    elif scenario == "concurrent":
        reached = 0
        both_ready = asyncio.Event()
        outputs = {}
        async def generate(request, stream):
            nonlocal reached
            label = next(part.text for content in request.contents for part in content.parts or [] if part.text)
            reached += 1
            if reached == 2:
                both_ready.set()
            await asyncio.wait_for(both_ready.wait(), 20)
            yield response(label + "-answer")
        runner, first = await make_runner(LlmAgent(name="concurrent", model=Model(generate)))
        second = await runner.session_service.create_session(app_name="native_probe", user_id="user")
        async def turn(conversation, label):
            with sasy.session(session_id=str(uuid4()), policy=policy, backend="souffle"):
                events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                                   new_message=user(label))]
                answer = next(event for event in state().events if event.text == label + "-answer")
                graph = ancestry(answer.id)
                assert any(node.text == label for node in graph.nodes)
                assert not any(node.text.startswith("concurrent-") and node.text not in (label, label + "-answer") for node in graph.nodes)
                assert events
                outputs[label] = answer.id
            assert adk._active.get() is None and not _current_input_ids.get()
        await asyncio.gather(turn(first, "concurrent-a"), turn(second, "concurrent-b"))
        assert len(set(outputs.values())) == 2

    elif scenario == "otel":
        import grpc
        from opentelemetry import trace
        from sasy.instrumentation.otel import OTelConfig, configure_otel

        configure_otel(OTelConfig(use_logfire=False, capture_logs=False, attach_logs_to_spans=False))
        provider = trace.get_tracer_provider()
        roots = []
        for label in ("otel-left", "otel-right"):
            def consume(value: str):
                """Consume a synthetic traced value."""
                assert_call_observed("consume", {"value": value})
                return {"value": value}
            async def generate(request, stream):
                if any(part.function_response for content in request.contents for part in content.parts or []):
                    yield response("trace-finished")
                else:
                    yield LlmResponse(content=types.Content(role="model", parts=[
                        types.Part(text="about to consume"),
                        types.Part(function_call=types.FunctionCall(name="consume", args={"value": label}, id=label)),
                    ]))
            runner, conversation = await make_runner(LlmAgent(name=label.replace("-", "_"), model=Model(generate), tools=[consume]))
            sid = str(uuid4())
            with sasy.session(session_id=sid, policy=policy, end_on_exit=False, backend="souffle"):
                with trace.get_tracer("native_probe").start_as_current_span(label) as span:
                    events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                                       new_message=user(label))]
                    assert events
                    roots.append((sid, format(span.get_span_context().span_id, "016x"),
                                  format(span.get_span_context().trace_id, "032x")))
        assert await asyncio.to_thread(provider.force_flush)
        for sid, span_id, trace_id in roots:
            stored = graph_client.GetSpan(obs.SpanRequest(span_id=span_id, session_id=sid), metadata=metadata, timeout=15)
            assert json.loads(stored.attributes_json)["sasy.session_id"] == sid
            graph = graph_client.GetTrace(obs.TraceRequest(trace_id=trace_id), metadata=metadata, timeout=15)
            operations = [item for item in graph.computations if json.loads(item.attributes_json).get("sasy.operation")]
            assert operations and all(json.loads(item.attributes_json)["sasy.session_id"] == sid for item in operations)
            assert any(item.name == "adk.output" for item in operations), "A multi-part output lost its span links"
            assert {item.output_message_id for item in operations if item.HasField("output_message_id")} <= {item.id for item in graph.messages}
            other = next(other_sid for other_sid, _, _ in roots if other_sid != sid)
            with pytest.raises(grpc.RpcError) as missing:
                graph_client.GetSpan(obs.SpanRequest(span_id=span_id, session_id=other), metadata=metadata, timeout=15)
            assert missing.value.code() == grpc.StatusCode.NOT_FOUND
        provider.shutdown()

    elif scenario.startswith(("state-", "artifact-", "forwarded-state-", "forwarded-artifact-")):
        from google.adk.agents import SequentialAgent

        approved = scenario.endswith("-approval") and not scenario.endswith("-no-approval")
        artifact = "artifact-" in scenario
        forwarded = scenario.startswith("forwarded-")
        executed = []
        def approve():
            """Record a synthetic approval."""
            return {"approved": True}
        def consume(value: str):
            """Consume the decision only when its ancestry contains approval."""
            graphs = assert_call_observed("consume", {"value": value})
            assert any(node.HasField("derived_from") and node.derived_from.name == "approve"
                       for graph in graphs for node in graph.nodes)
            assert not any("never-read-state" in node.text for graph in graphs for node in graph.nodes)
            assert not any("never-read-artifact" in node.text for graph in graphs for node in graph.nodes)
            executed.append(value)
            return {"consumed": value}
        writer_calls = reader_calls = 0
        async def save_report(value: str):
            """Save an artifact from the observed tool invocation."""
            assert_call_observed("save_report", {"value": value})
            version = await runner.artifact_service.save_artifact(app_name="native_probe", user_id="user",
                session_id=conversation.id, filename="decision.txt", artifact=types.Part(text=value))
            return {"version": version}
        async def write(request, stream):
            nonlocal writer_calls
            writer_calls += 1
            if approved and writer_calls == 1:
                yield response(call=("approve", {}))
            elif artifact and writer_calls == (2 if approved else 1):
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name="save_report", args={"value": "approved"}, id="save-report"))]))
            else:
                yield response("approved")
        async def read(request, stream):
            nonlocal reader_calls
            reader_calls += 1
            assert "approved" in str(request.config.system_instruction)
            if reader_calls == 1:
                yield response(call=("consume", {"value": "approved"}))
            else:
                yield response("decision-consumed")
        writer = LlmAgent(name="writer", model=Model(write), tools=[approve, save_report] if artifact else [approve],
                          output_key=None if artifact else "decision")
        reader = LlmAgent(name="reader", model=Model(read), tools=[consume],
                          include_contents="none", instruction="Decision: {artifact.decision.txt}" if artifact else "Decision: {decision}",
                          before_model_callback=without_history)
        if forwarded:
            from google.adk.tools.agent_tool import AgentTool
            delegation_calls = 0
            async def delegate(request, stream):
                nonlocal delegation_calls
                delegation_calls += 1
                yield response(call=("reader", {"request": "Consume the decision"})) if delegation_calls == 1 else response("finished")
            reader = LlmAgent(name="parent", model=Model(delegate), tools=[AgentTool(reader)],
                              instruction="Delegate to the reader.", before_model_callback=without_history)
        runner = InMemoryRunner(agent=SequentialAgent(name="workflow", sub_agents=[writer, reader]),
                                app_name="native_probe")
        conversation = await runner.session_service.create_session(app_name="native_probe", user_id="user",
                                                                    state={"unrelated": "never-read-state"})
        if artifact:
            await runner.artifact_service.save_artifact(app_name="native_probe", user_id="user", session_id=conversation.id,
                filename="unrelated.txt", artifact=types.Part(text="never-read-artifact"))
        state_policy = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "approve").\n'
        state_policy += 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "save_report").\n'
        state_policy += 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "reader").\n'
        state_policy += 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "consume"), '
        state_policy += 'CurrentDepends(source), ToolResult(source, "approve", _).\n'
        with sasy.session(policy=state_policy, backend="souffle"):
            events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                               new_message=user("Make a decision"))]
            assert executed == (["approved"] if approved else [])
            assert reader_calls == 2
            assert events

    elif scenario.startswith("persisted-artifact-"):
        import tempfile

        from google.adk.artifacts import FileArtifactService
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        approved = not scenario.endswith("-no-approval")
        executed = []
        holder = {}
        def approve():
            """Record a synthetic approval."""
            return {"approved": True}
        async def save_report(value: str):
            """Save a user-scoped artifact that outlives this runner."""
            version = await holder["runner"].artifact_service.save_artifact(app_name="native_probe", user_id="user",
                session_id=holder["conversation"].id, filename="user:decision.txt", artifact=types.Part(text=value))
            return {"version": version}
        def consume(value: str):
            """Consume the decision only when its ancestry contains approval."""
            graphs = assert_call_observed("consume", {"value": value})
            assert any(node.HasField("derived_from") and node.derived_from.name == "approve"
                       for graph in graphs for node in graph.nodes)
            assert not any("unattributed external input" in node.text for graph in graphs for node in graph.nodes)
            executed.append(value)
            return {"consumed": value}
        writer_calls = reader_calls = 0
        async def write(request, stream):
            nonlocal writer_calls
            writer_calls += 1
            if approved and writer_calls == 1:
                yield response(call=("approve", {}))
            elif writer_calls == (2 if approved else 1):
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name="save_report", args={"value": "approved"}, id="save-report"))]))
            else:
                yield response("saved")
        async def read(request, stream):
            nonlocal reader_calls
            reader_calls += 1
            assert "approved" in str(request.config.system_instruction)
            yield response(call=("consume", {"value": "approved"})) if reader_calls == 1 else response("decision-consumed")
        async def file_runner(agent, root):
            # A new service object and runner: nothing about the artifact is
            # known in this process except what the store and the engine hold.
            made = Runner(app_name="native_probe", agent=agent, session_service=InMemorySessionService(),
                          artifact_service=FileArtifactService(root))
            return made, await made.session_service.create_session(app_name="native_probe", user_id="user")
        persisted_policy = ''.join('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "' + name + '").\n'
                                   for name in ("approve", "save_report"))
        persisted_policy += ('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "consume"), '
                             'CurrentDepends(source), ToolResult(source, "approve", _).\n')
        with tempfile.TemporaryDirectory() as root, sasy.session(policy=persisted_policy, backend="souffle"):
            holder["runner"], holder["conversation"] = await file_runner(
                LlmAgent(name="writer", model=Model(write), tools=[approve, save_report]), root)
            assert [event async for event in holder["runner"].run_async(user_id="user",
                session_id=holder["conversation"].id, new_message=user("Make a decision"))]
            reader, reading = await file_runner(LlmAgent(name="reader", model=Model(read), tools=[consume],
                include_contents="none", instruction="Decision: {artifact.user:decision.txt}",
                before_model_callback=without_history), root)
            assert [event async for event in reader.run_async(user_id="user", session_id=reading.id,
                                                              new_message=user("Use the decision"))]
            assert executed == (["approved"] if approved else [])
            assert reader_calls == 2

    elif scenario == "concurrent-workflow-artifact":
        from tempfile import TemporaryDirectory

        from google.adk.artifacts import FileArtifactService
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools import ToolContext
        from google.adk.workflow import JoinNode, Workflow
        from sasy.reference_monitor import api as monitor

        published, release_writer, writer_completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        verdicts = []
        writer_calls = reader_calls = 0

        def read_sensitive():
            """Read synthetic sensitive input for the writer only."""
            return {"text": "sensitive synthetic input"}

        def read_untrusted():
            """Read synthetic untrusted input for the writer only."""
            return {"text": "untrusted synthetic input"}

        async def publish_report(tool_context: ToolContext):
            """Publish the report but withhold successful tool completion."""
            version = await tool_context.save_artifact("report.txt", types.Part(text="constant report"))
            assert version == 0
            published.set()
            await asyncio.wait_for(release_writer.wait(), 30)
            return {"published": version}

        async def inspect_report(tool_context: ToolContext):
            """Read a sibling's published artifact before and after its tool returns."""
            await asyncio.wait_for(published.wait(), 30)
            for completed in (False, True):
                if completed:
                    release_writer.set()
                    await asyncio.wait_for(writer_completed.wait(), 30)
                part = await tool_context.load_artifact("report.txt")
                assert part.text == "constant report"
                verdict = await monitor.check_tool_call_async("consume", "{}", [])
                verdicts.append(verdict.authorized)
                assert verdict.authorized is completed
                graphs = [ancestry(node) for node in _current_input_ids.get()]
                results = {node.derived_from.name for graph in graphs for node in graph.nodes
                           if node.HasField("derived_from")}
                assert {"read_sensitive", "read_untrusted"} <= results
                assert ("publish_report" in results) is completed
                leak = await monitor.check_tool_call_async("external_send", "{}", [])
                assert not leak.authorized, "Artifact lost sensitive/untrusted producer ancestry"
            return {"checked": True}

        async def write(request, stream):
            nonlocal writer_calls
            writer_calls += 1
            calls = ("read_sensitive", "read_untrusted", "publish_report")
            if writer_calls <= len(calls):
                name = calls[writer_calls - 1]
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name=name, args={}, id=name))]))
            else:
                writer_completed.set()
                yield response("writer complete")

        async def read(request, stream):
            nonlocal reader_calls
            reader_calls += 1
            if reader_calls == 1:
                yield response(call=("inspect_report", {}))
            else:
                yield response("reader complete")

        async def start(request, stream):
            yield response("start both branches")

        writer = LlmAgent(name="writer", model=Model(write),
                          tools=[read_sensitive, read_untrusted, publish_report], include_contents="none")
        reader = LlmAgent(name="reader", model=Model(read), tools=[inspect_report], include_contents="none")
        workflow = Workflow(name="concurrent_resources", edges=[("START",
            LlmAgent(name="coordinator", model=Model(start)), (writer, reader), JoinNode(name="join"))])
        selected_policy = ''.join('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "' + name + '").\n'
                                  for name in ("read_sensitive", "read_untrusted", "publish_report", "inspect_report", "external_send"))
        selected_policy += ('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "consume"), '
                            'CurrentDepends(source), ToolResult(source, "publish_report", _).\n')
        selected_policy += ('Unauthorized(idx) :- Actions(idx, a), IsTool(a, "external_send"), '
                            'CurrentDepends(sensitive), ToolResult(sensitive, "read_sensitive", _), '
                            'CurrentDepends(untrusted), ToolResult(untrusted, "read_untrusted", _).\n')
        with TemporaryDirectory(prefix="workflow-artifacts-") as directory:
            runner = Runner(node=workflow, app_name="native_probe", session_service=InMemorySessionService(),
                            artifact_service=FileArtifactService(directory))
            conversation = await runner.session_service.create_session(app_name="native_probe", user_id="user")
            with sasy.session(policy=selected_policy, backend="souffle"):
                async def run():
                    return [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                                     new_message=user("publish and inspect"))]
                events = await asyncio.wait_for(run(), 90)
                assert events and verdicts == [False, True]

    elif scenario.startswith(("agent-tool-", "context-state-", "context-artifact-", "native-single-turn-", "native-task-", "workflow-state-", "workflow-artifact-")):
        from google.adk.agents import SequentialAgent
        from google.adk.tools import ToolContext
        from google.adk.tools.agent_tool import AgentTool
        from sasy.reference_monitor import api as monitor

        failed_approval = scenario.endswith("-error")
        denied_launch = scenario.endswith("-denied")
        approved = not (scenario.endswith("no-approval") or failed_approval or denied_launch)
        mode = "single_turn" if scenario.startswith("native-single-turn-") else "task" if scenario.startswith("native-task-") else None
        delegated = scenario.startswith("agent-tool-") or mode is not None
        artifact = scenario.startswith(("context-artifact-", "workflow-artifact-"))
        workflow_resources = scenario.startswith(("workflow-state-", "workflow-artifact-"))
        executed = []
        writer_calls = reader_calls = 0

        async def approve(tool_context: ToolContext):
            """Write and approve a synthetic decision, or report an error."""
            if not delegated:
                await save_decision("same decision", tool_context)
            return {"error": "approval failed"} if failed_approval else {"approved": True}

        async def save_decision(value: str, tool_context: ToolContext):
            """Write a decision through the native tool context."""
            if artifact:
                await tool_context.save_artifact("decision.txt", types.Part(text=value))
            else:
                tool_context.state["decision"] = value
            return {"saved": True}

        async def consume_decision(tool_context: ToolContext):
            """Authorize a protected effect after actually reading its input."""
            value = ((await tool_context.load_artifact("decision.txt")).text if artifact
                     else tool_context.state["decision"])
            verdict = await monitor.check_tool_call_async("consume", json.dumps({"value": value}), [])
            if workflow_resources:
                graphs = [ancestry(node) for node in _current_input_ids.get()]
                assert any('"adk_resource"' in node.text and
                           ('"decision.txt"' if artifact else '"decision"') in node.text
                           for graph in graphs for node in graph.nodes)
            if verdict.authorized:
                graphs = [ancestry(node) for node in _current_input_ids.get()]
                assert any(node.HasField("derived_from") and node.derived_from.name == "approve"
                           for graph in graphs for node in graph.nodes)
                assert not any("never-read" in node.text for graph in graphs for node in graph.nodes)
                executed.append(value)
            return {"allowed": verdict.authorized}

        def consume(value: str):
            """Consume a delegated response after policy authorization."""
            assert_call_observed("consume", {"value": value})
            executed.append(value)
            return {"consumed": value}

        async def write(request, stream):
            nonlocal writer_calls
            writer_calls += 1
            if (approved or failed_approval) and writer_calls == 1:
                yield response(call=("approve", {}))
            elif not delegated and not (approved or failed_approval) and writer_calls == 1:
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name="save_decision", args={"value": "same decision"}, id="save-decision"))]))
            elif mode == "task":
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name="finish_task", args={"result": "same decision"}, id="finish-task"))]))
            else:
                yield response("same decision")

        async def read(request, stream):
            nonlocal reader_calls
            reader_calls += 1
            if delegated and reader_calls == 1:
                yield response(call=("reviewer", {"request": "Evaluate the decision"}))
            elif reader_calls == (2 if delegated else 1):
                name, arguments = ("consume", {"value": "same decision"}) if delegated else ("consume_decision", {})
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=
                    types.FunctionCall(name=name, args=arguments, id="consume-decision"))]))
            else:
                yield response("finished")

        writer = LlmAgent(name="reviewer", model=Model(write), tools=[approve] if delegated else [approve, save_decision])
        if mode is not None:
            writer = LlmAgent(name="reviewer", mode=mode, model=Model(write), tools=[approve], description="Evaluate a decision.")
            root = LlmAgent(name="parent", model=Model(read), sub_agents=[writer], tools=[consume])
        elif delegated:
            root = LlmAgent(name="parent", model=Model(read), tools=[AgentTool(writer), consume])
        else:
            reader = LlmAgent(name="reader", model=Model(read), tools=[consume_decision], include_contents="none",
                              instruction="Read the decision using consume_decision.", before_model_callback=without_history)
            if workflow_resources:
                from google.adk.workflow import JoinNode, Workflow
                async def branch_model(request, stream):
                    yield response("stateless branch complete")
                reader.before_model_callback = None
                root = Workflow(name="resources", edges=[("START", writer,
                    (LlmAgent(name="left", model=Model(branch_model)),
                     LlmAgent(name="right", model=Model(branch_model))),
                    JoinNode(name="merge"), reader)])
            else:
                root = SequentialAgent(name="sequence", sub_agents=[writer, reader])
        runner = (InMemoryRunner(node=root, app_name="native_probe") if workflow_resources
                  else InMemoryRunner(agent=root, app_name="native_probe"))
        conversation = await runner.session_service.create_session(app_name="native_probe", user_id="user",
            state={} if delegated else {"unrelated": "never-read-state"})
        if artifact:
            await runner.artifact_service.save_artifact(app_name="native_probe", user_id="user", session_id=conversation.id,
                filename="unused.txt", artifact=types.Part(text="never-read-artifact"))
        selected_policy = ''.join('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "' + name + '").\n'
                                  for name in ("approve", "reviewer", "save_decision", "consume_decision", "finish_task"))
        selected_policy += ('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "consume"), '
                            'CurrentDepends(source), ToolResult(source, "approve", _).\n')
        if denied_launch:
            selected_policy += 'Unauthorized(idx) :- Actions(idx, a), IsTool(a, "reviewer").\n'
        with sasy.session(policy=selected_policy, backend="souffle"):
            events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                               new_message=user("Make a decision"))]
            assert events
            assert executed == (["same decision"] if approved else [])
            assert reader_calls == (3 if delegated else 2)
            if denied_launch:
                assert writer_calls == 0, "Denied delegation launched its child"

    elif scenario == "cancelled-stream":
        entered, block = asyncio.Event(), asyncio.Event()
        model_calls = 0
        async def generate(request, stream):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                yield response("unfinished-partial", partial=True)
                entered.set()
                await block.wait()
            else:
                yield response("after-cancellation")
        runner, conversation = await make_runner(LlmAgent(name="cancelled", model=Model(generate)))
        with sasy.session(policy=policy, backend="souffle"):
            async def consume_stream():
                try:
                    async for _ in runner.run_async(user_id="user", session_id=conversation.id,
                        new_message=user("cancelled-input"), run_config=RunConfig(streaming_mode=StreamingMode.SSE)):
                        pass
                finally:
                    assert adk._active.get() is None and not _current_input_ids.get(), "Cancellation leaked task context"
            task = asyncio.create_task(consume_stream())
            await asyncio.wait_for(entered.wait(), 20)
            task.cancel()
            try:
                await task
                raise AssertionError("Cancelled turn did not propagate cancellation")
            except asyncio.CancelledError:
                pass
            assert not any(event.text == "unfinished-partial" for event in state().events)
            events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                new_message=user("retry-input"), run_config=RunConfig(streaming_mode=StreamingMode.SSE))]
            assert any(event.content and any(part.text == "after-cancellation" for part in event.content.parts or []) for event in events)
            assert adk._active.get() is None and not _current_input_ids.get()
    elif scenario in ("workflow-route-denied", "workflow-route-third-denied"):
        from google.adk.workflow import Workflow

        executed = []

        def select_route(tool_context):
            """Route according to the current session decision."""
            selected = tool_context.state["decision"]
            tool_context.actions.route = selected
            return {"selected": selected}

        def finalize():
            """Finalize the selected route."""
            executed.append(True)
            return {"finalized": True}

        async def router_model(request, stream):
            if any(part.function_response for content in request.contents for part in content.parts or []):
                yield response("constant")
            else:
                yield response(call=("select_route", {}))

        async def handler_model(request, stream):
            yield response("constant")

        async def finalizer_model(request, stream):
            if any(part.function_response for content in request.contents for part in content.parts or []):
                yield response("done")
            else:
                yield response(call=("finalize", {}))

        router = LlmAgent(name="router", model=Model(router_model), tools=[select_route])
        approved = LlmAgent(name="approved", model=Model(handler_model))
        rejected = LlmAgent(name="rejected", model=Model(handler_model))
        held = LlmAgent(name="held", model=Model(handler_model))
        finalizer = LlmAgent(name="finalizer", model=Model(finalizer_model), tools=[finalize])
        targets = {"approved": approved, "rejected": rejected}
        edges = [(approved, finalizer), (rejected, finalizer)]
        selected = "approved"
        if scenario == "workflow-route-third-denied":
            targets["held"] = held
            edges.append((held, finalizer))
            selected = "held"
        workflow = Workflow(name="conditional", edges=[("START", router, targets), *edges])
        runner = InMemoryRunner(node=workflow, app_name="native_probe")
        conversation = await runner.session_service.create_session(
            app_name="native_probe", user_id="user", state={"decision": selected})
        route_policy = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "select_route").\n'
        route_policy += 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "finalize").\n'
        route_policy += ('Unauthorized(idx) :- Actions(idx, a), IsTool(a, "finalize"), '
                         'CurrentDepends(source), ToolResult(source, "select_route", _).\n')
        before_ids = {event.id for event in state().events}
        with sasy.session(policy=route_policy, backend="souffle"):
            events = [event async for event in runner.run_async(
                user_id="user", session_id=conversation.id, new_message=user("route it"))]
            assert events and len(targets[selected].model._requests) == 1
            assert not any(agent.model._requests for route, agent in targets.items() if route != selected)
            assert not executed
            call = next(event for event in state().events
                        if event.id not in before_ids
                        and any(tool.name == "finalize" for tool in event.tools))
            lineage = ancestry(call.id).nodes
            assert any(node.HasField("derived_from") and node.derived_from.name == "select_route"
                       for node in lineage), "Final action lost the exclusive route's tool result"
            assert any('"adk_resource"' in node.text and '"decision"' in node.text
                       for node in lineage), "Final action lost the router's state read"
            assert not any(node.HasField("derived_from") and node.derived_from.name == "finalize"
                           for node in state().events if node.id not in before_ids), (
                "Policy denied the action after it executed")

    elif scenario.startswith("nested-argument-"):
        # A tool parameter ADK builds into a Pydantic model still reaches the
        # policy as plain JSON, under the field names the tool declares, so an
        # ordinary rule on a nested field decides the call.
        from pydantic import BaseModel

        class Destination(BaseModel):
            account: str

        blocked = scenario == "nested-argument-denied"
        account = "blocked" if blocked else "allowed"
        executed = []
        def pay(destination: Destination):
            """Pay a destination."""
            executed.append(destination.account)
            return {"paid": destination.account}
        async def generate(request, stream):
            if any(part.function_response for content in request.contents for part in content.parts or []):
                yield response("settled")
            else:
                yield response(call=("pay", {"destination": {"account": account}}))
        runner, conversation = await make_runner(LlmAgent(name="payer", model=Model(generate), tools=[pay]))
        nested = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "pay").\n'
        nested += ('Unauthorized(idx) :- Actions(idx, a), a = $CallTool("pay", args), '
                   '@json_get_str_path(args, "destination.account") = "blocked".\n')
        with sasy.session(policy=nested, backend="souffle"):
            events = [event async for event in runner.run_async(user_id="user", session_id=conversation.id,
                                                                new_message=user("pay-input"))]
            assert events
            assert executed == ([] if blocked else ["allowed"])
            # The call is recorded either way; only the allowed one establishes
            # a result. The tenant graph outlives one scenario, so each lane
            # looks for its own arguments alone.
            expected = '{"destination":{"account":"' + account + '"}}'
            recorded = [tool.arguments for event in state().events for tool in event.tools if tool.name == "pay"]
            assert expected in recorded, recorded
            results = [event.derived_from.arguments for event in state().events
                       if event.HasField("derived_from") and event.derived_from.name == "pay"]
            assert (expected in results) is not blocked, results

    else:
        raise AssertionError(f"Unknown native ADK probe: {scenario}")
    print(json.dumps({"native_adk": scenario, "passed": True}))


if __name__ == "__main__":
    import asyncio
    asyncio.run(_native_probe(sys.argv[1]))
