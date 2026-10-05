"""Exact text artifact production/consumption on native ADK services."""
import asyncio
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

if os.environ.get("SASY_REQUIRE_FRAMEWORKS") == "1":
    import google.adk  # noqa: F401
else:
    pytest.importorskip("google.adk")
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from sasy.instrumentation import adk_artifacts
from sasy.instrumentation.adk_state import EXTERNAL_SESSION_ORIGIN
from test_adk_instrumentation import (  # noqa: F401
    AdkInstrumentationError,
    LlmAgent,
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


async def _turn(runner, sess, message="request"):
    return [e async for e in runner.run_async(user_id="u", session_id=sess.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]))]


@pytest.mark.parametrize("approved", [True, False])
def test_save_template_preserves_actual_producer_ancestry(sink, approved):
    async def run():
        runner = None
        sess = None
        def approve():
            """Approve the request."""
            return "approved"
        async def save_text():
            """Save text."""
            return await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename="decision", artifact=types.Part(text="approved"))
        def consume():
            """Consume decision."""
            return "consumed"
        script = [calls(("approve", {}, "approve"))] if approved else []
        script += [calls(("save_text", {}, "save")), text("saved")]
        writer = LlmAgent(name="writer", model=ScriptedModel(script), tools=[approve, save_text])
        reader = LlmAgent(name="reader", model=ScriptedModel([calls(("consume", {}, "consume")), text("done")]),
            instruction="Decision {artifact.decision}", include_contents="none", tools=[consume])
        runner = InMemoryRunner(agent=SequentialAgent(name="pipeline", sub_agents=[writer, reader]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        check = next(c for c in sink.checks if c[0] == "consume")
        lineage = ancestors(sink, check[2])
        assert any("observed production" in e.text and "approved" in e.text for e in lineage)
        assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == approved
    asyncio.run(run())


def test_load_artifacts_tool_tracks_selected_content_and_gates_dispatch(sink):
    async def run():
        model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["selected"]}, "load")), text("answer")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, tools=[LoadArtifactsTool()]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        for name, content in [("selected", "selected contents"), ("unread", "unread secret")]:
            await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename=name, artifact=types.Part(text=content))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        assert [c[0] for c in sink.checks] == ["load_artifacts"]
        assert not any("unread secret" in e.text for e in sink.events.values())
        output = next(e for e in sink.events.values() if e.text == "answer")
        assert any("selected contents" in e.text for e in ancestors(sink, [output.id]))
        assert any("selected contents" in (p.text or "") for c in model._requests[1].contents for p in c.parts or [])
    asyncio.run(run())


@pytest.mark.parametrize("transform", [False, True])
def test_denied_load_does_not_present_contents_or_success(sink, transform):
    sink.allowed = transform
    sink.transforms = ["unsupported"] if transform else []
    async def run():
        model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["secret"]}, "load")), text("denied")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, tools=[LoadArtifactsTool()]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="secret", artifact=types.Part(text="hidden contents"))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        assert not any("hidden contents" in e.text for e in sink.events.values())
        assert not any(e.HasField("derived_from") for e in sink.events.values())
    asyncio.run(run())


def test_tool_read_updates_nested_action_inputs_and_returns_copy(sink):
    async def run():
        runner = None
        sess = None
        async def load_text():
            """Read an artifact."""
            part = await runner.artifact_service.load_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename="value")
            assert any("artifact" in e.text and "actual" in e.text for e in ancestors(sink, adk._current_input_ids.get()))
            part.text = "locally changed"
            return part.text
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=ScriptedModel([calls(("load_text", {}, "read")), text("done")]),
            tools=[load_text]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="value", artifact=types.Part(text="actual"))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        stored = await runner.artifact_service.load_artifact(app_name="test", user_id="u", session_id=sess.id, filename="value")
        assert stored.text == "actual"
        result = next(e for e in sink.events.values() if e.HasField("derived_from") and e.derived_from.name == "load_text")
        assert any("actual" in e.text for e in ancestors(sink, [result.id]))
    asyncio.run(run())


def test_artifact_alias_mutation_is_rejected_on_retained_read(sink):
    async def run():
        model = ScriptedModel([text("first")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, instruction="Value {artifact.value}"), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        alias = types.Part(text="original")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id, filename="value", artifact=alias)
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
            alias.text = "mutated"
            with pytest.raises(AdkInstrumentationError, match="without an observed producer"):
                await _turn(runner, sess, "again")
            assert len(model._requests) == 1
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["binary", "callback", "metadata", "delete", "versions", "cross_scope"])
def test_unqualified_artifact_operations_fail_closed(sink, mode):
    async def run():
        runner = None
        sess = None
        async def operation():
            """Use an artifact."""
            args = dict(app_name="test", user_id="u", session_id=sess.id, filename="value")
            service = runner.artifact_service
            if mode == "delete":
                return await service.delete_artifact(**args)
            if mode == "versions":
                return await service.list_versions(**args)
            if mode == "metadata":
                return await service.get_artifact_version(**args, version=0)
            if mode == "cross_scope":
                args["user_id"] = "other"
            return await service.load_artifact(**args)
        async def callback(callback_context, llm_request):
            await operation()
        model = ScriptedModel([calls(("operation", {}, "op"))])
        agent = LlmAgent(name="reader", model=model, tools=[operation],
            before_model_callback=callback if mode == "callback" else None)
        runner = InMemoryRunner(agent=agent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        part = types.Part(inline_data=types.Blob(data=b"data", mime_type="text/plain")) if mode == "binary" else types.Part(text="value")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id, filename="value", artifact=part)
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError):
            await _turn(runner, sess)
    asyncio.run(run())


def test_failed_observation_poisoned_saved_version_is_not_imported(sink):
    async def run():
        runner = None
        sess = None
        async def save_text():
            """Save text."""
            original = adk.observation.resolve_events_async
            async def fail_resource(snapshots):
                if any('"artifact-list"' in item.event.text and '"observed production"' in item.event.text for item in snapshots):
                    raise RuntimeError("resource observation failed")
                return await original(snapshots)
            adk.observation.resolve_events_async = fail_resource
            try:
                return await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                    filename="value", artifact=types.Part(text="produced"))
            finally:
                adk.observation.resolve_events_async = original
        model = ScriptedModel([calls(("save_text", {}, "save"))])
        agent = LlmAgent(name="writer", model=model, tools=[save_text])
        runner = InMemoryRunner(agent=agent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            with pytest.raises(RuntimeError, match="resource observation failed"):
                await _turn(runner, sess)
            agent.instruction = "Read {artifact.value}"
            with pytest.raises(AdkInstrumentationError, match="did not complete observation"):
                await _turn(runner, sess, "retry")
        assert not any("unattributed external input" in e.text and "produced" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_cancelled_save_names_its_unfinished_observation(sink):
    async def run():
        holder = SimpleNamespace(runner=None, sess=None, committed=asyncio.Event())

        class StallingService(InMemoryArtifactService):
            """Stores the version, then stalls so its caller can be cancelled mid-save."""
            async def save_artifact(self, *, app_name, user_id, filename, artifact,
                                    session_id=None, custom_metadata=None):
                stored = adk_artifacts._instrumented[InMemoryArtifactService]["save_artifact"]
                version = await stored(self, app_name=app_name, user_id=user_id, filename=filename,
                    artifact=artifact, session_id=session_id, custom_metadata=custom_metadata)
                holder.committed.set()
                await asyncio.Event().wait()
                return version

        async def save_text():
            """Save text."""
            return await holder.runner.artifact_service.save_artifact(app_name="test", user_id="u",
                session_id=holder.sess.id, filename="value", artifact=types.Part(text="produced"))
        async def load_text():
            """Load text."""
            part = await holder.runner.artifact_service.load_artifact(app_name="test", user_id="u",
                session_id=holder.sess.id, filename="value")
            return part.text if part else "missing"
        agent = LlmAgent(name="writer", model=ScriptedModel([calls(("save_text", {}, "save"))]),
            tools=[save_text, load_text])
        service = StallingService()
        holder.runner = Runner(app_name="test", agent=agent, session_service=InMemorySessionService(),
            artifact_service=service)
        holder.sess = await holder.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            saving = asyncio.create_task(_turn(holder.runner, holder.sess, "save"))
            await holder.committed.wait()
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await saving
            agent.model = ScriptedModel([calls(("load_text", {}, "load")), text("done")])
            with pytest.raises(AdkInstrumentationError, match="did not complete observation"):
                await _turn(holder.runner, holder.sess, "retry")
        assert not any("unattributed external input" in e.text and "produced" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_load_racing_an_outside_write_is_rejected(sink):
    async def run():
        holder = SimpleNamespace(runner=None, sess=None)

        class RacingService(InMemoryArtifactService):
            """Stores a further version from outside this session while a load is in flight."""
            async def load_artifact(self, *, app_name, user_id, filename, session_id=None, version=None):
                stored = adk_artifacts._instrumented[InMemoryArtifactService]
                await stored["save_artifact"](self, app_name=app_name, user_id=user_id, filename=filename,
                    artifact=types.Part(text="OUTSIDE WRITE"), session_id=session_id)
                return await stored["load_artifact"](self, app_name=app_name, user_id=user_id,
                    filename=filename, session_id=session_id, version=version)

        async def load_text():
            """Load text."""
            part = await holder.runner.artifact_service.load_artifact(app_name="test", user_id="u",
                session_id=holder.sess.id, filename="value")
            return part.text if part else "missing"
        model = ScriptedModel([calls(("load_text", {}, "load")), text("done")])
        service = RacingService()
        holder.runner = Runner(app_name="test", agent=LlmAgent(name="reader", model=model, tools=[load_text]),
            session_service=InMemorySessionService(), artifact_service=service)
        holder.sess = await holder.runner.session_service.create_session(app_name="test", user_id="u")
        await service.save_artifact(app_name="test", user_id="u", session_id=holder.sess.id,
            filename="value", artifact=types.Part(text="seeded"))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="changed without an observed save"):
            await _turn(holder.runner, holder.sess)
        assert len(model._requests) == 1
        assert not any("OUTSIDE WRITE" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_listing_racing_an_outside_write_is_rejected(sink):
    async def run():
        holder = SimpleNamespace(runner=None, sess=None, captured=False)

        class RacingService(InMemoryArtifactService):
            """Stores a further file from outside this session once the catalog is captured."""
            async def list_versions(self, *, app_name, user_id, filename, session_id=None):
                holder.captured = True
                return await adk_artifacts._instrumented[InMemoryArtifactService]["list_versions"](
                    self, app_name=app_name, user_id=user_id, filename=filename, session_id=session_id)

            async def list_artifact_keys(self, *, app_name, user_id, session_id=None):
                stored = adk_artifacts._instrumented[InMemoryArtifactService]
                if holder.captured:
                    holder.captured = False
                    await stored["save_artifact"](self, app_name=app_name, user_id=user_id, filename="outside",
                        artifact=types.Part(text="OUTSIDE WRITE"), session_id=session_id)
                return await stored["list_artifact_keys"](self, app_name=app_name, user_id=user_id,
                    session_id=session_id)

        async def list_text():
            """List saved files."""
            return await holder.runner.artifact_service.list_artifact_keys(app_name="test", user_id="u",
                session_id=holder.sess.id)
        model = ScriptedModel([calls(("list_text", {}, "list")), text("done")])
        service = RacingService()
        holder.runner = Runner(app_name="test", agent=LlmAgent(name="reader", model=model, tools=[list_text]),
            session_service=InMemorySessionService(), artifact_service=service)
        holder.sess = await holder.runner.session_service.create_session(app_name="test", user_id="u")
        await service.save_artifact(app_name="test", user_id="u", session_id=holder.sess.id,
            filename="value", artifact=types.Part(text="seeded"))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="listing changed"):
            await _turn(holder.runner, holder.sess)
        assert len(model._requests) == 1
        assert not any("outside" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_load_response_callback_cannot_select_unchecked_artifact(sink):
    async def callback(tool, args, tool_context, tool_response):
        return {"artifact_names": ["secret"]}
    async def run():
        model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["public"]}, "load")), text("must not run")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, tools=[LoadArtifactsTool()],
            after_tool_callback=callback), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="secret", artifact=types.Part(text="hidden contents"))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="changed after authorized dispatch"):
            await _turn(runner, sess)
        assert len(model._requests) == 1
        assert not any("hidden contents" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_cross_graph_artifact_producer_enters_as_external_input(sink):
    async def run():
        runner = None
        sess = None
        async def save_text():
            """Save text."""
            return await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename="user:value", artifact=types.Part(text="produced"))
        model = ScriptedModel([calls(("save_text", {}, "save")), text("done")])
        agent = LlmAgent(name="writer", model=model, tools=[save_text])
        runner = InMemoryRunner(agent=agent, app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        written = set(sink.events)
        agent.instruction = "Read {artifact.user:value}"
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        with session(end_on_exit=False):
            await _turn(runner, sess, "import")
        assert len(model._requests) == 3
        imported = [e for e in sink.events.values()
            if e.id not in written and EXTERNAL_SESSION_ORIGIN in e.text and "produced" in e.text]
        assert len(imported) == 1
        assert "unattributed external input" in imported[0].text
        assert not any(e.id in written for e in ancestors(sink, [imported[0].id]))
    asyncio.run(run())


def test_empty_artifact_preserves_native_missing_value_semantics(sink):
    async def run():
        model = ScriptedModel([text("done")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model,
            instruction="Optional: {artifact.empty?}"), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="empty", artifact=types.Part(text=""))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        assert any('"present":false' in e.text and '"empty"' in e.text for e in sink.events.values())
        assert "Optional:" in model._requests[0].config.system_instruction
    asyncio.run(run())


def test_child_task_artifact_read_rejects_before_caller_can_lose_context(sink):
    async def run():
        runner = None
        sess = None
        effects = []
        async def copy_text():
            """Copy an artifact."""
            part = await asyncio.create_task(runner.artifact_service.load_artifact(
                app_name="test", user_id="u", session_id=sess.id, filename="source"))
            effects.append(part.text)
            return await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename="destination", artifact=part)
        runner = InMemoryRunner(agent=LlmAgent(name="copier", model=ScriptedModel([calls(("copy_text", {}, "copy"))]),
            tools=[copy_text]), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="source", artifact=types.Part(text="source"))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="child task"):
            await _turn(runner, sess)
        assert not effects
        assert await runner.artifact_service.load_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="destination") is None
    asyncio.run(run())


@pytest.mark.parametrize("clone", [False, True])
def test_artifact_callback_filter_retains_exact_occurrence(sink, clone):
    def select(callback_context, llm_request):
        if any("Artifact selected is:" == part.text for content in llm_request.contents for part in content.parts or []):
            content = llm_request.contents[-1]
            llm_request.contents = [content.model_copy(deep=True) if clone else content]
    async def run():
        model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["selected"]}, "load")), text("answer")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, tools=[LoadArtifactsTool()],
            before_model_callback=select), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="selected", artifact=types.Part(text="selected contents"))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
            assert len(model._requests[1].contents) == 1
            output = next(e for e in sink.events.values() if e.text == "answer")
            assert any("selected contents" in e.text for e in ancestors(sink, [output.id]))
    asyncio.run(run())


@pytest.mark.parametrize("retain", ["none", "first"])
def test_filtered_artifact_bodies_do_not_taint_inventory_or_other_bodies(sink, retain):
    def select(callback_context, llm_request):
        def artifact(content):
            return any((part.text or "").startswith("Artifact ") for part in content.parts or [])
        bodies = [c for c in llm_request.contents if artifact(c)]
        if bodies:
            llm_request.contents = [c for c in llm_request.contents if not artifact(c)]
            if retain == "first":
                llm_request.contents.append(bodies[0])
    async def run():
        model = ScriptedModel([calls(("load_artifacts", {"artifact_names": ["first", "second"]}, "load")), text("answer")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model, tools=[LoadArtifactsTool()],
            before_model_callback=select), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        for name in ("first", "second"):
            await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
                filename=name, artifact=types.Part(text=name + " BODY"))
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(runner, sess)
        answer = next(e for e in sink.events.values() if e.text == "answer")
        lineage = ancestors(sink, [answer.id])
        assert not any("second BODY" in e.text for e in lineage)
        assert any("first BODY" in e.text for e in lineage) == (retain == "first")
        assert any("first BODY" in (part.text or "") for content in model._requests[1].contents
            for part in content.parts or []) == (retain == "first")
    asyncio.run(run())


def _file_runner(agent, root):
    """A runner whose artifacts outlive it in a directory, with its own ledger."""
    return Runner(app_name="test", agent=agent, session_service=InMemorySessionService(),
        artifact_service=FileArtifactService(root))


def _writer(root):
    holder = SimpleNamespace(runner=None, sess=None)
    def approve():
        """Approve the request."""
        return "approved"
    async def save_text():
        """Save text."""
        return await holder.runner.artifact_service.save_artifact(app_name="test", user_id="u",
            session_id=holder.sess.id, filename="user:decision", artifact=types.Part(text="approved"))
    model = ScriptedModel([calls(("approve", {}, "approve")), calls(("save_text", {}, "save")), text("saved")])
    holder.runner = _file_runner(LlmAgent(name="writer", model=model, tools=[approve, save_text]), root)
    return holder


def _reader(root):
    def consume():
        """Consume the decision."""
        return "consumed"
    model = ScriptedModel([calls(("consume", {}, "consume")), text("done")])
    return _file_runner(LlmAgent(name="reader", model=model, tools=[consume],
        instruction="Decision {artifact.user:decision}", include_contents="none"), root)


def _version_dir(root):
    return next(path.parent for path in root.rglob("metadata.json"))


def _forge(root, claim):
    path = _version_dir(root) / "metadata.json"
    stored = json.loads(path.read_text())
    stored["customMetadata"] = {"sasy_provenance": claim}
    path.write_text(json.dumps(stored))


def test_persisted_artifact_provenance_verifies_in_a_fresh_runner(sink, tmp_path):
    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(writer.runner, writer.sess)
            produced = set(sink.events)
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            await _turn(runner, sess, "read")
        check = next(c for c in sink.checks if c[0] == "consume")
        lineage = ancestors(sink, check[2])
        assert any(e.id in produced and "observed production" in e.text and "approved" in e.text for e in lineage)
        assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage)
        assert not any("unattributed external input" in e.text and "approved" in e.text for e in lineage)
    asyncio.run(run())


@pytest.mark.parametrize("saved,loaded,verified", [("alice", "alice", True), ("alice", "bob", True),
    ("alice", None, True), (None, None, True), (None, "bob", False)])
def test_persisted_provenance_replays_the_producer_entity(sink, tmp_path, saved, loaded, verified):
    from sasy.instrumentation.session import _entity_var

    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            token = _entity_var.set(saved)
            try:
                await _turn(writer.runner, writer.sess)
            finally:
                _entity_var.reset(token)
            produced = set(sink.events)
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            token = _entity_var.set(loaded)
            try:
                await _turn(runner, sess, "read")
            finally:
                _entity_var.reset(token)
        check = next(c for c in sink.checks if c[0] == "consume")
        lineage = ancestors(sink, check[2])
        assert any(e.id in produced and "approved" in e.text for e in lineage) == verified
        assert any(e.HasField("derived_from") and e.derived_from.name == "approve" for e in lineage) == verified
        assert any("unattributed external input" in e.text and "approved" in e.text for e in lineage) == (not verified)
    asyncio.run(run())


def test_persisted_provenance_from_another_session_is_an_external_input(sink, tmp_path):
    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(writer.runner, writer.sess)
        produced = set(sink.events)
        with session(end_on_exit=False):
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            await _turn(runner, sess, "read")
        check = next(c for c in sink.checks if c[0] == "consume")
        lineage = ancestors(sink, check[2])
        assert not any(e.id in produced for e in lineage)
        assert not any(e.HasField("derived_from") for e in lineage)
        assert any("unattributed external input" in e.text and "approved" in e.text for e in lineage)
    asyncio.run(run())


def test_observation_outage_does_not_downgrade_persisted_provenance(sink, tmp_path):
    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(writer.runner, writer.sess)
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            sink.failure = "references"
            with pytest.raises(RuntimeError, match="observation offline"):
                await _turn(runner, sess, "read")
        assert not any(c[0] == "consume" for c in sink.checks)
        assert not any("unattributed external input" in e.text and "approved" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_storage_failure_reading_a_provenance_claim_is_fatal(sink, tmp_path):
    """A failed metadata read decides nothing, so it must not silently downgrade."""
    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(writer.runner, writer.sess)
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            originals = adk_artifacts._originals(runner.artifact_service)
            stored = originals["get_artifact_version"]

            async def unavailable(*args, **kwargs):
                raise OSError("artifact metadata unavailable")

            originals["get_artifact_version"] = unavailable
            try:
                with pytest.raises(OSError, match="artifact metadata unavailable"):
                    await _turn(runner, sess, "read")
            finally:
                originals["get_artifact_version"] = stored
        assert not any(c[0] == "consume" for c in sink.checks)
        assert not any("unattributed external input" in e.text and "approved" in e.text
            for e in sink.events.values())
    asyncio.run(run())


@pytest.mark.parametrize("attack", ["tampered", "forged", "malformed", "absent"])
def test_unverifiable_persisted_provenance_enters_as_external_input(sink, tmp_path, attack):
    async def run():
        writer = _writer(tmp_path)
        writer.sess = await writer.runner.session_service.create_session(app_name="test", user_id="u")
        adk.instrument()
        with session(end_on_exit=False):
            await _turn(writer.runner, writer.sess)
            produced = set(sink.events)
            body = "approved"
            if attack == "tampered":
                body = "rejected"
                (_version_dir(tmp_path) / "decision").write_text(body)
            if attack == "forged":
                key = ("artifact", "test", "u", "user", "user:decision", 0)
                other = next(node for node, e in sink.events.items() if e.text == "saved")
                _forge(tmp_path, {"format": adk_artifacts._METADATA_FORMAT, "nodes": [other], "agent": "writer",
                    "origin": sink.aliases[other],
                    "digest": hashlib.sha256(adk_artifacts._record(
                        key, {"present": True, "text": body}).encode("utf-8")).hexdigest()})
            if attack == "malformed":
                _forge(tmp_path, {"format": adk_artifacts._METADATA_FORMAT, "nodes": "writer"})
            if attack == "absent":
                _forge(tmp_path, {})
            runner = _reader(tmp_path)
            sess = await runner.session_service.create_session(app_name="test", user_id="u")
            await _turn(runner, sess, "read")
        check = next(c for c in sink.checks if c[0] == "consume")
        lineage = ancestors(sink, check[2])
        assert any("unattributed external input" in e.text and body in e.text for e in lineage)
        assert not any(e.id in produced for e in lineage)
        assert not any(e.HasField("derived_from") for e in lineage)
    asyncio.run(run())


def test_binary_artifacts_stay_rejected_on_a_persistent_backend(sink, tmp_path):
    async def run():
        runner = _reader(tmp_path)
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="user:decision", artifact=types.Part(
                inline_data=types.Blob(data=b"approved", mime_type="application/octet-stream")))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError, match="plain-text"):
            await _turn(runner, sess, "read")
        assert not any("approved" in e.text for e in sink.events.values())
    asyncio.run(run())


def test_artifact_reference_part_cannot_resolve_another_file(sink):
    async def run():
        model = ScriptedModel([text("first")])
        runner = InMemoryRunner(agent=LlmAgent(name="reader", model=model,
            instruction="Value {artifact.pointer}"), app_name="test")
        sess = await runner.session_service.create_session(app_name="test", user_id="u")
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="secret", artifact=types.Part(text="hidden contents"))
        uri = f"memory://apps/test/users/u/sessions/{sess.id}/artifacts/secret/versions/0"
        await runner.artifact_service.save_artifact(app_name="test", user_id="u", session_id=sess.id,
            filename="pointer", artifact=types.Part(file_data=types.FileData(file_uri=uri, mime_type="text/plain")))
        adk.instrument()
        with session(end_on_exit=False), pytest.raises(AdkInstrumentationError):
            await _turn(runner, sess)
        assert not any("hidden contents" in e.text for e in sink.events.values())
    asyncio.run(run())
