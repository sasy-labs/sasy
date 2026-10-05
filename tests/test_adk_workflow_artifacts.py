"""Concurrent native Workflow artifact publications and exact read ancestry."""
import asyncio
import json
import threading
from contextvars import copy_context

import pytest
from google.adk.artifacts import file_artifact_service as file_backend
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import JoinNode, Workflow
from test_adk_instrumentation import (
    AdkInstrumentationError,
    LlmAgent,
    ScriptedModel,
    adk,
    calls,
    session,
    text,
    types,
)
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


def _agent(name, tool):
    return LlmAgent(name=name, include_contents="none", tools=[tool],
        model=ScriptedModel([calls((tool.__name__, {}, name)), text(name + " complete")]))


async def _run(root, branches, final=None, *, service=None):
    edges = ("START", tuple(branches), JoinNode(name="join"))
    if final is not None:
        edges += (final,)
    runner = Runner(node=Workflow(name="artifacts", edges=[edges]), app_name="test",
        session_service=InMemorySessionService(),
        artifact_service=service if service is not None else FileArtifactService(root))
    adk.instrument()
    native = await runner.session_service.create_session(app_name="test", user_id="u")
    with session(end_on_exit=False):
        async with asyncio.timeout(10):
            async for _ in runner.run_async(user_id="u", session_id=native.id,
                    new_message=types.Content(role="user", parts=[types.Part(text="review")])):
                pass
    return runner, native


def _artifact_events(sink, ids):
    result = []
    for event in ancestors(sink, ids):
        try:
            value = json.loads(event.text)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get("adk_resource", [None])[0] == "artifact":
            result.append((event, value))
    return result


@pytest.mark.parametrize("same_name", [False, True])
def test_overlapping_writes_use_native_versions_and_keep_both_producers(sink, tmp_path, monkeypatch, same_name):
    """Hold the first payload before publication while the second is published."""
    original = file_backend._write_metadata
    first_waiting = threading.Event()
    release_first = threading.Event()
    order = []
    def metadata(path, **kwargs):
        if threading.current_thread().name and not first_waiting.is_set():
            first_waiting.set()
            if not release_first.wait(5):
                raise RuntimeError("second writer did not publish")
        return original(path, **kwargs)
    monkeypatch.setattr(file_backend, "_write_metadata", metadata)
    async def run():
        async def first(tool_context):
            """Publish the first review."""
            result = await tool_context.save_artifact("report", types.Part(text="first evidence"))
            order.append(("first", result))
            return result
        async def second(tool_context):
            """Publish another review while the first remains staged."""
            await asyncio.to_thread(first_waiting.wait, 5)
            try:
                result = await tool_context.save_artifact("report" if same_name else "other",
                    types.Part(text="second evidence"))
                order.append(("second", result))
                return result
            finally:
                release_first.set()
        async def read(tool_context):
            """Read the independently published versions."""
            for filename, version, expected in (("report", 0, "first evidence"),
                    ("report" if same_name else "other", 1 if same_name else 0, "second evidence")):
                part = await tool_context.load_artifact(filename, version=version)
                assert part.text == expected
            return "read both"
        await _run(tmp_path, [_agent("first", first), _agent("second", second)], _agent("reader", read))
    try:
        asyncio.run(run())
    finally:
        release_first.set()
    assert order == [("second", 1 if same_name else 0), ("first", 0)]
    result = next(event for event in sink.events.values()
        if event.HasField("derived_from") and event.derived_from.name == "read")
    artifacts = _artifact_events(sink, [result.id])
    assert any(event.agent == "first" and value["value"]["text"] == "first evidence" for event, value in artifacts)
    assert any(event.agent == "second" and value["value"]["text"] == "second evidence" for event, value in artifacts)


def test_read_of_published_artifact_precedes_successful_producer_result(sink, tmp_path):
    async def run():
        published, consumed = asyncio.Event(), asyncio.Event()
        read_ids = []
        async def write(tool_context):
            """Publish evidence, then wait before completing the tool."""
            await tool_context.save_artifact("report", types.Part(text="sensitive evidence"))
            published.set()
            await consumed.wait()
            return "successful writer"
        async def read(tool_context):
            """Use evidence before the writer tool returns."""
            await published.wait()
            part = await tool_context.load_artifact("report")
            assert part.text == "sensitive evidence"
            read_ids.extend(adk._current_input_ids.get())
            lineage = ancestors(sink, read_ids)
            assert any(e.agent == "writer" and "observed production" in e.text for e in lineage)
            assert not any(e.HasField("derived_from") and e.derived_from.name == "write" for e in lineage)
            consumed.set()
            return "read evidence"
        await _run(tmp_path, [_agent("writer", write), _agent("reader", read)])
        # Completion must not retroactively add success to an earlier read.
        assert not any(e.HasField("derived_from") and e.derived_from.name == "write"
            for e in ancestors(sink, read_ids))
        assert any(e.HasField("derived_from") and e.derived_from.name == "write" for e in sink.events.values())
    asyncio.run(run())


def test_pending_first_version_is_absent_to_parallel_reader(sink, tmp_path, monkeypatch):
    original = file_backend._write_metadata
    pending, release = threading.Event(), threading.Event()
    def metadata(path, **kwargs):
        pending.set()
        if not release.wait(5):
            raise RuntimeError("reader did not inspect pending version")
        return original(path, **kwargs)
    monkeypatch.setattr(file_backend, "_write_metadata", metadata)
    async def run():
        async def write(tool_context):
            """Stage a not-yet-published review."""
            return await tool_context.save_artifact("report", types.Part(text="not visible"))
        async def read(tool_context):
            """Read while only the pending directory exists."""
            await asyncio.to_thread(pending.wait, 5)
            try:
                assert await tool_context.load_artifact("report") is None
                lineage = ancestors(sink, adk._current_input_ids.get())
                assert not any("not visible" in event.text for event in lineage)
            finally:
                release.set()
            return "absent"
        await _run(tmp_path, [_agent("writer", write), _agent("reader", read)])
    try:
        asyncio.run(run())
    finally:
        release.set()


@pytest.mark.parametrize("operation", ["state_read", "state_write", "list"])
def test_parallel_artifact_permission_does_not_enable_shared_state_or_listing(sink, tmp_path, operation):
    effects = []
    async def forbidden(tool_context):
        """Try an unqualified shared resource operation."""
        if operation == "state_read":
            tool_context.state.get("decision")
        elif operation == "state_write":
            tool_context.state["decision"] = "approved"
        else:
            await tool_context.list_artifacts()
        effects.append("dispatched")
        return "done"
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises(AdkInstrumentationError):
        asyncio.run(_run(tmp_path, [_agent("reader", forbidden), peer]))
    assert not effects


@pytest.mark.parametrize("filename", ["./report", "nested/../report", "report/", "user:./report"])
def test_noncanonical_artifact_names_are_rejected_in_parallel_branches(sink, tmp_path, filename):
    async def write(tool_context):
        """Attempt to publish under an alias."""
        return await tool_context.save_artifact(filename, types.Part(text="aliased evidence"))
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises((AdkInstrumentationError, InputValidationError)):
        asyncio.run(_run(tmp_path, [_agent("writer", write), peer]))
    assert not list(tmp_path.rglob("metadata.json"))


def test_unqualified_inmemory_service_remains_rejected_in_parallel(sink, tmp_path):
    async def write(tool_context):
        """Attempt an unqualified backend write."""
        return await tool_context.save_artifact("report", types.Part(text="evidence"))
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises(AdkInstrumentationError):
        asyncio.run(_run(tmp_path, [_agent("writer", write), peer], service=InMemoryArtifactService()))


def test_post_join_listing_retains_artifact_name_producers(sink, tmp_path):
    def writer(name):
        async def write(tool_context):
            """Publish a named report."""
            return await tool_context.save_artifact(name, types.Part(text=name + " evidence"))
        return _agent(name, write)
    async def inventory(tool_context):
        """List files after both writers finish."""
        assert sorted(await tool_context.list_artifacts()) == ["left", "right"]
        lineage = ancestors(sink, adk._current_input_ids.get())
        assert any(event.agent == "left" and "observed production" in event.text for event in lineage)
        assert any(event.agent == "right" and "observed production" in event.text for event in lineage)
        return "listed"
    asyncio.run(_run(tmp_path, [writer("left"), writer("right")], _agent("inventory", inventory)))


def test_latest_read_keeps_exact_selected_version_when_new_version_publishes(sink, tmp_path, monkeypatch):
    selected, newer = threading.Event(), threading.Event()
    original = file_backend._read_text_if_present
    def read_payload(path):
        if path.name == "report" and path.parent.name == "0":
            selected.set()
            if not newer.wait(5):
                raise RuntimeError("new version did not publish")
        return original(path)
    monkeypatch.setattr(file_backend, "_read_text_if_present", read_payload)
    async def run():
        published = asyncio.Event()
        captured = []
        async def write(tool_context):
            """Publish two versions around a concurrent reader."""
            assert await tool_context.save_artifact("report", types.Part(text="old evidence")) == 0
            published.set()
            await asyncio.to_thread(selected.wait, 5)
            try:
                assert await tool_context.save_artifact("report", types.Part(text="new evidence")) == 1
            finally:
                newer.set()
            return "written"
        async def read(tool_context):
            """Read latest while another version is being published."""
            await published.wait()
            part = await tool_context.load_artifact("report")
            assert part.text == "old evidence"
            captured.extend(adk._current_input_ids.get())
            return "read old"
        await _run(tmp_path, [_agent("writer", write), _agent("reader", read)])
        artifacts = _artifact_events(sink, captured)
        assert any(value["adk_resource"][-1] == 0 and value["value"]["text"] == "old evidence"
            for _, value in artifacts)
        assert not any(value["value"].get("text") == "new evidence" for _, value in artifacts)
    try:
        asyncio.run(run())
    finally:
        newer.set()


def test_failed_publication_never_exposes_payload_or_success(sink, tmp_path, monkeypatch):
    def fail_metadata(*args, **kwargs):
        raise OSError("metadata persistence failed")
    monkeypatch.setattr(file_backend, "_write_metadata", fail_metadata)
    async def write(tool_context):
        """Publish evidence on a failing disk."""
        return await tool_context.save_artifact("report", types.Part(text="failed evidence"))
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises(OSError, match="metadata persistence failed"):
        asyncio.run(_run(tmp_path, [_agent("writer", write), peer]))
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.rglob("*.pending"))
    assert not any(e.HasField("derived_from") and e.derived_from.name == "write" for e in sink.events.values())


def test_cancellation_after_publication_does_not_invent_tool_success(sink, tmp_path):
    async def run():
        published, read_done = asyncio.Event(), asyncio.Event()
        captured = []
        async def write(tool_context):
            """Publish then cancel before successful tool completion."""
            await tool_context.save_artifact("report", types.Part(text="published evidence"))
            published.set()
            await read_done.wait()
            raise asyncio.CancelledError()
        async def read(tool_context):
            """Consume the published write before cancellation."""
            await published.wait()
            assert (await tool_context.load_artifact("report")).text == "published evidence"
            captured.extend(adk._current_input_ids.get())
            read_done.set()
            return "read"
        try:
            await _run(tmp_path, [_agent("writer", write), _agent("reader", read)])
        except asyncio.CancelledError:
            pass
        assert captured
        assert any(event.agent == "writer" and "observed production" in event.text
            for event in ancestors(sink, captured))
        assert not any(e.HasField("derived_from") and e.derived_from.name == "write" for e in sink.events.values())
    asyncio.run(run())


def test_failed_write_observation_cleans_reservation_without_publishing(sink, tmp_path, monkeypatch):
    original = adk.observation.resolve_events_async
    async def reject_write(snapshots):
        if any('"artifact"' in item.event.text and '"observed production"' in item.event.text
                for item in snapshots):
            raise RuntimeError("write observation failed")
        return await original(snapshots)
    monkeypatch.setattr(adk.observation, "resolve_events_async", reject_write)
    async def write(tool_context):
        """Attempt an unrecordable publication."""
        return await tool_context.save_artifact("report", types.Part(text="unobserved evidence"))
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises(RuntimeError, match="write observation failed"):
        asyncio.run(_run(tmp_path, [_agent("writer", write), peer]))
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.rglob("*.pending"))


def test_cancelled_queued_worker_cleans_unconsumed_reservation(sink, tmp_path, monkeypatch):
    async def run():
        queued = asyncio.Event()
        original = asyncio.to_thread
        deferred = []
        async def hold_save(function, *args, **kwargs):
            if getattr(function, "__name__", "") == "_save_artifact_sync":
                deferred.append((copy_context(), function, args, kwargs))
                queued.set()
                await asyncio.Event().wait()
            return await original(function, *args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", hold_save)
        async def write(tool_context):
            """Queue a write whose native worker has not started."""
            return await tool_context.save_artifact("report", types.Part(text="queued evidence"))
        peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
        running = asyncio.create_task(_run(tmp_path, [_agent("writer", write), peer]))
        try:
            await asyncio.wait_for(queued.wait(), 5)
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        # A worker already submitted to an executor may start after cancellation.
        context, function, args, kwargs = deferred[0]
        with pytest.raises(AdkInstrumentationError, match="reserved publication"):
            await original(context.run, function, *args, **kwargs)
    asyncio.run(run())
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.rglob("*.pending"))
    assert not any(e.HasField("derived_from") and e.derived_from.name == "write" for e in sink.events.values())


def test_case_alias_cannot_downgrade_published_provenance(sink, tmp_path):
    probe = tmp_path / "CaseSensitivityProbe"
    probe.write_text("probe")
    insensitive = (tmp_path / "casesensitivityprobe").exists()
    probe.unlink()
    if not insensitive:
        pytest.skip("case-insensitive filesystem required")
    async def run():
        published = asyncio.Event()
        async def write(tool_context):
            """Publish sensitive evidence under its canonical spelling."""
            await tool_context.save_artifact("Report", types.Part(text="sensitive evidence"))
            published.set()
            return "published"
        async def read(tool_context):
            """Attempt to read through a filesystem case alias."""
            await published.wait()
            await tool_context.load_artifact("report")
            pytest.fail("case alias was accepted")
        with pytest.raises(AdkInstrumentationError):
            await _run(tmp_path, [_agent("writer", write), _agent("reader", read)])
    asyncio.run(run())
    assert not any("unattributed external input" in event.text and "sensitive evidence" in event.text
        for event in sink.events.values())


def test_parallel_artifact_instruction_template_observes_exact_file(sink, tmp_path):
    async def run():
        service = FileArtifactService(tmp_path)
        await service.save_artifact(app_name="test", user_id="u", filename="user:report",
            artifact=types.Part(text="external report evidence"))
        reader = LlmAgent(name="reader", instruction="Review {artifact.user:report}",
            include_contents="none", model=ScriptedModel([text("reviewed")]))
        peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
        await _run(tmp_path, [reader, peer], service=service)
        assert "external report evidence" in str(reader.model._requests[0].config.system_instruction)
        result = next(event for event in sink.events.values() if event.text == "reviewed")
        assert any("external report evidence" in event.text and "unattributed external input" in event.text
            for event in ancestors(sink, [result.id]))
    asyncio.run(run())


async def _turn(runner, native):
    async for _ in runner.run_async(user_id="u", session_id=native.id,
            new_message=types.Content(role="user", parts=[types.Part(text="review")])):
        pass


def test_started_worker_can_publish_after_cancellation_with_only_write_ancestry(sink, tmp_path, monkeypatch):
    from sasy.instrumentation import adk_artifacts
    blocked, release, published = threading.Event(), threading.Event(), threading.Event()
    original_metadata = file_backend._write_metadata
    original_replace = file_backend.os.replace
    def metadata(path, **kwargs):
        blocked.set()
        if not release.wait(5):
            raise RuntimeError("cancelled worker was not released")
        return original_metadata(path, **kwargs)
    def replace(source, destination):
        result = original_replace(source, destination)
        if str(source).endswith(".0.pending") and str(destination).endswith("/0"):
            published.set()
        return result
    monkeypatch.setattr(file_backend, "_write_metadata", metadata)
    monkeypatch.setattr(file_backend.os, "replace", replace)
    async def run():
        service = FileArtifactService(tmp_path)
        sessions = InMemorySessionService()
        async def write(tool_context):
            """Start an atomic publication that outlives cancellation."""
            return await tool_context.save_artifact("user:report", types.Part(text="late evidence"))
        peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
        workflow = Workflow(name="artifacts", edges=[("START", (_agent("writer", write), peer), JoinNode(name="join"))])
        runner = Runner(node=workflow, app_name="test", session_service=sessions, artifact_service=service)
        adk.instrument()
        native = await sessions.create_session(app_name="test", user_id="u")
        async def read(tool_context):
            """Read the write that actually published after cancellation."""
            assert (await tool_context.load_artifact("user:report")).text == "late evidence"
            return "read late publication"
        with session(end_on_exit=False):
            running = asyncio.create_task(_turn(runner, native))
            try:
                assert await asyncio.to_thread(blocked.wait, 5)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            finally:
                release.set()
            assert await asyncio.to_thread(published.wait, 5)
            stored = adk_artifacts._instrumented[FileArtifactService]
            assert await stored["list_versions"](service, app_name="test", user_id="u",
                session_id=native.id, filename="user:report") == [0]
            reader = Runner(agent=_agent("reader", read), app_name="test",
                session_service=sessions, artifact_service=service)
            fresh = await sessions.create_session(app_name="test", user_id="u")
            await _turn(reader, fresh)
        result = next(event for event in sink.events.values()
            if event.HasField("derived_from") and event.derived_from.name == "read")
        lineage = ancestors(sink, [result.id])
        assert any(event.agent == "writer" and "observed production" in event.text
            and "late evidence" in event.text for event in lineage)
        assert not any(event.HasField("derived_from") and event.derived_from.name == "write" for event in lineage)
        assert not any("unattributed external input" in event.text and "late evidence" in event.text for event in lineage)
    try:
        asyncio.run(run())
    finally:
        release.set()


@pytest.mark.parametrize("operation", ["load", "list"])
def test_reused_failed_version_cannot_inherit_unpublished_approval(sink, tmp_path, monkeypatch, operation):
    from sasy.instrumentation import adk_artifacts
    original = file_backend._write_metadata
    failing = True
    def metadata(path, **kwargs):
        if failing:
            raise OSError("publication failed")
        return original(path, **kwargs)
    monkeypatch.setattr(file_backend, "_write_metadata", metadata)
    async def run():
        nonlocal failing
        service = FileArtifactService(tmp_path)
        sessions = InMemorySessionService()
        def approve():
            """Approve the observed producer only."""
            return "approved"
        async def write(tool_context):
            """Publish an approved write which fails before visibility."""
            return await tool_context.save_artifact("user:report", types.Part(text="same evidence"))
        writer = LlmAgent(name="writer", tools=[approve, write], model=ScriptedModel([
            calls(("approve", {}, "approve")), calls(("write", {}, "write")), text("done")]))
        peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
        runner = Runner(node=Workflow(name="artifacts", edges=[("START", (writer, peer), JoinNode(name="join"))]),
            app_name="test", session_service=sessions, artifact_service=service)
        adk.instrument()
        native = await sessions.create_session(app_name="test", user_id="u")
        async def read(tool_context):
            """Read a separate unobserved production at the reused index."""
            if operation == "load":
                await tool_context.load_artifact("user:report")
            else:
                await tool_context.list_artifacts()
            pytest.fail("unpublished approval was trusted")
        with session(end_on_exit=False):
            with pytest.raises(OSError, match="publication failed"):
                await _turn(runner, native)
            failing = False
            stored = adk_artifacts._instrumented[FileArtifactService]
            version = await stored["save_artifact"](service, app_name="test", user_id="u",
                filename="user:report", artifact=types.Part(text="same evidence"))
            assert version == 0
            reader = Runner(agent=_agent("reader", read), app_name="test",
                session_service=sessions, artifact_service=service)
            fresh = await sessions.create_session(app_name="test", user_id="u")
            with pytest.raises(AdkInstrumentationError, match="publication no longer matches"):
                await _turn(reader, fresh)
        assert not any(event.HasField("derived_from") and event.derived_from.name == "read"
            for event in sink.events.values())
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["observation", "publication"])
def test_failed_reservation_name_cannot_be_listed_as_external(sink, tmp_path, monkeypatch, failure):
    original = adk.observation.resolve_events_async
    async def reject_write(snapshots):
        if any('"artifact"' in item.event.text and '"observed production"' in item.event.text
                for item in snapshots):
            raise OSError("write observation failed")
        return await original(snapshots)
    def reject_publication(*args, **kwargs):
        raise OSError("write publication failed")
    if failure == "observation":
        monkeypatch.setattr(adk.observation, "resolve_events_async", reject_write)
    else:
        monkeypatch.setattr(file_backend, "_write_metadata", reject_publication)
    async def write(tool_context):
        """Attempt to publish a name derived from the writer's input."""
        try:
            await tool_context.save_artifact("confidential-report", types.Part(text="private evidence"))
        except OSError:
            return {"error": "publication failed"}
        pytest.fail("fault injection did not run")
    async def inventory(tool_context):
        """List the filenames left by the completed branches."""
        await tool_context.list_artifacts()
        pytest.fail("unpublished filename was exposed as an external input")
    peer = LlmAgent(name="peer", model=ScriptedModel([text("peer")]))
    with pytest.raises(AdkInstrumentationError, match="listing contains unpublished reservations"):
        asyncio.run(_run(tmp_path, [_agent("writer", write), peer], _agent("reader", inventory)))
    assert not list(tmp_path.rglob("metadata.json"))
    assert not any(event.HasField("derived_from") and event.derived_from.name == "inventory"
        for event in sink.events.values())
