"""Immutable snapshot requests preserve caller objects and canonical identities."""
import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from google.protobuf.json_format import ParseDict
from sasy import capture
from sasy.instrumentation.session import _entity_var, _session_id_var
from sasy.observability import EventSnapshot, api, resolve_events, resolve_events_async
from sasy.observability._snapshots import CaptureDigestCache, content_digest
from sasy.proto.observability_pb2 import Edge, Event, IDs, Tool


@pytest.fixture
def boundary(monkeypatch):
    stub = SimpleNamespace(ResolveSnapshots=Mock(return_value=IDs(ids=["canonical"])), ResolveEvents=Mock(return_value=IDs(ids=["canonical"])))
    api._snapshot_cache.clear()
    monkeypatch.setattr(api, "get_stub", lambda *_: stub)
    monkeypatch.setattr(api, "get_async_stub", lambda *_: stub)
    monkeypatch.setattr(api, "_metadata", lambda: [("authorization", "test")])
    session = _session_id_var.set("snapshot-session")
    entity = _entity_var.set("actor")
    yield stub
    api._snapshot_cache.clear()
    _entity_var.reset(entity)
    _session_id_var.reset(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_snapshot_requests_clone_sanitize_and_stamp(boundary, monkeypatch, asynchronous):
    original_capture = api.capture_events
    def sanitize(events):
        copies = original_capture(events)
        for event in copies:
            event.text = "sanitized"
        return copies
    monkeypatch.setattr(api, "capture_events", sanitize)
    snapshot = EventSnapshot(event=Event(id="origin", text="private", derived_from=Tool(name="read")),
                             base_id="previous", dependencies=[Edge(source="previous", destination="origin")])
    before = snapshot.SerializeToString()
    if asynchronous:
        boundary.ResolveSnapshots = AsyncMock(return_value=IDs(ids=["canonical"]))
        assert await resolve_events_async([snapshot]) == ["canonical"]
    else:
        assert resolve_events([snapshot]) == ["canonical"]
    assert snapshot.SerializeToString() == before
    request = boundary.ResolveSnapshots.call_args.args[0]
    assert request.session_id == "snapshot-session"
    assert request.snapshots[0].event.id == "origin"
    assert request.snapshots[0].event.text == "sanitized"
    assert request.snapshots[0].event.entity == "actor"
    assert request.snapshots[0].dependencies[0].entity == "actor"
    assert request.snapshots[0].base_id == "previous"
    assert boundary.ResolveSnapshots.call_args.kwargs["metadata"] == [("authorization", "test")]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("compact", [False, True])
async def test_reuse_full_snapshot_and_clearing_provenance(boundary, asynchronous, compact):
    method = (AsyncMock if asynchronous else Mock)(return_value=IDs(ids=["canonical"]))
    setattr(boundary, "ResolveSnapshots" if compact else "ResolveEvents", method)
    request = EventSnapshot(event=Event(id="origin", text="unchanged"), base_id="canonical", reuse_dependencies=True)
    invoke = resolve_events_async if asynchronous else resolve_events
    for _ in range(2):
        if asynchronous:
            assert await invoke([request], compact=compact) == ["canonical"]
        else:
            assert invoke([request], compact=compact) == ["canonical"]
    sent = method.call_args.args[0].snapshots[0]
    assert sent.reuse_dependencies
    assert sent.HasField("content_hash") == compact
    assert sent.event.text == ("" if compact else "unchanged")
    assert not sent.event.HasField("derived_from")
    assert not sent.dependencies


@pytest.mark.parametrize("snapshot", [EventSnapshot(), EventSnapshot(event=Event(text="missing origin")),
    EventSnapshot(event=Event(id="origin"), reuse_dependencies=True),
    EventSnapshot(event=Event(id="origin"), base_id="base", reuse_dependencies=True,
                  dependencies=[Edge(source="base", destination="origin")])])
def test_invalid_snapshot_shape_stops_before_rpc(boundary, snapshot):
    with pytest.raises(ValueError):
        resolve_events([snapshot])
    boundary.ResolveSnapshots.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ids", [[], [""], ["one", "two"]])
async def test_invalid_canonical_reply_fails_closed(boundary, asynchronous, ids):
    boundary.ResolveSnapshots = (AsyncMock if asynchronous else Mock)(return_value=IDs(ids=ids))
    snapshots = [EventSnapshot(event=Event(id="origin"))]
    with pytest.raises(RuntimeError, match="canonical IDs"):
        if asynchronous:
            await resolve_events_async(snapshots)
        else:
            resolve_events(snapshots)


@pytest.mark.asyncio
async def test_empty_batch_no_rpc_and_rpc_errors_propagate(boundary):
    assert resolve_events([]) == []
    assert await resolve_events_async([]) == []
    boundary.ResolveSnapshots.assert_not_called()
    boundary.ResolveSnapshots.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_events([EventSnapshot(event=Event(id="origin"))])
    boundary.ResolveSnapshots = AsyncMock(side_effect=RuntimeError("unavailable"))
    with pytest.raises(RuntimeError, match="unavailable"):
        await resolve_events_async([EventSnapshot(event=Event(id="origin"))])


def reference(text="ordinary text", **event_fields):
    return EventSnapshot(event=Event(id="origin", text=text, **event_fields), base_id="canonical", reuse_dependencies=True)


def test_cross_language_content_vectors():
    vectors = json.loads((Path(__file__).parent / "fixtures/message-content-hashes.json").read_text())["vectors"]
    for vector in vectors:
        event = ParseDict(vector["event"], Event())
        before = event.SerializeToString()
        canonical = copy.deepcopy(event)
        canonical.ClearField("id")
        canonical.ClearField("principal")
        canonical.DiscardUnknownFields()
        assert canonical.SerializeToString(deterministic=True).hex() == vector["canonical_proto_hex"], vector["name"]
        assert content_digest(event).hex() == vector["sha256"], vector["name"]
        assert event.SerializeToString() == before


def test_digest_discards_top_level_and_nested_unknown_fields():
    # Unknown varint field 100; both Event and nested Tool preserve it in Python
    # protobuf objects but the server's known-field decoding drops it.
    unknown = bytes.fromhex("a00601")
    original = Event(text="known", tools=[Tool(name="read", arguments="{}")])
    extended = Event.FromString(original.SerializeToString() + unknown)
    extended.tools[0].CopyFrom(Tool.FromString(original.tools[0].SerializeToString() + unknown))
    assert extended.SerializeToString() != original.SerializeToString()
    assert content_digest(extended) == content_digest(original)


def test_exact_acknowledged_snapshots_skip_repeated_capture_without_sending_raw_fingerprints(boundary, monkeypatch):
    actual_capture = api.capture_events
    spy = Mock(wraps=actual_capture)
    monkeypatch.setattr(api, "capture_events", spy)
    snapshot = reference('https://example.invalid/?api_key=synthetic-secret', tools=[Tool(name="request", arguments='{"Authorization":"Bearer secret"}')])
    before = snapshot.SerializeToString()
    assert resolve_events([snapshot]) == ["canonical"]
    assert resolve_events([snapshot]) == ["canonical"]
    assert spy.call_count == 1
    sent = boundary.ResolveSnapshots.call_args.args[0].snapshots[0]
    full = copy.deepcopy(snapshot.event)
    full.entity = "actor"
    expected = content_digest(actual_capture([full])[0])
    assert sent.content_hash == expected
    assert sent.event == Event(id="origin")
    assert b"synthetic-secret" not in sent.SerializeToString()
    assert snapshot.SerializeToString() == before
    assert len(api._snapshot_cache) == 1
    assert all(len(key) == len(value) == 32 for key, value in api._snapshot_cache._entries.items())
    assert expected not in api._snapshot_cache._entries  # Raw keys have a separate domain.
    boundary.ResolveEvents.assert_not_called()


@pytest.mark.parametrize("field", ["text", "agent", "role", "entity", "tool_name", "tool_arguments", "tool_order", "derived_from", "optional_presence", "id", "principal"])
def test_every_current_event_field_affects_capture_cache_key(boundary, monkeypatch, field):
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    snapshot = reference(tools=[Tool(name="a", arguments="{}"), Tool(name="b", arguments="[]")])
    resolve_events([snapshot])
    if field == "tool_name":
        snapshot.event.tools[0].name = "changed"
    elif field == "tool_arguments":
        snapshot.event.tools[0].arguments = '{"Authorization":"changed"}'
    elif field == "tool_order":
        snapshot.event.tools.reverse()
    elif field == "derived_from":
        snapshot.event.derived_from.CopyFrom(Tool(name="producer"))
    elif field == "optional_presence":
        snapshot.event.tools[0].ClearField("arguments")
    elif field == "role":
        snapshot.event.role = 0  # Explicit zero differs from absent.
    else:
        setattr(snapshot.event, field, "changed")
    resolve_events([snapshot])
    assert spy.call_count == 2
    assert boundary.ResolveSnapshots.call_count == 2


def test_texts_that_differ_only_after_a_credential_marker_are_different_messages(boundary, monkeypatch):
    # These two once sanitized to one string and so shared a digest. A message
    # is recorded as it is, so they are two messages with two digests, and the
    # cache re-derives rather than answering from the first.
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    snapshot = reference("Authorization: first")
    resolve_events([snapshot])
    first = boundary.ResolveSnapshots.call_args.args[0].snapshots[0].content_hash
    snapshot.event.text = "Authorization: second"
    resolve_events([snapshot])
    assert spy.call_count == 2
    assert boundary.ResolveSnapshots.call_args.args[0].snapshots[0].content_hash != first


@pytest.mark.parametrize("changed", ["session", "base", "entity"])
def test_scope_changes_cannot_use_previous_capture_cache_entry(boundary, monkeypatch, changed):
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    snapshot = reference()
    resolve_events([snapshot])
    context = _session_id_var if changed == "session" else _entity_var
    token = context.set("different") if changed != "base" else None
    try:
        if changed == "base":
            snapshot.base_id = "another-version"
            boundary.ResolveSnapshots.return_value = IDs(ids=[snapshot.base_id])
        resolve_events([snapshot])
    finally:
        if token is not None:
            context.reset(token)
    assert spy.call_count == 2


def test_full_baseline_bypasses_cache_and_compact_rpc(boundary, monkeypatch):
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    snapshot = reference()
    for _ in range(2):
        resolve_events([snapshot], compact=False)
    assert spy.call_count == 2 and len(api._snapshot_cache) == 0
    assert boundary.ResolveEvents.call_count == 2
    boundary.ResolveSnapshots.assert_not_called()
    sent = boundary.ResolveEvents.call_args.args[0].snapshots[0]
    assert sent.event.text == snapshot.event.text and not sent.HasField("content_hash")
    resolve_events([snapshot])
    assert spy.call_count == 3
    resolve_events([snapshot], compact=False)
    assert spy.call_count == 4


def test_mixed_batch_keeps_output_full_and_validates_every_reference_before_cache_commit(boundary, monkeypatch):
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    output = EventSnapshot(event=Event(id="output", text="produced"), dependencies=[Edge(source="canonical", destination="output")])
    snapshots = [reference(), output]
    boundary.ResolveSnapshots.return_value = IDs(ids=["wrong-base", "output-version"])
    with pytest.raises(RuntimeError, match="base ID"):
        resolve_events(snapshots)
    assert len(api._snapshot_cache) == 0
    boundary.ResolveEvents.assert_not_called()
    sent = boundary.ResolveSnapshots.call_args.args[0]
    assert sent.snapshots[0].event == Event(id="origin")
    assert sent.snapshots[0].HasField("content_hash")
    assert sent.snapshots[1].event.text == "produced"
    assert not sent.snapshots[1].HasField("content_hash")
    assert sent.snapshots[1].dependencies[0].source == "canonical"
    boundary.ResolveSnapshots.return_value = IDs(ids=["canonical", "output-version"])
    assert resolve_events(snapshots) == ["canonical", "output-version"]
    assert spy.call_count == 4  # Neither prior speculative capture result was cached.
    assert len(api._snapshot_cache) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reference_reply_must_equal_exact_base(boundary, compact, asynchronous):
    method = (AsyncMock if asynchronous else Mock)(return_value=IDs(ids=["wrong-version"]))
    setattr(boundary, "ResolveSnapshots" if compact else "ResolveEvents", method)
    with pytest.raises(RuntimeError, match="base ID"):
        if asynchronous:
            await resolve_events_async([reference()], compact=compact)
        else:
            resolve_events([reference()], compact=compact)
    assert len(api._snapshot_cache) == 0


@pytest.mark.parametrize("failure", ["rpc", "cardinality", "empty_id", "different_id"])
def test_failed_reference_resolution_does_not_warm_cache(boundary, monkeypatch, failure):
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    if failure == "rpc":
        boundary.ResolveSnapshots.side_effect = RuntimeError("missing reference")
    else:
        boundary.ResolveSnapshots.return_value = IDs(ids={"cardinality": [], "empty_id": [""], "different_id": ["new"]}[failure])
    with pytest.raises(RuntimeError):
        resolve_events([reference()])
    assert len(api._snapshot_cache) == 0
    boundary.ResolveSnapshots.side_effect = None
    boundary.ResolveSnapshots.return_value = IDs(ids=["canonical"])
    resolve_events([reference()])
    assert spy.call_count == 2
    boundary.ResolveEvents.assert_not_called()


@pytest.mark.parametrize("compact", [False, True])
def test_callers_cannot_supply_precomputed_reference_hashes(boundary, compact):
    snapshot = reference()
    snapshot.content_hash = b""  # Optional presence alone is enough to reject.
    with pytest.raises(ValueError, match="full Event"):
        resolve_events([snapshot], compact=compact)
    boundary.ResolveSnapshots.assert_not_called()
    boundary.ResolveEvents.assert_not_called()


def test_replacing_capture_function_invalidates_previous_digest(boundary, monkeypatch):
    # The cache key carries the identity of the function that builds the
    # recorded event, so swapping that function cannot be answered from an
    # entry derived by the previous one.
    snapshot = reference("before")
    resolve_events([snapshot])
    first_digest = boundary.ResolveSnapshots.call_args.args[0].snapshots[0].content_hash
    replacement = Mock(side_effect=lambda events: [Event(text="after", entity="actor")])
    monkeypatch.setattr(api, "capture_events", replacement)
    resolve_events([snapshot])
    assert replacement.call_count >= 1
    second_digest = boundary.ResolveSnapshots.call_args.args[0].snapshots[0].content_hash
    assert second_digest != first_digest
    assert second_digest == content_digest(Event(text="after", entity="actor"))


@pytest.mark.parametrize("field", ["text", "tool_arguments", "derived_arguments"])
def test_capture_length_limit_is_enforced_before_cache_hit(boundary, monkeypatch, field):
    snapshot = reference("ok")
    if field == "text":
        snapshot.event.text = "long"
    elif field == "tool_arguments":
        snapshot.event.tools.append(Tool(arguments="long"))
    else:
        snapshot.event.derived_from.arguments = "long"
    resolve_events([snapshot])
    monkeypatch.setattr(capture, "MAX_CAPTURE_LENGTH", 3)
    with pytest.raises(ValueError, match="character limit"):
        resolve_events([snapshot])
    assert boundary.ResolveSnapshots.call_count == 1


def test_a_cached_entry_cannot_bypass_the_capture_length_limit(boundary, monkeypatch):
    # A message's own text no longer goes through the credential scanner, so
    # there is no scanning budget to bound here. The size limit still applies,
    # and an entry the cache already holds must not answer past a tightened one.
    nested = "Authorization: synthetic"
    for _ in range(4):
        nested = json.dumps(nested)
    snapshot = reference(nested)
    resolve_events([snapshot])
    resolve_events([snapshot])  # The exact previously validated content can hit.
    assert boundary.ResolveSnapshots.call_count == 2
    monkeypatch.setattr(capture, "MAX_CAPTURE_LENGTH", 3)
    with pytest.raises(ValueError, match="character limit"):
        resolve_events([snapshot])
    assert boundary.ResolveSnapshots.call_count == 2

def test_lru_is_bounded_and_hits_refresh_recency(boundary, monkeypatch):
    monkeypatch.setattr(api, "_snapshot_cache", CaptureDigestCache(2))
    spy = Mock(wraps=api.capture_events)
    monkeypatch.setattr(api, "capture_events", spy)
    for text in ("a", "b", "a", "c", "a"):
        resolve_events([reference(text)])
    assert spy.call_count == 3 and len(api._snapshot_cache) == 2
    resolve_events([reference("b")])
    assert spy.call_count == 4 and len(api._snapshot_cache) == 2


def test_concurrent_cache_use_stays_bounded_and_stores_only_digests(boundary, monkeypatch):
    monkeypatch.setattr(api, "_snapshot_cache", CaptureDigestCache(8))
    def call(index):
        _session_id_var.set("threads")
        return resolve_events([reference(str(index))])
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(result == ["canonical"] for result in executor.map(call, range(40)))
    assert len(api._snapshot_cache) == 8
    assert all(isinstance(key, bytes) and len(key) == 32 and isinstance(value, bytes) and len(value) == 32
               for key, value in api._snapshot_cache._entries.items())


def test_clear_prevents_inflight_rpc_from_repopulating_cache(boundary):
    entered, resume = ThreadEvent(), ThreadEvent()
    def rpc(request, **kwargs):
        entered.set()
        assert resume.wait(10)
        return IDs(ids=["canonical"])
    boundary.ResolveSnapshots.side_effect = rpc
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(copy_context().run, resolve_events, [reference()])
        assert entered.wait(10)
        api._snapshot_cache.clear()
        resume.set()
        assert future.result(timeout=10) == ["canonical"]
    assert len(api._snapshot_cache) == 0


@pytest.mark.asyncio
async def test_async_call_freezes_objects_and_batch_before_await(boundary, monkeypatch):
    entered, resume = asyncio.Event(), asyncio.Event()
    requests = []
    async def rpc(request, **kwargs):
        requests.append(request)
        entered.set()
        await resume.wait()
        return IDs(ids=["canonical"])
    boundary.ResolveSnapshots = AsyncMock(side_effect=rpc)
    snapshots = [reference("checked")]
    task = asyncio.create_task(resolve_events_async(snapshots))
    await entered.wait()
    snapshots[0].event.text = "changed during await"
    snapshots[0].base_id = "changed base"
    snapshots.append(reference("added later"))
    resume.set()
    assert await task == ["canonical"]
    assert len(requests[0].snapshots) == 1
    assert requests[0].snapshots[0].base_id == "canonical"
    expected = Event(id="origin", text="checked", entity="actor")
    assert requests[0].snapshots[0].content_hash == content_digest(expected)


@pytest.mark.asyncio
async def test_async_cancellation_does_not_publish_pending_cache(boundary):
    entered = asyncio.Event()
    async def rpc(request, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    boundary.ResolveSnapshots = AsyncMock(side_effect=rpc)
    task = asyncio.create_task(resolve_events_async([reference()]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(api._snapshot_cache) == 0
