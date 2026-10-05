"""Immutable message snapshots against an owned engine and real policy evaluator."""
import os
import uuid

import grpc
import pytest
from sasy.proto import observability_pb2 as obs
from sasy.proto import observability_pb2_grpc as obs_grpc

pytestmark = pytest.mark.integration


def _sid():
    return uuid.uuid4().hex


def _snapshot(alias, text="message", *, base=None, sources=(), reuse=False, **fields):
    values = dict(event=obs.Event(id=alias, text=text, **fields), reuse_dependencies=reuse,
                  dependencies=[obs.Edge(source=source, destination=alias, message_index=index,
                                         proximal=index == len(sources) - 1)
                                for index, source in enumerate(sources)])
    if base is not None:
        values["base_id"] = base
    return obs.EventSnapshot(**values)


def _resolve(client, session, *snapshots):
    return list(client.observability.ResolveEvents(obs.EventSnapshots(session_id=session, snapshots=snapshots),
                                                   metadata=client.metadata, timeout=30).ids)


def _slice(client, session, root):
    return client.observability.BackwardSlice(obs.SliceRequest(session_id=session, event_id=root),
                                               metadata=client.metadata, timeout=30)


def _state(observer):
    state = observer.updates.GetState(obs.StateRequest(), metadata=observer.metadata, timeout=30)
    # GetState combines shards without a collection-order guarantee. Preserve
    # all fields and multiplicities while comparing the actual graph contents.
    state.events.sort(key=lambda event: event.SerializeToString(deterministic=True))
    state.edges.sort(key=lambda edge: edge.SerializeToString(deterministic=True))
    return state


def _reject(call, code=grpc.StatusCode.INVALID_ARGUMENT):
    with pytest.raises(grpc.RpcError) as failure:
        call()
    assert failure.value.code() == code, failure.value.details()


def test_retries_and_unchanged_observations_are_quiet(engine):
    client, observer, session = engine.client(), engine.client(), _sid()
    snapshots = [_snapshot("input", role=obs.USER), _snapshot("output", sources=["input"], role=obs.LLM)]
    first = _resolve(client, session, *snapshots)
    assert len(set(first)) == 2 and not set(first) & {"input", "output"}
    before = _state(observer)
    # A request replay also models losing the response after the server commits.
    assert _resolve(client, session, *snapshots) == first
    assert _resolve(client, session,
                    _snapshot("input", base=first[0], reuse=True, role=obs.USER),
                    _snapshot("output", base=first[1], reuse=True, role=obs.LLM)) == first
    after = _state(observer)
    assert after.sequence == before.sequence
    assert after == before
    graph = _slice(client, session, first[1])
    assert {(edge.source, edge.destination) for edge in graph.edges} == {(first[0], first[1])}
    assert {event.principal for event in graph.nodes} == {"tenant-a-client"}


def test_dependency_only_version_preserves_old_graph_without_inheriting_approval(engine):
    client, session = engine.client(), _sid()
    client.set_policy('''
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send"),
    CurrentDepends(id), ToolResult(id, "approve", _).
''', session)
    approval, old = _resolve(client, session,
        _snapshot("approval", "approved", derived_from=obs.Tool(name="approve", arguments="{}")),
        _snapshot("output", "send", sources=["approval"]))
    assert client.check(session, "send", nodes=[old]).results[0].authorized
    new, = _resolve(client, session, _snapshot("output", "send", base=old, sources=[]))
    assert new != old
    assert {event.id for event in _slice(client, session, old).nodes} == {approval, old}
    assert {event.id for event in _slice(client, session, new).nodes} == {new}
    assert client.check(session, "send", nodes=[old]).results[0].authorized
    assert not client.check(session, "send", nodes=[new]).results[0].authorized


def test_explicit_mutation_removes_fields_without_merging_or_parenting_old_version(engine):
    client, session = engine.client(), _sid()
    old, = _resolve(client, session, _snapshot("message", "approved", agent="reviewer", entity="actor",
        role=obs.AGENT, tools=[obs.Tool(name="next", arguments="{}")],
        derived_from=obs.Tool(name="approve", arguments='{"amount":1}')))
    new, = _resolve(client, session, _snapshot("message", "revised", base=old))
    assert new != old
    graph = _slice(client, session, new)
    assert len(graph.nodes) == 1 and not graph.edges
    event = graph.nodes[0]
    assert event.text == "revised" and not event.tools
    for field in ("agent", "entity", "role", "derived_from"):
        assert not event.HasField(field), field
    original = _slice(client, session, old).nodes[0]
    assert original.text == "approved" and original.derived_from.name == "approve"
    assert original.tools[0].name == "next" and original.entity == "actor"


def test_changed_input_cannot_reuse_its_old_dependencies(engine):
    client, observer, session = engine.client(), engine.client(), _sid()
    old, = _resolve(client, session, _snapshot("message", "original"))
    before = _state(observer)
    _reject(lambda: _resolve(client, session, _snapshot("message", "changed", base=old, reuse=True)))
    assert _state(observer) == before
    assert _slice(client, session, old).nodes[0].text == "original"


@pytest.mark.parametrize("invalid", ["unknown-source", "duplicate-alias", "cycle", "missing-base", "invalid-text"])
def test_invalid_batches_commit_nothing(engine, invalid):
    client, observer, session = engine.client(), engine.client(), _sid()
    items = [_snapshot("valid-first")]
    if invalid == "unknown-source":
        items.append(_snapshot("invalid", sources=["nonexistent"]))
    elif invalid == "duplicate-alias":
        items.append(_snapshot("valid-first", "different"))
    elif invalid == "cycle":
        items += [_snapshot("a", sources=["b"]), _snapshot("b", sources=["a"])]
    elif invalid == "invalid-text":
        items.append(_snapshot("invalid", "bad\x00text"))
    else:
        items.append(_snapshot("invalid", base="nonexistent", reuse=True))
    before = _state(observer)
    _reject(lambda: _resolve(client, session, *items))
    assert _state(observer) == before


def test_batch_aliases_and_canonical_dependencies_resolve_identically(engine):
    client, session = engine.client(), _sid()
    source, = _resolve(client, session, _snapshot("source", "input"))
    out, = _resolve(client, session, _snapshot("out", "output", sources=[source]))
    # Dependency resolution may use an alias earlier or later in the same batch.
    outputs = _resolve(client, session, _snapshot("out", "output", sources=["source"]),
                       _snapshot("source", "input"))
    assert outputs == [out, source]
    distinct, = _resolve(client, session, _snapshot("another-origin", "input"))
    assert distinct != source


def test_versions_and_bases_are_scoped_to_tenant_and_session(engine):
    a, b, session, other_session = engine.client(), engine.client(engine.tenant_b_key), _sid(), _sid()
    first, = _resolve(a, session, _snapshot("same-origin", "private"))
    other_tenant, = _resolve(b, session, _snapshot("same-origin", "private"))
    other_conversation, = _resolve(a, other_session, _snapshot("same-origin", "private"))
    assert len({first, other_tenant, other_conversation}) == 3
    for client, sid in [(b, session), (a, other_session)]:
        _reject(lambda: _resolve(client, sid, _snapshot("same-origin", "private", base=first, reuse=True)))
        _reject(lambda: _resolve(client, sid, _snapshot("foreign-child", sources=[first])))
    _reject(lambda: _resolve(a, session, _snapshot("different-origin", "private", base=first, reuse=True)))


@pytest.mark.parametrize("method", ["event", "edge", "atomic"])
def test_legacy_writes_cannot_mutate_versioned_messages(engine, method):
    client, observer, session = engine.client(), engine.client(), _sid()
    source, target = _resolve(client, session, _snapshot("source"), _snapshot("target"))
    forged = obs.Event(id=target, text="legacy replacement")
    edge = obs.Edge(source=source, destination=target)
    before = _state(observer)
    if method == "event":
        def call():
            return client.observability.RegisterEvents(obs.Events(session_id=session, events=[forged]),
                                                                   metadata=client.metadata, timeout=10)
    elif method == "edge":
        def call():
            return client.observability.RegisterDependencies(obs.Dependencies(session_id=session, edges=[edge]),
                                                                         metadata=client.metadata, timeout=10)
    else:
        def call():
            return client.observability.RegisterEventsWithDependencies(obs.EventsWithDependencies(
                    session_id=session, events=[obs.Event(id="fresh-legacy", text="must not commit"), forged], edges=[edge]),
                    metadata=client.metadata, timeout=10)
    _reject(call, grpc.StatusCode.FAILED_PRECONDITION)
    assert _state(observer) == before


def test_retries_remain_stable_after_process_restart(engine_factory):
    owned = engine_factory()
    try:
        session, client = _sid(), owned.client()
        request = [_snapshot("input"), _snapshot("output", sources=["input"])]
        original = _resolve(client, session, *request)
        revised, = _resolve(client, session, _snapshot("output", "revised", base=original[1]))
        owned.restart()
        client, observer = owned.client(), owned.client()
        before = _state(observer)
        assert _resolve(client, session, *request) == original
        assert _resolve(client, session, _snapshot("output", "revised", base=original[1])) == [revised]
        assert _resolve(client, session, _snapshot("output", "revised", base=revised, reuse=True)) == [revised]
        assert _state(observer) == before
        assert {event.id for event in _slice(client, session, original[1]).nodes} == set(original)
        assert [event.id for event in _slice(client, session, revised).nodes] == [revised]
    finally:
        owned.stop()


@pytest.mark.skipif(os.environ.get("SASY_TEST_LARGE_MESSAGE_VERSIONS") != "1",
                    reason="set SASY_TEST_LARGE_MESSAGE_VERSIONS=1 for the >32K history lane")
def test_large_session_replay_does_not_generate_updates(engine_factory):
    owned = engine_factory()
    try:
        session, client = _sid(), owned.client()
        batches = [[_snapshot(f"message-{i}", "short synthetic message")
                    for i in range(start, min(start + 128, 32769))] for start in range(0, 32769, 128)]
        expected = [_resolve(client, session, *batch) for batch in batches]
        # Read the whole state only in this opt-in lane; its graph exceeds the
        # ordinary gRPC receive limit even with short synthetic message content.
        channel = grpc.secure_channel(owned.address, grpc.ssl_channel_credentials(owned.root_certificates),
                                      options=[("grpc.max_receive_message_length", 128 * 1024 * 1024)])
        try:
            updates = obs_grpc.ObservabilityUpdatesStub(channel)
            metadata = [("x-api-key", owned.tenant_a_key)]
            before = updates.GetState(obs.StateRequest(), metadata=metadata, timeout=60)
            for batch, ids in zip(batches, expected):
                assert _resolve(client, session, *batch) == ids
            after = updates.GetState(obs.StateRequest(), metadata=metadata, timeout=60)
            assert after.sequence == before.sequence
            assert len(after.events) == len(before.events) == 32769
        finally:
            channel.close()
    finally:
        owned.stop()


def test_mutable_legacy_parents_cannot_enter_immutable_history(engine):
    client, observer, session = engine.client(), engine.client(), _sid()
    client.observability.RegisterEvents(obs.Events(session_id=session,
        events=[obs.Event(id="mutable-parent", text="legacy content")]), metadata=client.metadata, timeout=10)
    before = _state(observer)
    _reject(lambda: _resolve(client, session, _snapshot("child", sources=["mutable-parent"])))
    assert _state(observer) == before


def test_dependency_metadata_is_versioned(engine):
    client, session = engine.client(), _sid()
    parent, first = _resolve(client, session, _snapshot("parent"), _snapshot("output", sources=["parent"]))
    replacement = _snapshot("output", base=first, sources=[parent])
    replacement.dependencies[0].message_index = 7
    replacement.dependencies[0].proximal = False
    second, = _resolve(client, session, replacement)
    assert first != second
    old_edge, = _slice(client, session, first).edges
    new_edge, = _slice(client, session, second).edges
    assert old_edge.message_index == 0 and old_edge.proximal
    assert new_edge.message_index == 7 and not new_edge.proximal
    assert new_edge.source == old_edge.source == parent


def test_concurrent_identical_requests_commit_one_version(engine):
    from concurrent.futures import ThreadPoolExecutor

    client, observer, session = engine.client(), engine.client(), _sid()
    items = [_snapshot("source"), _snapshot("destination", sources=["source"])]
    before = _state(observer)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: _resolve(client, session, *items), range(8)))
    assert all(ids == results[0] for ids in results)
    after = _state(observer)
    assert len(after.events) - len(before.events) == 2
    assert len(after.edges) - len(before.edges) == 1
    assert _resolve(client, session, *items) == results[0]
    assert _state(observer) == after


def test_authentication_stamps_before_snapshot_identity(engine):
    client, session = engine.client(), _sid()
    snapshots = [_snapshot("source", principal="forged-writer"),
                 _snapshot("output", sources=["source"], principal="forged-writer")]
    snapshots[1].dependencies[0].principal = "forged-edge-writer"
    ids = _resolve(client, session, *snapshots)
    assert _resolve(client, session, _snapshot("source"), _snapshot("output", sources=["source"])) == ids
    graph = _slice(client, session, ids[1])
    assert {event.principal for event in graph.nodes} == {"tenant-a-client"}
    assert {edge.principal for edge in graph.edges} == {"tenant-a-client"}


def _content_hash(event):
    import hashlib
    projection = obs.Event.FromString(event.SerializeToString())
    projection.ClearField("id")
    projection.ClearField("principal")
    projection.DiscardUnknownFields()
    return hashlib.sha256(b"sasy:event-content:v1\0" + projection.SerializeToString(deterministic=True)).digest()


def _reference(event, version):
    return obs.EventSnapshot(event=obs.Event(id=event.id), base_id=version,
                             reuse_dependencies=True, content_hash=_content_hash(event))


def _resolve_compact(client, session, *snapshots):
    return list(client.observability.ResolveSnapshots(obs.EventSnapshots(session_id=session, snapshots=snapshots),
                                                      metadata=client.metadata, timeout=30).ids)


def test_compact_references_and_new_outputs_share_atomic_aliases(engine):
    client, session = engine.client(), _sid()
    root = _snapshot("input", "actual input", entity="actor", role=obs.USER)
    first, = _resolve(client, session, root)
    reference = _reference(root.event, first)
    output = _snapshot("output", "result", sources=["input"])
    before = _state(client)
    returned = _resolve_compact(client, session, reference, output)
    assert returned[0] == first and returned[1] != first
    graph = _slice(client, session, returned[1])
    assert {(edge.source, edge.destination) for edge in graph.edges} == {(first, returned[1])}
    after = _state(client)
    assert after.sequence > before.sequence
    assert _resolve_compact(client, session, reference, output) == returned
    assert _state(client) == after


@pytest.mark.parametrize("invalid", ["hash", "hash-empty", "hash-short", "origin", "base", "missing-base",
                                     "body", "entity", "principal", "reuse", "dependencies", "legacy-rpc"])
def test_invalid_compact_reference_rejects_entire_batch(engine, invalid):
    client, session = engine.client(), _sid()
    root = _snapshot("input", "original")
    version, = _resolve(client, session, root)
    reference = _reference(root.event, version)
    if invalid == "hash":
        reference.content_hash = bytes(32)
    elif invalid == "hash-empty":
        reference.content_hash = b""
    elif invalid == "hash-short":
        reference.content_hash = bytes(31)
    elif invalid == "origin":
        reference.event.id = "different-origin"
    elif invalid == "base":
        reference.base_id = version[:-1] + ("a" if version[-1] != "a" else "b")
    elif invalid == "missing-base":
        reference.ClearField("base_id")
    elif invalid == "body":
        reference.event.text = ""
    elif invalid == "entity":
        reference.event.entity = "actor"
    elif invalid == "principal":
        reference.event.principal = "forged"
    elif invalid == "reuse":
        reference.reuse_dependencies = False
    elif invalid == "dependencies":
        reference.dependencies.append(obs.Edge(source=version, destination="input"))
    before = _state(client)
    invoke = _resolve if invalid == "legacy-rpc" else _resolve_compact
    _reject(lambda: invoke(client, session, _snapshot("must-not-commit"), reference))
    assert _state(client) == before


def test_compact_hash_is_bound_to_scope_and_captured_fields(engine):
    client, session = engine.client(), _sid()
    event = _snapshot("input", "body", agent="model", role=obs.LLM, entity="actor",
                      tools=[obs.Tool(name="call", arguments="{}")], derived_from=obs.Tool(name="source"))
    version, = _resolve(client, session, event)
    reference = _reference(event.event, version)
    _reject(lambda: _resolve_compact(client, _sid(), reference))
    _reject(lambda: _resolve_compact(engine.client(engine.tenant_b_key), session, reference))
    for field in ("text", "agent", "role", "entity", "tools", "derived_from"):
        changed = obs.Event.FromString(event.event.SerializeToString())
        changed.ClearField(field)
        _reject(lambda: _resolve_compact(client, session, _reference(changed, version)))
    assert _resolve_compact(client, session, reference) == [version]


def test_compact_references_remain_quiet_after_restart_and_concurrent_retries(engine_factory):
    from concurrent.futures import ThreadPoolExecutor
    owned = engine_factory()
    client, session = owned.client(), _sid()
    original = [_snapshot("input", "body"), _snapshot("output", "result", sources=["input"])]
    versions = _resolve(client, session, *original)
    references = [_reference(snapshot.event, version) for snapshot, version in zip(original, versions, strict=True)]
    assert _resolve_compact(client, session, *references) == versions
    owned.restart()
    client = owned.client()
    before = _state(client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _resolve_compact(client, session, *references), range(16)))
    assert results == [versions] * 16
    assert _state(client) == before


def test_compact_fingerprint_vectors_match_stored_proto_projection(engine):
    import json
    from pathlib import Path

    from google.protobuf.json_format import ParseDict
    fixture = next(parent / "tests/fixtures/message-content-hashes.json" for parent in Path(__file__).resolve().parents
                    if (parent / "tests/fixtures/message-content-hashes.json").is_file())
    client, session = engine.client(), _sid()
    for index, vector in enumerate(json.loads(fixture.read_text())["vectors"]):
        event = ParseDict(vector["event"], obs.Event())
        assert _content_hash(event).hex() == vector["sha256"]
        event.id = f"golden-{index}"
        version, = _resolve(client, session, obs.EventSnapshot(event=event))
        assert _resolve_compact(client, session, _reference(event, version)) == [version]
