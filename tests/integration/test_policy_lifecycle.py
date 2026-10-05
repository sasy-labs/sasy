"""Tenant/session boundaries, policy rollout, graph context and persistence."""
import uuid

import grpc
import pytest

from sasy.proto import observability_pb2 as obs
from sasy.proto import policy_engine_pb2 as pe

pytestmark = pytest.mark.integration
ALLOW = "IsAuthorized(idx) :- Actions(idx, _)."
DENY = "// Empty allowlist denies every action."


def sid():
    return uuid.uuid4().hex


def test_distinct_policy_bindings_and_rebind_preserve_graph(engine):
    client = engine.client()
    allowed, denied = sid(), sid()
    first = client.set_policy(ALLOW, allowed)
    second = client.set_policy(DENY, denied)
    assert first != second
    assert client.check(allowed).results[0].authorized
    assert not client.check(denied).results[0].authorized
    client.observability.RegisterEvents(obs.Events(session_id=allowed, events=[obs.Event(id="context", text="kept")]), metadata=client.metadata, timeout=10)
    assert client.set_policy(DENY, allowed) == second
    assert not client.check(allowed).results[0].authorized
    graph = client.observability.BackwardSlice(obs.SliceRequest(event_id="context", session_id=allowed), metadata=client.metadata, timeout=10)
    assert [(n.id, n.text) for n in graph.nodes] == [("context", "kept")]


def test_identical_source_deduplicates_and_installs_both_bindings(engine):
    client = engine.client()
    a, b = sid(), sid()
    source = ALLOW + "\n// unique source " + sid()
    first = client.set_policy(source, a)
    second = client.set_policy(source, b)
    assert first == second
    assert client.check(a).results[0].authorized
    assert client.check(b).results[0].authorized
    # Identity and both working bindings are deterministic assertions; wall-clock
    # cache speed belongs in the microbenchmark rather than a shared CI gate.


def test_same_session_and_node_ids_are_isolated_between_tenants(engine):
    a, b = engine.client(), engine.client(engine.tenant_b_key)
    session = sid()
    a.set_policy(ALLOW, session)
    b.set_policy(DENY, session)
    for client, text in [(a, "tenant A data"), (b, "tenant B data")]:
        client.observability.RegisterEvents(obs.Events(session_id=session, events=[obs.Event(id="same-id", text=text)]), metadata=client.metadata, timeout=10)
    for client, text in [(a, "tenant A data"), (b, "tenant B data")]:
        graph = client.observability.BackwardSlice(obs.SliceRequest(event_id="same-id", session_id=session), metadata=client.metadata, timeout=10)
        assert [(n.id, n.text) for n in graph.nodes] == [("same-id", text)]
    assert a.check(session).results[0].authorized
    assert not b.check(session).results[0].authorized
    foreign = sid()
    a.observability.RegisterEvents(obs.Events(session_id=session, events=[obs.Event(id=foreign, text="private")]), metadata=a.metadata, timeout=10)
    with pytest.raises(grpc.RpcError) as error:
        b.observability.BackwardSlice(obs.SliceRequest(event_id=foreign, session_id=session), metadata=b.metadata, timeout=10)
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_same_node_ids_in_two_sessions_are_isolated(engine):
    client = engine.client()
    sessions = [sid(), sid()]
    for session in sessions:
        client.observability.RegisterEvents(obs.Events(session_id=session, events=[obs.Event(id="same-id", text=session)]), metadata=client.metadata, timeout=10)
    for session in sessions:
        graph = client.observability.BackwardSlice(obs.SliceRequest(event_id="same-id", session_id=session), metadata=client.metadata, timeout=10)
        assert [(n.id, n.text) for n in graph.nodes] == [("same-id", session)]


def test_default_rollout_spares_pin_and_force_replaces_it(engine_factory):
    owned = engine_factory()
    try:
        admin = owned.client(owned.admin_key)
        pinned, fresh = sid(), sid()
        admin.set_policy(ALLOW, pinned)
        admin.set_policy(DENY, scope="default")
        assert admin.check(pinned).results[0].authorized
        assert not admin.check(fresh).results[0].authorized
        admin.set_policy(DENY, scope="force")
        assert not admin.check(pinned).results[0].authorized
    finally:
        owned.stop()


@pytest.mark.parametrize("scope", ["default", "force"])
def test_tenant_default_changes_require_admin(engine, scope):
    client = engine.client()
    target = pe.DefaultTarget() if scope == "default" else pe.ForceTarget()
    with pytest.raises(grpc.RpcError) as error:
        client.policy.SetPolicy(pe.SetPolicyRequest(policy_source=ALLOW, backend="souffle", scope=pe.PolicyScope(**{scope: target})), metadata=client.metadata, timeout=30)
    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_dependency_context_controls_authorization(engine):
    client, session = engine.client(), sid()
    source = '''
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read").
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send"),
    CurrentDepends(id), ToolResult(id, "approve", _).
'''
    client.set_policy(source, session)
    assert client.check(session, "read").results[0].authorized
    assert not client.check(session, "send").results[0].authorized
    client.observability.RegisterEventsWithDependencies(obs.EventsWithDependencies(session_id=session,
        events=[obs.Event(id="approval", text="approved", derived_from=obs.Tool(name="approve", arguments="{}")),
                obs.Event(id="current", text="send output"), obs.Event(id="unrelated", text="other output")],
        edges=[obs.Edge(source="approval", destination="current", proximal=True)]), metadata=client.metadata, timeout=10)
    assert client.check(session, "send", nodes=["current"]).results[0].authorized
    assert not client.check(session, "send", nodes=["unrelated"]).results[0].authorized
    assert not client.check(session, "send").results[0].authorized


def test_restart_restores_session_and_default_and_graph(engine_factory):
    owned = engine_factory()
    try:
        client, admin = owned.client(), owned.client(owned.admin_key)
        pinned = sid()
        client.set_policy(ALLOW, pinned)
        # A restrictive, nonempty default proves replay: a lost default would
        # deny "default_probe" along with everything else after restart.
        admin.set_policy('IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "default_probe").', scope="default")
        client.observability.RegisterEvents(obs.Events(session_id=pinned, events=[obs.Event(id="persisted", text="survives restart")]), metadata=client.metadata, timeout=10)
        assert client.check(pinned).results[0].authorized
        owned.restart()
        client, admin = owned.client(), owned.client(owned.admin_key)
        assert client.check(pinned).results[0].authorized
        assert admin.check(sid(), "default_probe").results[0].authorized
        assert not admin.check(sid(), "other_tool").results[0].authorized
        graph = client.observability.BackwardSlice(obs.SliceRequest(event_id="persisted", session_id=pinned), metadata=client.metadata, timeout=10)
        assert [(n.id, n.text) for n in graph.nodes] == [("persisted", "survives restart")]
        assert "restored persisted policy state" in owned.log_path.read_text()
    finally:
        owned.stop()


def test_fresh_store_has_no_session_policy(engine_factory):
    owned = engine_factory()
    try:
        with pytest.raises(grpc.RpcError) as error:
            owned.client().check(sid())
        # The current RPC maps a missing tenant registry to INTERNAL; either
        # status below is fail-closed, but unrelated failures must not pass.
        assert error.value.code() in (grpc.StatusCode.INTERNAL, grpc.StatusCode.FAILED_PRECONDITION)
        assert error.value.details() == "no policy registered for tenant 'tenant-a'"
    finally:
        owned.stop()


def test_end_session_drops_binding_and_approvals(engine):
    client, session = engine.client(), sid()
    source = 'IsAuthorized(idx) :- Actions(idx, _), PolicyMetadata("approved", "yes", _).'
    client.set_policy(source, session, metadata=[("approved", "yes")])
    assert client.check(session).results[0].authorized
    response = client.policy.EndSession(pe.EndSessionRequest(session_id=session), metadata=client.metadata, timeout=10)
    assert response.was_active
    client.set_policy(source, session)
    assert not client.check(session).results[0].authorized
