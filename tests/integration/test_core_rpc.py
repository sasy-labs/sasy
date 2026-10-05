"""Core service contracts using generated credentials and synthetic policies."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import uuid

import grpc
import pytest

from sasy.proto import credential_server_pb2 as cred
from sasy.proto import observability_pb2 as obs
from sasy.proto import policy_engine_pb2 as pe
from sasy.proto import reference_monitor_pb2 as rm

pytestmark = pytest.mark.integration
ALLOW = "IsAuthorized(idx) :- Actions(idx, _)."


@pytest.fixture
def client(engine):
    return engine.client()


@pytest.fixture
def session_id():
    return uuid.uuid4().hex


def test_health_and_sync_status(client):
    assert client.policy.Health(pe.HealthRequest(), metadata=client.metadata, timeout=10).healthy
    assert client.policy.GetSyncStatus(pe.SyncStatusRequest(), metadata=client.metadata, timeout=10).connected


@pytest.mark.parametrize("source,valid", [(ALLOW, True), ("", True), ("Bad(x) :- Missing(x).", False)])
def test_validate_policy(client, source, valid):
    result = client.policy.ValidatePolicy(pe.ValidatePolicyRequest(policy_source=source), metadata=client.metadata, timeout=60)
    assert result.valid is valid
    assert bool(result.error_output) is not valid
    if valid:
        assert result.desugared_source


@pytest.mark.parametrize("api_key", [None, "invalid-generated-test-key"])
def test_missing_or_invalid_credentials_rejected(engine, api_key):
    client = engine.client()
    with pytest.raises(grpc.RpcError) as error:
        client.policy.CheckAuthorization(pe.AuthorizationRequest(session_id="unauthenticated"),
                                         metadata=[] if api_key is None else [("x-api-key", api_key)], timeout=10)
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_credential_roundtrip_and_global_fallback(engine):
    client = engine.client(engine.proxy_key)
    for entity, service, pairs in [("test-user", "roundtrip", {"token": "synthetic-token", "secret": "synthetic-secret"}),
                                   ("*", "fallback", {"token": "synthetic-global"})]:
        result = client.credentials.SetCredentials(cred.SetCredentialsRequest(entity=entity, service=service,
            credentials=[cred.Credential(key=k, value=v) for k, v in pairs.items()]), metadata=client.metadata, timeout=10)
        assert result.response
        result = client.credentials.GetCredentials(cred.CredentialsRequest(entity="test-user", service=service), metadata=client.metadata, timeout=10)
        assert {c.key: c.value for c in result.credentials} == pairs
    assert not client.credentials.GetCredentials(cred.CredentialsRequest(entity="unknown", service="unknown"), metadata=client.metadata, timeout=10).credentials
    client.credentials.SetCredentials(cred.SetCredentialsRequest(entity="test-user", service="roundtrip",
        credentials=[cred.Credential(key="token", value="replacement")]), metadata=client.metadata, timeout=10)
    replaced = client.credentials.GetCredentials(cred.CredentialsRequest(entity="test-user", service="roundtrip"), metadata=client.metadata, timeout=10)
    assert {c.key: c.value for c in replaced.credentials} == {"token": "replacement", "secret": "synthetic-secret"}


@pytest.mark.parametrize("method", ["GetCredentials", "SetCredentials"])
def test_credential_permissions(client, method):
    request = cred.CredentialsRequest(entity="other", service="example") if method == "GetCredentials" else cred.SetCredentialsRequest(entity="other", service="example")
    with pytest.raises(grpc.RpcError) as error:
        getattr(client.credentials, method)(request, metadata=client.metadata, timeout=10)
    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_register_generated_and_preset_event_ids(client, session_id):
    result = client.observability.RegisterEvents(obs.Events(session_id=session_id, events=[
        obs.Event(text="input", role=obs.USER), obs.Event(text="output", role=obs.LLM), obs.Event(id="explicit", text="preset")]),
        metadata=client.metadata, timeout=10)
    assert len(set(result.ids)) == 3
    assert all(result.ids)
    assert result.ids[2] == "explicit"


def test_dependencies_slices_and_stamped_principal(client, session_id):
    client.observability.RegisterEvents(obs.Events(session_id=session_id, events=[
        obs.Event(id=n, text=n, role=obs.USER, principal="forged", entity="user-label") for n in ("a", "b", "c")]), metadata=client.metadata, timeout=10)
    response = client.observability.RegisterDependencies(obs.Dependencies(session_id=session_id, edges=[
        obs.Edge(source="a", destination="b", proximal=True), obs.Edge(source="b", destination="c", proximal=True)]), metadata=client.metadata, timeout=10)
    assert response.response == "ok"
    for method, root in [("BackwardSlice", "c"), ("ForwardSlice", "a")]:
        graph = getattr(client.observability, method)(obs.SliceRequest(event_id=root, session_id=session_id), metadata=client.metadata, timeout=10)
        assert {n.id for n in graph.nodes} == {"a", "b", "c"}
        assert {(e.source, e.destination) for e in graph.edges} == {("a", "b"), ("b", "c")}
        assert {n.principal for n in graph.nodes} == {"tenant-a-client"}
        assert {n.entity for n in graph.nodes} == {"user-label"}


def test_atomic_registration_and_state(engine, session_id):
    client = engine.client(engine.admin_key)
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    response = client.observability.RegisterEventsWithDependencies(obs.EventsWithDependencies(session_id=session_id,
        events=[obs.Event(id=a, text="source"), obs.Event(id=b, text="destination")], edges=[obs.Edge(source=a, destination=b)]), metadata=client.metadata, timeout=10)
    assert set(response.ids) == {a, b}
    state = client.updates.GetState(obs.StateRequest(), metadata=client.metadata, timeout=10)
    assert {a, b} <= {e.id for e in state.events}
    assert (a, b) in {(e.source, e.destination) for e in state.edges}
    status = client.policy.GetSyncStatus(pe.SyncStatusRequest(), metadata=client.metadata, timeout=10)
    assert status.node_count >= 2


def test_register_and_read_computation(client, session_id):
    span, trace = uuid.uuid4().hex[:16], uuid.uuid4().hex
    response = client.observability.RegisterComputations(obs.Computations(session_id=session_id, computations=[obs.Computation(
        trace_id=trace, span_id=span, name="test-span", start_time_ns=1000, end_time_ns=2000,
        duration_ns=1000, status_code=1, attributes_json='{"fixture":true}', events_json="[]")]), metadata=client.metadata, timeout=10)
    assert len(response.ids) == 1
    assert response.ids[0]
    stored = client.observability.GetSpan(obs.SpanRequest(span_id=span, session_id=session_id), metadata=client.metadata, timeout=10)
    assert stored.name == "test-span"
    assert stored.trace_id == trace
    assert stored.attributes_json == '{"fixture":true}'


def test_action_order_timing_and_transform_ids(client, session_id):
    source = '''
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "allowed").
ApplyTransform(idx, "synthetic_transform") :- Actions(idx, a), IsTool(a, "allowed").
'''
    client.set_policy(source, session_id)
    result = client.policy.CheckAuthorization(pe.AuthorizationRequest(session_id=session_id, actions=[
        pe.Action(tool_call=pe.ToolCallAction(fn_name=n, args="{}")) for n in ("allowed", "denied", "allowed")]), metadata=client.metadata, timeout=30)
    assert [(r.index, r.authorized) for r in result.results] == [(0, True), (1, False), (2, True)]
    assert list(result.results[0].transform_ids) == ["synthetic_transform"]
    assert result.results[1].deny_if_unauthorized
    assert result.results[1].trace.reasons
    assert result.HasField("timing")
    assert result.timing.total_us >= result.timing.eval_us
    assert result.timing.backend


def test_reference_monitor_uses_authenticated_roles(engine, session_id):
    source = 'IsAuthorized(idx) :- Actions(idx, _), HasRole("can-approve").'
    a, b = engine.client(), engine.client(engine.tenant_b_key)
    a.set_policy(source, session_id)
    b.set_policy(source, session_id)
    request = rm.ToolCallRequest(session_id=session_id, fn_name="probe", args="{}")
    assert a.monitor.CheckToolCall(request, metadata=a.metadata, timeout=30).authorized
    assert not b.monitor.CheckToolCall(request, metadata=b.metadata + [("x-entity", "tenant-a-client"), ("x-roles", "can-approve")], timeout=30).authorized
    delegated = [("x-api-key", engine.proxy_key), ("x-entity", "tenant-a-client"), ("x-roles", "can-approve")]
    assert a.monitor.CheckToolCall(request, metadata=delegated, timeout=30).authorized
    assert not b.check(session_id, principal="tenant-a-client", roles=["can-approve"]).results[0].authorized


def test_http_proxy_enforces_policy_and_blocks_internal_network(client, session_id):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"synthetic upstream response")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = rm.HTTPRequest(session_id=session_id, request=rm.BaseRequest(url=f"http://127.0.0.1:{server.server_port}/probe", method="GET"))
        client.set_policy(ALLOW, session_id)
        allowed = list(client.monitor.ProxyHTTP(iter([request]), metadata=client.metadata, timeout=30))
        assert allowed[0].status == 403
        assert b"disallowed (internal) addresses" in allowed[0].message.content
        client.set_policy("", session_id)
        denied = list(client.monitor.ProxyHTTP(iter([request]), metadata=client.metadata, timeout=30))
        assert denied[0].status == 403
        assert denied[0].message.content
        assert b"disallowed (internal) addresses" not in denied[0].message.content
        assert calls == []
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
