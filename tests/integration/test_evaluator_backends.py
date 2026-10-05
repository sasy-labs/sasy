"""Compiled evaluator administration and decisions, using owned engine state.

Interpreted evaluator equivalence is exercised by the Rust evaluator tests.
These RPC tests exercise the compiled backend used by the public server.
"""

import uuid

import grpc
import pytest
from sasy.proto import policy_engine_pb2 as pe
from sasy.proto import policy_engine_pb2_grpc as pe_grpc

pytestmark = pytest.mark.integration

ALLOW_ALL = "IsAuthorized(idx) :- Actions(idx, a).\n"
DENY_ALL = "DenyUnauthorized(idx) :- Actions(idx, a).\n"


@pytest.fixture
def evaluator_rpc(engine):
    with engine.channel() as channel:
        yield (
            pe_grpc.PolicyEngineStub(channel),
            [("x-api-key", engine.admin_key)],
            [("x-api-key", engine.tenant_a_key)],
        )


def _set(stub, metadata, source, session_id=None):
    scope = (
        pe.PolicyScope(session=pe.SessionTarget(session_id=session_id))
        if session_id else pe.PolicyScope(force=pe.ForceTarget())
    )
    return stub.SetPolicy(
        pe.SetPolicyRequest(policy_source=source, backend="souffle", scope=scope),
        metadata=metadata, timeout=300,
    )


def _check(stub, metadata, session_id, names=("read_data",)):
    return stub.CheckAuthorization(
        pe.AuthorizationRequest(
            session_id=session_id,
            actions=[pe.Action(tool_call=pe.ToolCallAction(fn_name=name, args="{}")) for name in names],
        ), metadata=metadata, timeout=30,
    )


def test_health_without_admin(evaluator_rpc):
    stub, _, user = evaluator_rpc
    assert stub.Health(pe.HealthRequest(), metadata=user, timeout=10).healthy


def test_status_after_default_install(evaluator_rpc):
    stub, admin, _ = evaluator_rpc
    response = _set(stub, admin, ALLOW_ALL)
    assert response.accepted, response.error_output
    status = stub.GetEvaluatorStatus(pe.EvaluatorStatusRequest(), metadata=admin, timeout=10)
    assert status.alive
    assert status.backend == "souffle"


def test_force_update_requires_admin(evaluator_rpc):
    stub, _, user = evaluator_rpc
    with pytest.raises(grpc.RpcError) as error:
        _set(stub, user, ALLOW_ALL)
    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_status_requires_admin(evaluator_rpc):
    stub, _, user = evaluator_rpc
    with pytest.raises(grpc.RpcError) as error:
        stub.GetEvaluatorStatus(pe.EvaluatorStatusRequest(), metadata=user, timeout=10)
    assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_self_pinned_policy_handles_known_and_unknown_tools(evaluator_rpc):
    stub, _, user = evaluator_rpc
    session_id = uuid.uuid4().hex
    installed = _set(stub, user, ALLOW_ALL, session_id)
    assert installed.accepted, installed.error_output
    result = _check(stub, user, session_id, ("read_data", "unknown_tool"))
    assert [item.index for item in result.results] == [0, 1]
    assert [item.authorized for item in result.results] == [True, True]


def test_timing_and_repeated_decisions(evaluator_rpc):
    stub, _, user = evaluator_rpc
    session_id = uuid.uuid4().hex
    installed = _set(stub, user, ALLOW_ALL, session_id)
    assert installed.accepted, installed.error_output
    for _ in range(5):
        result = _check(stub, user, session_id)
        assert len(result.results) == 1
        assert result.results[0].authorized
        assert result.HasField("timing")
        assert result.timing.backend
        assert result.timing.eval_us > 0


def test_force_update_changes_decision(evaluator_rpc):
    stub, admin, _ = evaluator_rpc
    session_id = uuid.uuid4().hex
    for source, expected in [(ALLOW_ALL, True), (DENY_ALL, False), (ALLOW_ALL, True)]:
        installed = _set(stub, admin, source)
        assert installed.accepted, installed.error_output
        result = _check(stub, admin, session_id)
        assert len(result.results) == 1
        assert result.results[0].authorized is expected


def test_empty_source_does_not_authorize(evaluator_rpc):
    stub, _, user = evaluator_rpc
    session_id = uuid.uuid4().hex
    installed = _set(stub, user, "", session_id)
    assert installed.accepted, installed.error_output
    result = _check(stub, user, session_id)
    assert len(result.results) == 1
    assert not result.results[0].authorized
