"""Run the public SDK/test harness in a fresh interpreter with owned credentials."""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("process_global_session", [False, True])
def test_sdk_session_graph_checks_and_feedback_helpers(engine, process_global_session):
    # A subprocess isolates SDK global configuration/channel caches from other
    # suites and prevents ambient .env credentials from entering this scenario.
    program = r'''
import os
import sasy
from sasy.instrumentation.session import SessionScopeError, current_wire_session_id
from sasy.instrumentation.testing import (
    check_services_running, check_tool_call, setup_test_graph, policy_test_context,
)
from sasy.observability.api import backward_slice

process_global = os.environ["SASY_TEST_PROCESS_GLOBAL"] == "1"
sasy.configure(ca_path=os.environ["TLS_CA_PATH"],
               **({"process_global_session": True} if process_global else {}))
port = int(os.environ["SASY_URL"].rsplit(":", 1)[1])
running, statuses = check_services_running(rm_port=port, obs_port=port)
assert running, statuses
default_session = current_wire_session_id() if process_global else None
def assert_configured_scope():
    if process_global:
        assert default_session and current_wire_session_id() == default_session
    else:
        try:
            current_wire_session_id()
        except SessionScopeError:
            pass
        else:
            raise AssertionError("configure() must not enable a default session")
assert_configured_scope()
source = 'IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "read").'
with sasy.session(policy=source, backend="souffle") as handle:
    outer = current_wire_session_id()
    assert outer and outer != default_session
    nodes = setup_test_graph("Synthetic user input", sync_delay=0)
    assert nodes and all(isinstance(n, str) for n in nodes)
    graph = backward_slice(nodes[0])
    assert len(graph.nodes) >= 2
    assert check_tool_call("read", "{}", input_node_ids=nodes).authorized
    assert not check_tool_call("write", "{}", input_node_ids=nodes).authorized
    with sasy.session(policy='IsAuthorized(idx) :- Actions(idx, _).'):
        assert current_wire_session_id() != outer
        assert check_tool_call("write", "{}", input_node_ids=[]).authorized
    assert current_wire_session_id() == outer
    assert not check_tool_call("write", "{}", input_node_ids=nodes).authorized
    handle.set_policy('IsAuthorized(idx) :- Actions(idx, _).')
    assert check_tool_call("write", "{}", input_node_ids=nodes).authorized
assert_configured_scope()
with policy_test_context() as result:
    pass
assert result.denials == []
result.assert_allowed("read")
try:
    result.assert_denied("missing")
except AssertionError:
    pass
else:
    raise AssertionError("assert_denied must reject a missing denial")
print("SDK session, graph, enforcement, rebind and feedback helpers passed")
'''
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               SASY_TEST_PROCESS_GLOBAL="1" if process_global_session else "0")
    result = subprocess.run([sys.executable, "-c", program], cwd=engine.root, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
