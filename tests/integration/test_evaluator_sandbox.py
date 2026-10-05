"""Custom functors cannot read an existing host file outside bubblewrap.

Linux qualification runs must set SASY_REQUIRE_SANDBOX=1. Compilation
or evaluator failures never count as proof of confinement: each probe must
return a normal decision, including a positive control that opens /dev/null.
"""

import json
import os
import platform
import shutil
import uuid

import pytest
from sasy.proto import policy_engine_pb2 as pe

pytestmark = [pytest.mark.integration, pytest.mark.linux_sandbox]

SAFE_FUNCTOR = r'''
#include <souffle/SouffleInterface.h>
extern "C" souffle::RamDomain file_exists(
    souffle::SymbolTable* symbols, souffle::RecordTable*, souffle::RamDomain path) {
    return symbols->decode(path).empty() ? 1 : 0;
}
'''

FILE_FUNCTOR = r'''
#include <souffle/SouffleInterface.h>
#include <fcntl.h>
#include <unistd.h>
extern "C" souffle::RamDomain file_exists(
    souffle::SymbolTable* symbols, souffle::RecordTable*, souffle::RamDomain path) {
    int fd = open(symbols->decode(path).c_str(), O_RDONLY);
    if (fd < 0) return 0;
    close(fd);
    return 1;
}
'''


@pytest.fixture
def sandbox_client(request):
    if platform.system() != "Linux":
        if os.environ.get("SASY_REQUIRE_SANDBOX") == "1":
            pytest.fail("Required evaluator sandbox qualification needs Linux")
        pytest.skip("Evaluator filesystem isolation requires Linux")
    if shutil.which("bwrap") is None:
        pytest.fail("Evaluator filesystem isolation tests need bubblewrap on PATH")
    engine = request.getfixturevalue("engine")
    return engine.client(engine.admin_key)


def _install(client, functor, path):
    # A successful read denies the action. The functor therefore participates
    # in the actual decision and cannot disappear as unused policy output.
    source = f'''
.functor file_exists(path: symbol): unsigned stateful
IsAuthorized(idx) :- Actions(idx, a), @file_exists({json.dumps(str(path))}) = 0.
DenyUnauthorized(idx) :- Actions(idx, a).
'''
    session_id = uuid.uuid4().hex
    response = client.policy.SetPolicy(
        pe.SetPolicyRequest(
            policy_source=source, functor_source=functor, backend="souffle",
            scope=pe.PolicyScope(session=pe.SessionTarget(session_id=session_id)),
        ), metadata=client.metadata, timeout=300,
    )
    assert response.accepted, response.error_output
    return session_id


def test_safe_custom_functor_evaluates(sandbox_client, tmp_path):
    session_id = _install(sandbox_client, SAFE_FUNCTOR, tmp_path / "unused")
    response = sandbox_client.check(session_id)
    assert len(response.results) == 1
    assert response.results[0].authorized


def test_host_file_is_hidden_and_engine_remains_healthy(sandbox_client, tmp_path):
    sentinel = tmp_path / "host-only.txt"
    sentinel.write_text("Synthetic host-only fixture; never a real credential")
    assert sentinel.read_text()
    safe_session = _install(sandbox_client, SAFE_FUNCTOR, sentinel)
    assert sandbox_client.check(safe_session).results[0].authorized

    # The same functor must decode and open an accessible path successfully.
    # This verifies both the ABI and the successful-open branch of the probe.
    accessible_session = _install(sandbox_client, FILE_FUNCTOR, "/dev/null")
    accessible = sandbox_client.check(accessible_session)
    assert len(accessible.results) == 1
    assert not accessible.results[0].authorized, "Positive control must open /dev/null"

    probe_session = _install(sandbox_client, FILE_FUNCTOR, sentinel)
    response = sandbox_client.check(probe_session)
    assert len(response.results) == 1
    assert response.results[0].authorized, "Custom functor opened a host file outside its sandbox"

    # Probing the host filesystem must not poison another session or service.
    recovery = uuid.uuid4().hex
    sandbox_client.set_policy("IsAuthorized(idx) :- Actions(idx, a).", recovery)
    assert sandbox_client.check(recovery).results[0].authorized
    assert sandbox_client.policy.Health(pe.HealthRequest(), metadata=sandbox_client.metadata, timeout=10).healthy
