"""Concurrent deployments keep configuration scoped to each session."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

import pytest
from sasy.proto import policy_engine_pb2 as pe

pytestmark = pytest.mark.integration
POLICY = '''
IsAuthorized(idx) :- Actions(idx, _).
Unauthorized(idx) :- Actions(idx, a), a = $CallTool(fn, _), PolicyMetadata("blocked", fn, _).
'''


@pytest.mark.parametrize("mode", ["distinct-source", "shared-source", "update-metadata"])
def test_concurrent_session_metadata(engine, mode):
    barrier = threading.Barrier(4)

    def worker(index):
        client = engine.client()
        tool = f"tool_{index}"
        source = POLICY + (f"\n// deployment {index}" if mode == "distinct-source" else "")
        outcomes = []
        for _ in range(3):
            session = uuid.uuid4().hex
            barrier.wait(timeout=180)
            client.set_policy(source, session, metadata=[] if mode == "update-metadata" else [("blocked", tool)])
            if mode == "update-metadata":
                response = client.policy.UpdatePolicyMetadata(pe.UpdatePolicyMetadataRequest(session_id=session,
                    facts=[pe.PolicyMetadataFact(rel="blocked", a=tool)]), metadata=client.metadata, timeout=30)
                assert response.accepted
            barrier.wait(timeout=180)
            outcomes.append((client.check(session, tool).results[0].authorized,
                             client.check(session, f"tool_{(index + 1) % 4}").results[0].authorized))
        return outcomes

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    # Every future is observed, so upload/RPC/thread failures cannot look like a
    # successful test with an empty list of "leaked" outcomes.
    assert results == [[(False, True)] * 3] * 4


def test_disabled_rule_does_not_relax_other_sessions(engine):
    source = '''
.decl Disabled()
Disabled() :- PolicyMetadata("rule_off", "outbound", _).
IsAuthorized(idx) :- Actions(idx, _).
Unauthorized(idx) :- Actions(idx, a), IsTool(a, "send"), !Disabled().
'''
    barrier = threading.Barrier(4)

    def worker(index):
        client = engine.client()
        outcomes = []
        for _ in range(3):
            session = uuid.uuid4().hex
            barrier.wait(timeout=180)
            client.set_policy(source, session, metadata=[("rule_off", "outbound")] if index == 0 else [])
            barrier.wait(timeout=180)
            outcomes.append(client.check(session, "send").results[0].authorized)
        return outcomes

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    assert results == [[True] * 3, [False] * 3, [False] * 3, [False] * 3]
