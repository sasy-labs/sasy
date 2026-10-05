"""Qualify `sasy.record` with a hand-written agent loop on an owned engine."""
import subprocess
import sys
from pathlib import Path

import pytest
import sasy

pytestmark = pytest.mark.integration

PROGRAM = r'''
import asyncio
import json
import sasy

POLICY = """
IsAuthorized(idx) :- Actions(idx, a), IsTool(a, "send_email").
// @deny_message: The email depends on a confidential document
Unauthorized(idx) :- Actions(idx, a), IsTool(a, "send_email"),
    CurrentDepends(id), ToolResult(id, "read_document", args),
    @json_get_str(args, "name") = "confidential-report".
"""
ARGS = {"to": "team@example.com", "body": "Summary"}

def request(document):
    task = sasy.record("Summarize and email the team", role="user")
    doc = sasy.record("Quarterly numbers", result_of=("read_document", {"name": document}))
    return sasy.record("", role="llm", inputs=[task, doc], tool_calls=[("send_email", ARGS)])

def check(version):
    return sasy.check_tool_call("send_email", json.dumps(ARGS), input_node_ids=[version])

with sasy.session(policy=POLICY):
    assert check(request("public-report")).authorized
    denied = check(request("confidential-report"))
    assert not denied.authorized
    assert "confidential document" in " ".join(denied.denial_reasons)
    # Only the listed inputs count: a message without the document is allowed.
    task = sasy.record("Email the team", role="user")
    clean = sasy.record("", role="llm", inputs=[task], tool_calls=[("send_email", ARGS)])
    assert check(clean).authorized

async def main():
    with sasy.session(policy=POLICY):
        task = await sasy.record_async("Summarize and email the team", role="user")
        doc = await sasy.record_async(
            "Quarterly numbers", result_of=("read_document", {"name": "confidential-report"}))
        version = await sasy.record_async(
            "", role="llm", inputs=[task, doc], tool_calls=[("send_email", ARGS)])
        assert not check(version).authorized

asyncio.run(main())
'''


def test_hand_written_loop_records_ancestry_that_policies_see(engine):
    env = dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
               TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
               PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))
    result = subprocess.run([sys.executable, "-c", PROGRAM], cwd=engine.root, env=env,
                            capture_output=True, text=True, timeout=240)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output
