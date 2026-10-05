"""Run the public message-flow demo against an owned engine without a model provider."""

import subprocess
import sys
from pathlib import Path

import pytest
import sasy

pytestmark = pytest.mark.integration


def _repo_root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "examples/message-flow/demo.py").is_file())


def _environment(engine):
    return dict(engine.env, SASY_URL=engine.address, SASY_API_KEY=engine.tenant_a_key,
                TLS_CA_PATH=str(engine.root / "tls/ca.pem"), SASY_CHANNEL_POOL_SIZE="1",
                PYTHONPATH=str(Path(sasy.__file__).resolve().parent.parent))


def _run(engine, arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=engine.root, env=_environment(engine),
                            capture_output=True, text=True, timeout=180)
    output = (result.stdout + result.stderr).replace(engine.tenant_a_key, "[REDACTED]")
    assert result.returncode == 0, output
    return output


def test_scripted_message_flow(engine):
    output = _run(engine, [str(_repo_root() / "examples/message-flow/demo.py"), "--quiet"])
    assert "sensitive only: ALLOW send to reviewer@example.net; sent=1" in output
    assert "untrusted + sensitive: DENY send to audit@partner.example; sent=0" in output


def test_provider_loop_with_scripted_responses(engine):
    program = r'''
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "examples/message-flow"))
sys.path.insert(0, sys.argv[2])
import demo
from test_message_flow_example import scenario
import sasy

sasy.configure(ca_path=os.environ["TLS_CA_PATH"])
for untrusted in (False, True):
    sent, decisions = demo.run_live(scenario(untrusted), "scripted", untrusted=untrusted, quiet=True)
    assert len(sent) == int(not untrusted)
    assert decisions[-1] == ("send_summary", not untrusted)
'''
    test_directory = Path(__file__).resolve().parents[1]
    _run(engine, ["-c", program, str(_repo_root()), str(test_directory)])
