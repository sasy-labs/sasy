"""Real evaluator oracle callbacks to an owned, deterministic HTTP provider."""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sasy.proto import observability_pb2 as obs

pytestmark = pytest.mark.integration

ORACLE_POLICY = """
IsAuthorized(idx) :- Actions(idx, a).

// @deny_message: The dependent user message was classified as blocked
Unauthorized(idx) :-
    Actions(idx, a), a = $CallTool(_, _),
    CurrentDepends(id), SentMessage(id, msg),
    msg.agent_role = $UserRole(),
    @llm_check_fn("Does the context contain [blocked]?", msg.contents) = 1.

DenyUnauthorized(idx) :- Actions(idx, a), a = $CallTool(_, _).
"""


@pytest.fixture(scope="module")
def oracle(engine_factory):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request))
            user_text = next(m["content"] for m in request["messages"] if m["role"] == "user")
            # Inspect only the context: the question itself contains [blocked].
            context = user_text.split("Context to analyze:\n", 1)[1]
            response = {
                "id": "local-oracle", "object": "chat.completion", "created": 0,
                "model": "local-oracle",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps({"result": "[blocked]" in context}),
                }}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        engine = engine_factory(
            LLM_ENABLED="true", LLM_PROVIDER="openai", LLM_CHECK_MODEL="local-oracle",
            LLM_TIMEOUT_SECS="5", OPENAI_API_KEY="local-test-placeholder",
            # An unsupported URL disables the optional cache without contacting
            # a developer's Redis/Valkey service on its default port.
            VALKEY_URL="disabled://local-test",
            OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
        )
        try:
            yield engine.client(), requests
        finally:
            engine.stop()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _chain(client, session_id, text):
    user_id, assistant_id = uuid.uuid4().hex, uuid.uuid4().hex
    client.observability.RegisterEventsWithDependencies(
        obs.EventsWithDependencies(
            session_id=session_id,
            events=[obs.Event(id=user_id, text=text, role=obs.USER),
                    obs.Event(id=assistant_id, text="Tool request", role=obs.LLM)],
            edges=[obs.Edge(source=user_id, destination=assistant_id)],
        ), metadata=client.metadata, timeout=10,
    )
    return assistant_id


@pytest.mark.parametrize("text, authorized", [
    ("Synthetic [blocked] content", False),
    ("Synthetic benign content", True),
])
def test_oracle_classification_controls_denial(oracle, text, authorized):
    client, requests = oracle
    session_id = uuid.uuid4().hex
    client.set_policy(ORACLE_POLICY, session_id)
    assistant_id = _chain(client, session_id, text)
    before = len(requests)
    response = client.check(session_id, "send_summary", nodes=[assistant_id])
    assert len(response.results) == 1
    assert response.results[0].authorized is authorized
    assert len(requests) > before, "A default or cached false result must not masquerade as an oracle call"
    path, request = requests[-1]
    assert path == "/v1/chat/completions"
    assert request["model"] == "local-oracle"
    assert any(text in message.get("content", "") for message in request["messages"])


def test_no_dependent_user_message_does_not_call_oracle(oracle):
    client, requests = oracle
    session_id = uuid.uuid4().hex
    client.set_policy(ORACLE_POLICY, session_id)
    before = len(requests)
    response = client.check(session_id, "send_summary")
    assert len(response.results) == 1
    assert response.results[0].authorized
    assert len(requests) == before
