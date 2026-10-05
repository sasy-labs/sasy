"""Owned engine processes for portable integration tests.

Set SASY_TEST_ENGINE to an explicitly built full engine. Tests never attach to
an existing service or load developer credentials. Missing prerequisites fail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import socket
import subprocess
import time

import grpc
import pytest

from sasy.proto import credential_server_pb2_grpc as credentials_grpc
from sasy.proto import observability_pb2_grpc as observability_grpc
from sasy.proto import policy_engine_pb2 as pe
from sasy.proto import policy_engine_pb2_grpc as policy_grpc
from sasy.proto import reference_monitor_pb2_grpc as monitor_grpc


class EngineClient:
    def __init__(self, engine, api_key):
        self.channel = engine.channel()
        self.metadata = [("x-api-key", api_key)]
        self.policy = policy_grpc.PolicyEngineStub(self.channel)
        self.observability = observability_grpc.ObservabilityStub(self.channel)
        self.updates = observability_grpc.ObservabilityUpdatesStub(self.channel)
        self.credentials = credentials_grpc.CredentialServerStub(self.channel)
        self.monitor = monitor_grpc.RMProxyStub(self.channel)

    def set_policy(self, source, session_id="", *, scope="session", backend="souffle", metadata=()):
        target = {"session": pe.SessionTarget(session_id=session_id)} if scope == "session" else {
            scope: pe.DefaultTarget() if scope == "default" else pe.ForceTarget()
        }
        response = self.policy.SetPolicy(
            pe.SetPolicyRequest(policy_source=source, backend=backend,
                                scope=pe.PolicyScope(**target),
                                policy_metadata=[pe.PolicyMetadataFact(rel=f[0], a=f[1], b=f[2] if len(f) > 2 else "") for f in metadata]),
            metadata=self.metadata, timeout=180,
        )
        assert response.accepted, response.error_output or response.message
        assert response.policy_id
        return response.policy_id

    def check(self, session_id, tool="probe", args="{}", *, nodes=(), **fields):
        return self.policy.CheckAuthorization(
            pe.AuthorizationRequest(session_id=session_id, current_node_ids=nodes,
                                    actions=[pe.Action(tool_call=pe.ToolCallAction(fn_name=tool, args=args))], **fields),
            metadata=self.metadata, timeout=30,
        )


class Engine:
    def __init__(self, binary, root, env_overrides):
        self.binary, self.root = binary, root
        self.process = None
        self.channels = []
        self.env = {k: os.environ[k] for k in ("PATH", "LANG", "SYSTEMROOT") if k in os.environ}
        self.env.update(HOME=str(root), TMPDIR=str(root), LLM_ENABLED="false",
                        PYTHON_DOTENV_DISABLED="1", OTEL_ENABLED="false", RUST_LOG="info",
                        TOKIO_WORKER_THREADS="2")
        self.env.update(env_overrides)
        self.admin_key, self.tenant_a_key, self.tenant_b_key, self.proxy_key = [secrets.token_urlsafe(32) for _ in range(4)]
        subprocess.run([str(binary), "guard-tls", "--output-dir", str(root / "tls"), "--entity", "integration"],
                       cwd=root, env=self.env, check=True, timeout=30, stdout=subprocess.DEVNULL)
        self.root_certificates = (root / "tls/ca.pem").read_bytes()
        auth = {"type": "api_key", "static_keys": dict(zip(
            [self.admin_key, self.tenant_a_key, self.tenant_b_key, self.proxy_key],
            ["admin", "tenant-a-client", "tenant-b-client", "proxy"]))}
        auth_path = root / "auth.json"
        auth_path.write_text(json.dumps(auth))
        auth_path.chmod(0o600)
        roles = ["reference-monitor-user", "observability-writer", "observability-reader", "policy-client"]
        (root / "roles.json").write_text(json.dumps({"default_tenant": "admin-tenant", "entities": {
            "admin": {"roles": roles + ["admin", "credential-reader", "credential-writer", "observability-admin"]},
            "tenant-a-client": {"tenant": "tenant-a", "roles": roles + ["can-approve"]},
            "tenant-b-client": {"tenant": "tenant-b", "roles": roles},
            "proxy": {"roles": roles + ["service-proxy", "credential-reader", "credential-writer"]},
        }}))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.address = f"127.0.0.1:{self.port}"
        self.log_path = root / "server.log"
        self.start()

    def channel(self):
        channel = grpc.secure_channel(self.address, grpc.ssl_channel_credentials(self.root_certificates))
        self.channels.append(channel)
        return channel

    def client(self, api_key=None):
        return EngineClient(self, api_key or self.tenant_a_key)

    def start(self):
        self.log = self.log_path.open("a")
        command = [str(self.binary), "serve", "--addr", self.address,
                   "--data-dir", str(self.root / "graph"), "--credentials-db", str(self.root / "credentials.db"),
                   "--auth-provider", str(self.root / "auth.json"), "--auth-config", str(self.root / "roles.json"),
                   "--tls-cert", str(self.root / "tls/server.pem"), "--tls-key", str(self.root / "tls/server-key.pem")]
        self.process = subprocess.Popen(command, cwd=self.root, env=self.env, stdout=self.log, stderr=self.log, start_new_session=True)
        channel = self.channel()
        ready = grpc.channel_ready_future(channel)
        deadline = time.monotonic() + 90
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("engine exited before becoming ready")
                try:
                    ready.result(timeout=0.2)
                    break
                except grpc.FutureTimeoutError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("engine did not become ready within 90 seconds") from None
        except BaseException:
            self.stop()
            raise RuntimeError(self.log_path.read_text()[-12000:]) from None
        finally:
            ready.cancel()

    def stop(self):
        for channel in self.channels:
            channel.close()
        self.channels.clear()
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if hasattr(self, "log"):
            self.log.close()

    def restart(self):
        self.stop()
        self.start()


@pytest.fixture(scope="session")
def engine_factory(tmp_path_factory):
    configured = os.environ.get("SASY_TEST_ENGINE")
    if not configured:
        pytest.fail("Integration tests require SASY_TEST_ENGINE=/absolute/path/to/sasy; build the full engine first")
    binary = Path(configured).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        pytest.fail(f"SASY_TEST_ENGINE is not an executable file: {binary}")
    instances = []

    def create(**env_overrides):
        root = tmp_path_factory.mktemp("engine")
        try:
            instance = Engine(binary, root, env_overrides)
        except BaseException:
            shutil.rmtree(root)
            raise
        instances.append(instance)
        return instance

    yield create
    for instance in reversed(instances):
        instance.stop()
        shutil.rmtree(instance.root)


@pytest.fixture(scope="session")
def engine(engine_factory):
    return engine_factory()
