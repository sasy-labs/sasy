"""Protected actions resolve pending computation inputs before issuing RPCs."""
import asyncio
from types import SimpleNamespace

import pytest
from sasy.instrumentation import dependencies
from sasy.instrumentation.session import (
    SessionScopeError,
    _session_id_var,
    current_wire_session_id,
    reset_default_session,
    session,
)
from sasy.proto.reference_monitor_pb2 import HTTPRequest, HTTPResponse, ToolCallResponse
from sasy.reference_monitor import api as rm


class Resolver:
    def __init__(self):
        self.task = dependencies._task()
        self.live = True
        self.ready = False
        self.fail = False
        self.wait = None

    def check(self):
        if not self.live or self.task is not dependencies._task():
            raise RuntimeError("Inactive or inherited computation")

    def inputs_sync(self):
        self.check()
        if self.fail:
            raise RuntimeError("Observation failed")
        self.ready = True
        return ["origin", "read"]

    async def inputs_async(self):
        if self.wait is not None:
            await self.wait.wait()
        return self.inputs_sync()


@pytest.fixture
def active_session():
    with session("scope-session", entity="scope-actor", end_on_exit=False):
        yield


@pytest.fixture
def scope(active_session):
    resolver = Resolver()
    token = dependencies.set_resolver(resolver)
    try:
        yield resolver
    finally:
        resolver.live = False
        dependencies.reset_resolver(token)


@pytest.fixture
def backend(monkeypatch, active_session):
    sent, captured = [], []
    monkeypatch.setattr(rm, "_metadata", lambda: [])
    monkeypatch.setattr(rm, "capture_decision", lambda *args, **kwargs: captured.append(args))

    def tool(request, **kwargs):
        sent.append(request)
        return ToolCallResponse(authorized=True)

    async def tool_async(request, **kwargs):
        return tool(request, **kwargs)

    def http(requests, **kwargs):
        sent.extend(requests)
        yield HTTPResponse()

    async def http_async(requests, **kwargs):
        async for request in requests:
            sent.append(request)
        yield HTTPResponse()

    monkeypatch.setattr(rm, "get_stub", lambda _: SimpleNamespace(CheckToolCall=tool, ProxyHTTP=http))
    monkeypatch.setattr(rm, "get_async_stub", lambda _: SimpleNamespace(CheckToolCall=tool_async, ProxyHTTP=http_async))
    return sent, captured


@pytest.mark.parametrize("kind", ["tool", "http"])
def test_sync_actions_resolve_reads_and_capture_effective_ids(scope, backend, kind):
    sent, captured = backend
    if kind == "tool":
        rm.check_tool_call("consume", "{}", [])
        assert captured[0][2] == ["origin", "read"]
    else:
        request = HTTPRequest(session_id="scope-session", input_node_ids=["explicit", "origin"])
        list(rm.proxy_http(request))
        assert list(request.input_node_ids) == ["explicit", "origin"]
    assert scope.ready
    assert list(sent[0].input_node_ids) == (["origin", "read"] if kind == "tool" else ["explicit", "origin", "read"])


@pytest.mark.parametrize("kind", ["tool", "http"])
def test_sync_failed_observation_never_reaches_rm(scope, backend, kind):
    scope.fail = True
    with pytest.raises(RuntimeError, match="Observation failed"):
        if kind == "tool":
            rm.check_tool_call("consume", "{}")
        else:
            list(rm.proxy_http(HTTPRequest(session_id="scope-session")))
    assert not backend[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "http"])
async def test_async_actions_resolve_reads_and_capture_effective_ids(backend, kind):
    scope = Resolver()
    token = dependencies.set_resolver(scope)
    try:
        if kind == "tool":
            await rm.check_tool_call_async("consume", "{}", [])
            assert backend[1][0][2] == ["origin", "read"]
        else:
            assert [r async for r in rm.proxy_http_async(HTTPRequest(session_id="scope-session"))]
        assert scope.ready
        assert list(backend[0][0].input_node_ids) == ["origin", "read"]
    finally:
        dependencies.reset_resolver(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "http"])
async def test_cancelled_flush_never_reaches_rm(backend, kind):
    entered = asyncio.Event()
    async def action():
        scope = Resolver()
        scope.wait = asyncio.Event()
        token = dependencies.set_resolver(scope)
        try:
            entered.set()
            if kind == "tool":
                await rm.check_tool_call_async("consume", "{}")
            else:
                async for _ in rm.proxy_http_async(HTTPRequest(session_id=current_wire_session_id())):
                    pass
        finally:
            dependencies.reset_resolver(token)
    task = asyncio.create_task(action())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not backend[0]


def test_http_freezes_request_at_creation_and_rejects_wrong_session(scope, backend):
    request = HTTPRequest(session_id="scope-session", input_node_ids=["explicit"])
    responses = rm.proxy_http(request)
    request.input_node_ids.append("late")
    list(responses)
    assert list(backend[0][0].input_node_ids) == ["explicit", "origin", "read"]
    for other in [None, "another-session"]:
        with pytest.raises(RuntimeError, match="different SASY session"):
            rm.proxy_http(HTTPRequest(session_id=other))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_lazy_http_cannot_escape_scope_or_task(backend, kind):
    scope = Resolver()
    token = dependencies.set_resolver(scope)
    response = (rm.proxy_http if kind == "sync" else rm.proxy_http_async)(HTTPRequest(session_id=current_wire_session_id()))
    dependencies.reset_resolver(token)
    with pytest.raises(RuntimeError, match="outside it"):
        if kind == "sync":
            next(response)
        else:
            await anext(response)
    token = dependencies.set_resolver(scope)
    response = (rm.proxy_http if kind == "sync" else rm.proxy_http_async)(HTTPRequest(session_id=current_wire_session_id()))
    async def child():
        if kind == "sync":
            return next(response)
        return await anext(response)
    try:
        with pytest.raises(RuntimeError, match="task"):
            await asyncio.create_task(child())
    finally:
        dependencies.reset_resolver(token)
    assert not backend[0]


def test_closed_scope_and_session_change_fail_before_dispatch(scope, backend):
    scope.live = False
    with pytest.raises(RuntimeError, match="Inactive"):
        rm.check_tool_call("consume", "{}")
    scope.live = True
    stream = rm.proxy_http(HTTPRequest(session_id="scope-session"))
    other = _session_id_var.set("another-session")
    try:
        with pytest.raises(RuntimeError, match="session"):
            next(stream)
    finally:
        _session_id_var.reset(other)
    assert not backend[0]


def test_session_without_computation_resolver_preserves_explicit_inputs(backend):
    rm.check_tool_call("consume", "{}", ["explicit"])
    assert list(backend[0][0].input_node_ids) == ["explicit"]


def test_sdk_without_session_rejects_custom_authorization():
    reset_default_session()
    with pytest.raises(SessionScopeError, match="active sasy.session"):
        rm.check_tool_call("consume", "{}", ["explicit"])
