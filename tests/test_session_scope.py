"""Explicit session boundaries and inherited task lifetimes."""
import asyncio
import importlib
from unittest.mock import Mock

import pytest

scope = importlib.import_module("sasy.instrumentation.session")
_real_end_session = importlib.import_module("sasy.policy.api").end_session


@pytest.fixture(autouse=True)
def isolated_scope(monkeypatch):
    scope.reset_default_session()
    monkeypatch.setattr("sasy.policy.api.end_session", Mock())
    monkeypatch.setattr(scope, "_session_flush_callbacks", [])
    yield
    scope.reset_default_session()


def test_default_is_inactive_and_configuration_preserves_opt_in():
    import sasy
    assert not scope.is_session_active()
    with pytest.raises(scope.SessionScopeError, match="active sasy.session"):
        scope.get_current_session_id()
    sasy.configure()
    assert not scope.is_session_active()
    sasy.configure(process_global_session=True)
    sid = scope.get_current_session_id()
    sasy.configure()
    sasy.configure(process_global_session=True)
    assert scope.get_current_session_id() == sid
    with scope.session("explicit"):
        assert scope.get_current_session_id() == "explicit"
    assert scope.get_current_session_id() == sid
    sasy.configure(process_global_session=False)
    assert not scope.is_session_active()


@pytest.mark.parametrize("end_on_exit", [True, False])
def test_child_task_inherits_scope_and_observes_its_lifetime(end_on_exit):
    async def run():
        release = asyncio.Event()
        started = asyncio.Event()
        async def child():
            assert scope.get_current_session_id() == "parent"
            started.set()
            await release.wait()
            return scope.get_current_session_id()
        with scope.session("parent", end_on_exit=end_on_exit):
            task = asyncio.create_task(child())
            await started.wait()
        scope.configure_default_session("fallback")
        release.set()
        if end_on_exit:
            with pytest.raises(scope.SessionScopeError, match="ended"):
                await task
        else:
            assert await task == "parent"
        assert scope.get_current_session_id() == "fallback"
    asyncio.run(run())


def test_captured_scope_cannot_reopen_closed_session():
    with scope.session("old"):
        captured = scope.capture_session_scope()
    with scope.session("new"):
        with pytest.raises(scope.SessionScopeError, match="ended"):
            with scope.bind_session_scope(captured):
                pytest.fail("closed scope reopened")
        assert scope.get_current_session_id() == "new"


def test_disabled_default_invalidates_captured_scope():
    scope.configure_process_global_session(True)
    captured = scope.capture_session_scope()
    scope.configure_process_global_session(False)
    with pytest.raises(scope.SessionScopeError, match="ended"):
        with scope.bind_session_scope(captured):
            pytest.fail("disabled scope reopened")


def test_flush_runs_inside_live_scope_before_end():
    seen = []
    scope.register_session_flush(lambda: seen.append(scope.get_current_session_id()))
    with scope.session("flush"):
        pass
    assert seen == ["flush"]
    assert not scope.is_session_active()


def test_global_session_is_explicit_and_restores_outer_scope():
    with scope.session("outer"):
        with scope.global_session():
            assert scope.is_session_active()
            assert scope.current_wire_session_id() is None
        assert scope.get_current_session_id() == "outer"
    assert not scope.is_session_active()


def test_manual_calls_raise_before_connecting(monkeypatch):
    from sasy.observability import api
    from sasy.reference_monitor import api as rm
    connect = Mock(side_effect=AssertionError("connection attempted"))
    monkeypatch.setattr(api, "get_stub", connect)
    monkeypatch.setattr(rm, "get_stub", connect)
    calls = [
        lambda: api.record_events([]),
        lambda: api.record_dependencies([]),
        lambda: api.record_events_with_dependencies([], []),
        lambda: api.register_computations([]),
        lambda: api.resolve_events([]),
        lambda: rm.check_tool_call("tool", "{}"),
    ]
    for call in calls:
        with pytest.raises(scope.SessionScopeError):
            call()
    connect.assert_not_called()


def test_async_manual_calls_raise_before_connecting(monkeypatch):
    from sasy.observability import api
    from sasy.reference_monitor import api as rm
    connect = Mock(side_effect=AssertionError("connection attempted"))
    monkeypatch.setattr(api, "get_async_stub", connect)
    monkeypatch.setattr(rm, "get_async_stub", connect)
    async def run():
        for call in [lambda: api.record_events_async([]), lambda: rm.check_tool_call_async("tool", "{}")]:
            with pytest.raises(scope.SessionScopeError):
                await call()
    asyncio.run(run())
    connect.assert_not_called()


def test_pending_action_cannot_cross_same_named_scope():
    from sasy.instrumentation.dependencies import ComputationScopeError, bind
    with scope.session("same", end_on_exit=False):
        pending = bind()
    with scope.session("same"):
        with pytest.raises(ComputationScopeError, match="different SASY session scope"):
            pending.check()


def test_async_authorization_refuses_allow_after_scope_ends(monkeypatch):
    from types import SimpleNamespace

    from sasy.reference_monitor import api as rm
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def check(*args, **kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(authorized=True)
        monkeypatch.setattr(rm, "get_async_stub", lambda *_: SimpleNamespace(CheckToolCall=check))
        monkeypatch.setattr(rm, "_metadata", lambda: [])
        with scope.session("pending"):
            task = asyncio.create_task(rm.check_tool_call_async("tool", "{}"))
            await entered.wait()
        release.set()
        with pytest.raises(scope.SessionScopeError, match="ended"):
            await task
    asyncio.run(run())


def test_captured_scope_preserves_absent_entity_after_default_changes(monkeypatch):
    monkeypatch.setattr(scope, "_default_entity", None)
    with scope.session("entity", end_on_exit=False):
        captured = scope.capture_session_scope()
    scope.configure_default_entity("later-actor")
    with scope.bind_session_scope(captured):
        assert scope.get_current_entity() is None
    assert scope.get_current_entity() == "later-actor"


def test_to_thread_inherits_active_scope():
    async def run():
        with scope.session("thread"):
            assert await asyncio.to_thread(scope.get_current_session_id) == "thread"
    asyncio.run(run())


@pytest.mark.parametrize("async_proxy", [False, True])
def test_manual_proxy_cannot_override_active_session(async_proxy):
    from sasy.instrumentation.dependencies import ComputationScopeError
    from sasy.proto.reference_monitor_pb2 import HTTPRequest
    from sasy.reference_monitor import api
    proxy = api.proxy_http_async if async_proxy else api.proxy_http
    with scope.session("active"):
        with pytest.raises(ComputationScopeError, match="different SASY session"):
            proxy(HTTPRequest(session_id="unrelated"))


def test_ending_continued_session_invalidates_child_even_after_reopen():
    async def run():
        release = asyncio.Event()
        started = asyncio.Event()
        async def child():
            started.set()
            await release.wait()
            return scope.get_current_session_id()
        with scope.session("continued", end_on_exit=False):
            task = asyncio.create_task(child())
            await started.wait()
        with scope.session("continued"):
            pass
        with scope.session("continued", end_on_exit=False):
            assert scope.get_current_session_id() == "continued"
            release.set()
            with pytest.raises(scope.SessionScopeError, match="ended"):
                await task
    asyncio.run(run())


@pytest.mark.parametrize("rpc_failure", [False, True])
def test_explicit_end_invalidates_all_scopes_and_flushes_before_rpc(monkeypatch, rpc_failure):
    from types import SimpleNamespace

    from sasy.policy import api
    # Keep the real function despite the fixture's context teardown mock.
    end_session = _real_end_session
    seen = []
    with scope.session("explicit-end", end_on_exit=False):
        first = scope.capture_session_scope()
    with scope.session("explicit-end", end_on_exit=False):
        second = scope.capture_session_scope()
    with scope.session("other", end_on_exit=False):
        other = scope.capture_session_scope()
    assert first.lease is not second.lease
    def flush():
        assert not first.lease.closed and not second.lease.closed
        seen.append("flush")
    def rpc(*args, **kwargs):
        assert first.lease.closed and second.lease.closed
        seen.append("rpc")
        if rpc_failure:
            raise RuntimeError("offline")
        return SimpleNamespace(was_active=True)
    scope.register_session_flush(flush)
    monkeypatch.setattr(api, "get_stub", lambda *_: SimpleNamespace(EndSession=rpc))
    monkeypatch.setattr(api, "_metadata", lambda: [])
    assert end_session("explicit-end") is not rpc_failure
    assert seen == ["flush", "rpc"]
    assert not other.lease.closed
    for captured in (first, second):
        with pytest.raises(scope.SessionScopeError, match="ended"):
            with scope.bind_session_scope(captured):
                pytest.fail("ended scope reopened")


def test_ending_default_session_requires_explicit_fresh_default(monkeypatch):
    from types import SimpleNamespace

    from sasy.policy import api
    scope.configure_default_session("default-ended")
    captured = scope.capture_session_scope()
    monkeypatch.setattr(api, "get_stub", lambda *_: SimpleNamespace(
        EndSession=lambda *args, **kwargs: SimpleNamespace(was_active=True)))
    monkeypatch.setattr(api, "_metadata", lambda: [])
    assert _real_end_session("default-ended")
    with pytest.raises(scope.SessionScopeError, match="ended"):
        scope.is_session_active()
    scope.configure_process_global_session(True)
    assert scope.get_current_session_id() != "default-ended"
    with pytest.raises(scope.SessionScopeError, match="ended"):
        with scope.bind_session_scope(captured):
            pytest.fail("old default revived")


def test_session_lease_registry_does_not_retain_completed_scopes():
    import gc
    import weakref
    with scope.session("collectible", end_on_exit=False):
        captured = scope.capture_session_scope()
        lease_ref = weakref.ref(captured.lease)
    del captured
    gc.collect()
    assert lease_ref() is None
    assert not any(sid == "collectible" for sid, _ in scope._session_leases)
