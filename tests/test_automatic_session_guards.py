"""Automatic adapters are transparent outside a scope and fail closed after it."""
import contextvars
import importlib
from unittest.mock import Mock

import pytest
from opentelemetry.trace import NoOpTracerProvider
from sasy.instrumentation import http, langroid

scope = importlib.import_module("sasy.instrumentation.session")


@pytest.fixture(autouse=True)
def inactive(monkeypatch):
    monkeypatch.setattr(scope, "_default_session_id", None)
    token = scope._session_id_var.set(None)
    lease = scope._session_lease_var.set(None)
    try:
        yield
    finally:
        scope._session_lease_var.reset(lease)
        scope._session_id_var.reset(token)


@pytest.fixture
def hooks(monkeypatch):
    wrappers = {}
    def register(module, name, wrapper):
        wrappers[name] = wrapper
    monkeypatch.setattr(http.wrapt, "wrap_function_wrapper", register)
    monkeypatch.setattr(langroid, "wrap_function_wrapper", register)
    monkeypatch.setattr(langroid, "get_tracer", lambda *_: NoOpTracerProvider().get_tracer(__name__))
    monkeypatch.setattr(http, "_instrumented", False)
    langroid.instrument.cache_clear()
    http.instrument()
    langroid.instrument()
    yield wrappers
    langroid.instrument.cache_clear()


@pytest.mark.asyncio
async def test_all_automatic_entry_hooks_are_transparent_without_session(hooks):
    # Deliberately invalid framework inputs ensure the inactive path does no
    # SASY-specific validation, provenance lookup, auth, or observation.
    for name, hook in hooks.items():
        value = object()
        calls = []
        if name.endswith("_async") or name.startswith("AsyncClient."):
            async def original(*args, **kwargs):
                calls.append((args, kwargs))
                return value
            result = await hook(original, None, (value,), {"extra": value})
        else:
            def original(*args, **kwargs):
                calls.append((args, kwargs))
                return value
            result = hook(original, None, (value,), {"extra": value})
        assert result is value, name
        assert calls == [((value,), {"extra": value})], name


@pytest.mark.parametrize("process_default", [False, True])
def test_http_enforces_in_active_and_process_default_scope(monkeypatch, hooks, process_default):
    import httpx
    def denied(*args, **kwargs):
        raise RuntimeError("reference monitor reached")
    monkeypatch.setattr(http, "proxy_http", denied)
    monkeypatch.setattr(http, "get_sasy_config", lambda: type("Config", (), {"auth_hook": type("Auth", (), {"get_metadata": lambda self: []})()})())
    request = httpx.Request("GET", "https://example.org/")
    original = Mock()
    def run():
        with pytest.raises(RuntimeError, match="reference monitor reached"):
            hooks["Client._send_single_request"](original, None, (request,), {})
    if process_default:
        scope.configure_process_global_session(True)
        try:
            run()
        finally:
            scope.configure_process_global_session(False)
    else:
        with scope.session("http-guard", end_on_exit=False):
            run()
    original.assert_not_called()


@pytest.mark.asyncio
async def test_closed_inherited_scope_never_falls_back_to_original(monkeypatch, hooks):
    monkeypatch.setattr("sasy.policy.api.end_session", lambda *_: None)
    with scope.session("closed-hook"):
        inherited = contextvars.copy_context()
    for name, hook in hooks.items():
        original = Mock()
        if name.endswith("_async") or name.startswith("AsyncClient."):
            # Creating the coroutine does not run its body; execute it in a
            # task created under the inherited context.
            import asyncio
            task = inherited.run(asyncio.create_task, hook(original, None, (), {}))
            with pytest.raises(scope.SessionScopeError):
                await task
        else:
            with pytest.raises(scope.SessionScopeError):
                inherited.run(hook, original, None, (), {})
        original.assert_not_called()


@pytest.mark.asyncio
async def test_to_thread_inherits_protected_http_scope(monkeypatch, hooks):
    import asyncio

    import httpx
    def deny(*args, **kwargs):
        assert scope.current_wire_session_id() == "thread-http"
        raise RuntimeError("checked in thread")
    monkeypatch.setattr(http, "proxy_http", deny)
    monkeypatch.setattr(http, "get_sasy_config", lambda: type("Config", (), {"auth_hook": type("Auth", (), {"get_metadata": lambda self: []})()})())
    original = Mock()
    with scope.session("thread-http", end_on_exit=False):
        with pytest.raises(RuntimeError, match="checked in thread"):
            await asyncio.to_thread(
                hooks["Client._send_single_request"], original, None,
                (httpx.Request("GET", "https://example.org/"),), {},
            )
    original.assert_not_called()
