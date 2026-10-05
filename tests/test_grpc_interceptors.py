"""Tests for sasy.auth.grpc_interceptors — RBAC decorators
and context management."""

import pytest
from sasy.auth.providers import AuthResult
from sasy.auth.grpc_interceptors import (
    _auth_context,
    disable_rbac,
    enable_rbac,
    get_auth_context,
    require_any_role,
    require_role,
)

from helpers import AbortError, FakeServicerContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_rbac_enabled():
    """Each test starts with RBAC enabled."""
    enable_rbac()
    yield
    enable_rbac()


@pytest.fixture(autouse=True)
def _clear_auth_context():
    """Reset the auth context between tests."""
    _auth_context.set(None)
    yield
    _auth_context.set(None)


# ---------------------------------------------------------------------------
# get_auth_context / set via _auth_context
# ---------------------------------------------------------------------------


class TestAuthContext:
    """Tests for get_auth_context / set_auth_context."""

    def test_set_get_roundtrip(self) -> None:
        auth = AuthResult(
            authenticated=True,
            entity="alice",
            roles=["admin"],
        )
        _auth_context.set(auth)
        assert get_auth_context() is auth

    def test_default_is_none(self) -> None:
        assert get_auth_context() is None

    def test_context_isolation(self) -> None:
        """Different contexts don't interfere."""
        import contextvars
        import threading

        results: list[AuthResult | None] = [None, None]

        def worker(idx, entity):
            ctx = contextvars.copy_context()

            def run():
                _auth_context.set(
                    AuthResult(
                        authenticated=True,
                        entity=entity,
                        roles=[],
                    )
                )
                results[idx] = get_auth_context()

            ctx.run(run)

        t1 = threading.Thread(target=worker, args=(0, "alice"))
        t2 = threading.Thread(target=worker, args=(1, "bob"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results[0] is not None
        assert results[1] is not None
        assert results[0].entity == "alice"
        assert results[1].entity == "bob"


# ---------------------------------------------------------------------------
# disable_rbac / enable_rbac
# ---------------------------------------------------------------------------


class TestRBACToggle:
    """Tests for disable_rbac / enable_rbac."""

    def test_disable_rbac_skips_checks(self) -> None:
        """With RBAC disabled, @require_role passes without auth."""
        disable_rbac()

        class Svc:
            @require_role("admin")
            def method(self, request, context):
                return "ok"

        svc = Svc()
        ctx = FakeServicerContext()
        assert svc.method("req", ctx) == "ok"

    def test_enable_rbac_enforces_checks(self) -> None:
        """With RBAC enabled, missing auth → abort."""
        enable_rbac()

        class Svc:
            @require_role("admin")
            def method(self, request, context):
                return "ok"

        svc = Svc()
        ctx = FakeServicerContext()
        with pytest.raises(AbortError):
            svc.method("req", ctx)


# ---------------------------------------------------------------------------
# @require_role
# ---------------------------------------------------------------------------


class TestRequireRole:
    """Tests for the @require_role decorator."""

    def test_sync_with_correct_role(self) -> None:
        class Svc:
            @require_role("writer")
            def write(self, request, context):
                return "written"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="alice",
                roles=["writer"],
            )
        )
        svc = Svc()
        ctx = FakeServicerContext()
        assert svc.write("req", ctx) == "written"

    def test_sync_without_role_aborts(self) -> None:
        class Svc:
            @require_role("admin")
            def admin_op(self, request, context):
                return "done"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="alice",
                roles=["reader"],
            )
        )
        svc = Svc()
        ctx = FakeServicerContext()
        with pytest.raises(AbortError) as exc_info:
            svc.admin_op("req", ctx)
        assert "PERMISSION_DENIED" in str(exc_info.value.code)

    def test_sync_unauthenticated_aborts(self) -> None:
        class Svc:
            @require_role("any")
            def op(self, request, context):
                return "done"

        # No auth context set → None
        svc = Svc()
        ctx = FakeServicerContext()
        with pytest.raises(AbortError):
            svc.op("req", ctx)

    @pytest.mark.asyncio
    async def test_async_with_correct_role(self) -> None:
        class Svc:
            @require_role("writer")
            async def write(self, request, context):
                return "async-written"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="bob",
                roles=["writer"],
            )
        )
        svc = Svc()
        ctx = FakeServicerContext()
        result = await svc.write("req", ctx)
        assert result == "async-written"

    @pytest.mark.asyncio
    async def test_async_without_role_aborts(self) -> None:
        class Svc:
            @require_role("admin")
            async def op(self, request, context):
                return "done"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="bob",
                roles=["reader"],
            )
        )
        svc = Svc()
        ctx = FakeServicerContext()
        with pytest.raises(AbortError):
            await svc.op("req", ctx)


# ---------------------------------------------------------------------------
# @require_any_role
# ---------------------------------------------------------------------------


class TestRequireAnyRole:
    """Tests for the @require_any_role decorator."""

    def test_sync_one_matching_role(self) -> None:
        class Svc:
            @require_any_role("admin", "editor")
            def op(self, request, context):
                return "ok"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="u",
                roles=["editor"],
            )
        )
        svc = Svc()
        assert svc.op("req", FakeServicerContext()) == "ok"

    def test_sync_no_matching_role_aborts(self) -> None:
        class Svc:
            @require_any_role("admin", "editor")
            def op(self, request, context):
                return "ok"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="u",
                roles=["viewer"],
            )
        )
        svc = Svc()
        with pytest.raises(AbortError):
            svc.op("req", FakeServicerContext())

    @pytest.mark.asyncio
    async def test_async_one_matching_role(self) -> None:
        class Svc:
            @require_any_role("admin", "editor")
            async def op(self, request, context):
                return "async-ok"

        _auth_context.set(
            AuthResult(
                authenticated=True,
                entity="u",
                roles=["admin"],
            )
        )
        svc = Svc()
        result = await svc.op("req", FakeServicerContext())
        assert result == "async-ok"

    def test_rbac_disabled_skips_check(self) -> None:
        disable_rbac()

        class Svc:
            @require_any_role("admin")
            def op(self, request, context):
                return "bypassed"

        svc = Svc()
        assert svc.op("req", FakeServicerContext()) == "bypassed"
