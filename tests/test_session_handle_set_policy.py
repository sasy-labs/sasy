"""Unit tests for ``_SessionHandle.set_policy`` failure modes.

The contract: when ``set_policy`` raises (server rejects, RPC
errors, etc.), the surrounding ``session()`` block must still
restore the outer ContextVar values on ``__exit__``. The
in-block ``_policy_id_var.set(new_id)`` after a successful call
also has to unwind cleanly via the original ``pid_token`` reset
without leaving residual state in the outer scope.

These are pure-Python tests with no server dependency — they
monkey-patch ``sasy.policy.api.set_session_policy`` to drive
the failure paths.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sasy.instrumentation.session import (
    _policy_id_var,
    _session_id_var,
    get_current_policy_id,
    get_current_session_id,
    session,
)


class _FakeResp:
    """Minimal stand-in for the SetPolicy gRPC response."""

    def __init__(
        self, accepted: bool, policy_id: str = "", message: str = "", error_output: str = ""
    ) -> None:
        self.accepted = accepted
        self.policy_id = policy_id
        self.message = message
        self.error_output = error_output


@pytest.fixture(autouse=True)
def _mock_end_session():
    """Session teardown must never contact a developer's configured server."""
    with patch("sasy.policy.api.end_session"):
        yield


def _outer_state() -> tuple[object, object]:
    """Snapshot of (session_id, policy_id) ContextVar values."""
    return _session_id_var.get(), _policy_id_var.get()


def test_set_policy_server_reject_propagates_and_restores_context():
    """Server rejection (``resp.accepted = False``) raises
    ``RuntimeError`` from ``handle.set_policy(...)``; the outer
    ``session()`` block's finally must still reset both ContextVars
    to their pre-block values."""
    before = _outer_state()

    initial = _FakeResp(accepted=True, policy_id="policy-initial")
    rejection = _FakeResp(accepted=False, message="bad policy", error_output="syntax error")

    # First call (session entry) succeeds; second call (set_policy) fails.
    with patch(
        "sasy.policy.api.set_session_policy",
        side_effect=[initial, rejection],
    ):
        with pytest.raises(RuntimeError, match="bad policy"):
            with session(session_id="reject-test", policy="// dummy") as h:
                # Mid-block: confirm we picked up the initial policy id.
                assert get_current_policy_id() == "policy-initial"
                assert get_current_session_id() == "reject-test"
                h.set_policy("// also dummy")  # raises

    # After the exception unwinds the block, both vars must be back
    # to their original values — no residue from the failed swap.
    assert _outer_state() == before


def test_set_policy_rpc_exception_restores_context():
    """An RPC-level exception (network/auth/etc.) also has to leave
    the outer scope clean. We simulate via ``side_effect``."""
    before = _outer_state()

    initial = _FakeResp(accepted=True, policy_id="policy-initial")

    def fake_set(*args, **kwargs):
        # First invocation succeeds (session entry); second blows up.
        if not fake_set.entered:
            fake_set.entered = True
            return initial
        raise ConnectionError("simulated network failure")

    fake_set.entered = False

    with patch("sasy.policy.api.set_session_policy", side_effect=fake_set):
        with pytest.raises(ConnectionError, match="simulated network failure"):
            with session(session_id="rpc-fail-test", policy="// dummy") as h:
                assert get_current_policy_id() == "policy-initial"
                h.set_policy("// also dummy")  # raises ConnectionError

    assert _outer_state() == before


def test_successful_set_policy_unwinds_on_block_exit():
    """When ``set_policy`` succeeds, the policy id flips mid-block —
    but on normal block exit the outer scope still has to be
    restored to its pre-block state, even though the in-block
    ``_policy_id_var.set(new_id)`` didn't save a token. The session
    block's ``reset(pid_token)`` is what unwinds it."""
    before = _outer_state()

    initial = _FakeResp(accepted=True, policy_id="policy-initial")
    swap = _FakeResp(accepted=True, policy_id="policy-swapped")

    with patch(
        "sasy.policy.api.set_session_policy",
        side_effect=[initial, swap],
    ):
        with session(session_id="success-test", policy="// dummy") as h:
            assert get_current_policy_id() == "policy-initial"
            h.set_policy("// dummy v2")
            # Mid-block: the swap is visible in the ContextVar.
            assert get_current_policy_id() == "policy-swapped"
            assert h._policy_id == "policy-swapped"

    # Block exit: both vars back to outer scope, despite the
    # mid-block flip on _policy_id_var.
    assert _outer_state() == before


def test_session_entry_failure_restores_context():
    """If ``policy=`` on entry fails (server rejects the initial
    upload), the block must never enter — and the ContextVars must
    be untouched, since the entry raises before ``_session_id_var.set``
    runs."""
    before = _outer_state()
    rejection = _FakeResp(accepted=False, message="bad", error_output="")

    with patch(
        "sasy.policy.api.set_session_policy",
        return_value=rejection,
    ):
        with pytest.raises(RuntimeError, match="bad"):
            with session(session_id="entry-fail", policy="// dummy"):
                pytest.fail("should not enter block on entry rejection")

    assert _outer_state() == before
