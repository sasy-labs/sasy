"""Tests for instrumentation.feedback — pure Python, no external deps."""

from sasy.instrumentation.feedback import (
    AuthorizationFeedback,
    FeedbackAccumulator,
    FeedbackScope,
    add_denial,
    add_feedback,
    add_tool_denial,
    get_current_feedback,
    reset_current_feedback,
    set_current_feedback,
)


# ===================================================================
# AuthorizationFeedback
# ===================================================================


class TestAuthorizationFeedback:
    """Tests for the AuthorizationFeedback dataclass."""

    def test_defaults(self) -> None:
        fb = AuthorizationFeedback()
        assert fb.message == ""
        assert fb.suggestions == []
        assert fb.request_url is None
        assert fb.fn_name is None

    def test_custom_fields(self) -> None:
        fb = AuthorizationFeedback(
            message="blocked",
            request_url="https://api.example.com",
            request_method="POST",
            suggestions=["Fix it"],
        )
        assert fb.message == "blocked"
        assert fb.request_url == "https://api.example.com"
        assert fb.request_method == "POST"
        assert fb.suggestions == ["Fix it"]


# ===================================================================
# FeedbackAccumulator
# ===================================================================


class TestFeedbackAccumulator:
    """Tests for the FeedbackAccumulator."""

    def test_add_single_feedback(self) -> None:
        acc = FeedbackAccumulator()
        fb = AuthorizationFeedback(message="denied!")
        acc.add(fb)
        assert len(acc.items) == 1
        assert acc.items[0].message == "denied!"

    def test_add_multiple(self) -> None:
        acc = FeedbackAccumulator()
        acc.add(AuthorizationFeedback(message="a"))
        acc.add(AuthorizationFeedback(message="b"))
        acc.add(AuthorizationFeedback(message="c"))
        assert len(acc.items) == 3

    def test_clear(self) -> None:
        acc = FeedbackAccumulator()
        acc.add(AuthorizationFeedback(message="x"))
        assert len(acc.items) == 1
        acc.clear()
        assert len(acc.items) == 0

    def test_add_denial(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_denial(
            url="https://api.example.com/data",
            message="Not authorized",
            method="POST",
            suggestions=["Request access"],
        )
        assert len(acc.items) == 1
        fb = acc.items[0]
        assert fb.request_url == "https://api.example.com/data"
        assert fb.request_method == "POST"
        assert fb.suggestions == ["Request access"]

    def test_add_tool_denial(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_tool_denial(
            fn_name="execute_code",
            message="Blocked",
            fn_args='{"code": "rm -rf /"}',
            suggestions=["Don't do that"],
        )
        fb = acc.items[0]
        assert fb.fn_name == "execute_code"
        assert fb.fn_args == '{"code": "rm -rf /"}'

    def test_get_denials(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_denial("u1", "denied 1")
        acc.add_denial("u3", "denied 2")

        denials = acc.get_denials()
        assert len(denials) == 2
        assert denials[0].request_url == "u1"
        assert denials[1].request_url == "u3"

    def test_has_denials(self) -> None:
        acc = FeedbackAccumulator()
        assert acc.has_denials() is False
        acc.add_denial("u", "msg")
        assert acc.has_denials() is True

    def test_format_summary_denials(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_denial(
            "https://api.example.com",
            "Blocked",
            method="POST",
            suggestions=["Fix it"],
        )
        summary = acc.format_summary()
        assert "Authorization issues" in summary
        assert "POST" in summary
        assert "Blocked" in summary
        assert "Fix it" in summary

    def test_format_summary_empty(self) -> None:
        acc = FeedbackAccumulator()
        assert acc.format_summary() == ""

    def test_format_for_llm_with_denials(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_denial(
            "https://api.example.com",
            "Not allowed",
            suggestions=["Ask admin"],
        )
        result = acc.format_for_llm()
        assert "AUTHORIZATION BLOCKED" in result
        assert "Not allowed" in result
        assert "Ask admin" in result

    def test_format_for_llm_no_denials(self) -> None:
        acc = FeedbackAccumulator()
        assert acc.format_for_llm() == ""

    def test_format_for_llm_tool_denial(self) -> None:
        acc = FeedbackAccumulator()
        acc.add_tool_denial("bad_fn", "Nope")
        result = acc.format_for_llm()
        assert "bad_fn()" in result


# ===================================================================
# Context variable functions
# ===================================================================


class TestContextVars:
    """Tests for context variable management."""

    def test_set_get_roundtrip(self) -> None:
        acc = FeedbackAccumulator()
        token = set_current_feedback(acc)
        try:
            assert get_current_feedback() is acc
        finally:
            reset_current_feedback(token)

    def test_default_is_none(self) -> None:
        # Should be None when not set (or after reset)
        assert get_current_feedback() is None

    def test_reset_restores_previous(self) -> None:
        outer = FeedbackAccumulator()
        inner = FeedbackAccumulator()

        t1 = set_current_feedback(outer)
        t2 = set_current_feedback(inner)

        assert get_current_feedback() is inner
        reset_current_feedback(t2)
        assert get_current_feedback() is outer
        reset_current_feedback(t1)

    def test_add_feedback_with_context(self) -> None:
        acc = FeedbackAccumulator()
        token = set_current_feedback(acc)
        try:
            add_feedback(
                AuthorizationFeedback(message="test")
            )
            assert len(acc.items) == 1
        finally:
            reset_current_feedback(token)

    def test_add_feedback_without_context(self) -> None:
        """add_feedback is a no-op when no accumulator is set."""
        add_feedback(AuthorizationFeedback(message="ignored"))
        # Should not raise

    def test_add_denial_convenience(self) -> None:
        acc = FeedbackAccumulator()
        token = set_current_feedback(acc)
        try:
            add_denial("url", "msg")
            assert len(acc.items) == 1
        finally:
            reset_current_feedback(token)

    def test_add_tool_denial_convenience(self) -> None:
        acc = FeedbackAccumulator()
        token = set_current_feedback(acc)
        try:
            add_tool_denial("fn", "msg")
            assert len(acc.items) == 1
        finally:
            reset_current_feedback(token)


# ===================================================================
# FeedbackScope
# ===================================================================


class TestFeedbackScope:
    """Tests for the FeedbackScope context manager."""

    def test_scope_sets_and_resets_context(self) -> None:
        assert get_current_feedback() is None
        with FeedbackScope("agent", "responder") as scope:
            assert get_current_feedback() is scope.accumulator
            assert scope.accumulator is not None
        assert get_current_feedback() is None

    def test_scope_accumulates_feedback(self) -> None:
        with FeedbackScope("agent", "responder") as scope:
            add_denial("url", "blocked")
            assert scope.accumulator is not None
            assert len(scope.accumulator.items) == 1

    def test_scope_nested(self) -> None:
        with FeedbackScope("outer", "r1") as outer:
            add_denial("u1", "m1")
            with FeedbackScope("inner", "r2") as inner:
                add_denial("u2", "m2")
                assert (
                    get_current_feedback() is inner.accumulator
                )
            assert get_current_feedback() is outer.accumulator
            assert len(outer.accumulator.items) == 1
