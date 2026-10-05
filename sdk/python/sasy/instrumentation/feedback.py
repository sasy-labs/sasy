"""
Authorization feedback accumulation for responder-scoped tracking.

This module provides a ContextVar-based mechanism to accumulate authorization
feedback (denials) within a responder scope. The accumulated feedback can be
surfaced to agents at responder boundaries.

Key design goals:
- Per-responder scope: feedback is isolated to a single LLM response generation
- Non-blocking: denied requests return informative 403 responses
- Avoid infinite loops: feedback is appended AFTER responder completes, not during
"""


from contextvars import ContextVar
from dataclasses import dataclass, field

from sasy.capture import capture_logger

from .config import get_config

logger = capture_logger(__name__)

@dataclass
class AuthorizationFeedback:
    """Single piece of authorization feedback for a denied request or tool call."""

    # Explanation
    message: str = ""
    suggestions: list[str] = field(default_factory=list)

    # HTTP request details (for HTTP denials)
    request_url: str | None = None
    request_method: str = "GET"

    # Tool call details (for tool call denials)
    fn_name: str | None = None
    fn_args: str | None = None


@dataclass
class FeedbackAccumulator:
    """
    Accumulates authorization feedback within a responder scope.

    This is used to collect all authorization decisions made during
    a single LLM response generation, so they can be surfaced to the
    agent after the response completes.
    """

    items: list[AuthorizationFeedback] = field(default_factory=list)

    def add(self, feedback: AuthorizationFeedback) -> None:
        """Add a feedback item."""
        self.items.append(feedback)

    def add_denial(
        self,
        url: str,
        message: str,
        method: str = "GET",
        suggestions: list[str] | None = None,
    ) -> None:
        """Add a denial feedback item."""
        self.items.append(
            AuthorizationFeedback(
                request_url=url,
                request_method=method,
                message=message,
                suggestions=suggestions or [],
            )
        )

    def add_tool_denial(
        self,
        fn_name: str,
        message: str,
        fn_args: str = "",
        suggestions: list[str] | None = None,
    ) -> None:
        """Add a tool call denial feedback item."""
        self.items.append(
            AuthorizationFeedback(
                fn_name=fn_name,
                fn_args=fn_args,
                message=message,
                suggestions=suggestions or [],
            )
        )

    def has_denials(self) -> bool:
        """Check if any denial feedback has been recorded."""
        return len(self.items) > 0

    def get_denials(self) -> list[AuthorizationFeedback]:
        """Return all denial items."""
        return list(self.items)

    def format_summary(self) -> str:
        """
        Format accumulated feedback as a human-readable summary.

        Returns:
            Formatted summary string, or empty string if no denials
        """
        if not self.items:
            return ""

        lines: list[str] = ["Authorization issues encountered:"]

        for f in self.items:
            if f.fn_name:
                line = f"  - {f.fn_name}(): {f.message}"
            else:
                line = f"  - {f.request_method} {f.request_url}: {f.message}"
            lines.append(line)
            for suggestion in f.suggestions:
                lines.append(f"    Suggestion: {suggestion}")

        return "\n".join(lines)

    def format_for_llm(self) -> str:
        """
        Format feedback for inclusion in LLM conversation.

        This formats denials in a way that informs the LLM without
        encouraging retry attempts for unauthorized actions.

        Returns:
            Formatted message for LLM, or empty string if no denials
        """
        if not self.items:
            return ""

        lines = [
            "[AUTHORIZATION BLOCKED - ACTION REQUIRED]",
            "",
        ]

        for f in self.items:
            if f.fn_name:
                # Tool call denial
                lines.append(f"Blocked: {f.fn_name}()")
            else:
                # HTTP request denial
                lines.append(f"Blocked: {f.request_method} {f.request_url}")
            lines.append(f"Reason: {f.message}")
            if f.suggestions:
                if len(f.suggestions) == 1:
                    lines.append(f"Required action: {f.suggestions[0]}")
                else:
                    lines.append("Required actions:")
                    for suggestion in f.suggestions:
                        lines.append(f"  - {suggestion}")
            lines.append("")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all accumulated feedback."""
        self.items.clear()


# Context variable for current feedback accumulator
# This propagates through nested async/sync calls within a responder scope
_current_feedback: ContextVar[FeedbackAccumulator | None] = ContextVar(
    "current_feedback", default=None
)


def get_current_feedback() -> FeedbackAccumulator | None:
    """
    Get the current feedback accumulator from context.

    Returns:
        The current FeedbackAccumulator if in a responder scope, None otherwise
    """
    return _current_feedback.get()


def set_current_feedback(accumulator: FeedbackAccumulator | None) -> object:
    """
    Set the current feedback accumulator in context.

    Args:
        accumulator: The accumulator to set, or None to clear

    Returns:
        Token that can be used to reset the context
    """
    return _current_feedback.set(accumulator)


def reset_current_feedback(token: object) -> None:
    """
    Reset the feedback context to a previous state.

    Args:
        token: Token returned from set_current_feedback
    """
    _current_feedback.reset(token)  # type: ignore


def add_feedback(feedback: AuthorizationFeedback) -> None:
    """
    Add feedback to the current accumulator if one exists.

    This is a convenience function for instrumentation code to add
    feedback without needing to check for accumulator existence.

    Args:
        feedback: The feedback to add
    """
    accumulator = get_current_feedback()
    if accumulator is not None:
        accumulator.add(feedback)


def add_denial(
    url: str,
    message: str,
    method: str = "GET",
    suggestions: list[str] | None = None,
) -> None:
    """
    Add a denial feedback item to the current accumulator.

    Args:
        url: The request URL that was denied
        message: Explanation of why the request was denied
        method: HTTP method (default: GET)
        suggestions: Optional list of suggestions for resolving the issue
    """
    accumulator = get_current_feedback()
    if accumulator is not None:
        accumulator.add_denial(url, message, method, suggestions)


def add_tool_denial(
    fn_name: str,
    message: str,
    fn_args: str = "",
    suggestions: list[str] | None = None,
) -> None:
    """
    Add a tool call denial feedback item to the current accumulator.

    Args:
        fn_name: The tool function name that was denied
        message: Explanation of why the tool call was denied
        fn_args: The tool arguments (for context)
        suggestions: Optional list of suggestions for resolving the issue
    """
    accumulator = get_current_feedback()
    if accumulator is not None:
        accumulator.add_tool_denial(
            fn_name=fn_name,
            message=message,
            fn_args=fn_args,
            suggestions=suggestions,
        )


class FeedbackScope:
    """
    Context manager for feedback accumulation within a responder scope.

    Creates a FeedbackAccumulator, sets it as the current feedback context,
    and handles cleanup on exit. Feedback processing is deferred to allow
    access to the response output.
    """

    def __init__(self, agent_name: str, responder_type: str):
        self.agent_name = agent_name
        self.responder_type = responder_type
        self.accumulator: FeedbackAccumulator | None = None
        self.token: object | None = None

    def __enter__(self) -> "FeedbackScope":
        self.accumulator = FeedbackAccumulator()
        self.token = set_current_feedback(self.accumulator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Reset the feedback context (processing is done separately via process())
        if self.token is not None:
            reset_current_feedback(self.token)

    def process(self, output, agent) -> None:
        """
        Process accumulated feedback after the responder completes.

        Args:
            output: The message output from the responder (may be None)
            agent: The agent instance that produced the output
        """
        if self.accumulator is None:
            return

        if not self.accumulator.items:
            return

        # Get feedback config
        config = get_config()

        # Log denials if configured
        if config.log_denials:
            summary = self.accumulator.format_summary()
            logger.warning(
                f"Authorization denials in {self.agent_name}.{self.responder_type}:\n{summary}"
            )

        # Call the callback with accumulator, output, and agent
        if config.feedback_callback is not None:
            try:
                config.feedback_callback(self.accumulator, output, agent)
            except Exception as e:
                logger.error(f"Feedback callback error: {e}")
