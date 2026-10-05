"""Policy enforcement test utilities.

Provides tools for verifying policy enforcement:

1. ``evaluate_policy`` — Evaluate a policy decision
   directly, without an LLM agent. Constructs a
   minimal graph with tool results and checks
   authorization::

       from sasy.instrumentation.testing import (
           evaluate_policy,
       )

       result = evaluate_policy(
           tool_name="cancel_order",
           tool_args={"order_id": "123"},
           tool_results={
               "get_order": {"status": "shipped"},
           },
       )
       assert not result.authorized

2. ``policy_test_context`` — Context manager for
   collecting authorization feedback during code
   execution.

3. ``setup_test_graph`` — Record user messages into the
   graph for policy testing.

4. ``check_tool_call`` — Low-level tool authorization
   check via the reference monitor.
"""


import socket
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from sasy.capture import capture_logger
from sasy.proto.observability_pb2 import (
    Edge,
    Event,
    Role,
    Tool,
)
from sasy.reference_monitor import check_tool_call as _check_tool_call

from .feedback import (
    AuthorizationFeedback,
    FeedbackAccumulator,
    reset_current_feedback,
    set_current_feedback,
)

logger = capture_logger(__name__)


@dataclass
class PolicyTestResult:
    """Result of a policy enforcement test."""

    denials: list[AuthorizationFeedback] = field(default_factory=list)

    def assert_denied(
        self,
        action_name: str,
        message_contains: str | None = None,
    ) -> None:
        """Assert an action was denied.

        Args:
            action_name: Name of the action (tool fn_name or URL pattern)
            message_contains: Optional substring to check in denial message

        Raises:
            AssertionError: If the action was not denied or message doesn't match
        """
        matching_denials = [
            d
            for d in self.denials
            if (d.fn_name and action_name in d.fn_name)
            or (d.request_url and action_name in d.request_url)
        ]

        if not matching_denials:
            denied_actions = [
                d.fn_name or d.request_url or "unknown" for d in self.denials
            ]
            raise AssertionError(
                f"Expected '{action_name}' to be denied, but it was not. "
                f"Denials: {denied_actions}"
            )

        if message_contains:
            for denial in matching_denials:
                if message_contains.lower() in denial.message.lower():
                    return
            messages = [d.message for d in matching_denials]
            raise AssertionError(
                f"Denial for '{action_name}' found, but message does not contain "
                f"'{message_contains}'. Messages: {messages}"
            )

    def assert_allowed(self, action_name: str) -> None:
        """Assert an action was allowed (not in denials).

        Args:
            action_name: Name of the action (tool fn_name or URL pattern)

        Raises:
            AssertionError: If the action was denied
        """
        for denial in self.denials:
            if (denial.fn_name and action_name in denial.fn_name) or (
                denial.request_url and action_name in denial.request_url
            ):
                raise AssertionError(
                    f"Expected '{action_name}' to be allowed, but it was denied: "
                    f"{denial.message}"
                )


@contextmanager
def policy_test_context() -> Generator[PolicyTestResult, None, None]:
    """Context manager for policy enforcement tests.

    Creates a FeedbackAccumulator, sets it as the current feedback context,
    and yields a PolicyTestResult that collects all feedback during the block.

    Usage:
        with policy_test_context() as result:
            # Execute code that triggers policy checks
            client.post("https://api.example.com/data")

        result.assert_denied("api.example.com", message_contains="unauthorized")
    """
    accumulator = FeedbackAccumulator()
    token = set_current_feedback(accumulator)

    result = PolicyTestResult()

    try:
        yield result
    finally:
        result.denials.extend(accumulator.items)
        reset_current_feedback(token)


def check_tool_call(
    fn_name: str,
    args: str,
    input_node_ids: list[str] | None = None,
    max_retries: int = 3,
):
    """Check authorization for a tool call via the reference monitor.

    Wrapper around rm_client.check_tool_call for test convenience.

    Args:
        fn_name: The tool function name
        args: The tool arguments (typically JSON string)
        input_node_ids: Node IDs from observability context
        max_retries: Maximum number of retry attempts for transient failures

    Returns:
        ToolCallResponse with authorization decision
    """
    return _check_tool_call(
        fn_name=fn_name,
        args=args,
        input_node_ids=input_node_ids,
        max_retries=max_retries,
    )


def setup_test_graph(
    user_message: str,
    agent_name: str = "LLMAgent",
    sync_delay: float = 2.0,
) -> list[str]:
    """Set up a test graph with a user message for policy testing.

    Creates a message node in the observability graph so the policy engine
    can query it via Depends() and SentMessage() relations.

    Args:
        user_message: The user message text (e.g., "I accidentally double-booked")
        agent_name: The agent name to use (default ``"LLMAgent"``; use the
            agent name your policy matches on)
        sync_delay: Seconds to wait for policy engine to sync graph updates.
            The policy engine syncs via streaming, so we need to allow time
            for updates to propagate. Default: 2.0 seconds.

    Returns:
        List of node IDs that can be passed as input_node_ids to check_tool_call

    Raises:
        RuntimeError: If observability services are not running
    """
    try:
        from sasy.observability.api import record_events_with_dependencies
    except ImportError:
        raise RuntimeError(
            "sasy.observability not available. "
            "Install with: pip install -e sdk/python"
        )

    # Pre-construct UUIDs so we can reference them in edges
    user_id = str(uuid.uuid4())
    current_id = str(uuid.uuid4())

    user_event = Event(
        text=user_message,
        agent=agent_name,
        role=Role.USER,
        id=user_id,
    )

    # "current" event simulates the tool call context responding to the user message
    current_event = Event(
        text="[Tool call context]",
        agent=agent_name,
        role=Role.AGENT,
        id=current_id,
    )

    # Edge(source, dest) means "dest depends on source"
    edge = Edge(
        source=user_id,
        destination=current_id,
        proximal=True,
    )

    # Single atomic call registers both events and the dependency
    event_ids = record_events_with_dependencies(
        [user_event, current_event], [edge]
    )

    if len(event_ids) < 2:
        raise RuntimeError("Failed to record test events in observability graph")

    # Wait for policy engine to sync the graph updates
    # The policy engine streams updates from the observability server,
    # so there's a delay before new nodes/edges are available for policy checks
    if sync_delay > 0:
        logger.debug(f"Waiting {sync_delay}s for policy engine to sync graph updates")
        time.sleep(sync_delay)

    # Return the current event ID as input_node_ids
    # The policy engine will use this to find the dependent user message
    return [event_ids[1]]


def is_service_running(host: str = "localhost", port: int = 10089) -> bool:
    """Check if a service is running on the given host:port.

    Args:
        host: Hostname to check
        port: Port to check

    Returns:
        True if the service is accepting connections, False otherwise
    """
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def check_services_running(
    rm_port: int = 10089,
    obs_port: int = 10089,
) -> tuple[bool, str]:
    """Check if required services are running.

    Args:
        rm_port: Reference monitor port (default: 10089)
        obs_port: Observability server port (default: 10089)

    Returns:
        Tuple of (all_running, error_message)
    """
    missing = []

    if not is_service_running(port=rm_port):
        missing.append(f"Reference Monitor (port {rm_port})")

    if not is_service_running(port=obs_port):
        missing.append(f"Observability Server (port {obs_port})")

    if missing:
        return False, (
            f"Required services not running: {', '.join(missing)}. "
            "Start the configured SASY and application services before running this check."
        )

    return True, ""


# ── Direct policy evaluation ──────────────────────────


# DenialReasonType.ASK — the soft "ask the human first" denial. The engine
# returns authorized=False for it (same wire shape as a hard block) but tags
# every reason ASK, so a three-valued verdict (allow / ask / deny) is
# recoverable. Mirrors the wire enum in policy_engine.proto (ASK = 5); kept as a
# literal to match scenario_assert._ASK_REASON_TYPE (one documented contract).
_ASK_REASON_TYPE = 5


@dataclass
class PolicyResult:
    """Result of a direct policy evaluation.

    Attributes:
        authorized: Whether the action was allowed.
        reasons: Denial reasons (empty if authorized).
        verdict: The three-valued decision — ``"ALLOW"``, ``"ASK"`` (soft, the
            human is asked first), or ``"DENY"`` (hard block). ``authorized`` is
            ``True`` iff ``verdict == "ALLOW"``; ``ASK`` and ``DENY`` both carry
            ``authorized=False`` but differ in whether the denial is all-ASK.
    """

    authorized: bool
    reasons: list[str] = field(default_factory=list)
    verdict: str = ""


def evaluate_policy(
    tool_name: str,
    tool_args: dict[str, str] | None = None,
    tool_results: dict[str, dict[str, str]]
    | None = None,
    sync_delay: float = 2.0,
) -> PolicyResult:
    """Evaluate a policy decision without an LLM agent.

    Constructs a minimal dependency graph with tool
    result events, then checks whether the given tool
    call is authorized by the current policy.

    Args:
        tool_name: The tool being invoked (e.g.,
            "cancel_reservation").
        tool_args: Arguments to the tool call as a dict
            (e.g., ``{"reservation_id": "ABC"}``).
        tool_results: Prior tool results to include in
            the graph. Maps tool name to a dict of
            field-value pairs that the tool returned.
            Example::

                {
                    "get_order": {
                        "status": "shipped",
                        "priority": "high",
                    }
                }
        sync_delay: Seconds to wait for the policy
            engine to sync graph updates.

    Returns:
        A ``PolicyResult`` with the authorization
        decision and any denial reasons.

    Example::

        result = evaluate_policy(
            tool_name="cancel_order",
            tool_args={"order_id": "123"},
            tool_results={
                "get_order": {
                    "status": "shipped",
                    "tier": "basic",
                },
            },
        )
        assert not result.authorized
    """
    import json

    from sasy.observability.api import (
        record_events,
        record_events_with_dependencies,
    )

    event_ids: list[str] = []
    args_json = json.dumps(tool_args or {})

    # Record tool result events
    if tool_results:
        for tr_tool, tr_fields in tool_results.items():
            tr_eid = str(uuid.uuid4())
            tr_args = json.dumps(tr_fields)
            tr_event = Event(
                text=json.dumps(tr_fields),
                role=Role.AGENT,
                agent="LLMAgent",
                id=tr_eid,
                derived_from=Tool(
                    name=tr_tool,
                    arguments=tr_args,
                ),
            )
            if event_ids:
                edge = Edge(
                    source=event_ids[-1],
                    destination=tr_eid,
                    proximal=True,
                )
                record_events_with_dependencies(
                    [tr_event], [edge]
                )
            else:
                record_events([tr_event])
            event_ids.append(tr_eid)

    # Record tool call event (the action being checked)
    tc_eid = str(uuid.uuid4())
    tc_event = Event(
        text="",
        role=Role.LLM,
        agent="LLMAgent",
        id=tc_eid,
        tools=[
            Tool(name=tool_name, arguments=args_json)
        ],
    )
    if event_ids:
        edge = Edge(
            source=event_ids[-1],
            destination=tc_eid,
            proximal=True,
        )
        record_events_with_dependencies(
            [tc_event], [edge]
        )
    else:
        record_events([tc_event])
    event_ids.append(tc_eid)

    # Wait for policy engine to sync
    if sync_delay > 0:
        time.sleep(sync_delay)

    # Check authorization
    resp = _check_tool_call(
        fn_name=tool_name,
        args=args_json,
        input_node_ids=event_ids,
    )

    reasons: list[str] = []
    if resp.authorized:
        verdict = "ALLOW"
    else:
        rs = (
            getattr(
                getattr(resp, "denial_trace", None),
                "reasons",
                None,
            )
            or []
        )
        reasons = [r.details for r in rs if r.details]
        # All-ASK reasons => the soft "ask the human first" verdict, not a hard
        # block. Any non-ASK reason makes it a real DENY. Mirrors
        # scenario_assert._run_one so both paths recover the same third verdict.
        if rs and all(
            getattr(r, "reason_type", None) == _ASK_REASON_TYPE
            for r in rs
        ):
            verdict = "ASK"
        else:
            verdict = "DENY"

    return PolicyResult(
        authorized=resp.authorized,
        reasons=reasons,
        verdict=verdict,
    )
