"""Rule-specific denial feedback through an isolated, authenticated engine."""

import uuid

import pytest
from sasy.proto import policy_engine_pb2 as pe
from sasy.proto import policy_engine_pb2_grpc as pe_grpc

pytestmark = pytest.mark.integration


@pytest.fixture
def feedback_rpc(engine):
    with engine.channel() as channel:
        yield pe_grpc.PolicyEngineStub(channel), [("x-api-key", engine.tenant_a_key)]


def _pin(rpc, policy):
    stub, metadata = rpc
    session_id = "feedback-" + uuid.uuid4().hex
    response = stub.SetPolicy(
        pe.SetPolicyRequest(
            policy_source=policy, backend="souffle",
            scope=pe.PolicyScope(session=pe.SessionTarget(session_id=session_id)),
        ), metadata=metadata, timeout=300,
    )
    assert response.accepted, response.error_output
    return session_id


def _check(rpc, session_id, fn_name, args):
    stub, metadata = rpc
    response = stub.CheckAuthorization(
        pe.AuthorizationRequest(
            session_id=session_id,
            actions=[pe.Action(tool_call=pe.ToolCallAction(fn_name=fn_name, args=args))],
        ), metadata=metadata, timeout=30,
    )
    assert len(response.results) == 1
    return response.results[0]


# Policy with three distinct Unauthorized rules, each gated on
# tool args via @json_get_str.  The policy compiler rewrites these into
# DenialReason(idx, reason, suggestion) rules.
FEEDBACK_POLICY = """\
// Default: authorize all actions
IsAuthorized(idx) :- Actions(idx, a).

// --- Three distinct deny rules, each with unique feedback ---

// @deny_message: Blocked: action targets a restricted region
// @suggestion: Use an approved region (us-east-1, eu-west-1)
Unauthorized(idx) :-
    Actions(idx, a),
    a = $CallTool(_, args),
    @json_get_str(args, "region") = region,
    region != "",
    region != "us-east-1",
    region != "eu-west-1".

// @deny_message: Blocked: admin operations require elevated role
// @suggestion: Request the admin role from your administrator
Unauthorized(idx) :-
    Actions(idx, a),
    a = $CallTool(fn_name, _),
    fn_name = "admin_delete",
    !HasRole("admin").

// @deny_message: Blocked: dangerous tool is forbidden
// @suggestion: Use a safe alternative tool instead
Unauthorized(idx) :-
    Actions(idx, a),
    a = $CallTool("dangerous_tool", _).

// Deny all denied tool calls (return error)
DenyUnauthorized(idx) :- Actions(idx, a), a = $CallTool(_, _).
"""


class TestDenialReasonFeedback:
    """Only fired deny rules contribute feedback to this session."""

    @pytest.fixture(autouse=True)
    def _feedback_session(self, feedback_rpc):
        self.rpc = feedback_rpc
        self.session_id = _pin(feedback_rpc, FEEDBACK_POLICY)

    def test_allowed_tool_no_feedback(self):
        """A tool call that matches no deny rules should be authorized."""
        result = _check(self.rpc, self.session_id, fn_name="read_data", args='{"region": "us-east-1"}')
        assert result.authorized, (
            f"Expected authorized, got denial_trace: {result.trace}"
        )

    def test_restricted_region_feedback(self):
        """Calling with a restricted region should return the region message."""
        result = _check(self.rpc, self.session_id, fn_name="deploy_service", args='{"region": "ap-south-1"}')
        assert not result.authorized
        trace = result.trace

        # Should have at least one denylist reason (the region rule)
        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        assert len(denylist_reasons) >= 1, (
            f"Expected DENYLISTED reason, got: {[r for r in trace.reasons]}"
        )
        assert any("restricted region" in r.details for r in denylist_reasons), (
            f"Expected 'restricted region' in details, got: "
            f"{[r.details for r in denylist_reasons]}"
        )

        # Suggestion should mention approved regions
        assert any("approved region" in s for s in trace.suggested_fixes), (
            f"Expected 'approved region' in suggestions, got: {trace.suggested_fixes}"
        )

        # Should NOT contain admin or dangerous tool messages
        all_details = " ".join(r.details for r in denylist_reasons)
        assert "admin" not in all_details.lower()
        assert "dangerous tool" not in all_details.lower()

    def test_admin_delete_feedback(self):
        """admin_delete without admin role should return the admin message."""
        result = _check(self.rpc, self.session_id, fn_name="admin_delete", args='{"target": "some-resource"}')
        assert not result.authorized
        trace = result.trace

        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        assert any("elevated role" in r.details for r in denylist_reasons), (
            f"Expected 'elevated role' in details, got: "
            f"{[r.details for r in denylist_reasons]}"
        )
        assert any("admin role" in s for s in trace.suggested_fixes), (
            f"Expected 'admin role' in suggestions, got: {trace.suggested_fixes}"
        )

        # Should NOT contain region or dangerous tool messages
        all_details = " ".join(r.details for r in denylist_reasons)
        assert "restricted region" not in all_details
        assert "dangerous tool" not in all_details

    def test_dangerous_tool_feedback(self):
        """dangerous_tool should return the dangerous tool message."""
        result = _check(self.rpc, self.session_id, fn_name="dangerous_tool", args='{}')
        assert not result.authorized
        trace = result.trace

        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        assert any("dangerous tool" in r.details for r in denylist_reasons), (
            f"Expected 'dangerous tool' in details, got: "
            f"{[r.details for r in denylist_reasons]}"
        )
        assert any("safe alternative" in s for s in trace.suggested_fixes), (
            f"Expected 'safe alternative' in suggestions, got: {trace.suggested_fixes}"
        )

        # Should NOT contain region or admin messages
        all_details = " ".join(r.details for r in denylist_reasons)
        assert "restricted region" not in all_details
        assert "elevated role" not in all_details

    def test_safe_tool_allowed(self):
        """A normal tool should be authorized (matches default allow-all)."""
        result = _check(self.rpc, self.session_id, fn_name="safe_tool", args='{}')
        assert result.authorized, (
            f"Expected authorized, got denial_trace: {result.trace}"
        )

    def test_multiple_rules_fire(self):
        """admin_delete in a restricted region should return BOTH reasons."""
        result = _check(self.rpc, self.session_id, fn_name="admin_delete", args='{"region": "ap-south-1"}')
        assert not result.authorized
        trace = result.trace

        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        details = [r.details for r in denylist_reasons]

        # Both the region rule AND the admin rule should fire
        assert any("restricted region" in d for d in details), (
            f"Expected region denial, got: {details}"
        )
        assert any("elevated role" in d for d in details), (
            f"Expected admin denial, got: {details}"
        )
        # Should have suggestions from both rules
        assert len(trace.suggested_fixes) >= 2, (
            f"Expected >=2 suggestions, got: {trace.suggested_fixes}"
        )


# Policy that tests IsAuthorized filtering behavior.
# - Only specific tools are allowlisted (not a blanket allow-all)
# - One tool is both allowlisted AND denylisted (denylist wins)
# - IsAuthorized rules have @deny_message annotations with tool patterns
#   so the metadata store can filter NOT_ALLOWLISTED feedback
ALLOWLIST_POLICY = """\
// --- Selective allowlist (no default allow-all) ---

// @deny_message: search_flights requires the booking role
// @suggestion: Request the booking role
// @tool_pattern: search_flights
IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("search_flights", _), HasRole("booking").

// @deny_message: read_data is available to all authenticated users
// @tool_pattern: read_data
IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("read_data", _).

// @deny_message: Only admins may use admin_delete
// @suggestion: Contact your administrator for access
// @tool_pattern: admin_delete
IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("admin_delete", _), HasRole("admin").

// restricted_read: unconditionally allowlisted AND unconditionally denylisted
// (denylist must override)
// @tool_pattern: restricted_read
IsAuthorized(idx) :- Actions(idx, a), a = $CallTool("restricted_read", _).

// @deny_message: restricted_read is temporarily suspended
// @suggestion: Use read_data instead
Unauthorized(idx) :-
    Actions(idx, a),
    a = $CallTool("restricted_read", _).

// --- Denylist: admin_delete is always blocked without admin role ---
// @deny_message: admin_delete is restricted to admins
// @suggestion: You must have the admin role
Unauthorized(idx) :-
    Actions(idx, a),
    a = $CallTool("admin_delete", _),
    !HasRole("admin").

// Deny unauthorized tool calls
DenyUnauthorized(idx) :- Actions(idx, a), a = $CallTool(_, _).
"""


class TestAllowlistFeedback:
    """Verify IsAuthorized filtering in denial traces."""

    @pytest.fixture(autouse=True)
    def _allowlist_session(self, feedback_rpc):
        self.rpc = feedback_rpc
        self.session_id = _pin(feedback_rpc, ALLOWLIST_POLICY)

    def test_allowlisted_tool_authorized(self):
        """read_data is allowlisted for all — should be authorized, no trace."""
        result = _check(self.rpc, self.session_id, fn_name="read_data", args='{}')
        assert result.authorized, (
            f"read_data should be authorized, got: {result.trace}"
        )

    def test_not_allowlisted_tool_denied(self):
        """unknown_tool is not on any allowlist — should be denied."""
        result = _check(self.rpc, self.session_id, fn_name="unknown_tool", args='{}')
        assert not result.authorized
        trace = result.trace

        not_allowlisted = [
            r for r in trace.reasons
            if r.reason_type == pe.NOT_ALLOWLISTED
        ]
        assert len(not_allowlisted) >= 1, (
            f"Expected NOT_ALLOWLISTED reason, got: "
            f"{[(r.reason_type, r.details) for r in trace.reasons]}"
        )

    def test_denylisted_overrides_allowlisted_no_allowlist_reason(self):
        """restricted_read is both unconditionally allowlisted and denylisted.

        Denylist overrides allowlist. The trace should show the DENYLISTED
        reason but NOT a NOT_ALLOWLISTED reason (because IsAuthorized did fire).
        """
        result = _check(self.rpc, self.session_id, fn_name="restricted_read", args='{}')
        assert not result.authorized
        trace = result.trace

        # Should have DENYLISTED reason from the Unauthorized rule
        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        assert any("temporarily suspended" in r.details for r in denylist_reasons), (
            f"Expected DENYLISTED reason with 'temporarily suspended', got: "
            f"{[r.details for r in denylist_reasons]}"
        )

        # Should NOT have NOT_ALLOWLISTED reason (IsAuthorized fires for restricted_read)
        not_allowlisted = [
            r for r in trace.reasons
            if r.reason_type == pe.NOT_ALLOWLISTED
        ]
        assert len(not_allowlisted) == 0, (
            f"restricted_read is allowlisted — should not have NOT_ALLOWLISTED "
            f"reason, but got: {[r.details for r in not_allowlisted]}"
        )

        # Suggestions should only come from the DenialReason rule
        assert any("read_data instead" in s for s in trace.suggested_fixes), (
            f"Expected DenialReason suggestion, got: {trace.suggested_fixes}"
        )

    def test_not_allowlisted_feedback_filtered_by_tool(self):
        """NOT_ALLOWLISTED feedback should only reference relevant rules.

        unknown_tool doesn't match any IsAuthorized rule's @tool_pattern,
        so the feedback should be generic (no tool-specific suggestions).
        In particular, it should NOT contain suggestions from the
        search_flights or admin_delete rules.
        """
        result = _check(self.rpc, self.session_id, fn_name="unknown_tool", args='{}')
        assert not result.authorized
        trace = result.trace

        # Should NOT contain tool-specific suggestions for other tools
        all_fixes = " ".join(trace.suggested_fixes)
        assert "booking role" not in all_fixes.lower(), (
            f"search_flights suggestion leaked to unknown_tool: {trace.suggested_fixes}"
        )
        assert "administrator for access" not in all_fixes.lower(), (
            f"admin_delete suggestion leaked to unknown_tool: {trace.suggested_fixes}"
        )

    def test_both_denylisted_and_not_allowlisted(self):
        """admin_delete without admin role is both denylisted AND not allowlisted.

        The IsAuthorized rule requires HasRole("admin"), which the test user
        lacks — so IsAuthorized doesn't fire. The Unauthorized rule also
        requires !HasRole("admin"), which IS true — so it fires.

        The trace should contain BOTH:
        - DENYLISTED with the DenialReason message
        - NOT_ALLOWLISTED with the IsAuthorized metadata message
        """
        result = _check(self.rpc, self.session_id, fn_name="admin_delete", args='{}')
        assert not result.authorized
        trace = result.trace

        denylist_reasons = [
            r for r in trace.reasons
            if r.reason_type == pe.DENYLISTED
        ]
        not_allowlisted = [
            r for r in trace.reasons
            if r.reason_type == pe.NOT_ALLOWLISTED
        ]

        # DENYLISTED reason from the Unauthorized/DenialReason rule
        assert any("restricted to admins" in r.details for r in denylist_reasons), (
            f"Expected DENYLISTED reason with 'restricted to admins', got: "
            f"{[r.details for r in denylist_reasons]}"
        )

        # NOT_ALLOWLISTED reason from the IsAuthorized metadata
        assert any("admin" in r.details.lower() for r in not_allowlisted), (
            f"Expected NOT_ALLOWLISTED reason mentioning admin, got: "
            f"{[r.details for r in not_allowlisted]}"
        )

        # Both categories of suggestions should be present
        all_fixes = " ".join(trace.suggested_fixes)
        assert "admin role" in all_fixes.lower(), (
            f"Expected DenialReason suggestion about admin role, got: "
            f"{trace.suggested_fixes}"
        )
