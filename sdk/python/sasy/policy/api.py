"""
Policy engine client API — upload, validate, and query status.

Uses the shared ``sasy.config`` channel for all connections.
"""


import re
from collections.abc import Iterable
from pathlib import Path

from sasy.capture import capture_logger
from sasy.config import get_config, get_stub
from sasy.proto import policy_engine_pb2 as pe
from sasy.proto import policy_engine_pb2_grpc as pe_grpc

logger = capture_logger(__name__)


def _metadata() -> list[tuple[str, str]]:
    return get_config().get_metadata()


# ── File helpers ──────────────────────────────────────────

def resolve_includes(path: Path) -> str:
    """Recursively resolve ``#include`` directives to produce a self-contained policy.

    Args:
        path: Path to the root .dl policy file.

    Returns:
        Policy source with all includes inlined.

    Raises:
        FileNotFoundError: If the policy file or an included file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#include"):
            m = re.search(r'"([^"]+)"', stripped)
            if m:
                inc_path = path.parent / m.group(1)
                lines.append(resolve_includes(inc_path))
            continue
        lines.append(line)
    return "\n".join(lines)


def find_functors(policy_path: Path) -> str:
    """Auto-detect custom C++ functors for a policy file.

    Looks for:
    - ``functors.cpp`` in the same directory
    - ``<name>_functors.cpp`` alongside ``<name>_policy.dl``

    Args:
        policy_path: Path to the policy .dl file.

    Returns:
        Functor source code, or empty string if none found.
    """
    candidates = [
        policy_path.parent / "functors.cpp",
        policy_path.parent / (
            policy_path.stem
            .replace("_policy", "_functors")
            .replace("_policy", "_functors")
            + ".cpp"
        ),
    ]
    stem = policy_path.stem
    if "_policy" in stem:
        base = stem.split("_policy")[0]
        candidates.append(policy_path.parent / f"{base}_functors.cpp")

    for fp in candidates:
        if fp.exists():
            logger.info("Found functors: %s", fp.name)
            return fp.read_text()
    return ""


# ── gRPC API ─────────────────────────────────────────────

#: One static config fact: ``(rel, a, b)``, materialized by the evaluator as a
#: ``PolicyMetadata(rel, a, b)`` EDB tuple when a session's evaluator
#: initializes. ``b`` is optional and defaults to "".
PolicyMetadataFact = tuple[str, str] | tuple[str, str, str]


def _metadata_facts(
    facts: Iterable[PolicyMetadataFact] | None,
) -> list[pe.PolicyMetadataFact]:
    """Normalize ``(rel, a)`` / ``(rel, a, b)`` tuples to proto facts."""
    out = []
    for fact in facts or ():
        rel, a, *rest = fact
        out.append(pe.PolicyMetadataFact(rel=rel, a=a, b=rest[0] if rest else ""))
    return out


def _set_policy_raw(
    policy_source: str,
    scope: pe.PolicyScope,
    functor_source: str = "",
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Low-level wrapper around ``SetPolicy``. Most callers use the
    higher-level :func:`set_session_policy` /
    :func:`set_default_policy` / :func:`force_policy_update`
    helpers, which build the right :class:`PolicyScope` for them.
    """
    stub = get_stub(pe_grpc.PolicyEngineStub)  # type: ignore
    return stub.SetPolicy(
        pe.SetPolicyRequest(
            policy_source=policy_source,
            functor_source=functor_source,
            backend=backend,
            scope=scope,
            policy_metadata=_metadata_facts(policy_metadata),
        ),
        metadata=_metadata(),
        timeout=timeout,
    )


def set_session_policy(
    session_id: str | None,
    policy_source: str,
    *,
    functor_source: str = "",
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Bind ``session_id`` to ``policy_source``.

    Uploads the source as a variant (server-side dedup means
    repeated identical uploads are free), then binds the session.
    If the session was already bound to a different policy, the
    live evaluator is evicted and the next ``CheckAuthorization``
    re-spawns under the new policy — graph state is preserved.

    ``session_id`` of ``None`` (or ``""``) binds the per-tenant
    **global session** — the cross-session view that observes events
    from every session in the tenant. Useful for break-glass /
    tenant-admin tooling. (``SessionTarget`` carries a plain string,
    where the empty string is the global target, so ``None`` is sent
    as ``""`` here.)

    ``policy_metadata`` carries static configuration alongside the policy —
    ``(rel, a)`` or ``(rel, a, b)`` tuples the evaluator materializes as
    ``PolicyMetadata(rel, a, b)`` facts at init. It keeps deployment-specific
    values (allowlists, thresholds, capability flags, a whole tool taxonomy)
    out of the rules, so one `.dl` can serve several deployments and adding a
    value does not edit the security-relevant file.
    """
    return _set_policy_raw(
        policy_source,
        pe.PolicyScope(session=pe.SessionTarget(session_id=session_id or "")),
        functor_source=functor_source,
        backend=backend,
        timeout=timeout,
        policy_metadata=policy_metadata,
    )


def set_default_policy(
    policy_source: str,
    *,
    functor_source: str = "",
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Set the caller's tenant default policy.

    Future unpinned sessions adopt this policy on first traffic.
    Existing sessions keep their current bindings — gradual
    rollout. Combine with a canary :func:`set_session_policy`
    upload of the same source to validate before promoting; the
    server dedupes identical content so the canary upload doesn't
    cost an extra recompile.

    Requires the ``admin`` role.

    ``policy_metadata`` carries static configuration alongside the policy —
    ``(rel, a)`` or ``(rel, a, b)`` tuples the evaluator materializes as
    ``PolicyMetadata(rel, a, b)`` facts at init. It keeps deployment-specific
    values (allowlists, thresholds, capability flags, a whole tool taxonomy)
    out of the rules, so one `.dl` can serve several deployments and adding a
    value does not edit the security-relevant file.
    """
    return _set_policy_raw(
        policy_source,
        pe.PolicyScope(default=pe.DefaultTarget()),
        functor_source=functor_source,
        backend=backend,
        timeout=timeout,
        policy_metadata=policy_metadata,
    )


def force_policy_update(
    policy_source: str,
    *,
    functor_source: str = "",
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Set the tenant default *and* evict every live session in the
    tenant. Sessions re-bind to the new default on next
    traffic. Disruptive — emergency hot reload / break-glass
    rollback.

    Requires the ``admin`` role.

    ``policy_metadata`` carries static configuration alongside the policy —
    ``(rel, a)`` or ``(rel, a, b)`` tuples the evaluator materializes as
    ``PolicyMetadata(rel, a, b)`` facts at init. It keeps deployment-specific
    values (allowlists, thresholds, capability flags, a whole tool taxonomy)
    out of the rules, so one `.dl` can serve several deployments and adding a
    value does not edit the security-relevant file.
    """
    return _set_policy_raw(
        policy_source,
        pe.PolicyScope(force=pe.ForceTarget()),
        functor_source=functor_source,
        backend=backend,
        timeout=timeout,
        policy_metadata=policy_metadata,
    )


def update_policy_metadata(
    session_id: str | None,
    facts: Iterable[PolicyMetadataFact],
    timeout: float = 30,
) -> pe.UpdatePolicyMetadataResponse:
    """Append ``PolicyMetadata`` facts to a live session, mid-run.

    Distinct from the ``policy_metadata`` argument on the upload helpers, which
    is static configuration fixed when the policy is installed. This is for
    facts that only become true while the session is running — an approval a
    human just gave, a verdict a scanner just returned — so a rule can consult
    them without the policy being re-uploaded or the evaluator restarted.

    Facts accumulate: appending never removes what is already there.
    """
    stub = get_stub(pe_grpc.PolicyEngineStub)  # type: ignore
    return stub.UpdatePolicyMetadata(
        pe.UpdatePolicyMetadataRequest(
            session_id=session_id or "",
            facts=_metadata_facts(facts),
        ),
        metadata=_metadata(),
        timeout=timeout,
    )


def validate_policy(
    policy_source: str,
    run_analyses: bool = False,
    extra_reach_targets: list[str] | None = None,
) -> pe.ValidatePolicyResponse:
    """Validate policy source without applying it.

    Args:
        policy_source: Soufflé policy source to validate.
        run_analyses: When ``True``, also run static analyses
            (contradiction detection, redundancy, reachability)
            on the desugared source. Adds a few hundred
            milliseconds for typical policies.
        extra_reach_targets: Additional reachability targets in
            addition to the defaults (``IsAuthorized``,
            ``Unauthorized``, ``Authorized``, ``DenyUnauthorized``,
            ``AllowPassthrough``).

    Returns:
        ValidatePolicyResponse with ``valid``, ``error_output``,
        ``desugared_source``, and (when ``run_analyses`` is set)
        ``analyses`` carrying contradictions/redundancies/reachability.
    """
    stub = get_stub(pe_grpc.PolicyEngineStub)  # type: ignore
    return stub.ValidatePolicy(
        pe.ValidatePolicyRequest(
            policy_source=policy_source,
            run_analyses=run_analyses,
            extra_reach_targets=list(extra_reach_targets or []),
        ),
        metadata=_metadata(),
    )


def end_session(session_id: str, timeout: float = 5.0) -> bool:
    """End a session on the policy engine.

    Drops the per-session evaluator and clears its policy binding,
    session metadata and ownership. Recorded graph observations remain
    in the persistent store. A later request for the same id does not
    retain its former policy or approvals; bind the intended policy
    again, or use a fresh id for a new conversation.

    Best-effort: failures are logged but not raised, so closing
    a session doesn't break the agent's teardown path. All local scopes for
    this ID are invalidated even if the RPC fails. Returns
    ``True`` iff the server reports an evaluator was actually
    dropped.
    """
    if not session_id:
        return False
    from sasy.instrumentation.session import close_session_scopes, flush_session_scopes

    # Invalidate before dispatch: an in-flight authorization must not return an
    # actionable allow once teardown starts, including when the RPC fails.
    flush_session_scopes(session_id)
    close_session_scopes(session_id)
    try:
        stub = get_stub(pe_grpc.PolicyEngineStub)  # type: ignore
        resp = stub.EndSession(
            pe.EndSessionRequest(session_id=session_id),
            metadata=_metadata(),
            timeout=timeout,
        )
        return bool(resp.was_active)
    except Exception as e:
        logger.debug("EndSession(%s) failed: %s", session_id, e)
        return False
    finally:
        # Scopes opened concurrently with teardown cannot retain a binding the
        # server may just have dropped.
        close_session_scopes(session_id)


def get_evaluator_status() -> pe.EvaluatorStatusResponse:
    """Get the current evaluator backend status.

    Returns:
        EvaluatorStatusResponse with ``backend``, ``alive``, and ``policy_path``.
    """
    stub = get_stub(pe_grpc.PolicyEngineStub)  # type: ignore
    return stub.GetEvaluatorStatus(
        pe.EvaluatorStatusRequest(),
        metadata=_metadata(),
    )


# ── Convenience: apply a policy from a file ────────────────────

def set_default_policy_file(
    path: Path | str,
    *,
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Set the tenant default policy from a .dl file (resolves
    ``#include`` and auto-detects companion functors). Wraps
    :func:`set_default_policy`.
    """
    path = Path(path)
    policy_source = resolve_includes(path)
    functor_source = find_functors(path)
    return set_default_policy(
        policy_source,
        functor_source=functor_source,
        backend=backend,
        timeout=timeout,
        policy_metadata=policy_metadata,
    )


def force_policy_update_file(
    path: Path | str,
    *,
    backend: str = "",
    timeout: float = 600,
    policy_metadata: Iterable[PolicyMetadataFact] | None = None,
) -> pe.SetPolicyResponse:
    """Force-update the tenant default from a .dl file (evicts all
    live sessions in tenant). Wraps :func:`force_policy_update`.
    """
    path = Path(path)
    policy_source = resolve_includes(path)
    functor_source = find_functors(path)
    return force_policy_update(
        policy_source,
        functor_source=functor_source,
        backend=backend,
        timeout=timeout,
        policy_metadata=policy_metadata,
    )
