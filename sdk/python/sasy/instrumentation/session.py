"""Session-scoped policy state.

Adds a context-local ``session_id`` (and an optional
``policy_id``) that gets attached to every graph-recording RPC and
policy-evaluation request. The reference monitor partitions its
graph relations by ``(tenant, session_id)`` so concurrent agents on
the same SASY binary — even across tenants — do not see each
other's state.

Three layers, in priority order (highest wins):

1. ``with sasy.session("foo", policy=policy_source):`` — context manager override.
   ``with sasy.global_session():`` explicitly selects the per-tenant
   ``global`` partition, represented by the :data:`GLOBAL_SESSION`
   sentinel.
2. ``sasy.configure(process_global_session=True)`` explicitly enables a
   generated process default; ``configure_default_session(id)`` can name it.
3. Nothing declared — instrumentation is inactive and custom recording or
   authorization APIs raise. A process default requires explicit opt-in.

The active session is ``str | GLOBAL_SESSION``: a string names a
concrete session, and :data:`GLOBAL_SESSION` is the per-tenant global
view. "Nothing declared" is the conventional ``None`` (unset), which
is inactive unless a process default was explicitly configured.

Three labels travel with a recorded message, and they mean different
things:

* ``principal`` — who the engine authenticated. The server stamps it
  from the caller's credentials; a client cannot set or change it.
* ``entity`` — the end user or actor the caller says this work is done
  on behalf of. Free-form, supplied by the caller, passed through
  untouched. Set it per block with ``session(entity=...)`` or
  process-wide with :func:`configure_default_entity`.
* ``agent`` — the conversation role a message was produced under (for
  example the name of the agent that wrote it). Policies usually match
  on this one.

Concurrency note: asyncio tasks and ``asyncio.to_thread`` inherit context
variables. They share a scope lease, so ending a session also invalidates
inherited scopes. ``end_on_exit=False`` keeps that lease open. Ordinary
executor workers need ``contextvars.copy_context().run`` or their own session.

"""

import contextvars
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TypeAlias, TypeGuard, cast
from uuid import uuid4
from weakref import WeakValueDictionary


class _GlobalSession:
    """Sentinel type for the per-tenant **global session** — the
    cross-session view that observes every session in the tenant.

    Distinct from a concrete session id (a ``str``) and from "no session
    declared" (``None``, inactive unless a process default was enabled). Use the module
    singleton :data:`GLOBAL_SESSION` and test for it with ``is``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "GLOBAL_SESSION"


#: Marker for the per-tenant global session. Reach it via
#: :func:`global_session`; detect it with
#: ``get_current_session_id() is GLOBAL_SESSION``.
GLOBAL_SESSION = _GlobalSession()

#: A resolved active session: a concrete id, or the global view.
SessionId = str | _GlobalSession
PolicySource: TypeAlias = str | bytes | os.PathLike[str]
PolicyMetadata: TypeAlias = Iterable[tuple[str, str] | tuple[str, str, str]]

# The active session in this context: ``None`` = nothing declared
# (uses an explicitly enabled process default), a ``str`` = a concrete
# session, ``GLOBAL_SESSION`` = the explicit tenant-global view.
_session_id_var: "contextvars.ContextVar[SessionId | None]" = contextvars.ContextVar(
    "sasy_session_id", default=None
)
_policy_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sasy_policy_id", default=None
)
# Per-call user-supplied actor in the caller's domain. Distinct from
# `principal` (server-stamped, immutable, the auth-bound writer) and
# from `agent` (conversation role label). Free-form, untouched by the
# server. Resolution mirrors session_id: per-block context manager,
# falling back to a process-wide default.
_entity_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sasy_entity", default=None
)
_entity_exact_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sasy_entity_exact", default=False
)

class SessionScopeError(RuntimeError):
    """Recording or authorization has no live SASY session scope."""


@dataclass
class _SessionLease:
    # Context copies share this object, so closing a scope invalidates inherited
    # task contexts as well as the context which entered the block.
    closed: bool = False


# Weak values discard registry entries when no context or captured scope retains
# a lease. Distinct blocks keep distinct identities, even for the same session.
_session_leases: WeakValueDictionary[tuple[str, int], _SessionLease] = WeakValueDictionary()
_session_leases_lock = RLock()


def _new_session_lease(session_id: str) -> _SessionLease:
    lease = _SessionLease()
    with _session_leases_lock:
        _session_leases[(session_id, id(lease))] = lease
    return lease


def flush_session_scopes(session_id: str) -> None:
    """Export completed observations before invalidating a session's scopes."""
    with _session_leases_lock:
        live = any(sid == session_id and not lease.closed
                   for (sid, _), lease in list(_session_leases.items()))
    if live:
        for flush in tuple(_session_flush_callbacks):
            try:
                flush()
            except Exception:
                pass


def close_session_scopes(session_id: str) -> None:
    """Invalidate all local scopes for a concrete session being ended.

    This also closes continued scopes retained by child tasks and process
    defaults. Reopening the same ID creates new leases, never revives old ones.
    """
    with _session_leases_lock:
        for (sid, _), lease in list(_session_leases.items()):
            if sid == session_id:
                lease.closed = True


_session_lease_var: contextvars.ContextVar[_SessionLease | None] = contextvars.ContextVar(
    "sasy_session_lease", default=None
)


_default_session_id: str | None = None
_default_session_lease: _SessionLease | None = None
_session_flush_callbacks: list[Callable[[], None]] = []
_default_entity: str | None = None


def configure_default_session(session_id: str | None = None) -> str:
    """Set the process-wide default session id.

    Used when no ``session(...)`` context is active. Pass ``None`` to
    auto-generate a UUID. Returns the chosen id so callers can log it
    or pass it to other systems.

    Typically called from the SDK's ``configure()`` flow at process
    init. Calling it more than once replaces the previous default.
    """
    global _default_session_id, _default_session_lease
    if _default_session_lease is not None:
        _default_session_lease.closed = True
    if session_id is None:
        session_id = str(uuid4())
    _default_session_lease = _new_session_lease(session_id)
    _default_session_id = session_id
    return session_id


def is_session_active() -> bool:
    """Whether hooks should instrument this context.

    A closed inherited scope raises: treating it as inactive would silently
    let background work escape enforcement after its owner exits.
    """
    lease = _session_lease_var.get()
    if lease is None and _session_id_var.get() is None:
        lease = _default_session_lease
    if lease is not None and lease.closed:
        raise SessionScopeError("The SASY session scope has ended; start a new sasy.session().")
    return _session_id_var.get() is not None or _default_session_id is not None


def require_active_session() -> None:
    """Require an explicit live scope or an opted-in process default."""
    if not is_session_active():
        raise SessionScopeError(
            "SASY instrumentation requires an active sasy.session(); use "
            "sasy.configure(process_global_session=True) to opt into a process default."
        )


def get_current_session_id() -> SessionId:
    """Return the explicit session or opted-in process default; otherwise raise."""
    require_active_session()
    sid = _session_id_var.get()
    if sid is not None:
        return sid
    assert _default_session_id is not None
    return _default_session_id


def configure_process_global_session(enabled: bool) -> None:
    """Enable a stable generated process default, or disable it."""
    if enabled:
        if (_default_session_id is None or _default_session_lease is None
                or _default_session_lease.closed):
            configure_default_session()
    else:
        reset_default_session()


@dataclass(frozen=True)
class CapturedSessionScope:
    session_id: SessionId
    lease: _SessionLease
    entity: str | None
    policy_id: str | None


def capture_session_scope() -> CapturedSessionScope:
    """Capture the live scope for deferred SDK work, retaining its lifetime."""
    sid = get_current_session_id()
    lease = _session_lease_var.get() or _default_session_lease
    if lease is None:
        raise SessionScopeError("The SASY session has no scope lease.")
    return CapturedSessionScope(sid, lease, get_current_entity(), get_current_policy_id())


@contextmanager
def bind_session_scope(scope: CapturedSessionScope) -> Iterator[None]:
    """Restore a captured SDK scope without reopening an ended session."""
    if scope.lease.closed:
        raise SessionScopeError("The captured SASY session scope has ended.")
    sid_token = _session_id_var.set(scope.session_id)
    lease_token = _session_lease_var.set(scope.lease)
    entity_token = _entity_var.set(scope.entity)
    exact_token = _entity_exact_var.set(True)
    policy_token = _policy_id_var.set(scope.policy_id)
    try:
        yield
    finally:
        _policy_id_var.reset(policy_token)
        _entity_var.reset(entity_token)
        _entity_exact_var.reset(exact_token)
        _session_lease_var.reset(lease_token)
        _session_id_var.reset(sid_token)


def register_session_flush(callback: Callable[[], None]) -> None:
    """Register an SDK exporter flush before an explicit session is ended."""
    if callback not in _session_flush_callbacks:
        _session_flush_callbacks.append(callback)


def current_wire_session_id() -> str | None:
    """The active session in the form the wire wants: a concrete session
    id, or ``None`` for the global session.

    ``GLOBAL_SESSION`` maps to ``None`` because the proto carries an
    optional ``session_id`` whose *unset* value the server reads as the
    tenant-global view. Used by the request builders so the global
    session sends no ``session_id`` rather than an empty string.
    """
    sid = get_current_session_id()
    # `isinstance(str)` (not `is GLOBAL_SESSION`) so the type narrows:
    # a concrete session is a str; anything else is GLOBAL_SESSION.
    return sid if isinstance(sid, str) else None


def reset_default_session() -> None:
    """Clear the process-wide default. Test helper; not part of
    the public agent API. Calls outside explicit scopes become inactive.
    """
    global _default_session_id, _default_session_lease
    if _default_session_lease is not None:
        _default_session_lease.closed = True
    _default_session_id = None
    _default_session_lease = None


def configure_default_entity(entity: str | None) -> None:
    """Set the process-wide default user-supplied entity.

    Used when no ``session(entity=...)`` context is active. Typically
    set once at SDK init for single-user agents (e.g.,
    ``sasy.config.configure(entity="alice")``, which calls this
    function). Multi-user gateways usually override per-session via the
    context manager and leave the process default unset.

    See the module docstring for the entity-vs-principal distinction.
    """
    global _default_entity
    _default_entity = entity


def get_current_entity() -> str | None:
    """Return the active user-supplied entity, or ``None`` if unset.

    Resolution order (highest priority first):
    1. ``with sasy.session(entity=...)`` — context manager value
    2. ``configure_default_entity(...)`` — process-wide default
    3. ``None`` — entity is left unset on emitted events; the
       server's ``principal`` (auth-derived) is still attached.
    """
    e = _entity_var.get()
    if e is not None or _entity_exact_var.get():
        return e
    return _default_entity


def get_current_policy_id() -> str | None:
    """Return the active policy id, or ``None`` if no session
    is pinning a specific variant. ``None`` means the server uses
    the caller's tenant default policy.
    """
    return _policy_id_var.get()


class _SessionHandle:
    """Handle yielded by :func:`session`.

    Compares and formats as the session id string, so
    ``with sasy.session(...) as sid: print(sid)`` prints the id, and
    exposes :meth:`set_policy` to rebind the live session to a
    different policy. Policy ids are server-internal — clients
    work with policy *source* (string or `Path`) on the public
    API.
    """

    __slots__ = ("session_id", "_policy_id", "_policy_token", "_scope")

    def __init__(
        self,
        session_id: SessionId,
        policy_id: str | None,
        policy_token: object,
    ) -> None:
        # A concrete id, or GLOBAL_SESSION for the per-tenant global view.
        self.session_id = session_id
        # Server-assigned id stays internal — surfaced only via the
        # private attribute so advanced callers (or test
        # introspection) can still see what the server bound to,
        # but routine code stays source-based.
        self._policy_id = policy_id
        self._policy_token = policy_token
        self._scope = capture_session_scope()

    # Treat the handle as the session id when used in str/format
    # contexts — keeps the yield-string ergonomics callers expect.
    # The global session stringifies to "".
    def __str__(self) -> str:
        return self.session_id if isinstance(self.session_id, str) else ""

    def __repr__(self) -> str:
        return f"_SessionHandle(session_id={self.session_id!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.session_id == other
        if isinstance(other, _SessionHandle):
            return self.session_id == other.session_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.session_id)

    def set_policy(
        self,
        policy: PolicySource,
        functors: PolicySource | None = None,
        # Empty string = keep the server's current backend for the
        # tenant. Hardcoding "souffle" here would silently switch
        # deployments running souffle-interpreted / flowlog whenever
        # a caller forgot to pass an explicit backend.
        backend: str = "",
        policy_metadata: PolicyMetadata | None = None,
    ) -> bool:
        """Rebind this session to a new policy for the rest of its
        scope. The next authorization check uses the new policy;
        graph state is preserved across the rebind.

        ``policy`` is dispatched by type: ``str``/``bytes`` is
        Soufflé source; ``Path``/``os.PathLike`` is a file path
        (``#include`` resolved, companion functors auto-detected).
        Same rule for ``functors``. Wrap a string in ``Path(...)``
        to force path semantics.

        The SDK calls ``SetPolicy(scope=Session)`` — server dedupes
        by content so re-binding to the same source as a previous
        upload is free. Returns ``True`` if the call succeeded.

        Atomicity caveat: the new policy id is only written to the
        ContextVar after the RPC returns success. Concurrent
        ``check_tool_call`` dispatches on the same session during
        the RPC's flight observe the *old* policy id client-side
        even though the server has already swapped. This is benign
        — the server is the source of truth, and the client-side id
        is informational — but agents that branch on
        ``get_current_policy_id()`` mid-call should be aware.

        ContextVar discipline: the in-block ``_policy_id_var.set``
        does not save a token because the enclosing ``session()``
        block's ``reset(pid_token)`` already unwinds *any* sets that
        happened after the original — so the outer scope is
        restored correctly regardless of how many times this gets
        called inside the block.
        """
        active = capture_session_scope()
        if active.lease is not self._scope.lease:
            raise SessionScopeError("set_policy() must run inside its original live session scope.")
        source, fns = _resolve_policy_source(policy, functors)

        from sasy.policy.api import set_session_policy as _set_session_policy

        resp = _set_session_policy(
            self.session_id if isinstance(self.session_id, str) else None,
            source,
            functor_source=fns,
            backend=backend,
            policy_metadata=policy_metadata,
        )
        if not resp.accepted:
            raise RuntimeError(
                f"set_policy(): rejected: {resp.message}\n{resp.error_output}"
            )
        new_id = resp.policy_id
        _policy_id_var.set(new_id)
        self._policy_id = new_id
        return True


def _resolve_policy_source(
    policy: PolicySource,
    functors: PolicySource | None,
) -> tuple[str, str]:
    """Common helper for ``session()`` and ``_SessionHandle.set_policy()``.

    Type-based dispatch (no heuristics):

    * ``policy`` of type ``str`` / ``bytes`` → Soufflé source.
    * ``policy`` of type ``Path`` / ``os.PathLike`` → file path;
      ``#include`` is resolved and companion functors are
      auto-detected via :func:`sasy.policy.find_functors`.

    Same rule for ``functors``. Wrap a string in ``Path(...)`` if
    you want it interpreted as a path.
    """
    from pathlib import Path as _Path

    from sasy.policy.api import find_functors, resolve_includes

    if not _is_path_like(policy):
        # str / bytes — Soufflé source.
        source = policy if isinstance(policy, str) else cast(bytes, policy).decode()
        if functors is None:
            fns = ""
        elif _is_path_like(functors):
            fns = _Path(functors).read_text()
        else:
            fns = functors if isinstance(functors, str) else cast(bytes, functors).decode()
    else:
        p = _Path(policy)
        source = resolve_includes(p)
        if functors is None:
            fns = find_functors(p)
        elif _is_path_like(functors):
            fns = _Path(functors).read_text()
        else:
            fns = functors if isinstance(functors, str) else cast(bytes, functors).decode()
    return source, fns


def _is_path_like(value: object) -> TypeGuard[os.PathLike[str]]:
    """True iff ``value`` is a filesystem path argument (``pathlib.Path``
    or ``os.PathLike``). Plain ``str`` is *always* treated as source —
    if you want a string interpreted as a path, wrap it in ``Path``.
    """
    if isinstance(value, (str, bytes)):
        return False
    # Anything else with `__fspath__` (Path, os.PathLike) is a path.
    return hasattr(value, "__fspath__") or isinstance(value, os.PathLike)


@contextmanager
def session(
    session_id: str | None = None,
    policy: PolicySource | None = None,        # str (source) | Path | None
    functors: PolicySource | None = None,      # str (source) | Path | None
    # Empty string = keep the server's current backend for the
    # tenant. Hardcoding "souffle" here would silently switch
    # deployments running souffle-interpreted / flowlog whenever
    # a caller forgot to pass an explicit backend.
    backend: str = "",
    entity: str | None = None,
    end_on_exit: bool = True,
    policy_metadata: PolicyMetadata | None = None,
) -> Iterator[_SessionHandle]:
    """Override the active session id for the duration of this block.

    All graph events recorded inside this block carry the chosen
    ``session_id``; all policy queries are evaluated against that
    session's partition. On exit, the previous session is restored.

    Pass ``None`` for ``session_id`` to auto-generate a fresh UUID —
    useful for one-shot runs (e.g., a single agent run) where the
    caller does not need a stable id.

    ``policy`` declares the policy this session evaluates under.
    Two forms, dispatched purely by Python type:

    * ``str`` / ``bytes`` — Soufflé source, uploaded inline as a
      non-default variant on block entry; the session is pinned to
      it for its lifetime.
    * ``pathlib.Path`` / any ``os.PathLike`` — file path, read from
      disk; ``#include`` is resolved and companion functors are
      auto-detected via :func:`sasy.policy.find_functors`, then
      uploaded as a variant.

    Plain ``str`` is *always* source — wrap a path in
    ``Path("...")`` to interpret it as a file. ``policy="my_policy.dl"``
    is therefore compiled as the text ``my_policy.dl`` and fails with a
    parser error; pass ``Path("my_policy.dl")`` to read the file.

    ``functors`` follows the same type-based dispatch and overrides
    the auto-detected companion functors for path-form policies.

    Omitting ``policy`` retains an existing binding for this session id.
    Without a session binding, the server uses the caller's tenant default.

    Server-assigned policy ids are not exposed on the public API —
    the SDK uploads, captures the id internally, and binds the
    session in one step. The server dedupes uploads by content
    hash, so re-running the same source repeatedly (e.g. across
    parametrized tests) is effectively free after the first
    compile.

    Once a session is bound to a policy, switching mid-session
    requires ``handle.set_policy(<new_source_or_path>)``.

    When ``end_on_exit`` is true (the default), ``__exit__`` calls
    ``EndSession`` to release the evaluator, policy binding, session
    metadata and ownership. Recorded graph observations survive, but
    reusing the id does not retain its policy or approvals. Use a fresh
    id for a new conversation, or ``end_on_exit=False`` to continue an
    existing session across blocks and end it explicitly when finished.
    Child tasks inherit the scope. Ending it invalidates their inherited scope;
    with ``end_on_exit=False`` they can continue after the block exits.
    Teardown is best-effort: RPC failures are swallowed and can leave
    server state in place without breaking the agent's teardown path.

    Args:
        session_id: Stable name for the session; ``None`` generates a UUID.
        policy: Policy to bind on entry, as described above. ``None``
            keeps whatever this session id is already bound to, or the
            tenant default if it has none.
        functors: C++ functor source or path; overrides the companion
            functors found next to a policy file.
        backend: Evaluator backend to request. ``""`` (the default)
            keeps the backend the engine is already using.
        entity: The end user or actor this block runs on behalf of.
            Stamped on messages recorded inside the block and sent with
            each authorization check. Caller-supplied, and distinct from
            the authenticated ``principal``.
        end_on_exit: End the session on exit (the default).
        policy_metadata: ``(rel, a)`` or ``(rel, a, b)`` facts installed
            with the policy and readable as ``PolicyMetadata(rel, a, b)``.

    Yields:
        A handle that compares and formats as the session id, and whose
        ``set_policy(...)`` rebinds the running session.

    Raises:
        RuntimeError: The engine rejected the policy. The message carries
            the compiler output, and the block body does not run.
        FileNotFoundError: ``policy`` is a path that does not exist.
        SasyEndpointNotConfigured: No endpoint is configured.
        grpc.RpcError: The engine could not be reached.

    Examples
    --------
    ::

        # 1. Inline source (most common in tests):
        with sasy.session(policy=ALLOW_ALL_POLICY) as s:
            assert check_tool_call("foo", "{}").authorized

        # 2. From a .dl file (resolves #include + finds functors).
        # ``str`` is always source — wrap the path so it's read
        # from disk instead of compiled as the literal text
        # "examples/message-flow/policy.dl".
        with sasy.session(policy=Path("examples/message-flow/policy.dl")):
            run_agent()

        # 3. Mid-session swap to a different policy:
        with sasy.session(policy=POLICY_A) as s:
            ...
            s.set_policy(POLICY_B)
            ...

        # 4. No policy declared → use tenant default:
        with sasy.session():
            ...
    """
    if session_id is None:
        session_id = str(uuid4())

    # Resolve `policy=` and bind the session to it via SetPolicy.
    # The server dedupes by content hash so repeated identical
    # sources across sessions collapse to one registry entry
    # without a recompile.
    resolved_policy_id: str | None = None
    if policy is not None:
        from sasy.policy.api import set_session_policy as _set_session_policy

        source, fns = _resolve_policy_source(policy, functors)
        resp = _set_session_policy(
            session_id,
            source,
            functor_source=fns,
            backend=backend,
            policy_metadata=policy_metadata,
        )
        if not resp.accepted:
            raise RuntimeError(
                f"session(): policy upload rejected: "
                f"{resp.message}\n{resp.error_output}"
            )
        resolved_policy_id = resp.policy_id
    lease = _new_session_lease(session_id)
    lease_token = _session_lease_var.set(lease)
    sid_token = _session_id_var.set(session_id)
    pid_token = _policy_id_var.set(resolved_policy_id)
    # Only enter an entity binding if the caller actually passed one;
    # leaving the var alone preserves whatever (outer block / process
    # default) was already set.
    eid_token = _entity_var.set(entity) if entity is not None else None
    try:
        yield _SessionHandle(session_id, resolved_policy_id, pid_token)
    finally:
        if end_on_exit:
            try:
                flush_session_scopes(session_id)
            finally:
                close_session_scopes(session_id)
        _session_lease_var.reset(lease_token)
        _session_id_var.reset(sid_token)
        _policy_id_var.reset(pid_token)
        if eid_token is not None:
            _entity_var.reset(eid_token)
        if end_on_exit:
            # Local import: avoid pulling the policy gRPC client
            # (and its proto deps) at module import time, which
            # matters because instrumentation/session.py is
            # imported eagerly by the SDK init path.
            try:
                from sasy.policy.api import end_session as _end_session
                _end_session(session_id)
            except Exception:
                # Final safety net — never propagate an exit-time
                # failure out of a context manager.
                pass


@contextmanager
def global_session(
    policy: PolicySource | None = None,
    functors: PolicySource | None = None,
    # Empty string = keep the server's current backend for the
    # tenant. Hardcoding "souffle" here would silently switch
    # deployments running souffle-interpreted / flowlog whenever
    # a caller forgot to pass an explicit backend.
    backend: str = "",
    entity: str | None = None,
    policy_metadata: PolicyMetadata | None = None,
) -> Iterator[_SessionHandle]:
    """Bind to the **per-tenant global session** — the
    cross-session view that observes events from every session in
    the caller's tenant. Useful for break-glass / tenant-admin
    tooling that wants to enforce a single policy across the whole
    tenant view without disturbing per-session bindings.

    The active session is set to :data:`GLOBAL_SESSION`, which sends no
    ``session_id`` on the wire (the server reads an unset session as the
    tenant-global partition). The auto-UUID default is suppressed and the
    meaning is explicit. Like :func:`session`, ``policy`` may be a source
    string or a path-like.

    Note: ``end_on_exit`` is forced ``False`` because the global
    session lifecycle is tenant-wide; dropping it would surprise
    other admins who expected the binding to persist.
    """
    lease_token = _session_lease_var.set(_SessionLease())
    sid_token = _session_id_var.set(GLOBAL_SESSION)
    resolved_policy_id: str | None = None
    if policy is not None:
        from sasy.policy.api import set_session_policy as _set_session_policy

        try:
            source, fns = _resolve_policy_source(policy, functors)
            resp = _set_session_policy(
                None,
                source,
                functor_source=fns,
                backend=backend,
                policy_metadata=policy_metadata,
            )
        except BaseException:
            # _resolve_policy_source or the RPC raised — restore
            # the ContextVar before the exception escapes so the
            # caller's next call doesn't observe the global session.
            _session_id_var.reset(sid_token)
            _session_lease_var.reset(lease_token)
            raise
        if not resp.accepted:
            _session_id_var.reset(sid_token)
            _session_lease_var.reset(lease_token)
            raise RuntimeError(
                f"global_session(): policy upload rejected: "
                f"{resp.message}\n{resp.error_output}"
            )
        resolved_policy_id = resp.policy_id

    pid_token = _policy_id_var.set(resolved_policy_id)
    eid_token = _entity_var.set(entity) if entity is not None else None
    try:
        yield _SessionHandle(GLOBAL_SESSION, resolved_policy_id, pid_token)
    finally:
        _session_lease_var.reset(lease_token)
        _session_id_var.reset(sid_token)
        _policy_id_var.reset(pid_token)
        if eid_token is not None:
            _entity_var.reset(eid_token)
