"""Resolve computation-local reads before crossing an action boundary.

Adapters own their observations and scopes. This bridge has no framework imports
and does not turn ordinary context getters into network operations.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import get_ident
from typing import Protocol

from .session import (
    CapturedSessionScope,
    capture_session_scope,
    current_wire_session_id,
    get_current_entity,
)


class ComputationScopeError(RuntimeError):
    """An action was created in one computation and used in another.

    A computation is one tool body, callback or agent step. Authorizing a
    call against a different computation's inputs would decide the wrong
    question, so the boundary refuses instead.
    """


class DependencyResolver(Protocol):
    """What an adapter supplies so this module can name a computation's inputs.

    ``check`` raises if the computation this resolver speaks for is no
    longer running; ``inputs_sync``/``inputs_async`` return the immutable
    version IDs of every message it has consumed so far.
    """

    def check(self) -> None: ...
    def inputs_sync(self) -> list[str]: ...
    async def inputs_async(self) -> list[str]: ...


_resolver: ContextVar[DependencyResolver | None] = ContextVar("sasy_dependency_resolver", default=None)


def set_resolver(resolver: DependencyResolver | None) -> Token:
    """Make *resolver* the one for the current context; returns a reset token."""
    return _resolver.set(resolver)


def reset_resolver(token: Token) -> None:
    """Restore the resolver that was active before the matching ``set_resolver``."""
    _resolver.reset(token)


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@dataclass(frozen=True)
class Binding:
    """The computation an action was started in, captured when it is created.

    Holds the resolver (the adapter scope that knows which messages this
    computation has consumed), the SASY session, the entity, the asyncio
    task and the thread. :meth:`check` raises if any of them differs from
    the caller's current context.

    An action created in one tool body, task or session must not be
    authorized against the inputs of another, so every boundary calls
    :meth:`check` before and after anything that can suspend.
    """

    resolver: DependencyResolver | None
    session: str | None
    entity: str | None
    task: object
    thread: int
    scope: CapturedSessionScope

    def check(self) -> None:
        """Raise :class:`ComputationScopeError` unless the context is still the bound one."""
        current_scope = capture_session_scope()
        if self.scope.lease is not current_scope.lease:
            raise ComputationScopeError("This action belongs to a different SASY session scope.")
        if self.session != current_wire_session_id():
            raise ComputationScopeError("This action belongs to a different SASY session.")
        if _resolver.get() is not self.resolver:
            raise ComputationScopeError(
                "This action was started inside one tool body and is being used "
                "outside it (for example from executor.submit, a background task, "
                "or after the tool returned). Make the call inside the tool body, "
                "or carry the context with asyncio.to_thread or "
                "contextvars.copy_context().run."
            )
        if self.resolver is not None:
            if (self.session != current_wire_session_id() or self.entity != get_current_entity()
                    or self.task is not _task() or self.thread != get_ident()):
                raise ComputationScopeError(
                    "This action was started in one task, thread, session or entity "
                    "and is being used in another. Make the call in the task and "
                    "session that started it, or carry the context with "
                    "asyncio.to_thread or contextvars.copy_context().run."
                )
            self.resolver.check()


def bind() -> Binding:
    """Capture the current computation, refusing if there is not one to act in."""
    binding = Binding(_resolver.get(), current_wire_session_id(), get_current_entity(), _task(), get_ident(), capture_session_scope())
    binding.check()
    return binding


def resolve_inputs(ids: list[str] | None = None, *, binding: Binding | None = None) -> list[str]:
    """Return *ids* followed by the current computation's own inputs, without duplicates."""
    bound = binding if binding is not None else bind()
    bound.check()
    explicit = list(ids or [])
    resolved = bound.resolver.inputs_sync() if bound.resolver is not None else []
    # Resolving can block. Re-check afterwards: the scope may have been
    # closed, or the session or entity changed, while the resolver ran, and
    # the ids must not be used for a computation that has ended.
    bound.check()
    return list(dict.fromkeys([*explicit, *resolved]))


async def resolve_inputs_async(ids: list[str] | None = None, *, binding: Binding | None = None) -> list[str]:
    """Async form of :func:`resolve_inputs`."""
    bound = binding if binding is not None else bind()
    bound.check()
    explicit = list(ids or [])
    resolved = await bound.resolver.inputs_async() if bound.resolver is not None else []
    # Awaiting the resolver suspends this task; re-check for the same reason
    # as in resolve_inputs.
    bound.check()
    return list(dict.fromkeys([*explicit, *resolved]))
