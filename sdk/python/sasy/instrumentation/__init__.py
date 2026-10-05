"""
Instrumentation for Langroid, Google ADK, LangChain and HTTP libraries.

Usage:
    import sasy
    sasy.instrument()  # every installed agent framework

    # Also route requests/httpx calls through the reference monitor, which
    # authorizes each request and injects credentials
    sasy.instrument(http=True)

    # Or selectively
    sasy.instrument(langchain=False, adk=False, langroid=False, http=True)  # HTTP hooks only

    # Check tool call authorization manually
    result = sasy.check_tool_call(fn_name="my_tool", args='{"key": "value"}')
"""

import functools
import importlib
import importlib.util
import inspect
import sys
import threading
import types
import warnings
from collections.abc import Callable, Collection
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from sasy.capture import capture_logger
from sasy.reference_monitor import check_tool_call

from . import otel, testing
from .config import FeedbackCallback, InstrumentationConfig, configure, get_config
from .feedback import (
    FeedbackAccumulator,
    add_denial,
    add_tool_denial,
    get_current_feedback,
)
from .http import instrument as _instrument_http
from .langroid import instrument as instrument_langroid

# Re-export for convenience
instrument_http = _instrument_http

logger = capture_logger(__name__)


class SasyInstrumentationWarning(UserWarning):
    """:func:`instrument` skipped an installed framework it cannot check.

    Given the first time the application uses that framework, once per
    process. That framework's agents run without SASY's checks. Pass the
    framework's flag as ``True`` to make this an error, or as ``False`` if the
    application does not use the framework (which also silences the warning).
    """


# flag -> (modules any of which means the framework is installed, name,
# distribution, what an application that needs the adapter uses).
# LangGraph does not depend on the ``langchain`` package, so a LangGraph-only
# application is detected by ``langgraph``: its ``ToolNode`` tool calls are
# what the LangChain adapter's backstop checks.
_FRAMEWORKS = {
    "adk": (("google.adk",), "Google ADK", "google-adk", "Google ADK"),
    "langchain": (("langchain", "langgraph"), "LangChain", "langchain", "LangChain or LangGraph"),
    "langroid": (("langroid",), "Langroid", "langroid", "Langroid"),
}

# The Langroid modules the Langroid adapter imports and patches when it is
# installed. The adapter's own module imports without Langroid, so these are
# imported up front: a Langroid the adapter cannot load with is then found
# before any patch is installed.
_LANGROID_MODULES = (
    "langroid",
    "langroid.agent.base",
    "langroid.agent.chat_agent",
    "langroid.agent.chat_document",
    "langroid.agent.task",
)


def _installed(module: str) -> bool:
    """Whether *module* can be found, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A dotted name imports its parent: no ``google`` means no ``google.adk``.
        return False


def _selected(flag_name: str, flag: bool | None) -> bool:
    """Whether a framework adapter is to be enabled.

    ``None`` enables it when the framework is installed, ``True`` requires it
    and ``False`` skips it.
    """
    if flag is False:
        return False
    modules, name, _, _ = _FRAMEWORKS[flag_name]
    if any(_installed(module) for module in modules):
        return True
    if flag is None:
        return False
    raise RuntimeError(
        f"{flag_name}=True, but {name} is not installed ({' or '.join(modules)} cannot be found). "
        f"Run: pip install 'sasy[{flag_name}]'"
    )


def _hint(flag_name: str) -> str:
    return (f"If this application does not use {_FRAMEWORKS[flag_name][3]}, pass "
            f"{flag_name}=False to sasy.instrument().")


def _unloadable(flag_name: str, error: ImportError) -> RuntimeError:
    """The error for an installed framework the adapter cannot even import with.

    For LangChain the framework may be LangGraph alone, without the
    ``langchain`` package the adapter imports.
    """
    _, name, distribution, _ = _FRAMEWORKS[flag_name]
    try:
        found = f"{distribution} {version(distribution)} is installed, but SASY's {name} adapter cannot load with it"
    except PackageNotFoundError:
        found = f"{distribution} is not installed, so SASY's {name} adapter cannot load"
    return RuntimeError(f"{found} ({error}). Run: pip install 'sasy[{flag_name}]'.")


def _skipped_message(flag_name: str, problem: Exception) -> str:
    """The warning for an installed framework whose agents SASY does not check."""
    _, name, _, uses = _FRAMEWORKS[flag_name]
    text = str(problem).rstrip()
    if not text.endswith("."):
        text += "."
    return (
        f"{text} SASY skipped its {name} adapter: {uses} agents in this process will NOT be "
        f"checked by SASY. Pass {flag_name}=True to sasy.instrument() to make this an error, "
        f"or {flag_name}=False if this application does not use {uses} (which also silences "
        "this warning)."
    )


# flag -> the public entry points every use of the framework passes through:
# (module, class or None for a module-level function, attributes, required).
# A skipped framework's adapter could not vouch for this version's internals,
# so only these public names are hooked, and only to warn. Attributes a
# version lacks are left alone; a required entry whose module or class cannot
# be loaded, or that has none of its attributes, means the warning is given
# at instrument() time instead.
_ENTRY_POINTS: dict[str, tuple[tuple[str, str | None, tuple[str, ...], bool], ...]] = {
    "adk": (("google.adk.runners", "Runner", ("run", "run_async", "run_live"), True),),
    "langchain": (
        # Every LangGraph graph, including the agents create_agent builds,
        # runs through these; LangGraph alone has no ``langchain`` package.
        ("langgraph.pregel", "Pregel", ("invoke", "ainvoke", "stream", "astream"), True),
        ("langchain.agents", None, ("create_agent",), False),
        ("langchain.agents.factory", None, ("create_agent",), False),
    ),
    "langroid": (
        ("langroid.agent.task", "Task", ("run", "run_async"), True),
        ("langroid.agent.chat_agent", "ChatAgent",
         ("llm_response", "llm_response_async", "llm_response_messages", "llm_response_messages_async"), True),
    ),
}

# Frameworks whose skipped-adapter warning has been given, or silenced by
# ``flag=False``: it is given at most once per process.
_warned: set[str] = set()
# Frameworks whose entry points carry the first-use warning.
_hooked: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(flag_name: str, message: str, stacklevel: int) -> None:
    """Warn (and log) that a framework is unchecked, once per process.

    *stacklevel* is relative to the caller, as for :func:`warnings.warn`.
    """
    if flag_name in _warned:
        return
    with _warn_lock:
        if flag_name in _warned:
            return
        _warned.add(flag_name)
    logger.warning(message)
    warnings.warn(message, SasyInstrumentationWarning, stacklevel=stacklevel + 1)


def _first_use_hook(flag_name: str, message: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """*original*, warning on its first use that the framework is unchecked.

    The call itself goes through unchanged, and the hook is the same kind of
    function as *original*: a coroutine function stays one, and a generator
    or async generator function stays one, warning when iteration starts and
    passing every yielded, sent and thrown value and close through. Anything
    else warns when called and returns what *original* returns.
    """
    hook: Callable[..., Any]
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def hook(*args: Any, **kwargs: Any) -> Any:
            # warnings.warn's frame 1 is _warn_once, 2 this hook, 3 the
            # application's call (for a coroutine, the code awaiting it).
            _warn_once(flag_name, message, stacklevel=2)
            return await original(*args, **kwargs)
    elif inspect.isasyncgenfunction(original):
        @functools.wraps(original)
        async def hook(*args: Any, **kwargs: Any) -> Any:
            # Frame 3 is the code iterating the generator.
            _warn_once(flag_name, message, stacklevel=2)
            # There is no ``yield from`` for async generators: delegate as
            # PEP 380 does, so asend, athrow and aclose reach *original*.
            generator = original(*args, **kwargs)
            try:
                value = await generator.__anext__()
            except StopAsyncIteration:
                return
            while True:
                try:
                    sent = yield value
                except GeneratorExit:
                    await generator.aclose()
                    raise
                except BaseException as error:
                    try:
                        value = await generator.athrow(error)
                    except StopAsyncIteration:
                        return
                else:
                    try:
                        value = await generator.asend(sent)
                    except StopAsyncIteration:
                        return
    elif inspect.isgeneratorfunction(original):
        @functools.wraps(original)
        def hook(*args: Any, **kwargs: Any) -> Any:
            # Frame 3 is the code iterating the generator.
            _warn_once(flag_name, message, stacklevel=2)
            return (yield from original(*args, **kwargs))
    else:
        @functools.wraps(original)
        def hook(*args: Any, **kwargs: Any) -> Any:
            _warn_once(flag_name, message, stacklevel=2)
            return original(*args, **kwargs)
    hook.__sasy_first_use_warning__ = flag_name  # type: ignore[attr-defined]
    return hook


def _rebind_module_globals(original: object, replacement: object, *, keep: Collection[str] = ()) -> None:
    """Point every module-level name bound to *original* at *replacement*, so
    a ``from ... import`` made before the patch gets it too.

    Nothing but a module global that is exactly *original* is touched, and
    the modules named in *keep* are left alone.
    """
    for name, module in list(sys.modules.items()):
        if name in keep or not isinstance(module, types.ModuleType):
            continue
        if issubclass(type(module), types.ModuleType):
            # Read past the module's own attribute lookup: for a module
            # deferred with importlib.util.LazyLoader any attribute access,
            # __dict__ included, would run its code. A module that has not
            # run cannot hold *original*.
            namespace = object.__getattribute__(module, "__dict__")
        else:
            # An object registered as a module that only claims to be one
            # (cffi's Lib, for one) computes its own attributes, so a failure
            # to read them skips it rather than failing the patch.
            try:
                namespace = module.__dict__
            except Exception:
                continue
            if not isinstance(namespace, dict):
                continue
        for attribute, value in list(namespace.items()):
            if value is original:
                namespace[attribute] = replacement


def _hook_entry_points(flag_name: str, message: str) -> bool:
    """Make a skipped framework's public entry points warn on first use.

    Returns ``False``, hooking nothing, when a required entry point cannot be
    loaded (for example a broken install).
    """
    if flag_name in _hooked:
        return True
    targets: list[tuple[object, str, Callable[..., Any]]] = []
    for module_name, class_name, attributes, required in _ENTRY_POINTS[flag_name]:
        try:
            owner: object = importlib.import_module(module_name)
            if class_name is not None:
                owner = getattr(owner, class_name)
        except Exception as error:  # noqa: BLE001 - any failure means "cannot hook"
            if required:
                logger.debug("cannot hook %s.%s: %r", module_name, class_name, error)
                return False
            continue
        found = [
            (owner, attribute, function)
            for attribute in attributes
            if isinstance(function := inspect.getattr_static(owner, attribute, None), types.FunctionType)
        ]
        if required and not found:
            logger.debug("cannot hook %s.%s: none of %s found", module_name, class_name, attributes)
            return False
        targets.extend(found)
    hooks: dict[int, Callable[..., Any]] = {}
    for owner, attribute, function in targets:
        if getattr(function, "__sasy_first_use_warning__", None) is not None:
            continue
        # One hook per function, however many names it is bound to.
        if id(function) not in hooks:
            hooks[id(function)] = _first_use_hook(flag_name, message, function)
            if isinstance(owner, types.ModuleType):
                # A module-level function may also be bound in modules that
                # imported it before instrument(). The skipped adapter's own
                # module keeps the original.
                _rebind_module_globals(function, hooks[id(function)], keep=(f"{__name__}.{flag_name}",))
        setattr(owner, attribute, hooks[id(function)])
    _hooked.add(flag_name)
    return True


def _skip(flag_name: str, problem: Exception) -> None:
    """Skip an installed framework in auto mode.

    Nothing is said now: an application that never uses the framework is not
    warned. Its public entry points instead warn on first use. When they cannot
    be loaded, as when the adapter's own import failed on a broken install,
    the warning is given here instead.
    """
    message = _skipped_message(flag_name, problem)
    if _hook_entry_points(flag_name, message):
        logger.debug("%s adapter skipped; warning on first use: %s", _FRAMEWORKS[flag_name][1], message)
        return
    # warnings.warn's frame 1 is _warn_once, 2 this function, 3 _adapter,
    # 4 instrument, 5 the application's call.
    _warn_once(flag_name, message, stacklevel=4)


def _adapter(
    flag_name: str,
    flag: bool | None,
    load: Callable[[], tuple[Callable[[], None], Callable[[], None]]],
) -> Callable[[], None] | None:
    """Load a selected framework's adapter and run its version check.

    *load* imports the adapter and returns its version check and its install
    function. An installed framework at a version the adapter does not
    support, or one the adapter cannot import with, is an error when the flag
    is ``True``. In auto mode (``None``) the framework is skipped (``None`` is
    returned) and a :class:`SasyInstrumentationWarning` is given when the
    application first uses it, since it may not use it at all.
    """
    cause: Exception
    try:
        check, install = load()
    except ImportError as error:
        problem, cause = _unloadable(flag_name, error), error
    else:
        try:
            check()
            return install
        except RuntimeError as error:
            problem = cause = error
    if flag is None:
        _skip(flag_name, problem)
        return None
    raise type(problem)(f"{problem} {_hint(flag_name)}") from cause


def _adk_adapter() -> tuple[Callable[[], None], Callable[[], None]]:
    from .adk import check_supported
    from .adk import instrument as install
    return check_supported, install


def _langchain_adapter() -> tuple[Callable[[], None], Callable[[], None]]:
    from .langchain import check_supported, instrument_langchain
    return check_supported, instrument_langchain


def _langroid_adapter() -> tuple[Callable[[], None], Callable[[], None]]:
    for module in _LANGROID_MODULES:
        importlib.import_module(module)
    # Langroid has no exact version pin, so there is no version check.
    return (lambda: None), instrument_langroid


def instrument(
    http: bool = False,
    langroid: bool | None = None,
    *,
    adk: bool | None = None,
    langchain: bool | None = None,
) -> None:
    """Install SASY's process-wide patches. Call once at start-up, after
    :func:`configure`.

    Each framework flag takes ``None`` (the default: enable the adapter if the
    framework is installed, and do nothing if it is not), ``True`` (require
    it) or ``False`` (skip it). With ``None``, an installed framework at a
    version its adapter does not support, or one its adapter cannot load
    with, is skipped: that framework's agents are then NOT checked by SASY.
    Nothing is said at this call; the first time the application uses the
    framework (builds a LangChain agent or runs a LangGraph graph, runs an ADK
    ``Runner``, runs a Langroid ``Task`` or asks a ``ChatAgent`` for an LLM
    response), a :class:`SasyInstrumentationWarning` is given, once per
    process, and logged. When even those entry points cannot be loaded, the
    warning is given at this call instead. Pass ``True`` to make it an error,
    or ``False`` for a framework the application does not use, which also
    silences the warning.

    Args:
        http: Route ``requests`` and ``httpx`` calls through the engine's
            reference monitor, which authorizes each request and injects
            credentials. Off by default; turn it on when the application needs
            credential injection or its outgoing HTTP requests checked by the
            policy. Requests to ``localhost``, ``127.0.0.1`` and ``::1`` are
            never routed.
        langroid: Patch Langroid's responders and tool handlers.
        adk: Patch Google ADK's ``Runner``, model and tool dispatch. Requires
            the exact ADK version pinned by ``pip install 'sasy[adk]'``.
        langchain: Make ``langchain.agents.create_agent`` build SASY's
            recorded and authorized agent
            (:func:`sasy.instrumentation.langchain.create_agent`), and refuse,
            inside a ``sasy.session`` or ``sasy.global_session``, every tool
            call of a LangGraph ``ToolNode`` that SASY did not build: inside
            a session, tools run only in agents SASY builds.
            Tools that graph code calls directly, not through a ``ToolNode``
            (for example a hand-written tool node), are not checked. Detected
            by ``langchain`` or ``langgraph`` being installed, and requires
            the exact versions pinned by ``pip install 'sasy[langchain]'``.

    Raises:
        RuntimeError: A framework flag is ``True`` but the framework is not
            installed, or is installed at a version its adapter does not
            support (for ADK, :class:`AdkInstrumentationError`, a subclass),
            or its adapter cannot load with it; or a framework hook the
            adapter needs is missing (raised while installing, so patches
            installed earlier in the same call stay in place). The version
            and load checks run before any patch is installed.

    The patches cannot be removed. Calling this again is harmless.
    """
    # False also silences the first-use warning of an earlier auto-mode call.
    _warned.update(flag_name for flag_name, flag in (("adk", adk), ("langchain", langchain),
                                                     ("langroid", langroid)) if flag is False)
    selected = [
        (flag_name, flag, load)
        for flag_name, flag, load in (
            ("adk", adk, _adk_adapter),
            ("langchain", langchain, _langchain_adapter),
            ("langroid", langroid, _langroid_adapter),
        )
        if _selected(flag_name, flag)
    ]
    # Every adapter is loaded and checked before any patch is installed. A
    # plain loop, not a comprehension: on Python 3.11 a comprehension is a
    # frame of its own, which would shift _skip's stacklevel.
    adapters: dict[str, Callable[[], None] | None] = {}
    for flag_name, flag, load in selected:
        adapters[flag_name] = _adapter(flag_name, flag, load)

    installs = [
        adapters.get("adk"),
        adapters.get("langchain"),
        _instrument_http if http else None,
        adapters.get("langroid"),
    ]
    for install in installs:
        if install is not None:
            install()


__all__ = [
    "instrument",
    "configure",
    "get_config",
    "InstrumentationConfig",
    "FeedbackCallback",
    "SasyInstrumentationWarning",
    "instrument_http",
    "instrument_langroid",
    "check_tool_call",
    "FeedbackAccumulator",
    "get_current_feedback",
    "add_denial",
    "add_tool_denial",
    "otel",
    "testing",
]
