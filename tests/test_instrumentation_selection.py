"""Framework selection: installed frameworks are enabled, required ones must be
installed, and an installed framework the adapter does not support is an error
when it is required and, in auto mode, is skipped with a warning on its first
use (or at once, when even its public entry points cannot be loaded). HTTP
routing is off unless asked for.

Whether a framework is installed is faked here, and so are the adapters and
the frameworks' public entry points, so the result does not depend on which
frameworks this environment really has.
"""
import asyncio
import contextlib
import inspect
import logging
import sys
import warnings
from types import ModuleType

import pytest
import sasy.instrumentation as instrumentation
from sasy.instrumentation import SasyInstrumentationWarning

# flag -> the modules any of which means the framework is installed.
MODULES = {"adk": ("google.adk",), "langchain": ("langchain", "langgraph"), "langroid": ("langroid",)}

# flag -> the module whose import the adapter needs; None in sys.modules makes
# that import fail, as for a framework version whose internals it cannot find.
ADAPTER_IMPORTS = {"adk": "sasy.instrumentation.adk", "langchain": "sasy.instrumentation.langchain",
                   "langroid": "langroid.agent.task"}


def _is_async(attribute):
    return attribute.startswith("a") and attribute[1:] in {"invoke", "stream"} or attribute.endswith("_async")


def _entry_function(module_name, attribute):
    """A stand-in for a framework entry point: it returns what it was called
    with, so a hook can be seen to call through unchanged."""
    if _is_async(attribute):
        async def entry(*args, **kwargs):
            return (module_name, attribute, args, kwargs)
    else:
        def entry(*args, **kwargs):
            return (module_name, attribute, args, kwargs)
    entry.__name__ = entry.__qualname__ = attribute
    return entry


def fake_entry_points(monkeypatch):
    """Register a fake module for every framework entry point, with a class
    or module-level function per attribute."""
    for entries in instrumentation._ENTRY_POINTS.values():
        for module_name, class_name, attributes, _ in entries:
            module = sys.modules.get(module_name)
            if module is None or not getattr(module, "sasy_test_fake", False):
                module = ModuleType(module_name)
                module.sasy_test_fake = True
                monkeypatch.setitem(sys.modules, module_name, module)
            functions = {attribute: _entry_function(module_name, attribute) for attribute in attributes}
            if class_name is None:
                for attribute, function in functions.items():
                    setattr(module, attribute, function)
            else:
                setattr(module, class_name, type(class_name, (), {"__module__": module_name, **functions}))


def entry_points(flag):
    """(owner, attribute) of each of *flag*'s faked entry points."""
    return [
        (sys.modules[module_name] if class_name is None else getattr(sys.modules[module_name], class_name), attribute)
        for module_name, class_name, attributes, _ in instrumentation._ENTRY_POINTS[flag]
        for attribute in attributes
    ]


def call(owner, attribute):
    """Call an entry point from this file, as an application would."""
    if isinstance(owner, type):
        instance = owner()
        result = getattr(instance, attribute)("go", key="value")
        expected_args = (instance, "go")
    else:
        result = getattr(owner, attribute)("go", key="value")
        expected_args = ("go",)
    if inspect.iscoroutine(result):
        result = asyncio.run(_await(result))
    module_name = owner.__module__ if isinstance(owner, type) else owner.__name__
    assert result == (module_name, attribute, expected_args, {"key": "value"})


async def _await(coroutine):
    return await coroutine


@pytest.fixture
def hidden():
    """Framework modules that cannot be found even when their framework is
    installed."""
    return set()


@pytest.fixture
def frameworks(monkeypatch, hidden):
    """Record each install; ``installed`` is the set of frameworks present."""
    calls = []
    installed = set(MODULES)
    checks = {}

    real = instrumentation.importlib.util.find_spec

    def find_spec(name, *args):
        owners = [key for key, modules in MODULES.items() if name in modules]
        if not owners:
            return real(name, *args)
        return object() if owners[0] in installed and name not in hidden else None

    def adapter(module_name, install_name, key):
        module = ModuleType(module_name)
        setattr(module, install_name, lambda: calls.append(key))

        def check_supported():
            if key in checks:
                raise checks[key]
        module.check_supported = check_supported
        monkeypatch.setitem(sys.modules, module_name, module)

    monkeypatch.setattr(instrumentation.importlib.util, "find_spec", find_spec)
    adapter("sasy.instrumentation.adk", "instrument", "adk")
    adapter("sasy.instrumentation.langchain", "instrument_langchain", "langchain")
    # The frameworks' public entry points, hooked when an adapter is skipped,
    # and the Langroid modules the Langroid adapter loads, whether or not the
    # frameworks are really installed here.
    fake_entry_points(monkeypatch)
    for name in instrumentation._LANGROID_MODULES:
        if name not in sys.modules or not getattr(sys.modules[name], "sasy_test_fake", False):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    # The first-use warning is given once per process; each test is a process.
    monkeypatch.setattr(instrumentation, "_warned", set())
    monkeypatch.setattr(instrumentation, "_hooked", set())
    monkeypatch.setattr(instrumentation, "_instrument_http", lambda: calls.append("http"))
    monkeypatch.setattr(instrumentation, "instrument_langroid", lambda: calls.append("langroid"))
    yield calls, installed, checks


@pytest.fixture
def no_warnings():
    """Fail on any warning raised inside the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.parametrize(("options", "present", "expected"), [
    # Auto-detection: every installed framework, and nothing else.
    ({}, {"adk", "langchain", "langroid"}, ["adk", "langchain", "langroid"]),
    ({}, set(), []),
    ({}, {"adk"}, ["adk"]),
    ({}, {"langchain"}, ["langchain"]),
    ({}, {"langroid"}, ["langroid"]),
    # HTTP routing only when asked for.
    ({"http": True}, {"adk", "langchain", "langroid"}, ["adk", "langchain", "http", "langroid"]),
    ({"http": True}, set(), ["http"]),
    # ADK no longer turns Langroid off: both are enabled when both are there.
    ({"adk": True}, {"adk", "langroid"}, ["adk", "langroid"]),
    # False skips an installed framework.
    ({"adk": False, "langchain": False, "langroid": False}, {"adk", "langchain", "langroid"}, []),
    ({"adk": False, "langchain": False, "langroid": False, "http": True}, {"adk", "langchain", "langroid"}, ["http"]),
    ({"langchain": False, "http": False}, {"adk", "langchain", "langroid"}, ["adk", "langroid"]),
    # True of an installed framework enables it.
    ({"langchain": True, "adk": False, "langroid": False}, {"adk", "langchain", "langroid"}, ["langchain"]),
])
def test_framework_selection(frameworks, no_warnings, options, present, expected):
    calls, installed, _ = frameworks
    installed.intersection_update(present)
    instrumentation.instrument(**options)
    assert calls == expected


def test_http_is_off_by_default(frameworks):
    calls, installed, _ = frameworks
    installed.clear()
    instrumentation.instrument()
    assert "http" not in calls
    instrumentation.instrument(http=True)
    assert calls == ["http"]


@pytest.mark.parametrize("flag", ["adk", "langchain", "langroid"])
def test_required_framework_must_be_installed(frameworks, flag):
    calls, installed, _ = frameworks
    installed.discard(flag)
    with pytest.raises(RuntimeError, match=rf"{flag}=True, but .* is not installed"):
        instrumentation.instrument(**{flag: True})
    assert calls == []


def test_framework_not_installed_is_silent_in_auto_mode(frameworks, no_warnings, caplog):
    calls, installed, _ = frameworks
    installed.clear()
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        instrumentation.instrument(http=True)
    assert calls == ["http"]
    assert caplog.records == []


@pytest.mark.parametrize("flag", ["adk", "langchain"])
def test_installed_but_unsupported_version_raises_when_required(frameworks, flag):
    calls, _, checks = frameworks
    checks[flag] = RuntimeError(f"{flag} 0.0 is installed; this adapter supports exactly 9.9.")
    with pytest.raises(RuntimeError, match=rf"supports exactly 9\.9\..*pass {flag}=False to sasy\.instrument\(\)"):
        instrumentation.instrument(http=True, **{flag: True})
    # The check runs before any patch, including HTTP, is installed.
    assert calls == []


def unsupported(frameworks, monkeypatch, flag):
    """Make *flag*'s adapter unusable: a version it does not support for ADK
    and LangChain, a Langroid module it cannot import for Langroid (whose
    public entry points still import)."""
    _, _, checks = frameworks
    if flag == "langroid":
        monkeypatch.setitem(sys.modules, "langroid.agent.chat_document", None)
    else:
        checks[flag] = RuntimeError(f"{flag} 0.0 is installed; this adapter supports exactly 9.9.")


@contextlib.contextmanager
def silent(caplog):
    """Fail on any warning, or any record logged at WARNING or above."""
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield
    assert caplog.records == []


@pytest.mark.parametrize("flag", ["adk", "langchain", "langroid"])
def test_unusable_adapter_is_skipped_silently_at_instrument_time(frameworks, monkeypatch, caplog, flag):
    """An application that has the framework installed but never uses it is
    not warned."""
    calls, _, _ = frameworks
    unsupported(frameworks, monkeypatch, flag)
    with silent(caplog):
        instrumentation.instrument(http=True)
    # The other frameworks and HTTP are still instrumented; this one is not.
    assert calls == [name for name in ("adk", "langchain", "http", "langroid") if name != flag]


@pytest.mark.parametrize(("flag", "index"), [
    (flag, index) for flag, entries in instrumentation._ENTRY_POINTS.items()
    for index in range(sum(len(attributes) for _, _, attributes, _ in entries))
])
def test_first_use_of_a_skipped_framework_warns(frameworks, monkeypatch, caplog, flag, index):
    unsupported(frameworks, monkeypatch, flag)
    with silent(caplog):
        instrumentation.instrument()
    hooked = entry_points(flag)
    owner, attribute = hooked[index]
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        with pytest.warns(SasyInstrumentationWarning) as caught:
            # The hook calls through: call() checks the original's result.
            call(owner, attribute)
    [warning] = caught
    message = str(warning.message)
    if flag == "langroid":
        assert "adapter cannot load with it" in message
    else:
        assert f"{flag} 0.0 is installed; this adapter supports exactly 9.9." in message
    assert "will NOT be checked by SASY" in message
    assert f"Pass {flag}=True to sasy.instrument() to make this an error" in message
    assert f"or {flag}=False if this application does not use" in message
    assert "silences this warning" in message
    # It points at the application's call, not at SASY's or the framework's code.
    assert warning.filename == __file__
    # And it is logged.
    assert [record.getMessage() for record in caplog.records] == [message]
    caplog.clear()
    # Once per process: no entry point warns again, and calling instrument()
    # again neither re-warns nor hooks twice.
    functions = [getattr(owner, attribute) for owner, attribute in hooked]
    with silent(caplog):
        for owner, attribute in hooked:
            call(owner, attribute)
        instrumentation.instrument()
    assert [getattr(owner, attribute) for owner, attribute in hooked] == functions


def test_every_entry_point_of_a_skipped_framework_is_hooked(frameworks, monkeypatch):
    for flag in ("adk", "langchain", "langroid"):
        unsupported(frameworks, monkeypatch, flag)
    originals = {flag: [getattr(owner, attribute) for owner, attribute in entry_points(flag)]
                 for flag in ("adk", "langchain", "langroid")}
    instrumentation.instrument()
    for flag, functions in originals.items():
        for (owner, attribute), original in zip(entry_points(flag), functions, strict=True):
            hook = inspect.getattr_static(owner, attribute)
            assert hook is not original and hook.__wrapped__ is original
            # An async entry point stays a coroutine function.
            assert inspect.iscoroutinefunction(hook) == inspect.iscoroutinefunction(original)
    # Each skipped framework warns on its own first use.
    for flag in ("adk", "langchain", "langroid"):
        owner, attribute = entry_points(flag)[0]
        with pytest.warns(SasyInstrumentationWarning, match=rf"{flag}=False"):
            call(owner, attribute)


def test_a_function_bound_to_two_names_gets_one_hook(frameworks, monkeypatch):
    """``langchain.agents`` re-exports ``langchain.agents.factory.create_agent``."""
    unsupported(frameworks, monkeypatch, "langchain")
    factory = sys.modules["langchain.agents.factory"]
    monkeypatch.setattr(sys.modules["langchain.agents"], "create_agent", factory.create_agent)
    instrumentation.instrument()
    assert sys.modules["langchain.agents"].create_agent is factory.create_agent
    assert hasattr(factory.create_agent, "__wrapped__")


def test_create_agent_imported_before_instrument_warns(frameworks, monkeypatch, caplog):
    """``from langchain.agents import create_agent`` before sasy.instrument(),
    as examples/langchain-information-flow/demo.py does, gets the hook too."""
    unsupported(frameworks, monkeypatch, "langchain")
    original = sys.modules["langchain.agents"].create_agent
    application = ModuleType("sasy_test_application")
    application.create_agent = original
    monkeypatch.setitem(sys.modules, application.__name__, application)
    # The skipped adapter's own module keeps the original.
    adapter = sys.modules["sasy.instrumentation.langchain"]
    monkeypatch.setattr(adapter, "_create_agent", original, raising=False)
    with silent(caplog):
        instrumentation.instrument()
    assert application.create_agent is sys.modules["langchain.agents"].create_agent
    assert application.create_agent.__wrapped__ is original
    assert adapter._create_agent is original
    with pytest.warns(SasyInstrumentationWarning, match=r"langchain=False") as caught:
        assert application.create_agent("go") == ("langchain.agents", "create_agent", ("go",), {})
    assert [warning.filename for warning in caught] == [__file__]


def _plain(log, n):
    return ("plain", n)


async def _coroutine(log, n):
    return ("coroutine", n)


def _generator(log, n):
    try:
        for value in range(n):
            try:
                log.append(("sent", (yield value)))
            except ValueError as error:
                log.append(("thrown", str(error)))
    finally:
        log.append("closed")
    return "done"


async def _async_generator(log, n):
    try:
        for value in range(n):
            try:
                log.append(("sent", (yield value)))
            except ValueError as error:
                log.append(("thrown", str(error)))
    finally:
        log.append("closed")


def _kind(function):
    return (inspect.iscoroutinefunction(function), inspect.isasyncgenfunction(function),
            inspect.isgeneratorfunction(function))


def _drive(function):
    """Use *function* from this file as an application would: send, throw
    and close through a generator, and return what came back and what
    *function* itself saw."""
    log = []
    if inspect.isgeneratorfunction(function):
        generator = function(log, 4)
        seen = [next(generator), generator.send("a"), generator.throw(ValueError("boom")), generator.send("b")]
        with pytest.raises(StopIteration) as stop:
            generator.send("c")
        seen.append(stop.value.value)
        early = function(log, 3)
        seen.append(next(early))
        early.close()
        return seen, log

    async def drive_async():
        if inspect.isasyncgenfunction(function):
            generator = function(log, 4)
            seen = [await generator.__anext__(), await generator.asend("a"),
                    await generator.athrow(ValueError("boom")), await generator.asend("b")]
            with pytest.raises(StopAsyncIteration):
                await generator.asend("c")
            early = function(log, 3)
            seen.append(await early.__anext__())
            await early.aclose()
            seen.append([value async for value in function(log, 2)])
            return seen
        return await function(log, 3)

    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        return asyncio.run(drive_async()), log
    return function(log, 3), log


@pytest.mark.parametrize("original", [_plain, _coroutine, _generator, _async_generator])
def test_first_use_hook_keeps_the_function_kind(monkeypatch, caplog, original):
    """ADK's Runner.run_async and LangGraph's Pregel.astream are async
    generator functions, and Pregel.stream and Runner.run generator
    functions: the hook is the same kind, passes every value through both
    ways, and warns once."""
    monkeypatch.setattr(instrumentation, "_warned", set())
    hook = instrumentation._first_use_hook("adk", "adk is unchecked", original)
    assert hook.__wrapped__ is original
    assert _kind(hook) == _kind(original)
    with pytest.warns(SasyInstrumentationWarning, match="adk is unchecked") as caught:
        result = _drive(hook)
    assert result == _drive(original)
    # It points at the application's use, not at SASY's code.
    assert [warning.filename for warning in caught] == [__file__]
    caplog.clear()
    with silent(caplog):
        assert _drive(hook) == result


def test_supported_framework_entry_points_are_not_hooked(frameworks, no_warnings):
    originals = {flag: [getattr(owner, attribute) for owner, attribute in entry_points(flag)]
                 for flag in ("adk", "langchain", "langroid")}
    instrumentation.instrument()
    for flag, functions in originals.items():
        assert [getattr(owner, attribute) for owner, attribute in entry_points(flag)] == functions


@pytest.mark.parametrize("flag", ["adk", "langchain", "langroid"])
def test_false_after_auto_mode_silences_the_first_use_warning(frameworks, monkeypatch, caplog, flag):
    unsupported(frameworks, monkeypatch, flag)
    instrumentation.instrument()
    instrumentation.instrument(**{flag: False})
    with silent(caplog):
        for owner, attribute in entry_points(flag):
            call(owner, attribute)


@pytest.mark.parametrize("flag", ["adk", "langchain"])
def test_false_silences_an_unsupported_framework(frameworks, no_warnings, flag):
    calls, _, checks = frameworks
    checks[flag] = RuntimeError(f"{flag} 0.0 is installed; this adapter supports exactly 9.9.")
    instrumentation.instrument(**{flag: False})
    assert flag not in calls


def test_unsupported_version_keeps_the_adapter_error_type(frameworks):
    calls, _, checks = frameworks

    class AdkInstrumentationError(RuntimeError):
        pass

    checks["adk"] = AdkInstrumentationError("google-adk 0.0 is installed")
    with pytest.raises(AdkInstrumentationError, match="adk=False"):
        instrumentation.instrument(adk=True)
    assert calls == []


def test_only_the_version_check_failure_is_downgraded(frameworks):
    """An error that is not the adapter's version check is not turned into a
    warning."""
    calls, _, checks = frameworks
    checks["adk"] = ValueError("not a version problem")
    with pytest.raises(ValueError, match="not a version problem"):
        instrumentation.instrument()
    assert calls == []


@pytest.mark.parametrize("flag", ["adk", "langchain", "langroid"])
def test_installed_framework_the_adapter_cannot_import_raises_when_required(frameworks, monkeypatch, flag):
    calls, _, _ = frameworks
    # A version whose internals the adapter's imports do not find.
    monkeypatch.setitem(sys.modules, ADAPTER_IMPORTS[flag], None)
    with pytest.raises(RuntimeError, match=rf"adapter cannot load with it.*pass {flag}=False"):
        instrumentation.instrument(http=True, **{flag: True})
    assert calls == []


@pytest.mark.parametrize("flag", ["adk", "langchain"])
def test_installed_framework_the_adapter_cannot_import_warns_on_first_use(frameworks, monkeypatch, caplog, flag):
    calls, _, _ = frameworks
    monkeypatch.setitem(sys.modules, ADAPTER_IMPORTS[flag], None)
    with silent(caplog):
        instrumentation.instrument(http=True)
    assert calls == [name for name in ("adk", "langchain", "http", "langroid") if name != flag]
    with pytest.warns(SasyInstrumentationWarning, match=rf"adapter cannot load with it.*will NOT be checked by "
                                                         rf"SASY\. Pass {flag}=True .* or {flag}=False"):
        call(*entry_points(flag)[0])


# flag -> the module of the entry point every use of the framework passes
# through; the Langroid one is also a module the Langroid adapter imports.
REQUIRED_ENTRY_POINT = {"adk": "google.adk.runners", "langchain": "langgraph.pregel", "langroid": "langroid.agent.task"}


@pytest.mark.parametrize("flag", ["adk", "langchain", "langroid"])
def test_entry_points_that_cannot_be_imported_warn_at_instrument_time(frameworks, monkeypatch, caplog, flag):
    """A broken install, where even the public entry points cannot be loaded
    to hook: the warning cannot wait for first use, so it is given at once."""
    calls, _, _ = frameworks
    monkeypatch.setitem(sys.modules, ADAPTER_IMPORTS[flag], None)
    monkeypatch.setitem(sys.modules, REQUIRED_ENTRY_POINT[flag], None)
    with caplog.at_level(logging.WARNING, logger=instrumentation.__name__):
        with pytest.warns(SasyInstrumentationWarning) as caught:
            instrumentation.instrument(http=True)
    [warning] = caught
    message = str(warning.message)
    assert "adapter cannot load with it" in message
    assert f"will NOT be checked by SASY. Pass {flag}=True" in message
    assert f"or {flag}=False" in message
    # It points at the application's call to instrument().
    assert warning.filename == __file__
    assert [record.getMessage() for record in caplog.records] == [message]
    assert calls == [name for name in ("adk", "langchain", "http", "langroid") if name != flag]
    caplog.clear()
    # Once per process.
    with silent(caplog):
        instrumentation.instrument()


@pytest.mark.parametrize("explicit", [None, True])
def test_langgraph_alone_selects_the_langchain_adapter(frameworks, hidden, explicit):
    """LangGraph does not depend on the ``langchain`` package; a LangGraph-only
    application still has ToolNode tool calls to check."""
    calls, installed, _ = frameworks
    hidden.add("langchain")
    installed.intersection_update({"langchain"})
    instrumentation.instrument(langchain=explicit)
    assert calls == ["langchain"]


@pytest.fixture
def langgraph_without_langchain(frameworks, hidden, monkeypatch):
    """LangGraph installed without the ``langchain`` package, so the adapter
    cannot import."""
    hidden.add("langchain")
    monkeypatch.setitem(sys.modules, "sasy.instrumentation.langchain", None)

    def version(distribution):
        raise instrumentation.PackageNotFoundError(distribution)

    monkeypatch.setattr(instrumentation, "version", version)
    return frameworks[0]


def test_langgraph_without_langchain_package_raises_an_actionable_error(langgraph_without_langchain):
    """Required, the adapter that cannot import is an error naming the way out,
    not a silently skipped backstop."""
    with pytest.raises(RuntimeError, match=r"langchain is not installed.*sasy\[langchain\].*"
                                           r"LangChain or LangGraph, pass langchain=False"):
        instrumentation.instrument(langchain=True)
    assert langgraph_without_langchain == []


def test_langgraph_without_langchain_package_warns_on_first_graph_run(langgraph_without_langchain, monkeypatch,
                                                                      caplog):
    for name in ("langchain.agents", "langchain.agents.factory"):
        monkeypatch.setitem(sys.modules, name, None)
    with silent(caplog):
        instrumentation.instrument()
    # The frameworks the adapters do load with are still instrumented.
    assert langgraph_without_langchain == ["adk", "langroid"]
    pregel = sys.modules["langgraph.pregel"].Pregel
    with pytest.warns(SasyInstrumentationWarning, match=r"langchain is not installed.*sasy\[langchain\].*"
                                                         r"LangChain or LangGraph agents in this process will "
                                                         r"NOT be checked by SASY"):
        call(pregel, "invoke")


def test_langchain_false_skips_langgraph_too(frameworks, hidden, no_warnings):
    calls, installed, _ = frameworks
    hidden.add("langchain")
    installed.intersection_update({"langchain"})
    instrumentation.instrument(langchain=False)
    assert calls == []


def test_real_find_spec_of_missing_parent_package_means_not_installed():
    assert instrumentation._installed("sasy_no_such_package.adk") is False
    assert instrumentation._installed("sasy") is True
