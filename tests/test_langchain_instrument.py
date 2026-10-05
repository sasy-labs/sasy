"""``sasy.instrument()`` for LangChain: the patched ``create_agent`` and the
LangGraph ``ToolNode`` backstop.

The patches are process-wide and permanent, so every test installs them itself
(installing is idempotent) and none relies on their being absent. LangChain's
original ``create_agent`` is taken from the adapter, which keeps it, and only
at call time: installing rebinds every module global that holds it, this
module's included.
"""
import asyncio
import importlib.util
import inspect
import os
import sys
import warnings
from types import ModuleType

import pytest

if os.environ.get("SASY_REQUIRE_FRAMEWORKS") == "1":
    import langchain  # noqa: F401
else:
    pytest.importorskip("langchain")
import langchain.agents
import langchain.agents.factory
import sasy.instrumentation as instrumentation
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from sasy.instrumentation import langchain as adapter
from sasy.instrumentation.session import GLOBAL_SESSION, _session_id_var
from sasy.proto.observability_pb2 import LLM, SYSTEM

from tests.test_langchain_instrumentation import ScriptedModel, parents, recorded
from tests.test_langchain_instrumentation import boundary as boundary


def original(*args, **kwargs):
    """LangChain's own create_agent, as it was before any patch."""
    return adapter._create_agent(*args, **kwargs)


def model(name="send", args=None):
    return ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{"name": name, "args": args or {"value": 1}, "id": "call-1", "type": "tool_call"}]),
        AIMessage(content="done"),
    ])


# What every refusal of a ToolNode SASY did not build says.
FOREIGN = (r"Inside a SASY session, tools run only in agents SASY builds: call sasy\.instrument\(\) "
           r"before building the agent with langchain\.agents\.create_agent, or use "
           r"sasy\.instrumentation\.langchain\.create_agent\. Custom LangGraph graphs are not "
           r"supported inside a session yet\.")


def run(agent, asynchronous):
    inputs = {"messages": [HumanMessage("go")]}
    return asyncio.run(agent.ainvoke(inputs)) if asynchronous else agent.invoke(inputs)


@pytest.fixture
def instrumented():
    adapter.instrument_langchain()
    return adapter._langchain_create_agent


@pytest.fixture
def no_session():
    token = _session_id_var.set(None)
    yield
    _session_id_var.reset(token)


# 2a: the patched create_agent -------------------------------------------------

@pytest.mark.parametrize("source", ["langchain.agents", "langchain.agents.factory"])
def test_patched_create_agent_builds_the_sasy_agent(boundary, instrumented, source):
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    create_agent = sys.modules[source].create_agent
    assert create_agent is instrumented
    agent = create_agent(model(), [send], system_prompt="Rules")
    assert isinstance(agent, CompiledStateGraph)
    output = agent.invoke({"messages": [HumanMessage("go")]})
    assert effects == [1]
    assert [check[:2] for check in boundary.checks] == [("send", {"value": 1})]
    assert output["messages"][-1].content == "done"


def test_explicit_form_still_works(boundary, instrumented):
    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    assert isinstance(adapter.create_agent(model(), [send]), adapter.SasyAgent)


@pytest.mark.parametrize("name", sorted(adapter._LANGCHAIN_OPTIONS))
def test_unrecordable_arguments_are_refused_by_name(boundary, instrumented, name):
    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    value = [object()] if name == "middleware" else True if name == "debug" else object()
    with pytest.raises(ValueError, match=rf"{name}=\.\.\..*docs\.sasy\.ai/integrations/langchain"):
        langchain.agents.create_agent(model(), [send], **{name: value})


def test_arguments_at_their_langchain_defaults_are_accepted(instrumented):
    agent = langchain.agents.create_agent(model(), None, middleware=[], checkpointer=None, debug=False)
    assert isinstance(agent, CompiledStateGraph)


def test_unknown_argument_is_a_type_error(instrumented):
    with pytest.raises(TypeError, match="unexpected keyword argument 'no_such_option'"):
        langchain.agents.create_agent(model(), [], no_such_option=1)


def test_unsupported_model_shape_still_raises(boundary, instrumented):
    with pytest.raises(ValueError, match="BaseChatModel"):
        langchain.agents.create_agent(object(), [])


# Both ways of building a SASY agent: the explicit function and the patched
# LangChain one.
BUILDERS = {
    "explicit": lambda *args, **kwargs: adapter.create_agent(*args, **kwargs),
    "patched": lambda *args, **kwargs: langchain.agents.create_agent(*args, **kwargs),
}


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_a_model_name_is_resolved_as_langchain_resolves_it(boundary, instrumented, monkeypatch, build):
    import langchain.chat_models

    resolved = []

    def init_chat_model(name, *args, **kwargs):
        resolved.append((name, args, kwargs))
        return model()

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", init_chat_model)
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    agent = BUILDERS[build]("openai:gpt-5", [send])
    # Exactly LangChain's own call: the name alone.
    assert resolved == [("openai:gpt-5", (), {})]
    output = agent.invoke({"messages": [HumanMessage("go")]})
    assert effects == [1]
    assert [check[:2] for check in boundary.checks] == [("send", {"value": 1})]
    assert output["messages"][-1].content == "done"


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_a_model_resolved_from_a_name_is_still_validated(boundary, instrumented, monkeypatch, build):
    import langchain.chat_models
    from langchain_openai import ChatOpenAI

    provider = ChatOpenAI(model="synthetic", api_key="synthetic", use_responses_api=True,
                          use_previous_response_id=True)
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda name: provider)
    with pytest.raises(adapter.InstrumentationError, match="Provider-retained"):
        BUILDERS[build]("openai:synthetic", [])
    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda name: object())
    with pytest.raises(ValueError, match="BaseChatModel"):
        BUILDERS[build]("openai:synthetic", [])


def test_a_model_name_resolves_to_the_model_langchain_builds(boundary, instrumented, monkeypatch):
    """With LangChain's real resolver: the agent calls the model LangChain's
    own create_agent would."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    sasy_model = _bound_model(adapter.create_agent("openai:gpt-5", []))
    langchain_model = _bound_model(original("openai:gpt-5", []))
    assert type(sasy_model) is type(langchain_model)
    assert sasy_model.model_name == langchain_model.model_name == "gpt-5"


def _bound_model(agent):
    """The chat model a compiled LangChain agent's model node calls."""
    graph = agent._graph if isinstance(agent, adapter.SasyAgent) else agent
    from langchain_core.language_models import BaseChatModel

    for node in graph.nodes.values():
        closure = getattr(getattr(node.bound, "func", None), "__closure__", None) or ()
        for cell in closure:
            if isinstance(cell.cell_contents, BaseChatModel):
                return cell.cell_contents
    raise AssertionError("no chat model found in the compiled graph")


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_name_reaches_the_compiled_graph(instrumented, build):
    agent = BUILDERS[build](model(), [], name="planner")
    assert (agent._graph if isinstance(agent, adapter.SasyAgent) else agent).name == "planner" == original(model(), [], name="planner").name
    # Unnamed, it is LangChain's default name.
    unnamed = BUILDERS[build](model(), [])
    assert (unnamed._graph if isinstance(unnamed, adapter.SasyAgent) else unnamed).name == original(model(), []).name


def test_name_survives_other_options_at_their_defaults(boundary, instrumented):
    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    agent = langchain.agents.create_agent(model(), [send], name="planner", debug=False, checkpointer=None, middleware=[])
    assert agent.name == "planner"
    # Unnamed, an option at its default leaves LangChain's default name.
    assert langchain.agents.create_agent(model(), [], checkpointer=None).name == original(model(), []).name
    agent.invoke({"messages": [HumanMessage("go")]})
    answers = [event for event, _ in boundary.events.values() if event.role == LLM]
    assert answers and {event.agent for event in answers} == {"planner"}


def _system_prompt_record(boundary, build, prompt):
    """What one run records for *prompt*: the system node the first model call
    read, less its per-run identity, and the shape of that call's inputs."""
    boundary.events.clear()
    boundary.edges.clear()
    boundary.checks.clear()

    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    BUILDERS[build](model(), [send], system_prompt=prompt).invoke({"messages": [HumanMessage("go")]})
    first_answer = next(node for node, (event, _) in boundary.events.items() if event.tools)
    inputs = sorted(parents(boundary, first_answer), key=lambda node: boundary.events[node][0].role)
    systems = [boundary.events[node][0] for node in inputs if boundary.events[node][0].role == SYSTEM]
    shape = [(boundary.events[node][0].role, boundary.events[node][0].text) for node in inputs]
    [system] = systems
    message = recorded(system)
    identity = message.pop("id")
    assert identity
    return (system.role, system.text, system.agent, list(system.tools), system.HasField("derived_from"), message,
            shape, [check[:2] for check in boundary.checks], len(boundary.events))


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_a_system_message_prompt_is_recorded_as_its_string_is(boundary, instrumented, build):
    as_string = _system_prompt_record(boundary, build, "Routing rules")
    as_message = _system_prompt_record(boundary, build, SystemMessage("Routing rules"))
    assert as_message == as_string
    assert as_string[1] == "Routing rules"


def test_a_system_message_prompt_is_fixed_when_the_agent_is_built(boundary, instrumented):
    prompt = SystemMessage("Routing rules")
    agent = adapter.create_agent(model(), [], system_prompt=prompt)
    prompt.content = "Changed later"
    agent.invoke({"messages": [HumanMessage("go")]})
    texts = {event.text for event, _ in boundary.events.values()}
    assert "Routing rules" in texts and "Changed later" not in texts
    # The caller's message is left as it was given.
    assert prompt.id is None


@pytest.mark.parametrize("build", sorted(BUILDERS))
def test_a_system_prompt_of_another_type_is_refused(boundary, instrumented, build):
    with pytest.raises(ValueError, match="system_prompt must be a string or a SystemMessage"):
        BUILDERS[build](model(), [], system_prompt=["Routing rules"])


# 2b: names bound before instrument() -----------------------------------------

def test_module_that_imported_create_agent_early_gets_the_sasy_version(monkeypatch):
    before = adapter._create_agent
    early = ModuleType("sasy_test_early_langchain_import")
    early.create_agent = before
    early.alias = before
    early.unrelated = len
    monkeypatch.setitem(sys.modules, early.__name__, early)
    adapter.instrument_langchain()
    assert early.create_agent is adapter._langchain_create_agent
    assert early.alias is adapter._langchain_create_agent
    assert early.unrelated is len
    # The adapter keeps the original it builds with.
    assert adapter._create_agent is before


def test_instrument_installs_the_langchain_adapter():
    instrumentation.instrument(http=False, adk=False, langroid=False, langchain=True)
    assert langchain.agents.create_agent is adapter._langchain_create_agent
    assert langchain.agents.factory.create_agent is adapter._langchain_create_agent
    # Idempotent: the ToolNode methods are wrapped once.
    run_one = ToolNode._run_one
    instrumentation.instrument(http=False, adk=False, langroid=False, langchain=True)
    assert ToolNode._run_one is run_one


def test_unpinned_langchain_is_skipped_and_warns_on_first_graph_run(monkeypatch):
    """With the real LangChain and LangGraph: nothing at instrument() time, a
    warning when a graph first runs, and the graph's own result."""
    monkeypatch.setattr(adapter, "version", lambda package: "0.0.1")
    installed = []
    monkeypatch.setattr(adapter, "instrument_langchain", lambda: installed.append("langchain"))
    # The first-use hooks are permanent; put the real entry points back after.
    monkeypatch.setattr(instrumentation, "_warned", set())
    monkeypatch.setattr(instrumentation, "_hooked", set())
    for module_name, class_name, attributes, _ in instrumentation._ENTRY_POINTS["langchain"]:
        owner = sys.modules[module_name] if class_name is None else getattr(sys.modules[module_name], class_name)
        for attribute in attributes:
            monkeypatch.setattr(owner, attribute, inspect.getattr_static(owner, attribute))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        instrumentation.instrument(adk=False, langroid=False)
    assert installed == []
    assert langchain.agents.create_agent.__wrapped__ is langchain.agents.factory.create_agent.__wrapped__

    def node(state):
        return {"messages": [AIMessage("done")]}

    graph = StateGraph(MessagesState)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    with pytest.warns(instrumentation.SasyInstrumentationWarning,
                      match=r"langchain 0\.0\.1 is installed.*will NOT be checked by SASY.*langchain=True.*langchain=False"
                      ) as caught:
        result = compiled.invoke({"messages": [HumanMessage("go")]})
    assert [message.content for message in result["messages"]] == ["go", "done"]
    assert [warning.filename for warning in caught] == [__file__]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = asyncio.run(compiled.ainvoke({"messages": [HumanMessage("again")]}))
        assert [message.content for message in result["messages"]] == ["again", "done"]
        instrumentation.instrument(adk=False, langroid=False)


def test_unpinned_langchain_raises_when_required(monkeypatch):
    monkeypatch.setattr(adapter, "version", lambda package: "0.0.1")
    with pytest.raises(RuntimeError, match=r"langchain 0\.0\.1 is installed.*pass langchain=False"):
        instrumentation.instrument(adk=False, langroid=False, langchain=True)


def test_lazily_loaded_modules_are_not_run(tmp_path, monkeypatch):
    """Rebinding reads module namespaces without running a LazyLoader module."""
    source = tmp_path / "sasy_test_lazy_module.py"
    source.write_text("raise RuntimeError('the lazy module ran')\n")
    spec = importlib.util.spec_from_file_location("sasy_test_lazy_module", source)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    loader.exec_module(module)
    adapter.instrument_langchain()
    # Still deferred: loading replaces the lazy class with ModuleType.
    assert type(module) is not ModuleType


def test_objects_that_only_claim_to_be_modules_are_skipped(monkeypatch):
    """Like cffi's Lib: registered in sys.modules, reporting ModuleType as its
    class, with attributes it computes itself."""
    class Imposter:
        @property
        def __class__(self):
            return ModuleType

        @property
        def __dict__(self):
            raise RuntimeError("computed attribute failed")

    monkeypatch.setitem(sys.modules, "sasy_test_imposter_module", Imposter())
    adapter.instrument_langchain()


# 2c: the ToolNode backstop ----------------------------------------------------

@pytest.mark.parametrize("asynchronous", [False, True])
def test_unprotected_agent_refuses_its_first_tool_call_in_a_session(boundary, instrumented, asynchronous):
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    async def asend(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    name = "asend" if asynchronous else "send"
    agent = original(model(name), [asend if asynchronous else send])
    with pytest.raises(adapter.InstrumentationError, match=rf"'{name}' comes from a LangGraph ToolNode that SASY did not build.*{FOREIGN}"):
        run(agent, asynchronous)
    assert effects == []
    assert boundary.checks == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unprotected_agent_refuses_its_first_tool_call_in_the_global_session(instrumented, asynchronous):
    """sasy.global_session binds a tenant-wide policy; an unprotected tool is
    refused there too."""
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    token = _session_id_var.set(GLOBAL_SESSION)
    try:
        with pytest.raises(adapter.InstrumentationError, match=f"'send' comes from.*{FOREIGN}"):
            run(original(model(), [send]), asynchronous)
    finally:
        _session_id_var.reset(token)
    assert effects == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unprotected_agent_runs_normally_outside_a_session(instrumented, no_session, asynchronous):
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    output = run(original(model(), [send]), asynchronous)
    assert effects == [1]
    assert output["messages"][-1].content == "done"


def test_unknown_tool_name_is_refused_in_a_session(boundary, instrumented):
    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    with pytest.raises(adapter.InstrumentationError, match=f"'missing' comes from.*{FOREIGN}"):
        original(model("missing"), [send]).invoke({"messages": [HumanMessage("go")]})


def test_unknown_tool_name_is_left_to_the_tool_node_outside_a_session(instrumented, no_session):
    def send(value: int) -> str:
        """Send a synthetic value."""
        return "sent"

    output = original(model("missing"), [send]).invoke({"messages": [HumanMessage("go")]})
    result = next(message for message in output["messages"] if isinstance(message, ToolMessage))
    assert result.status == "error" and "missing" in result.content


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_refusal_is_not_turned_into_a_tool_message(boundary, instrumented, asynchronous, wrapped):
    """A ToolNode that converts every tool error still raises; no wrapper or tool body runs.

    With a tool-call wrapper, the node converts whatever the wrapper raises, so
    the refusal has to come before the node's own handling, not from inside it.
    """
    effects = []
    wrapper_calls = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    def passthrough(request, handler):
        wrapper_calls.append(request)
        return handler(request)

    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([send], handle_tool_errors=True, wrap_tool_call=passthrough if wrapped else None))
    graph.add_edge(START, "tools")
    compiled = graph.compile()
    inputs = {"messages": [AIMessage("", tool_calls=[{"name": "send", "args": {"value": 1}, "id": "c", "type": "tool_call"}])]}
    with pytest.raises(adapter.InstrumentationError, match=FOREIGN):
        asyncio.run(compiled.ainvoke(inputs)) if asynchronous else compiled.invoke(inputs)
    assert effects == []
    assert wrapper_calls == []


@pytest.mark.parametrize("marked", [False, True])
def test_tool_substituted_by_a_middleware_is_refused(boundary, instrumented, marked):
    """An agent SASY did not build is refused at the node. On a node counted as
    SASY's own (marked here, since SASY never builds one with a middleware of
    the caller's), the second line of defence refuses the substituted tool when
    it is about to run."""
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(("send", value))
        return "sent"

    def other(value: int) -> str:
        """An unprotected tool."""
        effects.append(("other", value))
        return "other"

    from langchain_core.tools import StructuredTool
    protected = adapter._gate(StructuredTool.from_function(send))
    replacement = StructuredTool.from_function(other, name="send")

    class Swap(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            return handler(request.override(tool=replacement))

    agent = original(model(), [protected], middleware=[Swap()])
    if marked:
        adapter._mark_tool_nodes(agent, expected=True)
    with pytest.raises(adapter.InstrumentationError, match="'send' is not protected by SASY" if marked else FOREIGN):
        agent.invoke({"messages": [HumanMessage("go")]})
    assert effects == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sasy_agents_are_unaffected(boundary, instrumented, asynchronous):
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    output = run(adapter.create_agent(model(), [send]), asynchronous)
    assert effects == [1]
    assert [check[:2] for check in boundary.checks] == [("send", {"value": 1})]
    assert output["messages"][-1].content == "done"


def _tool_node_graph(node):
    graph = StateGraph(MessagesState)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    return graph.compile()


def _dispatch(name):
    return {"messages": [AIMessage("", tool_calls=[{"name": name, "args": {"value": 1}, "id": "c", "type": "tool_call"}])]}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_wrapper_cannot_run_a_dynamic_tool_in_a_session(boundary, instrumented, asynchronous):
    """A tool-call wrapper that runs a tool of its own for a name the node does
    not know would never reach the guarded execution methods; the call is
    refused before the wrapper runs. The synchronous case has only a sync
    wrapper; the asynchronous case has both."""
    effects = []

    def dynamic(value: int) -> str:
        """An unprotected tool the wrapper runs itself."""
        effects.append(value)
        return "ran"

    def run_it(request):
        return ToolMessage(dynamic(**request.tool_call["args"]), tool_call_id=request.tool_call["id"])

    def wrap(request, handler):
        return run_it(request)

    async def awrap(request, handler):
        return run_it(request)

    node = ToolNode([], handle_tool_errors=True, wrap_tool_call=wrap, awrap_tool_call=awrap if asynchronous else None)
    compiled = _tool_node_graph(node)
    with pytest.raises(adapter.InstrumentationError, match=f"'dynamic' comes from.*{FOREIGN}"):
        asyncio.run(compiled.ainvoke(_dispatch("dynamic"))) if asynchronous else compiled.invoke(_dispatch("dynamic"))
    assert effects == []
    assert boundary.checks == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_wrapper_cannot_run_another_tool_for_a_protected_name(boundary, instrumented, asynchronous):
    """The node knows the name and its tool is SASY-protected, but the node's
    wrapper runs a different tool itself and returns a ToolMessage without
    reaching the guarded handler. The call is refused before the wrapper runs,
    and neither tool body runs."""
    from langchain_core.tools import StructuredTool

    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(("send", value))
        return "sent"

    def other(value: int) -> str:
        """An unprotected tool the wrapper runs itself."""
        effects.append(("other", value))
        return "other"

    def run_other(request):
        return ToolMessage(other(**request.tool_call["args"]), tool_call_id=request.tool_call["id"])

    def wrap(request, handler):
        return run_other(request)

    async def awrap(request, handler):
        return run_other(request)

    protected = adapter._gate(StructuredTool.from_function(send))
    assert adapter._protected(protected)
    node = ToolNode([protected], handle_tool_errors=True, wrap_tool_call=wrap, awrap_tool_call=awrap)
    compiled = _tool_node_graph(node)
    with pytest.raises(adapter.InstrumentationError, match=f"'send' comes from.*{FOREIGN}"):
        asyncio.run(compiled.ainvoke(_dispatch("send"))) if asynchronous else compiled.invoke(_dispatch("send"))
    assert effects == []
    assert boundary.checks == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_sync_call_on_a_node_with_only_an_async_wrapper_is_refused(boundary, instrumented, asynchronous):
    """One rule for every node SASY did not build, whichever wrappers it has."""
    compiled = _tool_node_graph(ToolNode([], awrap_tool_call=lambda request, handler: handler(request)))
    with pytest.raises(adapter.InstrumentationError, match=FOREIGN):
        asyncio.run(compiled.ainvoke(_dispatch("missing"))) if asynchronous else compiled.invoke(_dispatch("missing"))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_unknown_name_without_a_wrapper_is_refused_in_a_session(boundary, instrumented, asynchronous):
    compiled = _tool_node_graph(ToolNode([]))
    inputs = _dispatch("missing")
    with pytest.raises(adapter.InstrumentationError, match=f"'missing' comes from.*{FOREIGN}"):
        asyncio.run(compiled.ainvoke(inputs)) if asynchronous else compiled.invoke(inputs)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_unknown_name_is_left_to_the_tool_node_outside_a_session(instrumented, no_session, asynchronous):
    compiled = _tool_node_graph(ToolNode([]))
    inputs = _dispatch("missing")
    output = asyncio.run(compiled.ainvoke(inputs)) if asynchronous else compiled.invoke(inputs)
    result = next(message for message in output["messages"] if isinstance(message, ToolMessage))
    assert result.status == "error" and "missing" in result.content


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("build", ["explicit", "patched"])
def test_a_tool_the_model_invented_in_a_sasy_agent_is_still_reported(boundary, instrumented, asynchronous, build):
    """SASY's own ToolNode carries SASY's middleware as its wrapper; a name the
    model invented is reported to the model by that middleware, as before."""
    effects = []

    def send(value: int) -> str:
        """Send a synthetic value."""
        effects.append(value)
        return "sent"

    create_agent = adapter.create_agent if build == "explicit" else langchain.agents.create_agent
    output = run(create_agent(model("does_not_exist"), [send]), asynchronous)
    failed = next(message for message in output["messages"] if isinstance(message, ToolMessage))
    assert failed.status == "error" and "no tool named does_not_exist" in failed.content
    assert output["messages"][-1].content == "done"
    assert effects == []
    assert boundary.checks == []
