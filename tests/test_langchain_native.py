"""Native graph entry points must preserve the checked invocation boundary."""
import asyncio
import threading
import time

import pytest

pytest.importorskip("langchain")
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from sasy.instrumentation import langchain as adapter

from tests.test_langchain_instrumentation import (
    ScriptedModel,
    ancestors,
    call,
    confidential_policy,
    content,
    contents,
    node_of,
    read_confidential,
    turn,
)
from tests.test_langchain_instrumentation import boundary as boundary


def build(responses=None, tools=(), model=None):
    adapter.instrument_langchain()
    import langchain.agents
    return langchain.agents.create_agent(model or ScriptedModel(responses=responses or [AIMessage("done")]), tools)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_batch_has_independent_scopes_and_marks(boundary, asynchronous):
    agent = build()
    assert isinstance(agent, CompiledStateGraph)
    assert "model" in agent.get_graph().nodes
    requests = [{"messages": [HumanMessage(text)]} for text in ("one", "two")]
    outputs = asyncio.run(agent.abatch(requests)) if asynchronous else agent.batch(requests)
    assert len(outputs) == 2
    for output, expected in zip(outputs, ("one", "two"), strict=True):
        answer = output["messages"][-1]
        identity = answer.response_metadata["sasy"]["node"]
        direct = {edge.source for edge in boundary.edges if edge.destination == identity}
        assert {content(boundary.events[node][0]) for node in direct} == {expected}
    assert adapter._run.get() is None


def test_with_config_preserves_native_registration(boundary):
    original = build()
    configured = original.with_config(tags=["test"], recursion_limit=12)
    variants = [original, configured, original.copy(), configured.with_config(tags=["second"])]
    for agent in variants:
        output = agent.invoke({"messages": [HumanMessage("go")]})
        assert output["messages"][-1].response_metadata["sasy"]["node"] in boundary.events
        assert isinstance(agent, CompiledStateGraph)
    assert len({id(agent.channels) for agent in variants}) == len(variants)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("per_item_config", [False, True])
@pytest.mark.parametrize("max_concurrency", [1, 2])
def test_batch_delegation_preserves_parent_and_returned_ancestry(boundary, monkeypatch, asynchronous, per_item_config, max_concurrency):
    from langchain_core.runnables.config import var_child_runnable_config

    decisions = confidential_policy(boundary, monkeypatch)
    outputs = []
    published = []

    def publish(text: str) -> str:
        """Publish externally."""
        published.append(text)
        return "published"

    # Both children inherit the confidential dispatch. Their returned answers
    # must also become parents of the checked delegation result.
    class ChildModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            answer = (AIMessage("child answer") if isinstance(messages[-1], ToolMessage)
                      else turn(call("publish", {"text": "child"})))
            return ChatResult(generations=[ChatGeneration(message=answer)])

    child = build(tools=[publish], model=ChildModel(responses=[])).with_config(max_concurrency=max_concurrency)
    requests = [{"messages": [HumanMessage(text)]} for text in ("one", "two")]
    config = [{"tags": ["one"]}, {"tags": ["two"]}] if per_item_config else None

    def delegate() -> str:
        """Delegate a batch of requests."""
        ambient = var_child_runnable_config.get()
        assert ambient and ambient.get("callbacks")
        outputs.extend(child.batch(requests, config))
        assert var_child_runnable_config.get() is ambient
        return "delegated"

    async def adelegate() -> str:
        """Delegate a batch of requests."""
        ambient = var_child_runnable_config.get()
        assert ambient and ambient.get("callbacks")
        outputs.extend(await child.abatch(requests, config))
        assert var_child_runnable_config.get() is ambient
        return "delegated"

    tool = adelegate if asynchronous else delegate
    outer = build([turn(call("read_confidential", id="read")), turn(call(tool.__name__, id="delegate")), AIMessage("done")],
                  [read_confidential, tool])
    request = {"messages": [HumanMessage("go")]}
    asyncio.run(outer.ainvoke(request)) if asynchronous else outer.invoke(request)

    assert decisions.count(("publish", True, False)) == 2
    assert not published
    result = node_of(boundary, lambda event: event.derived_from.name == tool.__name__)
    returned = {output["messages"][-1].response_metadata["sasy"]["node"] for output in outputs}
    assert len(returned) == 2
    assert returned <= ancestors(boundary, [result])
    for output, own, sibling in zip(outputs, ("one", "two"), ("two", "one"), strict=True):
        answer = output["messages"][-1].response_metadata["sasy"]["node"]
        ancestry = contents(boundary, ancestors(boundary, [answer]))
        assert {"BUDGET 42", own} <= ancestry
        assert sibling not in ancestry
    assert adapter._run.get() is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batched_confidential_child_does_not_taint_public_sibling(boundary, monkeypatch, asynchronous):
    decisions = confidential_policy(boundary, monkeypatch)
    published = []

    def publish(text: str) -> str:
        """Publish externally."""
        published.append(text)
        return "published"

    class ChildModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                answer = AIMessage("answer")
            elif messages[-1].content == "private":
                answer = turn(call("read_confidential"))
            else:
                answer = turn(call("publish", {"text": "public"}))
            return ChatResult(generations=[ChatGeneration(message=answer)])

    child = build(tools=[read_confidential, publish], model=ChildModel(responses=[])).with_config(max_concurrency=1)
    requests = [{"messages": [HumanMessage(task)]} for task in ("private", "public")]

    def delegate() -> str:
        """Delegate independent requests."""
        child.batch(requests)
        return "delegated"

    async def adelegate() -> str:
        """Delegate independent requests."""
        await child.abatch(requests)
        return "delegated"

    tool = adelegate if asynchronous else delegate
    outer = build([turn(call(tool.__name__, id="delegate")),
                   turn(call("publish", {"text": "combined"}, id="publish")), AIMessage("done")], [tool, publish])
    request = {"messages": [HumanMessage("go")]}
    asyncio.run(outer.ainvoke(request)) if asynchronous else outer.invoke(request)
    assert ("publish", False, True) in decisions
    assert ("publish", True, False) in decisions
    assert published == ["public"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batch_honors_bound_recursion_limit_and_per_call_override(boundary, asynchronous):
    agent = build().with_config(recursion_limit=1)
    requests = [{"messages": [HumanMessage("go")]}]
    with pytest.raises(GraphRecursionError):
        asyncio.run(agent.abatch(requests)) if asynchronous else agent.batch(requests)
    config = [{"recursion_limit": 3}]
    result = asyncio.run(agent.abatch(requests, config)) if asynchronous else agent.batch(requests, config)
    assert result[0]["messages"][-1].content == "done"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batch_honors_bound_max_concurrency(boundary, asynchronous):
    active, peak = 0, 0
    lock = threading.Lock()

    class MeasuredModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])

    agent = build(model=MeasuredModel(responses=[])).with_config(max_concurrency=1)
    requests = [{"messages": [HumanMessage(str(i))]} for i in range(3)]
    asyncio.run(agent.abatch(requests)) if asynchronous else agent.batch(requests)
    assert peak == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batch_return_exceptions_refused_before_any_item_runs(boundary, asynchronous):
    agent = build()
    requests = [{"messages": [HumanMessage("valid")]}, {"unsupported": "invalid"}]
    with pytest.raises(adapter.InstrumentationError, match="return_exceptions"):
        (asyncio.run(agent.abatch(requests, return_exceptions=True)) if asynchronous
         else agent.batch(requests, return_exceptions=True))
    assert not boundary.events
    assert adapter._run.get() is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("return_exceptions", [False, True])
def test_as_completed_batches_refused_before_any_item_runs(boundary, asynchronous, return_exceptions):
    agent = build()
    requests = [{"messages": [HumanMessage("go")]}]

    async def consume():
        return [item async for item in agent.abatch_as_completed(requests, return_exceptions=return_exceptions)]

    with pytest.raises(adapter.InstrumentationError, match="as-completed"):
        (asyncio.run(consume()) if asynchronous
         else list(agent.batch_as_completed(requests, return_exceptions=return_exceptions)))
    assert not boundary.events


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("per_item_config", [False, True])
@pytest.mark.parametrize("config", [{"callbacks": [object()]}, {"configurable": {"thread_id": "other"}}])
def test_batch_explicit_unsupported_config_is_refused(boundary, asynchronous, per_item_config, config):
    agent = build()
    requests = [{"messages": [HumanMessage("one")]}, {"messages": [HumanMessage("two")]}]
    option = [{}, config] if per_item_config else config
    with pytest.raises(adapter.InstrumentationError, match="config"):
        asyncio.run(agent.abatch(requests, option)) if asynchronous else agent.batch(requests, option)
    assert not boundary.events


@pytest.mark.parametrize("option", [
    {"callbacks": [object()]}, {"configurable": {"thread_id": "other"}}, {"unknown": True},
])
def test_config_cannot_install_untracked_execution(boundary, option):
    agent = build()
    with pytest.raises(adapter.InstrumentationError, match="config"):
        agent.invoke({"messages": [HumanMessage("go")]}, option)
    with pytest.raises(adapter.InstrumentationError, match="config"):
        agent.with_config(option)
    assert not boundary.events


@pytest.mark.parametrize("option", [
    {"context": {"hidden": "input"}}, {"stream_mode": "updates"},
    {"interrupt_before": ["model"]}, {"version": "v2"},
    {"output_keys": ["messages"]}, {"unknown": True},
])
def test_native_unsupported_options_refuse_before_model(boundary, option):
    with pytest.raises(adapter.InstrumentationError, match="invocation"):
        build().invoke({"messages": [HumanMessage("go")]}, **option)
    assert not boundary.events


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_stream_refused_before_model(boundary, asynchronous):
    agent = build()
    async def consume():
        return [item async for item in agent.astream({"messages": [HumanMessage("go")]})]
    with pytest.raises(adapter.InstrumentationError, match="direct streaming"):
        asyncio.run(consume()) if asynchronous else list(agent.stream({"messages": [HumanMessage("go")]}))
    assert not boundary.events


def test_tool_cannot_reuse_the_internal_stream_permission(boundary):
    agent = None
    def recurse() -> str:
        """Try starting another stream from inside a checked tool body."""
        list(agent.stream({"messages": [HumanMessage("hidden")]}))
        return "bad"
    agent = build([AIMessage("", tool_calls=[{"name": "recurse", "args": {}, "id": "r", "type": "tool_call"}])], [recurse])
    with pytest.raises(adapter.InstrumentationError, match="direct streaming"):
        agent.invoke({"messages": [HumanMessage("go")]})
    assert not any(content(e) == "hidden" for e, _ in boundary.events.values())
    assert adapter._run.get() is None


@pytest.mark.parametrize("edit", ["node", "triggers", "writers", "cache", "transformers"])
def test_graph_edits_and_nonconfig_copies_are_refused(boundary, edit):
    agent = build()
    with pytest.raises(adapter.InstrumentationError, match="copies"):
        agent.copy({"checkpointer": object()})
    if edit == "node":
        agent.nodes.pop("model")
    elif edit == "triggers":
        agent.nodes["model"].triggers.append("other")
    elif edit == "writers":
        agent.nodes["model"].writers.append(object())
    elif edit == "cache":
        agent.cache_policy = object()
    else:
        agent.stream_transformers = [object()]
    with pytest.raises(adapter.InstrumentationError, match="modified|not supported"):
        agent.invoke({"messages": [HumanMessage("go")]})
    assert not boundary.events


@pytest.mark.parametrize("asynchronous", [False, True])
def test_factory_outside_session_is_native_then_protected(boundary, asynchronous):
    from sasy.instrumentation.session import _session_id_var
    token = _session_id_var.set(None)
    try:
        agent = build()
        request = {"messages": [HumanMessage("outside")]}
        output = asyncio.run(agent.ainvoke(request)) if asynchronous else agent.invoke(request)
        assert "sasy" not in output["messages"][-1].response_metadata
        assert not boundary.events
        assert list(agent.stream(request))
    finally:
        _session_id_var.reset(token)
    output = asyncio.run(agent.ainvoke(request)) if asynchronous else agent.invoke(request)
    assert output["messages"][-1].response_metadata["sasy"]["node"] in boundary.events


def test_native_options_accepted_outside_rejected_at_protected_entry(boundary):
    import langchain.agents
    from langchain.agents.middleware import AgentMiddleware
    from sasy.instrumentation.session import _session_id_var
    effects = []
    class CustomMiddleware(AgentMiddleware):
        def before_model(self, state, runtime):
            effects.append("callback")
    adapter.instrument_langchain()
    token = _session_id_var.set(None)
    try:
        agent = langchain.agents.create_agent(ScriptedModel(responses=[AIMessage("done")]),
                                              middleware=[CustomMiddleware()])
        agent.invoke({"messages": [HumanMessage("outside")]})
        assert effects == ["callback"]
        assert not boundary.events
    finally:
        _session_id_var.reset(token)
    with pytest.raises(ValueError, match="middleware"):
        agent.invoke({"messages": [HumanMessage("inside")]})
    assert effects == ["callback"]


def test_native_process_default_and_closed_inherited_scope(boundary, monkeypatch):
    import importlib
    from contextvars import copy_context
    sessions = importlib.import_module("sasy.instrumentation.session")
    token = sessions._session_id_var.set(None)
    monkeypatch.setattr(sessions, "_default_session_id", "process-session")
    try:
        agent = build()
        output = agent.invoke({"messages": [HumanMessage("process")]})
        assert output["messages"][-1].response_metadata["sasy"]["node"] in boundary.events
        with sessions.session(end_on_exit=False):
            inherited = copy_context()
        inherited.get(sessions._session_lease_var).closed = True
        with pytest.raises(sessions.SessionScopeError, match="ended"):
            inherited.run(agent.invoke, {"messages": [HumanMessage("late")]})
    finally:
        sessions._session_id_var.reset(token)


def test_lazy_factory_freezes_prompt_and_tool_sequence(boundary):
    import langchain.agents
    from langchain_core.messages import SystemMessage
    from sasy.instrumentation.session import _session_id_var
    seen = []
    class WatchingModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append([message.content for message in messages])
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    adapter.instrument_langchain()
    prompt = SystemMessage("original prompt")
    tools = []
    token = _session_id_var.set(None)
    try:
        graph = langchain.agents.create_agent(WatchingModel(responses=[AIMessage("done")]), tools,
                                              system_prompt=prompt)
    finally:
        _session_id_var.reset(token)
    prompt.content = "mutated prompt"
    tools.append(object())
    graph.invoke({"messages": [HumanMessage("go")]})
    assert seen == [["original prompt", "go"]]
