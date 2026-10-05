"""Framework wrappers accept current wrapt and messages with only tool calls."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import wrapt
from langroid import ChatAgent, ChatAgentConfig, ChatDocument, Entity
from langroid.agent.chat_document import ChatDocMetaData
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    OpenAIToolCall,
    Role,
)
from opentelemetry.trace import NoOpTracerProvider
from sasy.instrumentation import langroid as instrumentation


@pytest.fixture(autouse=True)
def protected_session():
    from sasy.instrumentation.session import session
    with session("langroid-compat", end_on_exit=False):
        yield


@pytest.fixture
def registered_wrappers(monkeypatch):
    wrappers = {}
    signature = inspect.signature(wrapt.wrap_function_wrapper)

    def register(*args, **kwargs):
        # Binding against wrapt itself catches incompatible keyword names
        # without changing global Langroid methods for the rest of the suite.
        bound = signature.bind(*args, **kwargs)
        wrappers[bound.arguments["name"]] = bound.arguments["wrapper"]

    tracer = NoOpTracerProvider().get_tracer(__name__)
    monkeypatch.setattr(instrumentation, "wrap_function_wrapper", register)
    monkeypatch.setattr(instrumentation, "get_tracer", lambda *_: tracer)
    instrumentation.instrument.cache_clear()
    instrumentation.instrument()
    yield wrappers
    instrumentation.instrument.cache_clear()


def test_registers_responder_and_action_wrappers(registered_wrappers):
    assert {
        "ChatAgent.llm_response_messages",
        "ChatAgent.llm_response_messages_async",
        "Agent.handle_tool_message",
        "Agent.handle_tool_message_async",
    } <= registered_wrappers.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_response", [False, True])
@pytest.mark.parametrize("content", [None, "Text accompanying the tool call"])
async def test_tool_calls_survive_nullable_message_content(
    monkeypatch, registered_wrappers, async_response, content
):
    agent = ChatAgent(ChatAgentConfig(llm=None, use_tools=True, use_functions_api=False))
    agent.message_history = [
        LLMMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=[OpenAIToolCall(
                id="tool-call-1", type="function",
                function=LLMFunctionCall(name="lookup", arguments={"query": "example"}),
            )],
        )
    ]
    recorded = []

    def record(snapshots):
        recorded.extend(snapshot.event for snapshot in snapshots)
        return [snapshot.base_id or "canonical-" + snapshot.event.id for snapshot in snapshots]

    async def record_async(snapshots):
        return record(snapshots)

    monkeypatch.setattr(instrumentation, "resolve_events", record)
    monkeypatch.setattr(instrumentation, "resolve_events_async", record_async)
    instrumentation._record_output(agent.message_history[0], [], agent)
    output = ChatDocument(content="Tool result", metadata=ChatDocMetaData(sender=Entity.LLM))
    if async_response:
        async def respond(messages):
            return output

        result = await registered_wrappers["ChatAgent.llm_response_messages_async"](respond, agent, (agent.message_history,), {})
    else:
        result = registered_wrappers["ChatAgent.llm_response_messages"](lambda messages: output, agent, (agent.message_history,), {})

    assert result is output
    assert recorded
    tool_events = [event for event in recorded if event.tools]
    assert tool_events
    assert all(event.tools[0].name == "lookup" for event in tool_events)
    assert all(json.loads(event.tools[0].arguments) == {"query": "example"} for event in tool_events)
    # Accompanying text that is not a tool call adds nothing, and content that
    # is absent altogether is read without reaching into it.
    assert all(len(event.tools) == 1 for event in tool_events)
    assert instrumentation.get_tools(agent.message_history[0]) == list(tool_events[0].tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_tool", [False, True])
@pytest.mark.parametrize("transforms", [[], ["required-redaction"]])
async def test_required_tool_transforms_stop_dispatch(
    monkeypatch, registered_wrappers, async_tool, transforms
):
    tool = Mock()
    tool.default_value.return_value = "send"
    tool.model_dump.return_value = {"request": "send", "body": "example"}
    verdict = SimpleNamespace(authorized=True, transform_ids=transforms)
    monkeypatch.setattr(instrumentation, "get_current_input_ids", lambda: ["input"])
    monkeypatch.setattr(instrumentation, "get_config", lambda: SimpleNamespace(
        log_policy_decisions=False, tool_policy_fail_closed=False,
    ))
    monkeypatch.setattr(instrumentation, "add_tool_denial", Mock())
    # A responder is what puts a dispatch scope in place; without one the
    # adapter refuses the call, so the test enters the scope a responder would.
    scope = instrumentation._dispatch_outcomes.set([])
    try:
        if async_tool:
            monkeypatch.setattr(instrumentation, "rm_check_tool_call_async", AsyncMock(return_value=verdict))
            dispatch = AsyncMock(return_value="sent")
            result = await registered_wrappers["Agent.handle_tool_message_async"](dispatch, None, (tool,), {})
        else:
            monkeypatch.setattr(instrumentation, "rm_check_tool_call", Mock(return_value=verdict))
            dispatch = Mock(return_value="sent")
            result = registered_wrappers["Agent.handle_tool_message"](dispatch, None, (tool,), {})
    finally:
        instrumentation._dispatch_outcomes.reset(scope)
    if transforms:
        dispatch.assert_not_called()
        assert "Required tool transforms" in result
    else:
        dispatch.assert_called_once_with(tool)
        assert result == "sent"
