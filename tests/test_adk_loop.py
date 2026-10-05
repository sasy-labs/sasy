"""Native loop iterations preserve exact consumed message occurrences."""
import asyncio

import pytest
from test_adk_instrumentation import LlmAgent, ScriptedModel, adk, calls, conversation, text
from test_adk_instrumentation import sink as sink
from test_adk_state import ancestors


@pytest.mark.parametrize("retain_previous", [True, False])
def test_loop_consumes_only_selected_iteration_history(sink, retain_previous):
    from google.adk.agents import LoopAgent

    def approve():
        """Approve the first iteration."""
        return "approved"

    def consume():
        """Consume only the current model's inputs."""
        return "consumed"

    model = ScriptedModel([calls(("approve", {}, "approve")), text("same"),
                           calls(("consume", {}, "consume")), text("same")])
    def select(callback_context, llm_request):
        if not retain_previous and len(model._requests) >= 2:
            llm_request.contents = [content for content in llm_request.contents
                if any(part.text == "hello" for part in content.parts or [])]
    worker = LlmAgent(name="worker", model=model, tools=[approve, consume], before_model_callback=select)
    events, _, _ = asyncio.run(conversation(LoopAgent(name="loop", sub_agents=[worker], max_iterations=2)))
    assert len(model._requests) == 4
    inputs = next(check[2] for check in sink.checks if check[0] == "consume")
    assert any(event.HasField("derived_from") and event.derived_from.name == "approve"
               for event in ancestors(sink, inputs)) == retain_previous
    identical = [event for event in sink.events.values() if event.text == "same"]
    assert len(identical) == 2 and identical[0].id != identical[1].id
    assert len(events) == 6


def test_zero_iteration_loop_does_not_start_a_child(sink):
    from google.adk.agents import LoopAgent

    model = ScriptedModel([text("must not run")])
    events, _, _ = asyncio.run(conversation(LoopAgent(name="loop", max_iterations=0,
        sub_agents=[LlmAgent(name="child", model=model)])))
    assert not events and not model._requests and not sink.checks


def test_cancelled_loop_does_not_record_unfinished_iteration(sink):
    from google.adk.agents import LoopAgent
    from sasy.instrumentation import adk_state

    async def run():
        entered = asyncio.Event()
        class Blocking(ScriptedModel):
            async def generate_content_async(self, llm_request, stream=False):
                if not self._requests:
                    self._requests.append(llm_request)
                    yield text("first iteration")
                else:
                    entered.set()
                    await asyncio.Event().wait()
                    yield text("unfinished iteration")
        task = asyncio.create_task(conversation(LoopAgent(name="loop", max_iterations=2,
            sub_agents=[LlmAgent(name="child", model=Blocking([]))])))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adk._active.get() is None and adk_state._frame.get() is None
    asyncio.run(run())
    assert any(event.text == "first iteration" for event in sink.events.values())
    assert not any(event.text == "unfinished iteration" for event in sink.events.values())
