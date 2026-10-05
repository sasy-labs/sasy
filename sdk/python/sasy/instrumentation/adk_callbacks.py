"""Message scopes for callbacks in the qualified native ADK execution path."""
from __future__ import annotations

import copy
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import Any

from .session import is_session_active

_callback: ContextVar[bool] = ContextVar("sasy_adk_callback", default=False)
_lock = RLock()
_installed = False


def in_callback() -> bool:
    """Whether this task is executing application callback code."""
    return _callback.get()


def _adk():
    from . import adk
    return adk


def install() -> None:
    """Scope native callback pipelines without replacing callback objects."""
    global _installed
    from google.adk.agents import base_agent
    from google.adk.flows.llm_flows import (
        _tool_caller,
        _tool_error_handler,
        base_llm_flow,
    )

    with _lock:
        if _installed:
            return
        original_agent: Any = base_agent._run_callbacks

        @wraps(original_agent)
        async def agent_callbacks(callbacks, stop_condition, *args, **kwargs):
            sdk = _adk()
            if not is_session_active() or not callbacks or sdk._active.get() is None:
                return await original_agent(callbacks, stop_condition, *args, **kwargs)
            from .adk_context import callback_scope, validate_deltas
            # An agent callback consumes no message: its own state and artifact
            # reads are the whole of its observed input.
            context = kwargs["callback_context"]
            input_token = sdk._current_input_ids.set([])
            callback_token = _callback.set(True)
            try:
                async with callback_scope(context, [], root=True):
                    result = await original_agent(callbacks, stop_condition, *args, **kwargs)
                actions = context.actions
                if (actions.state_delta or actions.artifact_delta) and not validate_deltas(actions):
                    # ADK yields an agent callback's delta event straight from
                    # the agent, not through the model postprocessing the
                    # adapter observes, so the delta is checked here: once the
                    # event exists the Runner has already stored it, and a
                    # user:/app: key would outlive the refused run.
                    raise sdk.AdkInstrumentationError(
                        "A before/after_agent_callback changed ADK state or artifacts "
                        "outside its observed context. Write through the callback "
                        "context's `state` mapping and `save_artifact`, not through "
                        "`callback_context.actions`, so the write has an observed "
                        "producer."
                    )
                if result is not None:
                    raise sdk.AdkInstrumentationError(
                        "A before/after_agent_callback returned content instead of "
                        "letting the agent run. Returning content from an agent "
                        "callback is not supported by the SASY ADK adapter, because "
                        "the content would not come from an observed run; return None."
                    )
                return result
            finally:
                _callback.reset(callback_token)
                sdk._current_input_ids.reset(input_token)

        setattr(base_agent, "_run_callbacks", agent_callbacks)
        original_model: Any = base_llm_flow._run_callbacks

        @wraps(original_model)
        async def model_callbacks(callbacks, stop_condition, *args, **kwargs):
            if not is_session_active() or not callbacks:
                return await original_model(callbacks, stop_condition, *args, **kwargs)
            sdk = _adk()
            state = sdk._state()
            context = kwargs["callback_context"]._invocation_context
            agent = context.agent.name
            request = kwargs.get("llm_request")
            from . import adk_context, adk_state
            instruction = None
            originals = {}
            if request is not None:
                # The callback consumes the pre-filter request. The actual
                # model boundary independently resolves its final request.
                inputs = await state.inputs(request, agent)
                instruction = copy.deepcopy(request.config.system_instruction)
                # Keyed by id() because ADK content objects are unhashable and
                # compare by value; the entry keeps the object itself, so the
                # id cannot be recycled, and identity and the content key are
                # both re-checked on lookup below.
                originals = {id(content): (content, sdk._content_key(content)) for content in request.contents}
            else:
                response = kwargs["llm_response"]
                if response.partial:
                    raise sdk.AdkInstrumentationError(
                        "After-model callbacks on partial responses require observed chunk identities"
                    )
                output = state.outputs.get(agent)
                if output is None or not response.content:
                    raise sdk.AdkInstrumentationError("Model callback has no observed model output")
                inputs = list(output[1])
            input_token = sdk._current_input_ids.set(inputs)
            callback_token = _callback.set(True)
            try:
                async with adk_context.callback_scope(kwargs["callback_context"], inputs) as frame:
                    result = await original_model(callbacks, stop_condition, *args, **kwargs)
                if request is not None:
                    state.resources.callback_dependencies[agent] = list(dict.fromkeys([
                        *state.resources.callback_dependencies.get(agent, []), *frame.reads]))
                    if request.config.system_instruction != instruction:
                        adk_state.consume(inputs)
                    for content in request.contents:
                        previous = originals.get(id(content))
                        if previous is None or previous[0] is not content or previous[1] != sdk._content_key(content):
                            ids = await state.record(content, agent, list(dict.fromkeys([*inputs, *frame.reads])))
                            adk_state.register_rendering(state, agent, content, ids)
                elif frame.reads:
                    produced = result or kwargs["llm_response"]
                    if not produced.content:
                        raise sdk.AdkInstrumentationError("Callback output requires observed content")
                    ids = await state.record(produced.content, agent, list(dict.fromkeys([*inputs, *frame.reads])))
                    state.outputs[agent] = (sdk._content_key(produced.content), ids)
                return result
            finally:
                _callback.reset(callback_token)
                sdk._current_input_ids.reset(input_token)

        setattr(base_llm_flow, "_run_callbacks", model_callbacks)

        def tool_pipeline(original):
            @wraps(original)
            async def tool_callbacks(callbacks, stop_condition, *args, **kwargs):
                if not is_session_active() or not callbacks:
                    return await original(callbacks, stop_condition, *args, **kwargs)
                sdk = _adk()
                from . import adk_state
                context = kwargs["tool_context"]
                state, agent, origin = sdk._origin(context)
                key = (agent, context.function_call_id)
                inputs = adk_state.tool_inputs(origin.ids)
                after = "tool_response" in kwargs
                if after:
                    produced = state.results.get(key)
                    if not produced:
                        raise sdk.AdkInstrumentationError(
                            "An after_tool_callback ran for a tool whose execution the "
                            "adapter did not observe; the result has no recorded origin."
                        )
                    inputs = list(dict.fromkeys([*inputs, *produced]))
                input_token = sdk._current_input_ids.set(inputs)
                callback_token = _callback.set(True)
                try:
                    from .adk_context import callback_scope
                    async with callback_scope(context, inputs):
                        result = await original(callbacks, stop_condition, *args, **kwargs)
                    if not after and "error" not in kwargs and result is not None and not state.results.get(key):
                        # Stop before ADK invokes downstream after-tool callbacks
                        # with a response that did not come from an execution.
                        raise sdk.AdkInstrumentationError(
                            f"A before_tool_callback returned a result for {agent}'s "
                            "tool instead of letting it run. Short-circuiting a tool "
                            "from a callback is not supported by the SASY ADK adapter, "
                            "because the result would not come from an authorized "
                            "execution; return None."
                        )
                    return result
                finally:
                    _callback.reset(callback_token)
                    sdk._current_input_ids.reset(input_token)
            return tool_callbacks

        _tool_caller._run_callbacks = tool_pipeline(_tool_caller._run_callbacks)
        _tool_error_handler._run_callbacks = tool_pipeline(_tool_error_handler._run_callbacks)
        _installed = True
