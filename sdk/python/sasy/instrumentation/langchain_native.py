"""Native compiled graphs for the supported LangChain message-loop agent.

Only graphs registered by the SASY factory enter these scopes. The middleware
and checked tools remain the same as the explicit SasyAgent adapter. This is
not admission of arbitrary LangGraph nodes or a use of the research analyzer.
"""
from __future__ import annotations

import copy
import inspect
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from . import langchain as adapter
from .session import current_wire_session_id, is_session_active

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

_graphs: WeakKeyDictionary[Any, tuple[Any, ...]] = WeakKeyDictionary()
_lazy: WeakKeyDictionary[Any, tuple[Any, Any]] = WeakKeyDictionary()
_lock = RLock()
_installed = False
_execution: ContextVar[Any] = ContextVar("sasy_native_langchain_execution", default=None)
_CONFIG_KEYS = {"recursion_limit", "max_concurrency", "tags", "metadata", "run_name", "run_id"}


class _BatchCaller:
    """Freeze batch entry ancestry while forwarding answers and liveness checks."""

    def __init__(self, scope: Any):
        self._scope = scope
        self._inputs = scope.consumed() if scope.session_id == current_wire_session_id() else ()

    def consumed(self) -> tuple[str, ...]:
        return self._inputs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scope, name)


def _shape(graph: Any) -> tuple[Any, ...]:
    return (
        tuple((key, id(node), id(node.bound), tuple(node.channels), tuple(node.triggers),
               tuple(map(id, node.writers)), id(node.mapper), id(node.cache_policy),
               id(node.retry_policy), id(node.timeout), node.is_error_handler,
               node.error_handler_node, tuple(map(id, node.subgraphs)))
              for key, node in graph.nodes.items()),
        tuple((key, id(channel)) for key, channel in graph.channels.items()),
        graph.input_channels, tuple(graph.output_channels), tuple(graph.stream_channels),
        tuple(map(id, graph.stream_transformers)),
    )


def _config(config: Any) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise adapter.InstrumentationError("LangChain config must be a dict")
    for key, value in config.items():
        if key not in _CONFIG_KEYS and value not in (None, [], {}):
            raise adapter.InstrumentationError(f"LangChain config {key!r} is not supported by SASY")


def _validate(graph: Any, config: Any, options: dict[str, Any]) -> RunnableConfig:
    if _shape(graph) != _graphs[graph]:
        raise adapter.InstrumentationError("A SASY LangChain graph was modified after construction")
    for name in ("checkpointer", "store", "cache", "cache_policy", "interrupt_before_nodes",
                 "interrupt_after_nodes", "node_error_handler_map"):
        if getattr(graph, name, None):
            raise adapter.InstrumentationError(f"LangChain graph {name!r} is not supported by SASY")
    _config(graph.config)
    _config(config)
    # Keep the former SasyAgent spelling while accepting native RunnableConfig.
    options = dict(options)
    config = dict(config or {})
    if "recursion_limit" in options:
        config["recursion_limit"] = options.pop("recursion_limit")
    defaults = {"context": None, "stream_mode": "values", "print_mode": (), "output_keys": None,
                "interrupt_before": None, "interrupt_after": None, "durability": None,
                "control": None, "version": "v1"}
    for name, value in options.items():
        if name not in defaults or value != defaults[name]:
            raise adapter.InstrumentationError(f"LangChain invocation {name!r} is not supported by SASY")
    return cast("RunnableConfig", config)


def _finish(run: Any, scope: Any, foreign: Any, output: Any) -> Any:
    run.verify()
    # Models/caches can reuse message objects across concurrent invocations.
    # A caller's provenance mark must not be overwritten by another result.
    output = copy.deepcopy(output)
    adapter._stamp(run, output)
    if scope is not None:
        scope.record(run.latest_answers)
    elif foreign is not None:
        foreign.defer(adapter._foreign_answer(run, output))
    return output


def register(graph: Any) -> Any:
    """Keep the compiled graph, registering its supported invocation boundary."""
    _install()
    with _lock:
        if graph not in _graphs:
            _graphs[graph] = _shape(graph)
    return graph


def register_lazy(graph: Any, factory: Any) -> Any:
    """Keep native behavior outside sessions and qualify at protected entry."""
    _install()
    with _lock:
        _lazy[graph] = (_shape(graph), factory)
    return graph


def _protected_graph(graph: Any) -> Any:
    if graph not in _lazy or not is_session_active():
        return None
    shape, factory = _lazy[graph]
    if _shape(graph) != shape:
        raise adapter.InstrumentationError("A SASY LangChain graph was modified after construction")
    for name in ("checkpointer", "store", "cache", "cache_policy", "interrupt_before_nodes",
                 "interrupt_after_nodes", "node_error_handler_map"):
        if getattr(graph, name, None):
            raise adapter.InstrumentationError(f"LangChain graph {name!r} is not supported by SASY")
    _config(graph.config)
    result = register(factory())
    if graph.config:
        result = result.with_config(graph.config)
    return result


def _install() -> None:
    global _installed
    from langchain_core.runnables.config import (
        get_config_list,
        get_executor_for_config,
        merge_configs,
        var_child_runnable_config,
    )
    from langchain_core.runnables.utils import gather_with_concurrency
    from langgraph.pregel import Pregel

    with _lock:
        if _installed:
            return
        invoke, ainvoke = Pregel.invoke, Pregel.ainvoke
        batch, abatch = Pregel.batch, Pregel.abatch
        batch_as_completed, abatch_as_completed = Pregel.batch_as_completed, Pregel.abatch_as_completed
        stream, astream, copy_graph = Pregel.stream, Pregel.astream, Pregel.copy
        for method in (invoke, ainvoke, stream, astream):
            if "input" not in inspect.signature(method).parameters:
                raise adapter.InstrumentationError("LangGraph invocation hooks have changed")

        @wraps(invoke)
        def guarded_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            protected = _protected_graph(self)
            if protected is not None:
                return protected.invoke(input, config, **kwargs)
            if self not in _graphs:
                return invoke(self, input, config, **kwargs)
            config = _validate(self, config, kwargs)
            with adapter.SasyAgent(self)._invocation(input) as (run, scope, foreign, prepared):
                token = _execution.set([self, False])
                try:
                    return _finish(run, scope, foreign, invoke(self, prepared, config))
                finally:
                    _execution.reset(token)

        @wraps(ainvoke)
        async def guarded_ainvoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
            protected = _protected_graph(self)
            if protected is not None:
                return await protected.ainvoke(input, config, **kwargs)
            if self not in _graphs:
                return await ainvoke(self, input, config, **kwargs)
            config = _validate(self, config, kwargs)
            with adapter.SasyAgent(self)._invocation(input) as (run, scope, foreign, prepared):
                token = _execution.set([self, False])
                try:
                    output = await ainvoke(self, prepared, config)
                    return _finish(run, scope, foreign, output)
                finally:
                    _execution.reset(token)

        def batch_configs(graph: Any, inputs: Any, config: Any, options: Any) -> Any:
            # Validate caller-supplied config before LangChain expands it. Its
            # ambient callbacks and internal graph state are framework-owned;
            # guarded invoke admits only the caller's supported configuration.
            explicit = ([_validate(graph, item, options) for item in config]
                        if isinstance(config, list) else _validate(graph, config, options))
            inherited = var_child_runnable_config.get() or {}
            token = var_child_runnable_config.set(
                cast("RunnableConfig", {key: value for key, value in inherited.items() if key in _CONFIG_KEYS}))
            try:
                merged = ([merge_configs(graph.config, item) for item in explicit]
                          if isinstance(explicit, list) else merge_configs(graph.config, explicit))
                return get_config_list(merged, len(inputs))
            finally:
                # Restore before dispatch, retaining native parent tracing and
                # SASY's independent invocation/tool contexts in every worker.
                var_child_runnable_config.reset(token)

        @wraps(batch)
        def guarded_batch(self: Any, inputs: Any, config: Any = None, *,
                          return_exceptions: bool = False, **kwargs: Any) -> Any:
            protected = _protected_graph(self)
            if protected is not None:
                return protected.batch(inputs, config, return_exceptions=return_exceptions, **kwargs)
            if self not in _graphs:
                return batch(self, inputs, config, return_exceptions=return_exceptions, **kwargs)
            if return_exceptions:
                raise adapter.InstrumentationError("LangChain batch return_exceptions is not supported by SASY")
            configs = batch_configs(self, inputs, config, kwargs)
            if not inputs:
                return []
            scope = adapter.SasyAgent(self)._delegation()
            caller = _BatchCaller(scope) if scope is not None else None

            def run(input: Any, config: Any) -> Any:
                token = adapter._tool_scope.set(cast("adapter._ToolScope | None", caller))
                try:
                    return self.invoke(input, config)
                finally:
                    adapter._tool_scope.reset(token)

            if len(inputs) == 1:
                return [run(inputs[0], configs[0])]
            with get_executor_for_config(configs[0]) as executor:
                return list(executor.map(run, inputs, configs))

        @wraps(abatch)
        async def guarded_abatch(self: Any, inputs: Any, config: Any = None, *,
                                 return_exceptions: bool = False, **kwargs: Any) -> Any:
            protected = _protected_graph(self)
            if protected is not None:
                return await protected.abatch(inputs, config, return_exceptions=return_exceptions, **kwargs)
            if self not in _graphs:
                return await abatch(self, inputs, config, return_exceptions=return_exceptions, **kwargs)
            if return_exceptions:
                raise adapter.InstrumentationError("LangChain batch return_exceptions is not supported by SASY")
            configs = batch_configs(self, inputs, config, kwargs)
            if not inputs:
                return []
            scope = adapter.SasyAgent(self)._delegation()
            caller = _BatchCaller(scope) if scope is not None else None

            async def run(input: Any, config: Any) -> Any:
                token = adapter._tool_scope.set(cast("adapter._ToolScope | None", caller))
                try:
                    return await self.ainvoke(input, config)
                finally:
                    adapter._tool_scope.reset(token)

            return await gather_with_concurrency(
                configs[0].get("max_concurrency"),
                *(run(input, config) for input, config in zip(inputs, configs, strict=True)))

        def check_stream(graph: Any) -> None:
            if graph in _lazy and is_session_active():
                raise adapter.InstrumentationError("Use invoke/ainvoke or batch/abatch; direct streaming is not supported by SASY")
            if graph in _graphs:
                permit = _execution.get()
                if permit is None or permit[0] is not graph or permit[1]:
                    raise adapter.InstrumentationError(
                        "Use invoke/ainvoke or batch/abatch for SASY LangChain agents; "
                        "direct streaming and embedding the graph as a custom node are not supported yet")
                permit[1] = True

        @wraps(batch_as_completed)
        def guarded_batch_as_completed(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self in _graphs or (self in _lazy and is_session_active()):
                raise adapter.InstrumentationError("Use batch/abatch; as-completed batches are not supported by SASY")
            yield from batch_as_completed(self, *args, **kwargs)

        @wraps(abatch_as_completed)
        async def guarded_abatch_as_completed(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self in _graphs or (self in _lazy and is_session_active()):
                raise adapter.InstrumentationError("Use batch/abatch; as-completed batches are not supported by SASY")
            async for item in abatch_as_completed(self, *args, **kwargs):
                yield item

        @wraps(stream)
        def guarded_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            check_stream(self)
            yield from stream(self, *args, **kwargs)

        @wraps(astream)
        async def guarded_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            check_stream(self)
            async for item in astream(self, *args, **kwargs):
                yield item

        @wraps(copy_graph)
        def guarded_copy(self: Any, update: Any = None) -> Any:
            if self in _lazy:
                shape, factory = _lazy[self]
                if is_session_active():
                    if _shape(self) != shape:
                        raise adapter.InstrumentationError("A SASY LangChain graph was modified after construction")
                    if set(update or {}) - {"config"}:
                        raise adapter.InstrumentationError("Only config copies of a SASY LangChain graph are supported")
                    _config((update or {}).get("config"))
                result = copy_graph(self, {**(update or {}), "channels": dict(self.channels)})
                if _shape(self) != shape or set(update or {}) - {"config"}:
                    def factory():
                        raise adapter.InstrumentationError("A SASY LangChain graph was modified after construction")
                return register_lazy(result, factory)
            if self not in _graphs:
                return copy_graph(self, update)
            _validate(self, None, {})
            if set(update or {}) - {"config"}:
                raise adapter.InstrumentationError("Only with_config/config copies of a SASY LangChain graph are supported")
            _config((update or {}).get("config"))
            # Construction replaces the TASKS channel. Give the copy its own
            # mapping so with_config cannot mutate the registered original.
            result = copy_graph(self, {**(update or {}), "channels": dict(self.channels)})
            with _lock:
                _graphs[result] = _shape(result)
            return result

        Pregel.invoke = guarded_invoke  # type: ignore[method-assign]
        Pregel.ainvoke = guarded_ainvoke  # type: ignore[method-assign]
        Pregel.batch = guarded_batch  # type: ignore[method-assign]
        Pregel.abatch = guarded_abatch  # type: ignore[method-assign]
        Pregel.batch_as_completed = guarded_batch_as_completed  # type: ignore[method-assign]
        Pregel.abatch_as_completed = guarded_abatch_as_completed  # type: ignore[method-assign]
        Pregel.stream = guarded_stream  # type: ignore[method-assign]
        Pregel.astream = guarded_astream  # type: ignore[method-assign]
        Pregel.copy = guarded_copy  # type: ignore[method-assign]
        _installed = True
