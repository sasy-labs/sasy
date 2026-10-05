"""Opt-in decision-point capture for the reference monitor.

When ``SASY_CAPTURE_FILE`` is set, every CheckToolCall appends a JSON
line recording the proposed tool call, the backward slice of its
trigger node (full node + edge data), and the policy decision.
Captured traces can be converted into ``scripts/policy_test_harness.py``
scenarios for offline policy testing.

Capture is observational only — failures are swallowed (logged at
debug level) and never affect the authorization decision. The slice
query is one extra gRPC round-trip per decision point, which is
acceptable for capture runs and costs nothing when the env var is
unset. In async contexts the slice query runs synchronously on the
event loop; capture runs are not latency-sensitive.

Records are written with a single ``os.write`` to an O_APPEND fd
(same pattern as the latency log); lines larger than PIPE_BUF may
interleave under heavy multi-process concurrency, so consumers skip
unparseable lines.

Env vars:
    SASY_CAPTURE_FILE         append-mode JSONL path (unset = off)
    SASY_CAPTURE_SLICE_DEPTH  optional int; max backward-slice depth
                              (default: unbounded)
    SASY_CAPTURE_RUN_ID       optional label stamped into records
                              (default: ``<pid>-<start-time>``)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sasy.capture import capture_logger

logger = capture_logger(__name__)

CAPTURE_RECORD_VERSION = 1

# `entity` and `principal` are part of the policy-visible Message
# record, so a slice without them replays an authenticated message
# as an anonymous one.
_NODE_FIELDS = (
    "role", "agent", "text", "tools", "derived_from",
    "entity", "principal",
)
_EDGE_FIELDS = ("message_index", "proximal")

_capture_fd: int | None = None
_capture_path: str | None = None
_default_run_id = f"{os.getpid()}-{int(time.time())}"
_slice_failure_warned = False


def capture_enabled() -> bool:
    return bool(os.environ.get("SASY_CAPTURE_FILE"))


def _slice_depth() -> int | None:
    raw = os.environ.get("SASY_CAPTURE_SLICE_DEPTH")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.debug("bad SASY_CAPTURE_SLICE_DEPTH: %r", raw)
        return None


def _serialize_slice(graph: Any) -> dict[str, Any]:
    """Flatten a backward-slice DiGraph into JSON-able nodes/edges.

    Node data follows ``observability.utils.dict_from_event``
    (role is the proto enum name: SYSTEM/USER/LLM/AGENT); edge data
    follows ``dict_from_edge`` (message_index, proximal).
    """
    nodes = []
    for nid, data in graph.nodes(data=True):
        node: dict[str, Any] = {"id": nid}
        for key in _NODE_FIELDS:
            if key in data:
                node[key] = data[key]
        nodes.append(node)
    edges = []
    for src, dst, data in graph.edges(data=True):
        edge: dict[str, Any] = {"source": src, "destination": dst}
        for key in _EDGE_FIELDS:
            if key in data:
                edge[key] = data[key]
        edges.append(edge)
    return {"depth": _slice_depth(), "nodes": nodes, "edges": edges}


def _append(record: dict[str, Any]) -> None:
    global _capture_fd, _capture_path
    path = os.environ.get("SASY_CAPTURE_FILE")
    if not path:
        return
    if _capture_fd is None or _capture_path != path:
        try:
            _capture_fd = os.open(
                path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644
            )
            _capture_path = path
        except OSError as e:
            logger.debug("capture log open failed (%s): %s", path, e)
            return
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    try:
        os.write(_capture_fd, line)
    except OSError as e:
        logger.debug("capture log write failed (%s): %s", path, e)


def _warn_slice_failure_once(exc: Exception) -> None:
    """Say it loudly, once, when the backward-slice query fails.

    A capture whose records all carry ``slice: null`` is worthless —
    there is no dependency graph to convert into scenarios and no
    context to show the oracle — but nothing else surfaces it, so a
    whole run can complete before anyone notices. The usual cause is
    authorization: the slice query needs the ``observability-reader``
    role, which the default ``client`` entity does not hold.
    """
    global _slice_failure_warned
    if _slice_failure_warned:
        return
    _slice_failure_warned = True
    logger.warning(
        "SASY capture: backward-slice query failed, records will "
        "have no graph (%s). If this is a permissions error, run as "
        "an identity holding the 'observability-reader' role, using its "
        "complete provisioned API key.",
        exc,
    )


def capture_decision(
    fn_name: str,
    args: str,
    input_node_ids: list[str] | None,
    session_id: str | None,
    entity: str,
    result: Any = None,
    rpc_error: str | None = None,
    elapsed_s: float = 0.0,
    metadata: list | None = None,
) -> None:
    """Record one decision point. Never raises.

    ``result`` is a ``ToolCallResult`` (or None when the RPC failed,
    in which case ``rpc_error`` carries the error string).
    """
    if not capture_enabled():
        return
    try:
        record: dict[str, Any] = {
            "v": CAPTURE_RECORD_VERSION,
            "ts": time.time(),
            "run_id": os.environ.get(
                "SASY_CAPTURE_RUN_ID", _default_run_id
            ),
            "session_id": session_id or "",
            "entity": entity or "",
            "fn_name": fn_name,
            "args": args,
            "input_node_ids": list(input_node_ids or []),
            # Per-action facts the engine projects into ActionMetadata.
            # The live evaluator reasons over them, so a scenario built
            # without them writes an empty ActionMetadata relation and
            # any rule gated on operator-attested facts is certified
            # offline against a verdict live enforcement would not give.
            "action_metadata": [list(f) for f in (metadata or [])],
            "elapsed_s": elapsed_s,
        }

        decision: dict[str, Any] = {"rpc_error": rpc_error}
        if result is not None:
            decision["authorized"] = bool(result.authorized)
            decision["denial_reasons"] = list(result.denial_reasons)
            decision["suggestions"] = list(result.suggestions)
            decision["transform_ids"] = list(result.transform_ids)
        else:
            decision["authorized"] = None
        record["decision"] = decision

        roots = list(input_node_ids or [])
        tool_id = roots[-1] if roots else None
        record["tool_id"] = tool_id
        if roots:
            try:
                # Lazy import: avoids a hard networkx dependency
                # (and any import-order surprises) when capture is
                # off or the observability extra isn't installed.
                from sasy.observability.api import backward_slice

                # Every current node, not just the last. The evaluator
                # reasons over all of them and the converted scenario
                # keeps the whole list, so slicing only the final root
                # drops the provenance behind the others whenever they
                # are not its ancestors — and offline replay then
                # certifies a verdict the live graph would not give.
                merged: dict[str, Any] = {"nodes": [], "edges": []}
                seen_nodes: set[str] = set()
                seen_edges: set[tuple[str, str]] = set()
                for root in roots:
                    part = _serialize_slice(
                        backward_slice(root, max_depth=_slice_depth())
                    )
                    for node in part.get("nodes", []):
                        if node.get("id") not in seen_nodes:
                            seen_nodes.add(node.get("id"))
                            merged["nodes"].append(node)
                    for edge in part.get("edges", []):
                        key = (edge.get("source"), edge.get("destination"))
                        if key not in seen_edges:
                            seen_edges.add(key)
                            merged["edges"].append(edge)
                record["slice"] = merged
                record["slice_error"] = None
            except Exception as e:  # noqa: BLE001 — observational
                record["slice"] = None
                record["slice_error"] = str(e)
                _warn_slice_failure_once(e)
        else:
            record["slice"] = None
            record["slice_error"] = "no input_node_ids"

        _append(record)
    except Exception as e:  # noqa: BLE001 — never affect decisions
        logger.debug("capture_decision failed: %s", e)
