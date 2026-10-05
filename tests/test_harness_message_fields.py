"""The offline harness must present the same Message a live run would.

A field the harness pins to a constant does not make a rule reading it
fail — it makes the rule evaluate against the constant and *pass*. The
policy then converges against evidence the deployment will not
reproduce, and the gap only shows up as a live false-block. That is how
``tools_json`` behaved: hardcoded to ``"[]"`` while real captures carry
tool calls on ~16% of messages.

These tests assert on the serialized fact files rather than on a
Soufflé verdict, so they fail on the wiring itself instead of on
whichever rule happens to read it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from policy_test_harness import (  # noqa: E402
    adt_message,
    write_fact_files,
)

TOOLS_JSON = '[{"name": "get_user_details", "arguments": {"user_id": "u7340"}}]'


def _scenario() -> dict:
    return {
        "entity": "alice",
        "roles": ["admin"],
        "action": {
            "type": "CallTool",
            "fn_name": "cancel_reservation",
            "args": '{"reservation_id": "R1"}',
        },
        "graph": {
            "current_nodes": ["m2"],
            "edges": [["m1", "m2"]],
            "edges_data": [
                {"source": "m1", "destination": "m2",
                 "message_index": 0, "proximal": True},
            ],
            "sent_messages": [
                {"id": "m1", "contents": "looking that up",
                 "agent": "LLMAgent", "agent_role": "assistant",
                 "tools_json": TOOLS_JSON},
                {"id": "m2", "contents": "please cancel",
                 "agent": "LLMAgent", "agent_role": "user"},
            ],
            "tool_results": [],
        },
    }


def _facts(tmp_path: Path, name: str) -> list[list[str]]:
    write_fact_files(_scenario(), tmp_path)
    with (tmp_path / f"{name}.facts").open(newline="") as fh:
        return list(csv.reader(fh))


def test_adt_message_carries_the_tool_calls() -> None:
    msg = {"contents": "x", "agent": "A", "agent_role": "assistant",
           "tools_json": TOOLS_JSON}
    rendered = adt_message(msg)
    assert "get_user_details" in rendered, (
        "tools_json was dropped; a rule reading it would see an empty "
        "tool list offline and real data live"
    )


def test_adt_message_defaults_to_an_empty_list() -> None:
    msg = {"contents": "x", "agent": "A", "agent_role": "user"}
    assert '"[]"' in adt_message(msg) or "[]" in adt_message(msg)


def test_sent_message_facts_preserve_tool_calls(tmp_path: Path) -> None:
    rows = _facts(tmp_path, "SentMessage")
    assert len(rows) == 2
    blob = "".join("".join(r) for r in rows)
    assert "get_user_details" in blob
    assert "u7340" in blob


def test_edge_data_facts_are_populated(tmp_path: Path) -> None:
    """The same class of bug, one field over: an empty EdgeData starves
    every rule that orders turns by message_index."""
    rows = _facts(tmp_path, "EdgeData")
    assert rows, "EdgeData.facts was empty despite edges_data in the scenario"
    assert rows[0][0] == "m1" and rows[0][1] == "m2"
    assert "[1, 0]" in rows[0][2].replace(" ", " ")


def test_message_metadata_reaches_the_facts(tmp_path: Path) -> None:
    """A message's ``metadata`` must reach ``MessageMetadata.facts``.

    Without it the relation is empty offline while it is populated
    live, so a rule reading the adapter's record of a message
    converges against an emptiness the deployment will not reproduce.
    """
    scenario = _scenario()
    record = '{"s:additional_kwargs":{"s:marker":"exfil"},"s:content":"x"}'
    scenario["graph"]["sent_messages"][0]["metadata"] = record
    write_fact_files(scenario, tmp_path)
    with (tmp_path / "MessageMetadata.facts").open(newline="") as fh:
        rows = list(csv.reader(fh))
    # Sparse, as it is live: a row only for the message that carries it.
    assert rows == [["m1", record]]


def test_tool_results_reach_the_facts(tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["graph"]["tool_results"] = [
        {"id": "m1", "fn_name": "get_user_details",
         "args": '{"user_id": "u7340"}'}
    ]
    write_fact_files(scenario, tmp_path)
    with (tmp_path / "ToolResult.facts").open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows and rows[0][1] == "get_user_details"
