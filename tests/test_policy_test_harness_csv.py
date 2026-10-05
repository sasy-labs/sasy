"""RFC 4180 regression coverage for the shipped policy test harness."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import pytest

from scripts.policy_test_harness import (
    desugar_policy,
    parse_souffle_output,
    run_souffle,
    write_fact_files,
)

HOSTILE_TRANSFORM = (
    'inject,\\path\\to\\"quoted",value\r\nsecond line\n'
    '===============\n---------------\rfinal line'
)


def _read_csv_rows(path: Path) -> list[list[str]]:
    """Read RFC 4180 rows without normalizing embedded line endings.

    Args:
        path: CSV fact file to read.

    Returns:
        Parsed rows from the fact file.
    """
    with path.open("r", encoding="utf-8", newline="") as fact_file:
        return list(csv.reader(fact_file))


@pytest.mark.toolchain
def test_hostile_rfc4180_value_survives_common_policy(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Round-trip hostile input through common policy and ApplyTransform."""
    if shutil.which("souffle") is None:
        if (
            request.config.getoption("--run-toolchain", default=False)
            or os.environ.get("SASY_REQUIRE_TOOLCHAIN") == "1"
        ):
            pytest.fail("Souffle is required for the selected toolchain tests")
        pytest.skip("Souffle is not installed; run this test in nix develop")

    policy_path = tmp_path / "hostile_transform.dl"
    policy_path.write_text(
        """IsAuthorized(idx) :- Actions(idx, _).
ApplyTransform(idx, transform_id) :-
    Actions(idx, action),
    action = $CallTool(transform_id, _).
""",
        encoding="utf-8",
    )
    scenario = {
        "principal": HOSTILE_TRANSFORM,
        "entity": HOSTILE_TRANSFORM,
        "action": {
            "type": "CallTool",
            "fn_name": HOSTILE_TRANSFORM,
            "args": "{}",
        },
        "graph": {},
    }
    fact_dir = tmp_path / "facts"
    write_fact_files(scenario, fact_dir)

    assert _read_csv_rows(fact_dir / "Principal.facts") == [
        [HOSTILE_TRANSFORM]
    ]
    action_rows = _read_csv_rows(fact_dir / "Actions.facts")
    assert len(action_rows) == 1
    assert len(action_rows[0]) == 2
    assert "\r\n" in action_rows[0][1]

    desugared = desugar_policy(policy_path)
    try:
        stdout = run_souffle(desugared, fact_dir)
    finally:
        desugared.unlink()

    results = parse_souffle_output(stdout)
    assert results.authorized == {0}
    assert results.apply_transform == {0: [HOSTILE_TRANSFORM]}
