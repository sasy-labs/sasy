"""The release guard publishes only when the registry confirms a tag is new."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/require_unpublished_image.sh"


def guard(tmp_path, *, output, code):
    """Run the guard with a docker stand-in that answers as given."""
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\necho '{output}' >&2\nexit {code}\n")
    docker.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    return subprocess.run(
        ["bash", str(SCRIPT), "ghcr.io/sasy-labs/sasy:9.9.9"],
        capture_output=True, text=True, env=env, timeout=30,
    )


def test_an_existing_image_is_refused(tmp_path):
    assert guard(tmp_path, output="{}", code=0).returncode == 1


def test_registry_not_found_codes_allow_publishing(tmp_path):
    for output in ["manifest unknown", "no such manifest: x", "name unknown: x"]:
        assert guard(tmp_path, output=output, code=1).returncode == 0, output


def test_other_failures_fail_closed(tmp_path):
    for output in [
        "failed to fetch oauth token: unexpected status: 404 Not Found",
        "denied: requested access to the resource is denied",
        "dial tcp: connection refused",
    ]:
        assert guard(tmp_path, output=output, code=1).returncode == 1, output
