"""Require anonymous pulls of the exact qualified amd64 and arm64 image index."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
PLATFORMS = {("linux", "amd64"), ("linux", "arm64")}


def validate_manifest(manifest: dict, expected: dict[str, str]) -> None:
    """Reject missing/extra platforms, duplicate entries and unqualified digests."""
    children = manifest.get("manifests")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("Release index must contain exactly two qualified images.")
    found = {}
    for child in children:
        platform = child.get("platform", {})
        key = (platform.get("os"), platform.get("architecture"))
        if key not in PLATFORMS or key in found:
            raise ValueError(f"Unexpected or duplicate release platform: {key}")
        found[key] = child.get("digest")
    for architecture, reference in expected.items():
        if found.get(("linux", architecture)) != reference.rsplit("@", 1)[1]:
            raise ValueError(f"The {architecture} image differs from the qualified build.")


def verify_public_image(reference: str, expected: dict[str, str], tagged_reference: str) -> None:
    """Use a clean Docker configuration, independent of registry login/helpers.

    Pull by immutable index digest for each platform. A public manifest alone
    is insufficient: layer access must also work without registry credentials.
    """
    image, separator, digest = reference.rpartition("@")
    if not separator or not DIGEST.fullmatch(digest):
        raise ValueError("The release image must be pinned by sha256 digest.")
    if not re.fullmatch(re.escape(image) + r":\d+\.\d+\.\d+", tagged_reference):
        raise ValueError("The consumer tag must name a release in the same image repository.")
    if set(expected) != {"amd64", "arm64"}:
        raise ValueError("Both qualified architecture references are required.")
    for child in expected.values():
        child_image, child_separator, child_digest = child.rpartition("@")
        if not child_separator or child_image != image or not DIGEST.fullmatch(child_digest):
            raise ValueError("Qualified images must have matching repositories and sha256 digests.")
    with tempfile.TemporaryDirectory(prefix="sasy-anonymous-docker-") as directory:
        Path(directory, "config.json").write_text("{}\n")
        environment = {**os.environ, "DOCKER_CONFIG": directory}
        # Do not inherit alternate registry auth formats from the publishing job.
        environment.pop("DOCKER_AUTH_CONFIG", None)
        tagged = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", "--format", "{{.Manifest.Digest}}", tagged_reference],
            check=True, capture_output=True, text=True, env=environment, timeout=120,
        )
        if tagged.stdout.strip() != digest:
            raise ValueError("The consumer version tag differs from the qualified release index.")
        output = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", "--raw", reference],
            check=True, capture_output=True, text=True, env=environment, timeout=120,
        )
        validate_manifest(json.loads(output.stdout), expected)
        for architecture in ("amd64", "arm64"):
            subprocess.run(
                ["docker", "pull", "--platform", f"linux/{architecture}", reference],
                check=True, env=environment, timeout=600,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("--tagged-reference", required=True)
    parser.add_argument("--amd64", required=True)
    parser.add_argument("--arm64", required=True)
    arguments = parser.parse_args()
    verify_public_image(
        arguments.reference, {"amd64": arguments.amd64, "arm64": arguments.arm64}, arguments.tagged_reference
    )


if __name__ == "__main__":
    main()
