"""The SDK release requires both qualified images to be anonymously usable."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_public_image", ROOT / "scripts/verify_public_image.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
IMAGE = "ghcr.io/sasy-labs/sasy"
INDEX = IMAGE + "@sha256:" + "a" * 64
TAG = IMAGE + ":0.5.0"
EXPECTED = {arch: IMAGE + "@sha256:" + digest * 64 for arch, digest in [("amd64", "b"), ("arm64", "c")]}


def manifest():
    return {"manifests": [
        {"platform": {"os": "linux", "architecture": arch}, "digest": value.split("@")[1]}
        for arch, value in EXPECTED.items()
    ]}


def fake_docker(monkeypatch, *, denied=None, content=None, tag_digest=None):
    calls = []

    def run(command, **options):
        calls.append(command)
        config = Path(options["env"]["DOCKER_CONFIG"])
        assert config != Path(os.environ["DOCKER_CONFIG"])
        assert json.loads((config / "config.json").read_text()) == {}
        assert "DOCKER_AUTH_CONFIG" not in options["env"]
        if denied == "manifest" or (denied == "arm64" and "linux/arm64" in command):
            raise subprocess.CalledProcessError(1, command, stderr="unauthorized")
        output = (tag_digest or INDEX.split("@")[1]) if "--format" in command else json.dumps(content or manifest())
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setenv("DOCKER_CONFIG", "/publishing-job/authenticated-config")
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", "private credentials")
    monkeypatch.setattr(MODULE.subprocess, "run", run)
    return calls


def test_both_anonymous_architecture_pulls_are_required(monkeypatch):
    calls = fake_docker(monkeypatch)
    MODULE.verify_public_image(INDEX, EXPECTED, TAG)
    assert calls == [
        ["docker", "buildx", "imagetools", "inspect", "--format", "{{.Manifest.Digest}}", TAG],
        ["docker", "buildx", "imagetools", "inspect", "--raw", INDEX],
        ["docker", "pull", "--platform", "linux/amd64", INDEX],
        ["docker", "pull", "--platform", "linux/arm64", INDEX],
    ]


@pytest.mark.parametrize("denied", ["manifest", "arm64"])
def test_private_manifest_or_layers_fail_the_release_gate(monkeypatch, denied):
    fake_docker(monkeypatch, denied=denied)
    with pytest.raises(subprocess.CalledProcessError):
        MODULE.verify_public_image(INDEX, EXPECTED, TAG)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_digest", "extra"])
def test_only_the_two_qualified_architectures_can_pass(monkeypatch, problem):
    value = manifest()
    if problem == "missing":
        value["manifests"].pop()
    elif problem == "duplicate":
        value["manifests"][1] = value["manifests"][0]
    elif problem == "wrong_digest":
        value["manifests"][1]["digest"] = "sha256:" + "d" * 64
    else:
        value["manifests"].append(value["manifests"][0])
    calls = fake_docker(monkeypatch, content=value)
    with pytest.raises(ValueError):
        MODULE.verify_public_image(INDEX, EXPECTED, TAG)
    assert len(calls) == 2



def test_version_tag_must_resolve_to_the_qualified_index(monkeypatch):
    calls = fake_docker(monkeypatch, tag_digest="sha256:" + "d" * 64)
    with pytest.raises(ValueError, match="consumer version tag"):
        MODULE.verify_public_image(INDEX, EXPECTED, TAG)
    assert len(calls) == 1


def test_publication_graph_has_one_core_gate_and_no_independent_image_trigger():
    # BaseLoader preserves YAML's `on` key rather than treating it as a boolean.
    sdk = yaml.load((ROOT / ".github/workflows/sdk-python-release.yml").read_text(), Loader=yaml.BaseLoader)
    engine = yaml.load((ROOT / ".github/workflows/engine-release.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(engine["on"]) == {"workflow_call"}
    assert sdk["jobs"]["engine"]["uses"] == "./.github/workflows/engine-release.yml"
    assert sdk["jobs"]["core"]["if"] == "github.event_name == 'workflow_dispatch'"
    assert engine["jobs"]["core"]["uses"] == "./.github/workflows/ci-core.yml"
    assert set(engine["jobs"]["build"]["needs"]) == {"version", "core"}
    assert set(engine["jobs"]["publish"]["needs"]) == {"version", "build"}
    publish = sdk["jobs"]["publish"]
    assert set(publish["needs"]) == {"engine", "build"}
    for guard in ["needs.engine.result == 'success'", "needs.build.result == 'success'", "needs.engine.outputs.digest != ''", "github.repository == 'sasy-labs/sasy'", "!github.event.repository.private", "needs.engine.outputs.version"]:
        assert guard in publish["if"]
    assert set(engine["jobs"]["public_access"]["needs"]) == {"version", "publish"}
    access = engine["jobs"]["public_access"]["steps"][-1]
    assert access["id"] == "anonymous"
    assert "scripts/verify_public_image.py" in access["run"]
    assert "steps.anonymous.outputs.digest" in engine["jobs"]["public_access"]["outputs"]["digest"]


@pytest.mark.parametrize("push_fails", [False, True])
def test_publish_uses_successful_push_receipt_without_another_registry_request(tmp_path, push_fails):
    engine = yaml.load((ROOT / ".github/workflows/engine-release.yml").read_text(), Loader=yaml.BaseLoader)
    script = next(step["run"] for step in engine["jobs"]["publish"]["steps"] if step.get("id") == "index")
    (tmp_path / "dist").mkdir()
    for arch, reference in EXPECTED.items():
        (tmp_path / "dist" / f"sasy-linux-{arch}.image").write_text(reference + "\n")
    (tmp_path / "scripts").mkdir()
    guard = tmp_path / "scripts/require_unpublished_image.sh"
    guard.write_bytes((ROOT / "scripts/require_unpublished_image.sh").read_bytes())
    guard.chmod(0o755)
    docker = tmp_path / "docker"
    docker.write_text("""#!/bin/bash
case "$1 $2" in
  'login ghcr.io') cat >/dev/null ;;
  'manifest inspect') echo 'manifest unknown' >&2; exit 1 ;;
  'manifest create') exit 0 ;;
  'manifest push')
    if [[ "$PUSH_FAILS" == 1 ]]; then exit 1; fi
    echo 'Pushed ref ghcr.io/sasy-labs/sasy:0.5.0'
    echo "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ;;
  *) echo 'Unexpected post-publication registry request' >&2; exit 99 ;;
esac
""")
    docker.chmod(0o755)
    output = tmp_path / "outputs"
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                   "GH_TOKEN": "test", "GITHUB_ACTOR": "test", "IMAGE": IMAGE, "VERSION": "0.5.0",
                   "GITHUB_OUTPUT": str(output), "PUSH_FAILS": str(int(push_fails))}
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    if push_fails:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines() == [
            "digest=" + INDEX.split("@")[1],
            "amd64=" + EXPECTED["amd64"], "arm64=" + EXPECTED["arm64"],
        ]


@pytest.mark.parametrize('image, endpoint', [
    ('ghcr.io/sasy-labs/sasy', 'orgs/sasy-labs/packages/container/sasy'),
    ('ghcr.io/nilspalumbo/sasy-test', 'users/nilspalumbo/packages/container/sasy-test'),
])
@pytest.mark.parametrize('visibility', ['private', 'public'])
def test_rehearsal_checks_the_selected_package(monkeypatch, image, endpoint, visibility):
    import io
    import runpy
    import sys
    import urllib.request

    calls = []

    def open_package(request, **kwargs):
        calls.append(request.full_url)
        return io.StringIO(json.dumps({'visibility': visibility}))

    monkeypatch.setenv('IMAGE', image)
    monkeypatch.setenv('GH_TOKEN', 'synthetic-test-token')
    monkeypatch.setattr(sys, 'argv', ['check_rehearsal_registry.py'])
    monkeypatch.setattr(urllib.request, 'urlopen', open_package)
    if visibility == 'private':
        runpy.run_path(str(ROOT / 'scripts/check_rehearsal_registry.py'))
    else:
        with pytest.raises(SystemExit, match='nonprivate package'):
            runpy.run_path(str(ROOT / 'scripts/check_rehearsal_registry.py'))
    assert calls == ['https://api.github.com/' + endpoint]


def test_npm_release_stages_without_direct_publish():
    workflow = yaml.safe_load((ROOT / '.github/workflows/sdk-js-release.yml').read_text())
    steps = workflow['jobs']['publish']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert 'npm stage publish "$1"' in commands
    assert 'npm publish "$1"' not in commands
    assert 'npm dist-tag' not in commands
    assert 'npm >=11.15.0 is required' in commands
    assert 'GITHUB_STEP_SUMMARY' in commands
