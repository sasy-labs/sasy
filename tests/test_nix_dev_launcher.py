"""The Nix input must exclude local credentials and other checkout contents."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("symlink_tmpdir", [False, True])
@pytest.mark.parametrize("exit_code", [0, 37])
@pytest.mark.parametrize("existing_profile_dir", [False, True])
def test_launcher_stages_only_locked_inputs_and_cleans_up(
    tmp_path, exit_code, symlink_tmpdir, existing_profile_dir,
):
    project = tmp_path / "checkout with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/nix-dev.sh", scripts / "nix-dev.sh")
    (project / "flake.nix").write_text("{ outputs = _: {}; }\n")
    (project / "flake.lock").write_text("{}\n")
    (project / "nix").mkdir()
    (project / "nix/toolchain.nix").write_text("{}\n")
    (project / "nix/package.nix").write_text("# Not needed for a dev shell\n")
    (project / ".env").write_text("SASY_API_KEY=private-test-value\n")
    if existing_profile_dir:
        (project / ".nix").mkdir()
        (project / ".nix/private-note").write_text("private-profile-note")
    (project / "unrelated-source.py").write_text("# Must not become a Nix input.\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    probe = tools / "nix"
    probe.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, sys
stage = pathlib.Path(next(a[5:] for a in sys.argv if a.startswith('path:')))
profile = pathlib.Path(sys.argv[sys.argv.index('--profile') + 1])
assert profile.parent.is_dir()
profile.symlink_to(os.environ['NIX_TEST_STORE'])
report = {'argv': sys.argv[1:], 'cwd': os.getcwd(), 'stage': str(stage),
          'files': sorted(str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()),
          'flake': (stage / 'flake.nix').read_text(),
          'lock': (stage / 'flake.lock').read_text()}
pathlib.Path(os.environ['NIX_TEST_REPORT']).write_text(json.dumps(report))
sys.exit(int(os.environ['NIX_TEST_EXIT']))
""")
    probe.chmod(0o755)
    fake_store = tmp_path / "fake-store-environment"
    fake_store.mkdir()
    report = tmp_path / "report.json"
    env = dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
               NIX_TEST_REPORT=str(report), NIX_TEST_EXIT=str(exit_code),
               NIX_TEST_STORE=str(fake_store))
    temporary = tmp_path / "temporary real"
    temporary.mkdir()
    if symlink_tmpdir:
        linked = tmp_path / "temporary link"
        linked.symlink_to(temporary, target_is_directory=True)
        env["TMPDIR"] = str(linked)
    else:
        env["TMPDIR"] = str(temporary)
    command = ["--command", "printf", "%s", "argument with spaces"]
    result = subprocess.run(["bash", str(scripts / "nix-dev.sh"), *command],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    observed = json.loads(report.read_text())
    assert observed["files"] == ["flake.lock", "flake.nix", "nix/toolchain.nix"]
    assert observed["flake"] == (project / "flake.nix").read_text()
    assert observed["lock"] == (project / "flake.lock").read_text()
    assert Path(observed["cwd"]).resolve() == project.resolve()
    assert observed["argv"][-len(command):] == command
    assert "--no-update-lock-file" in observed["argv"]
    profile = Path(observed["argv"][observed["argv"].index("--profile") + 1])
    assert profile.parent == project.resolve() / ".nix"
    assert profile.is_symlink()
    assert profile.resolve() == fake_store.resolve()
    assert Path(observed["stage"]).parent == temporary.resolve()
    assert not Path(observed["stage"]).exists()
    assert "private-test-value" not in result.stdout + result.stderr
