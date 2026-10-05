"""Nix imports only the explicitly selected public engine build inputs."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_source(tmp_path):
    root = tmp_path / "private-checkout"
    (root / "nix").mkdir(parents=True)
    for name in ("stage-source.py", "build.sh", "toolchain.nix", "package.nix", "docker-root.nix"):
        shutil.copy2(ROOT / "nix" / name, root / "nix" / name)
    for name in ("flake.nix", "flake.lock"):
        shutil.copy2(ROOT / name, root / name)
    (root / "proto").mkdir()
    (root / "proto/example.proto").write_text("// Selected public input\n")
    (root / "nix/source-files.json").write_text(json.dumps([
        "proto/example.proto",
    ]))
    (root / ".env").write_text("DO_NOT_COPY_PRIVATE_CREDENTIAL\n")
    (root / "private-source.rs").write_text("DO_NOT_COPY_PRIVATE_SOURCE\n")
    return root


def test_stage_excludes_unlisted_files_and_preserves_the_proto_layout(tmp_path):
    root = make_source(tmp_path)
    stage = tmp_path / "stage"
    subprocess.run([sys.executable, str(root / "nix/stage-source.py"), str(stage)], check=True)
    paths = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() or p.is_symlink()}
    assert paths == {
        "flake.nix", "flake.lock", "nix/source-files.json", "nix/toolchain.nix",
        "nix/package.nix", "nix/docker-root.nix", "nix/stage-source.py", "nix/build.sh",
        "proto/example.proto",
    }
    assert (stage / "proto/example.proto").read_text() == "// Selected public input\n"
    # Refuse to overwrite an earlier stage or merge unlisted files into one.
    repeat = subprocess.run([sys.executable, str(root / "nix/stage-source.py"), str(stage)], capture_output=True)
    assert repeat.returncode != 0


def test_stage_refuses_selected_file_symlink_to_private_content(tmp_path):
    root = make_source(tmp_path)
    (root / "proto/example.proto").unlink()
    (root / "proto/example.proto").symlink_to(root / ".env")
    stage = tmp_path / "stage"
    result = subprocess.run([sys.executable, str(root / "nix/stage-source.py"), str(stage)], capture_output=True)
    assert result.returncode != 0
    assert not (stage / "proto/example.proto").exists()
    assert b"DO_NOT_COPY_PRIVATE_CREDENTIAL" not in result.stdout + result.stderr


def test_engine_inputs_are_selected_public_files():
    files = json.loads((ROOT / "nix/source-files.json").read_text())
    assert len(files) == len(set(files))
    assert "Cargo.lock" in files
    assert "crates/sasy-binary/src/main.rs" in files
    assert all(p.startswith(("crates/", "proto/", "souffle/", "Cargo.")) for p in files)
    manifest = ROOT / "release/public-core/manifest.json"
    if manifest.exists():
        public = set(json.loads(manifest.read_text())["core_files"])
        assert set(files) <= public
        required_rust = {p for p in public if p.startswith("") and p.endswith(".rs")}
    else:
        required_rust = {str(p.relative_to(ROOT)) for p in (ROOT / "crates").glob("*/src/**/*.rs")}
        required_rust.update(str(p.relative_to(ROOT)) for p in (ROOT / "crates").glob("*/build.rs"))
    assert required_rust <= set(files), f"Nix build omits Rust sources: {sorted(required_rust - set(files))}"


def test_stage_refuses_selected_file_through_symlinked_directory(tmp_path):
    root = make_source(tmp_path)
    (root / "proto").rename(root / "private-directory")
    (root / "private-directory/example.proto").write_text("DO_NOT_COPY_PRIVATE_CREDENTIAL")
    (root / "proto").symlink_to("private-directory")
    stage = tmp_path / "stage"
    result = subprocess.run([sys.executable, str(root / "nix/stage-source.py"), str(stage)], capture_output=True)
    assert result.returncode != 0
    assert not (stage / "proto/example.proto").exists()
    assert b"DO_NOT_COPY_PRIVATE_CREDENTIAL" not in result.stdout + result.stderr


def test_build_launcher_uses_canonical_stage_and_preserves_caller_arguments(tmp_path):
    import os

    root = make_source(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    fake_nix = tools / "nix"
    fake_nix.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, sys
stage = pathlib.Path(next(a[5:].split('#', 1)[0] for a in sys.argv if a.startswith('path:')))
assert stage == stage.resolve()
assert not (stage / '.env').exists()
assert not (stage / 'private-source.rs').exists()
pathlib.Path(os.environ['NIX_TEST_REPORT']).write_text(json.dumps({
    'stage': str(stage), 'argv': sys.argv[1:], 'cwd': os.getcwd(),
}))
sys.exit(37)
""")
    fake_nix.chmod(0o755)
    temporary = tmp_path / "temporary real"
    temporary.mkdir()
    linked = tmp_path / "temporary link"
    linked.symlink_to(temporary)
    report = tmp_path / "report.json"
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
               TMPDIR=str(linked), NIX_TEST_REPORT=str(report))
    args = ["--out-link", "caller result"]
    result = subprocess.run(["bash", str(root / "nix/build.sh"), *args], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 37
    observed = json.loads(report.read_text())
    assert observed["argv"][-len(args):] == args
    assert "--no-update-lock-file" in observed["argv"]
    option = observed["argv"].index("--option")
    assert observed["argv"][option:option + 3] == ["--option", "sandbox", "true"]
    assert observed["cwd"] == str(tmp_path)
    assert not Path(observed["stage"]).exists()
    assert not list(temporary.iterdir())
    assert "DO_NOT_COPY_PRIVATE_CREDENTIAL" not in result.stdout + result.stderr
