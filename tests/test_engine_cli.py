"""The SDK's `sasy engine` command: settings handling that needs no Docker."""

import datetime
import socket
import stat
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from dotenv import dotenv_values
from sasy import cli

SETTINGS = {
    "SASY_URL": "localhost:10089",
    "SASY_API_KEY": "synthetic-key",
    "TLS_CA_PATH": "/tmp/with space/ca.crt",
}


@pytest.fixture(autouse=True)
def isolated_local_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("SASY_ENGINE_PROFILE", raising=False)


def test_write_env_creates_a_private_file(tmp_path):
    path = tmp_path / ".env"
    written, kept = cli.write_env(path, SETTINGS)
    assert written == list(SETTINGS) and kept == []
    assert dotenv_values(path) == SETTINGS
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_env_keeps_existing_values_and_text(tmp_path):
    path = tmp_path / ".env"
    original = "# mine\nexport SASY_API_KEY=user-key\nOPENAI_API_KEY='x'"
    path.write_text(original)
    written, kept = cli.write_env(path, SETTINGS)
    assert kept == ["SASY_API_KEY"]
    assert written == ["SASY_URL", "TLS_CA_PATH"]
    assert path.read_text().startswith(original + "\n")
    values = dotenv_values(path)
    assert values["SASY_API_KEY"] == "user-key"
    assert values["TLS_CA_PATH"] == SETTINGS["TLS_CA_PATH"]
    # A second run has nothing left to add.
    saved = path.read_bytes()
    assert cli.write_env(path, SETTINGS) == ([], ["SASY_API_KEY"])
    assert path.read_bytes() == saved


def test_write_env_refuses_a_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("")
    (tmp_path / ".env").symlink_to(target)
    with pytest.raises(cli.EngineError):
        cli.write_env(tmp_path / ".env", SETTINGS)
    assert target.read_text() == ""


def test_write_env_rejects_values_that_would_change_meaning(tmp_path):
    with pytest.raises(cli.EngineError):
        cli.write_env(tmp_path / ".env", {"SASY_API_KEY": "a\nB=c"})
    assert not (tmp_path / ".env").exists()


def test_default_image_follows_the_sdk_version_unless_overridden(monkeypatch):
    monkeypatch.delenv("SASY_ENGINE_IMAGE", raising=False)
    assert cli.default_image().startswith("ghcr.io/sasy-labs/sasy:")
    monkeypatch.setenv("SASY_ENGINE_IMAGE", "sasy:dev")
    assert cli.default_image() == "sasy:dev"


def test_parser_defaults():
    args = cli._parser().parse_args(["engine", "start"])
    assert (args.name, args.port, args.image, args.volume) == (
        None,
        None,
        None,
        None,
    )
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["serve"])


def test_missing_docker_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cli.main(["engine", "status"]) == 1
    assert "docker was not found" in capsys.readouterr().err


def synthetic_ca_pem():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synthetic CA")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def test_tls_readiness_is_false_when_nothing_listens():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert cli._tls_ready(port, synthetic_ca_pem()) is False


def fake_docker(tmp_path, monkeypatch, *, stderr, code=1):
    """Put a docker executable on PATH that fails with the given message."""
    script = tmp_path / "docker"
    script.write_text(f"#!/bin/sh\necho '{stderr}' >&2\nexit {code}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))


def test_state_is_absent_only_when_docker_says_so(tmp_path, monkeypatch):
    fake_docker(tmp_path, monkeypatch, stderr="Error: No such object: sasy-engine")
    assert cli._state("sasy-engine") is None


def test_state_errors_surface_instead_of_reading_as_absent(tmp_path, monkeypatch):
    fake_docker(tmp_path, monkeypatch, stderr="Cannot connect to the Docker daemon")
    with pytest.raises(cli.EngineError):
        cli._state("sasy-engine")


CONFLICTED_DOCKER = """#!/bin/sh
# A concurrent start created the container just before this one tried to.
# Shell built-ins only: PATH holds nothing but this script.
log="${0%/*}/calls"
echo "$*" >> "$log"
case "$1" in
  info) echo 29.0 ;;
  inspect)
    if [ -e "${0%/*}/created" ]; then echo created; exit 0; fi
    echo "Error: No such object: sasy-engine" >&2; exit 1 ;;
  run)
    case "$2" in
      -d) : > "${0%/*}/created"
          echo 'Conflict. The container name "/sasy-engine" is already in use' >&2
          exit 125 ;;
      *) echo '{"api_key": "k", "ca_cert_pem": "PEM"}' ;;
    esac ;;
esac
"""


def test_a_name_conflict_leaves_the_other_start_container_alone(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(CONFLICTED_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(cli.EngineError, match="already in use"):
        cli.start("img", "sasy-engine", 10089, "vol", tmp_path, timeout=1)
    calls = (bin_dir / "calls").read_text().splitlines()
    assert not any(call.startswith("rm") for call in calls), calls


def test_start_leaves_a_container_another_start_just_created(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(CONFLICTED_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "created").write_text("")
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(cli.EngineError, match="is created"):
        cli.start("img", "sasy-engine", 10089, "vol", tmp_path, timeout=1)
    calls = (bin_dir / "calls").read_text().splitlines()
    assert not any(call.startswith(("rm", "run")) for call in calls), calls
