"""The ``sasy`` command installed with the SDK: run a local engine in Docker.

``sasy engine start`` publishes an authenticated TLS engine on loopback and
selects an owner-private connection profile in ``~/.sasy``. It does not write
into the current project unless ``--env-file`` is supplied.

This command is separate from the engine binary, which is also named ``sasy``
(``sasy serve``) and runs inside the container.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import dotenv_values

from sasy.local_profiles import ProfileError, load_profile, save_profile, selected_profile, validate_name

DEFAULT_NAME = "sasy-engine"
DEFAULT_PORT = 10089
STATE_DIR = "/data/local"

# Consume entire quoted values, including multiline ones, so their contents
# cannot be mistaken for settings. Same pattern as scripts/setup_local_env.py.
_BINDING = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?P<key>'[^'\n]+'|[^\s=#]+)[ \t]*"
    r"(?:=[ \t]*(?:'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|[^\r\n]*))?"
    r"[^\r\n]*(?:\r?\n|$)",
    re.MULTILINE,
)


class EngineError(RuntimeError):
    """A failure to report to the user without a traceback."""


def default_image() -> str:
    """Return the engine image matching this SDK's version.

    Returns:
        ``SASY_ENGINE_IMAGE`` if set, else the GHCR release for this version.
    """
    override = os.environ.get("SASY_ENGINE_IMAGE")
    if override:
        return override
    try:
        tag = version("sasy")
    except PackageNotFoundError:
        tag = "latest"
    return f"ghcr.io/sasy-labs/sasy:{tag}"


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a docker command and capture its output.

    Args:
        *args: Arguments after ``docker``.
        check: Raise :class:`EngineError` on a non-zero exit.

    Returns:
        The completed process.

    Raises:
        EngineError: Docker is missing, or the command failed with ``check``.
    """
    docker = shutil.which("docker")
    if docker is None:
        raise EngineError("docker was not found on PATH. Install Docker (or Colima, OrbStack).")
    result = subprocess.run([docker, *args], capture_output=True, text=True, timeout=600)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise EngineError(f"docker {args[0]} failed: {detail}")
    return result


def _require_daemon() -> None:
    """Fail with a clear message when the Docker daemon is not reachable.

    Raises:
        EngineError: The daemon does not answer.
    """
    if _docker("info", "--format", "{{.ServerVersion}}", check=False).returncode:
        raise EngineError("Docker is not running. Start Docker Desktop (or `colima start`) and try again.")


def _state(name: str) -> str | None:
    """Return the container's state (``running``, ...) or None if absent.

    Args:
        name: Container name.

    Returns:
        The Docker state string, or None when no such container exists.

    Raises:
        EngineError: Docker failed for another reason, so the state is unknown.
    """
    result = _docker("inspect", "--format", "{{.State.Status}}", name, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    if re.search(r"no such (object|container)", result.stderr, re.IGNORECASE):
        return None
    raise EngineError(f"docker inspect failed: {result.stderr.strip()}")


def _client_settings(*docker_args: str, init_args: tuple[str, ...] = ()) -> dict[str, str]:
    """Run ``sasy local-init`` in the image and parse its JSON.

    Args:
        *docker_args: ``docker`` arguments that end just before the command.

    Returns:
        The parsed client settings with ``api_key`` and ``ca_cert_pem``.

    Raises:
        EngineError: The output is not the expected JSON.
    """
    result = _docker(*docker_args, "sasy", "local-init", STATE_DIR, *init_args)
    try:
        settings = json.loads(result.stdout)
        if not all(isinstance(settings[key], str) and settings[key] for key in ("api_key", "ca_cert_pem")):
            raise ValueError("Invalid client settings")
        return {
            key: settings[key]
            for key in ("api_key", "ca_cert_pem", "client_entity", "admin_entity", "tenant", "trust_domain")
            if key in settings
        }
    except (ValueError, KeyError, TypeError) as error:
        raise EngineError("The engine image returned no client settings.") from error


def _tls_ready(port: int, ca_pem: str) -> bool:
    """Return whether the engine completes a TLS handshake on the port.

    Args:
        port: Host loopback port.
        ca_pem: The engine's CA certificate, used to verify the server.

    Returns:
        True when a verified handshake succeeds.
    """
    context = ssl.create_default_context(cadata=ca_pem)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as raw:
            with context.wrap_socket(raw, server_hostname="localhost"):
                return True
    except (OSError, ssl.SSLError):
        return False


def _wait_ready(name: str, port: int, ca_pem: str, timeout: float) -> None:
    """Wait until the engine accepts verified TLS connections.

    The engine's startup log line precedes TLS setup and binding, so readiness
    is a real handshake rather than a log match.

    Args:
        name: Container name.
        port: Host loopback port.
        ca_pem: The engine's CA certificate.
        timeout: Seconds to wait.

    Raises:
        EngineError: The container exited or did not become ready in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _state(name) != "running":
            logs = _docker("logs", name, check=False)
            tail = (logs.stdout + logs.stderr).strip()[-2000:]
            raise EngineError(f"The engine container stopped during startup:\n{tail}")
        if _tls_ready(port, ca_pem):
            return
        time.sleep(1)
    raise EngineError(f"The engine did not accept connections within {timeout:.0f}s; see `docker logs {name}`.")


def write_env(path: Path, settings: dict[str, str]) -> tuple[list[str], list[str]]:
    """Append settings missing from a dotenv file, keeping existing values.

    Args:
        path: The ``.env`` file; created with mode 0600 if absent.
        settings: Values to add when their key is not already bound.

    Returns:
        The keys written, and the keys whose different existing value was kept.

    Raises:
        EngineError: ``.env`` is a symlink, or a value cannot be written safely.
    """
    if path.is_symlink():
        raise EngineError(f"Refusing to write through the symlink {path}.")
    if any(c in v for v in settings.values() for c in ("\n", "\r")) or any("${" in v for v in settings.values()):
        raise EngineError("Settings cannot contain line breaks or ${...}.")
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError as error:
        raise EngineError(f"Cannot safely write {path}.") from error
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        os.close(fd)
        raise EngineError(f"Refusing to write an unsafe dotenv file {path}.")
    with os.fdopen(fd, "r+") as stream:
        existing = stream.read()
        present = {m["key"].strip("'").upper() for m in _BINDING.finditer(existing)}
        missing = {k: v for k, v in settings.items() if k not in present}
        current = dotenv_values(stream=io.StringIO(existing)) if existing else {}
        kept = [k for k in settings if k in present and current.get(k) != settings[k]]
        os.fchmod(stream.fileno(), 0o600)
        if missing and existing and not existing.endswith("\n"):
            stream.write("\n")
        for key, value in missing.items():
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            stream.write(f"{key}='{escaped}'\n")
    return list(missing), kept


def _running_settings(name: str, image: str, port: int, volume: str, explicit: set[str]) -> tuple[int, str, str]:
    """Verify loopback binding and reject incompatible explicit reuse settings."""
    info = json.loads(_docker("inspect", name).stdout)[0]
    bindings = info["NetworkSettings"]["Ports"].get("10089/tcp") or []
    if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
        raise EngineError(f"Container {name} does not have the expected loopback-only binding.")
    actual_port = int(bindings[0]["HostPort"])
    actual_image = info["Config"]["Image"]
    mounts = [m for m in info["Mounts"] if m["Destination"] == "/data" and m["Type"] == "volume"]
    if len(mounts) != 1:
        raise EngineError(f"Container {name} does not have a named /data volume.")
    actual_volume = mounts[0]["Name"]
    for field, requested, actual in [
        ("image", image, actual_image),
        ("port", port, actual_port),
        ("volume", volume, actual_volume),
    ]:
        if field in explicit and requested != actual:
            raise EngineError(f"Running engine {name} has a different {field}; stop it before changing {field}.")
    return actual_port, actual_image, actual_volume


def start(
    image: str,
    name: str,
    port: int,
    volume: str,
    project: Path | None = None,
    timeout: float = 120,
    *,
    profile: str = "local",
    env_file: Path | None = None,
    init_args: tuple[str, ...] = (),
    explicit: set[str] | None = None,
) -> int:
    """Start the engine (or reuse a running one) and select a local profile.

    Args:
        image: Engine image reference.
        name: Container name.
        port: Host loopback port to publish.
        volume: Named Docker volume holding keys, certificates and data.
        project: Reserved for compatibility; no project files are written.
        profile: Local profile to publish and select.
        env_file: Optional dotenv destination.
        init_args: Explicit local-init identity settings.
        explicit: Docker settings explicitly requested by the caller.
        timeout: Seconds to wait for the engine to become ready.

    Returns:
        The process exit code.
    """
    validate_name(profile)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", volume):
        raise EngineError("--volume must name a Docker volume, not a host path or mount specification.")
    _require_daemon()
    state = _state(name)
    if state == "running":
        print(f"Engine container {name} is already running.")
        port, image, volume = _running_settings(name, image, port, volume, explicit or set())
        client = _client_settings("exec", name, init_args=init_args)
        _wait_ready(name, port, client["ca_cert_pem"], timeout)
    else:
        if state in ("exited", "dead"):
            _docker("rm", name)
        elif state is not None:
            # A created container may belong to a start that is still running.
            raise EngineError(
                f"Container {name} is {state}. If no other `sasy engine start` "
                "is running, remove it with `sasy engine stop` and try again."
            )
        # Initialize before serving, so the server reuses these exact keys.
        client = _client_settings("run", "--rm", "-v", f"{volume}:/data", image, init_args=init_args)
        try:
            _docker(
                "run", "-d", "--init", "--name", name,
                "-p", f"127.0.0.1:{port}:10089", "-v", f"{volume}:/data", image,
            )  # fmt: skip
        except EngineError as error:
            # A failed run (such as a busy port) can leave a container that was
            # created but never started. After a name conflict the container
            # belongs to a concurrent start, so it is left alone.
            if "is already in use" not in str(error) and _state(name) == "created":
                _docker("rm", name, check=False)
            raise
        _wait_ready(name, port, client["ca_cert_pem"], timeout)
        print(f"Engine {name} is serving on localhost:{port} (volume {volume}).")

    data = save_profile(
        profile,
        {
            "url": f"localhost:{port}",
            "api_key": client["api_key"],
            "name": name,
            "image": image,
            "volume": volume,
            "port": str(port),
            **{k: v for k, v in client.items() if k not in ("api_key", "ca_cert_pem")},
        },
        client["ca_cert_pem"],
    )
    print(f"Selected local engine profile '{profile}' in ~/.sasy.")
    if env_file is not None:
        wanted = {"SASY_URL": data["url"], "SASY_API_KEY": data["api_key"], "TLS_CA_PATH": data["ca_path"]}
        written, kept = write_env(env_file, wanted)
        if written:
            print(f"Wrote {', '.join(written)} to {env_file}.")
        if kept:
            raise EngineError(
                f"Kept conflicting {', '.join(kept)} in {env_file}; remove or update those settings to use this engine."
            )
    return 0


def stop(name: str) -> int:
    """Stop the engine container; its volume and data are kept.

    Args:
        name: Container name.

    Returns:
        The process exit code.
    """
    _require_daemon()
    if _state(name) is None:
        print(f"No engine container named {name}.")
        return 0
    _docker("stop", name)
    _docker("rm", name)
    print(f"Stopped {name}. Its data volume is kept; `sasy engine start` reuses it.")
    return 0


def status(name: str) -> int:
    """Print whether the engine is running and where.

    Args:
        name: Container name.

    Returns:
        0 when running, 1 otherwise.
    """
    _require_daemon()
    state = _state(name)
    if state != "running":
        print(f"Engine {name} is not running.")
        return 1
    address = _docker("port", name, "10089/tcp").stdout.strip().splitlines()[0]
    image = _docker("inspect", "--format", "{{.Config.Image}}", name).stdout.strip()
    print(f"Engine {name} is running on {address} (image {image}).")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sasy",
        description="SASY SDK command line.",
        epilog="The engine binary's own commands (sasy serve, ...) run inside the engine image, not here.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    engine = commands.add_parser("engine", help="Run a local engine in Docker.")
    actions = engine.add_subparsers(dest="action", required=True)
    for action, text in [
        ("start", "Start the engine and select a local connection profile."),
        ("stop", "Stop the engine; its data volume is kept."),
        ("status", "Show whether the engine is running."),
    ]:
        sub = actions.add_parser(action, help=text, description=text)
        sub.add_argument("--name", default=None, help="Container name (default: profile container).")
        sub.add_argument("--profile", default=None, help="Local profile (default: selected profile, or local).")
        if action == "start":
            sub.add_argument("--image", default=None, help="Engine image.")
            sub.add_argument("--port", type=int, default=None)
            sub.add_argument("--volume", default=None, help="Data volume (default: NAME-data).")
            sub.add_argument("--timeout", type=float, default=120.0)
            sub.add_argument("--env-file", type=Path, help="Also write connection settings to this dotenv file.")
            for flag in ("client-entity", "admin-entity", "tenant", "trust-domain"):
                sub.add_argument(
                    f"--{flag}", help="Customize first-start identity; must match existing engine settings."
                )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``sasy`` command.

    Args:
        argv: Arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    args = _parser().parse_args(argv)
    try:
        profile = validate_name(args.profile if args.profile is not None else (selected_profile() or "local"))
        saved = load_profile(
            profile, missing_ok=(args.action == "start" or (args.profile is None and selected_profile() is None))
        )
        name = args.name or (
            saved["name"] if saved else (DEFAULT_NAME if profile == "local" else f"{DEFAULT_NAME}-{profile}")
        )
        if args.action == "start":
            if args.port is not None and not 1 <= args.port <= 65535:
                raise EngineError("--port must be between 1 and 65535.")
            flags: list[str] = []
            for flag in ("client_entity", "admin_entity", "tenant", "trust_domain"):
                value = getattr(args, flag)
                if value is not None:
                    flags.append(f"--{flag.replace('_', '-')}={value}")
            return start(
                image=args.image or (saved["image"] if saved else default_image()),
                name=name,
                port=args.port if args.port is not None else (int(saved["port"]) if saved else DEFAULT_PORT),
                volume=args.volume or (saved["volume"] if saved else f"{name}-data"),
                timeout=args.timeout,
                profile=profile,
                env_file=args.env_file,
                init_args=tuple(flags),
                explicit={key for key in ("image", "port", "volume") if getattr(args, key) is not None},
            )
        if args.action == "stop":
            return stop(name)
        return status(name)
    except (EngineError, ProfileError) as error:
        print(f"sasy: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
