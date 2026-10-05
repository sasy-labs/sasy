"""Explicitly selected, owner-private local engine connections in ``~/.sasy``.

The SDK never chooses an endpoint merely because a profile directory exists.
Only ``engine start`` selects a profile, or ``SASY_ENGINE_PROFILE`` names one.
CA files are immutable; the profile and selection are published atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ProfileError(RuntimeError):
    """Invalid or unsafe local engine configuration."""


def validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ProfileError("Profile names must contain 1–64 letters, digits, '_' or '-'.")
    return name


def profile_root() -> Path:
    return Path.home() / ".sasy"


def _check_private(fd: int, directory: bool) -> None:
    info = os.fstat(fd)
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or info.st_uid != os.getuid():
        raise ProfileError("Local engine settings must be ordinary files/directories owned by you.")
    if stat.S_IMODE(info.st_mode) & 0o077 or (not directory and info.st_nlink != 1):
        raise ProfileError("Local engine settings must be private (directories 0700, files 0600).")


@contextmanager
def _directory(parent: int, name: str, create: bool) -> Iterator[int]:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        _check_private(fd, True)
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _root(create: bool = False) -> Iterator[int]:
    home = os.open(Path.home(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _directory(home, ".sasy", create) as root:
            yield root
    finally:
        os.close(home)


def _read(parent: int, name: str) -> str:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        _check_private(fd, False)
        with os.fdopen(fd, "r", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _atomic_write(parent: int, name: str, value: str) -> None:
    # Refuse existing symlinks/hard links even though replace would not follow
    # them. Do not silently turn an unsafe settings tree into a trusted one.
    try:
        _read(parent, name)
    except FileNotFoundError:
        pass
    temporary = f".tmp-{uuid.uuid4().hex}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass


def selected_profile() -> str | None:
    """Read the explicit environment selector or the last startup selection."""
    if "SASY_ENGINE_PROFILE" in os.environ:
        return validate_name(os.environ["SASY_ENGINE_PROFILE"])
    try:
        # A pre-existing ~/.sasy directory alone does not select a connection.
        # Opening the tree below still validates ownership and every component.
        os.lstat(profile_root() / "selected-profile")
        with _root() as root:
            return validate_name(_read(root, "selected-profile").strip())
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ProfileError("Cannot safely read ~/.sasy/selected-profile.") from error


def load_profile(name: str | None = None, *, missing_ok: bool = False) -> dict[str, str] | None:
    """Return an explicitly named/selected profile; a missing selection errors."""
    name = validate_name(name) if name is not None else selected_profile()
    if name is None:
        return None
    try:
        with _root() as root, _directory(root, "profiles", False) as profiles:
            with _directory(profiles, name, False) as directory:
                data = json.loads(_read(directory, "profile.json"))
                if not isinstance(data, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in data.items()
                ):
                    raise ProfileError("Invalid local engine profile.")
                for field in ("url", "api_key", "ca_file", "name", "image", "volume", "port"):
                    if not data.get(field):
                        raise ProfileError(f"Local engine profile is missing {field}.")
                ca_file = data["ca_file"]
                if not re.fullmatch(r"ca-[0-9a-f]{64}\.crt", ca_file):
                    raise ProfileError("Invalid local engine CA filename.")
                pem = _read(directory, ca_file)
                if hashlib.sha256(pem.encode()).hexdigest() != ca_file[3:-4]:
                    raise ProfileError("Local engine CA does not match its profile.")
                data["ca_path"] = str(profile_root() / "profiles" / name / ca_file)
                return data
    except FileNotFoundError as error:
        if missing_ok:
            return None
        raise ProfileError(
            f"Local engine profile '{name}' is missing; run sasy engine start --profile {name}."
        ) from error
    except (OSError, ValueError) as error:
        raise ProfileError(f"Cannot safely read local engine profile '{name}'.") from error


def save_profile(name: str, settings: dict[str, str], ca_pem: str) -> dict[str, str]:
    """Publish a complete profile and select it, without writing to a project."""
    validate_name(name)
    ca_file = f"ca-{hashlib.sha256(ca_pem.encode()).hexdigest()}.crt"
    data = {**settings, "ca_file": ca_file}
    try:
        with _root(True) as root, _directory(root, "profiles", True) as profiles:
            with _directory(profiles, name, True) as directory:
                _atomic_write(directory, ca_file, ca_pem)
                _atomic_write(directory, "profile.json", json.dumps(data, indent=2) + "\n")
            _atomic_write(root, "selected-profile", name + "\n")
    except OSError as error:
        raise ProfileError(f"Cannot safely save local engine profile '{name}'.") from error
    return {**data, "ca_path": str(profile_root() / "profiles" / name / ca_file)}
