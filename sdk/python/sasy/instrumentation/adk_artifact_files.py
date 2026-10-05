"""Native FileArtifactService version reservations used by observed writes."""
from __future__ import annotations

import os
import shutil
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from . import adk_state as resources

_reservation: ContextVar[Reservation | None] = ContextVar("sasy_adk_file_reservation", default=None)
_methods: dict = {}
_public_methods: dict = {}
_allocate = None


@dataclass
class Reservation:
    directory: Path
    version: int
    staging: Path
    final: Path
    consumed: bool = False
    cancelled: bool = False
    lock: RLock = field(default_factory=RLock)

    def finish(self):
        # Cancellation cannot stop asyncio.to_thread. An unstarted worker must
        # fail before consuming this reservation; a started worker owns cleanup
        # and can publish only the dependencies recorded before it was started.
        with self.lock:
            if not self.consumed:
                self.cancelled = True
                shutil.rmtree(self.staging, ignore_errors=True)


def install():
    global _allocate
    if _allocate is not None:
        return
    from google.adk.artifacts import file_artifact_service as native
    cls = native.FileArtifactService
    for name in ("_save_artifact_sync", "_load_artifact_sync", "_artifact_dir", "_base_root", "_scope_root",
                 "_list_versions_sync", "_get_artifact_version_sync", "_list_artifact_keys_sync"):
        if hasattr(cls, name):
            _methods[name] = getattr(cls, name)
    for name in ("save_artifact", "load_artifact", "list_versions", "get_artifact_version", "list_artifact_keys"):
        _public_methods[name] = getattr(cls, name)
    _allocate = native._reserve_version_dir

    def reserve(directory):
        pending = _reservation.get()
        if pending is None:
            return _allocate(directory)
        with pending.lock:
            if pending.cancelled or pending.consumed or directory != pending.directory:
                resources._fail("Artifact save changed its reserved publication")
            pending.consumed = True
            return pending.version, pending.staging, pending.final

    native._reserve_version_dir = reserve


def qualified(service):
    from google.adk.artifacts.file_artifact_service import FileArtifactService
    return (type(service) is FileArtifactService and bool(_methods)
            and all(getattr(getattr(service, name), "__func__", None) is method
                    for name, method in _methods.items())
            and all(getattr(getattr(getattr(service, name), "__func__", None), "__sasy_original__", None) is method
                    for name, method in _public_methods.items()))


def allocate(service, app_name, user_id, session_id, filename):
    from google.adk.artifacts import file_artifact_service as native
    _validate_scope(service, app_name, user_id, session_id, filename)
    directory = service._artifact_dir(app_name, user_id, session_id, filename)
    scope = service._scope_root(service._base_root(app_name, user_id), session_id, filename).resolve()
    canonical = directory.relative_to(scope).as_posix()
    if filename.startswith("user:"):
        canonical = "user:" + canonical
    if filename != canonical or native._is_reserved_artifact_name(directory.name):
        resources._fail("Concurrent file artifacts require canonical, non-reserved filenames")
    # This native filesystem reservation is synchronous and has no suspension
    # point: cancellation cannot leave an allocation whose result was lost.
    assert _allocate is not None
    version, staging, final = _allocate(directory)
    pending = Reservation(directory, version, staging, final)
    try:
        _validate_scope(service, app_name, user_id, session_id, filename)
    except BaseException:
        pending.finish()
        raise
    return pending


def validate_name(service, app_name, user_id, session_id, filename):
    _validate_scope(service, app_name, user_id, session_id, filename)
    directory = service._artifact_dir(app_name, user_id, session_id, filename)
    scope = service._scope_root(service._base_root(app_name, user_id), session_id, filename).resolve()
    canonical = directory.relative_to(scope).as_posix()
    if filename.startswith("user:"):
        canonical = "user:" + canonical
    if filename != canonical:
        resources._fail("Concurrent file artifacts require canonical filenames")


def _validate_scope(service, app_name, user_id, session_id, filename):
    if session_id == "user":
        resources._fail("File artifact session ID 'user' conflicts with the user namespace")
    # ADK resolves paths, which hides symlinks and filesystem case aliases.
    # Check the supplied spelling before that resolution, including scope keys.
    path = Path(service.root_dir)
    pieces = ["apps", app_name, "users", user_id]
    if not filename.startswith("user:"):
        pieces.extend(["sessions", session_id])
    pieces.append("artifacts")
    pieces.extend(filename.removeprefix("user:").split("/"))
    for piece in pieces:
        if not isinstance(piece, str) or piece in ("", ".", "..") or "/" in piece or "\\" in piece:
            resources._fail("File artifacts require canonical scope and filename components")
        candidate = path / piece
        if candidate.is_symlink():
            resources._fail("File artifact paths cannot use symbolic link aliases")
        if candidate.exists():
            with os.scandir(path) as entries:
                if not any(entry.name == piece for entry in entries):
                    resources._fail("File artifact paths must use their exact filesystem spelling")
        path = candidate
