"""Content fingerprints and a bounded cache for acknowledged capture results."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

from sasy import capture
from sasy.proto.observability_pb2 import Event

#: Prefix of the immutable message-version IDs that ResolveEvents and
#: ResolveSnapshots issue. The adapters depend only on such IDs, because a
#: version cannot change after it is recorded. IDs from RegisterEvents have no
#: prefix; they are valid graph nodes, but the record behind one can be updated.
VERSION_ID_PREFIX = "sasy:mv1:"

_CONTENT_DOMAIN = b"sasy:event-content:v1\0"
_RAW_DOMAIN = b"sasy:event-capture-cache:v1\0"


def content_digest(event: Event) -> bytes:
    """Hash the captured Event wire representation shared with the server.

    Identity and authenticated principal are checked separately by the server.
    Optional presence and repeated-field order remain significant. Unknown
    fields are discarded recursively, matching server-side protobuf decoding.
    """
    content = Event()
    content.CopyFrom(event)
    content.ClearField("id")
    content.ClearField("principal")
    content.DiscardUnknownFields()
    return hashlib.sha256(_CONTENT_DOMAIN + content.SerializeToString(deterministic=True)).digest()


def raw_fingerprint(event: Event, session_id: str | None, base_id: str,
                    sanitizer: Callable[..., object]) -> bytes:
    """Inspect every current field; retain only a process-local cache key.

    Length limits are enforced even on cache hits. An exact acknowledged raw
    fingerprint has already passed the deterministic nested capture work budget;
    changed fields or capture limits must pass capture again. No raw fingerprint
    is transmitted, logged, or used as an authorization/version identity.
    """
    fields = ([event.text] if event.HasField("text") else [])
    fields.extend(tool.arguments for tool in event.tools if tool.HasField("arguments"))
    if event.HasField("derived_from") and event.derived_from.HasField("arguments"):
        fields.append(event.derived_from.arguments)
    if any(len(value) > capture.MAX_CAPTURE_LENGTH for value in fields):
        raise ValueError("telemetry capture text exceeds character limit")
    fingerprint = hashlib.sha256(_RAW_DOMAIN)
    for setting in (capture.MAX_CAPTURE_LENGTH, capture.MAX_CAPTURE_DEPTH,
                    id(sanitizer), id(capture.capture_text)):
        fingerprint.update(str(setting).encode("ascii") + b"\0")
    for value in (session_id, base_id):
        if value is None:
            fingerprint.update(b"\0")
        else:
            encoded = value.encode("utf-8")
            fingerprint.update(b"\1" + len(encoded).to_bytes(8, "big") + encoded)
    fingerprint.update(event.SerializeToString(deterministic=True))
    return fingerprint.digest()


class CaptureDigestCache:
    """Thread-safe LRU containing only fixed-size opaque digest pairs.

    Work happens outside the lock. Concurrent misses may repeat capture, while
    failures cannot publish speculative results. Clearing advances an epoch so
    an already-running RPC cannot repopulate a cleared cache when it completes.
    """
    def __init__(self, capacity: int = 4096):
        if capacity < 1:
            raise ValueError("Cache capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._entries: OrderedDict[bytes, bytes] = OrderedDict()
        self._epoch = 0

    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def get(self, key: bytes) -> bytes | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def commit(self, epoch: int, pending: dict[bytes, bytes]) -> None:
        with self._lock:
            if epoch != self._epoch:
                return
            for key, value in pending.items():
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._epoch += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
