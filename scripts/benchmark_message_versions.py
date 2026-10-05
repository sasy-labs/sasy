#!/usr/bin/env python3
"""Measure immutable snapshot RPCs against an owned local engine.

Example (from an installed development environment):
  python scripts/benchmark_message_versions.py --engine /absolute/path/to/sasy \
      --output /tmp/message-versions.json --large-session

No existing service or developer credentials are used. Results include request
and response protobuf sizes, warmed loopback TLS RPC times, and SDK telemetry
sanitation costs separately. Measurements do not imply WAN latency or a CI
performance threshold. Requires the repository's integration-test dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

import grpc
from sasy.capture import capture_events
from sasy.proto import observability_pb2 as obs
from sasy.proto import observability_pb2_grpc as obs_grpc
from sasy.proto import policy_engine_pb2 as pe


def _engine_class():
    root = Path(__file__).resolve().parents[1]
    # The benchmark reuses the integration suite's owned-engine fixture: it
    # starts a private engine process instead of touching a running service.
    # Some checkouts keep that fixture under a packaging directory rather than
    # in tests/, so look there too before giving up.
    relative = "tests/integration/conftest.py"
    candidates = [root / relative, *sorted(root.glob(f"*/*/*/{relative}"))]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("The owned-engine integration fixture is required")
    spec = importlib.util.spec_from_file_location("message_versions_owned_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine


def _summary(samples):
    ordered = sorted(samples)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))]}


def _requests(api, session, events, chunk_size, bases=None, reuse=False):
    for start in range(0, len(events), chunk_size):
        batch = events[start:start + chunk_size]
        if api == "register":
            yield obs.Events(session_id=session, events=batch)
        else:
            snapshots = []
            for index, event in enumerate(batch, start):
                snapshot = obs.EventSnapshot(event=event, reuse_dependencies=reuse)
                if bases is not None:
                    snapshot.base_id = bases[index]
                if api == "compact" and reuse:
                    projection = obs.Event.FromString(event.SerializeToString())
                    projection.ClearField("id")
                    projection.ClearField("principal")
                    snapshot.content_hash = hashlib.sha256(
                        b"sasy:event-content:v1\0" + projection.SerializeToString(deterministic=True)).digest()
                    snapshot.event.CopyFrom(obs.Event(id=event.id))
                snapshots.append(snapshot)
            yield obs.EventSnapshots(session_id=session, snapshots=snapshots)


def _send(client, api, requests):
    rpc = {"register": client.observability.RegisterEvents,
           "resolve": client.observability.ResolveEvents,
           "compact": client.observability.ResolveSnapshots}[api]
    ids, request_bytes, response_bytes = [], 0, 0
    started = time.perf_counter_ns()
    for request in requests:
        response = rpc(request, metadata=client.metadata, timeout=120)
        ids.extend(response.ids)
        request_bytes += request.ByteSize()
        response_bytes += response.ByteSize()
    return ids, (time.perf_counter_ns() - started) / 1e6, request_bytes, response_bytes


def _benchmark(engine, count, text_bytes, iterations, chunk_size):
    client = engine.client()
    client.policy.Health(pe.HealthRequest(), metadata=client.metadata, timeout=10)
    text = "x" * text_bytes
    events = [obs.Event(id=f"message-{index}", text=text, role=obs.USER) for index in range(count)]
    sanitation = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        clean = capture_events(events)
        sanitation.append((time.perf_counter_ns() - started) / 1e6)
        assert clean == events and clean[0] is not events[0]
    # Report the conservative scanner fallback as well as the plain-text
    # fast path. This synthetic credential is never a real provider secret.
    prefix = "https://example.invalid/path?api_key=synthetic-value\n"
    scanner_text = (prefix + text)[:text_bytes]
    scanner_events = [obs.Event(id=event.id, text=scanner_text, role=event.role) for event in events]
    scanner_samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        capture_events(scanner_events)
        scanner_samples.append((time.perf_counter_ns() - started) / 1e6)
    result = {"count": count, "text_bytes_each": text_bytes,
              "sdk_capture_events": _summary(sanitation),
              "sdk_capture_credential_syntax": _summary(scanner_samples), "rpc": {}}
    for api in ("register", "resolve", "compact"):
        result["rpc"][api] = {}
        phases = ["first_insert", "unchanged_resend", "mutation"]
        if api != "register":
            phases.insert(2, "unchanged_reuse")
        for phase in phases:
            elapsed, wire = [], []
            for _ in range(iterations):
                session = uuid.uuid4().hex
                original = None
                if phase != "first_insert":
                    original = _send(client, api, list(_requests(api, session, events, chunk_size)))[0]
                changed = events
                if phase == "mutation":
                    changed = [obs.Event(id=event.id, text="y" + text[1:], role=obs.USER) for event in events]
                # The reuse lane models consumption of a known base. Only
                # the compact API substitutes a reference for its full content.
                batches = list(_requests(api, session, changed, chunk_size,
                                         bases=original if api != "register" and phase in ("mutation", "unchanged_reuse") else None,
                                         reuse=phase == "unchanged_reuse"))
                ids, duration, sent, received = _send(client, api, batches)
                if phase.startswith("unchanged_"):
                    assert ids == original
                elif phase == "mutation" and api != "register":
                    assert all(new != old for new, old in zip(ids, original))
                elapsed.append(duration)
                wire.append({"request_bytes": sent, "response_bytes": received, "rpc_count": len(batches)})
            result["rpc"][api][phase] = dict(_summary(elapsed), wire=wire)
    return result


def _large_replay(engine, count, chunk_size):
    client, session = engine.client(), uuid.uuid4().hex
    events = [obs.Event(id=f"large-{index}", text="short synthetic message", role=obs.USER) for index in range(count)]
    batches = list(_requests("resolve", session, events, chunk_size))
    ids = _send(client, "resolve", batches)[0]
    channel = grpc.secure_channel(engine.address, grpc.ssl_channel_credentials(engine.root_certificates),
                                  options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)])
    try:
        updates = obs_grpc.ObservabilityUpdatesStub(channel)
        metadata = [("x-api-key", engine.tenant_a_key)]
        before = updates.GetState(obs.StateRequest(), metadata=metadata, timeout=120)
        result = {"count": count, "text_bytes_each": len(events[0].text)}
        for phase, api, requests in [("unchanged_resend", "resolve", batches), ("unchanged_reuse", "resolve", list(
                _requests("resolve", session, events, chunk_size, bases=ids, reuse=True))),
                ("compact_reuse", "compact", list(_requests("compact", session, events, chunk_size, bases=ids, reuse=True)))]:
            repeated, elapsed, sent, received = _send(client, api, requests)
            after = updates.GetState(obs.StateRequest(), metadata=metadata, timeout=120)
            assert repeated == ids and before.sequence == after.sequence
            assert len(after.events) == count
            result[phase] = {"elapsed_ms": elapsed, "request_bytes": sent,
                             "response_bytes": received, "sequence_delta": after.sequence - before.sequence}
        return result
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 1000])
    parser.add_argument("--text-bytes", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--large-session", action="store_true", help="Also replay 32769 short messages, beyond the broadcast ring")
    args = parser.parse_args()
    if min([args.iterations, args.text_bytes, args.chunk_size, *args.sizes]) < 1:
        parser.error("counts and sizes must be positive")
    source_binary = args.engine.expanduser().resolve(strict=True)
    # Pin one executable for every owned process: a concurrent rebuild of the
    # supplied target path must not mix engine versions across measurements.
    with tempfile.TemporaryDirectory(prefix="sasy-message-versions-binary-") as directory:
        binary = Path(directory) / source_binary.name
        shutil.copy2(source_binary, binary)
        engine_type = _engine_class()
        result = {"platform": platform.platform(), "python": sys.version,
                  "engine_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                  "engine_path": str(source_binary),
                  "inferred_build_profile": source_binary.parent.name if source_binary.parent.name in ("debug", "release") else "unknown",
                  "transport": "owned loopback TLS", "chunk_size": args.chunk_size,
                  "notes": "RPC timing excludes request construction and setup; sanitation measured separately. RegisterEvents mutations overwrite; snapshot mutations preserve history.",
                  "cases": []}
        for count in args.sizes:
            with tempfile.TemporaryDirectory(prefix="sasy-message-versions-") as directory:
                engine = engine_type(binary, Path(directory), {})
                try:
                    result["cases"].append(_benchmark(engine, count, args.text_bytes, args.iterations, args.chunk_size))
                finally:
                    engine.stop()
        if args.large_session:
            with tempfile.TemporaryDirectory(prefix="sasy-message-versions-large-") as directory:
                engine = engine_type(binary, Path(directory), {})
                try:
                    result["large_replay"] = _large_replay(engine, 32769, args.chunk_size)
                finally:
                    engine.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
