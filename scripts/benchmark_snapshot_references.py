#!/usr/bin/env python3
"""Compare full snapshots and compact references through the complete Python SDK.

Uses synthetic inputs and an owned loopback engine. No provider credentials or
existing services are used. Run with --engine /path/to/sasy --output /tmp/sdk.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import sasy
import sasy.config as sdk_config
from benchmark_message_versions import _engine_class, _summary
from sasy.auth.hooks import APIKeyAuthHook
from sasy.observability import api
from sasy.proto import observability_pb2 as obs


def _sdk_cases(engine, sizes, text_bytes, iterations):
    sdk_config._config = sdk_config.SasyConfig(_env_file=None)
    sdk_config.configure(url=engine.address, ca_path=str(engine.root / "tls/ca.pem"),
                         cert_path="", key_path="", auth_hook=APIKeyAuthHook(engine.tenant_a_key))
    sdk_config.get_config().channel_pool_size = 1
    cases = []
    for count in sizes:
        for shape in ("plain", "credential_syntax"):
            prefix = "" if shape == "plain" else "https://example.invalid/path?api_key=synthetic-value\n"
            text = (prefix + "x" * text_bytes)[:text_bytes]
            snapshots = [obs.EventSnapshot(event=obs.Event(id=f"message-{i}", text=text, role=obs.USER))
                         for i in range(count)]
            with sasy.session(session_id=uuid.uuid4().hex, end_on_exit=False):
                ids = api.resolve_events(snapshots)
                for item, version in zip(snapshots, ids, strict=True):
                    item.base_id = version
                    item.reuse_dependencies = True
                original = [item.SerializeToString() for item in snapshots]
                row = {"count": count, "text_bytes_each": text_bytes, "shape": shape, "modes": {}}
                for mode in ("full", "compact_cold", "compact_warm"):
                    compact = mode != "full"
                    api._snapshot_cache.clear()
                    assert api.resolve_events(snapshots, compact=compact) == ids
                    samples = []
                    for _ in range(iterations):
                        if mode == "compact_cold":
                            api._snapshot_cache.clear()
                        started = time.perf_counter_ns()
                        actual = api.resolve_events(snapshots, compact=compact)
                        samples.append((time.perf_counter_ns() - started) / 1e6)
                        assert actual == ids
                    assert [item.SerializeToString() for item in snapshots] == original
                    # Inspect one additional real RPC outside the timing loop.
                    # ByteSize work must not distort end-to-end SDK latency.
                    wire = []
                    real_get_stub = api.get_stub

                    class MeasuredStub:
                        def __init__(self, delegate):
                            self.delegate = delegate

                        def __getattr__(self, name):
                            rpc = getattr(self.delegate, name)

                            def measured(request, **kwargs):
                                reply = rpc(request, **kwargs)
                                wire.append({"rpc": name, "request_bytes": request.ByteSize(),
                                             "response_bytes": reply.ByteSize()})
                                return reply
                            return measured

                    try:
                        api.get_stub = lambda *args, **kwargs: MeasuredStub(real_get_stub(*args, **kwargs))
                        assert api.resolve_events(snapshots, compact=compact) == ids
                    finally:
                        api.get_stub = real_get_stub
                    row["modes"][mode] = dict(_summary(samples), wire=wire)
                cases.append(row)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 1000])
    parser.add_argument("--text-bytes", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if min([*args.sizes, args.text_bytes, args.iterations]) < 1:
        parser.error("sizes and iteration count must be positive")
    source = args.engine.expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="sasy-reference-benchmark-") as directory:
        root = Path(directory)
        binary = root / source.name
        shutil.copy2(source, binary)
        runtime = root / "runtime"
        runtime.mkdir()
        report = {"engine_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                  "engine_source": str(source), "build_profile": source.parent.name,
                  "platform": platform.platform(), "python": sys.version,
                  "sdk_source": str(Path(sasy.__file__).resolve()), "transport": "owned loopback TLS",
                  "method": "Full synchronous SDK call, one RPC per sample, including cloning, content fingerprinting, capture on misses, protobuf construction and transport. Initial registration and warmup excluded. compact_cold clears only the SDK capture cache outside timing; server content fingerprints may be warm."}
        engine = _engine_class()(binary, runtime, {})
        try:
            report["cases"] = _sdk_cases(engine, args.sizes, args.text_bytes, args.iterations)
        finally:
            sdk_config.reset_channel()
            api._snapshot_cache.clear()
            engine.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
