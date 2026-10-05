#!/usr/bin/env python3
"""Validate (and optionally analyze) a Soufflé policy file.

Tries the running SASY service's ``ValidatePolicy`` RPC first.
If the service isn't reachable, falls back to running
``sugar.py`` + ``souffle --show=transformed-datalog`` locally.

Usage:
    python scripts/validate_policy.py path/to/policy.dl
    python scripts/validate_policy.py --analyze path/to/policy.dl
    python scripts/validate_policy.py --analyze --json path/to/policy.dl
    python scripts/validate_policy.py --no-rpc path/to/policy.dl   # force local

Examples are also reachable via Make:
    make souffle-validate FILE=path/to/policy.dl
    python scripts/validate_policy.py --analyze examples/message-flow/policy.dl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUGAR_PY = REPO_ROOT / "." / "souffle" / "sugar.py"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", help="Path to the Soufflé policy .dl file")
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Also run static analyses (contradictions/redundancy/reachability)",
    )
    p.add_argument(
        "--json", action="store_true", help="Emit JSON output instead of text"
    )
    p.add_argument(
        "--no-rpc",
        action="store_true",
        help="Skip the RPC and run local sugar.py + souffle directly",
    )
    p.add_argument(
        "--address",
        default=os.environ.get("SASY_URL"),
        help="SASY gRPC address (default: SASY_URL env var)",
    )
    args = p.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"error: {policy_path} not found", file=sys.stderr)
        return 2
    policy_source = policy_path.read_text()

    if args.no_rpc:
        return run_local(policy_source, args.analyze, args.json)
    if rpc_available(args.address):
        rc = run_rpc(policy_source, args.analyze, args.json, args.address)
        if rc is not None:
            return rc
    return run_local(policy_source, args.analyze, args.json)


# ────────────────────────────────────────────────────────
#  RPC path
# ────────────────────────────────────────────────────────


def rpc_available(address: str | None) -> bool:
    """Cheap probe: does the sasy SDK import and is a service
    reachable on the configured endpoint? Returns True if we
    should at least try the RPC."""
    try:
        import sasy  # noqa: F401
    except ImportError:
        return False
    return address is not None or os.environ.get("SASY_URL") is not None


def run_rpc(
    policy_source: str, analyze: bool, as_json: bool, address: str | None
) -> int | None:
    """Try the RPC. Return an exit code on success; return None
    if the RPC fails in a way that suggests we should fall back."""
    try:
        import grpc
        from sasy.config import configure
        from sasy.policy import validate_policy
    except ImportError as e:
        print(f"warning: sasy SDK not importable: {e}", file=sys.stderr)
        return None

    if address:
        # A remote address is assumed to present a certificate the system trust
        # store already accepts, so the locally generated client certificate and
        # CA bundle are cleared for it; a localhost address keeps whatever the
        # environment configured (typically the certificates `make certs`
        # creates). An address written as 127.0.0.1 counts as remote here.
        is_cloud = "localhost" not in address
        configure(
            url=address,
            ca_path="" if is_cloud else None,
            cert_path="" if is_cloud else None,
            key_path="" if is_cloud else None,
        )

    try:
        resp = validate_policy(policy_source, run_analyses=analyze)
    except grpc.RpcError as e:
        print(
            f"warning: RPC failed ({e.code().name}: {e.details()}); "
            f"falling back to local",
            file=sys.stderr,
        )
        return None

    if as_json:
        # ValidatePolicyResponse → dict via google.protobuf.json_format.
        from google.protobuf.json_format import MessageToDict

        print(json.dumps(MessageToDict(resp, preserving_proto_field_name=True), indent=2))
    else:
        if resp.valid:
            print("✓ valid")
        else:
            print("✗ invalid:")
            print(resp.error_output)
        if analyze and resp.HasField("analyses"):
            print_analyses(resp.analyses)
    return 0 if resp.valid else 1


def print_analyses(analyses) -> None:
    print()
    print("=== Static analysis report ===")
    if analyses.broad_rules:
        print(f"\nBroad rules ({len(analyses.broad_rules)}):")
        for n in analyses.broad_rules:
            print(f"  • {n}")
    print(f"\nContradictions ({len(analyses.contradictions)}):")
    for f in analyses.contradictions:
        print(
            f"  [{f.category}] allow@{f.allow_location}  vs  deny@{f.deny_location}"
        )
        if f.message:
            print(f"      {f.message}")
        if getattr(f, "allow_body", ""):
            print(f"      allow: {f.allow_body}")
        if getattr(f, "deny_body", ""):
            print(f"      deny:  {f.deny_body}")
    print(f"\nRedundancies ({len(analyses.redundancies)}):")
    for f in analyses.redundancies:
        print(
            f"  [{f.head_relation}] {f.redundant_location}  subsumed by  "
            f"{f.covered_by_location}"
        )
        if getattr(f, "redundant_body", ""):
            print(f"      redundant:   {f.redundant_body}")
        if getattr(f, "covered_by_body", ""):
            print(f"      covered by:  {f.covered_by_body}")
    print("\nReachability:")
    for f in analyses.reachability:
        print(
            f"  {f.target}: {len(f.disjuncts)} disjunct(s), "
            f"{f.pruned} pruned, opaque {{{', '.join(f.opaque)}}}"
        )
        for i, d in enumerate(f.disjuncts, start=1):
            print(f"    [{i}] {d}")


# ────────────────────────────────────────────────────────
#  Local fallback path
# ────────────────────────────────────────────────────────


def run_local(policy_source: str, analyze: bool, as_json: bool) -> int:
    """Run the sugar.py preprocessor and souffle parse-check
    locally."""
    if not SUGAR_PY.exists():
        print(f"error: {SUGAR_PY} not found", file=sys.stderr)
        return 2

    # Write the policy to a temp file so sugar.py can resolve includes.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dl", delete=False
    ) as f:
        f.write(policy_source)
        tmp_path = Path(f.name)

    try:
        sugar = subprocess.run(
            [
                "python3",
                str(SUGAR_PY),
                "--resolve-includes",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if sugar.returncode != 0:
            print("✗ sugar.py preprocessing failed:", file=sys.stderr)
            print(sugar.stderr, file=sys.stderr)
            return 1
        desugared = sugar.stdout

        # Validate via souffle parse-check.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dl", delete=False
        ) as df:
            df.write(desugared)
            desugared_path = Path(df.name)
        try:
            souffle = subprocess.run(
                ["souffle", "--show=transformed-datalog", str(desugared_path)],
                capture_output=True,
                text=True,
            )
        finally:
            desugared_path.unlink(missing_ok=True)
        if souffle.returncode != 0:
            print("✗ souffle validation failed:", file=sys.stderr)
            print(souffle.stderr or souffle.stdout, file=sys.stderr)
            return 1

        if not analyze:
            print("✓ valid (local)")
            return 0

        # Run the policy-analyze binary on the desugared output.
        binary = (
            REPO_ROOT
            / "."
            / "target"
            / "debug"
            / "policy-analyze"
        )
        if not binary.exists():
            print(
                "info: building policy-analyze binary…",
                file=sys.stderr,
            )
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "-p",
                    "sasy-policy",
                    "--bin",
                    "policy-analyze",
                    "--quiet",
                ],
                cwd=REPO_ROOT / ".",
                check=True,
            )
        analyze_args = [str(binary)]
        if as_json:
            analyze_args.append("--json")
        analyze_args.append("/dev/stdin")
        analyze_proc = subprocess.run(
            analyze_args,
            input=desugared,
            text=True,
        )
        return analyze_proc.returncode
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
