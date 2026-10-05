#!/usr/bin/env bash
# Start the locked development shell without putting checkout files in the store.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
umask 077
flake_stage="$(mktemp -d "${TMPDIR:-/tmp}/sasy-nix-shell.XXXXXXXX")"
trap 'rm -rf -- "$flake_stage"' EXIT
flake_stage="$(cd -- "$flake_stage" && pwd -P)"
cp -- "$repo_root/flake.nix" "$repo_root/flake.lock" "$flake_stage/"
mkdir -- "$flake_stage/nix"
cp -- "$repo_root/nix/toolchain.nix" "$flake_stage/nix/"
# Keep the toolchain reachable by garbage collection after this shell starts.
# The profile stays outside the staged flake so it cannot become a source input.
mkdir -p -- "$repo_root/.nix"
cd -- "$repo_root"
nix --extra-experimental-features 'nix-command flakes' develop \
  --profile "$repo_root/.nix/develop-profile" \
  --no-update-lock-file "path:$flake_stage" "$@"
