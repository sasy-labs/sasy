#!/usr/bin/env bash
# Nix normally imports an entire Git flake before evaluating its source filter.
# Stage the explicit public build inputs first when building a private checkout.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
umask 077
stage_parent="$(mktemp -d "${TMPDIR:-/tmp}/sasy-nix-build.XXXXXXXX")"
trap 'rm -rf -- "$stage_parent"' EXIT
stage_parent="$(cd -- "$stage_parent" && pwd -P)"
python3 "$repo_root/nix/stage-source.py" "$stage_parent/source"
nix --extra-experimental-features 'nix-command flakes' build --option sandbox true --no-update-lock-file --max-jobs 1 --cores 2 "path:$stage_parent/source#sasy" "$@"
