#!/usr/bin/env bash
# Build the interpreted runtime used by policy differential tests.
# Run from any directory; output stays beside the shipped Souffle sources.
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
souffle="${SASY_SOUFFLE:-souffle}"
cxx="${SASY_CXX:-g++}"
souffle_path="$(command -v "$souffle")"
command -v "$cxx" >/dev/null
prefix="$(CDPATH= cd -- "$(dirname -- "$souffle_path")/.." && pwd)"
include="${SOUFFLE_INCLUDE:-${SASY_SOUFFLE_INCLUDE:-$prefix/include}}"
word_size="$("$souffle" --version | sed -n 's/^Word size: \([0-9][0-9]*\) bits/\1/p')"
case "$word_size" in
  32|64) ;;
  *) echo "Cannot determine Souffle word size (expected 32 or 64 bits)" >&2; exit 1 ;;
esac
case "$(uname -s)" in
  Darwin) library=libfunctors.dylib ;;
  Linux) library=libfunctors.so ;;
  *) echo "Unsupported policy test runtime platform" >&2; exit 1 ;;
esac

cd "$script_dir"
"$cxx" -std=c++17 -O2 -o souffle-interpreted interpreted_shim.cpp
"$cxx" -std=c++17 -O2 -shared -fPIC \
  "-DRAM_DOMAIN_SIZE=$word_size" "-I$include" functors.cpp -o "$library"
