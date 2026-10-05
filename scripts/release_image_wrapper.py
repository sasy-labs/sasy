#!/usr/bin/env python3
"""Adapt an owned-engine integration fixture to a local Linux Docker image."""
import argparse
from pathlib import Path
import shlex

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--image', required=True)
parser.add_argument('--label', required=True)
args = parser.parse_args()
# The fixture creates fresh synthetic credentials and runs from its own temp
# directory. Mount only that directory, never the user's repository or home.
args.output.write_text('''#!/bin/bash
set -euo pipefail
test "$(uname -s)" = Linux
test "$(id -u)" != 0
exec docker run --rm --network host --user "$(id -u):$(id -g)" \\
  --mount "type=bind,source=$PWD,target=$PWD" --workdir "$PWD" \\
  --env "HOME=$PWD" --env "TMPDIR=$PWD" --env "XDG_CACHE_HOME=$PWD/.cache" \\
  --env LLM_ENABLED=false --env OTEL_ENABLED=false --env TOKIO_WORKER_THREADS=2 \\
  --label ''' + shlex.quote(args.label) + ' --entrypoint /bin/sasy ' + shlex.quote(args.image) + ' "$@"\n')
args.output.chmod(0o755)
