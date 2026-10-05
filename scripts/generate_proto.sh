#!/usr/bin/env bash
# Regenerate the Python gRPC stubs (and, when the JavaScript toolchain is
# installed, the TypeScript ones) from proto/*.proto.
#
# The Python stubs are committed. They are generated with one pinned version of
# grpcio-tools so that regenerating them is reproducible and CI can check that
# the committed files match the .proto files:
#
#   uv pip install --python .venv/bin/python "grpcio-tools==$GRPCIO_TOOLS_VERSION"
#   PYTHON=.venv/bin/python scripts/generate_proto.sh
set -euo pipefail

GRPCIO_TOOLS_VERSION="1.80.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$ROOT_DIR/proto"
OUTPUT_DIR="$ROOT_DIR/sdk/python/sasy/proto"
TS_DIR="$ROOT_DIR/sdk/typescript"
PYTHON="${PYTHON:-python3}"

installed="$("$PYTHON" -c 'from importlib.metadata import version; print(version("grpcio-tools"))' 2>/dev/null || true)"
if [ "$installed" != "$GRPCIO_TOOLS_VERSION" ]; then
    echo "generate_proto.sh: grpcio-tools $GRPCIO_TOOLS_VERSION is required (found: ${installed:-none})." >&2
    echo "Install it into the interpreter named by PYTHON, for example:" >&2
    echo "  uv pip install --python .venv/bin/python 'grpcio-tools==$GRPCIO_TOOLS_VERSION'" >&2
    exit 1
fi

echo "Generating Python stubs into $OUTPUT_DIR with grpcio-tools $GRPCIO_TOOLS_VERSION..."
"$PYTHON" -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    "$PROTO_DIR/credential_server.proto" \
    "$PROTO_DIR/observability.proto" \
    "$PROTO_DIR/policy_engine.proto" \
    "$PROTO_DIR/policy_plugin.proto" \
    "$PROTO_DIR/reference_monitor.proto"

# protoc emits absolute imports between the generated modules; the package
# needs relative ones. Done in Python so the edit is the same on every platform.
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import pathlib, re, sys
for path in pathlib.Path(sys.argv[1]).glob("*_pb2*.py*"):
    text = path.read_text()
    fixed = re.sub(r"^import ([A-Za-z0-9_]+_pb2)", r"from . import \1", text, flags=re.M)
    if fixed != text:
        path.write_text(fixed)
PY
echo "Python stubs generated."

if command -v npm >/dev/null 2>&1 && { [ -d "$TS_DIR/node_modules" ] || [ -d "$ROOT_DIR/node_modules" ]; }; then
    echo "Generating TypeScript stubs into $TS_DIR/src/generated..."
    (cd "$TS_DIR" && npm run --silent proto:generate)
    echo "TypeScript stubs generated."
else
    echo "Skipping TypeScript stubs: run 'npm ci' in $ROOT_DIR and re-run, or"
    echo "'cd $TS_DIR && npm run proto:generate' once the toolchain is set up."
fi
