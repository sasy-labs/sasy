#!/bin/bash
# Succeed only when the registry confirms that an image reference does not
# exist yet. An existing image, or any answer other than the registry's
# not-found codes (such as a transient registry or auth error), fails, so a
# published version is never replaced.
set -uo pipefail
ref="$1"
if output=$(docker manifest inspect "$ref" 2>&1); then
  echo "$ref already exists; a published version is never replaced." >&2
  exit 1
fi
if grep -qiE 'manifest unknown|no such manifest|name unknown' <<< "$output"; then
  echo "$ref is not published yet."
  exit 0
fi
echo "Could not confirm that $ref is unpublished: $output" >&2
exit 1
