#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

exec python3 "$REPO_ROOT/tests/plugin/run_upstream.py" \
    --adapter "$SCRIPT_DIR/adapter.py" \
    "$@"
