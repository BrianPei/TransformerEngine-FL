#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/set_env.sh"
cd "$TE_PATH"
exec python3 "$PPU_TEST_DIR/run_suites.py" "$@"
