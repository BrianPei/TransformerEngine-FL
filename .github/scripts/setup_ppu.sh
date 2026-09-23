#!/usr/bin/env bash
# Source this script before invoking the common CI dispatcher.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/tests/plugin/backend/ppu/set_env.sh"
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
    export XML_LOG_DIR="$GITHUB_WORKSPACE/logs/ppu"
fi

# Checkpoint compatibility fixtures are immutable inputs to test_checkpoint.py.
# They must come from a reviewed, versioned artifact or an image/mount that
# already contains the same manifest; never generate them from this checkout.
readarray -t _checkpoint_config < <(python3 - "$TE_PATH/.github/configs/ppu.yml" <<'PY'
import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
fixture = config.get("checkpoint_fixture") or {}
print(fixture.get("version") or "")
print(fixture.get("path") or "")
print(fixture.get("archive_url") or "")
print(fixture.get("archive_sha256") or "")
PY
)
CHECKPOINT_VERSION="${PPU_CHECKPOINT_FIXTURE_VERSION:-${_checkpoint_config[0]:-}}"
CHECKPOINT_RELATIVE_PATH="${PPU_CHECKPOINT_FIXTURE_PATH:-${_checkpoint_config[1]:-}}"
CHECKPOINT_ARCHIVE_URL="${PPU_CHECKPOINT_FIXTURE_URL:-${_checkpoint_config[2]:-}}"
CHECKPOINT_ARCHIVE_SHA256="${PPU_CHECKPOINT_FIXTURE_SHA256:-${_checkpoint_config[3]:-}}"
if [ -z "$CHECKPOINT_VERSION" ]; then
    echo "checkpoint_fixture.version must identify an immutable fixture release" >&2
    exit 1
fi
if [ -z "$CHECKPOINT_RELATIVE_PATH" ] || [[ "$CHECKPOINT_RELATIVE_PATH" = /* ]] || \
   [[ "/$CHECKPOINT_RELATIVE_PATH/" = *"/../"* ]]; then
    echo "checkpoint_fixture.path must be a repository-relative path" >&2
    exit 1
fi
CHECKPOINT_DIR="$TE_PATH/$CHECKPOINT_RELATIVE_PATH"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    if [ -z "$CHECKPOINT_ARCHIVE_URL" ] || [ -z "$CHECKPOINT_ARCHIVE_SHA256" ]; then
        echo "PPU checkpoint fixture is missing: $CHECKPOINT_DIR" >&2
        echo "Configure a reviewed immutable archive URL and SHA256 in ppu.yml" >&2
        exit 1
    fi
    _fixture_tmp="$(mktemp -d)"
    trap 'rm -rf "$_fixture_tmp"' EXIT
    _archive="$_fixture_tmp/checkpoint-fixture.tar.gz"
    curl --fail --location --retry 3 --output "$_archive" "$CHECKPOINT_ARCHIVE_URL"
    printf '%s  %s\n' "$CHECKPOINT_ARCHIVE_SHA256" "$_archive" | sha256sum --check --status
    mkdir -p "$CHECKPOINT_DIR"
    tar --extract --gzip --file "$_archive" --directory "$CHECKPOINT_DIR" --strip-components=1
fi

export NVTE_TEST_CHECKPOINT_ARTIFACT_PATH="$CHECKPOINT_DIR"
python3 - "$CHECKPOINT_DIR" "$CHECKPOINT_VERSION" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_version = sys.argv[2]
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing checkpoint fixture manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
required_metadata = (
    "fixture_version",
    "source_repository",
    "source_commit",
    "transformer_engine_version",
    "torch_version",
    "generation_environment",
    "generation_command",
)
missing_metadata = [key for key in required_metadata if not manifest.get(key)]
if missing_metadata:
    raise SystemExit(
        "Checkpoint manifest is missing provenance fields: " + ", ".join(missing_metadata)
    )
if manifest["fixture_version"] != expected_version:
    raise SystemExit(
        f"Checkpoint fixture version mismatch: expected {expected_version}, "
        f"got {manifest['fixture_version']}"
    )
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("Checkpoint manifest must contain a non-empty files map")
required = {
    "linear.pt", "layernorm_linear.pt", "layernorm_mlp.pt", "layernorm.pt",
    "rmsnorm.pt", "transformer_layer.pt", "ops_linear.pt",
    "linear.fp8.pt", "ops_linear.fp8.pt", "linear.mxfp8.pt", "ops_linear.mxfp8.pt",
}
missing = sorted(required - files.keys())
if missing:
    raise SystemExit("Checkpoint manifest is missing required files: " + ", ".join(missing))
for relative, expected in files.items():
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"Invalid or missing checkpoint fixture file: {relative}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        raise SystemExit(f"Checkpoint fixture checksum mismatch for {relative}")
print(f"Validated checkpoint fixture: {len(files)} files")
PY
python3 - <<'PY'
import os
from pathlib import Path

import torch
import transformer_engine.pytorch
from transformer_engine.plugin.core.manager import get_default_manager

if not torch.cuda.is_available() or "PPU" not in torch.cuda.get_device_name():
    raise SystemExit("PPU runtime is unavailable")
source = Path(transformer_engine.__file__).resolve()
if not source.is_relative_to(Path(os.environ["TE_PATH"]).resolve()):
    raise SystemExit(f"TransformerEngine was imported from the wrong checkout: {source}")
torch.testing.assert_close((torch.ones(4, device="cuda") + 1).cpu(), torch.full((4,), 2.))
print("PPU runtime:", torch.__version__, torch.cuda.get_device_name(), torch.cuda.device_count())
print("TransformerEngine source:", source)
print("GEMM implementation:", get_default_manager().get_selected_impl_id("generic_gemm"))
PY
if [ -n "${GITHUB_ENV:-}" ]; then
    for key in TE_PATH PYTHONPATH XML_LOG_DIR NVTE_TEST_CHECKPOINT_ARTIFACT_PATH; do
        printf '%s=%s\n' "$key" "${!key}" >> "$GITHUB_ENV"
    done
fi
