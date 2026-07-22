#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${WHEEL_DIR:=$TE_PATH/dist/ascend}"
: "${PYTHON_BIN:=python3}"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch
export PLATFORM=ascend
export NVTE_DEVICE_TYPE=npu

"$PYTHON_BIN" -m pip install wheel
mkdir -p "$WHEEL_DIR"
find "$WHEEL_DIR" -maxdepth 1 -name 'transformer_engine-*.whl' -type f -delete

"$PYTHON_BIN" -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$WHEEL_DIR" \
    "$TE_PATH"

wheel_path=$(find "$WHEEL_DIR" -maxdepth 1 -name 'transformer_engine-*.whl' -type f \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$wheel_path" ]; then
    echo "No TransformerEngine-FL wheel was produced in $WHEEL_DIR." >&2
    exit 1
fi

install_dir=$(mktemp -d "$TE_PATH/.wheel-test-ascend.XXXXXX")
trap 'rm -rf "$install_dir"' EXIT
"$PYTHON_BIN" -m pip install --no-deps --target "$install_dir" "$wheel_path"

cd /tmp
if ! PYTHONPATH="$install_dir" "$PYTHON_BIN" - <<'PY'
import torch

import transformer_engine
import transformer_engine.pytorch as te

assert transformer_engine.te_device_type() == "npu"
assert torch.npu.is_available()

torch.npu.set_device(0)
layer = te.Linear(16, 8, device="npu", params_dtype=torch.float32)
inputs = torch.randn(4, 16, device="npu", dtype=torch.float32, requires_grad=True)
output = layer(inputs)
output.square().mean().backward()
torch.npu.synchronize()

assert output.shape == (4, 8)
assert inputs.grad is not None
assert layer.weight.grad is not None
print(f"WHEEL_ASCEND_OK version={transformer_engine.__version__} path={transformer_engine.__file__}")
PY
then
    echo "The isolated Ascend wheel validation failed." >&2
    exit 1
fi


echo "Ascend Python wheel build and isolated NPU import test passed: $wheel_path"
