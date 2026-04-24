# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

retry_command() {
    local attempts=$1
    local delay_seconds=$2
    shift 2

    local attempt
    for attempt in $(seq 1 "${attempts}"); do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -lt "${attempts}" ]; then
            echo "Command failed (attempt ${attempt}/${attempts}): $*"
            echo "Retrying in ${delay_seconds}s..."
            sleep "${delay_seconds}"
        fi
    done

    echo "Command failed after ${attempts} attempts: $*"
    return 1
}

# Paths
: "${TE_PATH:=$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
: "${MCORE_PATH:=/workspace/Megatron-LM-FL}"
: "${MCORE_REPO_URL:=https://github.com/flagos-ai/Megatron-LM-FL.git}"
: "${MCORE_REF:=main}"
: "${OUTPUT_DIR:=${TE_PATH}/qa/L1_pytorch_mcore_integration/output}"
: "${DATA_CACHE_PATH:=/tmp/data_cache}"

# Check whether FP8 is supported
WITH_FP8=
if command -v nvidia-smi &>/dev/null; then
    DEVICE_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/[^0-9]//g')
    if [[ ${DEVICE_ARCH} -ge 89 ]]; then
        WITH_FP8=1
    fi
elif command -v mx-smi &>/dev/null; then
    # Metax hardware does not support FP8; leave WITH_FP8 unset
    :
fi

# Download or sync Megatron-LM-FL to the requested repo/ref.
if [ ! -d "${MCORE_PATH}" ]; then
    pushd $(dirname ${MCORE_PATH})
    git config --global --unset-all credential.helper 2>/dev/null || true
    git config --system --unset-all credential.helper 2>/dev/null || true
    retry_command 3 5 git clone --depth 1 -b "${MCORE_REF}" "${MCORE_REPO_URL}" $(basename ${MCORE_PATH})
    popd
fi

if [ -d "${MCORE_PATH}/.git" ]; then
    git -C "${MCORE_PATH}" remote set-url origin "${MCORE_REPO_URL}"
    retry_command 3 5 git -C "${MCORE_PATH}" fetch --depth 1 origin "${MCORE_REF}"
    git -C "${MCORE_PATH}" checkout -B "${MCORE_REF}" "FETCH_HEAD"
fi

# Megatron-LM-FL tokenizer imports happen at module import time, so direct
# source execution needs these Python deps available before pretrain_gpt.py
# starts.
python3 - <<'PY' || python3 -m pip install --disable-pip-version-check six regex
import regex
import six
print(f"six available: {six.__version__}")
print(f"regex available: {regex.__version__}")
PY

CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}" /tmp/checkpoints

echo "Using Megatron-LM-FL repo: ${MCORE_REPO_URL}"
echo "Using Megatron-LM-FL ref: ${MCORE_REF}"
git -C "${MCORE_PATH}" rev-parse --short HEAD

# Megatron-LM-FL invocation. Keep the argument shape aligned with the
# previously validated tp1/pp1 mock-data GPT functional case while letting CI
# exit after a few steps.
COMMAND="
NVTE_TORCH_COMPILE=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_ALGO=Ring
CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun
--nnodes=1
--nproc_per_node=1

${MCORE_PATH}/pretrain_gpt.py
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--num-layers 12
--hidden-size 512
--num-attention-heads 8
--log-params-norm
--log-num-zeros-in-grad
--log-validation-ppl-to-tensorboard
--log-timers-to-tensorboard
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 4
--global-batch-size 32
--train-iters 50
--eval-iters 10
--timing-log-level 0
--lr-decay-iters 320000
--save ${CHECKPOINT_DIR}
--split 949,50,1
--tokenizer-type NullTokenizer
--vocab-size 8192
--mock-data
--distributed-backend nccl
--lr 0.00015
--lr-decay-style cosine
--min-lr 1.0e-5
--weight-decay 1e-2
--clip-grad 1.0
--lr-warmup-fraction .01
--log-interval 1
--save-interval 10000
--eval-interval 1000
--transformer-impl transformer_engine
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--deterministic-mode
--no-gradient-accumulation-fusion
--attention-softmax-in-fp32
--use-mcore-models
--ckpt-format torch_dist
--dist-ckpt-optim-fully-reshardable
--dist-ckpt-strictness log_all
--data-cache-path ${DATA_CACHE_PATH}
--bf16
--attention-backend unfused
--log-memory-to-tensorboard
--tensorboard-dir ${TENSORBOARD_DIR}
--exit-interval 4
${WITH_FP8:+--fp8-format hybrid}
"
COMMAND=$(echo "${COMMAND}" | tr '\n' ' ')

# Launch Megatron-LM-FL
bash -c "${COMMAND}"
