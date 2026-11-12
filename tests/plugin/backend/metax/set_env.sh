#!/usr/bin/env bash
# MetaX Backend Environment Setup

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PLATFORM="${PLATFORM:-metax}"
export TE_PATH="${TE_PATH:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
export TE_LIB_PATH="${TE_LIB_PATH:-$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/transformer_engine}"
export PYTHONPATH="${TE_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-1}"
# The CI image also ships flag_gems; keep fused normalization on the vendor
# backend, whose fp16/bf16 reductions match TE's tolerances.
export TE_FL_PER_OP="${TE_FL_PER_OP:-layernorm_fwd=vendor|flagos|reference;layernorm_bwd=vendor|flagos|reference;rmsnorm_fwd=vendor|flagos|reference;rmsnorm_bwd=vendor|flagos|reference;rmsnorm_bwd_add=vendor|flagos|reference}"
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
export CUDA_HOME="${CUDA_HOME:-/opt/maca}"
export MACA_HOME="${MACA_HOME:-/opt/maca}"
export PATH="${MACA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${LD_LIBRARY_PATH:-}"
