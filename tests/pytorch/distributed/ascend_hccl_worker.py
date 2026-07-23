# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-process HCCL and Transformer Engine smoke test for Ascend NPU."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

ASCEND_TEST_UTILS = Path(__file__).resolve().parents[2] / "test_utils" / "ascend"
sys.path.insert(0, str(ASCEND_TEST_UTILS))
from npu_patch import apply_ascend_npu_patch


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    try:
        collective = torch.tensor([rank + 1.0], device="npu", dtype=torch.float32)
        dist.all_reduce(collective)
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(collective.cpu(), torch.tensor([expected]))

        apply_ascend_npu_patch()

        import transformer_engine

        import transformer_engine.pytorch as te

        assert transformer_engine.te_device_type() == "npu"
        torch.manual_seed(2026)
        layer = te.Linear(64, 32, bias=True, device="npu", params_dtype=torch.float32)
        inputs = torch.randn(8, 64, device="npu", dtype=torch.float32)
        inputs = (inputs + rank * 0.125).requires_grad_(True)
        loss = layer(inputs).square().mean()
        loss.backward()

        local_weight_grad = layer.weight.grad.detach().clone()
        local_bias_grad = layer.bias.grad.detach().clone()
        gathered_weight_grads = [torch.empty_like(local_weight_grad) for _ in range(world_size)]
        gathered_bias_grads = [torch.empty_like(local_bias_grad) for _ in range(world_size)]
        dist.all_gather(gathered_weight_grads, local_weight_grad)
        dist.all_gather(gathered_bias_grads, local_bias_grad)
        expected_weight_grad = torch.stack(gathered_weight_grads).mean(dim=0)
        expected_bias_grad = torch.stack(gathered_bias_grads).mean(dim=0)

        dist.all_reduce(layer.weight.grad)
        layer.weight.grad.div_(world_size)
        dist.all_reduce(layer.bias.grad)
        layer.bias.grad.div_(world_size)
        torch.npu.synchronize()

        assert inputs.grad is not None
        assert torch.isfinite(loss).item()
        assert torch.isfinite(inputs.grad).all().item()
        assert torch.isfinite(layer.weight.grad).all().item()
        assert torch.isfinite(layer.bias.grad).all().item()
        torch.testing.assert_close(layer.weight.grad, expected_weight_grad)
        torch.testing.assert_close(layer.bias.grad, expected_bias_grad)
        dist.barrier()
        if rank == 0:
            print(f"ASCEND_HCCL_TE_OK world_size={world_size}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
