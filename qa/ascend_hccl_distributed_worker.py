# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-process HCCL and Transformer Engine smoke test for Ascend NPU."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


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

        import transformer_engine
        import transformer_engine.pytorch as te

        assert transformer_engine.te_device_type() == "npu"
        torch.manual_seed(2026)
        layer = te.Linear(64, 32, bias=True, device="npu", params_dtype=torch.float32)
        inputs = torch.randn(8, 64, device="npu", dtype=torch.float32, requires_grad=True)
        loss = layer(inputs).square().mean()
        loss.backward()

        dist.all_reduce(layer.weight.grad)
        layer.weight.grad.div_(world_size)
        torch.npu.synchronize()

        assert inputs.grad is not None
        assert torch.isfinite(loss).item()
        assert torch.isfinite(inputs.grad).all().item()
        assert torch.isfinite(layer.weight.grad).all().item()
        dist.barrier()
        if rank == 0:
            print(f"ASCEND_HCCL_TE_OK world_size={world_size}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
