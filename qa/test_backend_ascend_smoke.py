# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Smoke tests for the real TransformerEngine-FL backend on Ascend NPU."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="Ascend NPU is not available",
)


@pytest.fixture(scope="module", autouse=True)
def _select_npu():
    """Run smoke tests on the first NPU visible to this process."""
    torch.npu.set_device(0)
    yield
    torch.npu.synchronize()


def test_flag_gems_mm_matches_torch_npu():
    """FlagGems GEMM should match the native Torch-NPU result."""
    lhs = torch.randn(32, 64, device="npu", dtype=torch.float32)
    rhs = torch.randn(64, 16, device="npu", dtype=torch.float32)

    expected = torch.mm(lhs, rhs)
    actual = flag_gems.mm(lhs, rhs)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_transformer_engine_linear_forward_backward():
    """TE Linear should use the NPU backend for forward and backward passes."""
    import transformer_engine

    assert transformer_engine.te_device_type() == "npu"

    import transformer_engine.pytorch as te

    layer = te.Linear(64, 32, bias=True, device="npu", params_dtype=torch.float32)
    inputs = torch.randn(8, 64, device="npu", dtype=torch.float32, requires_grad=True)

    output = layer(inputs)
    loss = output.square().mean()
    loss.backward()
    torch.npu.synchronize()

    assert output.shape == (8, 32)
    assert inputs.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(inputs.grad).all().item()
    assert torch.isfinite(layer.weight.grad).all().item()
