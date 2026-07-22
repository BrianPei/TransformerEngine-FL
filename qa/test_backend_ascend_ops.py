# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Real-NPU unit tests for the FlagOS operators used by TransformerEngine-FL."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("flag_gems")

from transformer_engine.plugin.core.backends.flagos.impl.gemm import generic_gemm_fl
from transformer_engine.plugin.core.backends.flagos.impl.multi_tensor import (
    multi_tensor_l2_norm_fl,
    multi_tensor_scale_fl,
)
from transformer_engine.plugin.core.backends.flagos.impl.rmsnorm import (
    rmsnorm_bwd_fl,
    rmsnorm_fwd_fl,
)
from transformer_engine.plugin.core.backends.flagos.impl.softmax import (
    scaled_masked_softmax_backward_fl,
    scaled_masked_softmax_forward_fl,
)


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="Ascend NPU is not available",
)


@pytest.fixture(scope="module", autouse=True)
def _select_npu():
    torch.npu.set_device(0)
    yield
    torch.npu.synchronize()


@pytest.mark.parametrize("with_bias", [False, True])
def test_flagos_gemm_matches_torch_npu(with_bias):
    weight = torch.randn(16, 32, device="npu", dtype=torch.float32)
    inputs = torch.randn(8, 32, device="npu", dtype=torch.float32)
    bias = torch.randn(16, device="npu", dtype=torch.float32) if with_bias else None
    workspace = torch.empty(1, device="npu", dtype=torch.uint8)

    output, bias_grad, gelu_input, extra_output = generic_gemm_fl(
        A=weight,
        transA=True,
        B=inputs,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=torch.float32,
        bias=bias,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=workspace,
        workspace_size=workspace.numel(),
        accumulate=False,
        use_split_accumulator=False,
    )

    expected = torch.nn.functional.linear(inputs, weight, bias)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
    assert bias_grad is None
    assert gelu_input is None
    assert extra_output is None


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_flagos_rmsnorm_forward_backward_matches_torch_npu(zero_centered_gamma):
    eps = 1e-5
    inputs = torch.randn(4, 8, device="npu", dtype=torch.float32)
    weight = torch.randn(8, device="npu", dtype=torch.float32)
    output_grad = torch.randn_like(inputs)

    output, _, rsigma = rmsnorm_fwd_fl(
        input=inputs,
        weight=weight,
        eps=eps,
        ln_out=None,
        quantizer=None,
        odtype=None,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )
    input_grad, weight_grad = rmsnorm_bwd_fl(
        dy=output_grad,
        x=inputs,
        rsigma=rsigma,
        gamma=weight,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
        eps=eps,
    )

    ref_inputs = inputs.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    effective_weight = 1 + ref_weight if zero_centered_gamma else ref_weight
    ref_output = ref_inputs * torch.rsqrt(ref_inputs.square().mean(dim=-1, keepdim=True) + eps)
    ref_output = ref_output * effective_weight
    ref_output.backward(output_grad)

    torch.testing.assert_close(output, ref_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(input_grad, ref_inputs.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(weight_grad, ref_weight.grad, rtol=2e-3, atol=2e-3)


def test_flagos_scaled_masked_softmax_matches_torch_npu():
    scale = 0.5
    inputs = torch.randn(2, 2, 8, 8, device="npu", dtype=torch.float32)
    mask = torch.zeros(2, 1, 8, 8, device="npu", dtype=torch.int32)
    mask[..., -2:] = 1

    output = scaled_masked_softmax_forward_fl(inputs, mask, scale)
    expected = torch.softmax((inputs * scale).masked_fill(mask.expand_as(inputs) == 1, -10000), -1)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)

    output_grad = torch.randn_like(output)
    input_grad = scaled_masked_softmax_backward_fl(output_grad, output, scale)
    expected_grad = output * (output_grad - (output * output_grad).sum(dim=-1, keepdim=True))
    expected_grad = expected_grad * scale
    torch.testing.assert_close(input_grad, expected_grad, rtol=2e-3, atol=2e-3)


def test_flagos_multi_tensor_scale_and_l2_norm():
    noop_flag = torch.zeros(1, device="npu", dtype=torch.int32)
    source = [
        torch.tensor([1.0, 2.0], device="npu"),
        torch.tensor([3.0, 4.0], device="npu"),
    ]
    destination = [torch.zeros_like(tensor) for tensor in source]

    multi_tensor_scale_fl(1024, noop_flag, [source, destination], scale=2.0)
    torch.testing.assert_close(destination[0], source[0] * 2)
    torch.testing.assert_close(destination[1], source[1] * 2)

    total_norm, per_tensor_norms = multi_tensor_l2_norm_fl(
        1024,
        noop_flag,
        [source],
        per_tensor=True,
    )
    expected_per_tensor = torch.stack([torch.linalg.vector_norm(tensor) for tensor in source])
    expected_total = torch.linalg.vector_norm(torch.cat(source))
    torch.testing.assert_close(per_tensor_norms, expected_per_tensor, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(total_norm, expected_total, rtol=1e-4, atol=1e-4)
