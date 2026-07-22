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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transformer_engine_linear_low_precision_matches_torch_npu(dtype):
    """TE Linear should preserve low-precision NPU forward and backward behavior."""
    import transformer_engine.pytorch as te

    layer = te.Linear(32, 24, bias=True, device="npu", params_dtype=dtype)
    inputs = torch.randn(4, 32, device="npu", dtype=dtype, requires_grad=True)
    output_grad = torch.randn(4, 24, device="npu", dtype=dtype)

    output = layer(inputs)
    output.backward(output_grad)

    ref_inputs = inputs.detach().clone().requires_grad_(True)
    ref_weight = layer.weight.detach().clone().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True)
    ref_output = torch.nn.functional.linear(ref_inputs, ref_weight, ref_bias)
    ref_output.backward(output_grad)

    torch.npu.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(layer.weight.grad, ref_weight.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(layer.bias.grad, ref_bias.grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("module_name", ["RMSNorm", "LayerNorm"])
def test_transformer_engine_normalization_forward_backward(module_name):
    """TE normalization modules should match eager Torch-NPU autograd."""
    import transformer_engine.pytorch as te

    eps = 1e-5
    layer_class = getattr(te, module_name)
    layer = layer_class(32, eps=eps, device="npu", dtype=torch.float32)
    inputs = torch.randn(4, 32, device="npu", dtype=torch.float32, requires_grad=True)
    output_grad = torch.randn_like(inputs)

    output = layer(inputs)
    output.backward(output_grad)

    ref_inputs = inputs.detach().clone().requires_grad_(True)
    ref_weight = layer.weight.detach().clone().requires_grad_(True)
    if module_name == "RMSNorm":
        ref_output = ref_inputs * torch.rsqrt(
            ref_inputs.square().mean(dim=-1, keepdim=True) + eps
        )
        ref_output = ref_output * ref_weight
        ref_output.backward(output_grad)
        ref_bias_grad = None
    else:
        ref_bias = layer.bias.detach().clone().requires_grad_(True)
        ref_output = torch.nn.functional.layer_norm(
            ref_inputs,
            (32,),
            ref_weight,
            ref_bias,
            eps,
        )
        ref_output.backward(output_grad)
        ref_bias_grad = ref_bias.grad

    torch.npu.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(layer.weight.grad, ref_weight.grad, rtol=2e-3, atol=2e-3)
    if module_name == "LayerNorm":
        torch.testing.assert_close(layer.bias.grad, ref_bias_grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_transformer_engine_layernorm_linear_matches_torch_npu(
    normalization,
    zero_centered_gamma,
):
    """TE LayerNormLinear should match an eager Torch-NPU composition."""
    import transformer_engine.pytorch as te

    eps = 1e-5
    layer = te.LayerNormLinear(
        32,
        24,
        eps=eps,
        normalization=normalization,
        zero_centered_gamma=zero_centered_gamma,
        bias=True,
        device="npu",
        params_dtype=torch.float32,
    )
    inputs = torch.randn(4, 32, device="npu", dtype=torch.float32, requires_grad=True)
    output_grad = torch.randn(4, 24, device="npu", dtype=torch.float32)

    output = layer(inputs)
    output.backward(output_grad)

    ref_inputs = inputs.detach().clone().requires_grad_(True)
    ref_norm_weight = layer.layer_norm_weight.detach().clone().requires_grad_(True)
    effective_weight = 1 + ref_norm_weight if zero_centered_gamma else ref_norm_weight
    if normalization == "LayerNorm":
        ref_norm_bias = layer.layer_norm_bias.detach().clone().requires_grad_(True)
        normalized = torch.nn.functional.layer_norm(
            ref_inputs,
            (32,),
            effective_weight,
            ref_norm_bias,
            eps,
        )
    else:
        ref_norm_bias = None
        normalized = ref_inputs * torch.rsqrt(
            ref_inputs.square().mean(dim=-1, keepdim=True) + eps
        )
        normalized = normalized * effective_weight

    ref_weight = layer.weight.detach().clone().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True)
    ref_output = torch.nn.functional.linear(normalized, ref_weight, ref_bias)
    ref_output.backward(output_grad)

    torch.npu.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        layer.layer_norm_weight.grad,
        ref_norm_weight.grad,
        rtol=2e-3,
        atol=2e-3,
    )
    if normalization == "LayerNorm":
        torch.testing.assert_close(
            layer.layer_norm_bias.grad,
            ref_norm_bias.grad,
            rtol=2e-3,
            atol=2e-3,
        )
    torch.testing.assert_close(layer.weight.grad, ref_weight.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(layer.bias.grad, ref_bias.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_transformer_engine_layernorm_mlp_matches_torch_npu(
    monkeypatch,
    normalization,
    activation,
):
    """TE LayerNormMLP should match its eager Torch-NPU composition."""
    import transformer_engine.pytorch as te

    monkeypatch.setenv("NVTE_BIAS_GELU_NVFUSION", "0")
    eps = 1e-5
    layer = te.LayerNormMLP(
        16,
        32,
        eps=eps,
        normalization=normalization,
        activation=activation,
        bias=True,
        device="npu",
        params_dtype=torch.float32,
    )
    inputs = torch.randn(4, 16, device="npu", dtype=torch.float32, requires_grad=True)
    output_grad = torch.randn(4, 16, device="npu", dtype=torch.float32)

    output = layer(inputs)
    output.backward(output_grad)

    ref_inputs = inputs.detach().clone().requires_grad_(True)
    ref_norm_weight = layer.layer_norm_weight.detach().clone().requires_grad_(True)
    if normalization == "LayerNorm":
        ref_norm_bias = layer.layer_norm_bias.detach().clone().requires_grad_(True)
        normalized = torch.nn.functional.layer_norm(
            ref_inputs,
            (16,),
            ref_norm_weight,
            ref_norm_bias,
            eps,
        )
    else:
        ref_norm_bias = None
        normalized = ref_inputs * torch.rsqrt(
            ref_inputs.square().mean(dim=-1, keepdim=True) + eps
        )
        normalized = normalized * ref_norm_weight

    ref_fc1_weight = layer.fc1_weight.detach().clone().requires_grad_(True)
    ref_fc1_bias = layer.fc1_bias.detach().clone().requires_grad_(True)
    hidden = torch.nn.functional.linear(normalized, ref_fc1_weight, ref_fc1_bias)
    if activation == "gelu":
        hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
    else:
        hidden = torch.nn.functional.silu(hidden)
    ref_fc2_weight = layer.fc2_weight.detach().clone().requires_grad_(True)
    ref_fc2_bias = layer.fc2_bias.detach().clone().requires_grad_(True)
    ref_output = torch.nn.functional.linear(hidden, ref_fc2_weight, ref_fc2_bias)
    ref_output.backward(output_grad)

    torch.npu.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(
        layer.layer_norm_weight.grad,
        ref_norm_weight.grad,
        rtol=3e-3,
        atol=3e-3,
    )
    if normalization == "LayerNorm":
        torch.testing.assert_close(
            layer.layer_norm_bias.grad,
            ref_norm_bias.grad,
            rtol=3e-3,
            atol=3e-3,
        )
    for actual, expected in (
        (layer.fc1_weight.grad, ref_fc1_weight.grad),
        (layer.fc1_bias.grad, ref_fc1_bias.grad),
        (layer.fc2_weight.grad, ref_fc2_weight.grad),
        (layer.fc2_bias.grad, ref_fc2_bias.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    "module_name",
    ["LayerNorm", "RMSNorm", "Linear", "LayerNormLinear", "LayerNormMLP"],
)
def test_transformer_engine_deferred_init_materializes_on_npu(module_name):
    """Deferred core module parameters should materialize on the active NPU."""
    import transformer_engine.pytorch as te

    module_class = getattr(te, module_name)
    args = (
        (16, 32)
        if module_name in {"Linear", "LayerNormLinear", "LayerNormMLP"}
        else (16,)
    )
    kwargs = {
        "device": "meta",
        "params_dtype": torch.float32,
    }
    if module_name == "LayerNormMLP":
        kwargs["activation"] = "silu"

    module = module_class(*args, **kwargs)
    parameters = list(module.parameters())
    assert parameters
    assert all(parameter.is_meta for parameter in parameters)

    with torch.no_grad():
        module.reset_parameters()

    parameters = list(module.parameters())
    assert parameters
    assert all(parameter.device.type == "npu" for parameter in parameters)
    assert all(torch.isfinite(parameter).all().item() for parameter in parameters)


def test_ascend_backend_dispatch_selection():
    """Ascend should select FlagOS ops and use reference only for missing ops."""
    from transformer_engine.plugin.core import get_manager

    manager = get_manager()
    expected_impls = {
        "generic_gemm": "default.flagos",
        "rmsnorm_fwd": "default.flagos",
        "rmsnorm_bwd": "default.flagos",
        "layernorm_fwd": "reference.torch",
        "layernorm_bwd": "reference.torch",
        "gelu": "reference.torch",
        "dgelu": "reference.torch",
        "silu": "reference.torch",
        "dsilu": "reference.torch",
    }
    selected_impls = {
        op_name: manager.get_selected_impl_id(op_name) for op_name in expected_impls
    }
    assert selected_impls == expected_impls
