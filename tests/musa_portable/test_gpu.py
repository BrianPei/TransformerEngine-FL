"""Real MUSA device checks; run each backend in a separate pytest process."""

import os

import pytest
import torch
import torch_musa  # noqa: F401 -- register MUSA before importing TE
import torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine
from transformer_engine.plugin.core import get_manager


BACKEND = os.environ.get("TE_FL_PREFER")
if BACKEND not in ("reference", "flagos"):
    raise RuntimeError("Set TE_FL_PREFER=reference or flagos explicitly")
EXPECTED = "reference.torch" if BACKEND == "reference" else "default.flagos"


def check_backend(*ops):
    for op in ops:
        selected = get_manager().get_selected_impl_id(op)
        print(f"{op}: {selected}", flush=True)
        assert selected == EXPECTED, (op, selected, EXPECTED)


def test_visible_devices():
    print(f"TE source: {transformer_engine.__file__}; torch: {torch.__version__}")
    print(f"backend: {BACKEND}; devices: {torch.musa.device_count()}")
    assert torch.musa.is_available()
    assert torch.musa.device_count() > 0
    for index in range(torch.musa.device_count()):
        x = torch.ones((8, 8), device=f"musa:{index}")
        torch.testing.assert_close((x @ x).cpu(), torch.full((8, 8), 8.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 64), (4, 8, 128)])
@pytest.mark.parametrize("kind", ["linear", "layernorm", "rmsnorm"])
def test_forward_and_gradients(kind, shape, dtype):
    """Compare device output, input gradient and every parameter gradient to CPU."""
    torch.manual_seed(173)
    hidden = shape[-1]
    if kind == "linear":
        check_backend("generic_gemm")
        model = te.Linear(hidden, 96, device="musa", params_dtype=dtype)
        reference = torch.nn.Linear(hidden, 96)
    elif kind == "layernorm":
        check_backend("layernorm_fwd", "layernorm_bwd")
        model = te.LayerNorm(hidden, device="musa", dtype=dtype)
        reference = torch.nn.LayerNorm(hidden, eps=1e-5)
    else:
        check_backend("rmsnorm_fwd", "rmsnorm_bwd")
        model = te.RMSNorm(hidden, device="musa", dtype=dtype)
        reference = torch.nn.RMSNorm(hidden, eps=1e-5)
    reference.load_state_dict({k: v.detach().cpu().float() for k, v in model.named_parameters()})
    x = torch.randn(shape).to(device="musa", dtype=dtype).requires_grad_()
    cpu_x = x.detach().cpu().float().requires_grad_()
    actual = model(x)
    expected = reference(cpu_x)
    assert actual.device.type == "musa"
    tolerance = {torch.float32: 2e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(actual.cpu().float(), expected, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.cpu().float())
    torch.testing.assert_close(x.grad.cpu().float(), cpu_x.grad, atol=tolerance, rtol=tolerance)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad.cpu().float(),
            dict(reference.named_parameters())[name].grad,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["layernorm_linear", "layernorm_mlp"])
def test_composite_forward_and_gradients(kind, dtype):
    torch.manual_seed(173)
    check_backend("generic_gemm", "layernorm_fwd", "layernorm_bwd")
    cls = te.LayerNormLinear if kind == "layernorm_linear" else te.LayerNormMLP
    model = cls(64, 128, device="musa", params_dtype=dtype)
    parameters = {
        name: value.detach().cpu().requires_grad_() for name, value in model.named_parameters()
    }
    x = torch.randn((4, 8, 64)).to(device="musa", dtype=dtype).requires_grad_()
    cpu_x = x.detach().cpu().requires_grad_()
    normalized = F.layer_norm(
        cpu_x,
        (64,),
        parameters["layer_norm_weight"],
        parameters["layer_norm_bias"],
        1e-5,
    )
    if kind == "layernorm_linear":
        expected = F.linear(normalized, parameters["weight"], parameters["bias"])
    else:
        hidden = F.linear(normalized, parameters["fc1_weight"], parameters["fc1_bias"])
        expected = F.linear(
            F.gelu(hidden, approximate="tanh"),
            parameters["fc2_weight"],
            parameters["fc2_bias"],
        )
    actual = model(x)
    assert actual.device.type == "musa"
    # Match intermediate rounding by using the same dtype on CPU and MUSA.
    tolerance = {torch.float32: 2e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), atol=tolerance, rtol=tolerance
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.cpu())
    torch.testing.assert_close(
        x.grad.cpu().float(), cpu_x.grad.float(), atol=tolerance, rtol=tolerance
    )
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad.cpu().float(),
            parameters[name].grad.float(),
            atol=tolerance,
            rtol=tolerance,
        )
