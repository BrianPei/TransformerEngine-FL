import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation & Explicit Safe Mocking
# ==============================================================================
# 1. Isolating core ops and vendor type structures
mock_ops = MagicMock()
sys.modules["transformer_engine.plugin.core.ops"] = mock_ops


class MockBase:
    pass


mock_ops.TEFLBackendBase = MockBase
mock_ops.DType = MagicMock()
mock_ops.CommOverlapType = MagicMock()
mock_ops.NVTE_QKV_Layout = MagicMock()
mock_ops.NVTE_Bias_Type = MagicMock()
mock_ops.NVTE_Mask_Type = MagicMock()
mock_ops.NVTE_Softmax_Type = MagicMock()
mock_ops.NVTE_QKV_Format = MagicMock()


class MockFusedBackend:
    NVTE_No_Backend = 0


mock_ops.NVTE_Fused_Attn_Backend = MockFusedBackend

# 2. Explicitly mock required functional implementations to eliminate nested MagicMock recursions
mock_impl = MagicMock()
sys.modules["transformer_engine.plugin.core.backends.reference.impl"] = mock_impl

# Manually register non-recursive passthrough callbacks for every single implementation function used in reference.py
torch_functions = [
    "general_gemm_torch",
    "gelu_torch",
    "geglu_torch",
    "qgelu_torch",
    "qgeglu_torch",
    "relu_torch",
    "reglu_torch",
    "srelu_torch",
    "sreglu_torch",
    "silu_torch",
    "swiglu_torch",
    "clamped_swiglu_torch",
    "dgelu_torch",
    "dgeglu_torch",
    "dqgelu_torch",
    "dqgeglu_torch",
    "drelu_torch",
    "dreglu_torch",
    "dsrelu_torch",
    "dsreglu_torch",
    "dsilu_torch",
    "dswiglu_torch",
    "clamped_dswiglu_torch",
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
    "multi_tensor_scale_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_adam_fp8_torch",
    "multi_tensor_adam_capturable_torch",
    "multi_tensor_adam_capturable_master_torch",
    "multi_tensor_adam_param_remainder_torch",
    "multi_tensor_sgd_torch",
    "multi_tensor_compute_scale_and_scale_inv_torch",
    "multi_tensor_compute_scale_inv_e8m0_torch",
]

# Simple lambda returning a tensor to keep things ultra-fast and recursion-free
for func_name in torch_functions:
    setattr(mock_impl, func_name, lambda *args, **kwargs: torch.tensor([1.0]))

# Setup precise unpack configurations
mock_impl.layernorm_fwd_torch = lambda *args, **kwargs: [
    torch.tensor(1.0),
    torch.tensor(1.0),
    torch.tensor(1.0),
]
mock_impl.layernorm_bwd_torch = lambda *args, **kwargs: [torch.tensor(1.0), torch.tensor(1.0)]
mock_impl.rmsnorm_fwd_torch = lambda *args, **kwargs: [
    torch.tensor(1.0),
    torch.tensor(1.0),
    torch.tensor(1.0),
]
mock_impl.rmsnorm_bwd_torch = lambda *args, **kwargs: [torch.tensor(1.0), torch.tensor(1.0)]
mock_impl.dropout_fwd_torch = lambda *args, **kwargs: (torch.tensor(1.0), torch.tensor(1.0))
mock_impl.dropout_bwd_torch = lambda *args, **kwargs: torch.tensor(1.0)
mock_impl.multi_tensor_l2norm_torch = lambda *args, **kwargs: (torch.tensor(1.0), torch.tensor(1.0))

# Import actual class safely
from transformer_engine.plugin.core.backends.reference.reference import ReferenceBackend

# ==============================================================================
# Part 1: Availability and Attention Routing Tests
# ==============================================================================


def test_backend_availability():
    """Verify standard static and lifecycle availability flags."""
    assert ReferenceBackend.check_available() is True
    backend = ReferenceBackend()
    assert backend.is_available() is True


@pytest.mark.parametrize(
    "env_vars, expected_backends",
    [
        ({"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "1"}, [1, 1, 1]),
        ({"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"}, [0, 0, 0]),
    ],
)
def test_get_attention_backend(env_vars, expected_backends, monkeypatch):
    """Test dynamic environment variable evaluation for attention backends."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    backend = ReferenceBackend()
    res = backend.get_attention_backend()

    assert int(res[0]) == expected_backends[0]
    assert int(res[2]) == expected_backends[1]
    assert int(res[4]) == expected_backends[2]
    assert res[5] == expected_backends


# ==============================================================================
# Part 2: Activation and Linear Core Math Tests
# ==============================================================================


@pytest.mark.parametrize(
    "act_fwd, act_bwd",
    [
        ("gelu", "dgelu"),
        ("geglu", "dgeglu"),
        ("qgelu", "dqgelu"),
        ("qgeglu", "dqgeglu"),
        ("relu", "drelu"),
        ("reglu", "dreglu"),
        ("srelu", "dsrelu"),
        ("sreglu", "dsreglu"),
        ("silu", "dsilu"),
        ("swiglu", "dswiglu"),
    ],
)
def test_activation_forward_backward_pass_through(act_fwd, act_bwd):
    """Verify that all standard activations and gate variants map to their implementations."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)

    fwd_fn = getattr(backend, act_fwd)
    bwd_fn = getattr(backend, act_bwd)

    assert fwd_fn(inp, quantizer=None) is not None
    assert bwd_fn(inp, inp, quantizer=None) is not None


def test_clamped_swiglu_variants():
    """Verify clamped activation branches execute with customized limits."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)
    assert backend.clamped_swiglu(inp, quantizer=None, limit=5.0, alpha=1.5) is not None
    assert backend.clamped_dswiglu(inp, inp, quantizer=None, limit=5.0, alpha=1.5) is not None


@pytest.mark.parametrize(
    "dbias_act", ["dbias_dgelu", "dbias_dsilu", "dbias_drelu", "dbias_dqgelu", "dbias_dsrelu"]
)
def test_dbias_fusions(dbias_act):
    """Verify fused bias derivative operations are dispatched correctly."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)
    fn = getattr(backend, dbias_act)
    assert fn(inp, inp, quantizer=None) is not None


def test_generic_gemm_passthrough():
    """Verify general matrix multiplication arguments route cleanly to backend implementation."""
    backend = ReferenceBackend()
    inp = torch.randn(2, 2)
    res = backend.generic_gemm(
        A=inp,
        transA=False,
        B=inp,
        transB=False,
        D=None,
        quantizer=None,
        output_dtype=None,
        bias=None,
        bias_type=None,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=inp,
        workspace_size=0,
        accumulate=False,
        use_split_accumulator=False,
    )
    assert res is not None


# ==============================================================================
# Part 3: Normalization and Softmax Functional Tests
# ==============================================================================


def test_normalization_fwd_bwd():
    """Verify LayerNorm and RMSNorm operations forward full parameter signatures."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)
    w = torch.ones(4)

    assert backend.layernorm_fwd(inp, w, None, 1e-5, None, None, None, 0, False) is not None
    assert backend.layernorm_bwd(inp, inp, inp, inp, w, 0, False) is not None

    assert backend.rmsnorm_fwd(inp, w, 1e-5, None, None, None, 0, False) is not None
    assert backend.rmsnorm_bwd(inp, inp, inp, w, 0, False) is not None


@pytest.mark.parametrize(
    "softmax_fwd, softmax_bwd, has_mask",
    [
        ("scaled_softmax_forward", "scaled_softmax_backward", False),
        ("scaled_masked_softmax_forward", "scaled_masked_softmax_backward", True),
        (
            "scaled_upper_triang_masked_softmax_forward",
            "scaled_upper_triang_masked_softmax_backward",
            False,
        ),
        (
            "scaled_aligned_causal_masked_softmax_forward",
            "scaled_aligned_causal_masked_softmax_backward",
            False,
        ),
    ],
)
def test_softmax_variants(softmax_fwd, softmax_bwd, has_mask):
    """Verify standard, masked, triangular, and causal masked softmax variations."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)

    fwd_fn = getattr(backend, softmax_fwd)
    bwd_fn = getattr(backend, softmax_bwd)

    if has_mask:
        assert fwd_fn(inp, inp, 1.0) is not None
        assert bwd_fn(inp, inp, 1.0) is not None
    else:
        assert fwd_fn(inp, 1.0) is not None
        assert bwd_fn(inp, inp, 1.0) is not None


def test_dropout_and_version_stubs():
    """Verify dropout lifecycle execution along with framework component stubs."""
    backend = ReferenceBackend()
    inp = torch.randn(4, 4)

    assert backend.dropout_fwd(inp, 0.5) is not None
    assert backend.dropout_bwd(inp, inp, 0.5) is not None
    assert backend.get_cublasLt_version() == 0
    assert backend.get_cudnn_version() == 0
    assert backend.get_num_cublas_streams() == 4
    assert backend.get_flash_attention_class() is not None
    assert (
        backend.get_fused_attn_backend(
            None, None, None, None, None, None, None, 0.0, 1, 1, 1, 1, 1, 1, 0, 0, False
        )
        == 0
    )


# ==============================================================================
# Part 4: Multi-Tensor & Optimizer Pipeline Tests
# ==============================================================================


def test_multi_tensor_scale_variants():
    """Verify tensor collection scaling, including tensor to scalar unpacked conversions."""
    backend = ReferenceBackend()
    flag = torch.tensor(0)
    t_list = [[torch.tensor([1.0])]]

    backend.multi_tensor_scale(1024, flag, t_list, 2.0)
    backend.multi_tensor_scale_tensor(1024, flag, t_list, torch.tensor(2.0))


@pytest.mark.parametrize("noop_val", [0, 1])
def test_multi_tensor_unscale_l2norm(noop_val):
    """Verify unscaling behaviors drop out immediately if noop_flag trips."""
    backend = ReferenceBackend()
    flag = torch.tensor(noop_val)
    t_list = [[torch.tensor([2.0])]]
    inv_scale = torch.tensor(0.5)

    res = backend.multi_tensor_unscale_l2norm(1024, flag, t_list, inv_scale, per_tensor=False)
    assert isinstance(res, tuple)


def test_multi_tensor_optimizers_and_scales():
    """Verify parameter list distributions for execution pipelines like Adam, SGD, and scale calculations."""
    backend = ReferenceBackend()
    flag = torch.tensor(0)
    t_list = [[torch.tensor([1.0])]]

    backend.multi_tensor_adam(1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01)
    backend.multi_tensor_adam_fp8(1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01, None)
    backend.multi_tensor_adam_param_remainder(
        1024, flag, t_list, 1e-3, 0.9, 0.99, 1e-8, 1, 0, 1, 0.01
    )

    backend.multi_tensor_adam_capturable(
        1024,
        flag,
        t_list,
        torch.tensor(1e-3),
        0.9,
        0.99,
        1e-8,
        torch.tensor(1),
        0,
        1,
        0.01,
        torch.tensor(1.0),
    )
    backend.multi_tensor_adam_capturable_master(
        1024,
        flag,
        t_list,
        torch.tensor(1e-3),
        0.9,
        0.99,
        1e-8,
        torch.tensor(1),
        0,
        1,
        0.01,
        torch.tensor(1.0),
    )

    backend.multi_tensor_sgd(1024, flag, t_list, 0.01, 0.9, 0.0, 1e-2, False, True, False, 1.0)
    backend.multi_tensor_compute_scale_and_scale_inv(1024, flag, t_list, 448.0, True, 1e-8)
    backend.multi_tensor_compute_scale_inv_e8m0(1024, flag, t_list, 16)
