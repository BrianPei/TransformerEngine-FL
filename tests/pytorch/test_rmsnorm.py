import sys
import pytest
from unittest.mock import MagicMock
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation (Strategic Mocking)
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems

# 模拟 flag_gems.add 算子
mock_flag_gems.add = lambda x, y, *args, **kwargs: x + y

# 模拟前向和反向的核心 rms_norm 算子，保证返回符合形状预期的 Tensor
def mock_rms_norm_forward(input_tensor, normalized_shape, weight, eps):
    # 前向返回 (y, rstdevs)。特意让 rstdevs 默认多一个维度，以触发源码中的 shape != view 调整分支
    y = input_tensor * weight
    # 构造一个形状不匹配的 rstdevs（例如末尾多一个1维），强迫触发 .view(input.shape[:-1])
    rstdevs_shape = list(input_tensor.shape[:-1]) + [1]
    rstdevs = torch.ones(rstdevs_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    return y, rstdevs

def mock_rms_norm_backward(dy, x, rsigma, normalized_shape, gamma, eps):
    # 反向返回 (dx, dw)
    dx = dy * gamma
    dw = torch.ones_like(gamma)
    return dx, dw

mock_flag_gems.rms_norm_forward = mock_rms_norm_forward
mock_flag_gems.rms_norm_backward = mock_rms_norm_backward

# 直接导入被测的 Python 源码函数，绕过 OpManager 的动态路由拦截
from transformer_engine.plugin.core.backends.flagos.impl.rmsnorm import (
    rmsnorm_fwd_fl,
    rmsnorm_bwd_fl,
)

# ==============================================================================
# Part 1: rmsnorm_fwd_fl Forward Path Tests
# ==============================================================================

@pytest.mark.parametrize("zero_centered_gamma", [True, False])
@pytest.mark.parametrize("input_shape", [(4, 8), (2, 3, 4)])
def test_rmsnorm_fwd_lifecycle(zero_centered_gamma, input_shape):
    """Verify forward RMSNorm lifecycle, handling gamma centering and shape reshaping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.randn(input_shape, device=device)
    weight = torch.ones(input_shape[-1], device=device)
    
    y, _, rstdevs = rmsnorm_fwd_fl(
        input=inp,
        weight=weight,
        eps=1e-5,
        ln_out=None,
        quantizer=None,
        odtype=None,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma
    )
    
    # 验证输出类型和正确性
    assert isinstance(y, torch.Tensor)
    assert isinstance(rstdevs, torch.Tensor)
    
    # 核心覆盖检查：rstdevs 的形状必须完美契合 input.shape[:-1]
    assert rstdevs.shape == inp.shape[:-1]


# ==============================================================================
# Part 2: rmsnorm_bwd_fl Backward Path Tests
# ==============================================================================

@pytest.mark.parametrize("zero_centered_gamma", [True, False])
def test_rmsnorm_bwd_lifecycle(zero_centered_gamma):
    """Verify backward RMSNorm execution and scaling adjustments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dy = torch.randn(4, 8, device=device)
    x = torch.randn(4, 8, device=device)
    rsigma = torch.ones(4, device=device)
    gamma = torch.ones(8, device=device)
    
    dx, dw = rmsnorm_bwd_fl(
        dy=dy,
        x=x,
        rsigma=rsigma,
        gamma=gamma,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
        eps=1e-5
    )
    
    assert isinstance(dx, torch.Tensor)
    assert isinstance(dw, torch.Tensor)
    assert dx.shape == x.shape
    assert dw.shape == gamma.shape