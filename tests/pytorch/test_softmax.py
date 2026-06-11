import sys
import pytest
from unittest.mock import MagicMock
import torch

# ==============================================================================
# Part 0: Fine-Grained Dependency Isolation (Strategic Mocking)
# ==============================================================================
mock_flag_gems = MagicMock()
sys.modules["flag_gems"] = mock_flag_gems

# 模拟 flag_gems 算子行为，确保运算和类型转换平稳返回对应的 PyTorch 基础映射
mock_flag_gems.to_copy = lambda x, *args, **kwargs: x.to(kwargs.get("device", x.device)).to(kwargs.get("dtype", x.dtype))
mock_flag_gems.mul = lambda x, y, *args, **kwargs: x * (y.to(x.device) if isinstance(y, torch.Tensor) else y)
mock_flag_gems.add = lambda x, y, *args, **kwargs: x + y
mock_flag_gems.sub = lambda x, y, *args, **kwargs: x - y
mock_flag_gems.softmax = lambda x, dim, *args, **kwargs: torch.softmax(x, dim=dim)
mock_flag_gems.eq_scalar = lambda x, value, *args, **kwargs: (x == value)
mock_flag_gems.masked_fill = lambda x, mask, value, *args, **kwargs: torch.masked_fill(x, mask, value)
mock_flag_gems.all_dim = lambda x, dim, keepdim, *args, **kwargs: torch.all(x, dim=dim, keepdim=keepdim)
mock_flag_gems.sum_dim = lambda x, dim, keepdim, *args, **kwargs: torch.sum(x, dim=dim, keepdim=keepdim)

# 直接导入被测的 Python 源码函数，绕过算子路由拦截
from transformer_engine.plugin.core.backends.flagos.impl.softmax import (
    scaled_masked_softmax_forward_fl,
    scaled_masked_softmax_backward_fl,
)

# ==============================================================================
# Part 1: Forward Path (scaled_masked_softmax_forward_fl) Tests
# ==============================================================================

@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("scale_is_tensor", [True, False])
@pytest.mark.parametrize("device_mismatch", [True, False])
@pytest.mark.parametrize("is_4d_broadcast", [True, False])
def test_scaled_masked_softmax_fwd_matrix(mask_dtype, scale_is_tensor, device_mismatch, is_4d_broadcast):
    """Walk through all forward control branches including masking types and cross-device routing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造输入张量形状
    input_shape = (2, 2, 4, 4) if is_4d_broadcast else (4, 4)
    inp = torch.randn(input_shape, device=device)
    
    # 构造 Mask 形状及跨设备环境
    if is_4d_broadcast:
        mask_shape = (2, 1, 4, 4)
    else:
        mask_shape = input_shape
        
    mask_device = "cpu" if (device_mismatch and device == "cuda") else device
    
    if mask_dtype.is_floating_point:
        mask = torch.randn(mask_shape, device=mask_device, dtype=mask_dtype)
    else:
        # 整型 Mask，模拟部分遮蔽与全遮蔽情况
        mask = torch.ones(mask_shape, device=mask_device, dtype=mask_dtype)
        if mask_shape == input_shape:
            mask[0, 0] = 0  # 确保包含未被屏蔽的通路
            
    # 构造 Scale Factor
    if scale_is_tensor:
        scale_factor = torch.tensor(2.0, device=mask_device)  # 借用不同设备触发对应分流
    else:
        scale_factor = 2.0
        
    out = scaled_masked_softmax_forward_fl(
        input=inp,
        mask=mask,
        scale_factor=scale_factor
    )
    
    assert isinstance(out, torch.Tensor)
    assert out.shape == inp.shape


# ==============================================================================
# Part 2: Backward Path (scaled_masked_softmax_backward_fl) Tests
# ==============================================================================

@pytest.mark.parametrize("scale_is_tensor", [True, False])
@pytest.mark.parametrize("device_mismatch", [True, False])
def test_scaled_masked_softmax_bwd_matrix(scale_is_tensor, device_mismatch):
    """Walk through all backward control paths with float and tensor scale representations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_grad = torch.randn(4, 4, device=device, dtype=torch.float16)
    softmax_results = torch.randn(4, 4, device=device, dtype=torch.float16)
    
    if scale_is_tensor:
        scale_device = "cpu" if (device_mismatch and device == "cuda") else device
        scale_factor = torch.tensor(0.5, device=scale_device)
    else:
        scale_factor = 0.5
        
    grad_input = scaled_masked_softmax_backward_fl(
        output_grad_=output_grad,
        softmax_results_=softmax_results,
        scale_factor=scale_factor
    )
    
    assert isinstance(grad_input, torch.Tensor)
    assert grad_input.shape == output_grad.shape
    # 确保稳定返回到原始计算精度
    assert grad_input.dtype == output_grad.dtype