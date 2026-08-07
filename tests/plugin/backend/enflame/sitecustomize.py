"""Runtime patches for Enflame/GCU test execution."""

from __future__ import annotations

import os


def _skip(reason: str):
    return False, reason


def apply_enflame_patch() -> None:
    if os.environ.get("ENFLAME_ENABLE_PATCHES", "1") != "1":
        return

    try:
        import torch
        import torch_gcu
        from torch_gcu import transfer_to_gcu  # noqa: F401
    except Exception:
        return

    import transformer_engine

    transformer_engine.TE_DEVICE_TYPE = "gcu"
    transformer_engine.TE_PLATFORM = torch.gcu

    os.environ.setdefault("PLATFORM", "enflame")
    os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
    os.environ.setdefault("TE_FL_PREFER", "reference")
    os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
    os.environ.setdefault("NVTE_FLASH_ATTN", "0")
    os.environ.setdefault("NVTE_FUSED_ATTN", "0")
    os.environ.setdefault("NVTE_UNFUSED_ATTN", "1")
    os.environ.setdefault("NVTE_UnfusedDPA_Emulate_FP8", "1")

    try:
        import transformer_engine.plugin.core.backends.vendor.enflame.enflame as enflame_mod

        if not hasattr(enflame_mod, "_enflame_libs_loaded"):
            enflame_mod._enflame_libs_loaded = False
    except Exception:
        pass

    def _sync_torch_cuda_api() -> None:
        max_visible_devices = int(os.environ.get("ENFLAME_NPROC_PER_NODE", "2"))
        actual_device_count = torch.gcu.device_count()

        torch.cuda.is_available = torch.gcu.is_available
        torch.cuda.device_count = lambda: min(actual_device_count, max_visible_devices)
        torch.cuda.current_device = torch.gcu.current_device
        torch.cuda.set_device = torch.gcu.set_device
        torch.cuda.get_device_name = torch.gcu.get_device_name
        torch.cuda.manual_seed = torch.gcu.manual_seed
        torch.cuda.synchronize = torch.gcu.synchronize
        torch.cuda.empty_cache = torch.gcu.empty_cache
        torch.cuda.memory_allocated = torch.gcu.memory_allocated
        torch.cuda.max_memory_allocated = torch.gcu.max_memory_allocated
        torch.cuda.get_rng_state = torch.gcu.get_rng_state
        torch.cuda.set_rng_state = torch.gcu.set_rng_state
        torch.cuda.is_current_stream_capturing = torch.gcu.is_current_stream_capturing

        orig_get_device_properties = torch.cuda.get_device_properties

        def _get_device_properties(device=None):
            props = orig_get_device_properties(device)
            if hasattr(props, "gcnArchName"):
                return props

            class _Proxy:
                def __init__(self, inner):
                    self._inner = inner
                    self.gcnArchName = getattr(inner, "gcnArchName", "gcu")

                def __getattr__(self, name):
                    return getattr(self._inner, name)

                def __repr__(self):
                    return repr(self._inner)

            return _Proxy(props)

        torch.cuda.get_device_properties = _get_device_properties
        torch.cuda.get_device_capability = lambda device=None: (
            getattr(torch.gcu.get_device_properties(device), "major", 0),
            getattr(torch.gcu.get_device_properties(device), "minor", 0),
        )

    _sync_torch_cuda_api()

    import transformer_engine.pytorch as te_pytorch
    import transformer_engine.pytorch.quantization as quantization
    import transformer_engine.pytorch.utils as pytorch_utils

    te_pytorch.is_fp8_available = lambda return_reason=False: _skip(
        "FP8 execution is not available on Enflame/GCU."
    ) if return_reason else False
    te_pytorch.is_mxfp8_available = lambda return_reason=False: _skip(
        "MXFP8 execution is not available on Enflame/GCU."
    ) if return_reason else False
    te_pytorch.is_nvfp4_available = lambda return_reason=False: _skip(
        "NVFP4 execution is not available on Enflame/GCU."
    ) if return_reason else False
    te_pytorch.is_fp8_block_scaling_available = lambda return_reason=False: _skip(
        "FP8 block scaling is not available on Enflame/GCU."
    ) if return_reason else False

    pytorch_utils._get_device_compute_capability = lambda device: (3, 0)

    quantization.check_fp8_support = lambda: _skip("FP8 execution is not supported on Enflame/GCU.")
    quantization.check_mxfp8_support = lambda: _skip(
        "MXFP8 execution is not supported on Enflame/GCU."
    )
    quantization.check_nvfp4_support = lambda: _skip(
        "NVFP4 execution is not supported on Enflame/GCU."
    )
    quantization.check_fp8_block_scaling_support = lambda: _skip(
        "FP8 block scaling is not supported on Enflame/GCU."
    )


apply_enflame_patch()
