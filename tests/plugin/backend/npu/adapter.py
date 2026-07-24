"""Ascend NPU adapter for running upstream Transformer Engine tests."""

from npu_patch import apply_ascend_npu_patch


def apply() -> None:
    """Apply the Ascend runtime compatibility layer."""
    apply_ascend_npu_patch()
