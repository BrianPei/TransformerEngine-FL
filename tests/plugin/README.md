# TransformerEngine-FL Plugin Tests

This directory owns tests added for the TransformerEngine-FL plugin layer.
Upstream Transformer Engine tests remain in `tests/cpp`, `tests/jax`, and
`tests/pytorch`.

The test layout follows the implementation boundary:

- `plugin/`: plugin manager, policy, registry, and discovery behavior.
- `backend/`: shared backend contracts and operation suites.
- `backend/reference/`: reference backend tests.
- `backend/flagos/`: FlagOS backend tests that do not require a specific device.
- `backend/npu/`: Ascend NPU vendor backend-specific tests.

All pytest suites use the normal `python -m pytest` entry point. Platform runtime
compatibility belongs to the corresponding implementation under
`transformer_engine/plugin/core/backends/vendor/`, not to a test-only launcher
or the common CI workflow.
