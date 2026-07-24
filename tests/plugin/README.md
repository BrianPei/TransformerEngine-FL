# TransformerEngine-FL Plugin Tests

This directory owns tests added for the TransformerEngine-FL plugin layer.
Upstream Transformer Engine tests remain in `tests/cpp`, `tests/jax`, and
`tests/pytorch`.

The test layout follows the implementation boundary:

- `plugin/`: plugin manager, policy, registry, and discovery behavior.
- `backend/`: shared backend contracts and operation suites.
- `backend/reference/`: reference backend tests.
- `backend/flagos/`: FlagOS backend tests that do not require a specific device.
- `backend/npu/`: real NPU tests and the adapter used to run upstream tests.

Use `run_upstream.py` with a platform adapter when an upstream test needs
runtime compatibility setup. Platform-specific behavior should stay in the
adapter instead of being added to the common CI workflow.
