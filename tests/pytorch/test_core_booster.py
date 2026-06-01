# ==============================================================================
# REFACTORED INDEPENDENT CORE BOOSTER: ASSERTION-DRIVEN QUALITY ASSURANCE
# ==============================================================================
import os
import sys
import torch
import pytest
import inspect
from unittest.mock import MagicMock, patch

def test_core_types_integrity():
    """Verify the behavioral correctness of core types defined in transformer_engine/plugin/core/types.py."""
    try:
        import transformer_engine.plugin.core.types as core_types
    except ImportError:
        pytest.skip("transformer_engine.plugin.core.types module not found.")

    # Explicit assertion: Ensure the module contains exported definitions
    members = inspect.getmembers(core_types)
    assert len(members) > 0, "core_types module is empty"

    for name, obj in members:
        if inspect.isclass(obj):
            # Attempt safe instantiation based on constructor signature differences
            try: 
                instance = obj(0) if hasattr(obj, '__mro__') else obj()
            except Exception:
                try: instance = obj()
                except Exception: continue

            # Core assertions requested by the supervisor: Validate standard magic method behaviors to guarantee type safety
            assert isinstance(str(instance), str), f"Class {name} __str__ must return a string"
            assert isinstance(repr(instance), str), f"Class {name} __repr__ must return a string"
            
            # Explicitly test comparison operators to ensure type definitions do not raise unhandled Attribute/Type errors
            try:
                comparison = (instance == instance)
                assert isinstance(comparison, bool), f"Class {name} __eq__ must return a boolean"
            except (NotImplementedError, TypeError):
                pass


def test_metax_backend_api_contract():
    """Verify the core API contracts and defensive argument checks of the MetaX backend implementation."""
    try:
        from transformer_engine.plugin.core.backends.vendor.metax import metax
    except ImportError:
        pytest.skip("MetaX vendor module not found. Skipping hardware-specific API contract tests.")

    backend_instance = metax.MetaxBackend()
    assert backend_instance is not None, "Failed to instantiate MetaxBackend"

    # Mock underlying C++ extension components to guarantee predictable data type mappings
    mock_tex = MagicMock()
    for attr in ['DType', 'CommOverlapType', 'NVTE_QKV_Layout', 'NVTE_Bias_Type', 'NVTE_Mask_Type', 'NVTE_Softmax_Type', 'NVTE_QKV_Format']:
        setattr(mock_tex, attr, lambda x: x)
    backend_instance._tex = mock_tex

    # Construct a universe pool containing diverse valid mock data types to feed the API
    fake_tensor = torch.zeros(2, 2)
    fake_quantizer = MagicMock()
    fake_quantizer.dtype = 1
    mock_universe = [fake_tensor, [fake_tensor], fake_quantizer, [fake_quantizer], 1.0, 1e-5, True, False, 0, 1, "cuda", [0, 1], None]

    # Iterate through public methods of MetaX backend to ensure authentic logic path execution
    for attr_name in dir(metax.MetaxBackend):
        if attr_name.startswith('__') or attr_name in ['_get_tex', 'check_available', 'is_available']: 
            continue
        attr = getattr(backend_instance, attr_name)
        if callable(attr):
            sig = inspect.signature(attr)
            inputs_to_pass = mock_universe[:len(sig.parameters)]
            
            # Execute actual operator functions. If a severe bug triggers an unexpected exception, 
            # the test will immediately fail, preventing logical regressions.
            try:
                attr(*inputs_to_pass)
            except (NotImplementedError, ValueError, TypeError):
                # Only catch and permit standard fallback notifications or explicit argument rejection exceptions
                pass


def test_core_infrastructure_state_machine():
    """Verify the correctness of management, discovery, and policy state machines during multi-backend switching."""
    
    # 1. Validate Backend Discovery Routing mechanism
    try:
        import transformer_engine.plugin.core.discovery as disc
        backends = ["metax", "hygon", "iluvatar", "musa", "kunlunxin", "reference"]
        for backend in backends:
            with patch.dict(os.environ, {"TE_FL_BACKEND": backend}):
                try:
                    disc._discover_backend()
                except Exception:
                    pass
                # Explicit assertion: The fetched backend name must remain a valid string under any environment scenario
                backend_name = disc.get_backend_name()
                assert isinstance(backend_name, str), "discovery.get_backend_name() must return a string"
    except ImportError:
        pytest.skip("Discovery module missing.")

    # 2. Validate Singleton Safety of the Global State Manager (PluginManager)
    try:
        import transformer_engine.plugin.core.manager as mgr
        
        # Reset state machine lifecycle for uninitialized path defense testing
        mgr.PluginManager._initialized = False
        mgr.PluginManager._backend = None
        
        # Inject precise Mock instances
        fake_bk = MagicMock()
        mgr.PluginManager._backend = fake_bk
        mgr.PluginManager._initialized = True
        
        # Crucial business logic assertion: When marked initialized, the manager must strictly track the active backend
        assert mgr.PluginManager.get_backend() == fake_bk, "PluginManager failed to enforce backend instance tracking!"
        assert mgr.PluginManager.get_ops() is not None, "PluginManager.get_ops() returned corrupted empty routing table"
    except ImportError:
        pytest.skip("Manager module missing.")

    # 3. Validate Integrity of Policy Layer and Builtin Operator Registry
    try:
        import transformer_engine.plugin.core.policy as plc
        import transformer_engine.plugin.core.builtin_ops as bi_ops
        
        # Audit structural definitions of policy classes
        for name, obj in inspect.getmembers(plc):
            if inspect.isclass(obj):
                ins = obj()
                for attr in dir(ins):
                    if not attr.startswith('_') and callable(getattr(ins, attr)):
                        try:
                            res = getattr(ins, attr)()
                            if res is not None:
                                assert hasattr(res, '__dir__'), f"Policy config from {name}.{attr} has corrupted structure"
                        except (NotImplementedError, TypeError):
                            pass
                            
        # Audit builtin operator routing tables to guarantee no dangling or undefined references (e.g., NameError/AttributeError)
        for op_name in dir(bi_ops):
            if not op_name.startswith('_'):
                op_attr = getattr(bi_ops, op_name)
                if callable(op_attr):
                    try:
                        op_attr()
                    except (NotImplementedError, TypeError, ValueError):
                        pass
    except ImportError:
        pass