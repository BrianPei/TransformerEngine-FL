import torch

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class grouped_gemmTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Moe permute Operations",
            "Test correctness of all moe permute operations across backends",
        )
        self.backends = get_available_backends()
        self.device = device

    def test_grouped_gemm_equivalence(self, grad, has_bias, has_pre_gelu, single_output):
        print(
            "\n test te_general_grouped_gemm"
            f" grad:{grad} has_bias:{has_bias},has_pre_gelu:{has_pre_gelu},single_output:{single_output}"
        )
        import transformer_engine_torch_nv as tex

        num_gemms = 2
        m, k, n = 128, 32, 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16

        if dtype == torch.float16:
            te_dtype = tex.DType.kFloat16
        elif dtype == torch.float32:
            te_dtype = tex.DType.kFloat32
        elif dtype == torch.bfloat16:
            te_dtype = tex.DType.kBFloat16
        else:
            raise ValueError(f"不支持的 dtype: {torch_dtype}")

        torch.manual_seed(42)

        A_list = [torch.randn((k, n), device=device, dtype=dtype) for _ in range(num_gemms)]
        B_list = [torch.randn((m, k), device=device, dtype=dtype) for _ in range(num_gemms)]

        bias_list_py_bias = [
            (
                torch.randn(n, device=device, dtype=dtype)
                if has_bias
                else torch.empty(0, device=device, dtype=dtype)
            )
            for _ in range(num_gemms)
        ]
        bias_list_te = [b.clone() for b in bias_list_py_bias]

        pre_gelu_list_py = [
            (
                torch.randn(m, n, device=device, dtype=dtype)
                if has_pre_gelu
                else torch.empty(0, device=device, dtype=dtype)
            )
            for _ in range(num_gemms)
        ]
        pre_gelu_list_te = [p.clone() for p in pre_gelu_list_py]

        if single_output:
            D_list_py = [torch.empty(m * num_gemms, n, device=device, dtype=dtype)]
            D_list_te = [torch.empty(m * num_gemms, n, device=device, dtype=dtype)]
        else:
            D_list_py = [torch.empty(m, n, device=device, dtype=dtype) for _ in range(num_gemms)]
            D_list_te = [torch.empty(m, n, device=device, dtype=dtype) for _ in range(num_gemms)]
        workspace_py = [torch.empty(1024 * 1024, device=device, dtype=torch.uint8)]
        workspace_te = [torch.empty(1024 * 1024, device=device, dtype=torch.uint8)]

        tex.te_general_grouped_gemm(
            A_list,
            False,
            B_list,
            False,
            D_list_te,
            te_dtype,
            [],
            bias_list_te,
            te_dtype,
            single_output,
            pre_gelu_list_te,
            grad,
            workspace_te,
            1024 * 1024,
            False,
            False,
            0,
        )

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            print("backend:", backend)
            try:
                bias_list_py = [b.clone() for b in bias_list_py_bias]
                backend.te_general_grouped_gemm(
                    A_list,
                    False,
                    B_list,
                    False,
                    D_list_py,
                    te_dtype,
                    [],
                    bias_list_py,
                    te_dtype,
                    single_output,
                    pre_gelu_list_py,
                    grad,
                    workspace_py,
                    1024 * 1024,
                    False,
                    False,
                    0,
                )

                for py_d, te_d in zip(D_list_py, D_list_te):
                    self.assert_close(
                        py_d, te_d, rtol=1e-3, atol=1e-3, msg="Output D tensors mismatch!"
                    )

                if not grad and has_pre_gelu:
                    for py_p, te_p in zip(pre_gelu_list_py, pre_gelu_list_te):
                        self.assert_close(
                            py_p, te_p, rtol=1e-3, atol=1e-3, msg="Pre-GELU out tensors mismatch!"
                        )

                if grad or has_bias:
                    for py_b, te_b in zip(bias_list_py, bias_list_te):
                        self.assert_close(
                            py_b, te_b, rtol=1e-3, atol=1e-3, msg="Bias gradient tensors mismatch!"
                        )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘  {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ Test failed: {e}")

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("=" * 60)
        print(f"Available backends: {', '.join(self.backends)}")

        # gemm tests
        self.test_grouped_gemm_equivalence(False, False, False, False)
        self.test_grouped_gemm_equivalence(False, True, False, False)
        self.test_grouped_gemm_equivalence(False, False, True, False)

        self.test_grouped_gemm_equivalence(False, False, False, True)
        self.test_grouped_gemm_equivalence(False, True, False, True)
        self.test_grouped_gemm_equivalence(False, False, True, True)
        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = grouped_gemmTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
