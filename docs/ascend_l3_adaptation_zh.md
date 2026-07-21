# TransformerEngine-FL L3 QA 的 Huawei Ascend 适配与测试报告

## 1. 范围和目标

L3 当前只有：

```text
qa/L3_pytorch_FA_versions_test
```

本次新增：

```text
qa/L3_pytorch_FA_versions_test/test_ascend.sh
qa/run_l3_ascend.sh
```

原始 `qa/L3_pytorch_FA_versions_test/test.sh` 保持不变。

## 2. 运行环境

| 项目 | 配置 |
| --- | --- |
| 容器 | `TE-FL` |
| 项目路径 | `/data/TE-FL-clone-lt/TransformerEngine-FL` |
| 分支 | `feature/fix-build` |
| NPU | 16 x Huawei Ascend 910 |
| Python | 3.10.12 |
| PyTorch/Torch-NPU | 2.7.1+cpu / 2.7.1.post2 |
| CANN | 9.0.0 |
| FlagGems/Triton-Ascend | 5.0.2 / 3.2.0 |
| NVIDIA CUDA 与 SM 架构 | 不存在 |

## 3. 原测试的作用

原始脚本通过 `torch.cuda.get_device_capability(0)` 获取 NVIDIA GPU 的 SM 架构，再按架构安装多个 Dao-AILab FlashAttention 版本：

- SM 大于 90：测试 `flash-attn==2.8.3`；
- SM 90：测试 `2.7.3`、`2.8.3` 和 `3.0.0b1`；
- FlashAttention 3 从 `flash-attention/hopper` 构建。

每个版本安装后运行 `tests/pytorch/attention/test_attention.py`。所以该 L3 流程测试的是 Transformer Engine 与 NVIDIA CUDA FlashAttention 不同版本、不同 SM 架构的兼容性，并不是一个通用 attention 功能测试。

## 4. Ascend 适配方式

Dao-AILab `flash-attn` 2.x/3.x 编译 CUDA kernel，依赖 NVIDIA SM、CUDA toolchain，并且 3.x 的 Hopper 路径专门面向 NVIDIA Hopper。Huawei Ascend 没有对应 SM 架构，不能安装这些 wheel 或编译这些 kernel。

因此 `test_ascend.sh` 不尝试伪造 `torch.cuda.get_device_capability`，也不把 FlagGems attention 冒充成某个 CUDA FlashAttention 版本。它生成一个明确说明原因的 JUnit skipped 报告。

Ascend 上 TE backend、FlagGems GEMM 和真实 NPU 前反向能力已由 L0 测试覆盖；两卡 HCCL + TE 分布式能力由 L1 测试覆盖。这些结果不能替代“CUDA FlashAttention 版本矩阵兼容性”结论。

## 5. 运行与测试结果

```bash
docker exec -it TE-FL /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate
cd /data/TE-FL-clone-lt/TransformerEngine-FL
bash qa/run_l3_ascend.sh
```

服务器实际汇总：

```text
L3_pytorch_FA_versions_test: SKIPPED (CUDA FlashAttention version matrix)
```

JUnit 报告：

```text
logs/L3_pytorch_FA_versions_test-ascend/pytest.xml
```

报告计数为：

```text
tests=1
skipped=1
failures=0
errors=0
```

## 6. 后续方案

如果项目后续需要 L3 级别的 Ascend attention 兼容矩阵，应新定义面向 Ascend 的测试目标，例如 CANN、Torch-NPU、FlagGems 或 Ascend attention 实现的版本组合，并为每个组合执行相同的数值、dtype、mask、前反向和稳定性用例。该矩阵与 CUDA `flash-attn` 版本矩阵属于不同产品能力，不应共用同一个“通过”结论。
