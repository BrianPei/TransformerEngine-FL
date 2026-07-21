# TransformerEngine-FL L2 QA 的 Huawei Ascend 适配与测试报告

## 1. 范围

L2 包含两个目录：

```text
qa/L2_jax_distributed_unittest
qa/L2_jax_unittest
```

本次分别新增 `test_ascend.sh`，并增加统一入口 `qa/run_l2_ascend.sh`。原始 `test.sh` 未修改。

## 2. 运行环境

| 项目 | 配置 |
| --- | --- |
| 容器 | `TE-FL` |
| 项目路径 | `/data/TE-FL-clone-lt/TransformerEngine-FL` |
| 分支 | `feature/fix-build` |
| NPU | 16 x Huawei Ascend 910 |
| Python | 3.10.12，虚拟环境 `te-fl-py310` |
| PyTorch/Torch-NPU | 2.7.1+cpu / 2.7.1.post2 |
| CANN | 9.0.0 |
| JAX/JAXLIB | 未安装 |
| TE JAX-NPU native extension | 当前项目未提供 |

## 3. 原流程为何不能直接运行

`L2_jax_distributed_unittest/test.sh` 设置 `/usr/local/cuda/bin/ptxas`，使用 `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`，并执行 JAX GPU distributed collective 测试。

`L2_jax_unittest/test.sh` 同样依赖 PTXAS 和 XLA GPU custom calls，还运行 JAX MNIST、encoder 示例及 `--xla_gpu_deterministic_ops`。

这些不是普通的设备字符串差异。当前容器只有 PyTorch 的 Ascend 运行栈，没有可供 TransformerEngine-FL 使用的 JAX Ascend runtime、JAX-NPU custom call 库或对应 collective backend。因此不能通过安装 CPU 版 JAX 来代表 NPU 测试。

## 4. 新增内容

```text
qa/L2_jax_distributed_unittest/test_ascend.sh
qa/L2_jax_unittest/test_ascend.sh
qa/run_l2_ascend.sh
```

两个入口都调用公共工具：

```text
qa/ascend_write_junit_skip.py
```

它们为明确缺失的底层能力生成标准 JUnit skipped，不返回伪造的 passed。运行器会继续执行全部目录并给出状态汇总。

## 5. 运行与结果

```bash
docker exec -it TE-FL /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate
cd /data/TE-FL-clone-lt/TransformerEngine-FL
bash qa/run_l2_ascend.sh
```

服务器实际结果：

| L2 目录 | 状态 | JUnit |
| --- | --- | --- |
| `L2_jax_distributed_unittest` | SKIPPED | 1 test，1 skipped，0 failures，0 errors |
| `L2_jax_unittest` | SKIPPED | 1 test，1 skipped，0 failures，0 errors |

报告位置：

```text
logs/L2_jax_distributed_unittest-ascend/pytest.xml
logs/L2_jax_unittest-ascend/pytest.xml
```

运行器退出码为 0，因为这是经过识别并写入报告的“不支持”，不是脚本错误。CI 会把测试显示为 skipped，而不是 passed。

## 6. 后续转为真实测试的条件

要将这两个入口改为真实 Ascend 测试，至少需要：

1. 与 Python 3.10、CANN 9 和 Ascend 910 匹配的 JAX/JAXLIB Ascend runtime。
2. TransformerEngine-FL 的 JAX-NPU native extension 和 custom call 实现。
3. 可替换 XLA GPU/NCCL collective 的 Ascend distributed backend。
4. 对 MNIST、encoder、distributed dense/attention 用例进行数值和多卡验证。

满足这些条件前，当前 skipped 报告是准确的能力边界记录。
