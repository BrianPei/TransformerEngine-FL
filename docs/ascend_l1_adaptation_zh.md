# TransformerEngine-FL L1 QA 的 Huawei Ascend 适配与测试报告

## 1. 目标和适配原则

本次为 `qa` 下全部六个 `L1_*` 流程增加 Huawei Ascend 入口。所有原始 `test.sh` 均保持不变，每个目录只新增 `test_ascend.sh`。

适配遵循以下原则：

1. Ascend 已具备的 PyTorch、Torch-NPU、HCCL 和 Transformer Engine 能力运行真实 NPU 测试。
2. 原测试依赖 CUDA、NCCL、PTXAS、TensorRT 或仅支持 CUDA 的外部项目时，不把未执行伪装为通过，而是生成标准 JUnit `skipped`。
3. Ascend 测试日志默认写入仓库的 `logs/<suite>-ascend`，不覆盖 CUDA CI 的 `/logs` 产物。
4. 分层运行器执行全部目录并输出 `PASSED`、`SKIPPED` 或 `FAILED` 汇总。

## 2. 测试环境

| 项目 | 配置 |
| --- | --- |
| Docker 容器 | `TE-FL` |
| 镜像 | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 项目路径 | `/data/TE-FL-clone-lt/TransformerEngine-FL` |
| Git 分支 | `feature/fix-build` |
| 基础提交 | `3b5fbb58` |
| NPU | 16 x Huawei Ascend 910 |
| Python 环境 | `/data/TE-FL-clone-lt/te-fl-py310` |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cpu |
| Torch-NPU | 2.7.1.post2 |
| CANN | 9.0.0 |
| FlagGems | 5.0.2 |
| Triton-Ascend | 3.2.0 |
| TransformerEngine-FL | 2.14.0+3b5fbb58 |
| pytest | 8.2.1 |

服务器已确认 `torch.distributed.is_hccl_available()` 为 `True`，可见 NPU 数量为 16。当前没有安装 JAX；`/workspace/Megatron-LM-FL` 与 `/opt/pytorch/lightning-thunder` 也不存在。

## 3. 新增文件

```text
qa/ascend_hccl_distributed_worker.py
qa/test_ascend_hccl_distributed.py
qa/L1_cpp_distributed/test_ascend.sh
qa/L1_jax_distributed_unittest/test_ascend.sh
qa/L1_pytorch_distributed_unittest/test_ascend.sh
qa/L1_pytorch_mcore_integration/test_ascend.sh
qa/L1_pytorch_onnx_unittest/test_ascend.sh
qa/L1_pytorch_thunder_integration/test_ascend.sh
qa/run_l1_ascend.sh
```

公共的 `qa/ascend_write_junit_skip.py` 沿用 L0 适配，用于生成可被 CI 正确识别的 skipped 报告。

## 4. 真实 HCCL 分布式测试

`qa/test_ascend_hccl_distributed.py` 是 pytest/JUnit 外层入口，它用 `torch.distributed.run` 启动两个 worker。`qa/ascend_hccl_distributed_worker.py` 在两张 NPU 上执行：

1. 每个 rank 根据 `LOCAL_RANK` 选择 NPU。
2. 使用 `backend="hccl"` 初始化进程组。
3. 对 rank 张量执行 HCCL `all_reduce` 并检查求和结果。
4. 在每个 rank 创建真实的 `transformer_engine.pytorch.Linear` NPU 层。
5. 执行前向、loss、反向传播。
6. 对权重梯度执行 HCCL `all_reduce` 和平均。
7. 检查 loss、输入梯度和权重梯度均为有限值。
8. 两个 rank 完成 barrier 后由 rank 0 输出 `ASCEND_HCCL_TE_OK world_size=2`。

该测试同时覆盖真实 HCCL collective、TE NPU backend、FlagGems GEMM、autograd 和跨 rank 梯度通信。

## 5. 各目录修改与结果

| L1 目录 | Ascend 处理 | 服务器结果 |
| --- | --- | --- |
| `L1_cpp_distributed` | CUDA C++ `test_comm_gemm` 无 Ascend C++ 构建，运行两卡 HCCL + TE 等价能力测试 | `1 passed`，78.37 秒 |
| `L1_jax_distributed_unittest` | 原流程硬编码 PTXAS、XLA GPU、NCCL；当前没有 JAX Ascend runtime | `1 skipped` |
| `L1_pytorch_distributed_unittest` | 用两卡 HCCL + TE 前反向和梯度归约替代 CUDA/NCCL 用例集 | `1 passed`，74.79 秒 |
| `L1_pytorch_mcore_integration` | 原命令硬编码 NCCL/CUDA，且服务器没有 Megatron-LM-FL checkout | `1 skipped` |
| `L1_pytorch_onnx_unittest` | 原测试全面硬编码 CUDA tensor/custom op，ORT 无 Ascend execution provider | `1 skipped` |
| `L1_pytorch_thunder_integration` | Thunder TE executor 面向 CUDA/FP8，且外部 checkout 不存在 | `1 skipped` |

通过项的 JUnit 报告：

```text
logs/L1_cpp_distributed-ascend/pytest_hccl_te.xml
logs/L1_pytorch_distributed_unittest-ascend/pytest_hccl_te.xml
```

四个不支持项均在各自的 `logs/<suite>-ascend` 下生成一个 `tests=1, skipped=1, failures=0, errors=0` 的 XML。

## 6. 运行方法

进入容器并加载环境：

```bash
docker exec -it TE-FL /bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate
cd /data/TE-FL-clone-lt/TransformerEngine-FL
unset ASCEND_RT_VISIBLE_DEVICES
```

运行全部 L1：

```bash
bash qa/run_l1_ascend.sh
```

只运行指定项：

```bash
bash qa/run_l1_ascend.sh L1_cpp_distributed L1_pytorch_distributed_unittest
```

也可直接运行单个入口：

```bash
bash qa/L1_pytorch_distributed_unittest/test_ascend.sh
```

至少要让进程看到两张 NPU。需要限制设备时可使用：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

## 7. 已知边界

- `L1_cpp_distributed` 验证的是原 CUDA C++ 通信 GEMM 所对应的 HCCL、GEMM 和多卡执行能力，不是把 `.cu` 源码直接编译为 Ascend 二进制。
- 当前项目没有 TE JAX-NPU native extension，因此安装 CPU JAX 不能代表 Ascend 测试。
- ONNX Runtime 缺少 Ascend execution provider，不能用 CPU ORT 的结果宣称 NPU 集成通过。
- Megatron-LM-FL 和 Thunder 需要各自提供经验证的 Ascend 分支及依赖后，才能把 skipped 替换为真实集成测试。
- HCCL 运行中的 `get_device_capability isn't implemented`、未显式传 barrier device id 等信息是 Torch-NPU/FlagGems 警告，本次数值、梯度和 collective 断言均通过。
