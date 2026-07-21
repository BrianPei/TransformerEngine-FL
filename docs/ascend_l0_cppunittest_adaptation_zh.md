# TransformerEngine-FL Ascend L0 测试适配总结

## 1. 背景与结论

TransformerEngine-FL 原有的 `qa/L0_cppunittest/test.sh` 用于编译和运行 NVIDIA CUDA C++ 单元测试。该脚本依赖 CUDA Toolkit、`nvcc`、cuDNN、CUDA Runtime 和 CUDA 版 `libtransformer_engine.so`，不能直接在 Huawei Ascend NPU 环境运行。

本次适配没有修改原始 `test.sh`，也没有尝试把 CUDA `.cu` 测试源码伪装成 Ascend C++ 测试，而是新增独立的 Ascend QA 入口和真实 NPU smoke test：

```text
qa/L0_cppunittest/test_ascend.sh
transformer_engine/plugin/tests/test_backend_ascend_smoke.py
```

新的 Ascend 入口测试当前项目实际存在的 Torch-NPU、Triton-Ascend、FlagGems 和 FlagOS 插件后端。服务器实测结果为 128 个测试全部通过，0 个失败，0 个错误。

## 2. 运行环境

### 2.1 服务器与容器

| 项目 | 配置 |
| --- | --- |
| Docker 容器 | `TE-FL` |
| 容器镜像 | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 项目目录 | `/data/TE-FL-clone-lt/TransformerEngine-FL` |
| Git 分支 | `main` |
| 验证提交 | `3b5fbb58` |
| 加速设备 | 16 x Huawei Ascend 910 |

项目目录位于宿主机和容器共享的 `/data` 挂载点中，因此进入容器后可以直接访问源码，无需再次复制。

### 2.2 Python 与加速软件栈

| 软件 | 版本或位置 |
| --- | --- |
| Python 虚拟环境 | `/data/TE-FL-clone-lt/te-fl-py310` |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cpu |
| Torch-NPU | 2.7.1.post2 |
| TransformerEngine-FL | 2.14.0+3b5fbb58 |
| FlagGems | 5.0.2 |
| Triton-Ascend | 3.2.0 |
| pytest | 8.2.1 |
| CANN | 9.0.0 |

虽然 PyTorch 版本字符串包含 `+cpu`，但 `torch_npu` 会为 PyTorch 注册 `npu` 设备后端。设备可用性应通过以下代码判断：

```python
import torch
import torch_npu

print(torch.npu.is_available())
print(torch.npu.device_count())
```

不要使用 `torch.cuda.is_available()` 判断 Ascend NPU。

### 2.3 为什么使用 Python 3.10

容器默认 Python 是 3.12，但官方 `triton-ascend 3.2.0` 仅提供 Python 3.9 至 3.11 的 wheel。通用 Triton 不能驱动 Ascend，使用通用 Triton 时会出现：

```text
RuntimeError: 0 active drivers ([]). There should only be one.
```

因此使用系统 Python 3.10 创建了独立虚拟环境：

```bash
python -m pip install virtualenv
python -m virtualenv \
  -p /usr/bin/python3.10 \
  /data/TE-FL-clone-lt/te-fl-py310
```

### 2.4 CANN 9.0 兼容处理

Triton-Ascend 3.2.0 使用 CANN 8.5 的枚举名：

```text
RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE
```

CANN 9.0 中该枚举已改名为：

```text
RT_LIMIT_TYPE_SIMT_STACK_SIZE
```

当前虚拟环境已执行以下兼容替换：

```bash
sed -i \
  's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_STACK_SIZE/' \
  /data/TE-FL-clone-lt/te-fl-py310/lib/python3.10/site-packages/triton/backends/ascend/npu_utils.cpp
```

重新安装 `triton-ascend` 后需要重新应用此替换。

## 3. 原始 CUDA C++ 测试链路

原始文件保持不变：

```text
qa/L0_cppunittest/test.sh
```

其内容哈希与仓库 `HEAD` 一致：

```text
c7499282f44fc25bd0b3b875897f6b79f1b1e3ca
```

原始执行链路为：

```text
qa/L0_cppunittest/test.sh
        |
        +-- cd $TE_PATH/tests/cpp
        |
        +-- cmake -GNinja -Bbuild .
        |       |
        |       +-- tests/cpp/CMakeLists.txt
        |               |
        |               +-- tests/cpp/operator/CMakeLists.txt
        |               +-- tests/cpp/util/CMakeLists.txt
        |
        +-- cmake --build build
        |
        +-- ctest --test-dir build -j4
                |
                +-- $XML_LOG_DIR/ctest_cppunittest.xml
```

`tests/cpp/CMakeLists.txt` 明确声明：

```cmake
project(transformer_engine_tests LANGUAGES CUDA CXX)
find_package(CUDAToolkit REQUIRED)
```

测试源码直接使用以下 CUDA API：

```text
cudaMalloc
cudaMemcpy
cudaDeviceSynchronize
cudaStream_t
CUDA::cudart
CUDA::nvrtc
CUDNN::cudnn
```

因此原始测试只能验证 NVIDIA CUDA 原生实现，不能验证 Ascend/FlagOS Python 插件后端。

## 4. 新增代码及修改内容

### 4.1 原始文件未修改

以下文件没有代码变更：

```text
qa/L0_cppunittest/test.sh
tests/cpp/CMakeLists.txt
tests/cpp/operator/CMakeLists.txt
tests/cpp/util/CMakeLists.txt
tests/cpp/**/*.cu
```

这样可以保证 NVIDIA CUDA CI 的原有构建和测试行为不受 Ascend 适配影响。

### 4.2 新增 Ascend QA 入口

新增文件：

```text
qa/L0_cppunittest/test_ascend.sh
```

该脚本负责：

1. 从脚本位置自动推导项目根目录，避免依赖 `/opt/transformerengine`。
2. 创建 JUnit 报告目录。
3. 设置 `TE_FL_SKIP_CUDA=1`，避免加载 CUDA 原生后端。
4. 设置 `NVTE_FRAMEWORK=pytorch`。
5. 检查 `torch_npu` 是否安装。
6. 检查 `torch.npu.is_available()` 是否为真。
7. 在缺少 pytest 时自动安装 `pytest==8.2.1`。
8. 对每个适用测试文件启动独立 pytest 进程。
9. 为每个测试文件生成独立 JUnit XML。
10. 汇总失败状态，并通过脚本退出码反馈给 CI。

每个测试文件使用独立进程，是因为部分插件单元测试会向 `sys.modules` 注入 `MagicMock`。进程隔离可以防止前一个测试修改全局模块状态后污染后续测试。

### 4.3 新增真实 Ascend NPU smoke test

新增文件：

```text
transformer_engine/plugin/tests/test_backend_ascend_smoke.py
```

该测试不是 mock 测试，会在真实 Ascend NPU 上执行：

- Torch-NPU 设备选择和同步；
- Torch-NPU 原生矩阵乘；
- FlagGems 矩阵乘；
- FlagGems 与 Torch-NPU 结果对比；
- TransformerEngine-FL 设备类型检查；
- `transformer_engine.pytorch.Linear` 前向传播；
- `Linear` 反向传播；
- 输入梯度和权重梯度检查；
- 输出和梯度有限值检查。

关键断言包括：

```python
assert transformer_engine.te_device_type() == "npu"
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
assert inputs.grad is not None
assert layer.weight.grad is not None
```

### 4.4 Ascend 测试清单

`test_ascend.sh` 当前执行以下 9 个文件：

```text
test_backend_ascend_smoke.py
test_backend_flagos.py
test_backend_reference.py
test_backend_reference_activation.py
test_backend_reference_dropout.py
test_backend_reference_gemm.py
test_plugin_manager.py
test_plugin_policy.py
test_policy.py
```

其中真实 Ascend 计算由 `test_backend_ascend_smoke.py` 覆盖；其余测试覆盖插件注册、后端策略、实现选择和 reference 正确性。

### 4.5 为什么没有运行全部 `test_backend_flagos_*` 文件

以下既有文件是为“未安装真实 FlagGems”的环境编写的 CPU mock 单测：

```text
test_backend_flagos_fused_adam.py
test_backend_flagos_gemm.py
test_backend_flagos_multi_tensor.py
test_backend_flagos_rmsnorm.py
test_backend_flagos_softmax.py
```

这些文件会用 `MagicMock` 替换 `flag_gems`，并创建 CPU tensor。在真实 Ascend FlagGems 已加载的进程中，CPU tensor 不能传给 NPU kernel，否则会出现：

```text
ValueError: Expected a npu device, but got: cpu
```

因此它们不属于真实 Ascend 集成测试，未加入 Ascend QA 清单。真实 FlagGems/NPU 路径由新增 smoke test验证。

## 5. 运行方法

### 5.1 进入容器

```bash
docker exec -it TE-FL /bin/bash
```

### 5.2 加载环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate

cd /data/TE-FL-clone-lt/TransformerEngine-FL
```

### 5.3 运行 Ascend QA

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TE_PATH="$PWD"
export XML_LOG_DIR="$PWD/logs/L0_cppunittest-ascend"

bash qa/L0_cppunittest/test_ascend.sh
```

如果不设置 `ASCEND_RT_VISIBLE_DEVICES`，测试可以看到全部 16 张 NPU，但当前测试只选择第一个可见设备。

### 5.4 清理旧报告后重新运行

同名 JUnit 文件会被覆盖，但不再执行的旧测试报告不会被自动删除。为了保证报告目录只包含本次结果，可以执行：

```bash
rm -rf "$PWD/logs/L0_cppunittest-ascend"
export XML_LOG_DIR="$PWD/logs/L0_cppunittest-ascend"
bash qa/L0_cppunittest/test_ascend.sh
```

## 6. 实际验证效果

独立 `test_ascend.sh` 已在 `TE-FL` 容器内完整运行，结果如下：

| 测试文件 | 通过数量 |
| --- | ---: |
| `test_backend_ascend_smoke.py` | 2 |
| `test_backend_flagos.py` | 13 |
| `test_backend_reference.py` | 30 |
| `test_backend_reference_activation.py` | 9 |
| `test_backend_reference_dropout.py` | 4 |
| `test_backend_reference_gemm.py` | 7 |
| `test_plugin_manager.py` | 9 |
| `test_plugin_policy.py` | 12 |
| `test_policy.py` | 42 |
| 总计 | 128 |

JUnit 汇总结果：

```text
files=9
tests=128
failures=0
errors=0
skipped=0
```

终端最终输出：

```text
All Ascend plugin and NPU tests passed.
```

最后一次验证报告目录为：

```text
logs/L0_cppunittest-ascend-new
```

后续建议统一使用固定目录：

```text
logs/L0_cppunittest-ascend
```

## 7. JUnit 报告的用途

每个 pytest 文件生成一个 XML，例如：

```text
pytest_ascend_test_backend_ascend_smoke.xml
pytest_ascend_test_backend_flagos.xml
pytest_ascend_test_policy.xml
```

报告记录：

- 测试总数；
- 成功、失败、错误和跳过数量；
- 每个测试用例名称；
- 执行时间；
- 失败异常和 Python 调用栈。

Jenkins、GitLab CI、GitHub Actions 等系统可以采集：

```text
logs/L0_cppunittest-ascend/*.xml
```

并在 CI 页面显示测试结果。

## 8. 适配边界

本次适配验证的是 TransformerEngine-FL 当前已有的 Ascend Python 插件实现，不是 CUDA C++ 测试的逐算子等价移植。

如果需要让原始 `tests/cpp` 中的每个测试在 Ascend 上运行，需要额外完成：

1. 定义 Ascend 版 Transformer Engine C/C++ ABI；
2. 使用 ACL、AscendCL 或其他 NPU Runtime 替换 CUDA Runtime；
3. 使用 Ascend 数据类型替换 CUDA FP8/FP4 类型；
4. 将 `.cu` kernel 和测试迁移到 Ascend 编译工具链；
5. 替换 CUDA stream、memory 和 error API；
6. 新建 Ascend CMake 工程和原生动态库；
7. 为每个算子重新定义误差范围和硬件能力判断。

这属于完整的原生后端工程，不是 shell 脚本级修改。

## 9. 最终文件状态

原始 CUDA 测试入口保持不变：

```text
qa/L0_cppunittest/test.sh
```

本次新增：

```text
qa/L0_cppunittest/test_ascend.sh
transformer_engine/plugin/tests/test_backend_ascend_smoke.py
docs/ascend_l0_cppunittest_adaptation_zh.md
```

这样 CUDA 和 Ascend 测试入口相互独立：

```bash
# NVIDIA CUDA 环境
bash qa/L0_cppunittest/test.sh

# Huawei Ascend 环境
bash qa/L0_cppunittest/test_ascend.sh
```
