# TransformerEngine-FL 全部 L0 QA 的 Ascend 适配与测试报告

## 1. 目标与原则

本次工作为 `qa` 目录下全部 `L0_*` 流程增加 Huawei Ascend 环境入口，同时保留所有原始 `test.sh` 不变。

适配原则如下：

1. 每个 L0 目录新增独立的 `test_ascend.sh`。
2. 原始 `test.sh` 继续服务 NVIDIA CUDA 或原有 CI，不修改其行为。
3. 能在当前 Ascend 软件栈真实执行的测试必须运行真实代码和真实 NPU。
4. 硬件无关的 lint、license 测试复用原始逻辑，只修正项目路径和入口。
5. 当前环境没有实现基础的 JAX Ascend 流程生成标准 JUnit `skipped` 报告，不伪造通过。
6. 测试中发现的真实产品问题在最小范围内修复，并增加回归验证。

## 2. 运行环境

### 2.1 服务器与容器

| 项目 | 配置 |
| --- | --- |
| Docker 容器 | `TE-FL` |
| 容器镜像 | `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.12` |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 项目路径 | `/data/TE-FL-clone-lt/TransformerEngine-FL` |
| 服务器分支 | `feature/fix-build` |
| 基础提交 | `3b5fbb58` |
| NPU | 16 x Huawei Ascend 910 |

### 2.2 Python 与计算栈

| 软件 | 版本或路径 |
| --- | --- |
| Python 虚拟环境 | `/data/TE-FL-clone-lt/te-fl-py310` |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cpu |
| Torch-NPU | 2.7.1.post2 |
| CANN | 9.0.0 |
| TransformerEngine-FL | 2.14.0+3b5fbb58 |
| FlagGems | 5.0.2 |
| Triton-Ascend | 3.2.0 |
| pytest | 8.2.1 |

容器默认 Python 3.12 没有官方 Triton-Ascend wheel，因此使用 Python 3.10 独立虚拟环境。当前环境没有安装 `jax` 和 `jaxlib`，也没有经过项目验证的 JAX Ascend runtime。

### 2.3 环境加载

```bash
docker exec -it TE-FL /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate
cd /data/TE-FL-clone-lt/TransformerEngine-FL

export ASCEND_RT_VISIBLE_DEVICES=0
```

不设置 `ASCEND_RT_VISIBLE_DEVICES` 时可以看到全部 16 张 NPU；当前单卡 L0 测试选择第一个可见设备。

## 3. 新增文件

### 3.1 各目录 Ascend 入口

```text
qa/L0_cppunittest/test_ascend.sh
qa/L0_jax_distributed_unittest/test_ascend.sh
qa/L0_jax_lint/test_ascend.sh
qa/L0_jax_unittest/test_ascend.sh
qa/L0_jax_wheel/test_ascend.sh
qa/L0_license/test_ascend.sh
qa/L0_pytorch_debug_unittest/test_ascend.sh
qa/L0_pytorch_lint/test_ascend.sh
qa/L0_pytorch_unittest/test_ascend.sh
qa/L0_pytorch_wheel/test_ascend.sh
```

### 3.2 公共入口与测试工具

```text
qa/run_l0_ascend.sh
qa/ascend_write_junit_skip.py
transformer_engine/plugin/tests/test_backend_ascend_smoke.py
```

### 3.3 产品代码修复

```text
transformer_engine/plugin/core/backends/vendor/npu/__init__.py
transformer_engine/jax/triton_extensions/utils.py
```

### 3.4 原始文件保持不变

所有以下原始入口均未修改：

```text
qa/L0_cppunittest/test.sh
qa/L0_jax_distributed_unittest/test.sh
qa/L0_jax_lint/test.sh
qa/L0_jax_unittest/test.sh
qa/L0_jax_wheel/test.sh
qa/L0_license/test.sh
qa/L0_pytorch_debug_unittest/test.sh
qa/L0_pytorch_lint/test.sh
qa/L0_pytorch_unittest/test.sh
qa/L0_pytorch_wheel/test.sh
```

## 4. 各 L0 目录的修改和结果

### 4.1 `L0_cppunittest`

原始流程编译 `tests/cpp` 下的 CUDA `.cu` 文件，依赖 CUDA、`nvcc`、cuDNN 和 CUDA 版 TE C ABI，不能在 Ascend 上原样执行。

新增入口：

```text
qa/L0_cppunittest/test_ascend.sh
```

Ascend 入口执行：

- 真实 FlagGems NPU GEMM；
- TE `Linear` NPU 前向和反向；
- FlagOS 后端注册；
- reference 后端；
- plugin manager；
- plugin policy 和通用 policy。

结果：

```text
9 JUnit files
128 tests passed
0 failures
0 errors
```

### 4.2 `L0_jax_distributed_unittest`

原始流程硬编码：

```text
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
--xla_gpu_deterministic_ops
```

并运行 JAX 多 GPU 和 collective GEMM。当前环境仅提供 Torch-NPU，没有 JAX Ascend runtime，因此不能真实执行。

新增入口生成一个 JUnit skipped 报告：

```text
logs/L0_jax_distributed_unittest-ascend/pytest_jax_distributed_ascend.xml
```

结果：

```text
1 skipped
原因：no supported JAX Ascend runtime
```

### 4.3 `L0_jax_lint`

lint 是硬件无关的静态检查，因此新入口直接设置正确的 `TE_PATH` 并运行原始 lint。

首次运行发现：

```text
transformer_engine/jax/triton_extensions/utils.py:317:
E1123 Unexpected keyword argument 'constexprs'
```

原因是不同 Triton 版本的 `ASTSource` 构造函数签名不同。修复方式为运行时检查参数是否存在：

```python
ast_source_kwargs = {
    "fn": kernel_fn,
    "signature": signature_with_constexpr,
}
if "constexprs" in inspect.signature(tc.ASTSource).parameters:
    ast_source_kwargs["constexprs"] = constants
src = tc.ASTSource(**ast_source_kwargs)
```

修复后结果：

```text
JAX lint exit: 0
Pylint score: 10.00/10
```

### 4.4 `L0_jax_unittest`

原始测试使用 CUDA/XLA GPU、PTXAS 和 GPU custom calls。当前环境没有 `jax`、`jaxlib` 或项目支持的 JAX-NPU extension。

新增入口生成：

```text
logs/L0_jax_unittest-ascend/pytest_jax_ascend.xml
```

结果：

```text
1 skipped
原因：JAX Ascend runtime is not installed
```

### 4.5 `L0_jax_wheel`

原始 wheel 包含 CUDA core wheel、JAX native wheel 和 CUDA metapackage。当前项目没有可以功能验证的 JAX Ascend native extension，因此没有生成一个无法运行的“假 JAX Ascend wheel”。

新增入口生成：

```text
logs/L0_jax_wheel-ascend/pytest_jax_wheel_ascend.xml
```

结果：

```text
1 skipped
原因：no validated JAX Ascend runtime or native extension
```

### 4.6 `L0_license`

license 检查与硬件无关，新入口设置正确的项目路径后运行原始 checker。

本次新增的所有 `.sh` 和 `.py` 文件均通过版权与 license 头检查，输出为 `OK`。

完整仓库检查结果：

```text
exit: 1
214 missing copyright/license messages
```

这些失败来自仓库已有文件，主要集中在既有 FlagOS plugin 源码和部分旧 QA 文件，不是本次 Ascend 适配新增。完整日志位于：

```text
logs/L0_ascend_static/license.log
```

该结果被标记为“仓库基线失败”，没有通过修改配置或排除目录伪造成功。

### 4.7 `L0_pytorch_debug_unittest`

原始 debug 测试大量使用 `.cuda()`、FP8 CUDA tensor 和 `transformer_engine_torch` native API，不能整体迁移到当前 NPU 插件层。

Ascend 入口运行可移植的 debug 配置解析测试：

```text
tests/pytorch/debug/test_config.py
```

它验证 `nvdlfw-inspect` 配置加载、Transformer Engine feature 映射和 tensor/GEMM 规则解析。

结果：

```text
1 passed
0 failures
```

报告：

```text
logs/L0_pytorch_debug_unittest-ascend/pytest_debug_config_ascend.xml
```

### 4.8 `L0_pytorch_lint`

PyTorch lint 是硬件无关检查。新入口只负责推导正确项目路径，然后运行原始 C++ 和 Python lint。

结果：

```text
exit: 0
Pylint score: 10.00/10
```

完整日志：

```text
logs/L0_ascend_static/pytorch_lint.log
```

### 4.9 `L0_pytorch_unittest`

原始 PyTorch 测试大量硬编码 `device="cuda"`、`.cuda()`、CUDA graph 和 FP8 CUDA native API。

Ascend 入口复用 `L0_cppunittest/test_ascend.sh` 的真实 NPU 与插件测试清单，避免维护两套重复设备逻辑。

结果：

```text
9 JUnit files
128 tests passed
0 failures
0 errors
```

报告目录：

```text
logs/L0_pytorch_unittest-ascend
```

### 4.10 `L0_pytorch_wheel`

新入口使用以下环境构建纯 Python Ascend wheel：

```bash
TE_FL_SKIP_CUDA=1
NVTE_FRAMEWORK=pytorch
```

构建后不会卸载当前开发环境，而是：

1. 在 `dist/ascend` 生成 wheel；
2. 安装到临时隔离目录；
3. 从 `/tmp` 启动 Python，避免源码目录遮蔽 wheel；
4. 验证 wheel 中的 TE 设备类型为 `npu`；
5. 在真实 NPU 上执行 `Linear` 前向和反向；
6. 退出时删除临时安装目录。

首次测试发现 wheel 中没有包含：

```text
transformer_engine/plugin/core/backends/vendor/npu/patches.py
```

原因是 `vendor/npu` 缺少 `__init__.py`，`setuptools.find_packages()` 没有识别该目录。新增：

```text
transformer_engine/plugin/core/backends/vendor/npu/__init__.py
```

修复后结果：

```text
WHEEL_ASCEND_OK
NPU backend patches applied
generic_gemm using default.flagos
Linear forward/backward passed
```

wheel 产物：

```text
dist/ascend/transformer_engine-2.14.0+3b5fbb58-py3-none-any.whl
```

## 5. 公共工具

### 5.1 JUnit skipped 报告生成器

新增：

```text
qa/ascend_write_junit_skip.py
```

它为明确不支持的 JAX 流程生成标准 JUnit：

```xml
<testsuite tests="1" failures="0" errors="0" skipped="1">
  <testcase name="ascend_runtime_support">
    <skipped message="..." />
  </testcase>
</testsuite>
```

CI 可以正确显示为 skipped，而不是把“没有执行”误认为 passed。

### 5.2 全部 L0 运行器

新增：

```text
qa/run_l0_ascend.sh
```

默认执行全部 10 个 L0 Ascend 入口：

```bash
bash qa/run_l0_ascend.sh
```

也可以只运行指定目录：

```bash
bash qa/run_l0_ascend.sh \
  L0_jax_unittest \
  L0_pytorch_debug_unittest
```

运行器会继续执行后续项目，并汇总：

```text
PASSED
SKIPPED
FAILED
```

验证示例：

```text
L0_jax_unittest: SKIPPED (no JAX Ascend runtime)
L0_pytorch_debug_unittest: PASSED
```

## 6. 推荐运行命令

### 6.1 运行全部 L0

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/TE-FL-clone-lt/te-fl-py310/bin/activate
cd /data/TE-FL-clone-lt/TransformerEngine-FL

export ASCEND_RT_VISIBLE_DEVICES=0
bash qa/run_l0_ascend.sh
```

当前 `L0_license` 存在仓库基线失败，因此运行全部任务时总退出码预计为非零。其他任务仍会继续执行，最后统一汇总。

### 6.2 单独运行某项

```bash
bash qa/L0_cppunittest/test_ascend.sh
bash qa/L0_jax_lint/test_ascend.sh
bash qa/L0_license/test_ascend.sh
bash qa/L0_pytorch_debug_unittest/test_ascend.sh
bash qa/L0_pytorch_lint/test_ascend.sh
bash qa/L0_pytorch_unittest/test_ascend.sh
bash qa/L0_pytorch_wheel/test_ascend.sh
```

## 7. 测试结果总表

| L0 目录 | 状态 | 实际结果 |
| --- | --- | --- |
| `L0_cppunittest` | PASSED | 128 passed |
| `L0_jax_distributed_unittest` | SKIPPED | 1 JUnit skipped，无 JAX Ascend runtime |
| `L0_jax_lint` | PASSED | Pylint 10.00/10 |
| `L0_jax_unittest` | SKIPPED | 1 JUnit skipped，无 JAX Ascend runtime |
| `L0_jax_wheel` | SKIPPED | 1 JUnit skipped，无 JAX Ascend native wheel |
| `L0_license` | BASELINE FAILED | 新文件全部 OK；仓库已有 214 条缺失头消息 |
| `L0_pytorch_debug_unittest` | PASSED | 1 passed |
| `L0_pytorch_lint` | PASSED | Pylint 10.00/10 |
| `L0_pytorch_unittest` | PASSED | 128 passed |
| `L0_pytorch_wheel` | PASSED | wheel 构建、隔离导入、NPU 前反向通过 |

## 8. 适配边界

本次适配没有实现以下不存在的底层能力：

- 将 CUDA `.cu` C++ 单测直接编译为 Ascend kernel；
- JAX Ascend runtime；
- JAX-NPU native custom call extension；
- CUDA FP8 debug native API 在 NPU 上的一一对应实现；
- CUDA graph、NVRTC、cuDNN 测试的 NPU 等价实现。

如果后续获得经过支持的 JAX Ascend runtime，可以将三项 JAX skipped 入口替换为真实测试。在此之前，保留明确 skipped 比安装 CPU JAX 后错误宣称“Ascend 测试通过”更可靠。

## 9. 与本次工作无关的服务器改动

服务器工作区原先已有：

```text
transformer_engine/pytorch/module/layernorm_mlp.py
```

的未提交修改。本次工作没有修改或还原该文件。测试生成的 `logs/`、`dist/ascend/` 和 `torch_compile_debug/` 属于运行产物，不应与源码适配混为一组提交。
