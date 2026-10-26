# 精度对比示例

本示例演示 TorchDelegate 的端到端精度对比功能，验证两条执行路径的数值精度一致性：

- **Reference 路径**：Python `torch.compile(backend='npu')` 使用 GE backend
- **Delegate 路径**：C++ `td::DelegateSession + ge::GeSession::RunGraph()` 直接从 epair 文件执行

---

## 目录结构

```
accuracy_compare/
├── README.md                 # 本文档
├── run.sh                    # 运行脚本
├── __init__.py               # Python 包
├── accuracy_compare.py       # 精度对比主脚本
├── compile_to_air.py         # 导出 epair
├── export_program.py         # ExportedProgram 导出
├── ge_converters.py          # GE converter
├── backend_ops.py            # backend 算子
├── placeholder_to_custom_pass.py  # rewrite pass
├── ops.py                    # placeholder 算子
├── util.py                   # 工具函数
├── custom_op/                # 自定义算子编译
│   ├── custom_op.cpp
│   └── CMakeLists.txt
├── epair_runner/             # C++ runner
│   ├── CMakeLists.txt
│   ├── epair_runner.cpp
│   ├── tensor_io.cpp
│   └── tensor_io.h
└── generate/                 # 运行时生成目录
```

---

## 环境要求

1. **torch 2.9.0+cpu**（推荐使用 thj_delegate venv）
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.0.0+**
5. **libtorch_npu** 源码可用

---

## 环境变量配置

### 必填环境变量

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `TORCH_NPU_DIR` | torch_npu 源码目录 | `/path/to/pytorch/libtorch_npu` |

### 可选环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `ASCEND_HOME_PATH` 或 `CANN_ENV` | CANN 环境 | 优先检查 ASCEND_HOME_PATH |
| `TORCH_DELEGATE_REPO` | torch-delegate 仓库根目录 | 自动推导 |
| `TORCH_PKG_DIR` | Python torch 包目录 | 自动检测 |
| `DEVICE` | 导出设备 | `cpu` |
| `COMPILE_DEVICE` | 编译设备 | `npu:0` |

---

## 快速开始

```bash
# 1. 加载 CANN 环境
source ~/Ascend/cann/set_env.sh

# 2. 激活包含 torch 2.9.0 的 Python 环境
source /mnt/workspace/thj_delegate/bin/activate

# 3. 设置唯一必填环境变量
export TORCH_NPU_DIR=/mnt/workspace/gitCode/pytorch/libtorch_npu

# 4. 执行精度对比
cd /mnt/workspace/gitCode/torch-delegate/examples/accuracy_compare
bash run.sh compare
```

---

## 详细步骤

### step1：导出 ExportedProgram

```bash
bash run.sh step1
```

生成文件：
- `generate/sample.pt2` - ExportedProgram

### step2：编译 AIR 并导出 Epair

```bash
source ~/Ascend/cann/set_env.sh
bash run.sh step2
```

生成文件：
- `generate/sample.rewritten.pt2` - 重写后的 ExportedProgram
- `generate/rewritten.epair` - Epair 文件

### compare：精度对比

```bash
source ~/Ascend/cann/set_env.sh
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu

bash run.sh compare
```

输出结果：
- 控制台输出精度对比结果
- `generate/accuracy_result.json` - JSON 格式结果

---

## 精度对比结果示例

```
======================================================================
Accuracy Comparison Summary
======================================================================

[PASS] token_index
  Shape: [128, 1], Dtype: torch.int32
  Max Abs Diff: 0.000000

[PASS] token_ids
  Shape: [128, 1], Dtype: torch.int32
  Max Abs Diff: 0.000000

[PASS] log_probs
  Shape: [128, 1], Dtype: torch.bfloat16
  Max Abs Diff: 0.000000

======================================================================
Overall Result: PASS
======================================================================
```

---

## 技术要点

### Converter 注册时机分离

- export epair 时只注册 BeamSearch converter（触发 CANN 预注册 IR）
- Reference 路径动态注册所有 converter（BeamSearch + RopeInPlace）
- RopeInPlace 在 epair 中 source="" 触发 fallback codegen

### torch.compile 配置

Reference 路径必须使用 `fullgraph=True` + `torchair.CompilerConfig()`：

```python
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend, fullgraph=True)
```

### BeamSearch 输入 Layout

参考 `xllm_ops/torch_compile_example/python/compare_beam_search.py`：
- `REQUEST_NUM=32, BEAM_WIDTH=4, VOCAB_SIZE=32000`
- `top_k = beam_width` (不是 vocab_size)

---

## 故障排除

### 链接错误：undefined reference to `PyTorchStreamReader::getRecord`

**原因**: torch 版本不匹配。torch-delegate 需要 torch 2.9.0+cpu。

**解决**: 使用 thj_delegate venv：
```bash
source /mnt/workspace/thj_delegate/bin/activate
```

### 错误："环境变量 XXX 未设置"

**解决**：
1. 加载 CANN 环境：
```bash
source ~/Ascend/cann/set_env.sh
```

2. 设置唯一必填环境变量：
```bash
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu
```

### 错误："TopK 不支持 DT_BF16"

**解决**: 已在 `export_program.py` 中处理，转换为 float32 后执行 TopK。

---

## 遗留问题与解决方案

### 1. `_assert_tensor_metadata` 节点处理

**问题描述**：
`torch.export` 自动插入 `_assert_tensor_metadata` 元数据校验节点，该节点：
- 用于运行时校验 tensor 的 device/dtype/shape 是否与 export 时一致
- 不是计算节点，不产出任何 tensor（`-> ()`）
- 但 PyTorch 原生 op 无法被 `torch.compile` + torchair backend 识别
- 导致精度对比失败，因为 torchair 不会拒绝 PyTorch 内置 op

**当前解决方案**：
1. **导出时移除**：在 `export_program.py` 中实现 FX Graph Pass `_remove_assert_tensor_metadata()`，在 `torch.export.export` 后主动遍历 graph，擦除所有 `_assert_tensor_metadata` 节点，从源头避免生成
2. **兜底 converter**：`ge_converters.py` 中保留 `register_assert_tensor_metadata_converter()` 注册 no-op converter，作为防御性措施（如果上游未移除该节点，torchair 也不会拒绝）

**未来期望**：
将 `_remove_assert_tensor_metadata()` 推广为通用的 **FX Pass** 工具，处理类似的元数据校验节点：
- 支持配置化的 op 黑名单（不仅限于 `_assert_tensor_metadata`）
- 在 export 后统一清理不需要进入编译路径的节点
- 符合 PyTorch 最佳实践：使用 native FX API 而非依赖编译器后端

**注意**：
当前用例中"导出 EP，再执行 `torch.compile`"的方式**不推荐**，因为：
- `torch.export` 的导出结果不保证能被 `torch.compile` 完全处理（PyTorch 官方限制）
- 会导致"找不到 converter"的问题，这是预期行为
- 推荐的工作流：`export -> rewrite -> compile_to_air -> epair_runner`

### 2. beam_search const-correctness 编译错误

**问题描述**：
CANN 9.1.0 中安装的 `beam_search` 算子源码（TBE kernel）存在编译错误，错误信息：
```
error: cannot bind non-const lvalue reference of type 'TopkTiling&' to an rvalue of type 'const TopkTiling'
```

**根因**：
`beam_search.cpp` 中 `Init()` 函数签名中 `TopkTiling&` 参数应为 `const TopkTiling&`，因为 kernel 入口处通过 `GET_TILING_DATA(tiling_data, tiling)` 获取的结构体成员是 const 性质的。

**当前解决方案**：
构建依赖时，`build_accuracy_deps.sh` 从 xllm_ops 仓库（固定 `cann-9.0` 分支）克隆源码，自动应用 `patches/beam_search_const_tiling.patch` 修复 const-correctness 问题，然后通过 `xllm_ops/build.sh` 构建并自动安装算子包到 `${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/`（build.sh 内部逻辑 line 453 执行 `./xllm_ops.run`）。无需额外调用安装命令，也无需运行时再 patch。

**未来期望**：
xllm_ops/beam_search 上游修复后，可简化此逻辑：
- 更新 patch 指向新的 commit hash 或 tag
- 检测已修复版本后跳过 patch 应用

---

## 参考资源

- TorchDelegate 文档: `../../docs/build.md`
- export_and_load_example: `../export_and_load_example/`
- run_async example: `../run_async/`
