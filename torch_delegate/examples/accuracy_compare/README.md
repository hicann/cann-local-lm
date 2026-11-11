# 精度对比示例

简体中文 | [English](README_en.md)

本示例演示TorchDelegate的端到端精度对比功能，验证两条执行路径的数值精度一致性：

- **Reference路径**：Python `torch.compile(backend='npu')` 使用GE backend
- **Delegate路径**：C++ `td::EpairModelLoader`（`CompileAndLoad` / `RunModelWithStreamAsync`）直接从epair文件执行

---

## 目录结构

```text
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

1. **torch 2.9.0+cpu**（推荐使用 torch_delegate venv）
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.2.0+**
5. **libtorch_npu** 源码可用

---

## 环境变量配置

### 必填环境变量

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `TORCH_NPU_DIR` | torch_npu源码目录 | `/path/to/pytorch/libtorch_npu` |

### 可选环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `ASCEND_HOME_PATH` | CANN环境 | 优先检查ASCEND_HOME_PATH |
| `TORCH_PKG_DIR` | Python torch包目录 | 自动检测 |
| `DEVICE` | 导出设备 | `cpu` |
| `COMPILE_DEVICE` | 编译设备 | `npu:0` |

---

## 快速开始

```bash
# 1. 加载 CANN 环境
source ~/Ascend/cann/set_env.sh

# 2. 激活包含 torch 2.9.0 的 Python 环境
source /path/to/torch_delegate/bin/activate

# 3. 设置唯一必填环境变量
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 4. 执行精度对比
cd /path/to/torch-delegate/examples/accuracy_compare
bash run.sh compare
```

---

## 详细步骤

### 导出ExportedProgram

```bash
bash run.sh step1
```

生成文件：
- `generate/sample.pt2` - ExportedProgram

### 编译AIR并导出Epair

```bash
source ~/Ascend/cann/set_env.sh
bash run.sh step2
```

生成文件：
- `generate/sample.rewritten.pt2` - 重写后的ExportedProgram
- `generate/rewritten.epair` - Epair文件

### compare精度对比

```bash
source ~/Ascend/cann/set_env.sh
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu

bash run.sh compare
```

输出结果：
- 控制台输出精度对比结果
- `generate/accuracy_result.json` - JSON格式结果

---

## 精度对比结果示例

```text
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

### Converter注册时机分离

- export epair时只注册BeamSearch converter（触发CANN预注册IR）
- Reference路径动态注册所有converter（BeamSearch + RopeInPlace）
- RopeInPlace在epair中source="" 触发fallback codegen

### torch.compile配置

Reference路径必须使用 `fullgraph=True` + `torchair.CompilerConfig()`：

```python
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend, fullgraph=True)
```

### BeamSearch输入Layout

参考 `xllm_ops/torch_compile_example/python/compare_beam_search.py`：
- `REQUEST_NUM=32, BEAM_WIDTH=4, VOCAB_SIZE=32000`
- `top_k = beam_width` (不是vocab_size)

---

## 故障排除

### 链接错误undefined reference to `PyTorchStreamReader::getRecord`

**原因**: torch版本不匹配。torch-delegate需要torch 2.9.0+cpu。

**解决**：使用 torch_delegate venv：
```bash
source /path/to/torch_delegate/bin/activate
```

### 错误 "环境变量XXX未设置"

**解决**：
1. 加载CANN环境：
```bash
source ~/Ascend/cann/set_env.sh
```

2. 设置唯一必填环境变量：
```bash
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu
```

### 错误 "TopK不支持DT_BF16"

**解决**：已在 `export_program.py` 中处理，转换为float32后执行TopK。

---

## 遗留问题与解决方案

### `_assert_tensor_metadata` 节点处理

**问题描述**：
`torch.export` 自动插入 `_assert_tensor_metadata` 元数据校验节点，该节点：
- 用于运行时校验tensor的device/dtype/shape是否与export时一致
- 不是计算节点，不产出任何tensor（`-> ()`）
- 但PyTorch原生op无法被 `torch.compile` + torchair backend识别
- 导致精度对比失败，因为torchair不会拒绝PyTorch内置op

**当前解决方案**：
1. **导出时移除**：在 `export_program.py` 中实现FX Graph Pass `_remove_assert_tensor_metadata()`，在 `torch.export.export` 后主动遍历graph，擦除所有 `_assert_tensor_metadata` 节点，从源头避免生成
2. **兜底converter**：`ge_converters.py` 中保留 `register_assert_tensor_metadata_converter()` 注册no-op converter，作为防御性措施（如果上游未移除该节点，torchair也不会拒绝）

**未来期望**：
将 `_remove_assert_tensor_metadata()` 推广为通用的 **FX Pass** 工具，处理类似的元数据校验节点：
- 支持配置化的op黑名单（不仅限于 `_assert_tensor_metadata`）
- 在export后统一清理不需要进入编译路径的节点
- 符合PyTorch最佳实践：使用native FX API而非依赖编译器后端

**注意**：
当前用例中"导出EP，再执行 `torch.compile`"的方式**不推荐**，因为：
- `torch.export` 的导出结果不保证能被 `torch.compile` 完全处理（PyTorch官方限制）
- 会导致"找不到converter"的问题，这是预期行为
- 推荐的工作流：`export -> rewrite -> compile_to_air -> epair_runner`

### beam_search const-correctness编译错误

**问题描述**：
CANN 9.1.0中安装的 `beam_search` 算子源码（TBE kernel）存在编译错误，错误信息：
```text
error: cannot bind non-const lvalue reference of type 'TopkTiling&' to an rvalue of type 'const TopkTiling'
```

**根因**：
`beam_search.cpp` 中 `Init()` 函数签名中 `TopkTiling&` 参数应为 `const TopkTiling&`，因为kernel入口处通过 `GET_TILING_DATA(tiling_data, tiling)` 获取的结构体成员是const性质的。

**当前解决方案**：
构建依赖时，`build_accuracy_deps.sh` 从xllm_ops仓库（固定 `cann-9.0` 分支）克隆源码，自动应用 `patches/beam_search_const_tiling.patch` 修复const-correctness问题，然后通过 `xllm_ops/build.sh` 构建并自动安装算子包到 `${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/`（build.sh内部逻辑line 453执行 `./xllm_ops.run`）。无需额外调用安装命令，也无需运行时再patch。

**未来期望**：
xllm_ops/beam_search上游修复后，可简化此逻辑：
- 更新patch指向新的commit hash或tag
- 检测已修复版本后跳过patch应用

---

## 参考资源

- TorchDelegate文档: `../../docs/zh/guide/build.md`
- export_and_load_example: `../export_and_load_example/`
- run_async example: `../run_async/`
