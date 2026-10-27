# assert_tensor_metadata_demo

简体中文 | [English](README_en.md)

演示 `torch.export` 自动插入的 `aten._assert_tensor_metadata` 节点，以及如何在导出后将其移除，再导出epair并在C++侧加载执行。

## 背景

`torch.export` 在导出时会自动插入 `aten._assert_tensor_metadata` 运行时校验节点，用于在推理时检查tensor的dtype / device / layout是否与导出时一致。

该节点的schema为 `-> ()`，**没有计算语义**，但会被下游GE / torchair转换器拒绝，导致epair或 `torch.compile` 路径报错。

### 触发条件

`_assert_tensor_metadata` 由 `_insert_aten_to_metadata_assert_pass` 在导出过程中插入，位于每个 `aten.to.*` 节点之前。因此只要模型中使用了 `tensor.to()` / `tensor.float()` / `tensor.bfloat16()` 等类型转换操作，导出结果就会包含该节点。

> 参考: `torch/_export/utils.py::_insert_aten_to_metadata_assert_pass`

本示例**不使用任何自定义算子**（无需 `torch.library` / `xllm_placeholder`），也**不依赖fallback codegen**（无需torchair converter），仅靠标准 `aten.to.dtype` 即可触发。

## 模型

```python
class SimpleModel(nn.Module):
    def forward(self, x):
        y = torch.mm(x, self.weight)
        y = y.to(torch.float32)       # aten.to.dtype  -> _assert_tensor_metadata
        y = torch.relu(y)
        return y.to(torch.bfloat16)   # aten.to.dtype  -> _assert_tensor_metadata
```

导出后的graph（关键部分）:

```text
%mm = aten.mm.default(%x, %p_weight)
%_assert_tensor_metadata_default = aten._assert_tensor_metadata.default(%mm, dtype=float32, ...)
%to = aten.to.dtype(%mm, torch.float32)
%relu = aten.relu.default(%to)
%_assert_tensor_metadata_default_1 = aten._assert_tensor_metadata.default(%relu, dtype=bfloat16, ...)
%to_1 = aten.to.dtype(%relu, torch.bfloat16)
return (to_1,)
```

## 移除方法

```python
target = torch.ops.aten._assert_tensor_metadata.default
for node in list(ep.graph.nodes):
    if node.op == "call_function" and node.target is target:
        ep.graph.erase_node(node)
ep.graph.lint()
ep.graph_module.recompile()
```

与 `accuracy_compare/export_program.py::_remove_assert_tensor_metadata` 完全一致。

## 目录结构

```text
assert_tensor_metadata_demo/
├── README.md              # 本文档
├── export_demo.py         # Python 侧: 导出 epair + 保存输入/参考输出
├── main.cpp               # C++ 侧: EpairModelLoader 加载执行
├── CMakeLists.txt         # C++ 构建配置
└── run.sh                 # 一键验证脚本
```

## 环境要求

1. **torch 2.9.0+**
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.2.0+**

## 快速开始

```bash
# 1. 加载 CANN 环境
source /path/to/Ascend/cann/set_env.sh

# 2. 激活包含 torch 2.9.0 的 Python 环境
source /path/to/venv/bin/activate

# 3. 设置 TorchDelegate 环境（whl 安装后自动推导，无需 set_env.sh）
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# 4. 一键运行
cd examples/assert_tensor_metadata_demo
bash run.sh
```

`run.sh` 自动完成：
1. Python侧导出epair +保存输入二进制和参考输出
2. C++侧构建epair_loader
3. C++侧加载epair执行推理+精度比对

### 分步运行

#### 导出epair

```bash
python3 export_demo.py
```

生成文件：
- `generate/simple_model.pt2` - ExportedProgram
- `generate/simple_model.epair` - epair文件
- `generate/input_0.bin` - 示例输入
- `generate/ref_output.bin` - 参考输出
- `generate/graph_before.py` - 移除前的graph文本
- `generate/graph_after.py` - 移除后的graph文本

#### 构建C++示例

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build build -j$(nproc)
```

#### 运行C++加载器

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"
./build/epair_loader generate/simple_model.epair
```

## 流程

```python
torch.export.export(strict=False)
    │
    ▼
ExportedProgram  (含 _assert_tensor_metadata)
    │
    ├── print graph  (BEFORE)
    │
    ▼
remove_assert_tensor_metadata()
    │
    ├── print graph  (AFTER)
    │
    ▼
torch.export.save()  ──►  .pt2
    │
    ▼
torch_delegate.export_to_epair()  ──►  .epair
    │
    ▼
C++ EpairModelLoader
    │
    ├── CompileAndLoad(options, stream)
    ├── RunModelWithStreamAsync(stream, inputs, outputs)
    ├── aclrtSynchronizeStream(stream)
    └── 精度比对 (C++ 输出 vs Python 参考输出)
```

## 参考

- `accuracy_compare/export_program.py` — 原始 `_remove_assert_tensor_metadata` 实现
- `accuracy_compare/ge_converters.py` — 兜底converter（no-op）
- `torch/_export/utils.py::_insert_aten_to_metadata_assert_pass` — 上游插入逻辑
- `export_and_load_example/` — 基础epair加载示例
