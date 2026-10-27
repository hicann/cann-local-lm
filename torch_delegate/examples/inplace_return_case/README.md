# Inplace+Return测试用例

简体中文 | [English](README_en.md)

本示例测试fallback场景下的 **inplace输入+ Tensor返回值** 组合特性。

---

## 测试目标

### 核心场景inplace算子同时返回Tensor

```python
# inplace_scale_out: a *= scale (原地修改), 返回 a (修改后的张量)
my_ops.define("inplace_scale_out(Tensor(a!) a, float scale) -> Tensor")
```

这是 `dual_inplace_`（纯inplace，返回 `()`）和 `scale_add`（纯返回，无inplace）的组合：
- **inplace输入**：`a` 被标记为 `(a!)`，会被原地修改
- **Tensor返回值**：算子返回一个Tensor（而非 `()`）

验证要点：
- `auto_functionalized_v2` 正确解包inplace +返回值的组合
- REG_OP声明完整schema
- 模型同时使用返回值和原地修改后的输入
- GE图正确处理mutation输出和返回值输出

---

## 目录结构

```text
inplace_return_case/
├── custom_ops.py           # 算子定义 + CPU/NPU/Meta 实现
├── custom_ops_reg.cpp      # C++ 算子注册（CPU + PrivateUse1）
├── export_model.py         # 模型导出 + 生成测试数据
├── fallback_runner.cpp     # C++ runner（加载 epair + 执行）
├── compare_accuracy.py     # 精度对比脚本
├── CMakeLists.txt          # 构建配置
├── run.sh                  # 一键式运行脚本
├── README.md               # 本文档
└── output/                 # 运行时生成
    ├── model_inplace_return.epair
    └── input_*.bin / input_*.txt
```

---

## 环境要求

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.2.0+**
5. **libtorch_npu** 源码可用

---

## 快速开始

```bash
# 1. 加载 CANN 环境
source /path/to/cann/set_env.sh

# 2. 激活 Python 环境
source /path/to/venv_test/bin/activate

# 3. 执行测试
cd examples/inplace_return_case
bash run.sh
```

---

## 详细步骤

### 导出模型

```bash
python3 export_model.py
```

生成文件：
- `output/model_inplace_return.epair` - Epair文件
- `output/input_*.bin` / `output/input_*.txt` - 测试输入数据

### 构建C++ Runner

```bash
mkdir -p build && cd build
cmake .. -DCANN_ROOT=${ASCEND_HOME_PATH} -DTORCH_NPU_DIR=${TORCH_NPU_DIR}
cmake --build . -j$(nproc)
```

### 执行Runner

```bash
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR=./output_dir
./build/fallback_runner --custom-ops build/libcustom_ops.so output/model_inplace_return.epair
```

### 精度对比

```bash
python3 compare_accuracy.py
```

---

## 技术要点

### auto_functionalized_v2与inplace +返回值

PyTorch会将inplace算子包装为 `auto_functionalized_v2`。当算子同时有inplace输入和Tensor返回值时：

```json
{
    "target": "torch.ops.higher_order.auto_functionalized_v2",
    "inputs": [
        {"name": "", "arg": {"as_operator": "torch.ops.my_ops.inplace_scale_out.default"}},
        {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "scale", "arg": {"as_float": 2.0}}
    ]
}
```

`auto_functionalized_v2` 的输出包括：
1. **mutation输出**：被原地修改的tensor（`a!` 标记的参数）
2. **返回值输出**：算子返回的Tensor

fallback codegen需要正确区分这两类输出并生成对应的REG_OP。
