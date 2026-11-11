# Fallback Inplace/Optional测试用例

简体中文 | [English](README_en.md)

本示例测试fallback场景下的两个关键特性：

1. **多Inplace参数**：单个算子定义中有多个inplace参数
2. **多可选参数**：算子定义中有多个可选参数，调用时只传入部分

---

## 测试目标

### 多Inplace参数

```python
# dual_inplace_: a *= scale_a; b *= scale_b
my_ops.define("dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()")
```

验证要点：
- x和y都被原地修改
- REG_OP声明完整schema（a, b, scale_a, scale_b）
- auto_functionalized_v2正确解包

### 多可选参数

```python
# multi_optional_op: result = x * alpha + beta + gamma + delta
my_ops.define(
    "multi_optional_op(Tensor x, float alpha=1.0, float beta=0.0, float gamma=0.0, float delta=0.0) -> Tensor"
)
```

验证要点：
- 调用 `multi_optional_op(x, gamma=3.0)` 时，EP JSON只有x和gamma
- REG_OP声明全量5个参数（x, alpha, beta, gamma, delta）
- Execute从GE input读全量参数，缺失的alpha/beta/delta由GE提供默认值

- 调用 `multi_optional_op(x, alpha=2.0, gamma=3.0)` 时，EP JSON有x, alpha, gamma
- REG_OP声明全量5个参数
- Execute从GE input读全量参数

---

## 目录结构

```text
fallback_inplace_optional_case/
├── custom_ops.py           # 算子定义 + CPU/NPU/Meta 实现
├── custom_ops_reg.cpp      # C++ 算子注册（CPU + PrivateUse1）
├── export_model.py         # 模型导出 + 生成测试数据
├── fallback_runner.cpp     # C++ runner（加载 epair + 执行）
├── compare_accuracy.py     # 精度对比脚本
├── CMakeLists.txt          # 构建配置
├── run.sh                  # 一键式运行脚本
├── README.md               # 本文档
└── output/                 # 运行时生成
    ├── model_inplace_optional.epair
    └── input_*.bin / input_*.txt
```

---

## 环境要求

1. **torch 2.9.0+cpu**（推荐使用py311gcc13 venv）
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.2.0+**
5. **libtorch_npu** 源码可用

---

## 快速开始

```bash
# 1. 加载 CANN 环境
source ~/Ascend/cann/set_env.sh

# 2. 激活 Python 环境
conda activate py311gcc13

# 3. 设置 TORCH_NPU_DIR
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 4. 执行测试
cd /path/to/torch-delegate/examples/fallback_inplace_optional_case
bash run.sh
```

---

## 详细步骤

### 导出模型

```bash
python3 export_model.py
```

生成文件：
- `output/model_inplace_optional.epair` - Epair文件
- `output/input_*.bin` / `output/input_*.txt` - 测试输入数据

### 构建C++ Runner

```bash
mkdir -p build && cd build
cmake .. -DCANN_ROOT=${ASCEND_HOME_PATH} -DTORCH_NPU_DIR=${TORCH_NPU_DIR}
cmake --build . -j$(nproc)
```

生成文件：
- `build/fallback_runner` - C++执行器
- `build/libcustom_ops.so` - 自定义算子库

### 执行Runner

```bash
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR=./output_dir
./build/fallback_runner --custom-ops build/libcustom_ops.so output/model_inplace_optional.epair
```

### 精度对比

```bash
python3 compare_accuracy.py
```

---

## 预期输出

```text
========================================
[1/6] 检查环境
========================================
  [✓] ASCEND_HOME_PATH=/path/to/cann
  [✓] python3 3.11.0

========================================
[2/6] 准备 libtorch_npu
========================================
  [✓] TORCH_NPU_DIR=/path/to/libtorch_npu

========================================
[3/6] 导出模型
========================================
  导出 model_inplace_optional
  Fallback 算子 (3):
    - torch.ops.my_ops.dual_inplace_.default
    - torch.ops.my_ops.multi_optional_op.default (x2)

========================================
[4/6] 构建 C++ Runner + libcustom_ops
========================================
  [✓] fallback_runner 已编译
  [✓] libcustom_ops.so 已编译

========================================
[5/6] 执行 Runner
========================================
  [✓] custom_op.cpp (15000 bytes)
  [✓] libcust_opapi.so (80000 bytes)

========================================
[6/6] 精度对比
========================================
输入组 0 (batch_size=4, alpha=2.0, gamma=3.0):
    PASS (max_abs=1.23e-06, max_rel=2.34e-06)
输入组 1 (batch_size=8, alpha=1.5, gamma=2.5):
    PASS (max_abs=2.34e-06, max_rel=3.45e-06)
输入组 2 (batch_size=2, alpha=3.0, gamma=1.0):
    PASS (max_abs=1.11e-06, max_rel=2.22e-06)

========================================
测试总结
========================================
✓ 所有测试通过!
```

---

## 技术要点

### Inplace标记

必须使用 `(a!)` 和 `(b!)` 标记，否则PyTorch不会识别为inplace：

```python
# 正确
my_ops.define("dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()")

# 错误（不会被识别为 inplace）
my_ops.define("dual_inplace_(Tensor a, Tensor b, float scale_a, float scale_b) -> ()")
```

### auto_functionalized_v2

PyTorch会将inplace算子包装为 `auto_functionalized_v2`：

```json
{
    "target": "torch.ops.higher_order.auto_functionalized_v2",
    "inputs": [
        {"name": "", "arg": {"as_operator": "torch.ops.my_ops.dual_inplace_.default"}},
        {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "b", "arg": {"as_tensor": {"name": "y"}}},
        {"name": "scale_a", "arg": {"as_float": 0.5}},
        {"name": "scale_b", "arg": {"as_float": 2.0}}
    ]
}
```

fallback codegen需要正确解包并生成REG_OP。

### 可选参数默认值

当EP JSON中缺少某个参数时，fallback codegen会：
1. REG_OP仍然声明该参数（全量schema）
2. Execute从GE input读取该参数
3. GE Graph Builder为该参数创建Const节点（使用默认值）

---

## 故障排除

### 错误 "dual_inplace_ 未被识别为inplace"

**原因**： 算子定义中缺少 `(a!)` 或 `(b!)` 标记。

**解决**： 检查 `custom_ops.py` 中的算子定义，确保使用了正确的alias标记。

### 错误 "multi_optional_op参数数量不匹配"

**原因**： REG_OP声明的参数数量与GE图连接的输入数量不一致。

**解决**： 检查 `output_dir/*/custom_op.cpp` 中的REG_OP定义，确保声明了全量参数。

### 错误 "精度对比失败"

**原因**： inplace操作未正确执行，或可选参数默认值未正确应用。

**解决**： 检查 `output_dir/*/custom_op.cpp` 中的Execute方法，确保正确读取了所有参数。

---

## 参考资源

- sym_int_case: `../sym_int_case/`
- accuracy_compare: `../accuracy_compare/`
- TorchDelegate文档: `../../docs/zh/guide/build.md`
