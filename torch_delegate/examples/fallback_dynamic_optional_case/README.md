# Fallback Dynamic/Optional测试

简体中文 | [English](README_en.md)

验证fallback codegen对 **动态输入 (Tensor[])** 和 **可选输入 (Tensor?)** 的正确支持。

## 测试场景

| 场景 | 算子 | schema | 说明 |
|------|------|--------|------|
| 动态输入 | `my_addn` | `(Tensor[] tensors) -> Tensor` | Tensor[] 对应GE DYNAMIC_INPUT，运行时实例数可变 |
| 可选输入不占位 | `my_addmm` | `(Tensor, Tensor, Tensor, Tensor?, *, Scalar beta=1) -> Tensor` | bias=None，OPTIONAL_INPUT不连接 |
| 可选输入占位 | `my_addmm` | 同上 | bias=有效值，OPTIONAL_INPUT连接Const |

### 关键codegen验证点

- **DYNAMIC_INPUT**: `GetDynamicInputTensor(ir_idx, i)` + `GetIrInputInstanceInfo` 获取实例数
- **OPTIONAL_INPUT (Tensor?)**：`GetOptionalInputTensor(ir_idx)` 返回nullptr时不崩溃
- **OPTIONAL_INPUT (Scalar with default)**：有默认值的Scalar参数声明为OPTIONAL_INPUT，EP JSON缺失时用默认值
- **IR索引对齐**: `GetRequiredInputTensor`/`GetOptionalInputTensor` 使用IR声明索引，不受DYNAMIC_INPUT扩展和OPTIONAL不连接影响

## 文件结构

```text
fallback_dynamic_optional_case/
├── custom_ops.py              # Python 自定义算子定义 (my_addn, my_addmm)
├── custom_ops_reg.cpp         # C++ 自定义算子注册 (CPU + PrivateUse1)
├── export_model.py            # 模型导出 + 生成测试数据
├── fallback_runner.cpp        # C++ Runner: 加载 epair 并执行推理
├── compare_accuracy.py        # 精度对比
├── CMakeLists.txt             # 构建配置
├── run.sh                     # 一键式验证脚本
└── README.md                  # 本文件
```

## 运行

```bash
# 前置：source CANN 环境
source <cann_path>/cann/set_env.sh

# 一键执行
bash run.sh
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/model_addn.epair` | 动态输入模型 |
| `output/model_addmm_no_bias.epair` | 可选输入不占位模型 |
| `output/model_addmm_with_bias.epair` | 可选输入占位模型 |
| `output/input_*.bin` | 测试输入数据 |
| `output/delegate_output_*.bin` | delegate推理输出 |
