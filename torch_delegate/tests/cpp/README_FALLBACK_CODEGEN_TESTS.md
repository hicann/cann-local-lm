# Fallback Codegen Test Suite

本文档描述 `test_fallback_codegen.cc` 测试用例的设计和用途。

---

## 测试目的

验证 fallback codegen 模块的 JSON 解析和代码生成逻辑，包括：

1. 不同重载的自定义算子解析
2. `ascend_ir.source` 字段的过滤逻辑
3. Higher-Order Operators (HOPs) 的展开处理
4. 在线模式 vs 离线模式
5. 环境变量控制和错误处理

---

## Higher-Order Operators 处理

### auto_functionalized_v2

PyTorch 的 `auto_functionalized_v2` 是一个 Higher-Order Operator (HOP)，用于处理 inplace 类算子的函数化转换。真实算子名在第一个输入的 `as_operator` 字段中。

**JSON 结构**：
```json
{
  "target": "torch.ops.higher_order.auto_functionalized_v2",
  "inputs": [
    {
      "name": "",
      "arg": {
        "as_operator": "torch.ops.xllm_ascend.rope_in_place.default"
      },
      "kind": 1
    },
    {
      "name": "sin_cache",
      "arg": {"as_tensor": {"name": "sin_cache"}},
      "kind": 2
    }
  ]
}
```

**解析逻辑**：
1. 检测 `target == "torch.ops.higher_order.auto_functionalized"` 或 `auto_functionalized_v2`
2. 从 `inputs[0].arg.as_operator` 提取真实算子名
3. 对真实算子名应用标准解析逻辑（namespace、op_name、overload）

**测试用例**：`ParseAutoFunctionalizedV2Node`

---

## 测试场景

### 1. ParseMixedOverloadsFilterBySource

**目的**：测试解析包含不同重载算子的 JSON，并正确过滤需要 fallback 的节点。

**JSON 内容**：
- `add.Tensor`（source="converter"） → 不需要 fallback
- `add.Scalar`（无 source 字段） → 需要 fallback
- `mul.Tensor`（source=""） → 需要 fallback
- `relu.default`（source="converter"） → 不需要 fallback

**期望结果**：只解析出 `add.Scalar` 和 `mul.Tensor` 两个需要 fallback 的节点。

---

### 2. ParseAllConvertedNodesSkipped

**目的**：测试所有节点都已成功转换的情况，应返回空列表。

**JSON 内容**：所有节点都有 `source="converter"`。

**期望结果**：返回空列表（无节点需要 fallback）。

---

### 3. ParseNodesWithoutAscendIrField

**目的**：测试节点没有 `ascend_ir` 字段时的处理逻辑。

**注意**：根据代码实现逻辑，**没有 `ascend_ir` 字段的节点不会被处理**，因为：
- 没有 `ascend_ir` 字段的节点可能是 placeholder、output 等非计算节点
- 或者是其他不需要 fallback 的节点

**JSON 内容**：节点没有 `ascend_ir` 字段。

**期望结果**：返回空列表（无节点需要 fallback）。

---

### 4. ParseMixedSourcesField

**目的**：测试混合的 `source` 字段情况。

**JSON 内容**：
- `converted_node`（source="converter"） → 不需要 fallback
- `mock_node_no_field`（无 source 字段，但有 ascend_ir） → 需要 fallback
- `mock_node_empty_source`（source=""） → 需要 fallback
- `no_ascend_ir_node`（无 ascend_ir 字段） → **不会被处理**

**期望结果**：返回 2 个节点（`mock_node_no_field` 和 `mock_node_empty_source`）。

---

### 5. ParseAutoFunctionalizedV2Node

**目的**：测试 auto_functionalized_v2 HOP 的展开处理。

**JSON 内容**：
- target = `torch.ops.higher_order.auto_functionalized_v2`
- inputs[0].arg.as_operator = `torch.ops.xllm_ascend.rope_in_place.default`

**解析过程**：
1. 检测到 HOP target
2. 从 inputs[0].arg.as_operator 提取真实算子名
3. 应用标准解析：namespace=xllm_ascend, op_name=rope_in_place, overload=""

**期望结果**：
- 返回 1 个节点
- namespace = "xllm_ascend"
- op_name = "rope_in_place"
- overload = ""

---

### 6. GenerateFromModelJsonAllConvertedSkipped

**目的**：测试所有节点都已转换时，GenerateFromModelJson 应返回 skipped。

**期望结果**：
- `success = true`
- `skipped = true`
- `generated_op_names` 为空

---

### 7. GenerateFromModelJsonMissingEnvVars

**目的**：测试缺少必要环境变量时的错误处理。

**期望结果**：
- `success = false`
- `error_message` 包含 "TORCH_PKG_DIR" 和 "TORCH_NPU_DIR"

---

### 8. ParseTargetFormatVariations

**目的**：测试不同格式的 target 字段解析。

**target 格式**：
- `torch.ops.aten.topk.default` → namespace=aten, op_name=topk, overload=""
- `torch.ops.aten.add.Tensor` → namespace=aten, op_name=add, overload="Tensor"
- `torch.ops.custom_ns.my_op` → namespace="", op_name="custom_ns.my_op"
- `aten::matmul` → namespace="", op_name="aten::matmul"

**注意**：target 格式解析遵循 FX graph 格式：
- `torch.ops.<namespace>.<op_name>.<overload>` 为标准格式
- 其他格式可能无法正确解析

---

## 环境变量控制

### 必需的环境变量（仅当节点需要 fallback 时）

| 变量 | 说明 | 用途 |
|------|------|------|
| `TORCH_PKG_DIR` | Python torch 包目录 | 编译依赖 |
| `TORCH_NPU_DIR` | torch_npu 库目录 | 编译依赖 |

### 可选的环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TORCH_DELEGATE_OUTPUT_DIR` | 输出目录 | `./output_dir` |
| `TORCH_DELEGATE_KEEP_ARTIFACTS` | 在线模式下是否保留编译产物 | `0`（不保留） |

---

## 运行测试

### 方式 1：通过脚本单独编译运行

```bash
cd tests/cpp
./run_fallback_codegen_tests.sh
```

### 方式 2：通过 CMake 构建（需要完整的测试环境）

```bash
cd build
cmake .. -DBUILD_TESTS=ON -DPython3_EXECUTABLE=/path/to/python
make torch_delegate_tests
ctest -R FallbackCodegenTest
```

---

## 禁用的测试

以下测试标记为 `DISABLED_`，因为需要实际的环境设置：

- `DISABLED_GenerateFromModelJsonOnlineCompilation`：测试在线编译模式

要运行此测试，需要：
1. 设置 `TORCH_PKG_DIR` 和 `TORCH_NPU_DIR` 环境变量
2. 修改测试名称（去掉 `DISABLED_` 前缀）

---

## 测试结果示例

```
[==========] Running 11 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 11 tests from FallbackCodegenTest
[ RUN      ] FallbackCodegenTest.ParseMixedOverloadsFilterBySource
[       OK ] FallbackCodegenTest.ParseMixedOverloadsFilterBySource (0 ms)
[ RUN      ] FallbackCodegenTest.ParseAllConvertedNodesSkipped
[       OK ] FallbackCodegenTest.ParseAllConvertedNodesSkipped (0 ms)
[ RUN      ] FallbackCodegenTest.ParseNodesWithoutAscendIrField
[       OK ] FallbackCodegenTest.ParseNodesWithoutAscendIrField (0 ms)
[ RUN      ] FallbackCodegenTest.ParseMixedSourcesField
[       OK ] FallbackCodegenTest.ParseMixedSourcesField (0 ms)
[ RUN      ] FallbackCodegenTest.ParseAutoFunctionalizedV2Node
[       OK ] FallbackCodegenTest.ParseAutoFunctionalizedV2Node (0 ms)
[ RUN      ] FallbackCodegenTest.GenerateFromModelJsonAllConvertedSkipped
[       OK ] FallbackCodegenTest.GenerateFromModelJsonAllConvertedSkipped (0 ms)
[ RUN      ] FallbackCodegenTest.GenerateFromModelJsonMissingEnvVars
[       OK ] FallbackCodegenTest.GenerateFromModelJsonMissingEnvVars (0 ms)
[ RUN      ] FallbackCodegenTest.GenerateFromModelJsonInvalidJson
[       OK ] FallbackCodegenTest.GenerateFromModelJsonInvalidJson (0 ms)
[ RUN      ] FallbackCodegenTest.ParseModelJsonFromStringEmptyJson
[       OK ] FallbackCodegenTest.ParseModelJsonFromStringEmptyJson (0 ms)
[ RUN      ] FallbackCodegenTest.ParseTargetFormatVariations
[       OK ] FallbackCodegenTest.ParseTargetFormatVariations (0 ms)
[ RUN      ] FallbackCodegenTest.CodegenResultFieldsValidation
[       OK ] FallbackCodegenTest.CodegenResultFieldsValidation (0 ms)
[ DISABLED ] FallbackCodegenTest.DISABLED_GenerateFromModelJsonOnlineCompilation
[----------] 11 tests from FallbackCodegenTest (1 ms total)
[==========] 11 tests from 1 test suite ran. (1 ms total)
[  PASSED  ] 11 tests.
```
