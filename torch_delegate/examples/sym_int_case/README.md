# E2E 测试

简化的端到端测试，用于验证 fallback 机制对非 tensor 输入的支持。

## 测试模型

### SimpleMixedModel（动态 shape 场景）

`SimpleMixedModel` 包含以下特性：
- **动态 tensor 输入** (x，动态 batch)
- **静态 tensor 输入** (static_tensor)
- **静态非 tensor 输入** (scale, int)
- **动态非 tensor 输入** (batch_size，来自 x.size(0))
- **Converter 算子** (sigmoid, tanh, add)
- **Custom 算子** (scale_add, fallback)

### SimpleStaticModel（静态 shape 场景）

`SimpleStaticModel` 包含以下特性：
- **静态 tensor 输入** (x，固定 batch=4)
- **静态 tensor 输入** (static_tensor)
- **静态非 tensor 输入** (scale, int)
- **静态非 tensor 输入** (multiplier, int，非 SymInt)
- **Converter 算子** (sigmoid, tanh, add)
- **Custom 算子** (scale_add, fallback)

所有输入 shape 固定，不使用 `dynamic_shapes`，验证纯静态场景下 fallback 算子的 scalar 输入处理路径。

## 使用步骤

### Step 1: 检查环境
```bash
bash run_step1.sh
```
检查 Python、GCC、CANN、torch、torch_npu、torch_delegate 是否可用。

### Step 2: 导出模型
```bash
bash run_step2.sh
```
- 导出 `model_simple.epair`
- 解压并检查 JSON 中的 placeholders
- 验证动态/静态输入的正确性

### Step 3: 构建 C++ Runner
```bash
bash run_step3.sh
```
- 构建 `fallback_runner`
- 构建 `libcustom_ops.so`

### Step 4: 执行测试
```bash
bash run_step4.sh
```
- 执行 3 组不同 batch size 的输入
- 生成输出文件到 `output_dir/`

### Step 5: 精度对比
```bash
bash run_step5.sh
```
- 对比 PyTorch 参考输出和 delegate 输出
- 验证精度是否在容差范围内

## 预期结果

所有 5 个步骤都应该通过：
- Step 1: 环境检查通过
- Step 2: 模型导出成功，JSON 中包含正确的 placeholders
- Step 3: C++ runner 编译成功
- Step 4: 所有 3 组输入执行成功
- Step 5: 精度对比通过

## 注意事项

- 确保已安装 protobuf: `pip install protobuf`
- 确保已激活 py311gcc13 环境
- 确保已 source CANN 环境
