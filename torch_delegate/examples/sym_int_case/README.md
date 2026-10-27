# E2E测试

简体中文 | [English](README_en.md)

简化的端到端测试，用于验证fallback机制对非tensor输入的支持。

## 测试模型

### SimpleMixedModel（动态shape场景）

`SimpleMixedModel` 包含以下特性：
- **动态tensor输入** (x，动态batch)
- **静态tensor输入** (static_tensor)
- **静态非tensor输入** (scale, int)
- **动态非tensor输入** (batch_size，来自x.size(0))
- **Converter算子** (sigmoid, tanh, add)
- **Custom算子** (scale_add, fallback)

### SimpleStaticModel（静态shape场景）

`SimpleStaticModel` 包含以下特性：
- **静态tensor输入** (x，固定batch=4)
- **静态tensor输入** (static_tensor)
- **静态非tensor输入** (scale, int)
- **静态非tensor输入** (multiplier, int，非SymInt)
- **Converter算子** (sigmoid, tanh, add)
- **Custom算子** (scale_add, fallback)

所有输入shape固定，不使用 `dynamic_shapes`，验证纯静态场景下fallback算子的scalar输入处理路径。

## 使用步骤

### 检查环境
```bash
bash run_step1.sh
```
检查Python、GCC、CANN、torch、torch_npu、torch_delegate是否可用。

### 导出模型
```bash
bash run_step2.sh
```
- 导出 `model_simple.epair`
- 解压并检查JSON中的placeholders
- 验证动态/静态输入的正确性

### 构建C++ Runner
```bash
bash run_step3.sh
```
- 构建 `fallback_runner`
- 构建 `libcustom_ops.so`

### 执行测试
```bash
bash run_step4.sh
```
- 执行3组不同batch size的输入
- 生成输出文件到 `output_dir/`

### 精度对比
```bash
bash run_step5.sh
```
- 对比PyTorch参考输出和delegate输出
- 验证精度是否在容差范围内

## 预期结果

所有5个步骤都应该通过：
- Step 1: 环境检查通过
- Step 2: 模型导出成功，JSON中包含正确的placeholders
- Step 3: C++ runner编译成功
- Step 4: 所有3组输入执行成功
- Step 5: 精度对比通过

## 注意事项

- 确保已安装protobuf: `pip install protobuf`
- 确保已激活py311gcc13环境
- 确保已source CANN环境
