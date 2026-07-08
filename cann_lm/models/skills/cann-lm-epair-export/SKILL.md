---
name: cann-lm-epair-export
description: |
  CANN-LM 模型 epair 导出技能。当用户提到以下场景时使用：
  - 导出 CANN-LM 模型为 .pt2 或 .epair 格式
  - 编译/转换 cann_lm 命名空间的自定义算子（rope_in_place、fused_infer_attention_score、beam_search）
  - 注册 GE 转换器（ge_converter.py）
  - 使用 torchair 打桩库在无 NPU 设备时编译 epair
  - 修复 torchair 依赖问题（libhccl.so、torch_npu）
version: 1.0.0
metadata:
  author: assistant
  category: model-export
allowedTools:
  - read
  - write
  - bash
  - task
  - question
  - websearch
  - webfetch
---

# CANN-LM epair 导出技能

CANN-LM 模型导出为 epair 的完整流水线，包含自定义算子、GE 转换器注册和打桩编译。

## 流水线概览

```
export_program.py  →  sample.pt2      (step1: 导出 ExportedProgram)
ep_to_epair.py      →  sample.epair     (step2: 编译为 epair)
```

## 前置条件检查

```bash
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置>}"
python3 --version       # 需 >= 3.10
source .venv/bin/activate 2>/dev/null
python3 -c "import torch; import torchair; import torch_delegate; print('OK')"
```

## 环境准备

```bash
# CANN 环境
source /home/developer/Ascend/cann/set_env.sh

# 打桩库（无 NPU 设备时使用，提供 GE/hccl 等 stub .so）
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}

# Python 虚拟环境
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate
```

**关键依赖**：
- `torch`（2.9.0+cpu）
- `torchair`（从源码构建，含 `_torchair.so`）
- `torch_delegate`（从 torch-delegate-harmony 构建）
- `torch_npu` **不需要**（用打桩库替代）

## 文件结构

```
cann_lm/models/fake_ops/
├── __init__.py
├── ops.py               # 算子注册（cann_lm 命名空间）
├── export_program.py    # step1 导出
├── ep_to_epair.py       # step2 编译 epair
├── ge_converter.py      # GE 转换器注册
└── utils.py             # 工具函数
```

## 步骤

### Step 1: 导出 ExportedProgram

```bash
cd /mnt/workspace/project/CANN-LM_5883
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m cann_lm.models.fake_ops.export_program \
    --output ./generate/sample.pt2 --device cpu
```

**产出**：
- `generate/sample.pt2` — ExportedProgram 序列化文件
- `generate/sample.graph.py` — 计算图文本

**模型包含三个自定义算子**（定义在 `fake_ops/ops.py`，`cann_lm` 命名空间）：

| 算子 | 注册名 | 说明 |
|------|--------|------|
| `rope_in_place` | `torch.ops.cann_lm.rope_in_place` | inplace RoPE |
| `fused_infer_attention_score` | `torch.ops.cann_lm.fused_infer_attention_score` | 融合注意力 |
| `beam_search` | `torch.ops.cann_lm.beam_search` | 束搜索 |

### Step 2: 编译为 epair

```bash
cd /mnt/workspace/project/CANN-LM_5883
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m cann_lm.models.fake_ops.ep_to_epair
```

**参数**：
- `--ep-path`：输入 .pt2 路径（默认 `generate/sample.pt2`）
- `--epair-path`：输出 .epair 路径（默认 `generate/sample.epair`）
- `--rewrite-only`：只做重写，不编译

**流程**：
1. 加载 `.pt2`
2. 调用 `register_ge_converters()` 注册三个算子的 GE 转换器
3. 调用 `torch_delegate.export_to_epair()` 编译
4. 保存 `.epair`

**产出**：
- `generate/sample.epair` — epair 模型文件
- `generate/sample.rewritten.pt2` — 重写后的中间产物
- `generate/sample.rewritten.graph.py` — 重写后的计算图代码

## GE 转换器

定义在 `fake_ops/ge_converter.py`，注册三个转换器：

| 算子 | GE 映射 | 方式 |
|------|---------|------|
| `rope_in_place` | `custom_op("RopeInPlace")` | 自定义算子 |
| `fused_infer_attention_score` | `ge.FusedInferAttentionScore(...)` | GE 内置算子 |
| `beam_search` | `custom_op("BeamSearch")` | 自定义算子 |

**注意**：`fused_infer_attention_score` **不能**走 torchair 内置的 `npu_fused_infer_attention_score` 路径，因为后者的 GE 转换器注册在 `torch.ops.npu.npu_fused_infer_attention_score` 下，需要 `torch_npu`。本项目中直接注册到 `cann_lm` 命名空间并调用 `ge.FusedInferAttentionScore` 内置 GE 接口。

## 验证转换结果

```bash
cd /mnt/workspace/project/CANN-LM_5883
python3 -c "
import zipfile, json
with zipfile.ZipFile('generate/sample.epair') as z:
    m = json.loads(z.read('sample/extra/epair_manifest.json'))
    print(f'Converted: {m[\"converted_nodes\"]}/{m[\"total_nodes\"]}')
"
```

## 常见问题

### libhccl.so not found

torchair 的 `_torchair.so` 链接 `libhccl.so`，CANN 9.1.0 中该库从 `aarch64-linux/lib64/` 移除。解决方案：

```bash
# 从旧 CANN 创建软链接
ln -sf /home/developer/Ascend/cann-9.0.0-beta.2/aarch64-linux/lib64/libhccl.so \
       /home/developer/Ascend/cann-9.1.0/lib64/libhccl.so
```

或使用打桩库中的 `libhccl.so`（存在 `/mnt/workspace/torchair/build/stubs/`）。

### torch_npu not available

epair 编译**不需要** `torch_npu`。所有依赖通过打桩库（`torchair/build/stubs/`）提供。

### move_to_device_pass fails

`move_to_device_pass(ep, "npu")` 需要 `torch_npu` 注册 `npu` 设备类型。当前方案**跳过此步骤**，直接在 CPU 设备上调用 `export_to_epair`，打桩库会处理 GE graph 构建。

### 算子全部 source=converter 但 op 名为空

`ge.FusedInferAttentionScore` 返回的 GE 内置算子在 manifest 中 `op` 字段可能为空，但 `source=converter` 表示转换器已成功执行。这是正常的。
