# 权重格式转换设计文档

> epair 导出时把权重从 ND 行主序转为 NZ（FractNZ）块主序落盘，避免 NPU 运行期 transdata。
>
> 涵盖 `cann_lm/converter/export/weight_format/`（新包）+ 配置文件 `weight_format_rules.json` + epair_utils 集成 + C++ format 映射。
>
> 关联：导出主流程见 [export_design.md](./export_design.md)，量化见 [quant_design.md](../quant/quant_design.md)。
>
> 状态: Stage 1 已设计 | 最后核对: 2026-08-31

## 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [配置文件](#配置文件)
- [格式转换函数](#格式转换函数)
- [packer 集成](#packer-集成)
- [ep_json layout 标注](#ep_json-layout-标注)
- [C++ format 映射](#c-format-映射)
- [NZ shape 语义](#nz-shape-语义)
- [Stage 划分](#stage-划分)
- [待验证](#待验证)
- [后续迁移](#后续迁移)

---

## 概述

`cann_lm/converter/export/` 落盘权重时，默认按 ND（行主序）写 `weights_*.bin`。Ascend cube 计算原生以 NZ（FractNZ，16×16 块主序）布局工作，ND 落盘会在 OM 编译/运行期引入 transdata。本设计在落盘前把权重张量转成 NZ 物理字节，并同步标注 ep_json 的 `layout` 字段与 C++ 的 `ge::Format`，让权重 Const 节点直接以 `FORMAT_FRACTAL_NZ` 进入 GE 图。

### 能力边界

- **做（Stage 1）**：fp16/bf16 `MatMul` 类 Linear 权重（`*_proj`）→ NZ（标准 16×16 块转）
- **不做（Stage 2）**：量化 `QuantBatchMatmulV4` 的 int4 nibble-packed 权重（需先过 CANN 行为研究门，见 [Stage 划分](#stage-划分)）
- **保持 ND**：embed_tokens / lm_head / norm / bias / KV cache / 量化 weight_q + weight_scale

## 目录结构

```
cann_lm/converter/export/
├── weight_format/                 # ← 新包：格式转换函数 + 规则解析
│   ├── __init__.py                # re-export nd2nz / resolve_format / load_rules
│   ├── transforms.py              # nd2nz()（Stage 1）+ pack_int4_nz()（Stage 2 预留）
│   └── rules.py                   # 配置加载 + resolve_format(name, is_quantized)
├── weight_format_rules.json       # ← shipped 默认配置（export/ 根目录）
├── epair_utils.py                 # 集成：_pack_weights_to_dir 调 resolve_format + nd2nz
└── ...（既有文件）
```

> **预留**：`transforms.py` 当前仅含 `nd2nz`。Stage 2 的 `pack_int4_nz`（int4 nibble-aware NZ）也放此文件。`transdata_scale`（fp32→int64 位模式）语义上同属"格式转换"，现位于 `cann_lm/converter/export/quant_swap/materialize.py`（`swap.py` 在 `cann_lm/converter/swap.py`，绝对 import 它）；是否进一步合并到 `weight_format/transforms.py` 待定（`epair_utils` 当前从 `quant_swap.materialize` import）。

## 配置文件

shipped 默认 `cann_lm/converter/export/weight_format_rules.json`，converter 固定加载此默认配置（不暴露 CLI 覆盖）。该配置缺失时 → 全 ND（等价现状，不启用 NZ）。

### schema

```json
{
  "default": {"format": "ND"},
  "rules": [
    {
      "name_pattern": "*_proj.weight",
      "op_type": "MatMul",
      "format": "NZ", "block": 16, "pad": true, "pack": "nd2nz"
    },
    {
      "name_pattern": "*",
      "op_type": "QuantBatchMatmulV4",
      "format": "ND",
      "comment": "Stage 2 待研究门结论"
    }
  ]
}
```

### 字段语义

| 字段 | 含义 |
|------|------|
| `name_pattern` | 权重名 glob（`fnmatch` 语义），如 `*_proj.weight` |
| `op_type` | **分流键**：消费该权重的 GE 算子类型。`MatMul`（fp16 路径）/ `QuantBatchMatmulV4`（量化路径） |
| `format` | 目标格式：`ND` / `NZ` |
| `block` | NZ 块大小（默认 16） |
| `pad` | 非 block 倍数维是否补零 |
| `pack` | 变换函数名：`nd2nz`（标准 16×16）/ `int4_nz`（Stage 2 int4 nibble-aware） |

## 格式转换函数

`weight_format/transforms.py`：

### nd2nz（Stage 1）

```python
def nd2nz(tensor: torch.Tensor, block: int = 16, pad: bool = True) -> torch.Tensor:
    """[M, K] ND row-major → FractNZ 物理块 [ceil(M/b), ceil(K/b), b, b] block-major.
    非 b 倍数维内部补零（partial block）；返回同 dtype contiguous 供 tobytes()。"""
```

逻辑：`M, K = shape` → `m, k = ceil(M/b), ceil(K/b)` → `F.pad(..., (0, k*b-K, 0, m*b-M))` → `reshape(m, b, k, b)` → `permute(0, 2, 1, 3)` → `.contiguous()`。bf16 先 int16 view（沿用 `epair_utils` 既有约定）。

### pack_int4_nz（Stage 2 预留，当前未实现）

int8 nibble-packed int4 `[k//2, n]` → int4 元素级 16×16 NZ。**不能复用 `nd2nz`**（会打散 nibble 配对）。需先过 [Stage 2 研究门](#stage-划分) 拿到 CANN int4-NZ 字节确切布局再实现。

## packer 集成

`epair_utils._pack_weights_to_dir`（272-280 行）插入判定，仿 `transdata_scale` 先例：

```python
rule = resolve_format(name, is_quantized)   # 新增
raw = tensor.detach().cpu().contiguous()
if raw.dtype == torch.bfloat16:
    raw = raw.view(torch.int16)
elif "weight_scale" in name and raw.dtype == torch.float32:
    raw = transdata_scale(raw)
elif rule["format"] == "NZ" and raw.dim() == 2:           # ← 新增
    raw = nd2nz(raw, block=rule["block"], pad=rule["pad"])
    entry_format = "NZ"
else:
    entry_format = "ND"
data = raw.numpy().tobytes()
# index 条目加 "format": entry_format
```

### op_type 分流（is_quantized 探测）

单次导出要么纯 fp16、要么纯量化（量化时所有 Linear 换成 QuantLinear，无混合）。packer 探测 state_dict 是否含 `*_weight_scale` 张量 → 判 `is_quantized` → 选 `MatMul` 或 `QuantBatchMatmulV4` 规则集。**无需逐权重遍历图查消费算子**（避免给 packer 加图遍历负担）。prefill-decode 共享 pack 一次，NZ 只转一次（天然正确）。

## ep_json layout 标注

新增 `epair_utils.inline_weight_layout(ep_json_path, names_nz)`（仿 `inline_weight_scale_dtype`）：把命中张量的

- `tensor_values[name].layout`（pt2 layout）
- `ascend_ir.ops[].inputDesc/outputDesc[].layout`（字符串）

从 `"ND"` 改 `"NZ"`。`saver` 在 `consolidated_weights_save` 之后、写 ep_json 后调用（参照 `inline_weight_scale_dtype` 时机）。`weight_info.json` 条目也加 `"format"` 字段供下游辨识。

## C++ format 映射

权重 Const 节点的 format 完全由 ep_json `ascend_ir.desc.layout` 字符串驱动（`ge_graph_builder.cc:407 SetFormat(info.format)` ← `MapAscendIrDesc` ← `MapGeLayoutString`）。两处加 `"NZ"` case：

| 文件 | 函数 | 改动 |
|---|---|---|
| `aten_tensor_desc_mapper.cc:112` | `MapGeLayoutString` | `if (layout == "NZ") return ge::FORMAT_FRACTAL_NZ;`（权重 Const 主路径） |
| `ge_weight_loader.cc:115` | `Pt2LayoutFromGeString` | `if (layout == "NZ") return pt2::kLayoutStrided;`（inline const 兜底，pt2 无 NZ 概念，format 由 `MapGeLayoutString` 兜） |

`ge_graph_builder.cc:401 MakeTensorDesc` 已正确透传 `info.format`，无需改。`878/894/1116` 的 `FORMAT_ND` 硬编码是 Scalar/Sym/Data 节点（非权重），无关。

## NZ shape 语义

遵循 GE `FORMAT_FRACTAL_NZ` 标准契约：

- **ep_json shape = 逻辑 [M, K]**（非 padded）
- `nd2nz` 内部对非 16 倍数维补零（保证字节完整），写入物理 NZ 块
- GE kernel 读 partial block 的有效元素、忽略 pad 零值

**不写 padded shape**（`[m*16, k*16]`）——会让 MatMul 输出维度变 padded_M，破坏下游 shape。LLM Linear 维度实践上多 16 对齐（qwen3 hidden 1024/2048、intermediate 3072、head_dim 128×n_heads），pad 极少触发。

## Stage 划分

### Stage 1 — fp16 路径（本次交付，低风险）

1. `weight_format/transforms.py` 加 `nd2nz`
2. `weight_format/rules.py` 配置加载（`load_rules` 读 shipped 默认配置）+ `resolve_format`
3. `epair_utils._pack_weights_to_dir` 集成 + 新增 `inline_weight_layout`
4. C++ 两处 `"NZ"` case
5. 文档同步（本文档 + export_design.md / quant_design.md 注记 + INDEX.md）

### Stage 2 — 量化路径（独立后续，含强制研究门）

量化 weight_q 是 int8 nibble-packed int4 `[k//2, n]`，**标准 `nd2nz` 会打散 nibble 配对**，必须用专用 `int4_nz`。但 int4-NZ 字节布局由 CANN V4/V3 runtime kernel 决定，**不能凭推断实现**。先过研究门：

| 研究步骤 | 方法 | 判据 |
|---|---|---|
| ① 查 CANN 是否自动 ND→NZ | `atc --mode=1` 解析现有量化 OM 算子列表，看有无 `TransData`/`Transpose` 拦留 | 无残留→CANN 编译期自动转，量化 weight 保持 ND，Stage 2 仅文档化 |
| ② 若不自动转 | 读 GE 源码仓 `/mnt/workspace/gitCode/cann/ge/` 的 `quant_batch_matmul_v4` op_graph + 950-ops int4 NZ kernel | 拿到 int4-NZ 字节排列，据此实现 `pack_int4_nz`（含 weight_q/scale 协同分块） |

**关键事实**：当前 ND 量化 OM 已跑通（[quant_design.md](../quant/quant_design.md) "C++ 注册与产物"）。即 CANN 要么编译期自动转 NZ，要么 V4 runtime kernel `QuantBatchMatmulV4_ND_ND_int8_int4` 原生吃 ND。故量化预转 NZ 是**性能优化非正确性必需**，做错会打破能跑的 OM。

## 待验证

1. **MatMul NZ 输入接受性**：`reg_register.h` 的 MatMul 是否接受 weight Const `FORMAT_FRACTAL_NZ` 而不插运行期 transdata。需起最小 qwen3-0.6B NZ 导出 + OM 编译验证用例（参数化，放 `examples/`）。这是 Stage 1 性能收益能否兑现的关键。
2. **NZ shape 语义**：实现按 GE 标准（逻辑 shape + 内部 pad），OM 编译实测确认 GE kernel 正确读 partial block。

## 后续迁移

### 已完成（2026-08-31）：quant_swap 迁入 export/ + swap 升至 converter 根

`materialize.py` / `quant_linear.py` / `swap.py` 已从 `cann_lm/converter/quant/` 迁出：
- `materialize.py` / `quant_linear.py` → `cann_lm/converter/export/quant_swap/`（物化内部，与 export pipeline 同位）
- `swap.py` → `cann_lm/converter/swap.py`（converter 根，编排层 peer of `loader.py` / `saver.py`，绝对 import materialize/quant_linear）

旧 import 路径已干净切断：`swap_from_dopt` 走 `cann_lm.converter.swap`，物化符号走 `cann_lm.converter.export.quant_swap`。`quant/` 现仅留 dopt 编排（`auto_quant` / `dopt_wrapper`）+ vendored `dopt_lm` + 校准数据。详见 [quant_design.md](../quant/quant_design.md) "quant 包架构"。

### 待定：transdata_scale / pack_int4 合并到 weight_format/

`transdata_scale` / `pack_int4` 现位于 `export/quant_swap/materialize.py`，语义同属"格式转换"。是否进一步合并到 `weight_format/transforms.py`（与 `nd2nz` 同位）待定——`epair_utils` 当前从 `quant_swap.materialize` import `transdata_scale`，合并需同步改 import + 测试。迁移时同步更新本文档目录结构节与 [export_design.md](./export_design.md) 源文件索引。

## 文档关联

| 文档 | 关联 |
|------|------|
| [export_design.md](./export_design.md) | 导出主流程 / `epair_utils` / C++ `GeWeightLoader` / `inline_weight_scale_dtype` 先例 |
| [quant_design.md](../quant/quant_design.md) | 量化路径 / `transdata_scale` / `QuantBatchMatmulV4` / weight_scale 落盘 |
| [INDEX.md](./INDEX.md) | 导出模块索引（已同步补 weight_format 包条目） |
