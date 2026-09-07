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
- [环境变量总开关](#环境变量总开关)
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
│   ├── layout.py              # nd2nz()（Stage 1）+ pack_int4_nz()（Stage 2 预留）
│   └── rules.py                   # 配置加载 + resolve_format(name, is_quantized)
├── weight_format_rules.json       # ← shipped 默认配置（export/ 根目录）
├── epair_utils.py                 # 集成：_pack_weights_to_dir 调 resolve_format + nd2nz
└── ...（既有文件）
```

> **预留**：`layout.py` 当前仅含 `nd2nz`。Stage 2 的 `pack_int4_nz`（int4 nibble-aware NZ）也放此文件。`transdata_scale`（fp32→int64 位模式）语义上同属"格式转换"，现位于 `cann_lm/converter/export/quant_swap/materialize.py`（`swap.py` 在 `cann_lm/converter/swap.py`，绝对 import 它）；属**物化位模式打包**（非布局变换），**不并入** `weight_format/layout.py`，留 `materialize.py` 与 `pack_int4` 同属物化层（`epair_utils` 从 `quant_swap.materialize` import）。

## 配置文件

shipped 默认 `cann_lm/converter/export/weight_format_rules.json`，converter 固定加载此默认配置（不暴露 CLI 覆盖）。该配置缺失时 → 全 ND（等价现状，不启用 NZ）。另有环境变量总开关 `CANN_LM_ENABLE_WEIGHT_TRANS` 默认关闭（见 [环境变量总开关](#环境变量总开关)）——即使配置文件命中 NZ，env 未开也不转换。

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

## 环境变量总开关

NZ 转换默认**关闭**（等价现状，全 ND，零字节变化）。导出时设环境变量 `CANN_LM_ENABLE_WEIGHT_TRANS=1` 才启用整个 `weight_format` 流程：`resolve_format` + `nd2nz`（写 NZ 字节）+ `weight_info.json` 条目的 `format` 字段（标 `"NZ"`）。未设或非 `1` → packer 跳过 `resolve_format`（含 `is_quantized` 探测），所有权重落 ND + `format` 全 `"ND"`，C++ 侧 `CreateConstNode` 走 `MapPt2Layout`（现状 ND 路径）——代码 dormant，零风险。

**设计意图**：NZ 是性能优化（避运行期 transdata），非正确性必需（当前 ND 量化 OM 已跑通，见 [Stage 划分](#stage-划分)）。Stage 1 的 MatMul NZ 输入接受性未验证（[待验证](#待验证)），故默认关闭、env 开启，便于代码先 dormant 合入，验证通过后再议默认开启或废弃开关。

**作用范围**：Python 导出侧（`epair_utils._pack_weights_to_dir` 写 NZ 字节 + `weight_info.json` 标 `format`）+ C++ 加载侧（`ge_graph_builder.cc:CreateConstNode` 读 `weight_info.json` 的 `format` 字段分流，见 [C++ format 映射](#c-format-映射)）。env 无 C++ 读取——env 关闭时 `weight_info.format` 全 `"ND"`，C++ 走 `MapPt2Layout`（ND）；env 开启时 `format` 为 `"NZ"`，C++ 走 `MapGeLayoutString("NZ")`（待实现）。

**命名约定**：`CANN_LM_ENABLE_WEIGHT_TRANS`（`1` 启用，其他/未设 = 关闭）。沿用 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` 的 `ENABLE` 布尔开关风格；`CANN_LM_` 前缀因仅作用于 cann_lm converter 导出侧。用 `TRANS`（非 `NZ`）因总开关 gate 整个 `weight_format` 流程——Stage 1 `nd2nz` + Stage 2 `pack_int4_nz` + 未来其他格式转换，非仅 NZ。

## 格式转换函数

`weight_format/layout.py`：

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

`epair_utils._pack_weights_to_dir`（:232 函数）循环体内插入判定，仿 `transdata_scale` 先例。**dtype view 与 NZ 转换是正交步骤**（bfloat16 先 `view(int16)` 不改字节、只改 dtype 解释；NZ 再重排物理块主序），不可塞进同一个 `elif` 链——否则 bfloat16 权重永远命中第一分支、NZ 分支不可达：

```python
import os
_TRANS_ENABLED = os.environ.get("CANN_LM_ENABLE_WEIGHT_TRANS") == "1"   # 默认关闭

# is_quantized 探测仅 env 开启时执行（见 op_type 分流）
for name in names:
    tensor = state_dict[name]
    raw = tensor.detach().cpu().contiguous()
    # ① dtype / 特殊位模式（互斥：bfloat16 view / weight_scale transdata）
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.int16)
    elif "weight_scale" in name and raw.dtype == torch.float32:
        raw = transdata_scale(raw)
    # ② NZ 转换（独立步骤，可与 ① 叠加；env 关闭 → 全 ND）
    entry_format = "ND"
    if _TRANS_ENABLED and raw.dim() == 2:
        rule = resolve_format(name, is_quantized)
        if rule["format"] == "NZ":
            raw = nd2nz(raw, block=rule["block"], pad=rule["pad"])
            entry_format = "NZ"
    data = raw.numpy().tobytes()
    # index 条目加 "format": entry_format
```

### op_type 分流（is_quantized 探测）

单次导出要么纯 fp16、要么纯量化（量化时所有 Linear 换成 QuantLinear，无混合）。packer 探测 state_dict 是否含 `*_weight_scale` 张量 → 判 `is_quantized` → 选 `MatMul` 或 `QuantBatchMatmulV4` 规则集。**无需逐权重遍历图查消费算子**（避免给 packer 加图遍历负担）。prefill-decode 共享 pack 一次，NZ 只转一次（天然正确）。

> env 关闭（`CANN_LM_ENABLE_WEIGHT_TRANS` 未设）时，`resolve_format` 不被调用，`is_quantized` 探测一并跳过——packer 退化为现状（纯 dtype/位模式处理 + 全 ND 落盘），零额外开销。

## ep_json layout 标注

> **机制修正（2026-09-05 探明）：原设计假设 NZ 经 `inline_weight_layout` 改 ep_json，此路不通**：
> - `tensor_values[p_*].layout` 是 **pt2 枚举**（7=kLayoutStrided=ND），pt2 无 NZ 枚举 → 写不进 NZ
> - 权重是 `tensor_values` 的 Data 节点（`p_*`），**不是 ascend_ir Const 算子**（ascend_ir 的 Const 全是小 inline 常量，最大 4 元素）→ 无 `ascend_ir.desc.layout` 可改
> - 消费算子 `inputDesc[i].layout`（字符串 "ND"）虽可改 "NZ"，但权重 Const 仍 ND → 反触发 TransData

**NZ 信号改走 `weight_info.json` 的 `"format":"NZ"` 字符串**（`_pack_weights_to_dir` 已落盘），由 C++ `CreateConstNode`/`GeWeightLoader` 读取 → `MapGeLayoutString`（见 [C++ format 映射](#c-format-映射)）。`inline_weight_layout` 不再实现。

## C++ format 映射

> **机制修正（2026-09-05 探明）：原设计写「权重 Const format 由 `ascend_ir.desc.layout` 驱动（`MapGeLayoutString`）」——错函数**。实探 `ge_graph_builder.cc`：

**权重 Const 路径 = `CreateConstNode`**（`ge_graph_builder.cc:473-509`，在 `:1213-1221` 给 Parameter/Buffer/TensorConstant placeholder 调用）：
- `:478` `format = mapper.MapPt2Layout(holder.format, ...)`——`holder` 来自 `GeWeightLoader::ReadTensorData`，`holder.format` 是 **pt2 枚举**（来自 ep_json `tensor_values[p_*].layout`=7）
- `:487` `ge::TensorDesc desc(ge::Shape(holder.dims), format, dtype)` → Const 节点（`--external_weight` 时 GE 编译期 `ConvertConstToFileConst`→FileConstant）

**`MapPt2Layout`**（`aten_tensor_desc_mapper.cc:67`）只 `kLayoutStrided→FORMAT_ND`，**无 NZ**；pt2 无 NZ 枚举 → 权重 Const 当前表达不了 NZ。

**`MapGeLayoutString`**（`aten_tensor_desc_mapper.cc:119`）只被 `ValidateConstOutputDesc`（`:517`，inline Const 校验）+ `RestoreInlineConstTensor`（inline Const 的 pt2-layout-string 转换）用——**不是权重 Const 路径**。

### NZ C++ 改动（待实现）

| 文件 | 函数 | 改动 | 调用方 |
|---|---|---|---|
| `aten_tensor_desc_mapper.cc:119` | `MapGeLayoutString` | 加 `if (layout == "NZ") return ge::FORMAT_FRACTAL_NZ;` | 须被下方权重 Const 路径调到 |
| `ge_weight_loader.cc`（`TensorRecordInfo`/`TensorHolder`）+ `ge_graph_builder.cc:478`（`CreateConstNode`） | 读 `weight_info.json` 的 `"format"` 字符串 + `CreateConstNode` 据此分流 | `holder`/`TensorRecordInfo` 携带 `format` 字符串；`CreateConstNode` 命中 `"NZ"` → `MapGeLayoutString("NZ")→FORMAT_FRACTAL_NZ`，否则 `MapPt2Layout(holder.format)`（现状 ND） | 权重 Const 主路径（`:1213-1221`） |

注：`ge_graph_builder.cc:401 MakeTensorDesc`（`:407 SetFormat(info.format) ← MapAscendIrDesc`）是 **ascend_ir op desc 路径**（inline Const / 普通算子 inputDesc/outputDesc），**非权重 Const**；权重 Const format 在 `CreateConstNode:487` 直接构造 `TensorDesc(format, dtype)`，不经 MakeTensorDesc。

## NZ shape 语义

遵循 GE `FORMAT_FRACTAL_NZ` 标准契约：

- **ep_json shape = 逻辑 [M, K]**（非 padded）
- `nd2nz` 内部对非 16 倍数维补零（保证字节完整），写入物理 NZ 块
- GE kernel 读 partial block 的有效元素、忽略 pad 零值

**不写 padded shape**（`[m*16, k*16]`）——会让 MatMul 输出维度变 padded_M，破坏下游 shape。LLM Linear 维度实践上多 16 对齐（qwen3 hidden 1024/2048、intermediate 3072、head_dim 128×n_heads），pad 极少触发。

## Stage 划分

### Stage 1 — fp16 路径（本次交付，低风险）

1. `weight_format/layout.py` 加 `nd2nz`
2. `weight_format/rules.py` 配置加载（`load_rules` 读 shipped 默认配置）+ `resolve_format`
3. `epair_utils._pack_weights_to_dir` 集成（env-gated，见 [环境变量总开关](#环境变量总开关)）——写 NZ 字节 + `weight_info.json` 标 `format`（`inline_weight_layout` 已弃用，见 [ep_json layout 标注](#ep-json-layout-标注)）
4. C++ `CreateConstNode`/`GeWeightLoader` 读 `weight_info.format` + `MapGeLayoutString` 加 NZ case（见 [C++ format 映射](#c-format-映射)）
5. 文档同步（本文档 + export_design.md / quant_design.md 注记 + INDEX.md）

### Stage 2 — 量化路径（独立后续，含强制研究门）

量化 weight_q 是 int8 nibble-packed int4 `[k//2, n]`，**标准 `nd2nz` 会打散 nibble 配对**，必须用专用 `int4_nz`。但 int4-NZ 字节布局由 CANN V4/V3 runtime kernel 决定，**不能凭推断实现**。先过研究门：

| 研究步骤 | 方法 | 判据 |
|---|---|---|
| ① 查 CANN 是否自动 ND→NZ | `atc --mode=1` 解析现有量化 OM 算子列表，看有无 `TransData`/`Transpose` 拦留 | 无残留→CANN 编译期自动转，量化 weight 保持 ND，Stage 2 仅文档化 |
| ② 若不自动转 | 读 GE 源码仓 `/mnt/workspace/gitCode/cann/ge/` 的 `quant_batch_matmul_v4` op_graph + 950-ops int4 NZ kernel | 拿到 int4-NZ 字节排列，据此实现 `pack_int4_nz`（含 weight_q/scale 协同分块） |

**关键事实**：当前 ND 量化 OM 已跑通（[quant_design.md](../quant/quant_design.md) "C++ 注册与产物"）。A3 实测 `QuantBatchMatmulV4` 有 `ND_ND` 与 `ND_NZ` 两个并列 kernel 变体——喂 ND 权重选 `ND_ND`（现状，内部处理 transpose），喂 NZ 权重选 `ND_NZ`（跳过内部 transpose，即 Stage 2 目标）；NZ kernel 存在 = int4-NZ 布局由 CANN 定义，研究门收窄为「逆向 `ND_NZ` kernel 期望的确切 int4-NZ 字节排列」。故量化预转 NZ 是**性能优化非正确性必需**，做错会打破能跑的 OM。

## 待验证

> **背景修正**:ND→NZ 转换有两种位置——① HOST 侧 `TransData` 算子(OM 里可见);② **kernel 内部**(OM 里无 TransData,但 MatMul kernel 每次执行做 ND→NZ)。无 host TransData ≠ 无开销——可能转进 kernel 内部(per-execution 性能下降)。CLI-015 的目标是避免 kernel 内部转换,前提**成立**。
>
> **CANN 官方 NZ 路径**:CANN 提供 `aclnnTransMatmulWeight`(device ND→NZ)+ `aclnnMatmulWeightNz`(NZ matmul),A3 支持。NZ 4D = `[n1, k1, k0, n0]`(k0=16, n0=16)。nn.Linear 用 view/strides 处理转置(无 Transpose 算子)。这是 **ACLNN runtime 路径**,非 GE 图/OM 编译路径。
>
> **torchair 无辜**:plain `aten.mm`/`aten.matmul` converter = `ge.MatMul(self, mat2, None)`——不插 Transpose。Transpose_1 是 **torch.export 分解 `F.linear`** 的产物(nn.Linear → weight.T + mm),非 torchair 主动插。bare-matmul(`x @ weight[K,N]`,无 nn.Linear)→ ep_json 只有 MatMul(无 Transpose_1)。

### L2 验证结果(bare-matmul, A3, 2026-09-07)

**✅ 通路跑通!NPU 输出 NZ OM == ND OM,max_diff=0.000000e+00。**

三件套(缺一不可):

| # | 组件 | 内容 | 作用 |
|---|------|------|------|
| ① | Python `nd2nz` | `[K,N]` → `reshape(k1,16,n1,16)` → **`permute(2,0,1,3)`** → `[n1,k1,k0,n0]` | CANN-correct NZ 字节 |
| ② | C++ `CreateConstNode` | **2D shape `holder.dims`** + format `FRACTAL_NZ`(**不手动 reshape 4D**) | GE 自己 `TransShapeToFracNz` 算出 4D storage(不撞 6D) |
| ③ | C++ input desc NZ 覆盖 | MatMul input[1] desc format → `FRACTAL_NZ`(引用 NZ 权重时) | Const NZ + MatMul input NZ 一致 → format refiner 不 revert |

OM 结构(env-on):
```
Const p_q_proj_weight  out_layout=FRACTAL_NZ  out_shape=[8,8,16,16]  ← GE 自己算的 4D(非 6D!)
MatMul MatMul          in_layout=['ND', 'FRACTAL_NZ']                 ← input[1]=NZ,没被 revert
total=4  TransData=0
```

### 关键教训(L2 过程中踩的坑)

1. **不要手动 reshape 成 4D**:设 4D storage `[8,8,16,16]` → GE `TransShapeToFracNz` 当 NCHW `[8,8,16,16]` 再转一次 → 6D `[8,8,1,1,16,16]`(H=16,W=16 → ceil/16=1)→ format_transfer/InferShape/format refiner 三机制全否决。**保持 2D origin + format FRACTAL_NZ** → GE 自己算出正确 4D storage(不撞 6D)。
2. **nd2nz permute 顺序**:必须匹配 CANN 文档 `[n1,k1,k0,n0]`。`permute(0,2,1,3)` 产出 `[n1,k1,n0,k0]`(内层反了)→ NPU max_diff=41(错)。`permute(0,2,3,1)` 产出 `[k1,n1,n0,k0]`(全反)→ max_diff=39(错)。**`permute(2,0,1,3)`** 产出 `[n1,k1,k0,n0]`(正确)→ max_diff=0。
3. **input desc NZ 覆盖必须同时设**:仅 Const NZ + MatMul input ND(不覆盖)→ format refiner 改回 ND(no-op)。Const NZ + MatMul input NZ(覆盖)→ 一致 → 保留 → 编译过。
4. **GE FRACTAL_NZ ≠ CANN API NZ**:GE 的 `FORMAT_FRACTAL_NZ` 是 NCHW-based(`TransShapeToFracNz` 对任意 shape 当 NCHW 处理);CANN API `aclnnMatmulWeightNz` 用 matmul 专用 4D `[n1,k1,k0,n0]`(直接用)。但**保持 2D origin + FRACTAL_NZ → GE 自己算出 4D(与 CANN API 一致)**——关键是不要手动设 4D。
5. **torchair 不插 Transpose**:Transpose_1 来自 torch.export 分解 `F.linear`(nn.Linear → weight.T + mm)。bare-matmul(`x @ weight[K,N]`)无 F.linear → 无 Transpose_1。

### L2 失败路径(L2 过程中排除的方案)

| 尝试 | 结果 | 根因 |
|------|------|------|
| 4D storage + FRACTAL_NZ(手动 reshape) | ❌ 编译失败 | GE 当 NCHW 再转 → 6D → IsShapeArgValid 4D-vs-6D |
| 2D + FRACTAL_NZ(无 input 覆盖) | ❌ no-op | Const NZ vs MatMul input ND → format refiner 改回 ND |
| C++ Transpose perm 覆盖(①) | ❌ 不可调和 | format_transfer 用 6D storage,InferShape 用 4D origin → perm 无法同时满足 |
| Lock(`ATTR_NAME_STORAGE_FORMAT`/`FORMAT_LOCKED`) | ❌ SDK 不可行 | `ge_attr_define.h`/`AttrUtils` 不在 CANN 0715 公开头 |
| `inline_weight_layout`(ep_json layout 改写) | ❌ 此路不通 | `tensor_values.layout` 是 pt2 枚举(无 NZ),权重不是 ascend_ir Const 算子 |

### L2 成功路径(bare-matmul)

**bare-matmul**(`x[K] @ weight[K,N]`,无 nn.Linear)→ ep_json 只有 `ge.MatMul`(无 Transpose_1)→ env-on 导出 → cann_lm_compile → OM 编译过(Const FRACTAL_NZ + MatMul input NZ + 0 TransData)→ NPU 输出 vs ND OM **max_diff=0**。

### L2 验证结果:nn.Linear 场景(Transpose_1, A3, 2026-09-07)

**✅ nn.Linear 也跑通!NPU max_diff=0.000000e+00。**

nn.Linear 的 `F.linear` 被 torch.export 分解为 `Transpose(weight) + mm`——Transpose 算子的 2D perm `[1,0]` 在 NZ 权重 Const 的 4D storage 上撞 `format_transfer_transpose`(4D-vs-2D)+ `TransposeCommonInferShape`(origin 2D,perm 值越界)。**perm 不可调和**(format_transfer 要 4D perm,InferShape 要 2D)。

解法:**跳过 Transpose + 设 MatMul `transpose_x2=True`**——GE MatMul 原生属性,内部转置,无 Transpose op。

四件套(nn.Linear 场景,缺一不可):

| # | 组件 | 内容 | 作用 |
|---|------|------|------|
| ① | Python `nd2nz` | `[K,N]` → reshape(k1,16,n1,16) → `permute(2,0,1,3)` → `[n1,k1,k0,n0]` | CANN-correct NZ 字节 |
| ② | C++ `CreateConstNode` | 2D shape `holder.dims` + `FRACTAL_NZ`(不 reshape) | GE 自己 `TransShapeToFracNz` 算 4D storage |
| ③ | C++ pre-pass + skip Transpose | Transpose on NZ weight → 跳过 + 注册 pass-through(`Transpose:0 → Const`) | 消除 Transpose 算子(避 2D-perm-on-4D) |
| ④ | C++ MatMul `transpose_x2=True` + input NZ 覆盖 | binding 到 Const + `transpose_x2=True` + input desc `FRACTAL_NZ` | GE 内部转置 + Const NZ + MatMul input NZ 一致 |

关键实现细节:
- `nz_transpose_passthrough` 必须放在 `GraphBuildContext`(跨节点共享)——Transpose 在 FX node[0],MatMul 在 node[1],局部变量不跨节点。
- MatMul 的 input 引用可能是 FX 别名(如 `"t"`),需先 `ResolveBoundaryInputSourceRef` 解析到 GE ref(如 `"Transpose:0"`)再查 passthrough。
- `transpose_x2=True` 通过 `ge::AttrValue::SetAttrValue(true)` 加到 attrs。

OM 结构(env-on):
```
Const p_q_proj_weight  out_layout=FRACTAL_NZ  out_shape=[8,8,16,16]
MatMul MatMul          in_layout=['ND', 'FRACTAL_NZ']  ← transpose_x2=True, 无 Transpose 算子
total=4  TransData=0
```

### L2 通路验证总结

| 场景 | OM 结构 | NPU max_diff | 状态 |
|------|---------|-------------|------|
| bare-matmul(`x @ weight[K,N]`,无 nn.Linear) | Const NZ + MatMul input NZ + 0 TransData | **0** | ✅ |
| nn.Linear(`F.linear`,有 Transpose_1) | Const NZ + MatMul input NZ + 0 TransData + transpose_x2=True | **0** | ✅ |
| bare-matmul 130×130(非 16 对齐) | — | — | ❌ GE 限制(见下) |

两个场景 **NPU 输出 NZ OM == ND OM,max_diff=0**。NZ 权重预转换在 A3 完整跑通。

### 非对齐 shape 限制(2026-09-07)

**16 对齐是 NZ 的硬约束**。非对齐 shape(如 130×130)在 GE 图编译路径(Const + ge.MatMul)不可行,根因是 GE 两套检查矛盾:

| GE 检查 | 期望 shape | 130×130 结果 |
|---------|-----------|-------------|
| `CheckTensorIsValid`(`graph_prepare.cc:2270`) | `data_size == Product(shape) * elem_size` | 2D `[130,130]`:33800 ≠ 41472(NZ padded)→ ❌ |
| `MatMul InferShape`(`matrix_calculation_matmul.cc:473`) | `dim[0] == input_k`(k-axis 匹配) | 4D `[9,9,16,16]`:k=9 ≠ 130(input k)→ ❌ |

- **2D `[130,130]` + FRACTAL_NZ**:CheckTensorIsValid 失败(NZ padded bytes 41472 ≠ logical 33800)。
- **4D `[9,9,16,16]` + FRACTAL_NZ**:InferShape 失败(GE 看到 dim[0]=9 作为 k-axis,不认 NZ 块语义)。
- **padded 2D `[144,144]`**:InferShape 失败(k=144 ≠ input k=130;input 是 runtime Data 不能 pad)。
- **128×128 能过**:因为 16 对齐 → logical size = NZ padded size(32768=32788),两个检查都满足。

**根因**:GE 的 `CheckTensorIsValid` 不理解 NZ padding(检查 `data_size == Product(logical_shape) * elem_size`,不认 NZ 块 padding)。CANN 官方 `aclnnMatmulWeightNz` 走 ACLNN runtime(4D NZ 直传,不走 GE `CheckTensorIsValid`),所以官方路径能处理非对齐;GE 图编译路径不行。

**当前实现**:16 对齐 shape 用 NZ(已验证);非对齐自动回落 ND(需在 `_pack_weights_to_dir` 加 shape 对齐检查)。LLM 实际权重维度多 16 对齐(qwen3 hidden 1024/2048、intermediate 3072、head_dim 128×n_heads),非对齐极少。

### L3 整网(A3)

qwen3-0.6B `--mode single` + `CANN_LM_ENABLE_WEIGHT_TRANS=1` 导出 → OM 编译 → `atc --mode=1` 查 `*_proj` 对应 MatMul 无 TransData → NPU 出字 vs ND 版一致。

### L4 Kirin(延后)

CANN-LM Kirin 算子支持落地后,复 L2/L3 于 kirin9030/kirinx90(hdc 真机 + `kirin_cann_test` 流程)。Kirin9030 有 ND↔NZ TransData(NZ 概念成立),matmul NZ kernel + GE 行为待验。

### 核心判据(修正)

NZ 重排成功 ⟺ **NPU 输出 NZ OM == ND OM**(max_diff=0)+ OM 里权重 Const `layout=FRACTAL_NZ` + 0 host TransData。TransData 计数不是唯一指标(kernel 内部转换无 TransData 但有开销)。

## 后续迁移

### 已完成（2026-08-31）：quant_swap 迁入 export/ + swap 升至 converter 根

`materialize.py` / `quant_linear.py` / `swap.py` 已从 `cann_lm/converter/quant/` 迁出：
- `materialize.py` / `quant_linear.py` → `cann_lm/converter/export/quant_swap/`（物化内部，与 export pipeline 同位）
- `swap.py` → `cann_lm/converter/swap.py`（converter 根，编排层 peer of `loader.py` / `saver.py`，绝对 import materialize/quant_linear）

旧 import 路径已干净切断：`swap_from_dopt` 走 `cann_lm.converter.swap`，物化符号走 `cann_lm.converter.export.quant_swap`。`quant/` 现仅留 dopt 编排（`auto_quant` / `dopt_wrapper`）+ vendored `dopt_lm` + 校准数据。详见 [quant_design.md](../quant/quant_design.md) "quant 包架构"。

### 已决定（2026-09-05）：transdata_scale / pack_int4 不并入 weight_format/

`transdata_scale` / `pack_int4` 现位于 `export/quant_swap/materialize.py`，语义同属"格式转换"。**不并入** `weight_format/layout.py`——二者是**物化位模式打包**（`transdata_scale`：fp32→int64 位模式；`pack_int4`：int4→nibble-packed int8），非布局变换，与 `nd2nz` / `pack_int4_nz` 的**布局重排**职责正交，留 `export/quant_swap/materialize.py`。`epair_utils` 维持从 `quant_swap.materialize` import `transdata_scale`，无需改 import / 测试。

## 文档关联

| 文档 | 关联 |
|------|------|
| [export_design.md](./export_design.md) | 导出主流程 / `epair_utils` / C++ `GeWeightLoader` / `inline_weight_scale_dtype` 先例 |
| [quant_design.md](../quant/quant_design.md) | 量化路径 / `transdata_scale` / `QuantBatchMatmulV4` / weight_scale 落盘 |
| [INDEX.md](./INDEX.md) | 导出模块索引（已同步补 weight_format 包条目） |
