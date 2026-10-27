# 权重格式转换设计文档

简体中文 | [English](../../../en/cann_lm_converter/export/weight_format_design.md)

> epair 导出时把权重从 ND 行主序转为分形块主序落盘（fp16 MatMul → FractNZ；量化 QuantBatchMatmulV4 weight_q → FractZ），避免 NPU 运行期/内核内 transdata。
>
> 涵盖 `cann_lm/converter/export/weight_format/`（新包）+ 配置文件 `weight_format_rules.json` + epair_utils 集成 + C++ format 映射。
>
> 关联：导出主流程见 [export_design.md](./export_design.md)，量化见 [quant_design.md](../quant/quant_design.md)。
>
> 状态: Stage 1/2 已落地并验证（A3 NZ max_diff=0；KirinX90 Z 板端 cos=0.999912）；总开关废弃设计定稿（规则级 enable：MatMul 关 / QuantBatchMatmulV4 开）| 最后核对: 2026-10-20

<a id="目录结构"></a>
## 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [配置文件](#配置文件)
- [环境变量总开关](#环境变量总开关)
- [总开关废弃设计（2026-10-20 定稿）](#总开关废弃设计2026-10-20-定稿)
- [格式转换函数](#格式转换函数)
- [packer 集成](#packer-集成)
- [ep_json layout 标注](#ep_json-layout-标注)
- [ep_json 标注同步机制（_sync_* 补录）](#ep_json-标注同步机制_sync_-补录)
- [C++ format 映射](#c-format-映射)
- [NZ shape 语义](#nz-shape-语义)
- [Stage 划分](#stage-划分)
- [待验证](#待验证)
- [后续迁移](#后续迁移)

---

<a id="概述"></a>
## 概述

`cann_lm/converter/export/` 落盘权重时，默认按 ND（行主序）写 `weights_*.bin`。Ascend cube 计算原生以 NZ（FractNZ，16×16 块主序）布局工作，ND 落盘会在 OM 编译/运行期引入 transdata。本设计在落盘前把权重张量转成 NZ 物理字节，并同步标注 ep_json 的 `layout` 字段与 C++ 的 `ge::Format`，让权重 Const 节点直接以 `FORMAT_FRACTAL_NZ` 进入 GE 图。

### 能力边界

- **做（Stage 1，已实现）**：fp16/bf16 `MatMul` 类 Linear 权重（`*_proj`）→ NZ（标准 16×16 块转）；A3 实测 max_diff=0（见 [待验证](#待验证) L2/L3）
- **做（Stage 2，已实现）**：量化 `QuantBatchMatmulV4` 的 weight_q → Z（`pack_int4_nz`，int4 nibble-aware FractZ `[K//64, N//16, 16, 8]`）；KirinX90 板端实测 cos=0.999912（``op_npu_verify/GUIDE_kirinx90_board.md``）
- **保持 ND**：embed_tokens / lm_head / norm / bias / KV cache / 量化 weight_scale（int64 位模式，非布局对象，见 [ep_json 标注同步机制](#ep_json-标注同步机制_sync_-补录)）

## 目录结构

```
cann_lm/converter/export/
├── weight_format/                 # ← 新包：格式转换函数 + 规则解析
│   ├── __init__.py                # re-export nd2nz / pack_int4_nz / resolve_format / load_rules
│   ├── layout.py              # nd2nz()（Stage 1, FractNZ）+ pack_int4_nz()（Stage 2, FractZ,已实现）
│   └── rules.py                   # 配置加载 + resolve_format(name, is_quantized)
├── weight_format_rules.json       # ← shipped 默认配置（export/ 根目录）
├── epair_utils.py                 # 集成：_pack_weights_to_dir 调 resolve_format + nd2nz
└── ...（既有文件）
```

> **职责边界**：`layout.py` 含 `nd2nz`（Stage 1）与 `pack_int4_nz`（Stage 2，已实现）。`transdata_scale`（fp32→int64 位模式）语义上同属"格式转换"，现位于 `cann_lm/converter/export/quant_swap/materialize.py`（`swap.py` 在 `cann_lm/converter/swap.py`，绝对 import 它）；属**物化位模式打包**（非布局变换），**不并入** `weight_format/layout.py`，留 `materialize.py` 与 `pack_int4` 同属物化层（`epair_utils` 从 `quant_swap.materialize` import）。

<a id="配置文件"></a>
## 配置文件

shipped 默认 `cann_lm/converter/export/weight_format_rules.json`，converter 固定加载此默认配置（不暴露 CLI 覆盖）。该配置缺失时 → 全 ND（等价现状，不启用 NZ）。另有环境变量总开关 `CANN_LM_ENABLE_WEIGHT_TRANS` 默认关闭（见 [环境变量总开关](#环境变量总开关)）——即使配置文件命中 NZ，env 未开也不转换。

### schema

**当前 shipped 实况**（[`cann_lm/converter/export/weight_format_rules.json`](../../../../cann_lm/converter/export/weight_format_rules.json)）：

```json
{
  "default": {"format": "ND"},
  "rules": [
    {
      "name_pattern": "*_proj.weight",
      "op_type": "MatMul",
      "format": "NZ",
      "block": 16,
      "pad": true,
      "pack": "nd2nz"
    },
    {
      "name_pattern": "*",
      "op_type": "QuantBatchMatmulV4",
      "format": "Z",
      "pack": "pack_int4_nz"
    }
  ]
}
```

<a id="wildcard-陷阱与代码-guard2026-10-20-记录"></a>
### wildcard 陷阱与代码 guard（2026-10-20 记录）

上面第二条规则的 `"name_pattern": "*"` **过宽**：它匹配 state_dict 中一切 2D 张量，包括 lifted tensor constants（rotary cos/sin 缓存等）。量化导出时它们会被错误地 Z-pack，导致 GE format 传播 infershape 失败（Reshape_11 编译错误）。当前靠代码层 guard 兜住——`_transform_weight_tensor`（`epair_utils.py:273`）在调 `resolve_format` **之前**额外要求 `name.endswith("weight_q")`。

问题：

1. **配置撒谎**：rules.json 声称"所有 QuantBatchMatmulV4 2D 张量都 Z"，实际只有 `*.weight_q` 会被转换；读配置的人会得到错误预期。
2. **两 packer 语义不一致（现存隐患）**：v1 `_pack_weights_to_dir`（`epair_utils.py:748`）**没有**这层 guard——量化模型若走 v1 路径，`weight_scale`（2D int64）会被 `"*"` 规则命中直接送进 `pack_int4_nz`。

总开关废弃迁移（见 [总开关废弃设计](#总开关废弃设计2026-10-20-定稿)）会把 pattern 收紧为 `"*.weight_q"` 并让两 packer 复用同一 transform，一并消除这两点。

### 字段语义

| 字段 | 含义 |
|------|------|
| `name_pattern` | 权重名 glob（`fnmatch` 语义），如 `*_proj.weight` / `*.weight_q` |
| `op_type` | **分流键**：消费该权重的 GE 算子类型。`MatMul`（fp16 路径）/ `QuantBatchMatmulV4`（量化路径） |
| `format` | 目标格式：`ND` / `NZ`（FractNZ）/ `Z`（FractZ,量化 int4pack 专用） |
| `block` | NZ 块大小（默认 16；Z 路径分形常数硬编码于 `pack_int4_nz`，不走此字段） |
| `pad` | 非 block 倍数维是否补零 |
| `pack` | 变换函数名：`nd2nz`（标准 16×16）/ `pack_int4_nz`（int4 nibble-aware FractZ） |
| `enable` | **规则级开关**（2026-10-20 迁移设计新增，当前代码尚未读取）：`false` → 规则视同不存在（继续匹配后续规则/落 default）；省略 = `true`。落地见 [总开关废弃设计](#总开关废弃设计2026-10-20-定稿) |

> 当前整套规则仍受 env 总开关门控（[环境变量总开关](#环境变量总开关)）；迁移设计见 [总开关废弃设计](#总开关废弃设计2026-10-20-定稿)——落地后本表即控制的唯一权威。

<a id="环境变量总开关"></a>
## 环境变量总开关

> **⚠️ 本节描述的开关已被 [总开关废弃设计](#总开关废弃设计2026-10-20-定稿) 取代（2026-10-20）**，下文保留为历史设计与事故根因记录。

NZ 转换默认**关闭**（等价现状，全 ND，零字节变化）。导出时设环境变量 `CANN_LM_ENABLE_WEIGHT_TRANS=1` 才启用整个 `weight_format` 流程：`resolve_format` + `nd2nz`（写 NZ 字节）+ `weight_info.json` 条目的 `format` 字段（标 `"NZ"`）。未设或非 `1` → packer 跳过 `resolve_format`（含 `is_quantized` 探测），所有权重落 ND + `format` 全 `"ND"`，C++ 侧 `CreateConstNode` 走 `MapPt2Layout`（现状 ND 路径）——代码 dormant，零风险。

**设计意图**：NZ 是性能优化（避运行期 transdata），非正确性必需（当前 ND 量化 OM 已跑通，见 [Stage 划分](#stage-划分)）。Stage 1 的 MatMul NZ 输入接受性未验证（[待验证](#待验证)），故默认关闭、env 开启，便于代码先 dormant 合入，验证通过后再议默认开启或废弃开关。

**作用范围**：Python 导出侧（`epair_utils._pack_weights_to_dir` 写 NZ 字节 + `weight_info.json` 标 `format`）+ C++ 加载侧（`ge_graph_builder.cc:CreateConstNode` 读 `weight_info.json` 的 `format` 字段分流，见 [C++ format 映射](#c-format-映射)）。env 无 C++ 读取——env 关闭时 `weight_info.format` 全 `"ND"`，C++ 走 `MapPt2Layout`（ND）；env 开启时 `format` 为 `"NZ"`，C++ 走 `MapGeLayoutString("NZ")`（待实现）。

**命名约定**：`CANN_LM_ENABLE_WEIGHT_TRANS`（`1` 启用，其他/未设 = 关闭）。沿用 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` 的 `ENABLE` 布尔开关风格；`CANN_LM_` 前缀因仅作用于 cann_lm converter 导出侧。用 `TRANS`（非 `NZ`）因总开关 gate 整个 `weight_format` 流程——Stage 1 `nd2nz` + Stage 2 `pack_int4_nz` + 未来其他格式转换，非仅 NZ。

---

<a id="总开关废弃设计2026-10-20-定稿"></a>
## 总开关废弃设计（2026-10-20 定稿）

### 动机

总开关的设计初衷是"dormant 合入，验证通过后再议"（见上文设计意图）。截至 2026-10-20，验证已全部通过：

- Stage 1（MatMul ND→NZ）：A3 整网 L2/L3，NPU 输出 NZ OM == ND OM，max_diff=0
- Stage 2（QuantBatchMatmulV4 weight_q Z-pack）：KirinX90 板端 q_proj 单算子 cos=0.999912（omc→MRT→golden 全链）

继续保留 env 开关的理由已经消失，而它的成本已被实测证明——**门控不对称事故**（见 [ep_json 标注同步机制](#ep_json-标注同步机制_sync_-补录)）：打包无条件执行、标注同步被 env 门控，env 未开即产出自相矛盾的 epair，错误延迟到编译期才爆 size mismatch，离根因两跳。控制权应移交给 `weight_format_rules.json`——配置可版本化、可 review，且"清空 rules 即全 ND"的逃生门本来就有（配置缺失 → 全 ND = 现状）。

### 门控面盘点：env 实际门控的 4 类动作

去掉 env 之前必须先拆开它门控的东西——**不是每一类都该进 rules.json**：

| # | 动作 | 位置 | 现状 | 去向 |
|---|---|---|---|---|
| ① | weight_q Z-pack（`pack_int4_nz`）与 fp16 `*_proj` NZ（`nd2nz`） | `epair_utils.py:273` `_transform_weight_tensor` | env 门 + 代码 `endswith("weight_q")` 兜底 | **rules.json**（layout 选择正是配置的本职） |
| ② | ep_json 标注同步（`_sync_input_desc_layout` / `_sync_weight_scale_dtype` / converter 侧 `x2.desc.layout="Z"`） | `epair_utils.py:210/:483`、`ge_converters/_quant_matmul_converter.py:103` | env 门 | **不进配置**——无条件跟随①；packer 命中 Z/NZ 即同步 |
| ③ | weight_scale transdata_scale（fp32→int64 位模式） | `epair_utils.py:260` | **本就无条件** | **不进配置**——kernel 语义契约（arch31 s8s4 要求 x2_scale uint64 ND），不是可选项 |
| ④ | `is_quantized` 探测 + rules 加载 | `epair_utils.py:303` `_load_weight_trans_rules` | env 门 | 与①合并：rules 恒载、探测恒做（json 缺失→全 ND 兜底已存在） |

**②③ 不进配置的原则**：rules.json 的字段语义锁定在"layout/format 选择"这一层。dtype 重标注、物化位模式是"命中某 format 后的必然伴随动作"——把必做动作表达成可关动作，正是上次事故的模式。若确需把 weight_scale 处理显式化，至多在 QuantBatchMatmulV4 rule 上加 `"materialize": "transdata_scale"` 这种**声明性**字段（永远生效、仅自文档），而非开关性字段。

### rules.json 目标形态（2026-10-20 修订：每规则独立 `enable`，MatMul 关 / QuantBatchMatmulV4 开）

```json
{
  "default": {"format": "ND"},
  "rules": [
    {
      "name_pattern": "*_proj.weight",
      "op_type": "MatMul",
      "format": "NZ",
      "block": 16,
      "pad": true,
      "pack": "nd2nz",
      "enable": false
    },
    {
      "name_pattern": "*.weight_q",
      "op_type": "QuantBatchMatmulV4",
      "format": "Z",
      "pack": "pack_int4_nz",
      "enable": true
    }
  ]
}
```

四条变化：

1. **新增每规则 `enable` 字段**（替代 env 总开关的粒度单元）：`enable: false` → 该规则视同不存在（继续匹配后续规则，最终落 default ND）；省略 = `true`（列进 rules 即意图启用）。控制粒度从"环境变量一处翻转全部"细化为"每条转换路径单独声明"——这才是"控制权进配置"的完整形态。
2. **MatMul 规则默认关（不做转换）**：fp16 NZ 是性能优化非正确性必需（`ND_ND` kernel 变体本就正确，见 [Stage 划分](#stage-划分)），且启用会改变**所有** fp16 Linear 导出的图结构（C++ 侧需 Transpose skip + `transpose_x2` 机器）——默认关使迁移对 fp16 导出**零行为变化、零回归面**；需要 A3 L3 性能时显式翻 `enable: true`。量化导出受 `op_type` 分流本就只走 QuantBatchMatmulV4 规则集，此规则仅影响 fp16 路径（对当前 q_proj 量化流程无感）。
3. **QuantBatchMatmulV4 规则默认开（必须转换）**：kirinx90 s8s4 kernel 契约要求 x2 = FractZ int4pack、x2_scale = uint64——**正确性必需**路径（此规则若不生效，量化导出直接产出坏 epair），也是本能力当前唯一活跃使用者。
4. pattern 从 `"*"` **收紧为 `"*.weight_q"`**——wildcard 误伤 rotary cos/sin 的 guard 从代码 hack 移回配置层显式表达（见 [wildcard 陷阱](#wildcard-陷阱与代码-guard2026-10-20-记录)）。

逃生门语义不变：rules.json 缺失或清空 `rules` → 全 ND。**`enable` 必须与代码支持（`resolve_format` 读取该字段）同 PR 落地**——当前 `resolve_format` 忽略未知字段，先改 json 后改代码会让 `enable: false` 静默不生效（配置撒谎，重演"数据与标注不同源"事故模式）。

### 双源头一致性

去 env 后有一个粘合点必须接住：packer（写 Z 字节 + weight_info format）与 converter（`_quant_matmul_converter` 设 `x2.desc.layout="Z"`）是**两处独立决策**，现在靠"同一个 env 值"粘合。迁移后两处都必须**同读同一份 rules**——converter 侧再 `load_rules()`（shipped 路径、每次现读、无缓存），据 `resolve_format(name, is_quantized).format == "Z"` 决定是否设 layout，杜绝二处漂移。

### 改动清单

全部 `os.environ.get("CANN_LM_ENABLE_WEIGHT_TRANS")` 读取点（python 侧 8 处；C++ 侧无 env 读取，见上文作用范围）：

| 文件:行 | 改动 |
|---|---|
| `epair_utils.py:303` `_load_weight_trans_rules` | 去 env 判定；rules 恒载（缺失→全 ND）；`is_quantized` 恒探测 |
| `weight_format/rules.py` `resolve_format` | 读取 `enable` 字段：`false` 规则跳过（视同不存在，继续扫描），省略=true——"配置与行为同源"的落点 |
| `weight_format_rules.json` | 更新为"rules.json 目标形态"：MatMul 规则 `enable: false`（不做转换）、QuantBatchMatmulV4 规则 `*.weight_q` + `enable: true`——与 rules.py 支持**同 PR**（防配置先于代码生效） |
| `epair_utils.py:210` / `:483`（consolidated/PD 两出口） | `_sync_input_desc_layout` + `_sync_weight_scale_dtype` 调用**无条件化**（As soon as save_weights）——本项即历史欠账 [P1 修复]，移门后自动消解 |
| `epair_utils.py:273`（v2 transform） | `trans_enabled` 参数删除；guard 完全移入 name_pattern（`*.weight_q`） |
| `epair_utils.py:725-748`（v1 `_pack_weights_to_dir`） | 不再各自内联 transform，**复用 `_transform_weight_tensor`**——顺路修复 v1 缺 guard 的 `"*"` 误伤 bug（见 wildcard 陷阱第 2 点） |
| `ge_converters/_quant_matmul_converter.py:103` | 去 env；`is_s8s4` 且 `resolve_format(...).format=="Z"` 时设 `x2.desc.layout`（与 packer 同源决策） |
| `tests/python/converter/test_weight_format.py` | `env-off` 两用例改为"空 rules json"用例；`env-on` 用例去 monkeypatch（ND/NZ 字节断言不变）；新增 `enable: false` 规则跳过用例（含匹配后续规则的顺序语义） |
| 文档 | 本文（本节 + [ep_json 标注同步机制](#ep_json-标注同步机制_sync_-补录)）、`TODO_notes.md` env 总开关注记、`op_npu_verify/GUIDE_kirinx90_board.md` §1.2 与故障速查 #2（标"上游已修"）；`export_design.md` 经查无 env 提及、无需改 |

### 回归验收

按 `回归锚点` 执行，判据从强到弱：

1. **fp16 零行为变化**（MatMul `enable:false`）：非量化模型新链导出 vs 迁移前 env=0 旧链——weights bin + ep_json **逐位一致**（新链下 MatMul 规则关闭 → 全 ND，与"不转换"完全等价）。
2. **单算子逐位复现**（QuantBatchMatmulV4 `enable:true`）：q_proj 去 env 重导（无 `CANN_LM_ENABLE_WEIGHT_TRANS`）→ 五产物 md5 与 2026-10-20 env=1 基线**逐位一致**（omc `65be769d…`、SubGraph_0.weight `2366b036…`、输入 `4f6b5f3…`）→ 板跑 output_0 md5 复现 `b1344d8e…`（免重算 cos）。
3. **e2e 全模对账**：同输入双路径（env=1 旧链 vs 去 env 新链）的**量化模型**导出，weights bin + ep_json diff 为空。
4. **单测**：`pytest tests/python/converter/test_weight_format.py` 全绿（"空 rules → 全 ND"逃生门、`enable:false` 跳过两族用例）。

### 决策点记录

- **规则级开关（2026-10-20 修订）**：每规则独立 `enable`，取代最初的"全局默认开"姿态——**MatMul（fp16 NZ）默认关**（性能优化非必需、启用改变全量 fp16 图结构，显式 opt-in），**QuantBatchMatmulV4（量化 Z）默认开**（kirinx90 kernel 正确性契约，必须转换）。逃生门=清空 rules；规则级粒度是 env 总开关废弃后的完整替代形态。
- **废弃时序**：迁移 PR 一次移除全部读取点，不做 env/配置并存的双轨期（双轨=两头漂移的温床，第二次机会主义门控的起点）。旧 env 静默忽略即可（无 env 读取点），在 CHANGELOG/发布说明标注。

<a id="格式转换函数"></a>
## 格式转换函数

`weight_format/layout.py`：

### nd2nz（Stage 1）

```python
def nd2nz(tensor: torch.Tensor, block: int = 16, pad: bool = True) -> torch.Tensor:
    """[M, K] ND row-major → FractNZ 物理块 [ceil(M/b), ceil(K/b), b, b] block-major.
    非 b 倍数维内部补零（partial block）；返回同 dtype contiguous 供 tobytes()。"""
```

逻辑：`M, K = shape` → `m, k = ceil(M/b), ceil(K/b)` → `F.pad(..., (0, k*b-K, 0, m*b-M))` → `reshape(m, b, k, b)` → `permute(0, 2, 1, 3)` → `.contiguous()`。bf16 先 int16 view（沿用 `epair_utils` 既有约定）。

### pack_int4_nz（Stage 2，已实现）

int4pack int32 `[K, N//8]`（1 个 int32 打包 8 个 int4）→ FractZ 物理块 `[K//64, N//16, 16, 8]`。**不能复用 `nd2nz`**（int4 值横跨 int32 字节边界，元素级 16×16 转换会打散 nibble 配对与 int32 承载结构）。padding 内部消化（nibble 平面域补齐）；分形常数 `64/16/8` 与 `epair_utils._sync_input_desc_layout` 的 `_B_FRAC_K/_B_FRAC_N/_INT4_PER_INT32` 一致（见 [ep_json 标注同步机制](#ep_json-标注同步机制_sync_-补录)——两处必须同源）。字节布局经 KirinX90 板端实证（cos=0.999912，见 `GUIDE_kirinx90_board.md` 回归锚点）。

<a id="packer-集成"></a>
## packer 集成

`epair_utils` 三个 packer（v1 `_pack_weights_to_dir` / v2 `_pack_weights_v2` / PD `_pack_weights_pd_dedup`）中，**v2 与 PD 经共享的 `_transform_weight_tensor`（:246）**执行转换；核心结构：**dtype view 与布局转换是正交步骤**（bfloat16 先 `view(int16)` 不改字节、只改 dtype 解释；布局再重排物理块主序），不塞同一个 `elif` 链——否则 bfloat16 权重永远命中第一分支、布局分支不可达。v1 现为**独立内联的重复实现**（历史遗留，guard 语义与 v2 不一致——见 wildcard 陷阱），迁移后统一复用 `_transform_weight_tensor`：

```python
# _transform_weight_tensor 现状骨架（epair_utils.py:246；迁移目标的最终形态见"总开关废弃设计"）
if raw.dtype == torch.bfloat16:
    raw = raw.view(torch.int16)
elif "weight_scale" in name and raw.dtype == torch.float32:
    raw = transdata_scale(raw)      # 无条件：kernel 语义契约
entry_format = "ND"
if trans_enabled and raw.dim() == 2 and "weight_scale" not in name and name.endswith("weight_q"):
    rule = resolve_format(name, is_quantized, rules=rules)
    if rule.get("format") == "NZ":
        raw = nd2nz(raw, block=rule["block"], pad=rule["pad"]); entry_format = "NZ"
    elif rule.get("format") == "Z":
        raw = pack_int4_nz(raw); entry_format = "Z"
# 注：现 shipped rules 第二条 name_pattern="*"，靠函数内 endswith("weight_q") 兜底防误伤
#（wildcard 陷阱，见"配置文件"节）；迁移后收紧为 "*.weight_q"，guard 移入配置层。
```

### op_type 分流（is_quantized 探测）

单次导出要么纯 fp16、要么纯量化（量化时所有 Linear 换成 QuantLinear，无混合）。packer 探测 state_dict 是否含 `*_weight_scale` 张量 → 判 `is_quantized` → 选 `MatMul` 或 `QuantBatchMatmulV4` 规则集。**无需逐权重遍历图查消费算子**（避免给 packer 加图遍历负担）。prefill-decode 共享 pack 一次，NZ/Z 只转一次（天然正确）。

> env 关闭时的"跳过 resolve_format + is_quantized"行为随总开关废弃一并移除（见 [总开关废弃设计](#总开关废弃设计2026-10-20-定稿) 动作④）——rules 恒载、探测恒做，缺失 json 时 `resolve_format` 走 default ND，行为退化等价且无结合 env 才成立的两态。

<a id="ep_json-layout-标注"></a>
## ep_json layout 标注

> **机制修正（2026-09-05 探明）：原设计假设 NZ 经 `inline_weight_layout` 改 ep_json，此路不通**：
> - `tensor_values[p_*].layout` 是 **pt2 枚举**（7=kLayoutStrided=ND），pt2 无 NZ 枚举 → 写不进 NZ
> - 权重是 `tensor_values` 的 Data 节点（`p_*`），**不是 ascend_ir Const 算子**（ascend_ir 的 Const 全是小 inline 常量，最大 4 元素）→ 无 `ascend_ir.desc.layout` 可改
> - 消费算子 `inputDesc[i].layout`（字符串 "ND"）虽可改 "NZ"，但权重 Const 仍 ND → 反触发 TransData

**NZ 信号改走 `weight_info.json` 的 `"format":"NZ"` 字符串**（`_pack_weights_to_dir` 已落盘），由 C++ `CreateConstNode`/`GeWeightLoader` 读取 → `MapGeLayoutString`（见 [C++ format 映射](#c-format-映射)）。`inline_weight_layout` 不再实现。

<a id="ep_json-标注同步机制_sync_-补录"></a>
## ep_json 标注同步机制（_sync_* 补录）

> 2026-10-20 补录：代码中存在但此前无文档的两个 ep_json 后处理函数。它们是 Stage 2（Z）路径能编译通过的关键环节，也是 2026-10-20 板端事故的根因所在。

权重字节转换发生在 packer（`_transform_weight_tensor`），但 torchair lite 导出生成的 ep_json 仍带着**原始 ND 语义的标注**。落盘前必须把标注同步成转换后语义，两处修补：

### `_sync_input_desc_layout`（`epair_utils.py:491`）

读 weight_info 的 blob `format` 字段，对非 ND 张量修补 ep_json 三点：

1. **inputDesc layout**：`"ND"` → blob format（如 `"Z"`）
2. **inputDesc shape（Z only）**：2D `[K, N_packed]` → 4D `[K//64, N//16, 16, 8]`（N = N_packed×8，8 个 int4 打包进 1 个 int32）
3. **tensor_values sizes（Z only）**：2D → 4D，匹配 Z 分形形状

Z 分形常数（`_B_FRAC_K=64, _B_FRAC_N=16, _INT4_PER_INT32=8`）必须与 `layout.py` 的 `pack_int4_nz` 一致。不修 #2/#3 的后果：GE infershape 把 2D shape 穿过 Z layout 传播，产生错误 StridedSlice 输出维度（如 `[1,1,16,8]` 而非 `[2,1]`），引发 Reshape 编译失败。张量名映射：`op.weight_q` → inputDesc `x2`（QuantBatchMatmulV4）。

### `_sync_weight_scale_dtype`（`epair_utils.py:615`）

`transdata_scale` 把 weight_scale 数据从 fp32 物化成 uint64 位模式，ep_json dtype 标注同步修补两点：

1. **tensor_values**：dtype 7（float32）→ 100（synthesized uint64）——不修则 C++ 侧重装载 `WEIGHT_SIZE_MISMATCH`
2. **inputDesc**：`x2_scale` 的 `DT_FLOAT` → `DT_UINT64`——不修则 arch31 tiling 拒绝（arch31 s8s4 要求 x2_scale 为 uint64 ND）

自检信号：导出日志出现 `Synced N inputDesc shape(s) 2D → 4D for Z layout` / `Synced N tensor_values sizes 2D → 4D for Z layout` / `Synced weight_scale dtype: N tensor_values, N inputDesc` 三行。

### 2026-10-20 门控不对称事故（根因记录）

**现象**：单算子导出（env 未开）→ 编译期 `PreparePlaceholders failed: load tensor data error, placeholder=b_op_weight_scale`。

**根因**：同一张 weight_scale，两条处理路径的门控不一致——

| 处理 | 执行条件 |
|---|---|
| `transdata_scale` 打包（fp32→uint64 位模式，8B） + `pack_int4_nz` | **无条件**（`_transform_weight_tensor` 内联执行） |
| `_sync_weight_scale_dtype` / `_sync_input_desc_layout` 标注同步 | **仅 `CANN_LM_ENABLE_WEIGHT_TRANS=1`**（`epair_utils.py:210/:483` 的 if 门） |

env 未开 → 产出自相矛盾的 epair（weights_1.bin 8B×40960=327680B 数据，ep_json 却标 4B float32）→ C++ `BuildTensorHolder` expected_bytes=163840 ≠ 实际 327680 → 编译期报 size mismatch，**离根因两跳**（错误不指向导出侧）。

**教训（已沉淀为总开关废弃设计的核心论据）**：把"必然伴随动作"表达成"可关动作"必然制造两头漂移空间。数据与其标注必须同门同源——迁移方案（[总开关废弃设计](#总开关废弃设计2026-10-20-定稿) 动作②）将其无条件化，从机制上消灭这类矛盾。

<a id="c-format-映射"></a>
## C++ format 映射

> **机制修正（2026-09-05 探明）：原设计写「权重 Const format 由 `ascend_ir.desc.layout` 驱动（`MapGeLayoutString`）」——错函数**。实探 `ge_graph_builder.cc`：

**权重 Const 路径 = `CreateConstNode`**（`ge_graph_builder.cc:473-509`，在 `:1213-1221` 给 Parameter/Buffer/TensorConstant placeholder 调用）：
- `:478` `format = mapper.MapPt2Layout(holder.format, ...)`——`holder` 来自 `GeWeightLoader::ReadTensorData`，`holder.format` 是 **pt2 枚举**（来自 ep_json `tensor_values[p_*].layout`=7）
- `:487` `ge::TensorDesc desc(ge::Shape(holder.dims), format, dtype)` → Const 节点（`--external_weight` 时 GE 编译期 `ConvertConstToFileConst`→FileConstant）

**`MapPt2Layout`**（`aten_tensor_desc_mapper.cc:67`）只 `kLayoutStrided→FORMAT_ND`，**无 NZ**；pt2 无 NZ 枚举 → 权重 Const 当前表达不了 NZ。

**`MapGeLayoutString`**（`aten_tensor_desc_mapper.cc:119`）只被 `ValidateConstOutputDesc`（`:517`，inline Const 校验）+ `RestoreInlineConstTensor`（inline Const 的 pt2-layout-string 转换）用——**不是权重 Const 路径**。

### C++ format 改动（已落地，NZ + Z 两 case）

| 文件 | 函数 | 改动 | 状态 |
|---|---|---|---|
| `aten_tensor_desc_mapper.cc:119` | `MapGeLayoutString` | `"NZ"` → `ge::FORMAT_FRACTAL_NZ`（:127 注释即此来源）；Z 路径经 `QuantBatchMatmulV4` inputDesc 的 `"Z"` 走 `MapAscendIrDesc` | ✅ |
| `ge_weight_loader.cc`（`TensorRecordInfo`/`TensorHolder`）+ `ge_graph_builder.cc:478`（`CreateConstNode`） | 读 `weight_info.json` 的 `"format"` 字符串 + `CreateConstNode` 据此分流 | `holder`/`TensorRecordInfo` 携带 `format`；命中 `"NZ"` → `MapGeLayoutString("NZ")→FORMAT_FRACTAL_NZ`，否则 `MapPt2Layout(holder.format)`（ND） | ✅ |

注：`ge_graph_builder.cc:401 MakeTensorDesc`（`:407 SetFormat(info.format) ← MapAscendIrDesc`）是 **ascend_ir op desc 路径**（inline Const / 普通算子 inputDesc/outputDesc），**非权重 Const**；权重 Const format 在 `CreateConstNode:487` 直接构造 `TensorDesc(format, dtype)`，不经 MakeTensorDesc。

## Z shape 语义（Stage 2 量化路径）

与 NZ 一节平行的 FractZ 契约（`pack_int4_nz` 产物， kirinx90 QuantBatchMatmulV4 s8s4 kernel 消费）：

- inputDecl 2D `[K, N_packed]`（int32，8 int4/元素）→ Z 物理 4D `[K//64, N//16, 16, 8]`（N = N_packed×8）
- ep_json shape 标注必须同步改 4D（`_sync_input_desc_layout`，见 [ep_json 标注同步机制](#ep_json-标注同步机制_sync_-补录)），否则 GE infershape 沿 2D 穿 Z 传播出错维
- q_proj 实测：`[2048, 320]` int32 → `[40, 128(=2048/16), 16, 8]`

<a id="nz-shape-语义"></a>
## NZ shape 语义

遵循 GE `FORMAT_FRACTAL_NZ` 标准契约：

- **ep_json shape = 逻辑 [M, K]**（非 padded）
- `nd2nz` 内部对非 16 倍数维补零（保证字节完整），写入物理 NZ 块
- GE kernel 读 partial block 的有效元素、忽略 pad 零值

**不写 padded shape**（`[m*16, k*16]`）——会让 MatMul 输出维度变 padded_M，破坏下游 shape。LLM Linear 维度实践上多 16 对齐（hidden 1024/2048、intermediate 3072、head_dim 128×n_heads 等），pad 极少触发。

<a id="stage-划分"></a>
## Stage 划分

### Stage 1 — fp16 路径（本次交付，低风险）

1. `weight_format/layout.py` 加 `nd2nz`
2. `weight_format/rules.py` 配置加载（`load_rules` 读 shipped 默认配置）+ `resolve_format`
3. `epair_utils._pack_weights_to_dir` 集成（env-gated，见 [环境变量总开关](#环境变量总开关)）——写 NZ 字节 + `weight_info.json` 标 `format`（`inline_weight_layout` 已弃用，见 [ep_json layout 标注](#ep_json-layout-标注)）
4. C++ `CreateConstNode`/`GeWeightLoader` 读 `weight_info.format` + `MapGeLayoutString` 加 NZ case（见 [C++ format 映射](#c-format-映射)）
5. 文档同步（本文档 + export_design.md / quant_design.md 注记 + INDEX.md）

### Stage 2 — 量化路径（已实现并板端验证）

量化 weight_q 是 int4pack int32 `[K, N//8]`，**标准 `nd2nz` 会打散 nibble 配对**，实现为专用的 `pack_int4_nz`（FractZ，见 [格式转换函数](#格式转换函数)）。历史研究门记录（逆向 CANN kernel 期望的确切字节排列）已随 KirinX90 板端 cos=0.999912 关闭；oracle 判决方法（常数输入枚举 f-bit 分解）沉淀于板端指南。

**关键事实**：A3 实测 `QuantBatchMatmulV4` 有 `ND_ND` 与 `ND_NZ` 两个并列 kernel 变体——喂 ND 权重选 `ND_ND`（内部处理 transpose），喂 Z 权重选跳过内部转换的变体；预转布局是**性能优化非正确性必需**，做错会打破能跑的 OM（这正是历史上设研究门、并且 shipped rules 曾标 `"format": "ND"` 的原因）。

<a id="待验证"></a>
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

**当前实现**:16 对齐 shape 用 NZ(已验证);非对齐自动回落 ND(需在 `_pack_weights_to_dir` 加 shape 对齐检查)。LLM 实际权重维度多 16 对齐(hidden 1024/2048、intermediate 3072、head_dim 128×n_heads 等),非对齐极少。

### L3 整网(A3)

LLM `--mode single` + `CANN_LM_ENABLE_WEIGHT_TRANS=1` 导出 → OM 编译 → `atc --mode=1` 查 `*_proj` 对应 MatMul 无 TransData → NPU 出字 vs ND 版一致。

### L4 Kirin(2026-10-20 部分达成)

KirinX90 **量化 Z 路径（Stage 2）**已实测打通:q_proj 单算子(E4B int4pack W4A8)→ epair(Z-packed weight_q + uint64 scale)→ `cann_lm_compile` → 骨架 omc + 外置 SubGraph_0.weight → 板上 MRT V2 → 板端输出 vs dopt fakequant golden **cos=0.999912**(跨语义判据 ≥0.99、同语义 ≥0.999 双过)。全流程命令模板与回归锚点(md5 基线)固化在 ``examples/test_converter/dopt_op_benchmark/op_npu_verify/GUIDE_kirinx90_board.md``。fp16 MatMul NZ 于 Kirin 的 L2/L3 复验仍延后(kirinx90 板端不支持多流,matmul ND/Z kernel 行为另验)。

> 注意 Z ≠ NZ 语义:量化 weight_q 走 `pack_int4_nz` 落 **FractZ**(`[K//64,N//16,16,8]`,政策由 QuantBatchMatmulV4 kernel 定义),与 fp16 `*_proj` 的 FractNZ(16×16)是不同布局。rules.json 的 `"format": "Z"` 与 `"NZ"` 字段分别对应两者。

### 核心判据(修正)

NZ 重排成功 ⟺ **NPU 输出 NZ OM == ND OM**(max_diff=0)+ OM 里权重 Const `layout=FRACTAL_NZ` + 0 host TransData。TransData 计数不是唯一指标(kernel 内部转换无 TransData 但有开销)。

<a id="后续迁移"></a>
## 后续迁移

### 已完成（2026-08-31）：quant_swap 迁入 export/ + swap 升至 converter 根

`materialize.py` / `quant_linear.py` / `swap.py` 已从 `cann_lm/converter/quant/` 迁出：
- `materialize.py` / `quant_linear.py` → `cann_lm/converter/export/quant_swap/`（物化内部，与 export pipeline 同位）
- `swap.py` → `cann_lm/converter/swap.py`（converter 根，编排层 peer of `loader.py` / `saver.py`，绝对 import materialize/quant_linear）

旧 import 路径已干净切断：`swap_from_dopt` 走 `cann_lm.converter.swap`，物化符号走 `cann_lm.converter.export.quant_swap`。`quant/` 现仅留 CANN 轻量化工具编排（`auto_quant` / `dopt_wrapper`）+ vendored `dopt_lm` + 校准数据。详见 [quant_design.md](../quant/quant_design.md) "quant 包架构"。

### 已决定（2026-09-05）：transdata_scale / pack_int4 不并入 weight_format/

`transdata_scale` / `pack_int4` 现位于 `export/quant_swap/materialize.py`，语义同属"格式转换"。**不并入** `weight_format/layout.py`——二者是**物化位模式打包**（`transdata_scale`：fp32→int64 位模式；`pack_int4`：int4→nibble-packed int8），非布局变换，与 `nd2nz` / `pack_int4_nz` 的**布局重排**职责正交，留 `export/quant_swap/materialize.py`。`epair_utils` 维持从 `quant_swap.materialize` import `transdata_scale`，无需改 import / 测试。

### 已定稿（2026-10-20）：CANN_LM_ENABLE_WEIGHT_TRANS 废弃迁移

见 [总开关废弃设计（2026-10-20 定稿）](#总开关废弃设计2026-10-20-定稿)。要点：控制权移交 `weight_format_rules.json`（pattern 收紧 `*` → `*.weight_q`、默认生效、清空 rules 即全 ND 逃生门）；标注同步 `_sync_*` 无条件化；v1 packer 复用 `_transform_weight_tensor` 修复 wildcard 误伤；迁移 PR 一次性移除全部 8 处 env 读取，无双轨期。**未实施**——本节为设计定稿，落地时按该节"改动清单"+"回归验收"执行。

## 文档关联

| 文档 | 关联 |
|------|------|
| [export_design.md](./export_design.md) | 导出主流程 / `epair_utils` / C++ `GeWeightLoader` / `inline_weight_scale_dtype` 先例 |
| [quant_design.md](../quant/quant_design.md) | 量化路径 / `transdata_scale` / `QuantBatchMatmulV4` / weight_scale 落盘 |
| [INDEX.md](./INDEX.md) | 导出模块索引（已同步补 weight_format 包条目） |
| ``op_npu_verify/GUIDE_kirinx90_board.md`` | 用户级端到端指南：单算子导出→omc 编译→KirinX90 板跑→比对；含本能力的使用契约（导出期 env、3 行 Synced 自检）与回归锚点 |
