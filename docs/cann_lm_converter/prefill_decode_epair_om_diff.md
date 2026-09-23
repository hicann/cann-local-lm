# Prefill 与 Decode epair/OM 差异分析

> 基于 qwen3.5-0.8b（24 层 linear attention，`--mode prefill-decode --dtype bfloat16 --seq-len 8`）实证。prefill seq=8 / decode seq=1，两次独立 `torch.export` shape-specialized 导出。

## 结论

- **epair**：差 42 node / 114 GE op，**全在 view/slice/clone**（激活张量 seq 维处理），核心计算算子（MatMul/RmsNorm/ChunkGatedDeltaRule/Conv2D）逐层一致。
- **OM**：GE fusion 消除 ~60% 辅助算子，残差 128 个是 **seq=8 真实计算需求**（Transpose/Add/StridedSlice），非冗余。
- **OM weight**：模型参数（326 hash）一份共享、内容一致；md5 不同源于 **GE 编译期派生常量**（prefill +8 / decode +5，编码 seq_len）+ 排列顺序，非两份不同权重。
- **根因**：torch.export shape-specialized trace——同一模型代码在 seq=8/1 下展开成不同算子数。算子是模型代码写的，数量差异是 trace 优化决策。
- **规整**：OM fusion 已自动消除辅助差异；要 EP 统一需 converter 图后处理 pass；完全一致只有 dynamic shape（代价大，不推荐）。

---

## 1. epair 层差异

### 结构总览

| 指标 | PREFILL | DECODE | Δ |
|------|---------|--------|-----|
| FX nodes | 2040 | 1998 | -42 |
| GE op 总数 | 3886 | 3772 | -114 |
| inputs / outputs | 397 / 49 | 397 / 49 | 0 |
| 独有 GE op | 0 | 0 | — |

**无算子种类差异**，仅数量不同。

### FX node 差 42（4 个 target）

| FX target | PF | DC | Δ |
|-----------|----|----|----|
| `clone.default` | 30 | 6 | **+24** |
| `slice.Tensor` | 91 | 73 | **+18** |
| `_unsafe_view.default` | 211 | 193 | **+18** |
| `reshape.default` | 497 | 515 | **-18**（decode 切片退化成 reshape） |

### FX→GE 对应（差 114，核对闭合）

| FX | → GE op | PF | DC | Δ |
|----|---------|----|----|----|
| clone | Identity | 30 | 6 | +24 |
| slice.Tensor | StridedSliceV2 | 91 | 73 | +18 |
| slice 参数（每 slice 4 个 Const） | Const | 364 | 292 | +72 |
| reshape + unsafe_view | Reshape（和一致） | 708 | 708 | 0 |

`72+24+18 = 114` ✅；`24+18+18−18 = 42` ✅。

### 差异归属（stack_trace 定位模型代码）

算子是模型 forward 代码写的，非 trace 凭空插入：

| 算子 | 模型代码 | 行 |
|------|---------|----|
| slice | `position_ids = position_ids[1:]` | modeling_qwen3_5.py:889 |
| clone | `gate = gate.reshape(*input_shape, -1)` 附近（防 aliasing） | :683 |
| unsafe_view | `self.in_proj_qkv(...)` → `F.linear(...)` 后（MatMul 输出重排） | :550 |

stack_trace 顶层：`export_wrapper.py:61 → modeling:959(model) → :921(decoder_layer) → :765(self_attn)`，差异集中在 linear attention 层 + wrapper 的 KV/delta 处理。slice 前驱含 `conv1d+silu`（qwen3.5 short-conv 输出切片），有层号命中的分布在层 3,7,11,15,19,23（hybrid 架构每 4 层一组）。

### 核心算子完全一致

MatMul 187/187、RmsNorm 79/79、ChunkGatedDeltaRule 18/18、Conv2D 18/18、ScatterNdUpdate 12/12——计算语义不依赖 seq_len。

---

## 2. OM 层差异（GE fusion 后）

| | PREFILL | DECODE | Δ |
|---|---------|--------|-----|
| OM op | 1551 | 1423 | -128 |
| vs EP 消除率 | 60% | 62% | — |

差异算子（fusion 消不掉的残差）：

| GE op | PF | DC | Δ | 原因 |
|-------|----|----|----|------|
| Transpose | 108 | 48 | -60 | seq 维转置（[8,heads,dim]→[heads,8,dim]） |
| Add | 66 | 18 | -48 | 残差广播加（decode 部分被 InplaceAddRmsNorm 吸收） |
| StridedSliceD | 79 | 61 | -18 | seq 维切片（同 EP） |
| 核心算子 | 全等 | | | MatMul/RmsNorm/ChunkGatedDeltaRule/Conv2D/RotaryMul 一致 |

残差是 **seq=8 的真实计算需求**，static 导出无法消除。

---

## 3. OM weight 差异

### epair 层：权重一份共享

`weight_info.json` + `weights_0.bin`(1.5G) + `host_weights_0.bin`(485M) 各一份，导出 log `shared weights`，decode 段 `save_weights=False`。

### OM 层：各固化一份，md5 不同

| | prefill/weight | decode/weight |
|---|---|---|
| combined 大小 | 1.5G | 1.5G |
| md5 | a89cd2... | 8e3a4b... **DIFF** |

### 原因：派生常量 + 排列

meta.json 用 hash 内容寻址：

| | hash 数 | 共有 | 独有 |
|---|---|---|---|
| prefill | 334 | 326 | **+8** |
| decode | 331 | 326 | **+5** |

- **326 共有 hash = 模型参数，内容完全一致**（hash 相同）。
- **独有的是 GE 编译期派生常量**，其值直接编码 seq_len：

| prefill 独有（8） | decode 独有（5） |
|---|---|
| `8`（slice size=seq_len） | `1`（seq=1 退化） |
| `[2048,2049,2050,2051...]`（8 个位置索引） | `2048`（单个位置） |
| `[0,2,1]`/`[0,2,1,3]`（transpose perm） | `[0,0,1]`（退化 perm） |
| `0x01010101`（broadcast）/ zero padding | `0x01010101` / zero padding |

- **排列不同**：326 共有权重中 324 个 offset 不同（图算子遍历顺序差异）。

`GE_EXTERNAL_WEIGHT=2` 把所有 Const（模型权重 + 派生常量）都分离到 combined，所以派生常量差异 + 排列差异共同导致 md5 不同。**模型参数没有"两份不同"**。

---

## 4. 根因与规整

### 根因：torch.export shape-specialized trace

torch.export 是 concrete（非 symbolic）trace，seq_len 具体值进入 trace 路径。同一行模型代码（如 `position_ids[1:]`）：
- prefill seq=8 → 切 `[8,...]` 需显式 `slice`
- decode seq=1 → 切 `[1,...]` = 整个张量，**优化成 `reshape`**（slice 退化）

算子是模型代码写的，**数量差异是 torch.export 的 shape-specialized 优化决策**（slice 退化、防 aliasing clone 插入、view 选择）。torchair FX→GE 仅类型映射，不改变数量差。

### 三层作者

| 层 | 角色 | 产生差异 |
|----|------|---------|
| 模型代码 | 显式写 slice/reshape/clone | 触发源 |
| **torch.export shape-specialized** | 同一代码不同 seq_len → 不同算子数 | **差异产生处** |
| torchair | slice→StridedSliceV2 等类型映射 | 否 |

### 规整方案

| 方案 | 可行性 | 效果 |
|------|--------|------|
| OM fusion（已自动） | 已生效 | 消除 Reshape/Identity/Cast（免费） |
| converter 图后处理 pass | 最可行 | slice[0:1]→reshape、消冗余 clone、统一 view（需改 converter） |
| 模型代码改写 | 难，且 trace 仍可能差异 | 不彻底 |
| dynamic shape export | 代价大 | 完全一致，但性能 -20~50%、converter 重构、不推荐 |

残差 128 个 OM 差异是 seq=8 真实计算需求，**static 下无法消除**。prefill/decode 分离特化是业界主流（vLLM/TensorRT-LLM/SGLang），不追求"算子完全一致"——分离特化各自优化才是最优。
