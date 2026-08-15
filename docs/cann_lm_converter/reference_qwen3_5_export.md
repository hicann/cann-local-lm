# Qwen3.5 导出问题专题

> 把 `converter_design.md` 中散布的 qwen3.5 导出相关内容汇集于此，便于单独追踪。
>
> 同时记录 2026-08-09 新诊断的 OM 编译维度约定错误（`converter_design.md` 遗留任务 3 尚未反映此状态）。
>
> 最后更新: 2026-08-09

## 目录

- [概述](#概述)
- [1. 模型架构差异](#1-模型架构差异)
- [2. Qwen3_5ExportWrapper 子类](#2-qwen3_5exportwrapper-子类)
- [3. mask 4D 透传](#3-mask-4d-透传)
- [4. 线性注意力导出（核心问题）](#4-线性注意力导出核心问题)
  - [4.1 原始 segfault（纯 torch for 循环）](#41-原始-segfault纯-torch-for-循环)
  - [4.2 融合算子方案（epair 导出层已解决）](#42-融合算子方案epair-导出层已解决)
  - [4.3 OM 编译维度约定错误（已修复）](#43-om-编译维度约定错误已修复)
  - [4.4 delta_cache tensor 化（全链改动）](#44-delta_cache-tensor-化全链改动2026-08-09)
- [5. scatter-write 改造（遗留任务 5）](#5-scatter-write-改造遗留任务-5)
- [6. 全量版难点](#6-全量版难点)
- [7. 精度验证](#7-精度验证)
- [8. ACL 运行 OM + 数值差异调试](#8-acl-运行-om--数值差异调试)
  - [8.1 现状总表](#81-现状总表)
  - [8.2 nan 根因：random input](#82-nan-根因random-input)
  - [8.3 OM vs EP 对比：确认 OM pipeline 有问题](#83-om-vs-ep-对比确认-om-pipeline-有问题)
  - [8.4 ChunkGatedDeltaRule 单算子对比：scale 根因确认](#84-chunkgateddeltarule-单算子对比scale-根因确认)
  - [8.5 全模型差异归属修正：非 chunk](#85-全模型差异归属修正非-chunk)
  - [8.6 initial_state 未生效（独立问题）](#86-initial_state-未生效独立问题)
  - [8.7 下一步：逐算子定位全模型 15.78](#87-下一步逐算子定位全模型-1578)
  - [8.8 相关脚本](#88-相关脚本)
  - [8.9 复现](#89-复现)
  - [8.10 embeddings 入图验证](#810-embeddings-入图验证2026-08-10)

- [9. npu-op-elimination 改造（2026-08-10）](#9-npu-op-elimination-改造2026-08-10)

---

## 概述

Qwen3.5-0.8B 是**混合架构**：6 层 full_attention（标准 GQA + KV cache）+ 18 层 linear_attention（GatedDeltaNet + recurrent state / delta_cache）。这种混合架构给导出带来三方面特殊性：

1. **线性注意力的 recurrent state 不是 KV buffer 格式**：`[num_v_heads, k_head_dim, v_head_dim]` 的 recurrent state（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——需单独设计 recurrent buffer 的增量协议。
2. **delta_cache 跨调用持久**：linear attention 的 recurrent state 必须跨 prefill/decode 调用持久，wrapper 用 `list` 容器（`_delta_cache_box`）承载，避 `nn.Module` tensor 属性赋值检测。
3. **CANN 融合算子 `ChunkGatedDeltaRule` 维度约定**：与标准 GQA attention 不同，需 THD 3D 布局（见 [4.3](#43-om-编译维度约定错误已修复)）。

### 导出链路现状

| 阶段 | 状态 | 说明 |
|------|------|------|
| wrapper 精度验证 | ✅ PASS | `verify_wrapper_qwen3_5.py` logits max_diff=0.0 |
| wrapper vs HF | ✅ PASS | logits max_diff=0.019（bfloat16 正常），tokens 一致 |
| `torch.export` | ✅ 成功 | prefill 2858 / decode 2864 节点（融合算子路径） |
| `export_to_epair`（GE 转换） | ✅ 成功 | ep_json 生成（delta_states 图输入，无 dict 赋值 copy / 常量 mutate） |
| Python ep 回环 | ✅ PASS | `verify_qwen3_5_epair.py` ep vs wrapper vs HF 一致（prefill+decode） |
| **C++ OM 编译（prefill）** | ✅ **PASS** | `qwen3_5.om` 2.03GB（[4.3](#43-om-编译维度约定错误已修复) 维度修复 + [4.4](#44-delta_cache-tensor-化) tensor 化） |
| **C++ OM 编译（decode）** | ✅ **PASS** | `qwen3_5_decode.om` 2.03GB（delta_states 图输入输出） |
| **ACL 运行 OM（NPU）** | ✅ 真实 input 不 nan，OM vs EP maxdiff=0.28，三方 top5 一致 | acl 加载执行（89 输入 49 输出）；修复 npu_chunk 前 l2norm + scale（参考 cann-recipes-infer），OM vs EP 0.28 / OM vs HF 0.375，top5_overlap 5/5，见 [8](#8-acl-运行-om--数值差异调试) |
| **embeddings 入图** | ✅ PASS | `--use-embeddings`：input_ids 不入图，host 查表 embeddings 喂 OM；OM vs HF 0.375 / top5 5/5，与 input_ids 入图数值等价，见 [8.10](#810-embeddings-入图验证2026-08-10) |

> 历史：`debug_qwen3_5/README.md` 曾称"遗留任务 3 已解决"，但那只到 epair 导出层；2026-08-09 跑 C++ OM 编译暴露维度约定 bug（[4.3](#43-om-编译维度约定错误已修复)），同日修复；又暴露 delta_cache 对象在 export 路径的 dict 赋值 copy + in-place mutate 问题（[4.4](#44-delta_cache-tensor-化)），全链 tensor 化修复；prefill+decode 双图 OM 编译打通；2026-08-09~10 ACL 运行 OM 发现 random input nan + OM vs EP 15.78，逐算子定位到 ChunkGatedDeltaRule kernel 未做 l2norm + scale 默认 1.0（参考 cann-recipes-infer 修复：modeling 前 l2norm + 传 scale），OM vs EP 0.28 / OM vs HF 0.375，三方 top5 一致（见 [8](#8-acl-运行-om--数值差异调试)）。

---

## 1. 模型架构差异

qwen3.5 与 gemma4 / qwen3 的接口不一致，converter 通过注册表 + adapter 抹平。以下差异表均摘自 `converter_design.md`。

### Config 加载差异

| 模型 | from_pretrained | from_json_file | to_dict | config_json_key |
|------|:-:|:-:|:-:|---|
| qwen3_5 | 有 | 有 | **无** | `text_config` |

qwen3_5 的 config 嵌套在 `config.json` 的 `text_config` 子字典下，`build_config` 从 `text_config` 取子字典再按 dataclass 字段过滤。`to_dict` 无，故 `save_model_config` 走 `dataclasses.asdict` fallback。

### forward 签名差异

| 参数 | gemma4 | qwen3 | qwen3_5 |
|------|:------:|:-----:|:-------:|
| position_ids | 无（内部计算） | 有 | 无（内部计算） |
| past_key_values | `list` | `list` | **`SimpleKVCache`** |
| output_hidden_states | 有 | 有 | **无** |
| use_cache | 无 | 无 | **有** |
| **delta_cache** | 无 | 无 | **有（独有）** |

### forward 返回差异

| 返回 key | gemma4 | qwen3 | qwen3_5 |
|----------|--------|-------|---------|
| past_key_values | 有 | 有 | 有 |
| hidden_states | 有 | 有 | **无** |
| **delta_cache** | 无 | 无 | **有（独有）** |

### 权重 key 重映射差异

| 模型 | 重映射策略 | 说明 |
|------|-----------|------|
| qwen3_5 | `model.language_model.*` → `model.*` | 通过 `_remap_key` 去掉 VLM 前缀（同 gemma4） |

---

## 2. Qwen3_5ExportWrapper 子类

`cann_lm/models/qwen3_5/export_wrapper.py` 的 `Qwen3_5ExportWrapper` 继承 `ExportWrapper` 基类，实现 `_drive_layers`。

### 与其他子类的差异

| 子类 | 原生 cache 类型 | forward kwargs 差异 | 特殊处理 |
|------|----------------|---------------------|---------|
| Qwen3_5ExportWrapper | `SimpleKVCache` | `delta_cache`、`use_cache` | list → SimpleKVCache 转换；`delta_cache=None`；无 PLE |

### 关键实现

- **list ↔ SimpleKVCache 互转**：外部传入的 `list[(k,v)]` 转成 `SimpleKVCache` 再传给模型；模型返回的 `SimpleKVCache` 展平为 `[k_0, v_0, k_1, v_1, ...]`。
- **delta_cache 持久化**：`self._delta_cache_box = [None]`（list 容器），承载 linear attention 的 recurrent state，跨 prefill/decode 调用持久。用 list 容器而非 `nn.Module` tensor 属性，避 torch.export 报 attribute 赋值检测。
- **full_attention 层索引映射**：`_get_full_attn_layer_indices()` 从 `config.layer_types` 取 full_attention 层索引（截断到实际层数），KV cache 层索引通过此映射（full_attention 层不是连续的）。
- **prefill vs decode 的 past 区分**：
  - prefill（`get_sample_inputs` 输出，24 对，全层）：按 `layer_idx` 取 full_attn 层
  - decode（`reconstruct_kv` 输出，6 对，只 full_attn，顺序 `full_attn_indices`）：按 `idx` 配对

### 纯 embeddings 模式

qwen3_5 无 PLE，`embeddings` 非 None 时只传 `inputs_embeds`（不传 `input_ids`，input_ids 不入图），与 gemma4（额外 `per_layer_token_identity`）不同。返回 7-tuple，对齐 `forward` 签名（`per_layer_token_identity` 默认 None）。

---

## 3. mask 4D 透传

模型的 `_prepare_causal_mask` / `_create_causal_mask` 内部有动态控制流（依赖运行时 seq 长度），torch.export 无法静态追踪。传 4D mask 时走透传路径（`return attention_mask`），跳过动态构造。

| 模型 | 透传代码位置 |
|------|-------------|
| qwen3_5 | `modeling_qwen3_5.py` 第 616 行 `if attention_mask.dim() == 4: return attention_mask`（**新增**） |

qwen3_5 的 4D 透传是为导出新增的（gemma4 / qwen3 原生就有），是新模型接入指南里"mask 4D 透传"检查项的参考实现。

---

## 4. 线性注意力导出（核心问题）

qwen3.5 有 18 层线性注意力（GatedDeltaNet），这是导出问题的集中点。`GatedDeltaNet.forward`（`modeling_qwen3_5.py:415-506`）的 prefill 路径调 `torch_chunk_gated_delta_rule`（~78 行 Python for 循环），decode 路径调 `torch_recurrent_gated_delta_rule`（逐 token for 循环）。

### 4.1 原始 segfault（纯 torch for 循环）

**问题**：`torch_chunk_gated_delta_rule` 的 Python for 循环（63 次迭代 + `.clone()`）导致 `torch.export` 展开约 **19418 节点**，`export_to_epair` 转 GE IR 时图爆炸 → **segfault**。

**根因**：纯 Python fallback 版本循环展开，图节点过多。

**当时状态**（`converter_design.md` 遗留任务 3 记录）：
- `torch.export` 成功（19242 节点），wrapper 精度验证通过（4 层 ALL PASS）
- `export_to_epair` 崩溃

**当时考虑的解决方向**：
- 方案 A：安装 `flash-linear-attention` (fla) + triton + GPU 环境，用编译的 triton kernel 替换 Python 循环（当前 CPU-only 环境不可用）
- 方案 B：将 delta rule 封装为自定义 GE 算子（类似 `FusedInferAttentionScore`），不展开内部循环
- 方案 C：在模型代码中集成 fla kernel（和 transformers 一致：有 fla 用 kernel，无 fla fallback）

### 4.2 融合算子方案（epair 导出层已解决）

最终采用**方案 B 的变体**：复用 CANN 已有的 `ChunkGatedDeltaRule` 融合算子（在 `aic-ascend910_93-ops-info-transformer.json` 有 op_info，`arch22` 有编译好的 kernel `.o`），而非自己封装自定义 GE 算子。

**落地三件套**：

1. **`cann_lm/models/fused_ops/chunk_gated_delta_rule.py`** — 注册 `npu::npu_chunk_gated_delta_rule`（schema + fake meta + CPU eager impl）：
   - schema: `(Tensor query, Tensor key, Tensor value, *, Tensor? beta=None, Tensor? initial_state=None, Tensor? actual_seq_lengths=None, float? scale=None, Tensor? g=None) -> (Tensor out, Tensor final_state)`
   - fake meta: `out = query.new_empty(query.shape[:-1] + (value.shape[-1],))`，`final_state = initial_state.new_empty(initial_state.shape)`
   - CPU impl: 转换 BHSD→BSHD 调纯 torch `torch_chunk_gated_delta_rule`（做数值回归用）

2. **`cann_lm/converter/export/ge_converter.py`** — 手动创建 `ge.ChunkGatedDeltaRule` op（torchair auto_generated 未生成此算子，torchair 自带的 `npu_chunk_gated_delta_rule` converter 直接 `raise RuntimeError`，此处覆盖）：
   - CANN op_info: `aic-ascend910_93-ops-info-transformer.json "ChunkGatedDeltaRule"`
   - inputs: query/key/value/beta/initial_state/actual_seq_lengths + g(optional)
   - outputs: out/final_state，attr: `scale_value`（float，default 1.0）

3. **`cann_lm/models/qwen3_5/modeling_qwen3_5.py:487`** — modeling 用 `torch.ops.npu.npu_chunk_gated_delta_rule(...)` 替代 `torch_chunk_gated_delta_rule`，单 GE 节点替代 19418 节点展开。

4. **`examples/export_llm_model/om_npu_verify/reg_register.h:451`** — C++ 侧注册 `REG_OP(ChunkGatedDeltaRule)`（inputs/outputs/attr）。

**效果**：`torch.export` 节点数从 19418 降到 2866，`export_to_epair` 不再 segfault，ep_json 生成，Python ep 回环验证通过。

### 4.3 OM 编译维度约定错误（未解决，新发现）

epair 导出层解决后，跑 C++ `aclgrphBuildModel` 编译 OM 时，**infershape 阶段失败**：

```
[ERROR] [chunk_gated_delta_rule_infershape.cpp:66]
  OpName:[ChunkGatedDeltaRule] value dim num should be 3, but got 4
[ERROR] InferShapeOnCompile: ... op_desc[ChunkGatedDeltaRule], ret[-1]
[ERROR] aclgrphBuildModel failed, status: 1343242270
```

> `acl.op` / `atc --singleop` 单算子实测（2026-08-09）补充确认：
> - 4D BHSD `[1,16,8,128]`：`atc --singleop` 编译失败，报 tiling 阶段 "Template constraint"
> - 3D THD `[8,16,128]`（T=1×8）：`ATC run success` 编译通过
>
> 即图模式 infershape pass 与单算子 tiling 阶段**双重拒绝 4D**，只是报错点不同（图模式 infershape 更严格先报；单算子模式到 tiling 才失败）。

#### 根因

**维度约定错误**：CANN `ChunkGatedDeltaRule` 期望 query/key/value 是 **3D THD 布局**，但 modeling 传的是 **4D BHSD**。

**CANN 期望（THD 3D）**——四重确认：
1. **官方 IR proto**（`ops-transformer/attention/chunk_gated_delta_rule/op_graph/chunk_gated_delta_rule_proto.h`）注释明确：
   ```c
   .INPUT(query, ...)   // A 3D tensor. bfloat16
   .INPUT(key, ...)     // A 3D tensor. bfloat16
   .INPUT(value, ...)  // A 3D tensor. bfloat16
   .INPUT(beta, ...)   // A tensor. bfloat16
   .INPUT(initial_state, ...)  // A tensor. (DT_BF16, DT_FLOAT)  ← 新增 FP32 state
   .INPUT(actual_seq_lengths, ...)  // int32
   .OPTIONAL_INPUT(g, ...)  // float32
   .OUTPUT(out, ...)        // bfloat16
   .OUTPUT(final_state, ...)  // (DT_BF16, DT_FLOAT)
   .ATTR(scale_value, Float, 1.0)
   ```
2. **同系列 `npu_recurrent_gated_delta_rule` 文档**（`torch_npu/_op_plugin_docs.py`）明确维度约定：
   - query/key: `(T, Nk, Dk)` 3D，value: `(T, Nv, Dv)` 3D，beta/g: `(T, Nv)` 2D，state: `(BlockNum, Nv, Dv, Dk)` 4D，actual_seq_lengths: `(B,)`
   - **T = Σ batch 序列长度**（batch 拼接，无 batch 维）
3. **NPU 实测**（`npu_recurrent_gated_delta_rule` eager 接口，2026-08-09）传入 THD 3D 跑通，out shape `(6, 8, 128) = (T, Nv, Dv)`，T=6=2×3（bs×mtp）。
4. **`atc --singleop` 单算子编译实测**（2026-08-09）：
   - 4D BHSD `[1,16,8,128]`：编译失败（tiling "Template constraint"）
   - 3D THD `[8,16,128]`（T=1×8）：`ATC run success` 编译通过
   - 这是 chunk 算子本身的实测（非跨算子推断），直接确认算子要 3D THD
5. **`chunk_gated_delta_rule_tiling.cpp` 源码**（`ops-transformer` 仓库，2026-08-09 核对）——维度约束的**代码级确认**：
   ```cpp
   const size_t QKV_DIM_NUM = 3;                    // query/key/value 3D
   const size_t BETA_DIM_NUM = 2;                  // beta 2D
   const size_t STATE_DIM_NUM = 4;                 // initial_state 4D
   const size_t ACTUAL_SEQ_LENGTHS_DIM_NUM = 1;   // actual_seq_lengths 1D
   const size_t G_DIM_NUM = 2;                     // g 2D
   ```
   `AnalyzeShapes()` 用 `CheckDim(value, QKV_DIM_NUM=3, "value")` 逐个校验 rank；`CheckExpectedShapes` 期望 shape 从锚点解析：
   ```cpp
   t = queryShape[0]; nk = queryShape[1]; dk = queryShape[2];  // query (T,Nk,Dk)
   nv = valueShape[1]; dv = valueShape[2];                      // value (T,Nv,Dv)
   b  = stateShape[0];                                          // initial_state (B,Nv,Dv,Dk)
   // expectBeta=(T,Nv) 2D, expectOut=(T,Nv,Dv) 3D, expectG=(T,Nv) 2D
   ```
   - `CheckDerivedDimConstraints` 附加约束：`nk<=64`, `nv<=64`, `dk<=128`, `dv<=128`, `nv%nk==0`（GQA），`T/B/Nk/Dk/Nv/Dv>0`
   - `AnalyzeDtype`：`initial_state`/`final_state` 的 `DT_FLOAT`（FP32 state）**仅 Ascend950 支持**，非 950 报错。本机 Ascend910_9362 不支持 FP32 state，只能 bfloat16。
   - qwen3.5 维度（nv=16, nk=4, dk=128, dv=128）满足全部约束。

> 注：infershape.cpp（OM 编译图模式 infershape pass 报错点）源码 gitcode 渲染异常未取到，但其报错 `value dim num should be 3` 与 tiling.cpp 的 `CheckDim(value, 3)` 逻辑完全对应（两个阶段都校验维度数，文案略异：infershape 旧 "dim num"，tiling 新 "number of dimensions"）。

**modeling 实际传（BHSD 4D）**（`modeling_qwen3_5.py:467-491`）：
- query/key/value: `[B, H, seq, D]` 4D（BHSD，经 `transpose(1,2)` 从 BSHD 转来）
- beta/g: `[B, H, seq]` 3D
- initial_state: `[B, Nv, Dv, Dk]` 4D（这个对了）

`chunk_gated_delta_rule.py` 的注释也错误声称"NPU 算子签名 BHSD"，是当初实现时的错误假设。infershape 检查 value 维度数，4D ≠ 3D，报错。

#### 为什么 epair 导出层没暴露

`torch.export` 不校验 CANN infershape（fake meta 只声明 shape）；`export_to_epair` 的 GE 转换不调 infershape（只构造 op 节点）；真正调 CANN infershape 的是 C++ `aclgrphBuildModel`。**之前从未跑过 OM 编译**，故 bug 潜伏至今。

#### 修复方向

**改 modeling 传 THD 3D + converter 直接透传**（参考 torch_npu 的 `npu_recurrent_gated_delta_rule` converter 设计——它用 `ge.RecurrentGatedDeltaRule(...)` 直接透传 query/key/value，**不做任何维度转换**，因 eager 和 GE 共用同一 THD 约定）：

| 文件 | 改动 |
|------|------|
| `modeling_qwen3_5.py` | 调 npu 算子时把 query/key/value 从 BHSD reshape 成 THD `(T,N,D)`，beta/g 从 BHS reshape 成 TH `(T,N)`，initial_state 保持 4D |
| `chunk_gated_delta_rule.py` | schema/meta/CPU impl 全改 THD 约定（CPU impl 做 THD↔BSHD 调 `torch_chunk_gated_delta_rule`） |
| `ge_converter.py` | 恢复直接透传（去掉 BHSD↔THD 转换），与 recurrent converter 设计一致 |

**为何不在 converter 里做转换**：曾尝试在 `ge_converter` 加 `ge.Reshape`+`ge.TransposeD` 做 BHSD→THD 转换，但 `ge.Const(torch.tensor(...))`（构造 Reshape 的 shape tensor）在 torchair stub 环境卡死（TransposeD 能过，卡在 Const 构造）。torch_npu 的 recurrent converter 不做转换正是正确设计——让 modeling 传 THD，converter 透传，根本不需要转换节点。

#### 额外发现

- `reg_register.h` 的 `initial_state`/`final_state` 只注册了 `DT_BF16`，但 IR proto 最新版支持 `DT_FLOAT`（"FP32 state" 提交）。当前 bfloat16 不影响，后续 FP32 state 要补。
- fusion switch 配置文件用旧格式 `{"Switch":{...}}` 被新 CANN 9.2.0 拒绝（报 "not on or off"），导致 `aclgrphBuildInitialize` 失败。这是独立的**环境配置问题**（与维度无关），改 cfg 格式为 `{"UBFusion":{"AutomaticUbFusion":"off"}}`（去掉外层 `Switch`）即可。

### 4.4 delta_cache tensor 化（全链改动，2026-08-09）

维度修复（[4.3](#43-om-编译维度约定错误已修复)）让 prefill OM 编译通过，但 **decode 图导出/编译**暴露 delta_cache 对象在 export 路径的新问题，最终演化为 prefill+decode 双图统一 tensor 化。

#### 问题链

1. **decode 图导出**：`delta_cache`（`SimpleDeltaNetCache` Python 对象）的 `conv_states`/`recurrent_states` tensor 被 torch.export 当**常量 lift**，但 modeling 的 `update_conv_state`/`update_recurrent_state`（dict 赋值 mutate）+ `torch_causal_conv1d_update` 的 `conv_state.copy_()`（in-place mutate）触发 `RuntimeError: Constant ... is mutated in the forward method. Pls register it as buffer`。
2. **wrapper 层构造 SimpleDeltaNetCache + dict 赋值**（`delta_cache.layers[i]["conv_states"] = conv_flat[idx]`）被 trace 成 `aten.copy` 节点，C++ `GetGEGraph` 的 NoOp converter 检查 input/output tensor meta mismatch → 失败。
3. **decode 的 conv1d weight** 经 `squeeze(1).unsqueeze(1)`（torch.export 生成 unsqueeze 节点，desc 空），`_convert_conv1d` 拿不到 weight shape → Conv2D infershape 失败。
4. **prefill export 路径**的 `F.pad` 产生 `aten.pad` 节点（无 GE converter）→ 失败。

#### 解法：modeling 直接用 tensor，不经过 SimpleDeltaNetCache dict 对象

全链改动让 delta_cache 的 conv_state/recurrent_state 作为**图输入**（展平 tensor），modeling 直接用 tensor（不构造 dict 对象），函数式返回 updated（不 mutate）：

| 文件 | 改动 |
|------|------|
| `modeling_qwen3_5.py` `torch_causal_conv1d_update` | 去 in-place `conv_state.copy_()`，函数式返回 `(out, new_conv_state)` |
| `modeling_qwen3_5.py` `GatedDeltaNet.forward` | 加 `conv_state`/`recurrent_state` tensor 参数（export 路径，不经过 SimpleDeltaNetCache dict）；`new_conv_state` 用 slice 替代 `F.pad`（避免 `aten.pad`）；返回 `(output, new_conv_state, last_recurrent_state)`；eager 路径（cache_params）保留 mutate |
| `modeling_qwen3_5.py` `DecoderLayer.forward` | `delta_cache` → `delta_states`（tuple\|SimpleDeltaNetCache\|None），传递 + 返回 tuple |
| `modeling_qwen3_5.py` `TextModel.forward` | `delta_cache` → `delta_states`，层循环按 linear 层索引分发 tuple，收集 new_conv/new_rec list 返回 |
| `modeling_qwen3_5.py` `ForCausalLM.forward` + `generate` | `delta_cache` → `delta_states`/`delta_out`（eager 兼容） |
| `export_wrapper.py` 基类 `forward` + `_drive_layers` | 加 `delta_states` 参数，返回 `*delta_out_flat` |
| `qwen3_5/export_wrapper.py` `_drive_layers` | 不构造 SimpleDeltaNetCache，直接传 `delta_states` tuple；eager/prefill 传 `_delta_cache_box`；override `get_sample_inputs` 构造零 delta_states（prefill 图输入）；`new_conv_state` 用 slice 替代 `F.pad` |
| `qwen3/export_wrapper.py` `_drive_layers` | 返回加空 `[]`（兼容基类签名） |

#### prefill + decode 双图 delta tensor 流

```
prefill 图:  delta_states(零, 36 输入: 18 conv + 18 recurrent) → updated delta(36 输出)
                ↓ host 持久化
decode 图:   delta_states(prefill 输出, 36 输入) → updated delta(36 输出)
                ↓ host 持久化, 下次 decode 输入
```

prefill 和 decode 的 delta 完全是图 I/O（不依赖 `_delta_cache_box` Python 副作用），**C++ 运行 prefill+decode 分离 OM 可直接用 delta tensor 流转**。这对应 `converter_design.md` 遗留任务 5 + 全量版难点的核心实现。

#### 验证

| 验证 | 结果 |
|------|------|
| prefill 图 OM 编译（含 delta_states） | ✅ `qwen3_5.om` 2.03GB |
| decode 图 OM 编译 | ✅ `qwen3_5_decode.om` 2.03GB |
| ep 回环 vs wrapper（EP==WR） | ✅ True |
| vs HF（HF==EP==WR） | ✅ True，`[PASS] ep 回环 vs wrapper 一致 + HF 一致` |
| eager 兼容（model.generate 路径） | ✅（`update_conv_state` 条件 `if cache_params is not None` 恢复，`_has_state` 正常流转） |

---

## 5. scatter-write 改造（遗留任务 5）

当前方案 B（预分配 buffer + `index_copy` scatter-write）仅实现 gemma4。qwen3.5 的 attention 层同样需要从 `torch.cat` 改为 `index_copy` + 预分配 buffer。

**基类 `ExportWrapper.get_sample_inputs` 的 mask 已统一为 scatter-write 语义**（causal-over-buffer `[kv_max]`），故 qwen3.5 在完成 scatter 改造前，走 CLI（`get_sample_inputs` 传预分配 KV buffer + scatter mask）导出会因 attn K 维度（`kv_max + seq_len`）与 mask（`kv_max`）不匹配报 broadcast 错误。

### qwen3.5 的特殊难点

qwen3.5 是混合架构（6 full + 18 linear）：
- **full_attention 层**：标准 GQA，可按 gemma4 思路改 `torch.cat` → `index_copy`
- **linear_attention 层**：`recurrent_state` 不是 KV buffer 格式，需要**单独设计 recurrent buffer 的增量协议**

### 绕过（临时）

`examples/export_llm_model/om_npu_verify/export_llm_model_epair.py` 用 `_build_simple_sample_inputs`（`past=None`，模型内部自建 cache，K=seq_len）绕过 scatter mask 不匹配问题，可在 scatter 改造完成前导出单图验证精度（非全量静态图，无法覆盖 prefill+decode）。

---

## 6. 全量版难点

 ExportWrapper 轻量版（当前里程碑）子类内部调原生 `forward`（带 `if past is None` 分支）。**全量版**（后续）需各 attention 加 scatter-write 路径，一张静态图覆盖 prefill+decode。

qwen3.5 的全量难点：

> 线性注意力（GatedDeltaNet）的 cache 是 `[num_v_heads, k_head_dim, v_head_dim]` 的 **recurrent state**（delta rule 增量更新），不是 `[kv_cache_max_len, ...]` 的 KV buffer——需单独设计 recurrent buffer 的增量协议。

这是 qwen3.5 全量版最核心的待设计点，与 [第 4 节](#4-线性注意力导出核心问题)的融合算子维度修复、[第 5 节](#5-scatter-write-改造遗留任务-5)的 scatter-write 改造相互关联。

---

## 7. 精度验证

### 验证结果（cann_lm 自定义模型 vs HF transformers）

| 模型 | logits max_diff (vs HF) | tokens match | 层数 | 架构特点 |
|------|----------------------|-------------|------|---------|
| qwen3.5-0.8B | 0.019 | ✅ | 24 | 混合 (6 full + 18 linear) |

所有模型生成的 token 序列与 HF 完全一致，精度差异在 bfloat16 正常范围内。

### chunk vs 非 chunk delta 差异

qwen3.5 linear attention 的 `delta_cache` 在一次性 prefill（非 chunk）时 recurrent state 处理与 HF 不一致；**分 chunk 更接近 HF**（数值路径不同）。符合 cann-recipes-infer 的 prefill/decode 分离设计（参考其 `modeling_qwen3_5_moe.py`）。

---

## 8. ACL 运行 OM + 数值差异调试

仿照 `om_npu_verify/README.md` 的 `run_gemma4_om.py`，写 `run_qwen3_5_om.py` 验证 qwen3.5 OM 在 NPU 的实际运行（[4.4](#44-delta_cache-tensor-化全链改动2026-08-09) 后）。本节记录 2026-08-09 的完整调试链：从 nan 定位到 pipeline 正确性验证，到 scale 根因确认，再到全模型差异归属修正。

### 8.1 现状总表

| 步骤 | 结果 |
|------|------|
| OM load + acl execute | ✅ 成功（89 输入 49 输出，flatten 正确：input_ids/mask/pos/past(24对)/new_kv/output_pos/delta(36)） |
| OM logits（random input_ids） | ❌ 全 nan（`has_nan=True`） |
| OM logits（真实 input_ids "Hello, my name is"） | ✅ 不 nan（`has_nan=False`，top5 正常数值） |
| OM（NPU）vs EP（CPU，相同真实输入） | ❌ maxdiff=15.78，top5_overlap=0/5 |
| EP（CPU）vs HF | ✅ maxdiff=0.019（cann_lm 逻辑正确） |
| 单算子 ChunkGatedDeltaRule OM vs CPU（scale 修复后） | ✅ maxdiff=0.006（算子本身一致） |

### 8.2 nan 根因：random input

**nan 和输入有关**：random input_ids（`torch.randint`）导致 OM logits 全 nan；真实 input_ids（"Hello, my name is" tokenize）不 nan。

- ep.module()（CPU）跑 random input 不 nan（verify PASS）—— cann_lm 图逻辑对 random 容忍
- OM（NPU）跑 random input nan —— NPU 算子对 random 输入（可能含特殊 token id）的处理异常

推测：random id 可能命中特殊 token（pad/eos/unk），NPU 算子（embedding 查表或后续）对特殊 token 的处理产生 nan；或 NPU 算子对 random 数值范围的 bf16 计算溢出。**后续调试均用真实 input（不 nan），聚焦数值差异**。

### 8.3 OM vs EP 对比：确认 OM pipeline 有问题

**方法**：venv 跑 `ep.module()`（CPU，chunk 走 CPU impl）存 logits npy（`dump_ep_logits.py`），系统 Python 跑 OM（NPU acl）读 npy 对比（`run_qwen3_5_om.py`，相同 input_ids + `get_sample_inputs` 确定性构造）。

| 对比 | maxdiff | top5_overlap | 结论 |
|---|---|---|---|
| EP vs HF | 0.019 | — | ✅ cann_lm 逻辑正确 |
| **OM vs EP**（相同输入） | **15.78** | 0/5 | ❌ **OM pipeline 有问题** |
| OM vs HF | 15.75 | 0/5 | ❌（和 OM vs EP 几乎相同） |

→ `input_ids match: True`，输入一致。差异不在 cann_lm 逻辑（EP 对），而在 **OM NPU 执行 vs EP CPU 执行**。

### 8.4 ChunkGatedDeltaRule 单算子对比：scale 根因确认

**方法**：系统 Python 跑单算子 OM（acl，`debug_chunk_om.py`）存输入+输出 npy，venv 跑 CPU impl（`cpu_chunk_compare.py`，`npu_chunk_gated_delta_rule` cpu impl = `torch_chunk_gated_delta_rule`）读 npy 对比（相同输入）。

**原始（CPU `apply_scale=True`，scale query）**：

| 输出 | OM vs CPU maxdiff | 比值 mean | 结论 |
|---|---|---|---|
| out | 0.762 | 11.34 ≈ sqrt(128) | ❌ OM 值约 CPU 的 11 倍 |
| final_state | 0.005 | — | ✅ 接近 |

```
out OM[:5]  = [-0.00522, 0.00230, 0.0253, -0.0141, 0.00983]   (≈ CPU × 11.3)
out CPU[:5] = [-0.000460, 0.000203, 0.00223, -0.00124, 0.000866]
```

**根因定位**：
- CPU `torch_chunk_gated_delta_rule`（`modeling_qwen3_5.py:212`）：`scale = 1/(query.shape[-1]**0.5); query = query * scale` —— CPU 内部缩放 query
- NPU `ge_converter.py:113`：`op.attr["scale_value"].f = scale if scale is not None else 1.0` —— modeling 未传 scale，默认 1.0，kernel 不缩放 query
- kernel 源码（`arch22/chunk_gated_delta_rule_stage1.h:676`）：`Muls(qUbFloat_, qUbFloatCon_, scale_, ...)` —— 确实用 scale 缩放 query
- 比值 mean=11.34 ≈ sqrt(D_k)=sqrt(128)=11.314 —— 数值吻合

**验证（CPU 改 `apply_scale=False`，不 scale query）**：

| 状态 | out maxdiff | 比值 mean |
|---|---|---|
| apply_scale=True（CPU scale query） | 0.762 | 11.34 |
| **apply_scale=False（不 scale）** | **0.006** | **1.00** |

→ **[PASS] OM == CPU**。scale 确实是 chunk 算子的根因。

> **注**：曾尝试两种 scale 修复均导致 OM nan：
> - 传 `scale=1/sqrt(D)`（scale_value=0.0884）→ nan
> - 后乘 `Mul(out, 1/sqrt(D))` → nan
>
> 两者均触发 kernel/图优化问题，未深究。当前保留 `apply_scale=False`（chunk 算子本身一致），全模型差异另查（见 8.5）。

### 8.5 全模型差异归属修正：非 chunk

scale 修复让单算子 chunk 一致（maxdiff 0.006），但**全模型 OM vs EP 仍 15.78**（没变）：

- EP logits[:5] 仅 13.0625→13.125 微变（apply_scale True→False），说明 **chunk 的 scale 差异被后续 RMSNorm 吸收**，对最终 logits 影响微小
- 之前"chunk 11 倍导致全模型 15.78"的推断**错误**——chunk scale 被 norm 吸收
- **全模型 15.78 来自别的算子**（非 chunk）

### 8.6 initial_state 未生效（独立问题）

单算子 OM 实测（`debug_chunk_om.py`，零/非零 initial_state）：**都不 nan**，排除 ChunkGatedDeltaRule 产生 nan。

但发现：零/非零 initial_state 输出**逐位相同**（`out[0,0,:5]` 和 `final_state[0,0,0,:3]` 完全一致），说明 initial_state input 可能没生效（kernel 未读 init，或被 zero 覆盖）。独立问题，待查证。

### 8.7 逐算子定位全模型 15.78（2026-08-09）

chunk 已排除（scale 修复后一致），全模型 15.78 来自别的算子。用**逐层 mini model 二分**定位差异起点层。

#### 逐层 mini model 二分

构造截断前 N 层的 mini model（`mini_model_test.py` export + EP dump，`run_mini_om.py` OM run 对比）：

| N（层数） | OM vs EP maxdiff | 层类型（0..N-1） |
|---|---|---|
| 24（全） | 15.78 | 6 full + 18 linear |
| 12 | 9.82 | 前 12 层 |
| 6 | 11.70 | 前 6 层 |
| 3 | 9.66 | 0,1,2 linear |
| **1** | **6.05** | layer 0 linear |

→ 差异在第 1 层（layer 0, GatedDeltaNet）就有 6.05，不随层单调累积（截断后 norm+lm_head 放大效应）。

#### 逐算子单测（layer 0）

对 layer 0 的各算子，venv dump 输入/weight/输出，系统 Python NPU eager 跑对比：

| 算子 | NPU eager vs CPU maxdiff | 结论 |
|---|---|---|
| **ChunkGatedDeltaRule** | 0.006（apply_scale=False） | ✅ 排除（scale 根因已确认，被 norm 吸收） |
| **Conv2D depthwise**（conv1d, groups=6144） | 0.0 | ✅ 排除 |
| **RMSNorm**（gdn_norm + final_norm） | 0.016 / 0.0 | ✅ 排除 |
| **MatMul**（in_qkv/in_z/out_proj/lm_head） | 0.031 / 0.031 / 0.004 / 0.125 | ✅ 排除（bf16 正常精度） |

→ **所有单算子 NPU eager vs CPU 差异都小（< 0.125）**，但 mini1 OM vs EP 6.05。

#### 关 AUB 实验

试关 `AutomaticBufferFusionOp`（`GE_FUSION_SWITCH_FILE` + `{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}`）编译 mini1 OM：

- 关 AUB 后 mini1 OM **nan**（has_nan=True）
- 有 AUB 时 mini1 OM maxdiff=6.05（不 nan）

→ AUB 不是差异源（关 AUB 反而 nan，AUB 融合帮助不 nan）。

#### 当前结论

mini1 OM vs EP 6.05 差异**不在单算子 NPU eager 实现**（都 < 0.125），而在**图编译层面**：

1. **图算子 ≠ eager 实现**：OM 用 CANN 编译后的 GE kernel，NPU eager 用 torch_npu 的 eager 实现，两者可能不同（单测 eager 一致不代表图 kernel 一致）
2. **bf16 累积路径差异**：OM 图有 Cast 32 + MatMul 9 + Mul 10，bf16 运算累积路径可能和 EP（CPU trace）不同
3. **关 AUB nan**：说明图融合对稳定性有影响，但非 6.05 差异源

#### 下一步

- 用 `atc --singleop` 编译单算子 OM（图 kernel），对比 CPU（而非 NPU eager）
- 或改 OM 加中间层输出，逐层对比 mini1 OM vs EP
- 或对比 mini1 OM 的 GE 图（ep_json）vs EP torch 图结构，找算子顺序/融合差异

#### atc --singleop 图 kernel 对比（2026-08-10）

对 mini1 的关键算子，用 `atc --singleop` 编译单算子 OM（CANN 图 kernel），NPU 跑对比 CPU：

| 算子 | 图 kernel vs CPU maxdiff | 结论 |
|---|---|---|
| MatMul（in_qkv, 2D） | 0.0625 | ✅ 排除 |
| Swish（silu） | 0.0086 | ✅ 排除 |

→ MatMul/Swish 图 kernel 和 NPU eager 一致，排除。

#### chunk context 对比（关键定位）

单算子 chunk random 输入一致（maxdiff 0.006），但 **context 输入（layer 0 实际值）不一致**：

| 测试 | OM vs EP maxdiff |
|---|---|
| 单算子 random 输入 | 0.006（一致） |
| **context 输入（layer 0）** | **0.35（不一致！）** |

```
OM out[:3]=[-1.45e-7, -8.3e-8, 2.4e-6]  (值小 ~10倍)
EP out[:3]=[-2.77e-6, -1.59e-6, 4.67e-5]
比值 mean=0.55
```

context 输入值特征（layer 0）：
- g: min=-5.98, max≈0, mean=-0.64（含接近 0 值）
- beta: 0.047~0.99
- q absmax=4.69, k absmax=1.59, v absmax=5.03

**根因定位**：ChunkGatedDeltaRule kernel 对 **context 实际输入**（g≈0 + 实际 beta/q/k/v 组合）行为和 CPU torch_chunk 不同。单算子 random（g 范围 -585~-0.08）一致，但 context（g 范围 -5.98~0）触发偏离——说明触发条件在 context 输入的值范围（疑似 g≈0 或 g_cum_exp 处理）。

**mini1 OM vs EP 6.05 差异的主要来源**：chunk kernel context 偏离 0.35，累积 18 层 + RMSNorm/lm_head 放大到 6.05。

#### 下一步（chunk kernel context 偏离）

- 深入查 chunk kernel 对 context 输入的偏离点：g_cum_exp 处理 / delta rule 公式 / beta 缩放
- 对比 kernel 源码（`arch22/chunk_gated_delta_rule_stage*.h`）和 CPU `torch_chunk_gated_delta_rule` 的中间步骤
- 构造 controlled 输入（单独变化 g/beta/q）定位哪个参数触发偏离

#### 根因确认 + 修复（2026-08-10）

参考 cann-recipes-infer `modeling_qwen3_5_moe.py`（`_forward_prefill_with_fused_kernel` line 1428-1451）发现：

**参考实现 npu_chunk 调用前手动 l2norm + 传 scale**：
```python
query = F.normalize(query, p=2, dim=-1)   # line 1428
key = F.normalize(key, p=2, dim=-1)       # line 1429
scale = 1.0 / (self.head_k_dim ** 0.5)     # line 1430
core_attn_out = torch_npu.npu_chunk_gated_delta_rule(..., scale=scale, ...)
```

**我们的 bug**：modeling 未在 npu_chunk 前 l2norm（靠 CPU impl 的 `use_qk_l2norm_in_kernel=True` 在 torch_chunk 内部做），但 **NPU kernel 无 `use_qk_l2norm` 参数，不做 l2norm**！且未传 scale（converter 默认 1.0）。导致 NPU query/key 未归一化 + 未缩放，与 CPU（l2norm + scale）偏离。

**修复**（3 处）：

1. `modeling_qwen3_5.py` npu_chunk 调用前加 l2norm + 传 scale：
```python
# 用 mean*Dk 替代 sum (ge converter 不支持 aten.sum fallback cpu; ReduceMean 支持)
_dk = self.head_k_dim
q_inv = torch.rsqrt((query_t * query_t).mean(dim=-1, keepdim=True) * _dk + 1e-6)
k_inv = torch.rsqrt((key_t * key_t).mean(dim=-1, keepdim=True) * _dk + 1e-6)
query_t = query_t * q_inv
key_t = key_t * k_inv
scale = 1.0 / (self.head_k_dim ** 0.5)
core_attn_out_t, ... = torch.ops.npu.npu_chunk_gated_delta_rule(..., scale=scale)
```
> 注：用 `mean(dim)*Dk` 替代 `sum(dim)`，因 ge converter 不支持 `aten.sum`（fallback 为 `cpu` op，C++ GetGEGraph 失败）；`ReduceMean` 支持。数学等价 `sum(x) = mean(x)*n`。

2. `chunk_gated_delta_rule.py` CPU impl 改 `apply_scale=True` + `use_qk_l2norm_in_kernel=False`（modeling 前已 l2norm，不内部重复）。

**修复后结果**（全模型 24 层，真实 input "Hello, my name is"）：

| 对比 | 修复前 | 修复后 |
|---|---|---|
| OM vs EP maxdiff | 15.78 | **0.281** |
| OM vs HF maxdiff | 15.75 | **0.375** |
| top5_overlap (OM vs EP) | 0/5 | **5/5** |
| top5_overlap (OM vs HF) | 0/5 | **5/5** |
| has_nan (真实 input) | False | False |

OM top5 = EP top5 = HF top5 = [198, 271, 0, 353, 2921] **三方一致**。

**额外发现（cann-recipes-infer 参考）**：参考实现 prefill 用 `npu_chunk_gated_delta_rule`，decode 用 `npu_recurrent_gated_delta_rule`（两个算子）。我们当前 prefill+decode 都用 `npu_chunk_gated_delta_rule`。prefill 精度已修复，decode 路径后续可考虑改用 recurrent 算子（与参考实现对齐）。

#### past KV 优化：24对 → 6对（2026-08-10）

**问题**：prefill `get_sample_inputs` 按基类构造 24 对 past KV（全层），但 linear_attention 层用 `delta_states` 不用 KV——18 linear 层的 past KV 是冗余零 buffer（不读不写，72MB 无用 IO）。

**修复**（`qwen3_5/export_wrapper.py`）：
1. `get_sample_inputs`：past KV 从基类 24 对减到 full_attn 6 对（`base_list[3] = [base_list[3][i] for i in full_idx]`）
2. `_drive_layers`：prefill+decode 统一按 `idx` 配 `full_attn_indices`（去掉 prefill 24对按 layer_idx 分支）

**效果**：prefill OM 输入 89 → 53（减 36 冗余 KV），prefill + decode past KV 结构完全一致（6 对 full_attn）。精度不变（OM vs EP 0.281，OM vs HF 0.375，top5 5/5），生成文本一致。

#### decode 精度验证（2026-08-10）

重新 export decode 图 + atc 编译 decode OM（l2norm+scale 修复后），跑 decode OM vs EP（seq_len=1, past 6对零, delta 36零, decode mask, input_ids=[9419]）：

| 阶段 | OM vs EP maxdiff | top5_overlap | has_nan |
|---|---|---|---|
| prefill | 0.281 | 5/5 | False |
| **decode** | **0.203** | **5/5** | **False** |

decode top5 = EP top5 = [30263, 95256, 96507, 177498, 132578]，delta 输出（18 conv + 18 recurrent）也不 nan。

**prefill + decode 双图精度都通过**。decode 用 chunk 算子（seq_len=1 时 chunk 内 pad 63x 计算浪费但结果正确），与 EP（CPU）一致。chunk 算子共用 prefill+decode 可行（虽 decode 效率不如 recurrent 算子）。

#### chunked prefill + mask minv 修复（2026-08-10）

实现长 prompt chunked prefill（仿 `run_gemma4_gen.py`，`run_qwen3_5_gen.py` 加 `build_prefill` 支持分块 + KV/delta 累积传递）。

**nan 根因**：mask 的 minv 用 `finfo(bf16).min`（-3.4e38），NPU SoftmaxV2 对 bf16 min + 多 attend 位置（chunk 1 attend buffer[0..8]，mask buffer[9..]=minv）溢出 nan。chunk 0（attend 少位置）不触发。

**修复**：mask minv 用 `-1e4`（足够 mask，不溢出）：
```python
# run_qwen3_5_gen.py build_prefill
minv = torch.tensor(-1e4, dtype=dtype)  # 而非 torch.finfo(dtype).min
```

**验证**：
| prompt | chunks | 生成 |
|---|---|---|
| "Hello, my name is"（5 token） | 1 | "John. I am a 20-year-old male. I am a student at a university." |
| "The capital of China is Beijing and the capital of Japan is Tokyo today"（14 token） | 2 | 连贯复述 + 续写 |

prefill+decode 双图 + chunked prefill（长 prompt 分块累积 KV/delta）全链路验证通过。

#### actual_seq_lengths 分析（2026-08-10）

尝试把 `actual_seq_lengths` 从 modeling 内部 `torch.full` 常量改为 **wrapper 图输入**（支持 chunked 变长），结论：**不可行，需回退**。

**CANN 算子约定**：THD 布局把 batch 维折叠进 T 维，`T = Σ actual_seq_lengths`：
```
query shape = [T, N, D]   ← T 维 = 所有 batch token 拼接
actual_seq_lengths = [B]  ← 每个 batch 的长度, T = Σ
```
batch=1 时 `actual_seq_lengths[0] == query.shape[0] == T`（**必须相等**）。

**tiling 源码确认**（`chunk_gated_delta_rule_tiling.cpp`）：
```cpp
t = queryShape[0];      // T 直接从 query shape 取
b = stateShape[0];      // batch 数从 state shape 取
// 约定: t == Σ actual_seq_lengths (算子用 actual_seq_lengths 做 batch 切分)
```

**尝试 [real_len] 的矛盾**：
- OM seq=8 特化 → query 固定 `[8, N, D]`（T=8，含 pad 补到 8）
- chunked 最后一块 real<8，但 query pad 到 8（T=8）
- 传 `actual_seq_lengths=[real_len]`（如 [5]）但 query `[8]` → **T=8 与 [5] 矛盾**
- 算子按 [5] 只处理前 5 token，后 3 pad 位未定义 → recurrent_state 累积异常

**实验验证**（`actual_seq_lengths=[real_len]` 图输入）：

| actual_seq_lengths | 短 prompt 生成 | 分析 |
|---|---|---|
| [8]（=T，含 pad） | "John. I am a 20-year-old male..."（和 HF 前 15 一致）✅ | 算子处理 8 token，pad 参与 chunk 但 mask 屏蔽，output_pos 取 real 位 |
| [5]（=real_len） | "John is a very good friend..."（和 HF 不一致）❌ | T=8 与 [5] 矛盾，算子切分错位 |

长 prompt（复述）生成不变——复述由 full_attn KV 驱动，actual_seq_lengths 只影响 linear_attn delta。

**结论**：
- `actual_seq_lengths[0]` **必须 = query.shape[0] = T = 8**（seq=8 特化）
- 当前 pad 到 8，actual_seq_lengths=[8]（常量）正确
- 要支持 `[real_len]`，需 query 也 `[real_len]`（不 pad）→ **动态 seq OM**（dynamic shape export），当前不支持
- **已回退** actual_seq_lengths 图输入改动，恢复为 modeling 内部 `torch.full` 常量 [8]（OM 53 输入，无 actual_seq_lengths 输入）

### 8.8 相关脚本

| 脚本 | 作用 |
|------|------|
| `dump_ep_logits.py`（venv） | 跑 `ep.module()`（CPU）存 logits npy 供 OM 对比 |
| `run_qwen3_5_om.py`（系统 Python） | 跑 OM（NPU acl），对比 OM vs EP + OM vs HF |
| `run_qwen3_5_decode_om.py`（系统 Python） | 跑 decode OM（seq=1），验证 decode 阶段 |
| `debug_chunk_om.py`（系统 Python） | 单算子 ChunkGatedDeltaRule OM 运行 + 存 npy |
| `cpu_chunk_compare.py`（venv） | 跑 CPU impl 对比单算子 OM 输出 |

### 8.9 复现

```bash
cd examples/export_llm_model/om_npu_verify
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:/home/developer/Ascend/cann-9.2.0/python.site-packages:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# OM vs EP + OM vs HF（系统 Python, NPU）
/opt/buildtools/Python-3.11.4/bin/python3 run_qwen3_5_om.py
# → OM vs EP maxdiff=1.5781e+01, OM vs HF maxdiff=1.5750e+01, MISMATCH

# EP dump（venv, CPU）
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
/mnt/workspace/torch-delegate-harmony/.venv/bin/python dump_ep_logits.py

# 单算子 chunk OM vs CPU（系统 Python 存 npy → venv 对比）
/opt/buildtools/Python-3.11.4/bin/python3 debug_chunk_om.py    # 存 chunk_om_dump.npz
/mnt/workspace/torch-delegate-harmony/.venv/bin/python cpu_chunk_compare.py  # 对比 → PASS (maxdiff 0.006)
```

### 8.10 embeddings 入图验证（2026-08-10）

把 `embeddings` 作为图输入（`input_ids` 不入图，`embed_tokens` 查表外置到 host），参考 gemma4 的 `--use-embeddings` 路径，但 qwen3.5 无 PLE（少 `per_layer_token_identity`）。

#### cann_lm 代码已支持（无需改 cann_lm）

qwen3.5 的 `Qwen3_5ExportWrapper` 已支持 `use_embeddings=True`（基类 `ExportWrapper` 统一处理）：

| 能力 | 实现位置 | 说明 |
|------|----------|------|
| `get_sample_inputs(use_embeddings=True)` | 基类 + qwen3.5 override | 基类返回 `(None, mask, pos, past, new_kv, embeddings, output_pos, *extras)`；qwen3.5 override 把 past 减到 6 对 + 追加 `None`（PLE placeholder）/`None`（scale）/`delta_states` = 10 元素 |
| `_drive_layers` 收 embeddings | qwen3.5 override | `embeddings is not None` → `kwargs["inputs_embeds"] = embeddings`（不传 input_ids），model forward 跳过 embed_tokens 查表 |
| `_build_external_extras` | 基类默认返回 `()` | qwen3.5 无 PLE，extras 空（对比 gemma4 返回 `per_layer_token_identity`） |
| `host_side_weight_names` | 基类默认 `["model.model.embed_tokens.weight"]` | tie_word_embeddings=True，detach 后 lm_head 仍留图内（storage identity 去重） |
| `detach_externalized_lookups` | 基类默认删 `embed_tokens` | torch.export 不 lift embed_tokens → weights_*.bin 不含冗余查表权重 |

#### tie_word_embeddings 处理

qwen3.5 `tie_word_embeddings=true`：`embed_tokens` 与 `lm_head` 共享 storage。`--use-embeddings` 路径 `detach_externalized_lookups` 删 `embed_tokens` 后，`lm_head` 仍留在图内（lm_head weight 经 storage identity 去重保留）。OM 图内无 embed_tokens 查表，host 验证用 CPU 模型 `model.model.embed_tokens(input_ids)` 查表算 embeddings 喂 OM。

#### 导出 + 编译 + 验证

```bash
# 1. 导出 embeddings 入图 epair（venv, CPU）
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
cd examples/export_llm_model/om_npu_verify
/mnt/workspace/torch-delegate-harmony/.venv/bin/python export_llm_model_epair.py \
  --model qwen3_5 --hf-path /mnt/workspace/data/models/Qwen3.5-0.8B \
  --seq-len 8 --dtype bfloat16 --output-dir epair_qwen3_5_emb --use-embeddings

# 2. 编译 OM（系统环境，关 AUB，cfg 格式 {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}）
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg
export SOC_VERSION=Ascend910_9362
./build/export_llm_model epair_qwen3_5_emb/ep_json_with_ascend_ir.json \
  epair_qwen3_5_emb/weight_info.json qwen3_5_emb_noaub

# 3. 验证（系统 Python, NPU acl + HF 对比）
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python.site-packages:${PYTHONPATH:-}
/opt/buildtools/Python-3.11.4/bin/python3 run_qwen3_5_om_emb.py
# → OM vs HF maxdiff=0.375, top5_overlap=5/5, MATCH
```

#### OM I/O（seq_len=8 特化，embeddings 入图）

**输入 53 个**（对比 input_ids 入图：input_ids → embeddings 替换）：

| 输入 | 形状 | dtype | 说明 |
|---|---|---|---|
| **embeddings** | [1,8,1024] | bf16 | host 查表 embed_tokens 结果（**替代 input_ids**） |
| attention_mask | [1,1,8,2048] | bf16 | causal-over-buffer |
| position_ids | [1,8] | int64 | |
| past_key_values_0_0..5_1（12） | [1,2,2048,256] | bf16 | 6 对 full_attn KV |
| new_kv_cache_pos | [8] | int64 | KV scatter 写入位 |
| output_pos | [1] | int32 | =[7] 选最后 token |
| delta_states_0_0..17（18） | [1,6144,4] | bf16 | 18 conv_state |
| delta_states_1_0..17（18） | [1,16,128,128] | bf16 | 18 recurrent_state |

**输出 49 个**：logits [1,1,248320] + 12 KV（6 对）+ 36 delta（18 conv + 18 recurrent 交替）。

#### 验证结果（真实 input "Hello, my name is"）

| OM | AUB 开 (`qwen3_5_emb.om`) | AUB 关 (`qwen3_5_emb_noaub.om`) |
|----|---|---|
| OM vs HF maxdiff | 0.375 | 0.375 |
| top5_overlap (OM vs HF) | 5/5 | 5/5 |
| has_nan | False | False |
| delta 输出 nan | False | False |
| OM top5 | [198, 271, 0, 353, 2921] | [198, 271, 0, 353, 2921] |

OM top5 = HF top5 = [198, 271, 0, 353, 2921]，**与 [8.4](#84-chunkgateddeltarule-单算子对比scale-根因确认) 的 input_ids 入图结果完全一致**——embeddings 入图与 input_ids 入图数值等价（`embeddings = embed_tokens(input_ids)`，cann_lm embed_tokens weight 来自 HF）。AUB 开/关数值一致（AUB 只融合元素算子，不改语义）。

#### 修复

- `export_llm_model_epair.py` input_0 保存：`use_embeddings=True` 时 `inp` 是 bfloat16，`numpy.tofile` 不支持 bfloat16 → 改 `arr.view(torch.int16).numpy().tofile()`（与 acl 验证脚本 `t_bytes`/`b2t` 一致）。
- fusion switch cfg 首编失败（`aclgrphBuildInitialize failed, status: 4294967295`）：疑 GE session 残留状态，重试成功。cfg 格式 `{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}` 可用（[4.3](#43-om-编译维度约定错误已修复) "额外发现"称旧格式 Switch 被拒不准，实测可用）。

#### 相关脚本

| 脚本 | 作用 |
|------|------|
| `export_llm_model_epair.py --use-embeddings`（venv） | 导出 embeddings 入图 epair |
| `run_qwen3_5_om_emb.py`（系统 Python） | host 查表 embeddings + OM (acl) run + vs HF（支持 `QWEN35_OM` env 切换 OM） |

---

## 9. npu-op-elimination 改造（2026-08-10）

参考 [`reference_npu_op_elimination.md`](reference_npu_op_elimination.md) 把 qwen3.5 导出图里 NPU 不支持/低效的算子改造成 NPU 友好形式。SKILL 跨模型适用性表标注 qwen3.5 三处"待应用"，本次落地改造 2 + 改造 3（改造 1 host mask 已应用，见[第 3 节](#3-mask-4d-透传)）。

### 改造 1：host 预构 mask（已应用）

见[第 3 节](#3-mask-4d-透传)——`_create_causal_mask` 第 691 行 `if attention_mask.dim() == 4: return attention_mask` 4D 透传，图内无 `ge.MaskedFill`。OM 算子表 `MaskedFill=0` ✓。

### 改造 2：repeat_kv reshape+cat（BroadcastTo → ConcatV2D）

`modeling_qwen3_5.py:repeat_kv` 原用 `expand`（生成 `BroadcastTo`），改 `reshape+cat+view`（kv_head-major，生成 `ConcatV2D`）：

```python
def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch * num_kv_heads, 1, slen, head_dim)
    hidden_states = torch.cat([hidden_states] * n_rep, dim=1)
    return hidden_states.view(batch, num_kv_heads * n_rep, slen, head_dim)
```

数值恒等（maxdiff=0，`expand+reshape` 与 `reshape+cat+view` 逐元素相等）。

### 改造 3：RoPE 预缓存表 + Gather（Pow/Cos/Sin → Gather）

`modeling_qwen3_5.py:Qwen3_5TextRotaryEmbedding` 原在 `forward` 动态算 cos/sin（`inv_freq = base^exp` → `freqs = inv_freq @ position_ids` → `apply_interleaved_mrope` → `emb.cos()/sin()`），生成 `Pow`/`Neg`/`Cos`/`Sin`。改为 `__init__` 预缓存 cos/sin 表 + `forward` Gather。

qwen3.5 是 **mRoPE**（multimodal RoPE，3 维 position），`TextModel.forward:790` 把 position_ids reshape 成 3D `[3,batch,seq]`（文本模式 3 维相同 via `expand`）。预缓存表按单维 position 算（3 维相同时等价于原 3D 动态算 + `apply_interleaved_mrope` 交错），`forward` 对 3D position_ids 取 `[0]` 索引：

```python
def __init__(self, config, device=None):
    ...  # inv_freq 计算保留 (init 算, 不进图)
    cos_cached, sin_cached = self._build_cos_sin_cache(self.max_seq_len_cached, device)
    self.register_buffer("cos_cached", cos_cached, persistent=False)
    self.register_buffer("sin_cached", sin_cached, persistent=False)

def _build_cos_sin_cache(self, max_seq, device):
    """预算 [max_seq, dim] 表 (文本模式: 3 维 position 相同, mRoPE 退化)."""
    pos = torch.arange(max_seq, device=device, dtype=torch.float32)[None, :]  # [1, max_seq]
    pos_3d = pos[None, ...].expand(3, 1, max_seq)  # [3, 1, max_seq]
    inv_exp = self.inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
    pos_exp = pos_3d[:, :, None, :].float()
    freqs = (inv_exp @ pos_exp).transpose(2, 3)  # [3, 1, max_seq, dim/2]
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)  # [1, max_seq, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [1, max_seq, dim]
    cos = (emb.cos() * self.attention_scaling).squeeze(0)  # [max_seq, dim]
    sin = (emb.sin() * self.attention_scaling).squeeze(0)
    return cos.to(torch.float32), sin.to(torch.float32)

def forward(self, x, position_ids):
    # 文本模式: position_ids 3D [3,batch,seq] (3 维相同) 或 2D [batch,seq]
    if position_ids.ndim == 3:
        position_ids = position_ids[0]  # 取第 0 维 (3 维相同)
    cos = self.cos_cached[position_ids]  # Gather, 无动态算子
    sin = self.sin_cached[position_ids]
    return cos.to(x.dtype), sin.to(x.dtype)
```

- `max_position_embeddings=262144`，`partial_rotary_factor=0.25` → dim=64，cos+sin 表 [262144,64]×2 float32 ≈ 128MB（`persistent=False` 不进 state_dict，但 register_buffer → 图内 Const）
- 数值恒等（maxdiff=0，2D/3D position_ids 路径均验证）

### 验证

#### eager 数值验证（恒等变换）

| 改造 | 验证 | maxdiff | 结论 |
|------|------|---------|------|
| repeat_kv expand→reshape+cat | old expand vs new cat | 0.00 | PASS |
| RotaryEmbedding 动态→预缓存+Gather (2D pos) | old 动态 vs new Gather | 0.00 | PASS |
| RotaryEmbedding 3D pos (TextModel 路径) | old 3D 动态 vs new 取[0]+Gather | 0.00 | PASS |

#### OM 算子表（关 AUB，改造前 vs 后）

| 算子 | 改造前 | 改造后 | 说明 |
|------|---:|---:|------|
| **Pow** | >0 | **0** ✓ | inv_freq=`base^exp` 消除（init 算） |
| **Cos/Sin** | 各1 | **0** ✓ | 动态 cos/sin 消除（预缓存表 Gather） |
| **BroadcastTo** | 0 | **0** ✓ | TextModel.forward position_ids `expand(4,...)` → `cat` 消除（见改造 4） |
| **ConcatV2D** | 43 | **55** | repeat_kv cat(+11) + position_ids cat(+1) |
| **GatherV2/Index** | 1/1 | 1/3 | RoPE cos/sin Gather（Index 增 2） |
| **Const** | 348 | 342 | 含 cos/sin 预缓存表（~128MB） |
| Neg | 30 | 30 | apply_rotary_pos_emb 的 `rotate_half(-x2)`，**不在改造 3 范围**（gemma4 用 `npu_rotary_mul` 融合算子消除，qwen3.5 用纯 torch apply_rotary_pos_emb 保留） |
| total ops | 2307 | 2266 | |

**改造 3 核心目标达成**：Pow/Cos/Sin 全消除。Neg(30) 来自 `apply_rotary_pos_emb.rotate_half`（`torch.cat((-x2, x1))` 的取负），不在 SKILL 改造 3 范围（SKILL 改造 3 只改 `RotaryEmbedding.forward` 的动态算子，不改 `apply_rotary_pos_emb`/`rotate_half`）。gemma4 用 `npu_rotary_mul` 融合算子替代整个 apply_rotary_pos_emb 消除 Neg，qwen3.5 未用融合算子（后续可考虑）。

### 改造 4：TextModel position_ids expand → cat（消除残留 BroadcastTo）

改造 2/3 后残留 1 个 `BroadcastTo`，来自 `TextModel.forward:794` 的 mRoPE 4 维 position_ids 构造：

```python
position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)  # [1,1,seq] -> [4,batch,seq]
```

改造 3 把 `RotaryEmbedding.forward` 简化为纯 Gather（无 expand）后，这个 `expand(4,...)` 失去融合机会，独立暴露成 BroadcastTo（改造前被 RotaryEmbedding 内部 expand 一起常量折叠）。改 `cat` 替代 `expand`（和 repeat_kv 同思路）：

```python
position_ids = position_ids.view(1, 1, -1)
position_ids = torch.cat([position_ids] * 4, dim=0)        # dim0 1->4 (text + 3 mrope, 4 维相同)
if batch_size > 1:
    position_ids = torch.cat([position_ids] * batch_size, dim=1)  # dim1 1->batch_size
```

数值等价（`expand` 与 `cat` 都是复制 4 份相同 position）。BroadcastTo 1→0 ✓，ConcatV2D 54→55。

#### OM 精度验证（embeddings 入图，真实 input "Hello, my name is"）

| 对比 | 改造前 | 改造后 |
|------|---|---|
| OM vs HF maxdiff | 0.375 | **0.195** |
| top5_overlap (OM vs HF) | 5/5 | 5/5 |
| OM top5 | [198,271,0,353,2921] | [198,271,0,353,2921] |
| has_nan | False | False |

改造后**精度提升**（0.375→0.195）：RoPE 预缓存表用 float32 算 cos/sin（init 时高精度），原动态算在 forward 里有 bfloat16 中间精度损失。top5 仍 5/5 一致。

### 文件

| 文件 | 改动 |
|------|------|
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py:repeat_kv` | 改造 2（expand→reshape+cat） |
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5TextRotaryEmbedding` | 改造 3（`__init__` 预缓存 + `_build_cos_sin_cache` + `forward` Gather，处理 3D position_ids） |
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py:Qwen3_5TextModel.forward` | 改造 4（position_ids `expand(4,...)` → `cat`，消除残留 BroadcastTo） |

---

## 文档关联

- [`converter_design.md`](./converter_design.md) — converter 全流程设计（本文档抽取自其中 qwen3.5 相关片段）
- [`compositte_op.md`](../cann_lm_models/fused_ops/compositte_op.md) — 融合算子汇总与 GE 映射分析（含 qwen3.5 线性注意力算子的 P0/P1 落地排序）
- [`examples/export_llm_model/llm_engine_on_cpu/debug_qwen3_5/README.md`](../../examples/export_llm_model/llm_engine_on_cpu/debug_qwen3_5/README.md) — qwen3.5 调试与验证脚本集
- [`examples/export_llm_model/om_npu_verify/README.md`](../../examples/export_llm_model/om_npu_verify/README.md) — gemma4 epair→OM 编译验证（qwen3.5 OM 编译复用此 C++ 框架）
