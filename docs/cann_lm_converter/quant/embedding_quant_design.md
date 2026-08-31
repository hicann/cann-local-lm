# Embedding 量化接入设计文档

> 纯 embeddings 模式(`--use-embeddings`)下,把外置到 host 的 embedding 查表(weight per-vocab int8 + scale)接到图里:图输入改为 `int8 + scale`,图入口 dequant 回 float,后续 RMSNorm/QuantLinear 流程不变。
>
> 量化策略:dopt `Quant_Embed_MinMax`(weight 8-bit per-channel min_max,sym signed),走 **dopt 实际数据**(非 converter 自算)。
>
> 状态: 当前采用（函数名已对齐 materialize_int8_per_channel_symmetric） | 最后核对: 2026-08-25

## 目录

- [背景与动机](#背景与动机)
- [量化契约](#量化契约)
- [dopt 产物现状(已验证)](#dopt-产物现状已验证)
- [改造点](#改造点)
- [端到端流程](#端到端流程)
- [验证方案](#验证方案)
- [gemma4 follow-up](#gemma4-follow-up)
- [风险与未决](#风险与未决)
- [与现有设计的关系](#与现有设计的关系)

---

## 背景与动机

纯 embeddings 模式([reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md) / [converter_design.md](../converter_design.md) 模式 B)已把 `embed_tokens`(及 gemma4 `embed_tokens_per_layer`)查表外置到 host:host 用 `host_weights_*.bin` 里的 **float** 权重查表 → float embeddings 喂图,图内 `embed_tokens` 经 `detach_externalized_lookups()` 不 lift。

本设计把 host 查表权重从 float 改成 **int8 + per-vocab scale**(dopt `Quant_Embed_MinMax`),host 查表后 gather int8+scale 喂图,图入口 dequant 回 float。收益:

- host_weights 从 float([vocab,hidden] fp16/bf16)→ int8+scale([vocab,hidden] int8 + [vocab] fp32),省 ~2x 存储/带宽
- 与 W4A8 全链路量化风格一致(embedding 也量化,而非例外)

> 注:[quant_design.md](./quant_design.md) 当前"量化范围:全部 nn.Linear,**除 embed_tokens 与 lm_head**"。本设计把 embed_tokens(+ gemma4 embed_tokens_per_layer)纳入量化,lm_head 改走标准 QuantLinear W4A8。

---

## 量化契约

### 数据流

```
dopt 产物: embed_tokens.weight [vocab, hidden] (fakequant float = int8_q * scale)
           + s [vocab] (per-vocab scale, sym, offset=0)
           + bit=8, offset=0 (全 0)

materialize_int8_per_channel_symmetric:
  q_signed = round(weight / s[:,None]).clamp(-128, 127).to(int8)   # [vocab, hidden] int8
  scale    = s                                                      # [vocab] fp32
  (精确重建: weight 本就是 q*s, round(weight/s)=q, maxdiff=0)

host 查表+gather:
  int8_emb [b,s,hidden] = gather(int8_weight, input_ids)
  scale     [b,s]        = gather(scale,           input_ids)

图输入:  int8 [b,s,hidden] + scale [b,s]
图入口 dequant:  emb = int8_emb.float() * scale[b,s,1]    # signed, 无 zero-point, aten cast+mul
→ float → input_layernorm(RMSNorm) → 首个 QuantLinear(DynamicQuant) → ... 完全不变
```

### 关键决策(已定)

| 项 | 决策 | 说明 |
|---|---|---|
| int8 编码 | **signed** `[-128,127]` | dequant = `int8 * scale`,无 zero-point;dopt 实测 offset=0(sym) |
| 粒度 | **per-vocab-row**(per-channel) | scale `[vocab]`,dopt `find_params(weight, perchannel=True)` 按 shape[0] |
| scale 来源 | **dopt 实际数据** | 从 dopt `Quant_Embed_MinMax` 产物 port(`.s`),非 converter 自算 min_max |
| 图入口处理 | **dequant 回 float**(方案 A) | int8+scale → float → 现有 float 流程;不引入量化 RMSNorm(CANN 无现成 int8 RMSNorm) |
| dequant 算子 | aten `to`+`mul` | GE converter 已支持,无新算子 |
| lm_head | **QuantLinear W4A8** | 从 DEFAULT_SKIP 移除,走标准 W4A8(同其他 Linear),dopt 用 `Quant_INT8-4_ECO` |
| host_weights 存储 | **swap 返回 dict 直存** | materialized int8+scale 不走 wrapper.state_dict,swap 直接返回 dict 给 saver |

### gemma4 PLE 契约(同构)

`embed_tokens_per_layer [ple_vocab, ple_embed_dim]`(dopt wrap 成 QEmbedding,因 `Gemma4TextScaledWordEmbedding` 是 `nn.Embedding` 子类)→ int8 + scale `[ple_vocab]` → host gather → int8 `[b,s,L,h_pl]`(reshape)+ scale `[b,s]` → dequant `ple_q.float()*scale[b,s,1,1]`。

---

## dopt 产物现状(已验证)

`<dopt 产物目录>/model.safetensors`(qwen3-0.6B 的 dopt 产物, `--output-dir` 指定):

```
model.embed_tokens.quant_op.weight_quantizer.bit:            [] int64 = 8
model.embed_tokens.quant_op.weight_quantizer.s:               [151936]=[vocab] fp32   ← per-vocab scale
model.embed_tokens.quant_op.weight_quantizer.max/min/offset:  [151936] fp32   (offset 全 0 = sym)
model.embed_tokens.quant_op.weight_quantizer.unsigned_quant: [] fp32 = 0.0
model.embed_tokens.weight:                                    [151936, 1024]=[vocab,hidden] fp32  ← fakequant
lm_head.quant_op.*:                                           11 keys (input_quantizer bit/max/min/offset)
embed_tokens_per_layer.*:                                     0 keys (qwen3 无 PLE)
```

实测:
- `bit=8, unsigned_quant=0, offset=0`(sym signed)
- `materialize_int8_per_channel_symmetric`: `q=round(w/s).clamp(-128,127)`,`q*s vs weight` **maxdiff=0.0000**(精确重建,因 w 本就是 fakequant = q*s)
- dopt `do_opt.py` 通过 `visible_type_dict` 按原始精确类型/类名自动选择 wrapper：`nn.Embedding → QEmbedding`，`Gemma4TextScaledWordEmbedding → QGemma4ScaledEmbedding`

---

## 改造点

### 1. `cann_lm/converter/export/quant_swap/materialize.py` — 新增

```python
def materialize_int8_per_channel_symmetric(weight_fp: torch.Tensor,
                                 scale_per_channel: torch.Tensor
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """[vocab, hidden] fp + [vocab] scale → (int8_signed [vocab,hidden], scale [vocab])。

    q = round(weight / scale[:,None]).clamp(-128, 127).to(int8); scale 原样返回。
    精确重建: dopt weight 是 fakequant = q*s, round(w/s)=q, maxdiff=0。
    """
```

### 2. `cann_lm/converter/swap.py` — 扩展

- **新增 `swap_embeddings_from_dopt(model, sd) -> dict`**:
  - 遍历 `nn.Embedding`(embed_tokens + gemma4 `embed_tokens_per_layer` 子类),读 `{name}.weight` + `{name}.quant_op.weight_quantizer.s`
  - 调 `materialize_int8_per_channel_symmetric` → 返回 `{name: (int8, scale)}` dict
  - **不 swap 成图模块**(embedding 外置给 host,非图内 matmul)——区别于 Linear→QuantLinear
- `swap_from_dopt`:
  - `DEFAULT_SKIP` 移除 `lm_head`(→ swap QuantLinear W4A8,同其他 Linear)
  - embed_tokens/embed_tokens_per_layer 仍**不**走 QuantLinear(走 `swap_embeddings_from_dopt` 返回 dict)
  - 返回值加 `embed_quant: {name: (int8, scale)}`(供 saver 存 host_weights)

### 3. `cann_lm/converter/saver.py` — host_weights 携 int8+scale

- `save_epair`/`save_epair_prefill_decode`:quant-embeddings 模式下,host_weights 用 swap 返回的 `embed_quant` dict(**int8+scale**,而非 float `embed_tokens.weight`)
- `_pack_host_side_weights_to_dir` 已通用(int8/fp32 自动落盘 + dtype/shape 记录),无需改
- `model_config.json` 记 `quant_embeddings: true` + scheme(per-vocab int8 sym, scale `[vocab]`)+ lm_head W4A8

### 4. `cann_lm/models/export_wrapper.py` + 子类 — wrapper

- `forward`:加 `embeddings`(复用, 传 int8)+`embeddings_scale`(+ gemma4 `per_layer_token_identity` 保留),基类 forward 入口反量化 int8* scale→float (非 `_drive_layers`)
- `_drive_layers`:收 float `embeddings`→ 传 `inputs_embeds`(+ gemma4 经实例方法替换传 float `per_layer_token_identity`)→ model forward 不变 (dequant 已在基类 forward 完成)
- `get_sample_inputs(use_embeddings=True, quant_embeddings=True)`:host 用 materialized int8+scale(从 swap dict 或 host_weights),gather → int8+scale sample
- `detach_externalized_lookups`:不变(仍 del embed_tokens/embed_tokens_per_layer;int8+scale 已在 host_weights)

### 5. CLI `cann_lm/converter/cann_lm_converter.py`

- embed 量化自动检测(`--quant-path` 加载 dopt 产物时, `swap_embeddings_from_dopt` 检测 embed scale 存在则自动做, 无需额外参数)
- wrapper quant-embeddings 模式 + host_weights int8+scale 由 `embed_quant` 非空触发

### 6. 图入口 dequant

aten `to`(int8→float)+ `mul`(scale `[b,s,1]` 广播)→ `cann_lm/converter/export/ge_converters/ge_converter.py` 已支持 aten.to/mul → `ge.Cast`/`ge.Mul`,**无新算子**。

---

## 端到端流程

```
cann_lm_converter.py --quant-path <dopt产物>   # embed 量化自动检测 (dopt 跑了 Quant_Embed_MinMax 则自动做)
  │
  ├─ load_model(hf_path)                    cann_lm 模型(float 权重)
  │
  ├─ swap_from_dopt(model, dopt_path)
  │   ├─ Linear (非 embed/lm_head) → QuantLinear W4A8 (int4 weight + per-group scale, 图内)
  │   ├─ lm_head → QuantLinear W4A8 (从 DEFAULT_SKIP 移除, 同其他 Linear)
  │   └─ swap_embeddings_from_dopt → {embed_tokens: (int8, scale[vocab]),
  │                                    embed_tokens_per_layer: (int8, scale[ple_vocab])}  (gemma4)
  │
  ├─ wrapper = wrapper_cls(model) + get_sample_inputs(use_embeddings=True, quant_embeddings=True)
  │   └─ host 用 int8+scale gather → int8+scale sample (embeddings(int8), embeddings_scale, [per_layer_token_identity])
  │
  ├─ _extract_host_side_weights → int8+scale (从 swap dict, 非 float)
  ├─ detach_externalized_lookups()          del embed_tokens/embed_tokens_per_layer
  ├─ torch.export → 图含: int8+scale 输入 + dequant(cast+mul) 节点 + QuantLinear(quant_matmul)
  ├─ export_to_epair → ep_json
  └─ consolidated_weights_save
      ├─ weights_*.bin: 图内权重(QuantLinear int4 + dequant 无权重)
      ├─ host_weights_*.bin: int8 embed_tokens + scale[vocab] (+ int8 embed_tokens_per_layer + scale)
      └─ model_config.json: quant_embeddings + scheme
```

**host 推理时**:
```
host 从 host_weights 读 int8 embed_tokens.weight + scale[vocab]
host: int8_emb = gather(int8_weight, input_ids)         # [b,s,hidden]
      scale     = gather(scale,          input_ids)      # [b,s]
      喂图: (int8_emb, scale, attention_mask, position_ids, kv, ...)
图入口: emb = int8_emb.float() * scale[b,s,1] → RMSNorm → QuantLinear → ...
```

---

## 验证方案(qwen3,dopt 产物已就绪)

1. **物化精确性**:`materialize_int8_per_channel_symmetric(w, s)` → `int8*scale` vs dopt fakequant `w`,maxdiff=0(已初步验证)
2. **host gather + dequant 回环**:`get_sample_inputs(quant_embeddings=True)` → `ep.module()` 输出 vs float embeddings 输出,maxdiff 应=0(dopt fakequant 精确)
3. **端到端 OM 验证 vs HF**:扩展 `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_om.py` 支持 embed 量化样本(host 用 int8+scale gather 喂图),OM-Q vs wrapper(in-graph float) vs HF。原 `verify_qwen3_epair.py` 已重构进 OM runner
4. **C++ OM 编译**:quant-embeddings epair → `examples/test_converter/export_llm_model/build/cann_lm_compile` → .om(图入口 dequant 用 aten cast+mul,应可编译)

---

## gemma4 follow-up(dopt 侧,非本设计代码)

gemma4 dopt 产物**不含** embed_tokens/embed_tokens_per_layer scale → 需重跑 dopt 加 `Quant_Embed_MinMax` 策略给两者:

- dopt `do_opt.py` 已支持按原始类型自动映射 Embedding wrapper；layer_strategy 给 `embed_tokens` + `embed_tokens_per_layer` 配 `Quant_Embed_MinMax` 即可，无需 `target_op`
- 重跑后 cann_lm 改造代码(qwen3 验证过的路径)直接复用
- PLE 的 `ple_q`+`ple_scale`:int8 `[b,s,L,h_pl]` + scale `[b,s]`,dequant `ple_q.float()*scale[b,s,1,1]`

---

## 风险与未决

1. **lm_head dopt 策略对齐**:lm_head 与其他 Linear 统一使用 `Quant_INT8-4_ECO`，对应 QuantLinear W4A8（`act_quant=True`）。
2. **tied 模型**(qwen3-0.6B `tie_word_embeddings=True`):`embed_tokens.weight == lm_head.weight` → embed 量化(int8 per-vocab for host lookup)+ lm_head 量化(int4 W4A8 for graph matmul)是**同一权重的两种量化表示**,各自独立,可行(host 用 int8+scale 查表,图内 lm_head 用 int4+group_scale matmul)。
3. **dopt embed offset=0 vs quant.py zero=128**:实测 dopt embed 是 `offset=0` signed(非 `quant.py` GPTQ 量子的 `zero=128` unsigned)。`materialize_int8_per_channel_symmetric` 用 signed `round(w/s).clamp(-128,127)` 已对齐(精确重建)。CANN/GE 侧 int8 signed 存储匹配(aten cast int8→float)。
4. **精度回归**:int8 量化误差(dopt fakequant 已含 round/clamp 误差,但 `q*s` 精确重建 dopt 产物,故 host gather + dequant = dopt fakequant 值,与"float embeddings"差异 = dopt 量化误差本身)。需 vs 原始 float embeddings(未量化)做端到端 logits 回归,确认 int8 量化误差可接受。
5. **host_weights 存 int8+scale 的机制**:swap 返回 dict 直存(已定)。需确认 `_extract_host_side_weights` / `host_side_weight_names` 在 quant-embeddings 模式下从 swap dict 取而非 wrapper.state_dict(因 detach 后 float embed_tokens 已删,且本设计用 int8+scale 不用 float)。

---

## 与现有设计的关系

- 扩展 [quant_design.md](./quant_design.md):把 embed_tokens(+ embed_tokens_per_layer)纳入量化范围,新增 `materialize_int8_per_channel_symmetric` + `swap_embeddings_from_dopt`;lm_head 从"排除"改"QuantLinear W4A8"
- 扩展 [converter_design.md](../converter_design.md) 模式 B(纯 embeddings):host_weights 从 float → int8+scale;wrapper forward/get_sample_inputs/_drive_layers 加 quant-embeddings 模式
- 复用 [reference_pipeline_walkthrough.md](../reference_pipeline_walkthrough.md) 的导出+OM 流程;图入口仅多 dequant(cast+mul)节点
- 不改 modeling(`modeling_gemma4.py` 等):forward 收 float `inputs_embeds`,无感
- 不引入新 GE 算子:dequant 用 aten cast+mul(已有 converter)
