# Embedding 外置 Host 设计(纯 embeddings 模式)

> `use_embeddings=True`(CLI `--use-embeddings` 默认,CLI-002)下,`embed_tokens` 查表从图里移到 host:host 查表得 `embeddings` 作图输入(替代 `input_ids`)。图不含 `embed_tokens` 子模块 + 查表,`weights_*.bin` 不含查表权重(≈ OM);查表权重单独打包 `host_weights_*.bin`。
>
> 支持:float embed(非量化)+ int8 embed(quant,CANN 轻量化工具 `Quant_Embed_MinMax`)。gemma4 特有 PLE(`per_layer_token_identity`,float 查表)。

## 目录

- [纯 embeddings 模式流程](#纯-embeddings-模式流程use_embeddingstrue默认)
- [float embed(非量化)](#float-embed非量化)
- [int8 embed(quant)处理流程](#int8-embedquant-处理流程)
- [关键组件](#关键组件)
- [lift 概念(torch.export)](#lift-概念torchexport)
- [gemma4 PLE 特殊](#gemma4-ple-特殊per_layer_token_identity)
- [轻量化工具对 gemma4 的实现](#轻量化工具对-gemma4-的实现)
- [数据流](#数据流)
- [产物布局](#产物布局)

---

## 纯 embeddings 模式流程(use_embeddings=True,默认)

### 10 步

1. **`use_embeddings=True`**(convert `--use-embeddings` → saver)
2. **`get_sample_inputs(seq_len, use_embeddings, quant_embeddings)`**(export_wrapper.py:93-168):host 查表生成 `embeddings`,`input_ids=None`(placeholder)
3. **`host_side_weight_names()`**(基类 L182-184 / gemma4 L285-287):返回需外置的查表权重 FQN
4. **`_extract_host_side_weights(wrapper, use_embeddings)`**(saver.py:25-31 + L97):打包查表权重到 `host_side_weights`
5. **`detach_externalized_lookups()`**(saver.py:100-101 + 基类 L186-194 / gemma4 L289-295):`del` 查表子模块 → torch.export 不 lift
6. **`torch.export.export(wrapper, sample_inputs, strict=False)`**(saver.py:103):`embed_tokens` weight 不 lift(不进图输入),图输入 `embeddings`(+ scale)
7. **`_assert_externalized_not_lifted(ep, names)`**(saver.py:105-106):检查 `embed_tokens` weight 不在 lifted
8. **`export_to_epair` + `consolidated_weights_save`**(saver.py:110, L119-124):图权重 → `weights_*.bin`;查表权重 → `host_weights_*.bin`
9. **`save_model_config(use_embeddings=true)`**(saver.py:127-128):记录模式
10. **host 推理**(下游 Python):读 `model_config use_embeddings` + `host_weights` → gather → `embeddings` → 传图

---

## float embed(非量化)

`get_sample_inputs`(基类 L146-157,quant_embeddings=False):
```python
embeddings = self.model.model.embed_tokens(input_ids).detach()   # host 查表 float
return (None, attention_mask, position_ids, past_key_values, new_kv_cache_pos,
        embeddings, output_pos, *extras)
```
- 图输入:**`embeddings`(float)**——一个,无 scale
- forward(export_wrapper.py:57-60):`embeddings_scale is None` → 不走 dequant,直接用 float
- host 推理:读 `host_weights_*.bin`(`embed_tokens.weight` float)→ gather `input_ids` → float embeddings → 传图

---

## int8 embed(quant)处理流程

### 1. 轻量化工具量化 embed(Quant_Embed_MinMax)

`dopt_wrapper.build_layer_strategy`(dopt_wrapper.py:41-43):`nn.Embedding → Quant_Embed_MinMax`(W8 per-channel int8 + scale)。

轻量化工具 GPTQ 产物 `model.safetensors` 含:
- `embed_tokens.weight`(fakequant fp16)
- `embed_tokens.quant_op.weight_quantizer.s`(per-channel scale `[vocab]`)

`dopt_config.json`:`embed_tokens: {quant_strategy: "Quant_Embed_MinMax"}`

### 2. swap 抽 int8+scale(`swap_embeddings_from_dopt`,swap.py:35-56)

```python
def swap_embeddings_from_dopt(sd, model) -> dict[str, tuple[Tensor, Tensor]]:
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Embedding): continue
        wkey = f"{name}.weight"; skey = f"{name}.quant_op.weight_quantizer.s"
        if wkey not in sd or skey not in sd: continue   # 跳过无 scale(如 gemma4 PLE 未量化)
        int8, scale = materialize_int8_per_channel_symmetric(sd[wkey], sd[skey])
        out[f"model.{name}.weight"] = (int8, scale)
    return out
```
- 遍历 `nn.Embedding`,从 sd 读 fakequant weight + per-channel scale
- `materialize_int8_per_channel_symmetric` → **(int8 `[vocab,hidden]`, scale `[vocab]`)**
- 返回 `embed_quant = {fqn: (int8, scale)}`,跳过无 scale 的

### 3. saver 注入 wrapper(`set_quant_embeddings`,saver.py:82-84)

```python
quant_embeddings = bool(embed_quant)   # 非空(轻量化工具量化 embed)→ True
if quant_embeddings:
    wrapper.set_quant_embeddings(embed_quant)   # 存 self._quant_embeddings
```
`wrapper._quant_embeddings = {fqn: (int8, scale)}`(export_wrapper.py:36-38)。

### 4. get_sample_inputs(quant):host gather int8+scale(基类 L130-145)

```python
q_w, q_s = self._quant_embeddings["model.model.embed_tokens.weight"]
emb_q = q_w[input_ids]      # host gather int8 weight[input_ids] → int8 embeddings [seq,hidden]
emb_scale = q_s[input_ids]  # host gather scale[input_ids] → per-token scale [seq,1]
return (None, attention_mask, position_ids, past_key_values, new_kv_cache_pos,
        emb_q, output_pos, ple, emb_scale)
```
- **host 先 gather**:`q_w[input_ids]`(int8 按输入 id 索引)+ `q_s[input_ids]`(scale 按输入 id 索引)
- 图输入:`emb_q`(int8,per-token)+ `emb_scale`(per-token scale)——**host gather 后的值**,不是完整 weight/scale
- `input_ids=None`(placeholder)

### 5. saver pack host_weights(完整 int8+scale,saver.py:90-95)

```python
if quant_embeddings:
    host_side_weights = [
        (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
    ] + [
        (fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()
    ]
```
- 从 `embed_quant` 直接 pack **完整 int8 `[vocab,hidden]` + scale `[vocab]`**(带 `_q`/`_scale` 后缀)
- → `host_weights_*.bin`(供 host 推理时 gather)

### 6. detach + torch.export(saver.py:100-103)

```python
if use_embeddings:
    wrapper.detach_externalized_lookups()   # del embed_tokens(剥离,不 lift)
ep = torch.export.export(wrapper, sample_inputs, strict=False)
```
- `del text_model.embed_tokens` → weight 不 lift(不进 `weights_*.bin`)
- 图输入:`emb_q`(int8)+ `emb_scale`(scale)+ 其他;图含 forward dequant

### 7. forward dequant(图内,export_wrapper.py:57-60)

```python
if embeddings is not None and embeddings_scale is not None:   # quant 时都非 None
    dt = self._model_dtype
    embeddings = embeddings.to(dt) * embeddings_scale.to(dt).unsqueeze(-1)   # int8*scale → float
```
`int8 * per-token scale → float` 给 `_drive_layers`。

### 8. host_weights 打包 + model_config

- `consolidated_weights_save`:图权重 → `weights_*.bin` + `weight_info.json`;`host_side_weights`(int8 + scale,带 `_q`/`_scale`)→ `host_weights_*.bin` + `host_weights_info.json`
- `save_model_config`:`quant_embeddings: true`(下游判断图第一输入是 int8 embeddings + scale)

### 9. host 推理(下游 Python)

- 读 `model_config quant_embeddings=true` → 图第一输入是 int8 embeddings + scale
- 读 `host_weights_*.bin`(`embed_tokens.weight_q` int8 `[vocab,hidden]` + `embed_tokens.weight_scale` `[vocab]`)
- host 查表:`input_ids → embed_q gather → int8 embeddings`;`input_ids → embed_scale gather → per-token scale`
- 传 int8 embeddings + scale 给图 → 图 forward `int8*scale→float` dequant

---

## 关键组件

### `set_quant_embeddings`(export_wrapper.py:36-38)

注入 `embed_quant`(`{fqn: (int8_weight, per_channel_scale)}`)到 `wrapper._quant_embeddings`,启用 quant-embeddings 模式(host int8 查表 + forward dequant)。触发条件:`bool(embed_quant)` True(轻量化工具量化 embed 时)。

### `detach_externalized_lookups`(export_wrapper.py:186-194 / gemma4:289-295)

```python
def detach_externalized_lookups(self):
    text_model = self.model.model
    if hasattr(text_model, "embed_tokens"):
        del text_model.embed_tokens   # gemma4 还 del embed_tokens_per_layer
```
剥离查表子模块 → torch.export 不 lift 其权重 → `weights_*.bin` 不含查表权重(≈ OM)。

### `_assert_externalized_not_lifted`(saver.py:34-48 + L105-106)

```python
def _assert_externalized_not_lifted(ep, names):
    lifted = set()
    gs = ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for _placeholder, logical_name in mapping.items():
            lifted.add(logical_name)
    leaked = [n for n in names if n in lifted]
    if leaked:
        raise RuntimeError(f"detach_externalized_lookups ineffective: {leaked} still lifted. ...")
```
检查 `embed_tokens` weight 不在 `inputs_to_parameters`/`buffers`/`lifted_tensor_constants`(若在,detach 无效,raise:forward 触碰了查表子模块)。

### `_extract_host_side_weights`(saver.py:25-31)

```python
def _extract_host_side_weights(wrapper, use_embeddings):
    if not use_embeddings:
        return []
    candidate_names = wrapper.host_side_weight_names()
    sd = wrapper.state_dict()
    return [(name, sd[name].detach().cpu()) for name in candidate_names if name in sd]
```
use_embeddings 时,从 `host_side_weight_names()` + `state_dict()` 取查表权重,打包 `[(name, tensor)]`(float embed)。

### `host_side_weight_names`(基类 L182-184 / gemma4 L285-287)

- 基类:`["model.model.embed_tokens.weight"]`
- gemma4:`["model.model.embed_tokens.weight", "model.model.embed_tokens_per_layer.weight"]`(PLE 额外)

---

## lift 概念(torch.export)

**lift**:torch.export 把模型的 parameter/buffer「提升」为图的输入 placeholder(权重作为图输入,而非内建常量),让图与权重解耦。

- **普通 nn.Module**:weight 是模块属性(`module.weight`),forward 内直接用(内建常量)
- **torch.export trace 后**:weight 被 lift 成图输入 placeholder(一个 Tensor 输入参数),图内用 placeholder 引用
- **目的**:图与权重解耦——同一图可用不同权重(换权重不重新 trace)

### 表现(ExportedProgram.graph_signature)

- `inputs_to_parameters`:weight placeholder → parameter 逻辑名映射
- `inputs_to_buffers`:buffer 同理
- `inputs_to_lifted_tensor_constants`:tensor 常量

### epair/saver 上下文

| | weight 状态 | 进 weights_*.bin? |
|---|---|---|
| **lift**(子模块在模型里) | 图输入 placeholder | ✅(冗余,纯 embeddings 不该有) |
| **detach 后不 lift**(子模块剥离) | 不在图 | ❌(≈ OM,查表权重走 host_weights) |

`detach_externalized_lookups` 让 `embed_tokens` weight **不 lift**(不进 `weights_*.bin`,避免冗余);查表权重改走 `host_weights_*.bin`。`_assert_externalized_not_lifted` 兜底检查。

---

## gemma4 PLE 特殊(per_layer_token_identity)

### per_layer_token_identity 无 scale

gemma4 `_build_external_extras`(gemma4/export_wrapper.py:273-283):
```python
def _build_external_extras(self, input_ids):
    """gemma4 PLE: host 算好 per_layer_token_identity (embed_tokens_per_layer 查表结果)"""
    m = self.model.model
    token_identity = m.embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape, m.config.num_hidden_layers, m.hidden_size_per_layer_input
    ).detach()
    return (token_identity,)   # 单 tensor,无 scale
```
- `token_identity = embed_tokens_per_layer(input_ids)`(直接 float 查表,reshape + detach)
- shape `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`
- **返回单 tensor,无 scale 配对**

forward 签名(export_wrapper.py:44-55):`per_layer_token_identity` 单 tensor(无 `_scale` 参数),不像 `embeddings` 有 `embeddings_scale`。

### 与 embed_tokens 量化对比

| | embed_tokens(quant) | PLE(embed_tokens_per_layer) |
|---|---|---|
| 图输入 | `emb_q`(int8)+ `emb_scale`(scale)——**两个** | `per_layer_token_identity`(float)——**一个** |
| forward | `int8*scale→float` dequant | 直接用 float |
| host gather | `q_w[input_ids]` + `q_s[input_ids]` | `embed_tokens_per_layer(input_ids)`(float 查表) |
| host_weights | int8 + scale(带 `_q`/`_scale`) | float weight(`_extract_host_side_weights`) |

### 关键

`_build_external_extras` 用 `m.embed_tokens_per_layer(input_ids)`(**直接 float 查表**),即使 `embed_tokens` 量化(`quant_embeddings=True` 走 `emb_q`/`emb_scale` int8 路径),**PLE 仍走 float 查表**——`get_sample_inputs` quant 分支只从 `_quant_embeddings` 取 `embed_tokens` 的 int8+scale,PLE 仍用 `_build_external_extras` 的 float 结果。

即 **embed_tokens 量化(W8 int8+scale),PLE 不量化(float)**。

---

## 轻量化工具对 gemma4 的实现

### 1. embed_tokens(Gemma4TextScaledWordEmbedding)→ QGemma4ScaledEmbedding

**do_opt.py:22-26**(visible_type_dict):
```python
visible_type_dict = {
    torch.nn.Embedding: QEmbedding,
    torch.nn.Linear: QLinear,
    "Gemma4TextScaledWordEmbedding": QGemma4ScaledEmbedding,
}
```
`get_target_op_cls`(L29-43):优先 `type(m)`,然后 `m.__class__.__name__`(匹配 "Gemma4TextScaledWordEmbedding")。

**qembed.py:67-106**(QGemma4ScaledEmbedding):
- weight per-channel int8 量化(强制 `per_channel=True`,L80-82)
- **`embed_scale`**(gemma4 特有):`m.embed_scale` 或 `scalar_embed_scale`(L94-98)——gemma4 scaled embed = `embed_tokens * embed_scale`
- `tie_lmhead`(可选,L88-92)
- 即 gemma4 embed_tokens 走 **QGemma4ScaledEmbedding**(int8 + embed_scale),不是普通 QEmbedding

### 2. PLE `per_layer_model_projection`(Linear)→ GPTQ int4

**ptq_loop.py:267-297**:
```python
ple_proj_key = 'model.per_layer_model_projection'
ple_weight_config = _get_weight_config(args.quant_config, ple_proj_key)
if ple_weight_config is not None:
    ple_proj_layer = model.model.per_layer_model_projection.to(dev)
    ple_proj_quant = WeightQuantHessian(ple_proj_layer)   # GPTQ Hessian
    ple_proj_quant.quantizer.configure(wbits, perchannel=True, sym=args.sym, ...)
    for ids_cpu in input_ids_list:
        inputs_embeds = model.model.embed_tokens(ids_cpu.to(dev))
        out = ple_proj_layer(inputs_embeds)   # projection forward
        ple_proj_quant.add_batch(inputs_embeds.data, out.data)   # Hessian 累积
    ple_proj_quant.fasterquant(blocksize, percdamp, groupsize, ...)   # GPTQ int4
```
- **`per_layer_model_projection`**(PLE projection Linear 层)走 **GPTQ Hessian int4**(bit/groupsize/blocksize,perchannel,sym)
- 不是 `embed_tokens_per_layer` 查表,是 projection 层

### 3. PLE per_layer_input(查表 + projection,ptq_loop.py:149-164)

```python
ple_enabled = bool(getattr(model.model, "hidden_size_per_layer_input", None))   # gemma4 PLE 标志
if ple_enabled:
    ids_cpu = input_ids_list[j]
    orig_embed = model.model.embed_tokens(ids_cpu).to(dev)   # embed_tokens 查表
    pli = model.model.get_per_layer_inputs(ids_cpu.to(dev), None)   # PLE per_layer_inputs
    pli = model.model.project_per_layer_inputs(orig_embed, pli)   # projection
    per_layer_input = pli[:, :, i, :]   # 每层输入
```
- PLE 启用(`hidden_size_per_layer_input`)时,`embed_tokens(input_ids)` + `get_per_layer_inputs` + `project_per_layer_inputs` → per_layer_input(每层输入)
- 传给层 forward(L182:`per_layer_input=per_layer_input`)

### embed_tokens_per_layer 查表(→ per_layer_token_identity)

轻量化工具不专门量化 `embed_tokens_per_layer` 查表(只作 `get_per_layer_inputs` 输入,float)→ cann_lm `per_layer_token_identity` 也 float(无 scale)。

### 轻量化工具 gemma4 量化汇总

| 组件 | 轻量化工具处理 | 有 scale? |
|------|---------|----------|
| `embed_tokens`(Gemma4TextScaledWordEmbedding) | QGemma4ScaledEmbedding(int8 + embed_scale) | ✅(weight scale + embed_scale) |
| `per_layer_model_projection`(PLE projection Linear) | GPTQ int4 | ✅(group_size/scale) |
| `embed_tokens_per_layer`(PLE 查表) | 不专门量化(只作 `get_per_layer_inputs` 输入,float) | ❌(float 查表) |
| **per_layer_token_identity**(cann_lm) | `embed_tokens_per_layer(input_ids)` float 查表 | ❌(无 scale) |

**轻量化工具量化的是 `per_layer_model_projection`(projection Linear,int4),不是 `embed_tokens_per_layer`(查表 Embedding)**。`embed_tokens_per_layer` 在轻量化工具里只作 `get_per_layer_inputs` 的查表输入(float),不量化 → cann_lm 的 `per_layer_token_identity`(= embed_tokens_per_layer 查表)也无 scale(float)。

---

## 数据流

### float embed(非 quant)

```
input_ids (host)
  ↓ host 查表 (host_weights_*.bin: embed_tokens.weight gather)
  ↓ embeddings (float)
  ↓ 传图 (图第一输入 embeddings,替代 input_ids)
图 (forward 直接用 float,无 dequant)
```

### int8 embed(quant)

```
轻量化工具产物 (embed_tokens fakequant fp16 + scale [vocab])
  ↓ swap materialize_int8_per_channel_symmetric
int8 [vocab,hidden] + scale [vocab] (embed_quant)
  ↓ saver pack host_weights (完整)
host_weights_*.bin (int8 + scale, 带 _q/_scale 后缀)
  ↓ get_sample_inputs: host gather input_ids
图输入: emb_q (int8 per-token) + emb_scale (per-token scale)
  ↓ torch.export (detach embed_tokens, 不 lift)
图 (forward dequant: int8*scale → float)
  ↓ host 推理
host 读 host_weights → gather input_ids → int8+scale → 传图 → dequant
```

### int8 embed 位置对照

| 位置 | int8/scale 状态 | 形状 |
|------|----------------|------|
| 轻量化工具产物 | fakequant fp16 weight + scale buffer | `[vocab,hidden]` + `[vocab]` |
| swap `embed_quant` | materialize 后 int8 + scale | `[vocab,hidden]` + `[vocab]` |
| `host_weights_*.bin` | **完整** int8 + scale(带 `_q`/`_scale`) | `[vocab,hidden]` + `[vocab]` |
| 图输入(sample_inputs) | **host gather 后** per-token | `[seq,hidden]` + `[seq,1]` |
| 图 forward | dequant `int8*scale→float` | `[seq,hidden]` |

---

## 产物布局

### use_embeddings=True

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # 图 IR(无 embed_tokens 查表)
├── decode_ep_json_with_ascend_ir.json
├── weight_info.json                       # 图权重索引(不含 embed_tokens)
├── weights_0.bin                          # 图权重(Linear int4+scale + LayerNorm,不含查表权重 ≈ OM)
├── host_weights_info.json                 # host 权重索引
├── host_weights_0.bin                     # 查表权重(float embed_tokens.weight 或 int8+scale 带 _q/_scale)
├── model_config.json                     # use_embeddings=true(quant_embeddings=true if int8)
└── tokenizer files
```

### C++ 加载侧

- `GeWeightLoader` 读 `weight_info.json` + `weights_*.bin`(图权重)
- **不读 `host_weights`**——host 查表在 Python 推理脚本侧(`llm_models_run_on_cpu/debug_*` 等)
- `model_config use_embeddings` 供下游判断图第一输入是 `input_ids` 还是 `embeddings`

---

## 关键点汇总

1. **embed_tokens 不进图**(`detach_externalized_lookups` → 不 lift),查表权重在 `host_weights_*.bin`
2. **图输入是 host gather 后的 per-token**(`embeddings`/`emb_q`+`emb_scale`),不是完整 weight/scale
3. **forward dequant `int8*scale→float`**(quant 时,图内)
4. **gemma4 PLE(`per_layer_token_identity`)float 查表,无 scale**;轻量化工具量化 `per_layer_model_projection`(projection Linear int4),不量化 `embed_tokens_per_layer` 查表
5. **embed_tokens 量化**(W8 int8+scale)与 **PLE 不量化**(float)并存——embed_tokens 走 int8,PLE 走 float

## 关联

- 设计文档:[export_design.md](./export/export_design.md)「纯 embeddings 模式」「saver」「host_weights 机制」
- 代码:`cann_lm/converter/saver.py`、`cann_lm/models/export_wrapper.py`、`cann_lm/converter/swap.py`(`swap_embeddings_from_dopt`)、`cann_lm/models/gemma4/export_wrapper.py`
- 轻量化工具:`cann_lm/converter/quant/dopt_lm/do_opt.py`、`modules/qembed.py`、`opt_funcs/weight_quant_ptq/ptq_algo_hessian/ptq_loop.py`
