# debug_qwen3_5 — Qwen3.5 调试与验证脚本

Qwen3.5-0.8B 模型的推理、精度验证、chunk 推理、epair 导出全流程调试脚本集。

- **模型**:Qwen3.5-0.8B(0.75B params,24 层,6 full_attention + 18 linear_attention,混合架构)
- **路径**:`$QWEN3_5_HF_PATH`(由 check_env.sh / 基底设置)
- **依赖环境**:torchair 源码编译 + stubs(CPU-only,无 NPU)

---

## 环境设置

运行任何脚本前必须先 source converter 环境:

```bash
# source 基底 (① converter: CPU venv + torchair stubs; 或 source ../check_env.sh 一次性设好 HF 路径)
source ../../../../scripts/set_env_export.sh
export QWEN3_5_HF_PATH=<qwen3.5 model dir>   # 必填
```

- **venv**:torch 2.9.0+cpu, torchair 源码编译 NO_ASCEND_SDK=1, torch_delegate 0.1.0, transformers 5.14.1
- **stubs**:必须用 torchair 编译生成的 `build/stubs`(ABI 匹配)
- qwen3.5 有 fla(flash-linear-attention)依赖,transformers 会 fallback 到纯 torch 实现(若未装 fla + triton)
- 详见 `skills/` 下的 SKILL.md

---

## 脚本列表

当前 4 个脚本:

| 脚本 | 测试场景 | 验证对象 |
|------|---------|---------|
| `verify_qwen3_5_vs_hf.py` | pure-torch vs HF 数值对齐 | logits max_diff(greedy token 一致) |
| `verify_qwen3_5_epair.py` | epair 导出 + ep 回环推理 | ep.module prefill + wrapper decode vs wrapper direct vs HF |
| `verify_qwen3_5_chunk.py` | chunk 推理一致性 | 单轮 chunk vs 非 chunk invariant + vs HF |
| `verify_qwen3_5_multiturn.py` | 多轮对话(cann/wrapper + chunk) | 跨轮上下文,每轮 prefill(含历史)+ decode |

---

## 1. `test_modeling_qwen3_5.py` — 单元测试（已移至 tests/）

> 已按「单元测试归 tests/py/」约定移至 [`tests/py/llm_models/test_modeling_qwen3_5.py`](../../../../tests/py/llm_models/test_modeling_qwen3_5.py)（含 68 项：RMSNorm/RoPE/attention/linear_attn/delta_cache/model/generate）。用 converter venv（无 torch_npu）跑：`pytest tests/py/llm_models/test_modeling_qwen3_5.py` 或直接 `python tests/py/llm_models/test_modeling_qwen3_5.py`。

---

## 2. `verify_qwen3_5_vs_hf.py` — pure-torch vs HF 数值对齐

**测试场景**:
- cann_lm pure-torch 模型 vs HuggingFace transformers Qwen3.5
- chat prompt + random 32-token 序列的 logits 对比(max_diff / argmax_match / top5_overlap)
- greedy 生成 8 token 对比

**用法**:
```bash
python verify_qwen3_5_vs_hf.py --model $QWEN3_5_HF_PATH --gen-tokens 8
```

**参数**:
- `--model`:checkpoint 路径(必填)
- `--prompt`:输入文本
- `--gen-tokens`:greedy 生成 token 数(默认 8)

**验证结果**:✅ PASS
- logits `max|Δ|=0.000e+00`(完全一致),argmax_match=100%, top5_overlap=100%
- greedy 8 tokens identical: `你好！我是 Qwen3.5`

---

## 3. `verify_qwen3_5_epair.py` — epair 导出 + ep 回环推理

**测试场景**(2 模式):
- `--mode ep`(默认):`ep.module()` prefill(验证 logits)+ wrapper decode(ep 回环),对比 ep vs wrapper direct vs HF
- `--mode wrapper`:wrapper direct 推理(cat KV,past=None)

**qwen3.5 特殊处理**:
- **delta_cache 持久化**:ep.module() 是 trace 后的图,不执行 `_delta_cache_box[0]=...`(Python 副作用),故 ep 回环:
  1. wrapper prefill(non-ep,cat 模式):获取 delta_cache(更新 `_delta_cache_box`)
  2. ep.module prefill(scatter 模式):获取 KV_ep(验证 logits_ep == logits_w)
  3. wrapper decode:用 KV_ep + `_delta_cache_box`(delta_cache 来自 step 1)
- **形状特化**:ep.module 按 prompt 的实际 seq_len 导出(每个 seq_len 缓存一个 ep)
- **chunk_gated_delta_rule**:linear attention 用 cann_lm 融合算子(单 GE 节点,避免 19418 节点 segfault)

**用法**:
```bash
# ep 回环 vs wrapper vs HF (默认)
python verify_qwen3_5_epair.py

# wrapper direct 推理
python verify_qwen3_5_epair.py --mode wrapper

# 单 prompt
python verify_qwen3_5_epair.py --prompt "你好"

# 不对比 HF
python verify_qwen3_5_epair.py --no-vs-hf
```

**参数**:`--mode ep/wrapper` / `--vs-hf`(默认开,`--no-vs-hf` 关闭)/ `--num-steps`(10)/ `--prompt`

**验证结果**:✅ PASS(ep 回环 == wrapper == HF,三 prompt tokens 一致;ep prefill logits max_diff 0.25-0.31 bfloat16 数值差异但 tokens match)

---

## 4. `verify_qwen3_5_chunk.py` — chunk 推理一致性

单轮 chunk invariant: A=非 chunk prefill + decode vs B=chunk prefill + decode, 验证 A==B. 可选 `--vs-hf` 加 HF 三方对比.

qwen3.5 混合架构: full_attention SimpleKVCache + linear_attention delta_cache. `attention_mask=None` 时 model `_create_causal_mask` 自动用 `past_len` 算 total_len.

**用法**:
```bash
# cann, chunk vs 非 chunk (默认三 prompt)
python verify_qwen3_5_chunk.py

# 加 HF 对比
python verify_qwen3_5_chunk.py --vs-hf

# 真分 chunk (chunk_size < seq_len)
python verify_qwen3_5_chunk.py --chunk-size 4 --num-steps 8

# 单 prompt + token 级详细
python verify_qwen3_5_chunk.py --prompt "你好" --token-level
```

**参数**:
- `--mode`: `cann`(默认)/ `wrapper`
- `--chunk-size`: chunk prefill size(需 < seq_len 才真分 chunk, 默认 8)
- `--num-steps`: decode token 数(默认 10)
- `--vs-hf`: 加 HF transformers 对比
- `--prompt`: 单 prompt(默认三 prompt)
- `--token-level`: 打印 token 级详细对比

**验证结果**(`--chunk-size 4 --num-steps 8 --vs-hf`):
- prompt1/2: chunk vs 非 chunk 一致(A==B)
- prompt3: `chunk==HF`(对), 非 chunk != HF(偏)——linear attention `delta_cache` 一次性 prefill 与 HF 数值路径不同, 分 chunk 更接近 HF

---

## 5. `verify_qwen3_5_multiturn.py` — 多轮对话

多轮对话(跨轮上下文): 每轮独立 prefill(含历史, 重算) + decode. 上下文对(input_ids 含历史), KV/delta 不跨轮. 可选 `--vs-hf`.

**用法**:
```bash
# cann, 3 轮 (默认)
python verify_qwen3_5_multiturn.py --num-steps 16

# wrapper + chunk
python verify_qwen3_5_multiturn.py --mode wrapper --chunk-size 4 --num-steps 16

# vs HF 精度对比
python verify_qwen3_5_multiturn.py --vs-hf --num-steps 16

# 自定义多轮
python verify_qwen3_5_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI"
```

**参数**:
- `--mode`: `cann`(默认)/ `wrapper`
- `--chunk-size`: chunk prefill size(0=非 chunk, 默认 8)
- `--num-steps`: 每轮 decode token 数(默认 16)
- `--prompts`: 多轮 prompt, `||` 分隔(默认 3 轮)
- `--vs-hf`: 加 HF transformers 多轮对比

**验证结果**(`--vs-hf --num-steps 16`): cann/wrapper 3 轮 `==HF` True; 示例 Turn2 "刚才我说了什么？" → "刚才我并没有说任何具体的话,因为我是 Qwen3.5"

**注**: 当前"每轮独立 prefill(含历史, 重算)"模式(上下文对, 但 KV/delta 不跨轮累积). 真跨轮(只 prefill 新 token, past=上轮 KV/delta)需精确 token 对齐(chat_template 编码 assistant ≠ decode tokens), 待后续.

---

## 待补充/未测

> ~~多轮对话~~(已补充):`verify_qwen3_5_chunk_multiturn.py` `--num-turns>1` 支持 `--mode cann/wrapper` + `--chunk-size`,每轮独立 prefill(含历史)+ decode,验证多轮上下文。
>
> ~~wrapper chunk 推理~~(已解决):`Qwen3_5ExportWrapper._delta_cache_box`(list 容器)持久化 delta_cache,`verify_qwen3_5_chunk_multiturn.py` `--mode wrapper` 支持 chunk。

**注**:当前多轮用"每轮独立 prefill(含历史,重算)"模式(上下文对,但 KV/delta 不跨轮累积)。真跨轮(只 prefill 新 token,past=上轮 KV/delta)需精确 token 对齐(chat_template 编码 assistant ≠ decode tokens),待后续。

## 常见问题

### qwen3.5 特有

1. **混合架构**:6 full_attention(标准 GQA + KV cache)+ 18 linear_attention(GatedDeltaNet + recurrent state/delta_cache)
2. **scatter-write 已改造**(遗留任务 5):`SimpleKVCache.update` 支持 `index_copy`(scatter,K=kv_max 固定)+ cat fallback;`Qwen3_5ExportWrapper._drive_layers` 区分 prefill(24 对)/decode(6 对)past
3. **delta_cache 自动创建**:`Qwen3_5TextModel.forward` `use_cache=True` 时自动创建 `SimpleDeltaNetCache`(对齐 generate)
4. **chunk vs 非 chunk delta 差异**:一次性 prefill 的 `torch_chunk_gated_delta_rule` 与分 chunk 数值路径不同(bfloat16),chunk 更接近 HF
5. **export_to_epair segfault(遗留任务 3)**:linear attention `torch_chunk_gated_delta_rule` Python for 循环(63 次 + clone)→ torch.export 展开 19418 节点 → export_to_epair GE 转换图爆炸 segfault
   - 解决方向(参考 cann-recipes-infer):用 NPU fused 算子 `npu_chunk_gated_delta_rule`(prefill)+ `npu_recurrent_gated_delta_rule`(decode),或自定义 GE 算子,避免循环展开

### 通用(同 gemma4)

- **验证 logits 用 greedy 而非 sampling**:sampling 随机性无法判断 bug
- **bfloat16**:模型原始精度
- 详见 `skills/` 下的 SKILL.md

## 文档关联

- `skills/`:qwen3.5 调试经验
- `../debug_gemma4/README.md`:gemma4 调试流程参考(slider-window 修复、scatter-write 等)
- `../../../../docs/cann_lm_converter/README.md`:converter 全流程设计 + 精度差异分析 + 遗留任务(qwen3.5 scatter/linear attention segfault)
