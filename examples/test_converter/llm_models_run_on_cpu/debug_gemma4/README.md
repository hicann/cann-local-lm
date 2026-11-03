# debug_gemma4 — Gemma4 调试与验证脚本

gemma4-E2B 模型的导出、推理、精度验证、chunk 推理、epair 回环全流程调试脚本集。

- **模型**:gemma-4-E2B-it(5B params,35 层,20 层 KV 共享,sliding + full attention)
- **路径**:`$GEMMA4_HF_PATH`(由 check_env.sh / 基底设置)
- **依赖环境**:torchair 源码编译 + stubs(CPU-only,无 NPU)

---

## 环境设置

运行任何脚本前必须先 source converter 环境:

```bash
# source 基底 (① converter: CPU venv + torchair stubs; 或 source ../check_env.sh 一次性设好 HF 路径)
source ../../../../scripts/set_env_export.sh
export GEMMA4_HF_PATH=<gemma4 model dir>   # 必填
```

- **venv**:torch 2.9.0+cpu, torchair 源码编译 NO_ASCEND_SDK=1, torch_delegate 0.1.0, transformers 5.14.1
- **stubs**:必须用 torchair 编译生成的 `build/stubs`(ABI 匹配),不要用真实 CANN lib64(会 `aclrtSetDevice 107001` 无设备)
- 详见 `skills/gemma4-debug/SKILL.md`

---

## 脚本列表

当前 4 个脚本:

| 脚本 | 测试场景 | 验证对象 |
|------|---------|---------|
| `verify_gemma4_chunk.py` | chunk 推理 + logits 精度对比 | chunk vs 非 chunk + vs HF logits |
| `verify_gemma4_epair.py` | ep 回环 + wrapper direct 推理 | ep.module 回环 vs wrapper |
| `verify_gemma4_multiturn.py` | 多轮对话(cann/wrapper + chunk) | 跨轮上下文,每轮 prefill(含历史)+ decode |
| `verify_gemma4_vs_hf.py` | pure-torch vs HF 数值对齐 | logits max_diff/argmax/top5 + greedy token 一致 |

> wrapper 重写验证（host mask==model 4D + bypass vs model.forward）与可导出性（FX 图 masked_fill==0）已抽到 `tests/python/llm_models/test_gemma4.py`（GEMMA4-024/025），示例脚本 `verify_gemma4_wrapper_bypass.py` / `verify_gemma4_wrapper_export.py` 已删。
>
> `verify_gemma4_epair_quant.py` 已删（quant PLE int8 回环，需 CANN 轻量化工具 GPTQ 产物，从未实际验证过）。quant embeddings 的 sample-inputs 构造已由 `test_gemma4.py::test_wrapper_get_sample_inputs_quant_embeddings_mode`（GEMMA4-015）覆盖；完整 quant epair 回环若需复测，可在 `verify_gemma4_epair.py` 加 `--quant-embeddings` 模式（需轻量化工具产物）。
>
> `compare_hf_vs_ep.py` 已删（HF vs EP 对比，被 `verify_gemma4_epair.py --vs-hf` 覆盖：greedy 模式 token 一致是强判定，已在 epair 验证；sampling 同 seed 因 bfloat16 tie-breaking 预期分叉，非强判定）。

---

## 1. `verify_gemma4_chunk.py` — chunk 推理 + logits 精度对比

**测试场景**:
- chunk vs 非 chunk 生成对比(token 一致)
- 可选 vs HF(generate 黑盒对比)
- 可选 `--logits-diff`:逐 step logits 精度对比(HF model.language_model + lm_head + softcap 拆解,验证数值精度)
- 模式:`cann`(model.forward cat KV)/ `wrapper`(Gemma4ExportWrapper.forward)

**用法**:
```bash
# cann 模式, chunk vs 非 chunk (默认三 prompt)
python verify_gemma4_chunk.py

# wrapper 模式
python verify_gemma4_chunk.py --mode wrapper

# 加 HF 三方对比
python verify_gemma4_chunk.py --vs-hf

# 逐 step logits 精度对比 (非 chunk, 隐含 --vs-hf)
python verify_gemma4_chunk.py --logits-diff --num-steps 4

# 真分 chunk (chunk_size < seq_len)
python verify_gemma4_chunk.py --chunk-size 4 --num-steps 8

# 单 prompt + token 级详细
python verify_gemma4_chunk.py --prompt "Hello" --token-level
```

**参数**:
- `--mode`:`cann`(默认)/ `wrapper`
- `--vs-hf`:加 HF transformers `model.generate` 三方对比
- `--logits-diff`:逐 step logits max_diff/allclose(隐含 `--vs-hf`,非 chunk);**token match 即 PASS**(logits allclose 是 bfloat16 数值参考)
- `--chunk-size`:默认 8(seq_len < chunk_size 时单 chunk,需 < seq_len 才真分 chunk)
- `--num-steps`:decode 步数(默认 10)
- `--prompt`:单 prompt(默认三 prompt)
- `--token-level`:打印 token 级详细对比

**验证结果**:
- chunk vs 非 chunk:`A==B` True(chunk 推理一致)
- `--logits-diff`:逐 step token match(tok_a==tok_hf),max_diff 0.69-1.69(bfloat16 `nn.Linear` vs HF `Gemma4ClippableLinear` 预期差异)

---

## 2. `verify_gemma4_epair.py` — ep 回环 + wrapper direct 推理

**测试场景**:
- ep 模式:`ep.module()` prefill(padded 到 EXPORT_SEQ_LEN)+ wrapper decode,对比 ep 回环 vs wrapper direct
- wrapper 模式:wrapper direct 推理(不 pad,scatter-write + q_start 正确)
- greedy / sampling 切换
- 可选 vs HF

**用法**:
```bash
# ep 回环 vs wrapper, greedy (默认三 prompt)
python verify_gemma4_epair.py

# wrapper direct 推理
python verify_gemma4_epair.py --mode wrapper

# sampling 模式
python verify_gemma4_epair.py --sampling

# 加 HF 三方对比
python verify_gemma4_epair.py --vs-hf

# 单 prompt
python verify_gemma4_epair.py --prompt "Hello" --num-steps 16
```

**参数**:
- `--mode`:`ep`(默认,对比 ep 回环 vs wrapper)/ `wrapper`(direct 推理)
- `--sampling`:sampling(默认 greedy)
- `--vs-hf`:加 HF 对比
- `--num-steps`:decode 步数(默认 10)
- `--prompt`:单 prompt(默认三 prompt)
- `--temp`/`--top-k`/`--top-p`:sampling 参数

**ep 模式 mask 策略**(scatter + pad):
- prefill:pad 到 EXPORT_SEQ_LEN=8,mask causal-over-buffer + pad 行屏蔽
- decode:mask 屏蔽 pad 位置 `[real_seq_len, EXPORT_SEQ_LEN)` + 未写入 `[cached_len, kv_max)`;position_ids 跳过 pad(`real_seq_len+n-1`);buffer 写位置跳 pad(`EXPORT_SEQ_LEN+n-1`)

**验证结果**:greedy `ep==wrapper` True(回环一致)

---

## 3. `verify_gemma4_multiturn.py` — 多轮对话(cann/wrapper + chunk)

**测试场景**:
- 多轮对话(多 user 输入,跨轮上下文)
- 支持 `--mode cann`/`wrapper` + `--chunk-size`(chunk prefill)
- 每轮独立 prefill(含完整历史,重算)+ decode
- 可选 `--vs-hf`(HF 多轮 `model.generate` 对比)
- gemma4 标准 transformer,无 delta_cache;每轮重算(past=None, shared={} 重置)

**gemma4 特有**:
- BOS 由 chat_template 自动加(gemma4 必须 BOS 起始)
- cann model.forward 需手动 4D mask(causal/chunk/decode)+ shared_kv_states(PLE)
- wrapper 双 mask BOOL(full/sliding)+ new_kv_cache_pos 绝对位 + 零 KV buffer(scatter-write)

**用法**:
```bash
# cann, 3 轮 (默认)
python verify_gemma4_multiturn.py --num-steps 16

# wrapper + chunk
python verify_gemma4_multiturn.py --mode wrapper --chunk-size 4 --num-steps 16

# vs HF 精度对比
python verify_gemma4_multiturn.py --vs-hf --num-steps 16

# 自定义多轮
python verify_gemma4_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI"
```

**参数**:
- `--mode`:`cann`(默认)/ `wrapper`
- `--chunk-size`:chunk prefill size(0=非 chunk,默认 8)
- `--num-turns`:对话轮数(默认 3)
- `--num-steps`:每轮 decode token 数(默认 16)
- `--prompts`:多轮 prompt,`||` 分隔
- `--vs-hf`:加 HF transformers 多轮对比

**注**:当前"每轮独立 prefill(含历史,重算)"模式(上下文对,但 KV 不跨轮累积). 真跨轮(只 prefill 新 token,past=上轮 KV)需精确 token 对齐,待后续.

---

## 4. `verify_gemma4_vs_hf.py` — pure-torch vs HF 数值对齐

cann_local_lm Gemma4ForCausalLM vs HF Gemma4ForConditionalGeneration (text: `language_model` + `lm_head` + `softcap` 拆解), 对比:
- per-position logits max_diff / argmax_match / top5_overlap (chat prompt + random 32-token)
- prefill cache-vs-nocache invariant (mine 内部一致性)
- greedy 生成 token-id 一致

**gemma4 特有**: BOS 必须起始(chat_template 自动加; random 前置 `BOS_TOKEN_ID=2`); bfloat16 必须; cann forward 需手动 4D mask + `shared_kv_states` (PLE); HF logits 拆解 `lm_head(language_model(...).last_hidden_state)` + softcap tanh.

**用法**:
```bash
python verify_gemma4_vs_hf.py --model $GEMMA4_HF_PATH
python verify_gemma4_vs_hf.py --model /path/to/gemma-4-E2B-it --gen-tokens 8
```

**参数**:
- `--model`: gemma4 checkpoint 路径(必填)
- `--prompt`: 输入文本(默认 "Hello, how are you?")
- `--gen-tokens`: greedy 生成 token 数(默认 8)

**验证结果**: ✅ PASS (logits argmax 100%, top5 100%, greedy token 一致; bfloat16 max|Δ| 数值差异预期)

---

## 常见问题

详见 `skills/gemma4-debug/SKILL.md`(17 条调试经验 + 精度策略)。关键:

1. **必须加 BOS**(id=2):Gemma tokenizer 不自动加,不加 BOS 输出退化重复(" is is is...")
2. **必须用 bfloat16**:float16 输出质量差(重复/无意义)
3. **验证 logits 用 greedy 而非 sampling**:sampling 随机性无法判断 bug;greedy 确定性,logits 对则 token 一致
4. **sliding window decode bug**(已修复):`_prepare_causal_mask` 4D sliding 用 `new_kv_cache_pos` 算 q 位置(非 `arange(q_len)`),见 SKILL.md #16
5. **scatter-write 下 bfloat16 精度策略**:Q@K bfloat16 / softmax float32 / @v bfloat16(已验证,不要改)

## 文档关联

- `skills/gemma4-debug/SKILL.md`:gemma4 调试经验(17 条)+ 验证脚本索引 + 精度策略
- `skills/gemma4-pure-torch/SKILL.md`:纯 PyTorch 实现说明
- `../../../../docs/zh/cann_lm_converter/README.md`:converter 全流程设计 + 精度差异分析
