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

当前 8 个脚本:

| 脚本 | 测试场景 | 验证对象 |
|------|---------|---------|
| `verify_gemma4.py` | 单元测试 + 端到端推理 | cann_lm 模型定义 + wrapper I/O |
| `compare_hf_vs_ep.py` | HF vs EP 对比(greedy/sampling) | wrapper export + scatter 推理 |
| `verify_gemma4_chunk.py` | chunk 推理 + logits 精度对比 | chunk vs 非 chunk + vs HF logits |
| `verify_gemma4_epair.py` | ep 回环 + wrapper direct 推理 | ep.module 回环 vs wrapper |
| `verify_gemma4_wrapper_bypass.py` | wrapper 重写验证 | host mask == model 4D + bypass vs model.forward |
| `verify_gemma4_wrapper_export.py` | wrapper 可导出 + 图无 masked_fill | torch.export + FX 图 masked_fill==0 |

---

## 1. `verify_gemma4.py` — 单元测试 + 端到端推理

**测试场景**:
- **单元测试**(10 项 pytest, 已抽到 [`tests/llm_models/test_gemma4.py`](../../../../tests/llm_models/test_gemma4.py)):config / components(RMSNorm/RoPE/apply_rotary/repeat_kv/rotate_half)/ attention / MLP / decoder layer / TextModel / ForCausalLM / KV sharing / layer truncation / **wrapper vs model 单步**(max_diff=0.0)
- **端到端推理**(本脚本):text → tokenize(chat_template) → forward → generate → 文本输出(支持随机权重快测 / 真实 safetensors)

**用法**:
```bash
# 单元测试 (pytest, 随机权重, 快测 shape/pipeline):
pytest tests/llm_models/test_gemma4.py

# 真实权重 e2e 出字 (chat_template):
python verify_gemma4.py --model $GEMMA4_HF_PATH --prompt "Hello, my name is" --max-new-tokens 16

# 截断层 + 随机权重 (快测):
python verify_gemma4.py --layers 4 --prompt "Hi"
```

**参数**:
- `--prompt`:输入文本(默认 "Hello, how are you?")
- `--model`:safetensors 路径(None = 随机权重, 走 ByteTokenizer fallback)
- `--layers`/`--hidden`:随机权重的层数/隐藏维度
- `--max-new-tokens`:生成 token 数(默认 64)
- `--device`:cpu/cuda/auto(默认 auto)

**验证层次**:
1. 单元测试(45 项,含 `test_wrapper_vs_model`:wrapper 单步 forward vs 原始 model,max_diff=0.0,验证 wrapper 是薄封装不引入误差)
2. 单 token sanity check(forward pass OK)
3. 文本生成(BOS + bfloat16)

---

## 2. `compare_hf_vs_ep.py` — HF vs EP 对比

**测试场景**:
- HF `model.generate()` vs EP(`ep.module` prefill + wrapper decode)对比
- **greedy**(定位/验证 logits)或 **sampling**(确认推理效果)切换
- 验证 wrapper export + scatter 推理与 HF 一致

**用法**:
```bash
# 默认 sampling (temp=1.0, top_k=64, top_p=0.95)
python compare_hf_vs_ep.py

# greedy (定位 bug / 验证 logits 对)
# 修改脚本顶部:TEMPERATURE = 0.0, generate_hf 的 do_sample=False
```

**参数**(脚本顶部常量):
- `TEMPERATURE`/`TOP_K`/`TOP_P`:sampling 参数
- `MAX_NEW_TOKENS`:30
- `EXPORT_SEQ_LEN`:8(prefill pad 到此长度)
- `BOS_TOKEN_ID`:2(gemma4 必须)

**关键**(sliding window 修复后):
- greedy:EP == HF == CLM 完全一致(prompt1/3 验证)
- sampling:EP 不退化,第一 token 多数对,分叉后输出连贯(符合 bfloat16 tie-breaking 预期)

**调试经验**:用 greedy 验证(sampling 随机无法判断 bug),见 `skills/gemma4-debug/SKILL.md` #18。

---

## 3. `verify_gemma4_chunk.py` — chunk 推理 + logits 精度对比

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

## 4. `verify_gemma4_epair.py` — ep 回环 + wrapper direct 推理

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
- `../../../../docs/cann_lm_converter/converter_design.md`:converter 全流程设计 + 精度差异分析
