# debug_qwen3_5 — Qwen3.5 调试与验证脚本

Qwen3.5-0.8B 模型的推理、精度验证、chunk 推理、epair 导出全流程调试脚本集。

- **模型**:Qwen3.5-0.8B(0.75B params,24 层,6 full_attention + 18 linear_attention,混合架构)
- **路径**:`/mnt/workspace/data/models/Qwen3.5-0.8B`
- **依赖环境**:torchair 源码编译 + stubs(CPU-only,无 NPU)

---

## 环境设置

运行任何脚本前必须先 source converter 环境:

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0           # 阻止 torch autoload npu backend
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0  # 不走 fallback codegen
```

- **venv**:`/mnt/workspace/torch-delegate-harmony/.venv`(torch 2.9.0+cpu, torchair 源码编译 NO_ASCEND_SDK=1, torch_delegate 0.1.0, transformers 5.14.1)
- **stubs**:必须用 torchair 编译生成的 `/mnt/workspace/torchair/build/stubs`(ABI 匹配)
- qwen3.5 有 fla(flash-linear-attention)依赖,transformers 会 fallback 到纯 torch 实现(若未装 fla + triton)
- 详见 `skills/` 下的 SKILL.md

---

## 脚本列表

当前 6 个脚本:

| 脚本 | 测试场景 | 验证对象 |
|------|---------|---------|
| `test_modeling_qwen3_5.py` | 单元测试(68 项) | cann_lm 模型定义(RMSNorm/RoPE/attention/linear_attn/delta_cache/model/generate) |
| `run_qwen3_5.py` | 端到端推理 | cann_lm 模型 + chat_template + model.generate |
| `verify_qwen3_5_vs_hf.py` | pure-torch vs HF 数值对齐 | logits max_diff(greedy token 一致) |
| `verify_wrapper_qwen3_5.py` | wrapper 自测 | Qwen3_5ExportWrapper vs 原始 model(max_diff=0.0) |
| `verify_qwen3_5_all.py` | 单步 + 多轮 decode + epair 导出 | wrapper vs model / vs HF / torch.export + ep.module 回环 |
| `verify_qwen3_5_chunk.py` | chunk 推理 + vs HF | 分 chunk prefill vs 非 chunk,验证一致性 + vs HF |
| `verify_qwen3_5_multiturn.py` | 多轮对话(cann/wrapper + chunk) | 跨轮上下文,每轮 prefill(含历史)+ decode |

---

## 1. `test_modeling_qwen3_5.py` — 单元测试

**测试场景**:
- 68 项单元测试:RMSNorm / RMSNormGated / RotaryEmbedding(mrope)/ rotate_half / apply_rotary_pos_emb / l2norm / repeat_kv / apply_mask_to_padding_states / causal_conv1d_update / torch_chunk_gated_delta_rule / GatedDeltaNet(linear attention)/ Qwen3_5Attention(full attention)/ DecoderLayer / TextModel(含 delta_cache)/ ForCausalLM / generate / inputs_embeds / 增量 decode / 错误处理

**用法**:
```bash
python test_modeling_qwen3_5.py
```

**验证结果**:✅ 68 passed, 0 failed

---

## 2. `run_qwen3_5.py` — 端到端推理

**测试场景**:
- 用 cann_lm 纯 PyTorch 实现 + `transformers.AutoTokenizer`
- `chat_template` + `model.generate`(cat 模式)+ greedy/sampling

**用法**:
```bash
# greedy
python run_qwen3_5.py --model /mnt/workspace/data/models/Qwen3.5-0.8B --prompt "你好" --greedy --max-new-tokens 16

# sampling
python run_qwen3_5.py --model /mnt/workspace/data/models/Qwen3.5-0.8B --prompt "你好，请介绍一下你自己。" --max-new-tokens 256

# 多 prompt (|| 分隔)
python run_qwen3_5.py --model /path/to/model --prompt "你好||介绍AI"
```

**参数**:
- `--model`:Qwen3.5 checkpoint 路径(必填)
- `--prompt`:输入文本(默认 "你好，请介绍一下你自己。")
- `--max-new-tokens`:默认 256
- `--temperature`:默认 0.6
- `--greedy`:greedy 解码(默认 sampling)
- `--dtype`:bfloat16(默认)/ float16 / float32

**验证结果**:✅ greedy "你好！很高兴见到你。有什么我可以帮助你的吗?"

---

## 3. `verify_qwen3_5_vs_hf.py` — pure-torch vs HF 数值对齐

**测试场景**:
- cann_lm pure-torch 模型 vs HuggingFace transformers Qwen3.5
- chat prompt + random 32-token 序列的 logits 对比(max_diff / argmax_match / top5_overlap)
- greedy 生成 8 token 对比

**用法**:
```bash
python verify_qwen3_5_vs_hf.py --model /mnt/workspace/data/models/Qwen3.5-0.8B --gen-tokens 8
```

**参数**:
- `--model`:checkpoint 路径(必填)
- `--prompt`:输入文本
- `--gen-tokens`:greedy 生成 token 数(默认 8)

**验证结果**:✅ PASS
- logits `max|Δ|=0.000e+00`(完全一致),argmax_match=100%, top5_overlap=100%
- greedy 8 tokens identical: `你好！我是 Qwen3.5`

---

## 4. `verify_wrapper_qwen3_5.py` — wrapper 自测

**测试场景**:
- `Qwen3_5ExportWrapper` 单步 forward vs 原始 model
- 验证 wrapper I/O 转换不引入误差(预期 max_diff=0.0)
- 对比 logits + KV cache(6 full_attention 层)

**用法**:
```bash
python verify_wrapper_qwen3_5.py
```

**验证结果**:✅ PASS
- logits `max_diff=0.000000e+00`
- KV cache 6 full_attention layers match(max_kv_diff=0.0)

---

## 5. `verify_qwen3_5_all.py` — 单步 + 多轮 decode + epair 导出

**测试场景**:
- Layer 1: 单步 forward 精度(wrapper vs 原始 model)
- Layer 2: 多轮 decode(3 steps,wrapper vs 原始 model)
- Layer 3: wrapper vs HF transformers
- Layer 4: epair 导出(torch.export)+ ep.module 回环 vs HF

**用法**:
```bash
python verify_qwen3_5_all.py
```

**验证结果**:✅ 全 PASS
- Layer 1: diff=0.0
- Layer 2: PASS
- Layer 3: wrapper vs HF diff=0.02(bfloat16 正常)
- Layer 4: torch.export OK(19418 节点),ep.module 回环 vs HF diff=0.02

---

## 6. `verify_qwen3_5_chunk.py` — chunk 推理 + vs HF

**测试场景**:
- chunk vs 非 chunk prefill + greedy decode
- 验证分 chunk 推理与一次性 prefill 的一致性
- 可选 vs HF(model.generate)
- qwen3.5 `_create_causal_mask` 自动用 `past_len` 算 total_len → `attention_mask=None` 即可
- `delta_cache` 跨 chunk 累积(linear attention recurrent state)

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
- `--vs-hf`:加 HF transformers 对比
- `--chunk-size`:默认 8(seq_len < chunk_size 时单 chunk,需 < seq_len 才真分 chunk)
- `--num-steps`:decode 步数(默认 10)
- `--prompt`:单 prompt(默认三 prompt)
- `--token-level`:打印 token 级详细对比

**验证结果**(`--chunk-size 4 --num-steps 8 --vs-hf`):
- prompt1/2:chunk vs 非 chunk 一致(A==B)
- prompt3:`chunk == HF`(对),非 chunk != HF(偏)
  - **发现**:qwen3.5 linear attention 的 `delta_cache` 在一次性 prefill(非 chunk)时 recurrent state 处理与 HF 不一致;**分 chunk 更接近 HF**(数值路径不同)
  - 符合 cann-recipes-infer 的 prefill/decode 分离设计(参考其 `modeling_qwen3_5_moe.py`)

---

## 7. `verify_qwen3_5_multiturn.py` — 多轮对话(cann/wrapper + chunk)

**测试场景**:
- 多轮对话(多 user 输入,跨轮上下文)
- 支持 `--mode cann`/`wrapper` + `--chunk-size`(chunk prefill)
- 每轮独立 prefill(含完整历史,重算)+ decode
- 可选 `--vs-hf`(HF 多轮 `model.generate` 对比)

**用法**:
```bash
# cann, 3 轮 (默认)
python verify_qwen3_5_multiturn.py --num-steps 16

# wrapper + chunk
python verify_qwen3_5_multiturn.py --mode wrapper --chunk-size 4 --num-steps 16

# vs HF 精度对比
python verify_qwen3_5_multiturn.py --vs-hf --num-steps 16
python verify_qwen3_5_multiturn.py --mode wrapper --chunk-size 4 --vs-hf --num-steps 16

# 自定义多轮
python verify_qwen3_5_multiturn.py --prompts "你好||刚才我说了什么？||介绍一下人工智能"
```

**参数**:
- `--mode`:`cann`(默认)/ `wrapper`
- `--chunk-size`:chunk prefill size(0=非 chunk,默认 8)
- `--num-steps`:每轮 decode token 数(默认 16)
- `--prompts`:多轮 prompt,`||` 分隔(默认 3 轮)
- `--vs-hf`:加 HF transformers 多轮对比

**验证结果**(`--vs-hf --num-steps 16`):
- cann 模式:3 轮 `cann==HF` True
- wrapper + chunk(`--mode wrapper --chunk-size 4`):3 轮 `wrapper==HF` True
- 示例:Turn2 "刚才我说了什么？" → cann/wrapper/HF 都 "刚才我并没有说任何具体的话,因为我是 Qwen3.5"

**注**:当前"每轮独立 prefill(含历史,重算)"模式 —— 上下文对(因 input_ids 含历史),但 KV/delta 不跨轮累积。真跨轮(只 prefill 新 token,past=上轮 KV/delta)需精确 token 对齐(chat_template 编码 assistant ≠ decode tokens),待后续。

---

## 待补充/未测

> ~~多轮对话~~(已补充):`verify_qwen3_5_multiturn.py` 支持 `--mode cann/wrapper` + `--chunk-size`,每轮独立 prefill(含历史)+ decode,验证多轮上下文。
>
> ~~wrapper chunk 推理~~(已解决):`Qwen3_5ExportWrapper._delta_cache_box`(list 容器)持久化 delta_cache,`verify_qwen3_5_chunk.py` 支持 `--mode wrapper`。

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
- `../debug_gemma4/README.md`:gemma4 调试流程参考(slider-window 修复、scatter-write、ONNX 导出等)
- `../../../../../docs/cann_lm_converter/converter_design.md`:converter 全流程设计 + 精度差异分析 + 遗留任务(qwen3.5 scatter/linear attention segfault)
