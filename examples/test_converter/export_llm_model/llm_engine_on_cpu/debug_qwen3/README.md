# debug_qwen3 — Qwen3 调试与验证脚本

Qwen3-0.6B 模型的推理、精度验证、chunk 推理、多轮对话、epair 导出全流程调试脚本集。

- **模型**:Qwen3-0.6B(0.596B params,28 层,标准 GQA,thinking 模式)
- **路径**:`/mnt/workspace/data/models/qwen3-0.6B`
- **依赖环境**:torchair 源码编译 + stubs(CPU-only,无 NPU)

---

## 环境设置

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
```

- **venv**:`/mnt/workspace/torch-delegate-harmony/.venv`(torch 2.9.0+cpu, torchair, torch_delegate, transformers)
- **stubs**:`/mnt/workspace/torchair/build/stubs`(ABI 匹配,勿用真实 CANN lib64)
- 详见 `skills/` 下的 SKILL.md

---

## 脚本列表

当前 5 个脚本(精简自 10 个,合并方式参考 debug_gemma4):

| 脚本 | 测试场景 | 合并自 |
|------|---------|--------|
| `verify_qwen3.py` | 单元测试 + 端到端推理 + wrapper 自测 | test_qwen3 + run_qwen3 + verify_wrapper |
| `verify_qwen3_vs_hf.py` | pure-torch/wrapper/embeddings vs HF | verify_qwen3_vs_hf + wrapper_vs_hf + wrapper_vs_hf_embeddings |
| `verify_qwen3_epair.py` | epair 导出 + 回环 + 动态 shape | verify_epair_vs_hf + export_dynamic_shape |
| `verify_qwen3_chunk.py` | chunk 推理 + vs HF | (保留) |
| `verify_qwen3_multiturn.py` | 多轮对话 vs HF | (保留) |

---

## 1. `verify_qwen3.py` — 单元测试 + 端到端推理 + wrapper 自测

**测试场景**:
- 单元测试(35 项):config / RMSNorm / RoPE / attention / MLP / decoder / model / generate / loss / truncation / **wrapper vs model**(max_diff=0.0)
- 端到端推理:chat_template + model.generate(BOS + bfloat16)

**用法**:
```bash
# 单元测试 only
python verify_qwen3.py --test-only

# 真实权重推理
python verify_qwen3.py --model /mnt/workspace/data/models/qwen3-0.6B --prompt "你好" --max-new-tokens 16

# 跳过单元,只推理
python verify_qwen3.py --model /path/to/model --no-tests --prompt "Hi"
```

**参数**:`--test-only` / `--model` / `--prompt` / `--max-new-tokens`(64)/ `--device`(auto)/ `--no-tests`

**验证结果**:✅ 35 passed + 端到端 "好的,用户发来了一条消息:你好..."

---

## 2. `verify_qwen3_vs_hf.py` — pure-torch/wrapper/embeddings vs HF

**测试场景**(3 模式):
- `--mode full`(默认):pure-torch model.forward vs HF(float32,max_diff)
- `--mode wrapper`:Qwen3ExportWrapper vs HF(bfloat16,prefill+decode,tokens match)
- `--mode embeddings`:use_embeddings=True(图外查表)vs HF(bfloat16)

**用法**:
```bash
# full (float32, max_diff=0.0)
python verify_qwen3_vs_hf.py --mode full --model /mnt/workspace/data/models/qwen3-0.6B

# wrapper (bfloat16, tokens match)
python verify_qwen3_vs_hf.py --mode wrapper

# embeddings (bfloat16, tokens match)
python verify_qwen3_vs_hf.py --mode embeddings
```

**验证结果**:
- full:✅ max_diff=0.0,greedy identical
- wrapper:✅ tokens match(bfloat16 diff 0.25)
- embeddings:✅ tokens match

---

## 3. `verify_qwen3_epair.py` — epair 导出 + 回环 + 动态 shape

**测试场景**:
- ep 模式(默认):torch.export + ep.module() prefill + wrapper decode(scatter mask kv_max + 屏蔽未写入)
- wrapper 模式:wrapper direct 推理(不 pad,cat KV)
- `--dynamic`:动态 shape(seq_len + batch symbolic)epair 导出
- `--vs-hf`:HF model.generate(attn_implementation="eager")对比

**用法**:
```bash
# ep 回环 vs HF
python verify_qwen3_epair.py

# wrapper direct
python verify_qwen3_epair.py --mode wrapper

# 动态 shape 导出
python verify_qwen3_epair.py --dynamic

# sampling
python verify_qwen3_epair.py --sampling
```

**验证结果**:✅ PASS(ep==wrapper==HF,tokens match)

---

## 4. `verify_qwen3_chunk.py` — chunk 推理 + vs HF

**测试场景**:分 chunk prefill vs 非 chunk + vs HF(cat KV,past=[None]*N,EOS 停)

**用法**:
```bash
# cann, chunk vs 非 chunk
python verify_qwen3_chunk.py

# wrapper 模式
python verify_qwen3_chunk.py --mode wrapper

# vs HF
python verify_qwen3_chunk.py --vs-hf --chunk-size 4 --num-steps 16
```

**参数**:`--mode cann/wrapper` / `--vs-hf` / `--chunk-size`(8)/ `--num-steps`(10)/ `--prompt` / `--token-level`

**验证结果**:✅ PASS(三 prompt chunk==非chunk==HF)

---

## 5. `verify_qwen3_multiturn.py` — 多轮对话 vs HF

**测试场景**:多轮对话(wrapper + chunk,greedy)vs HF,每轮独立 prefill(含历史)+ decode

**用法**:
```bash
# cann vs HF
python verify_qwen3_multiturn.py --vs-hf

# wrapper + chunk vs HF
python verify_qwen3_multiturn.py --mode wrapper --chunk-size 4 --vs-hf

# 自定义多轮
python verify_qwen3_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI" --vs-hf
```

**参数**:`--mode cann/wrapper` / `--chunk-size`(0=非 chunk)/ `--num-steps`(16)/ `--prompts`(\|\| 分隔)/ `--vs-hf`

**验证结果**:✅ PASS(cann + wrapper+chunk 3 轮全 ==HF)

---

## 常见问题

### qwen3 特有

1. **scatter-write 已改造**(遗留任务 5):`Qwen3Attention.forward` 加 `new_kv_cache_pos` + `index_copy`(scatter,K=kv_max);`DecoderLayer`/`TextModel`/`ForCausalLM`/wrapper 传参链;`_prepare_causal_mask` 4D 透传 + attention_mask=None 用 `total_len`
2. **thinking 模式**:`enable_thinking=False` 关闭 thinking(手动 decode 需此);`model.generate` 自动处理
3. **chunk prefill 需 `past=[None]*N`**(非 None):`past=None` → attention 不存 KV → decode 每步从头(乱)
4. **decode 需 EOS 停**:手动 decode 不加 EOS → 生成超 content 后乱
5. **bfloat16 精度**:float32 max_diff=0.0(逻辑对);bfloat16 diff 0.03-0.25(BLAS 舍入,tokens match)
6. **HF 需 `attn_implementation="eager"`**:对齐 cann_lm eager attention(非 sdpa)

### 通用

- **bfloat16**:模型原始精度(skill #2)
- **验证 logits 用 greedy 而非 sampling**(skill #18)
- 详见 `skills/` 下的 SKILL.md

## 文档关联

- `skills/`:qwen3 调试经验
- `../debug_gemma4/README.md`:gemma4 调试流程参考
- `../debug_qwen3_5/README.md`:qwen3.5 调试流程参考(delta_cache 持久化、linear attention segfault)
- `../../../../../docs/cann_lm_converter/converter_design.md`:converter 全流程设计 + 精度差异分析
