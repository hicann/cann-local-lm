# om_npu_verify — NPU 侧 OM 运行验证

把导出的 epair 编译为 OM，在 NPU 上运行验证：单 OM vs eager 对比、prefill+decode 文本生成（出字）。

> 前置：`.run` 包已安装（`/opt/cann-device-lm/`），`cann_lm` whl 已装进 P1 venv + 系统 python。

## 目录结构

```
om_npu_verify/
├── run_gemma4_om.py             # ACL python OM 运行验证 (单 OM vs eager)
├── run_gemma4_gen.py            # ACL python OM 文本生成 (prefill+decode pipeline, 出字)
├── run_qwen3_5_om*.py          # qwen3.5 OM 验证 (OM vs EP/HF)
├── README.md                    # 本文件
├── epair_gemma4/                # 导出产物 (ep_json + weight_info + weights_*.bin)
├── epair_gemma4_pd/             # prefill+decode 两图产物 (出字用)
└── epair_qwen3_5_emb/          # qwen3.5 embeddings 入图产物
```

> OM 编译工具 `cann_lm_compile` 在 `.run` 包内（`/opt/cann-device-lm/cann_lm_compile`）。`reg_register.h` 仅注册 `ChunkGatedDeltaRule`（CANN 库已内置其余算子）。

## 环境

```bash
# 导出 epair 用（converter 模式）
source $HOME/Ascend/cann/set_env.sh
source /opt/cann-device-lm/set_env.sh
export CONVERTER_PYTHON=/path/to/venv/bin/python

# NPU 出字验证用（NPU 模式，自动探测 DOPT_PYTHON）
source /opt/cann-device-lm/set_env.sh --npu
```

> HF 模型路径由调用方必填：`export GEMMA4_HF_PATH=<model dir>` / `export QWEN3_5_HF_PATH=<model dir>`

## 使用流程

### 1. 导出 epair

```bash
$CONVERTER_PYTHON -m cann_lm.converter.cann_lm_converter \
  --hf-path "$GEMMA4_HF_PATH" \
  --output-dir epair_gemma4 \
  --mode prefill-decode --target-soc Ascend910B
```

- 纯 embeddings（`--use-embeddings` 默认 True）：embed_tokens + PLE 查表外置到 host，图内不含。
- prefill+decode 两图（出字用）：`--mode prefill-decode`。
- dtype 由 `--target-soc` 派生（Ascend→bfloat16, Kirin→float16）。

### 2. 编译 epair → OM

```bash
source /opt/cann-device-lm/set_env.sh   # 设 GE_FUSION_SWITCH_FILE + SOC_VERSION

/opt/cann-device-lm/cann_lm_compile \
  epair_gemma4/prefill_ep_json_with_ascend_ir.json \
  epair_gemma4/weight_info.json \
  epair_gemma4/prefill

/opt/cann-device-lm/cann_lm_compile \
  epair_gemma4/decode_ep_json_with_ascend_ir.json \
  epair_gemma4/weight_info.json \
  epair_gemma4/decode
# → prefill/prefill.om + decode/decode.om
```

> 编译结束后可能看到 `Aborted` 报错——这是 GE 库已知问题，不影响结果。检查 `.om` 文件存在即成功。

### 3. 查看 OM 算子/I/O

```bash
atc --mode=1 --om=prefill/prefill.om --json=/tmp/om.json   # OM→JSON
atc --mode=6 --om=prefill/prefill.om                        # model info
```

### 4. NPU 出字验证

```bash
source /opt/cann-device-lm/set_env.sh --npu

cd examples/test_converter/export_llm_model/om_npu_verify
PREFILL_OM=epair_gemma4/prefill/prefill.om \
DECODE_OM=epair_gemma4/decode/decode.om \
HF="$GEMMA4_HF_PATH" \
$DOPT_PYTHON run_gemma4_gen.py "Hello, my name is"
```

**流程**：
1. chat template tokenize → input_ids
2. 按 PREFILL_SEQ 切块，逐块跑 prefill OM（KV 跨块累积）
3. 最后一块得首字 logits → argmax → next token
4. decode loop：每步构造 decode 输入（seq=1，新 token + 上步 KV + mask）→ decode OM → logits + KV → argmax → 吐字
5. EOS 或 MAX_NEW 步停止 → decode tokens → text

### 单 OM vs eager 验证

```bash
GEMMA4_OM=<om_path> HF="$GEMMA4_HF_PATH" $DOPT_PYTHON run_gemma4_om.py
```

- OM logits vs eager logits maxdiff < 1.0 → MATCH（运行机制正确）

## 编译开关（env）

| env | GE key | 取值/说明 |
|---|---|---|
| `GE_FUSION_SWITCH_FILE` | `ge.fusionSwitchFile` | fusion switch cfg 路径；按 pass 名 on/off（默认关 AutomaticUbFusion，由 .run 包 set_env.sh 设置） |
| `GE_BUFFER_OPT` | `ge.bufferOptimize` | `l2_optimize`(默认)/`off_optimize`（实测 off 会编译失败） |
| `GE_OO_LEVEL` | `ge.oo.level` | `O1`/`O3`(默认)，降优化级 |
| `SOC_VERSION` | `ge.socVersion` | 默认 `Ascend910_9362` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | — | `1` 打开 GE 详细日志 |

## OM 算子表（gemma4，关 AUB，20 种）

| 算子 | 数量 | 说明 |
|---|---|---|
| Const | 593 | 权重 + cos/sin 预缓存表 |
| MatMul | 277 | 线性投影（q/k/v/o、mlp、PLE、lm_head） |
| RmsNorm | 207 | 融合 RMSNorm |
| Mul | 105 | 元素乘（scale/PLE combine） |
| Transpose | 105 | |
| Add | 106 | 残差+mask 加 |
| Data | 77 | 图输入 |
| ConcatV2D | 70 | repeat_kv cat |
| BatchMatMul | 70 | q@kᵀ、attn@v |
| Gelu | 70 | MLP 激活 |
| Cast | 67 | dtype 转换 |
| RotaryMul | 50 | 融合 RoPE |
| MemcpyAddrAsync | 40 | 共享层 KV 透传（层 15-34，20×2） |
| Muls | 37 | 标量乘 |
| GatherV2 | 35 | PLE 每层切片 |
| SoftmaxV2 | 35 | attention softmax |
| InplaceAddRmsNorm | 35 | 融合 add+rmsnorm |
| ScatterUpdate | 30 | KV scatter-write（15 非共享层×2） |
| Index | 5 | rope cos/sin Gather(4)+output_pos(1) |
| NetOutput | 1 | 图输出 |

**无 MaskedFill / Where / SelectV2**（wrapper host 化 mask）；**无 Pow/Neg/Cos/Sin**（RoPE 预缓存表+Gather）。

## I/O（gemma4，seq_len=8 特化）

**输入 77 个**：

| 输入 | 形状 | dtype | 含义 |
|---|---|---|---|
| embeddings | [1,8,1536] | bf16 | host 查表 embed_tokens |
| mask_full | [1,1,8,2048] | bf16 | causal-over-buffer（全 attention 层） |
| mask_sliding | [1,1,8,2048] | bf16 | causal+window（sliding 层） |
| position_ids | [1,8] | int64 | |
| new_kv_cache_pos | [8] | int64 | KV scatter 写入位 |
| output_pos | [1] | int32 | 选最后 token |
| per_layer_token_identity | [1,8,35,256] | bf16 | PLE host 查表 |
| past_key_values_0_0..34_1 (70) | [1,1,2048,head_dim] | bf16 | 35层×2 KV cache |

**输出 71 个**（NetOutput）：

| 输出 | 数量 | 形状 | 含义 |
|---|---|---|---|
| logits | 1 | [1,1,262144] | output_pos 选位 |
| scatter KV | 30 | [1,1,2048,head_dim] | 15 非共享层(0-14)×2 |
| 共享 KV | 40 | [1,1,2048,head_dim] | 20 共享层(15-34)×2 透传 |

> head_dim 逐层不同：full_attention 层 `global_head_dim=512`、sliding 层 `head_dim=256`。真实 e2b 的 full 层在 4,9,14,19,24,29,34（每 5 层一个）。

## 注意事项

- **每次编译默认关 AUB**：`.run` 包的 `set_env.sh` 已设 `GE_FUSION_SWITCH_FILE`（关 AutomaticUbFusion），元素算子保持独立、图清晰可读。
- **wrapper host 化 mask**：mask 由 host 预构输入，图内只 `Add` 应用；图内无 MaskedFill（NPU 不支持）。
- **RoPE 预缓存表+Gather**：cos/sin 烘焙为 Const，forward 只 Gather；消除 Pow/Neg/Cos/Sin；代价 OM +~400MB。
- **KV scatter-write**：用 `ScatterUpdate`（替代 `index_copy`），15 非共享层×2。
- 真实 e2b 配置：35 层（后 20 KV 共享）、head_dim=256/global=512、hidden=1536、num_kv_heads=1(n_rep=8)、PLE 开、sliding=512、kv_max=2048、vocab=262144。

## ring KV cache 测试用例（sliding 层 ring shrink, sliding_kv_pos 输入, 0 FloorMod）

gemma4 sliding 层 KV cache 从 `kv_cache_max_len`(2048) 缩到 ring `R=sliding_window+prefill_seq`(512+64=576)，full 层仍 2048。ring slot `pos%R` 由 **host 预算**作 `sliding_kv_pos` 输入喂进 OM（**图内无 FloorMod**，取模在 host Python），scatter 直接用该输入当槽位 + ring wrap mask。下列 4 个 prompt 覆盖 4 个临界点，均用同一个 ring OM（`prefill.om`+`decode.om`，R=576 固定，80 输入含 `sliding_kv_pos`）跑通。

prompt 基串：`The quick brown fox jumps over the lazy dog. `（11 tok/重复），按重复次数凑到目标 token 数。

| 用例 | 重复 | tokens | 临界点（测什么） | prefill chunk | OM 输出 |
|---|---|---|---|---|---|
| **<64** | ×5 | 51 | 单 chunk prefill（pad 到 64，1 chunk） | 1 | "That is the famous **pangram**!" |
| **>64** | ×8 | 81 | 多 chunk prefill + KV 跨 chunk 传（chunk0→chunk1） | 2 | "That is the famous **pangram**!" |
| **>576** | ×60 | 601 | **ring wrap 触发**（pos≥576，chunk9 pos576..600） | 10 | "This is a bit of a mixed input! It seems" |
| **>2048** | ×210 | 2101 | ring wrap + **full KV 溢出**（full 层 kv_max=2048，pos≥2048 散列覆盖旧位） | 33 | "jumps over the lazy dog. The quick brown fox jumps" |

跑法（须 `PYTHONPATH=repo` 用 repo 的 ring cann_lm，venv whl 是 stale 无 ring 版）：

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/CANN-LM_5883:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH}
export HF=<gemma-4-E2B-it 路径>
export PREFILL_OM=<ring prefill.om>  DECODE_OM=<ring decode.om>
export PREFILL_SEQ=64  MAX_NEW=10
cd examples/test_converter/export_llm_model/om_npu_verify

# case1 <64 (单 chunk)
python3 run_gemma4_gen.py "$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*5)")"
# case2 >64 (多 chunk)
python3 run_gemma4_gen.py "$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*8)")"
# case3 >576 (ring wrap)
python3 run_gemma4_gen.py "$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*60)")"
# case4 >2048 (full KV 溢出)
python3 run_gemma4_gen.py "$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*210)")"
```

**临界点说明**：
- `<64`：prompt 不足 64，单 prefill chunk（pad 到 64）。验基础 prefill+decode 链路。
- `>64`：prefill 分多 chunk（每 chunk 64），验多 chunk prefill + KV 跨 chunk 传（chunk0→chunk1）。
- `>576`：sliding 层 ring R=576，pos≥576 触发 **ring wrap**（`sliding_kv_pos=pos%576` 绕回，mask 两段 attend）+ scatter 覆盖旧槽位。验 ring wrap 正确性。
- `>2048`：full 层 kv_max=2048，pos≥2048 散列 `pos%2048` 覆盖旧 KV（full attention 丢早期上下文）；ring（sliding）层 576 早已 wrap。本用例 prompt 高度重复，溢出被重复模式掩盖仍出字；非重复 prompt >2048 会退化（full KV 限制，非 ring 问题）。

> ring shrink 仅作用于 sliding 层（R=576，slot 由 `sliding_kv_pos` 输入控制，图内 0 FloorMod）；full 层仍 kv_max=2048，>2048 受 full KV 限制。要支持更长需调大 `kv_cache_max_len`（full 层）。
