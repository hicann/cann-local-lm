# om_npu_verify — gemma4 epair→OM 编译与验证

把 cann_lm 定义的 gemma4(新 wrapper:host 预构 mask、绕过 forward、无 masked_fill;新 repeat_kv:reshape-cat-view)导出为 epair,再用 CANN `aclgrphBuildModel` 编译为 OM,并查看 OM 中的算子与输入输出。

## 目录结构

```
om_npu_verify/
├── run.sh                    # 批量导出脚本 (调 export_llm_model_epair.py)
├── export_llm_model_epair.py # 导出 epair (config→model→weights→wrapper→torch.export→epair)
├── main.cpp                  # C++ OM 编译器 (EpairModelLoader→GE Graph→aclgrphBuildModel→.om)
├── reg_register.h            # CANN 算子注册宏 (main.cpp include, 同目录)
├── CMakeLists.txt            # 构建 main.cpp (链 ge_compiler/ge_runner/acl_rt + torch_delegate_backend)
├── build/export_llm_model   # 编译好的可执行 (aarch64)
├── epair_gemma4/             # 导出产物 (ep_json + weight_info + weights_*.bin)
├── run_gemma4_om.py          # ACL python OM 运行验证 (单 OM vs eager)
├── run_gemma4_gen.py         # ACL python OM 文本生成 (prefill+decode pipeline, 吐字)
├── gemma4.om                 # 默认融合 OM (4.7G, 新 rope, AUB 开, 对比用)
└── gemma4_noaub.om           # 关 AUB OM (4.7G, 新 rope, **推荐用**)
```

## 环境

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export SOC_VERSION=Ascend910_9362
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}
VENV=/mnt/workspace/torch-delegate-harmony/.venv
```
> 导出用 converter venv(torchair stubs,CPU-only);OM 编译用真 CANN(aarch64,本机 Ascend910)。

## 使用流程

### 1. 构建 C++ 编译器(一次性)
```bash
export TORCH_PKG_DIR=/mnt/workspace/torch-delegate-harmony/.venv/lib/python3.11/site-packages/torch
cmake -S . -B build -DCANN_ROOT="$ASCEND_HOME_PATH" -DTORCH_PKG_DIR="$TORCH_PKG_DIR"
cmake --build build -j4
```

### 2. 导出 epair(新 wrapper + 新 repeat_kv)
```bash
$VENV/bin/python export_llm_model_epair.py \
  --model gemma4 --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
  --seq-len 8 --dtype bfloat16 --output-dir epair_gemma4 --use-embeddings
```
- `--use-embeddings`:embed_tokens + PLE embed_tokens_per_layer 查表外置到 host(图内不含,host 喂 embeddings + per_layer_token_identity);detach 避免冗余打包。
- 产物:`epair_gemma4/ep_json_with_ascend_ir.json`(GE IR)+ `weight_info.json` + `weights_*.bin`。
- 或 `bash run.sh gemma4`(run.sh 默认非 use_embeddings;gemma4 新 wrapper 需 `--use-embeddings`,见下"注意")。

### 3. 编译 epair→OM(**每次编译都关 AUB**)
> **默认每次编译都关 `AutomaticUbFusion`**(产生 `AutomaticBufferFusionOp` 的 UB 融合 pass),让元素算子(Mul/Add/Gelu 等)保持独立、图结构清晰可读。见下"关掉 AutomaticBufferFusionOp"。先建 cfg(一次性):
```bash
# /tmp/fusion_autoub_off.cfg (一次性创建, 复用):
cat > /tmp/fusion_autoub_off.cfg <<'EOF'
{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}
EOF
```
```bash
# 每次编译都带 GE_FUSION_SWITCH_FILE 关 AUB:
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg
./build/export_llm_model epair_gemma4/ep_json_with_ascend_ir.json \
  epair_gemma4/weight_info.json gemma4_noaub
# → gemma4_noaub.om (3 阶段: Load epair → Extract GE Graph → Compile)
```
- 不关 AUB(默认融合,有 141 个 `AutomaticBufferFusionOp`):`unset GE_FUSION_SWITCH_FILE` 后同上,产出 `gemma4.om`(对比/调试用)。
> 末尾 `Aborted (exit 134)` + `corrupted size vs. prev_size` 是 **GE finalize 退出时的堆损坏**(在 `aclgrphSaveModel` 之后触发,OM 已保存),不影响产物。属已知 GE quirk。

### 4. 查看 OM 算子/I/O(dump 成 JSON)
```bash
atc --mode=1 --om=gemma4_noaub.om --json=/tmp/gemma4_om.json   # OM→JSON
atc --mode=6 --om=gemma4_noaub.om                              # 显示 model info (大小/stream 等)
```

## 编译开关(env,免改代码重编即可实验)

main.cpp 已加 env 注入(改完需 `cmake --build build` 重编一次 main.cpp):

| env | GE key | 注入位置 | 取值/说明 |
|---|---|---|---|
| `GE_FUSION_SWITCH_FILE` | `ge.fusionSwitchFile` | init_options (`aclgrphBuildInitialize`) | fusion switch cfg 路径;按 pass 名 on/off |
| `GE_BUFFER_OPT` | `ge.bufferOptimize` | build_options (`aclgrphBuildModel`) | `l2_optimize`(默认)/`l1_optimize`/`off_optimize`(**实测 off 会让编译失败,勿用**) |
| `GE_OO_LEVEL` | `ge.oo.level` | build_options | `O1`/`O3`(默认),降优化级 |
| `GE_OPT_SWITCH` | `ge.optimizationSwitch` | build_options | `"pass1:off;pass2:on"` |
| `SOC_VERSION` | `ge.socVersion` | init_options | 默认 `Ascend910_9362` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | — | 环境 | `1` 打开 GE 详细日志 |

### 关掉 AutomaticBufferFusionOp(UB 融合)— **每次编译默认关**
**推荐每次编译都关 `AutomaticUbFusion`**(产生 `AutomaticBufferFusionOp` 的 UB 融合 pass)。理由:关后元素算子(Mul/Add/Gelu/Muls)保持独立、图清晰可读、便于排查;且不影响结构性算子(MatMul/RmsNorm/BatchMatMul/ConcatV2D/SoftmaxV2/RotaryMul/ScatterUpdate/InplaceAddRmsNorm/GatherV2),OM 大小几乎不变(4.7G)。

`AutomaticBufferFusionOp` 由 UB 融合 pass **`AutomaticUbFusion`** 产生(查 `fusion_result.json` 的 `ub_fusion` 字段可得 pass 名)。只关它一个(其余融合保留,图不重构):
```bash
# cfg 内容: {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg   # 见上 "3. 编译" 的建 cfg 步骤
./build/export_llm_model epair_gemma4/ep_json_with_ascend_ir.json epair_gemma4/weight_info.json gemma4_noaub
```
- 一键全关(blunt,会把图重构成 PartitionedCall 子图,**不推荐**):`{"Switch":{"GraphFusion":{"ALL":"off"},"UBFusion":{"ALL":"off"}}}`。
- `ge.bufferOptimize=off_optimize` **实测会让编译失败**(status 50331649),勿用。

### 查融合 pass 名
编译后看 `fusion_result.json`(根目录或 OM 同级),结构 `{session_and_graph_id_0_0: {graph_fusion: {...}, ub_fusion: {PassName: {effect_times, match_times}}}}`。或开 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 看 plog。

## 当前 OM 算子表(新 rope,默认融合 gemma4.om,1874 个/19 种)

> 新 rope = `Gemma4TextRotaryEmbedding` 预缓存 cos/sin 常量表 + `cos_cached[position_ids]` Gather(npu_tuned 2.3),消除 Pow/Neg/Cos/Sin(原各 2)。代价:cos/sin 表烘焙为 Const,OM +~400MB(sliding 128MB + full 256MB,bf16)。

| 算子 | 数量 | 说明 |
|---|---|---|
| Const | 593 | 权重/常量(含 cos/sin 预缓存表) |
| **MatMul** | 277 | 线性投影(q/k/v/o、mlp gate/up/down、PLE per_layer_model_projection、lm_head) |
| **RmsNorm** | 207 | 融合 RMSNorm(`npu_rms_norm`) |
| AutomaticBufferFusionOp | 141 | UB 融合节点(`AutomaticUbFusion` pass 产出,吸收元素算子;**关 AUB 后为 0**) |
| Transpose | 105 | |
| **Data** | 77 | 图输入 |
| **ConcatV2D** | 70 | repeat_kv 的 cat(35层×2 K/V)← 新 repeat_kv |
| BatchMatMul | 70 | q@kᵀ、attn@v |
| Cast | 67 | dtype 转换 |
| RotaryMul | 50 | 融合 RoPE(`npu_rotary_mul`) |
| MemcpyAddrAsync | 40 | 共享层 KV 透传(层 15-34,20×2) |
| GatherV2 | 35 | PLE 每层切片 `per_layer_inputs[:,:,i,:]` |
| SoftmaxV2 | 35 | attention softmax |
| InplaceAddRmsNorm | 35 | 融合 add+rmsnorm(`npu_add_rms_norm`) |
| Add | 35 | 残差 + mask 应用(attn_weights+mask) |
| ScatterUpdate | 30 | KV scatter-write(`index_copy` converter;15 非共享层×2) |
| **Index** | 5 | rope 的 cos/sin Gather(4)+ output_pos 选位(1)← 新 rope |
| Muls / NetOutput | 1 / 1 | 标量乘 / 图输出 |

**无 MaskedFill / Where / SelectV2**(新 wrapper host 化 mask)**无 Pow/Neg/Cos/Sin**(新 rope 预缓存表)。mask 由 host 预构输入,图内只 `Add` 应用;KV 用 `ScatterUpdate` scatter-write;RoPE 用 `RotaryMul` + 预缓存表 `Index` Gather。

## 推荐 OM:关 AutomaticUbFusion(gemma4_noaub.om,新 rope,2015 个/20 种)

| 算子 | 数量 | 说明 |
|---|---|---|
| Const | 593 | 权重 + cos/sin 预缓存表 |
| MatMul | 277 | 线性投影 |
| RmsNorm | 207 | 融合 RMSNorm |
| **Mul** | 105 | 元素乘(scale/PLE combine,AUB 关后独立) |
| Transpose | 105 | |
| **Add** | 106 | 残差+mask 加(独立) |
| Data | 77 | 图输入 |
| ConcatV2D | 70 | repeat_kv cat |
| BatchMatMul | 70 | q@kᵀ、attn@v |
| **Gelu** | 70 | MLP 激活(独立) |
| Cast | 67 | |
| RotaryMul | 50 | 融合 RoPE |
| MemcpyAddrAsync | 40 | 共享 KV 透传 |
| **Muls** | 37 | 标量乘(独立) |
| GatherV2 | 35 | PLE 每层切片 |
| SoftmaxV2 | 35 | attention softmax |
| InplaceAddRmsNorm | 35 | 融合 add+rmsnorm |
| ScatterUpdate | 30 | KV scatter |
| Index | 5 | rope Gather(4)+output_pos(1) |
| NetOutput | 1 | |

## 对比:默认(AUB 开)vs 关 AUB(均新 rope)

| 算子 | AUB 开 | AUB 关 | 差 |
|---|---|---|---|
| **AutomaticBufferFusionOp** | 141 | **0** | -141 |
| **Mul** | 0 | 105 | +105(元素乘重新独立) |
| **Add** | 35 | 106 | +71(残差+mask 加) |
| **Gelu** | 0 | 70 | +70(MLP 激活) |
| **Muls** | 1 | 37 | +36 |
| MatMul / RmsNorm / BatchMatMul / ConcatV2D / SoftmaxV2 / RotaryMul / ScatterUpdate / InplaceAddRmsNorm / GatherV2 / Transpose / Cast / MemcpyAddrAsync / Data / Index | 不变 | 不变 | 0 |
| **总数** | 1874 | **2015** | **+141** |

`AutomaticUbFusion` 专管**元素算子融合**(Mul/Add/Muls/Gelu),压成 `AutomaticBufferFusionOp`(~2:1);结构性算子(MatMul/RmsNorm/BatchMatMul/ConcatV2D/SoftmaxV2/RotaryMul/ScatterUpdate/InplaceAddRmsNorm/GatherV2)**不受其影响**。新 rope 已把 Pow/Neg/Cos/Sin 在两端都消成 0(与 AUB 无关)。

## I/O(seq_len=8 特化)

**输入 77 个**:
| 输入 | 形状 | dtype | 含义 |
|---|---|---|---|
| embeddings | [1,8,1536] | bf16 | host 查表 embed_tokens |
| mask_full | [1,1,8,2048] | bf16 | causal-over-buffer(全 attention 层) |
| mask_sliding | [1,1,8,2048] | bf16 | causal+window(sliding 层) |
| position_ids | [1,8] | int64 | |
| new_kv_cache_pos | [8] | int64 | KV scatter 写入位 |
| output_pos | [1] | int32 | =[7] 选最后 token |
| per_layer_token_identity | [1,8,35,256] | bf16 | PLE host 查表 |
| past_key_values_0_0..34_1(70) | [1,1,2048,head_dim] | bf16 | 35层×2;head_dim=256(sliding)/512(full),num_kv_heads=1 |

**输出 71 个**(NetOutput):
| 输出 | 数量 | 形状 | 来源 | 含义 |
|---|---|---|---|---|
| logits | 1 | [1,1,262144] | Index→Cast | output_pos 选位 |
| scatter KV | 30 | [1,1,2048,head_dim] | Cast | 15 非共享层(0-14)×2 |
| 共享 KV | 40 | [1,1,2048,head_dim] | MemcpyAddrAsync | 20 共享层(15-34)×2,透传 |

> head_dim 逐层不同:gemma4 的 full_attention 层用 `global_head_dim=512`、sliding 层用 `head_dim=256`(modeling_gemma4.py:278-280),KV buffer 跟着各层 attention head_dim 走。真实 e2b 的 full 层在 4,9,14,19,24,29,34(每 5 层一个)。

## ACL python 运行 OM

两个 python 脚本用 CANN `acl` python 接口(`acl.mdl.load_from_file` + `acl.mdl.execute`)在 NPU 上运行 OM,不依赖 torch_export/torchair。

### 环境

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python.site-packages:/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
```

用系统 Python(`/opt/buildtools/Python-3.11.4/bin/python3`),含 `acl` + `torch_npu` + `transformers`。

### run_gemma4_om.py — 单 OM vs eager 验证

验证 OM 运行机制正确性:用 `Gemma4ExportWrapper.get_sample_inputs` 构造 77 输入(mask_full/mask_sliding/position_ids/70kv/new_kv_cache_pos/embeddings/output_pos/per_layer_token_identity),展平后 `acl.mdl.execute`,读 logits,对比 eager wrapper(NPU)输出。

```bash
cd examples/export_llm_model/om_npu_verify
/opt/buildtools/Python-3.11.4/bin/python3 run_gemma4_om.py
```

- OM logits vs eager logits maxdiff < 1.0 → MATCH(运行机制正确)
- 关键:`acl.init` 需容错(torch_npu 可能已 init,ret=100002)
- bf16 字节处理:`tensor.view(int16).numpy().tobytes()`(H2D)+ `frombuffer(int16).view(bfloat16)`(D2H)

### run_gemma4_gen.py — prefill+decode 文本生成(吐字)

完整生成 pipeline:prefill OM(seq=128)处理 prompt → logits + 70 kv;decode OM(seq=1)逐 token 生成。支持超长 prompt 分 chunk prefill。

```bash
cd examples/export_llm_model/om_npu_verify
/opt/buildtools/Python-3.11.4/bin/python3 run_gemma4_gen.py "Hello, my name is"
/opt/buildtools/Python-3.11.4/bin/python3 run_gemma4_gen.py "Please explain the theory of relativity in detail."
```

**prompt 自动套 chat template**(`tok.apply_chat_template`),确保模型正常对话生成(裸 prompt 会退化成重复输出)。

**流程**:
1. chat template tokenize → input_ids
2. 按 PREFILL_SEQ(128)切块,逐块跑 prefill OM(KV 累积:上一块输出→下一块输入)
3. 最后一块得首字 logits → argmax → next token
4. decode loop:每步用 wrapper mask helpers 构造 decode 输入(seq=1,新 token embeddings/PLE + 上步 KV + mask[k≤cached_len])→ decode OM → logits + KV → argmax → 吐字
5. EOS 或 MAX_NEW 步停止 → decode tokens → text

**超长 prompt 支持(prompt > 128 token)**:
- 分 chunk:chunk 0 tokens[0:128] → KV 输出;chunk 1 tokens[128:...] → 用 chunk 0 的 KV → ...
- 每块的 `new_kv_cache_pos`/`position_ids` 按 `start_pos` 偏移;mask 用 `wrapper._build_mask_full/sliding(PREFILL_SEQ, KV_MAX, dtype, new_kv_cache_pos)`(q 位=buffer 绝对位)
- KV_MAX=2048(prompt > 2048 截断+警告)

**关键设计点**:
- 用 `wrapper.get_sample_inputs` 的 mask/kv 约定(gemma4 shared KV 要求 pad 位不被 all-mask,否则污染 store 层 KV → 共享层输出错误)
- prefill OM 的 77 输入 = wrapper sample 展平(跳过 None input_ids/embeddings_scale,70 kv 拆开)
- OM 产物路径在脚本常量 `PREFILL_OM`/`DECODE_OM` 里(当前指向 `/tmp/opencode/gemma4_pd/`)
- 需先导出 prefill+decode OM:`cann-lm-converter.py --mode prefill-decode --use-embeddings` → C++ `main.cpp` 编译两图

## 注意事项

- **每次编译默认关 AUB**:`export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg`(`{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}`)后编译,产出 `gemma4_noaub.om`(元素算子独立、图清晰)。不关则 `gemma4.om`(有 141 个 `AutomaticBufferFusionOp`,对比用)。
- **run.sh 默认非 `--use-embeddings`**:gemma4 新 wrapper 的 forward 收 10 参(mask_full/mask_sliding 等),非 use_embeddings 走 `wrapper.get_sample_inputs(use_embeddings=False)`(已适配);若要 PLE 外置+detach,用 `--use-embeddings`。
- **repeat_kv 已改 expand→reshape-cat-view**:`modeling_gemma4.py:56` 用 `reshape(b*kv,1,s,d).cat([x]*n_rep,dim=1).view(b,kv*n_rep,s,d)`,保持 kv_head-major(与 expand 逐元素相等),图里 `ConcatV2D` 替代 `BroadcastTo`。naive `cat([x]*n_rep,dim=1)` 不 reshape 是 rep-major,会与 q 错位,**不可用**。
- **RoPE 已改预缓存表+Gather**:`Gemma4TextRotaryEmbedding.__init__` 预算 `cos_cached`/`sin_cached`(register_buffer, init 算 inv_freq+outer+cat+cos/sin, 不进图),forward 只 `cos_cached[position_ids]`(Gather→`Index`)。消除 Pow/Neg/Cos/Sin(原各 2,全 0);数值与旧动态逐位相等(maxdiff 0)。代价:cos/sin 表烘焙为 Const,OM +~400MB(sliding 128MB + full 256MB,bf16)。
- **新 wrapper 绕过 forward**:`Gemma4ExportWrapper._drive_layers` 直接驱动 `tm.layers`,sliding mask 由 host 预构(mask_full/mask_sliding),图内无 masked_fill(NPU 不支持 `ge.MaskedFill`)。eager `model.generate` 仍走原 `Gemma4TextModel.forward`(`_prepare_causal_mask` None/2D 分支),未改。
- **abort-at-exit**:OM 编译末尾 `exit 134` + `corrupted size vs. prev_size` 是 GE finalize 堆损坏(OM 已 saved),不影响产物。
- 真实 e2b 配置:35 层(后 20 KV 共享)、head_dim=256/global=512、hidden=1536、num_kv_heads=1(n_rep=8)、PLE 开、sliding=512、kv_max=2048、vocab=262144。
