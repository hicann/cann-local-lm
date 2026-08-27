# gemma4 OM 算子等价替换（对齐 fused_ops.md）

> 目的：把 gemma4 OM 终态里**未在 `fused_ops.md` 覆盖的算子**，用 fused_ops.md 覆盖的算子（或 GE 框架算子）等价替换，使 OM 算子与 fused_ops.md 定义对齐。
> 范围：gemma4 无量化（`epair_gemma4_no_quant`）prefill-decode 模型。
> 验证：导出 → OM 编译（prefill+decode 3 stages PASS）→ NPU 出字（`The capital of China is` → `**Beijing**.`）。

## 替换清单

### 1. KV cache scatter-write: `index_copy` → `npu_scatter_nd_update`

| 　　　　　　　 | 原　　　　　　　　　　　　　　　　　　　　　　　| 替换　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ----------------| -------------------------------------------------| ----------------------------------------------------------------|
| Python　　　　 | `cache.index_copy(2, kv_pos, kv_new)`　　　　　 | `torch.ops.npu.npu_scatter_nd_update(cache, indices, updates)` |
| ascend_ir　　　| `ge.InplaceUpdate` + `ge.Transpose`　　　　　　 | `ge.ScatterNdUpdate`（+ `ge.TensorMove`）　　　　　　　　　　　|
| OM 终态　　　　| `ScatterUpdate` 30　　　　　　　　　　　　　　　| `ScatterNdUpdate` 30　　　　　　　　　　　　　　　　　　　　　 |
| fused_ops.md | ❌（InplaceUpdate 融合成 ScatterUpdate，不匹配） | ✅ fuse_op `ScatterNdUpdate` 直接命中　　　　　　　　　　　　　 |

- **文件**：`cann_lm/models/gemma4/modeling_gemma4.py` `scatter_kv_to_cache` + `cann_lm/converter/export/_scatter_nd_update_converter.py`（cann_lm 自带 converter 覆盖 torchair，产出 `ge.ScatterNdUpdate`）
- **语义**：构造 `[b*n*seq, 3]` 多维坐标 indices `(b,n,kv_pos)` + `[b*n*seq, d]` updates，命中 `ge.ScatterNdUpdate` 标准多维 scatter，等价 `index_copy(2, kv_pos, kv_new)`
- **用 out-of-place 版（无 `_`）**：in-place 版 schema 有 alias annotation `Tensor(a!)`，torch.export functionalization（`run_decompositions`/aot_autograd）不支持自定义算子输出带 alias；out-of-place `-> Tensor out` 无 alias，converter 加 `ge.TensorMove(self)` 保证不原地改

### 2. indices stack → cat+transpose（消除 `Pack`）

| | 原 | 替换 |
|---|---|---|
| Python | `torch.stack([col_b, col_n, col_pos], dim=-1)` | `torch.cat([col_b.view(1,-1), col_n.view(1,-1), col_pos.view(1,-1)], dim=0).transpose(0,1)` |
| ascend_ir | `ge.Pack` | `ge.ConcatV2D` + `ge.Transpose` |
| OM 终态 | `Pack` 30 | `Pack` 0（→ ConcatV2D+Transpose） |
| fused_ops.md | ❌ Pack 未列 | ✅ single_op `concatD`/`Transpose` 同族 |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache`
- **原因**：`ge.Pack` 是 torchair 把 `aten.stack` 转换的产物（行交错打包），fused_ops.md 未列；`stack` = `unsqueeze+cat`，用 `cat([view(1,-1)...], 0)+transpose` 等价，命中 ConcatV2D+Transpose
- **注**：`cat([unsqueeze...], -1)`（dim=-1）会引入 `Unsqueeze`（ascend_ir +90），而 `cat([view(1,-1)...], 0)+transpose` 用 `Reshape`（OM 终态优化掉 0）

### 3. indices arange → Const 预计算（消除 `Range`）

| | 原 | 替换 |
|---|---|---|
| Python | `torch.arange(n)` | `torch.tensor([[i]*kv_max for i in range(n)])` |
| ascend_ir | `ge.Range` | `ge.Const`（python list 静态预计算） |
| OM 终态 | `Range` 60 | `Range` 0 |
| fused_ops.md | ❌ Range 未列（框架算子） | ✅ Const 框架算子（类 Const/Data） |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache` `n_grid`
- **原因**：`n`/`kv_max` 静态（config 决定），`arange(n)` 结果可预计算成 Const，避免动态 `Range` 算子

### 4. indices expand → StridedSliceD+ConcatV2D（消除 `BroadcastTo`）

| | 原 | 替换 |
|---|---|---|
| Python | `tensor.view(1,n,1).expand(b,n,seq)` | `n_grid[:, :seq].reshape(-1)`（col_n）+ `torch.cat([kv_pos]*n)`（col_pos） |
| ascend_ir | `ge.BroadcastTo` | `ge.StridedSliceV2`（→ OM StridedSliceD）+ `ge.ConcatV2D` |
| OM 终态 | `BroadcastTo` 60 | `BroadcastTo` 0（→ StridedSliceD+ConcatV2D） |
| fused_ops.md | ❌ BroadcastTo 未列 | ✅ single_op `StridedSliceV2`/`concatD` 同族 |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache` `col_n`/`col_pos`
- **关键约束**：`col_n`（n_idx 每个重复 seq 次，**seq 是动态 SymInt**）**无法用 concat 替代**（concat 需固定 tensor 数）。只能用 `StridedSliceD` 从预计算 Const 网格 `[n, kv_max]` 切 `[:, :seq]`。`col_pos`（kv_pos tile n，n 静态）可用 `concat([kv_pos]*n)`

### 5. rotary cos/sin: advanced indexing → `index_select`（消除 `Index`）

| | 原 | 替换 |
|---|---|---|
| Python | `cos_cached[position_ids]` | `cos_cached.index_select(0, pos_flat).reshape(*position_ids.shape, -1)` |
| ascend_ir | `ge.Index` | `ge.GatherV2` |
| OM 终态 | `Index` 5 | `Index` 0（→ GatherV2 40） |
| fused_ops.md | ❌ Index 未列 | ✅ single_op `GatherV2` |

- **文件**：`modeling_gemma4.py` `Gemma4Rotary.forward`（cos/sin_cached 取位置嵌入）
- **原因**：`cos_cached[position_ids]`（advanced indexing）→ `ge.Index`；改 `index_select(0, pos_flat)` → `ge.GatherV2`

### 6. logits 取位: advanced indexing → `index_select`（消除 `IndexByTensor`）

| | 原 | 替换 |
|---|---|---|
| Python | `logits[:, output_pos, :]` | `logits.index_select(1, output_pos.to(torch.int64))` |
| ascend_ir | `ge.IndexByTensor` | `ge.GatherV2` |
| OM 终态 | `IndexByTensor` 1 | `IndexByTensor` 0（→ GatherV2） |
| fused_ops.md | ❌ IndexByTensor 未列 | ✅ single_op `GatherV2` |

- **文件**：`cann_lm/models/gemma4/export_wrapper.py` `forward`（取 output_pos 位置 logits）
- **注**：`output_pos` 是 int32，`index_select` 要 int64，加 `.to(int64)`（→ `ge.Cast`，fused_ops.md 有 Cast）

### 7. 关残差融合: RmsNorm+Add → 分开（消除 `InplaceAddRmsNorm`）

| | 原 | 替换 |
|---|---|---|
| 来源 | GE 编译期 `AddRmsNormFusionGraphPass` + `InplaceAddRmsNormFusionPass` 把 `Add+RmsNorm` 残差融合成 `InplaceAddRmsNorm` | fusion switch 关这两个 pass，保留分开的 `RmsNorm+Add` |
| OM 终态 | `InplaceAddRmsNorm` 35 | `InplaceAddRmsNorm` 0（RmsNorm 207→242, Add 71→106） |
| fused_ops.md | ❌ InplaceAddRmsNorm 未列 | ✅ fuse_op `Rmsnorm` + single_op `Add` |

- **文件**：`scripts/set_env.sh` om 分支 fusion cfg
- **配置**：
  ```json
  {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"},
             "GraphFusion":{"AddRmsNormFusionGraphPass":"off","InplaceAddRmsNormFusionPass":"off"}}}
  ```
- **注**：`InplaceAddRmsNorm` 不是 modeling 显式调用（modeling 用拆分的 `add + npu_rms_norm`，`forward_add`），而是 GE 编译期识别 `Add+RmsNorm` 残差模式自动融合。35 个 = 每层 `pre_feedforward_layernorm.forward_add` 1 次 × 35 层。关融合后 OM 多 35 算子但对齐 fused_ops.md；如需融合，删 GraphFusion 两项即可

### 8. `Muls`: 保留（替换不划算）

|                | 原（保留）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 尝试替换　　　　　　　　　　　　　　　　|
| ----------------| ---------------------------------------------------------------------------| -----------------------------------------|
| Python         | `x * float`（`per_layer_input_scale`/`per_layer_model_projection_scale`） | `x * torch.tensor(scale)`　　　　　　　 |
| ascend_ir      | `ge.Muls` 2　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | `ge.Mul` +2 + `ge.Fill` 35 + Const 大增 |
| fused_ops.md | ❌ Muls 未列（框架算子）　　　　　　　　　　　　　　　　　　　　　　　　　 | Mul ✅ 但 Fill ❌　　　　　　　　　　　　 |

- **文件**：`export_wrapper.py`（`* tm.per_layer_model_projection_scale` / `* tm.per_layer_input_scale`）
- **决定**：**保留 Muls**。`Muls` 是 GE 高效标量乘专用算子（框架算子，类 Const/Data/MemcpyAddrAsync，本就不属 fused_ops.md 范畴）。替换成 `Mul+Const` 会引入 `Fill` 35（标量 broadcast 填充），净 +35 算子，得不偿失

## 端到端验证

| 阶段 | 结果 |
|---|---|
| converter 导出（`run_pd6_cases.sh 2`） | PASS（prefill+decode ep_json + weights 4761MB） |
| ascend_ir 验证 | Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm 全 0 |
| OM 编译（prefill+decode） | 3 stages 全 PASS |
| OM 终态（ATC） | 非框架算子全命中 fused_ops.md |
| NPU 出字 | `The capital of China is` → `**Beijing**.` ✅ |

## 最终 OM 算子（prefill.om，1979 个）

| 算子　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 数量　| fused_ops.md 命中　　　　　　|
| ------------------------------------------------------------------| ------:| --------------------------------|
| Const　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 593　 | 框架算子　　　　　　　　　　　 |
| MatMul　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 277　 | ⚠️ MatmulV3 同族　　　　　　　　|
| RmsNorm　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 242　 | ✅ fuse Rmsnorm　　　　　　　　 |
| Add　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 106　 | ✅ single Add　　　　　　　　　 |
| Mul　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 105　 | ✅ single Mul　　　　　　　　　 |
| Transpose　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 100　 | ✅ single Transpose　　　　　　 |
| ConcatV2D　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 85　　| ⚠️ concatD 同族　　　　　　　　 |
| Data　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 77　　| 框架算子　　　　　　　　　　　 |
| Gelu　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 70　　| ✅ single Gelu　　　　　　　　　|
| Cast　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 66　　| ✅ single Cast　　　　　　　　　|
| StridedSliceD　　　　　　　　　　　　　　　　　　　　　　　　　　| 60　　| ⚠️ StridedSliceV2 同族　　　　　|
| RotaryMul　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 50　　| ✅ fuse RotaryPositionEmbedding |
| GatherV2　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 40　　| ✅ single GatherV2　　　　　　　|
| MemcpyAddrAsync　　　　　　　　　　　　　　　　　　　　　　　　　| 40　　| 框架算子　　　　　　　　　　　 |
| FusedInferAttentionScore　　　　　　　　　　　　　　　　　　　　 | 35　　| ✅ fuse FusedInferAttention　　 |
| ScatterNdUpdate　　　　　　　　　　　　　　　　　　　　　　　　　| 30　　| ✅ fuse ScatterNdUpdate　　　　 |
| Muls　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 2　　 | 框架算子（保留）　　　　　　　 |
| NetOutput　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 1　　 | 框架算子　　　　　　　　　　　 |
| **Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm** | **0** | ✅ 全消除　　　　　　　　　　　 |

## 结论

- **fuse_op**：4/6 命中（Rmsnorm/FusedInferAttention/RotaryPositionEmbedding/ScatterNdUpdate），2 未用是模型架构差异（ChunkGatedDeltaRule SSM、l2norm）
- **single_op**：9/14 命中（含 3 同族改名 StridedSliceD/ConcatV2D/MatMul），5 未用是架构差异
- **框架算子**（Const/Data/NetOutput/MemcpyAddrAsync/Muls）：本就不属 fused_ops.md 范畴
- **未覆盖算子全部消除**：Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm = 0

gemma4 OM 终态非框架算子**全部命中 fused_ops.md 定义**，只剩 `Muls` 2 个是 GE 框架标量乘算子（保留最优）。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `cann_lm/models/gemma4/modeling_gemma4.py` | `scatter_kv_to_cache`（cat+transpose+Const+StridedSliceD+ConcatV2D）+ rotary `index_select` |
| `cann_lm/models/gemma4/export_wrapper.py` | logits `index_select` |
| `cann_lm/converter/export/_scatter_nd_update_converter.py` | 新建，产出 `ge.ScatterNdUpdate`（覆盖 torchair） |
| `cann_lm/converter/export/ge_converter.py` | 注册 `npu_scatter_nd_update(_)` 覆盖 |
| `cann_lm/models/fused_ops/scatter_nd_update.py` | docstring 更新（gemma4 已落地） |
| `docs/cann_lm_models/fused_ops/fused_ops.md` | §3 torchair converter 映射 + 备注（原 support_ops §4）|
| `docs/cann_lm_models/fused_ops/fused_ops.md` | §4.2.6 / §5.3 P2 标记已落地（原 compositte_op §2.6）|
| `scripts/set_env.sh` | om fusion cfg 关 AddRmsNorm/InplaceAddRmsNorm 融合 |

---

## FusedInferAttentionScore：qwen3.5 调查与现状（当前全 eager，FIA decode 修复已验证但暂搁置）

> **当前状态（2026-08-22）**：qwen3.5 attention **全 eager**（prefill + decode 都用 `eager_attention_forward`）。
> FIA decode 修复（`actual_seq_kvlen=kv_max` + 显式 atten_mask）已开发并验证 decode 正确（committed `2f4755d`），但因 **prefill 多 chunk 限制**（FIA 静态 kvlen 无法跨 chunk；kv_max 对 q_len>1 退化）+ **方案 B 撞 torch.export 分支常量 lift 崩溃**，**暂回退全 eager**。FIA 相关代码（`fused_attention_forward` / `_pad_mask_to_kv_square` / saver fix `kvlen_value`）保留为 dead/no-op，待 prefill 多 chunk 解决后重启用。全 eager 下单/多 chunk prefill + decode 都正确。

### decode FIA 回归根因与修复（已验证，code 保留为 dead/no-op）

从实际 decode ep_json 的 FIA 节点 input[4]/[5]/[6] 抽取的 ground truth：

| | qwen3.5 decode（回归） | qwen3.5 prefill（正常） | gemma4 decode（正常） |
|---|---|---|---|
| `atten_mask` (input[4]) | **空（None）** | `cat_10`（有） | `cat`/`cat_8`（有） |
| `actual_seq_qlen` (input[5]) | `[1]` | `[8]` | `[1]` |
| `actual_seq_kvlen` (input[6]) | `[1]`（saver fix InlineSeqKvlen） | `[8]` | `view_8`（Data 引用，**无 saver fix**） |
| `sparse_mode` | 3 | 3 | 3（full 层）/ 4（sliding 层） |

两个独立 bug 叠加致 qwen3.5 decode 回归：

1. **saver fix 把 `actual_seq_kvlen` 烘成静态 `ge.Const(q_len=1)`**（`epair_utils.py:inline_fused_attn_seq_constants`，decode q_len=1）。`actual_seq_kvlen` 语义是"kv 的有效长度"（torchair 文档），静态 Const=1 让 FIA kernel **静态截断 kv 读取到 1 个 token**（kv[0:1]），丢全部 past。
2. **qwen3.5 decode FIA 无显式 `atten_mask`**：`Qwen3_5TextModel.forward` 原 `if seq_len==1 and past_len>0: causal_mask=None`（为 eager cat-cache 设计，单 query attend 全有效 kv 无需 mask），致 decode full-attn 层拿到 `attention_mask=None` → `fused_attention_forward` 的 `_pad_mask_to_kv_square` 返回 None → FIA 无 atten_mask。

**gemma4 为何"对"**：gemma4 工作的那份 ep_json（`om_npu_verify/epair_gemma4_pd/`）kvlen 是 `view_8`（Data 引用，**未过 saver fix**→PartitionedCall 动态路径），kernel 读更大 kv 范围 + 显式 mask 决定 attend；**不是 gemma4 模型配置对 kvlen=1 免疫**。gemma4 的 full-attn 层（sparse_mode=3 + pre_tokens=int32max + 显式 mask）与 qwen3.5 配置一致，唯一结构差异就是有无显式 atten_mask + kvlen 是 Const 还是 Data 引用。

#### decode 修复（committed `2f4755d`，code 保留）

| 配置 | decode FIA kvlen | atten_mask | `1+1=` 输出 | 结论 |
|---|---|---|---|---|
| 旧 FIA（无 mask） | `[1]`（saver fix Const） | 无 | `2019-01-01` | ❌ |
| **方案1**（只加 mask） | `[1]`（saver fix Const） | 有（`cat_10`） | `2019-01-01` | ❌ 方案1证伪：静态 Const=1 仍截断，mask 被忽略 |
| kvlen=kv_max（无 mask） | `[2048]` | 无 | `!!` 乱码 | ❌（早期文档试过） |
| **完整修复**（mask + kvlen=kv_max） | `[2048]` | 有（`cat_10`） | **`2+1=3+1=4+1=5+1=6+1=7`** | ✅ **正确** |

→ decode 的 mask 与 kvlen=kv_max **两者都要**：mask 决定实际 attend 范围（0..cached_len），kvlen=kv_max 让 kernel 读全 buffer（mask 兜底屏蔽 garbage 位）。单独任一都不行。

修复 code（保留为 dead/no-op）：
- `modeling_qwen3_5.py`：去掉 decode `causal_mask=None` 特例（decode 也构造 causal_mask，4D 透传）+ `fused_attention_forward`（`_pad_mask_to_kv_square` + atten_mask）。**当前 call site 回退 eager**，`fused_attention_forward`/`_pad_mask_to_kv_square` 留为 dead code 便于重启用。
- `epair_utils.py:inline_fused_attn_seq_constants`：加 `kvlen_value` 参数（默认 None→用 qlen_value）；decode 调用传 `kvlen_value=kv_cache_max_len`（`saver.py`）。**全 eager 下无 FIA 节点→saver fix no-op**。
- `fused_attention_forward` 内 `actual_kv_len` 仍设 `q_len`（eager/cat 路径用），export decode 被 saver fix 覆盖为 kv_max。

### prefill FIA 限制：kv_max 对 q_len>1 退化 + 多 chunk 静态 kvlen 无法跨 chunk

decode 的 `kvlen=kv_max` 修复**不能照搬到 prefill**：

1. **prefill kvlen=kv_max 对 q_len>1 退化**（实证，Ascend910）：
   - patch prefill ep_json 的 `InlineSeqKvlen` `[seq_len]→[kv_max]`，重编 OM，测单 chunk（mask 最简单正确）：
     - gemma4（q_len=128）：输出全 pad token 0（退化）
     - qwen3.5（q_len=8）：输出 `22222...`（卡在重复"2"）
   - **已排除 mask 构造原因**：kvlen=128/2048 用**同一个** `[kv_max,kv_max]` padded mask（gen 脚本 `_build_mask_full(seq, KV_MAX)` 永远建 kv_max 宽），唯一变量是 kvlen 值；kvlen=seq_len→正常，kvlen=kv_max→退化。buffer 128..kv_max 位是零（wrapper `torch.zeros`+H2D）非垃圾。故根因是 **FIA kernel 对 `q_len>1 + kvlen>actual写入位` 的限制**（读越界零位即便 mask 屏蔽也产出退化），与 mask/buffer 无关。decode（q_len=1）无此问题。
2. **多 chunk prefill（prompt > seq_len）FIA 静态 kvlen 无法跨 chunk**：
   - modeling `fused_attention_forward` 把 `actual_seq_kvlen` 设成 `q_len`（静态），saver fix 内联成 `ge.Const(seq_len)`。多 chunk 时 chunk 1+（q_pos=start..start+seq）的 FIA 只读 kv[0:seq_len]=chunk 0 的 kv，**读不到自己位置的 kv**（start..start+seq）→ 跨 chunk 上下文丢失 → `...` 乱码。
   - 正确值应是动态 `new_kv_cache_pos.max()+1`（chunk 0=seq_len、chunk 1=2×seq_len…），但静态 `ge.Const` 无法表达 → FIA prefill 不支持多 chunk。
3. **mask/buffer/KV-handoff 均已排除**：eager attention（matmul 全 buffer + mask，无 kvlen）多 chunk prefill 正确（见下"当前全 eager 验证"），证明多 chunk 问题是 FIA-kvlen 专属，非深层 mask/handoff。

### 方案 B（eager prefill + FIA decode，modeling 内分支）尝试：撞 torch.export 分支常量 lift 崩溃

尝试在 `Qwen3_5Attention.forward` 按 `if query_states.shape[2] == 1` 分支（decode→FIA，prefill→eager），靠 torch.export 形状特化让两图各 trace 一条分支。prefill ep_json 正确（FIA=0，eager 标志 BatchMatMul/SoftmaxV2），但 **decode OM 编译崩**（C++ `GeGraphBuilder` Extract GE Graph 阶段 heap corruption `corrupted size vs. prev_size`）。

**根因**：torch.export 对 `if q_len==1` 分支**不只 trace 命中分支**，还把**未命中分支的常量也 lift** 进 ep_json。decode ep_json 含 24 个 `tensor_constant`（vs eager prefill 的 12 个），多出的 12 个是 FIA 分支的 `actual_seq` 提升常量；saver fix 用 `InlineSeq` Const 替换了 FIA input[5]/[6] 引用但**未从 input_specs 剥离这些 dead 提升常量**。decode ep_json 的 input_specs 引用了 weight_info.json（从 eager prefill 打包）里**不存在**的常量名 → C++ `PreparePlaceholders` 加载失败 → 堆崩溃。

（2f4755d 纯 FIA 版本不崩，因 prefill 也是 FIA → 同名 actual_seq 常量已进 weight_info；方案 B 的 prefill=eager 打破了这个共享。）

修复需在 saver fix 做**传递性 dead 常量剥离**（从 input_specs 删掉被 InlineSeq Const 替换后不再被任何 op 引用的 actual_seq 提升常量，含经 View/Reshape 间接引用的），非平凡。**暂搁置**。

### 当前全 eager 验证（2026-08-22，Ascend910）

`Qwen3_5Attention.forward` 用 `eager_attention_forward`，`causal_mask` 保持 always-build（decode 也构造，eager+scatter 需 mask 屏蔽零位）。

| 测试 | prefill | 结果 |
|---|---|---|
| 单 chunk `1+1=` | 1 chunk | `2+1=3+1=4+1=5+1=6+1=7` ✅ |
| 多 chunk 算术递增（~30 tok，9 chunk） | 9 chunk | `11. 11+1=12. 12+1=13.` ✅（跨 chunk 续数） |
| 多 chunk 首都（~30 tok） | 多 chunk | ` Rome. ...` ✅（"意大利首都"→罗马，跨 chunk 上下文） |
| decode OM 算子 | — | 1463 ops，FIA=0，SoftmaxV2=6，BatchMatMul=12（全 eager） |

### OM 算子状态（全 eager）

- FusedInferAttentionScore: 0（用 eager）
- SoftmaxV2: 6（eager attention 的 softmax，6 full-attn 层）
- BatchMatMul: 12（eager 的 QK^T + @V，6 层×2）
- SoftplusV2: 18、L2Normalize: 36（g 计算 / l2norm，保留）
- fused_ops.md fuse_op 命中 5/8（FIA 不用 → ChunkGatedDeltaRule/Rmsnorm/l2norm/RotaryPositionEmbedding/ScatterNdUpdate ✅；FIA/DynamicQuant/QuantBatchMatmulV4 ❌）

### 未来重启用 FIA 的前提

1. **prefill 多 chunk 动态 kvlen**：modeling `fused_attention_forward` 把 `actual_kv_len` 从 `q_len` 改成 `new_kv_cache_pos.max()+1`（动态，chunk N = start+seq），saver 对 prefill **不内联** kvlen（留 Data 引用，接受 PartitionedCall）；decode 仍用 `kvlen=kv_max`（q_len=1 静态可用）。
2. **或干净的混合方案**（不靠 modeling 内 `if` 分支）：例如 prefill/decode 用不同 wrapper 或 modeling 路径，避免 torch.export 双分支常量 lift；或 saver fix 补传递性 dead 常量剥离。
3. gemma4 同样有 prefill 多 chunk 限制（q_len=128 + kv_max 退化同因），FIA prefill 多 chunk 修复需两模型共适用。

### 涉及文件（FIA 调查 + 当前全 eager）

| 文件 | 状态 |
|---|---|
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py` | call site 当前 eager；`fused_attention_forward`/`_pad_mask_to_kv_square` 留 dead code；`causal_mask` always-build（decode 也构造） |
| `cann_lm/converter/export/epair_utils.py` | `inline_fused_attn_seq_constants` 加 `kvlen_value`（committed 2f4755d，全 eager 下 no-op） |
| `cann_lm/converter/saver.py` | decode 调用传 `kvlen_value=kv_cache_max_len`（committed，全 eager 下 no-op）；prefill 多 chunk 限制注释 |
| `docs/cann_lm_models/fused_ops/fused_ops.md` | qwen3.5 OM 算子表（§5.4，全 eager 态） |
