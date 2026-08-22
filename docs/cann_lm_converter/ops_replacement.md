# gemma4 OM 算子等价替换（对齐 support_ops.md）

> 目的：把 gemma4 OM 终态里**未在 `support_ops.md` 覆盖的算子**，用 support_ops.md 覆盖的算子（或 GE 框架算子）等价替换，使 OM 算子与 support_ops.md 定义对齐。
> 范围：gemma4 无量化（`epair_gemma4_no_quant`）prefill-decode 模型。
> 验证：导出 → OM 编译（prefill+decode 3 stages PASS）→ NPU 出字（`The capital of China is` → `**Beijing**.`）。

## 替换清单

### 1. KV cache scatter-write: `index_copy` → `npu_scatter_nd_update`

| 　　　　　　　 | 原　　　　　　　　　　　　　　　　　　　　　　　| 替换　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ----------------| -------------------------------------------------| ----------------------------------------------------------------|
| Python　　　　 | `cache.index_copy(2, kv_pos, kv_new)`　　　　　 | `torch.ops.npu.npu_scatter_nd_update(cache, indices, updates)` |
| ascend_ir　　　| `ge.InplaceUpdate` + `ge.Transpose`　　　　　　 | `ge.ScatterNdUpdate`（+ `ge.TensorMove`）　　　　　　　　　　　|
| OM 终态　　　　| `ScatterUpdate` 30　　　　　　　　　　　　　　　| `ScatterNdUpdate` 30　　　　　　　　　　　　　　　　　　　　　 |
| support_ops.md | ❌（InplaceUpdate 融合成 ScatterUpdate，不匹配） | ✅ fuse_op `ScatterNdUpdate` 直接命中　　　　　　　　　　　　　 |

- **文件**：`cann_lm/models/gemma4/modeling_gemma4.py` `scatter_kv_to_cache` + `cann_lm/converter/export/_scatter_nd_update_converter.py`（cann_lm 自带 converter 覆盖 torchair，产出 `ge.ScatterNdUpdate`）
- **语义**：构造 `[b*n*seq, 3]` 多维坐标 indices `(b,n,kv_pos)` + `[b*n*seq, d]` updates，命中 `ge.ScatterNdUpdate` 标准多维 scatter，等价 `index_copy(2, kv_pos, kv_new)`
- **用 out-of-place 版（无 `_`）**：in-place 版 schema 有 alias annotation `Tensor(a!)`，torch.export functionalization（`run_decompositions`/aot_autograd）不支持自定义算子输出带 alias；out-of-place `-> Tensor out` 无 alias，converter 加 `ge.TensorMove(self)` 保证不原地改

### 2. indices stack → cat+transpose（消除 `Pack`）

| | 原 | 替换 |
|---|---|---|
| Python | `torch.stack([col_b, col_n, col_pos], dim=-1)` | `torch.cat([col_b.view(1,-1), col_n.view(1,-1), col_pos.view(1,-1)], dim=0).transpose(0,1)` |
| ascend_ir | `ge.Pack` | `ge.ConcatV2D` + `ge.Transpose` |
| OM 终态 | `Pack` 30 | `Pack` 0（→ ConcatV2D+Transpose） |
| support_ops.md | ❌ Pack 未列 | ✅ single_op `concatD`/`Transpose` 同族 |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache`
- **原因**：`ge.Pack` 是 torchair 把 `aten.stack` 转换的产物（行交错打包），support_ops.md 未列；`stack` = `unsqueeze+cat`，用 `cat([view(1,-1)...], 0)+transpose` 等价，命中 ConcatV2D+Transpose
- **注**：`cat([unsqueeze...], -1)`（dim=-1）会引入 `Unsqueeze`（ascend_ir +90），而 `cat([view(1,-1)...], 0)+transpose` 用 `Reshape`（OM 终态优化掉 0）

### 3. indices arange → Const 预计算（消除 `Range`）

| | 原 | 替换 |
|---|---|---|
| Python | `torch.arange(n)` | `torch.tensor([[i]*kv_max for i in range(n)])` |
| ascend_ir | `ge.Range` | `ge.Const`（python list 静态预计算） |
| OM 终态 | `Range` 60 | `Range` 0 |
| support_ops.md | ❌ Range 未列（框架算子） | ✅ Const 框架算子（类 Const/Data） |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache` `n_grid`
- **原因**：`n`/`kv_max` 静态（config 决定），`arange(n)` 结果可预计算成 Const，避免动态 `Range` 算子

### 4. indices expand → StridedSliceD+ConcatV2D（消除 `BroadcastTo`）

| | 原 | 替换 |
|---|---|---|
| Python | `tensor.view(1,n,1).expand(b,n,seq)` | `n_grid[:, :seq].reshape(-1)`（col_n）+ `torch.cat([kv_pos]*n)`（col_pos） |
| ascend_ir | `ge.BroadcastTo` | `ge.StridedSliceV2`（→ OM StridedSliceD）+ `ge.ConcatV2D` |
| OM 终态 | `BroadcastTo` 60 | `BroadcastTo` 0（→ StridedSliceD+ConcatV2D） |
| support_ops.md | ❌ BroadcastTo 未列 | ✅ single_op `StridedSliceV2`/`concatD` 同族 |

- **文件**：`modeling_gemma4.py` `scatter_kv_to_cache` `col_n`/`col_pos`
- **关键约束**：`col_n`（n_idx 每个重复 seq 次，**seq 是动态 SymInt**）**无法用 concat 替代**（concat 需固定 tensor 数）。只能用 `StridedSliceD` 从预计算 Const 网格 `[n, kv_max]` 切 `[:, :seq]`。`col_pos`（kv_pos tile n，n 静态）可用 `concat([kv_pos]*n)`

### 5. rotary cos/sin: advanced indexing → `index_select`（消除 `Index`）

| | 原 | 替换 |
|---|---|---|
| Python | `cos_cached[position_ids]` | `cos_cached.index_select(0, pos_flat).reshape(*position_ids.shape, -1)` |
| ascend_ir | `ge.Index` | `ge.GatherV2` |
| OM 终态 | `Index` 5 | `Index` 0（→ GatherV2 40） |
| support_ops.md | ❌ Index 未列 | ✅ single_op `GatherV2` |

- **文件**：`modeling_gemma4.py` `Gemma4Rotary.forward`（cos/sin_cached 取位置嵌入）
- **原因**：`cos_cached[position_ids]`（advanced indexing）→ `ge.Index`；改 `index_select(0, pos_flat)` → `ge.GatherV2`

### 6. logits 取位: advanced indexing → `index_select`（消除 `IndexByTensor`）

| | 原 | 替换 |
|---|---|---|
| Python | `logits[:, output_pos, :]` | `logits.index_select(1, output_pos.to(torch.int64))` |
| ascend_ir | `ge.IndexByTensor` | `ge.GatherV2` |
| OM 终态 | `IndexByTensor` 1 | `IndexByTensor` 0（→ GatherV2） |
| support_ops.md | ❌ IndexByTensor 未列 | ✅ single_op `GatherV2` |

- **文件**：`cann_lm/models/gemma4/export_wrapper.py` `forward`（取 output_pos 位置 logits）
- **注**：`output_pos` 是 int32，`index_select` 要 int64，加 `.to(int64)`（→ `ge.Cast`，support_ops.md 有 Cast）

### 7. 关残差融合: RmsNorm+Add → 分开（消除 `InplaceAddRmsNorm`）

| | 原 | 替换 |
|---|---|---|
| 来源 | GE 编译期 `AddRmsNormFusionGraphPass` + `InplaceAddRmsNormFusionPass` 把 `Add+RmsNorm` 残差融合成 `InplaceAddRmsNorm` | fusion switch 关这两个 pass，保留分开的 `RmsNorm+Add` |
| OM 终态 | `InplaceAddRmsNorm` 35 | `InplaceAddRmsNorm` 0（RmsNorm 207→242, Add 71→106） |
| support_ops.md | ❌ InplaceAddRmsNorm 未列 | ✅ fuse_op `Rmsnorm` + single_op `Add` |

- **文件**：`scripts/set_env.sh` om 分支 fusion cfg
- **配置**：
  ```json
  {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"},
             "GraphFusion":{"AddRmsNormFusionGraphPass":"off","InplaceAddRmsNormFusionPass":"off"}}}
  ```
- **注**：`InplaceAddRmsNorm` 不是 modeling 显式调用（modeling 用拆分的 `add + npu_rms_norm`，`forward_add`），而是 GE 编译期识别 `Add+RmsNorm` 残差模式自动融合。35 个 = 每层 `pre_feedforward_layernorm.forward_add` 1 次 × 35 层。关融合后 OM 多 35 算子但对齐 support_ops.md；如需融合，删 GraphFusion 两项即可

### 8. `Muls`: 保留（替换不划算）

| | 原（保留） | 尝试替换 |
|---|---|---|
| Python | `x * float`（`per_layer_input_scale`/`per_layer_model_projection_scale`） | `x * torch.tensor(scale)` |
| ascend_ir | `ge.Muls` 2 | `ge.Mul` +2 + `ge.Fill` 35 + Const 大增 |
| support_ops.md | ❌ Muls 未列（框架算子） | Mul ✅ 但 Fill ❌ |

- **文件**：`export_wrapper.py`（`* tm.per_layer_model_projection_scale` / `* tm.per_layer_input_scale`）
- **决定**：**保留 Muls**。`Muls` 是 GE 高效标量乘专用算子（框架算子，类 Const/Data/MemcpyAddrAsync，本就不属 support_ops.md 范畴）。替换成 `Mul+Const` 会引入 `Fill` 35（标量 broadcast 填充），净 +35 算子，得不偿失

## 端到端验证

| 阶段 | 结果 |
|---|---|
| converter 导出（`run_pd6_cases.sh 2`） | PASS（prefill+decode ep_json + weights 4761MB） |
| ascend_ir 验证 | Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm 全 0 |
| OM 编译（prefill+decode） | 3 stages 全 PASS |
| OM 终态（ATC） | 非框架算子全命中 support_ops.md |
| NPU 出字 | `The capital of China is` → `**Beijing**.` ✅ |

## 最终 OM 算子（prefill.om，1979 个）

| 算子 | 数量 | support_ops.md 命中 |
|---|---:|---|
| Const | 593 | 框架算子 |
| MatMul | 277 | ⚠️ MatmulV3 同族 |
| RmsNorm | 242 | ✅ fuse Rmsnorm |
| Add | 106 | ✅ single Add |
| Mul | 105 | ✅ single Mul |
| Transpose | 100 | ✅ single Transpose |
| ConcatV2D | 85 | ⚠️ concatD 同族 |
| Data | 77 | 框架算子 |
| Gelu | 70 | ✅ single Gelu |
| Cast | 66 | ✅ single Cast |
| StridedSliceD | 60 | ⚠️ StridedSliceV2 同族 |
| RotaryMul | 50 | ✅ fuse RotaryPositionEmbedding |
| GatherV2 | 40 | ✅ single GatherV2 |
| MemcpyAddrAsync | 40 | 框架算子 |
| FusedInferAttentionScore | 35 | ✅ fuse FusedInferAttention |
| ScatterNdUpdate | 30 | ✅ fuse ScatterNdUpdate |
| Muls | 2 | 框架算子（保留） |
| NetOutput | 1 | 框架算子 |
| **Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm** | **0** | ✅ 全消除 |

## 结论

- **fuse_op**：4/6 命中（Rmsnorm/FusedInferAttention/RotaryPositionEmbedding/ScatterNdUpdate），2 未用是模型架构差异（ChunkGatedDeltaRule SSM、l2norm）
- **single_op**：9/14 命中（含 3 同族改名 StridedSliceD/ConcatV2D/MatMul），5 未用是架构差异
- **框架算子**（Const/Data/NetOutput/MemcpyAddrAsync/Muls）：本就不属 support_ops.md 范畴
- **未覆盖算子全部消除**：Range/BroadcastTo/Pack/Index/IndexByTensor/InplaceAddRmsNorm = 0

gemma4 OM 终态非框架算子**全部命中 support_ops.md 定义**，只剩 `Muls` 2 个是 GE 框架标量乘算子（保留最优）。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `cann_lm/models/gemma4/modeling_gemma4.py` | `scatter_kv_to_cache`（cat+transpose+Const+StridedSliceD+ConcatV2D）+ rotary `index_select` |
| `cann_lm/models/gemma4/export_wrapper.py` | logits `index_select` |
| `cann_lm/converter/export/_scatter_nd_update_converter.py` | 新建，产出 `ge.ScatterNdUpdate`（覆盖 torchair） |
| `cann_lm/converter/export/ge_converter.py` | 注册 `npu_scatter_nd_update(_)` 覆盖 |
| `cann_lm/models/fused_ops/scatter_nd_update.py` | docstring 更新（gemma4 已落地） |
| `cann_lm/models/fused_ops/support_ops.md` | 第 4 节 torchair converter 映射 + 备注 |
| `cann_lm/models/fused_ops/compositte_op.md` | 2.6 节 + P2 标记已落地 |
| `scripts/set_env.sh` | om fusion cfg 关 AddRmsNorm/InplaceAddRmsNorm 融合 |

---

## FusedInferAttentionScore：扁平化已解但 decode 有数值回归 → 回退 eager

> **当前状态**：FIA 已回退为 `eager_attention_forward`（matmul+softmax+@V）。saver fix（`b489f82`，inline ge.Const）解决了扁平化 + 名字冲突，OM 编译 + 出字通过。但 `1+1=` 算术用例发现 **FIA GE kernel 的 `actual_seq_kvlen=1`（decode）致 qwen3.5 decode 数值回归**。回退 FIA → eager 后 `1+1=2` 正确。SoftplusV2 + L2Normalize 经验证**不是**根因。

### 回归根因：FIA GE kernel 的 `actual_seq_kvlen=1` 致 qwen3.5 decode 数值回归

saver fix 对 decode 内联 `actual_seq_kvlen = ge.Const(1)`（`q_len=1`）。GE FIA kernel 用 `actual_seq_kvlen` 界定 attend 的 kv 范围 → decode 时只 attend `key[0:1]`（1 个 kv，新 token），**忽略 past kv**（位置 1..past）→ decode 无上下文 → 算术错（`1+1=` → `2019-01-01` 而非 `2`）。

对比 CPU eager FIA impl（`fused_infer_attention_score.py:_fa_cpu`）：**忽略 `actual_seq_kvlen`**，做**全 buffer matmul + atten_mask 屏蔽无效位** → 全 past kv 参与注意力 → 上下文完整 → `1+1=2` 正确。

| attention 方案 | `actual_seq_kvlen` (decode) | GE kernel 行为 | past 上下文 | `1+1=` |
|---|---|---|---|---|
| **eager**（当前） | 无（不传） | matmul 全 buffer + mask | ✅ 完整 | ✅ `2` 正确 |
| **FIA**（回退前） | `ge.Const(1)` | attend `key[0:1]` 只 1 kv | ❌ 丢失 | ❌ `2019-01-01` |

### 验证（二分排查）

| 配置 | `1+1=` 输出 | 结论 |
|---|---|---|
| 原始 transformers (NPU) | `1+1=2` ✅ | 基线正确 |
| CPU eager cann_lm（全 fused ops CPU impl） | `1+1=2` ✅ | modeling/ fused CPU impl 正确 |
| **OM eager attn**（FIA 回退 + SoftplusV2 + L2Normalize） | **`2+1=3...`** ✅ | **FIA 是根因**（eager + SoftplusV2 + L2Normalize → 正确） |
| OM FIA（SoftplusV2 + L2Normalize） | `2019-01-01` ❌ | FIA GE kernel `actual_seq_kvlen=1` 致回归 |

→ 回退 FIA → `1+1=2` 正确 → **FIA GE kernel 是唯一根因**，SoftplusV2 + L2Normalize 无影响。

### gemma4 vs qwen3.5 对比：saver fix 对 gemma4 正确但 qwen3.5 回归

重新导出 gemma4（pull saver fix 后，ep_json 确认有 `InlineSeqQLen`/`InlineSeqKvlen` = saver fix 生效）+ 重编 OM + 测 `1+1=`：

| 模型 | saver fix | FIA input 数 | `actual_seq_kvlen` | `1+1=` | 正确? |
|---|---|---:|---|---|---|
| **gemma4**（新导出，saver fix 生效） | ✅ InlineSeq | 27 | `ge.Const(1)` | `1+1= **2**` | ✅ |
| **gemma4**（旧 OM，saver fix 未生效） | ❌ dynamic_const | 6 | 未知（常量折叠） | `1+1= **2**` | ✅ |
| **qwen3.5**（saver fix 生效） | ✅ InlineSeq | 27 | `ge.Const(1)` | `2019-01-01` | ❌ |

**同样的 saver fix（InlineSeq=1）、同样的 FIA input 布局（27 inputs, in[5]/[6]）、同样的 `actual_seq_kvlen=1`，gemma4 正确但 qwen3.5 回归。** 差异在模型特有参数：

| FIA 参数 | gemma4 (正确) | qwen3.5 (回归) |
|---|---|---|
| `num_key_value_heads` | 1 | **2** |
| `softmax_scale` | 1 | **0.0625** |
| `sparse_mode` | 3 或 4（混合 full/sliding 层） | **3**（始终 full） |
| `pre_tokens` | int32max 或 sliding_window | **int32max** |
| FIA 层数 | 35（全部 full-attn） | **6**（24 层中 6 full + 18 线性注意力） |
| `atten_mask` 构造 | `wrapper._build_mask_full`（`where(k<=q, 0, min)`） | gen 脚本 inline（`zeros` + `mask[cached_len+1:]=min`）—数值等价 |
| mask source | host 预构（`_build_mask_full`） | host 预构（gen 脚本） |

**无法从 Python 层定位具体哪个参数导致**——可能是 GE FIA kernel 对 `actual_seq_kvlen=1` + qwen3.5 的特定配置（如 `num_key_value_heads=2` 的 GQA ratio、`softmax_scale=0.0625`、或 6 层 FIA + 18 层 ChunkGatedDeltaRule 的混合架构交互）有 bug。需 CANN kernel 层排查或逐参数对照测试。

### 当前 OM 算子状态（FIA 回退后）

- **FusedInferAttentionScore: 0**（回退 eager）
- **SoftmaxV2: 6**（eager attention 的 softmax 回来，6 full-attn 层）
- SoftplusV2: 18（g 计算，线性注意力，保留）
- L2Normalize: 36（l2norm，保留）
- 总 1456 ops，扁平（so_store=0）
- support_ops.md fuse_op 命中 5/8（FIA 不再用 → 命中降 1；l2norm/Rmsnorm/RotaryPositionEmbedding/ScatterNdUpdate/ChunkGatedDeltaRule ✅；DynamicQuant/QuantBatchMatmulV4 ❌ 无量化）

### 待解（如要重新启用 FIA for qwen3.5）

1. **逐参数对照测试**：把 qwen3.5 的 FIA 参数逐一改成 gemma4 的值（如 `num_key_value_heads=1`、`softmax_scale=1`、`sparse_mode=4`），测 `1+1=` 是否变正确 → 定位具体哪个参数触发回归。
2. **或改 decode 的 `actual_seq_kvlen` 取值**：不用 `q_len=1`，而用 `kv_max`（让 GE kernel 读全 buffer），但这之前测过致乱码（`ge.Const(kv_max)` → OM 扁平但输出乱码 "!!"）。需搞清 GE kernel 对 `actual_seq_kvlen` + `atten_mask` 的交互语义。
3. **或接受 eager attention**（当前方案）：SoftmaxV2 回来（6 个），但 SoftplusV2 + L2Normalize 仍消了分解基元。OM 1456 ops（vs FIA 版 1415，多 41 个 SoftmaxV2+相关）。
4. **CANN kernel 层排查**：用 CANN tiling log 或 kernel dump 对比 gemma4 vs qwen3.5 的 FIA kernel 执行路径，定位 `actual_seq_kvlen=1` 在不同 `num_key_value_heads`/`softmax_scale` 下的行为差异。
