# FusedInferAttentionScore 静态 kv_max 可行性调查与执行计划

> 承接 [ops_replacement.md](./ops_replacement.md) "FusedInferAttentionScore:qwen3.5 调查与现状" 节。该节结论:qwen3.5 全 eager,FIA decode 修复已验证但暂搁置;gemma4 FIA prefill 用 `kvlen=seq_len`(单 chunk 正确)。
>
> 本文档目标:**找一组 attr/mask 组合,让静态 `kv_max` 对 prefill(q_len>1)也正确**,覆盖 decode + 单/多 chunk prefill,使 FIA 在静态图(无 PartitionedCall)下全场景可用。
>
> 约束:静态(无 PartitionedCall)、两组 SoC(Ascend910 arch22 + Ascend950 arch35)、多 chunk 必须通过。

## 1. 为什么要静态 kv_max

`actual_seq_kvlen`(FIA input[6])语义是"kv 有效长度"。多 chunk prefill 的正确值是逐 chunk 累计的 `new_kv_cache_pos.max()+1`(chunk N = start+seq),**动态、随 chunk 变**,单一静态 `ge.Const` 表达不了。

唯一的静态候选是 `kv_max`(kv buffer 容量,固定):让 kernel 读全 buffer,靠 mask 兜底屏蔽未写位。但实证(见 [ops_replacement.md](./ops_replacement.md) "prefill FIA 限制" 节)显示 `kvlen=kv_max` 对 `q_len>1` 退化(gemma4 全 pad 0 / qwen3.5 卡 "22222"),decode(q_len=1)则正常。

→ **核心问题:退化是 kernel 硬限制,还是可被 attr/mask 绕过?** 若可绕过,静态 `kv_max` 即统一答案;若不可,静态路径死,回退动态 kvlen(算子原生支持,见 §2)。

## 2. FIA 算子对 actual_seq_kvlen 的语义(ops-transformer 源码考证)

来源:gitcode `cann/ops-transformer` 仓 `attention/fused_infer_attention_score/`(本会话已逐文件精读)。

### 2.1 算子定义层

`op_host/fused_infer_attention_score_def.cpp`:
- `actual_seq_lengths_kv`(input[6])`ParamType(OPTIONAL)` + `ValueDepend(OPTIONAL)`——**算子原生支持运行期 data-dependent kvlen**,"静态"是 saver 选择,非算子要求。

`op_host/fused_infer_attention_score_tiling_register.cpp`:
- `TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX, ...}, {TILING_ON_HOST, TILING_ON_AICPU})`——kvlen 在 → tiling 读其 data → PartitionedCall(动态子图);kvlen 缺 → 无此 data dependency → 全静态 tiling。

### 2.2 host tiling 层:`s2Size` 怎么来

`op_host/fused_infer_attention_score_tiling_info_parser.cpp` `FiaInfoParser::GetS2Size`:

```cpp
if (kvStorageMode_ == BATCH_CONTINUOUS) return GetS2SizeForBatchContinuous();
// ...
GetS2SizeForBatchContinuous() {
    if (kvLayout_ == TND || kvLayout_ == NTD) return GetS2SizeFromActualSeqLens();  // 强制读 kvlen tensor
    // BSH/BSND/BNSD:  ← gemma4/qwen3.5 用 BNSD
    s2Size_ = keyShape_->GetShapeS();          // ← 从 key tensor 的 S 维取
    kvListSeqLens_.push_back(s2Size_);
}
```

**关键结论**:BNSD layout 下,`s2Size` = `key.shape[2]` = kv_max(kv buffer 预分配为 `[b,n,kv_max,d]`),**不读 kvlen tensor**。只有 TND/NTD 才强制读 kvlen。

→ **"不传 kvlen" 在 BNSD 下与 "kvlen=Const(kv_max)" 在 `s2Size` 维度完全等价**(都 = kv_max)。差异在 `actualSeqLenFlag_`(见 §3)。

### 2.3 三种"kv_max"表达

| 表达 | s2Size | actualSeqLenFlag | PartitionedCall | 文档测过? |
|---|---|---|---|---|
| `kvlen=Const(kv_max)`(saver fix) | kv_max | true | saver 内联后无 | ✓(退化复现) |
| `kvlen=None`(不传) | kv_max(=key.shape) | **false** | 天然无 | ✗(**未测,本计划 variant B**) |
| `kvlen=Const(seq_len)`(当前 gemma4 prefill) | —(qlen 同) | true | saver 内联后无 | ✓(单 chunk 正确) |

## 3. 退化现象与根因

### 3.1 现象(Ascend910 arch22 实证)

来自 [ops_replacement.md](./ops_replacement.md) "prefill FIA 限制" 节,patch ep_json `InlineSeqKvlen [seq_len]→[kv_max]` 重编 OM 测单 chunk:

| 模型 | q_len | mask | kvlen | 输出 |
|---|---|---|---|---|
| gemma4 | 128 | `[kv_max,kv_max]` padded | kv_max | 全 pad token 0 |
| qwen3.5 | 8 | `[kv_max,kv_max]` padded(实际 `[b,1,seq,kv_max]`,见 §4) | kv_max | `22222...` 卡重复 |
| decode(两模型) | 1 | mask 兜底 | kv_max | **正确** |

已排除变量:mask 构造(kvlen=128/2048 用同一 mask,唯一变量是 kvlen 值);buffer 内容(128..kv_max 是 `torch.zeros`+H2D,非垃圾)。根因落在 kernel 自身。

### 3.2 根因推断

- kernel flash-loop bound = `s2Size` = kv_max,loop 覆盖 `[0:kv_max]`,含 `[actual:kv_max]` 零位块。
- mask 只在 loop **内部**过滤(逐块屏蔽),**不缩 loop bound**。
- `q_len>1` 时,零位 KV 块即便被 mask 屏蔽,仍 corrupt 输出(疑似 online softmax merge 对全掩码块 `max=-inf`/`sum=0` 的数值处理缺陷,或 kernel 不跳过全掩码块而是照常处理零 KV);decode(q_len=1)无此问题。

→ **mask 单独控制不了"实际输入范围"**:loop bound 由 s2Size 决定(来自 kvlen 或 key.shape),mask 兜不住零位块 corrupt。这是 §6 旋钮 A/B/C/D/E 要探的:换 attr/mask 路径,看能否让 kernel 正确跳过零位块或避免数值 corrupt。

### 3.3 mask 形状不是直接杠杆(但仍是可旋钮变量)

FIA 实际收到的 atten_mask 是 **bool 方阵 `[b,1,kv_max,kv_max]`**,不是矩形——经 `modeling_gemma4.py` 的 `_pad_mask_to_kv_square` 把 graph 输入的 float 矩形 `[b,1,seq,kv_max]` 的 q 维 pad 到 kv_max(后 `kv_max-seq` 行全 True=mask out,无 query,不影响因果)。方阵 padding 本身不是多 chunk 问题,但说明 mask rank(方阵 vs 矩形)是可旋钮变量(§5 D)。

## 4. 多 chunk mask 分析

### 4.1 mask 值构造:多 chunk 正确

**gemma4**(`cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full`):
```python
q = new_kv_cache_pos.to(dtype).unsqueeze(1)   # [seq,1] 绝对 buffer 位 (chunk N = start..start+seq-1)
k = torch.arange(kv_max).unsqueeze(0)          # [1, kv_max]
mask = torch.where(k <= q, 0.0, finfo.min)     # [seq, kv_max] 绝对位 causal
```

**qwen3.5**(`examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen.py` 的 inline mask):
```python
q_pos = torch.arange(start_pos, start_pos+PREFILL_SEQ).unsqueeze(1)  # 绝对位
k_pos = torch.arange(KV_MAX).unsqueeze(0)
mask = torch.where(k_pos <= q_pos, 0.0, minv=-1e4)  # 绝对位 causal + pad 行全 mask
```

→ 两模型的 mask 值都用**绝对 buffer 位**,chunk N 的 query@绝对位 `start+q` attend `kv[0:start+q+1]`,**因果正确**。mask 是 runtime 输入(不进图),逐 chunk 重建值即可,图静态。

### 4.2 但 sparse_mode=3 + 静态 qlen 不符号多 chunk

- `sparse_mode=3`(causal)的 kernel 自带 causal,依赖 `actual_seq_qlen` 做绝对定位(算 "query 在绝对位 p attend kv[0:p]")。
- 当前 saver 把 qlen 内联成 `Const(seq_len)`(**chunk 大小**,非累计 `start+seq`)。chunk N(start>0)时 kernel 以为 q 结束在 `seq_len` → query i attend `kv[0:i+1]`(**相对**)→ **丢 `kv[start:start+i]` 上下文**。
- 静态 `ge.Const` 表达不了逐 chunk 累计的 `start+seq`。这是文档"多 chunk FIA 静态 kvlen 无法跨 chunk"的根因之一(qlen 同理)。
- 即便给了正确的绝对位 mask,sparse_mode=3 仍用自己的(基于错误 qlen 的)causal 取交集 → chunk N 上下文丢失。**mask 对了也没用**。

### 4.3 多 chunk 正确所需的最小设置

| 要素 | 当前 | 多 chunk 需要 |
|---|---|---|
| mask 值 | 绝对位 causal ✓ | 绝对位 causal ✓(已对) |
| causal 驱动 | sparse_mode=3(依赖 qlen 绝对定位) | **sparse_mode=1(ALL_MASK)**,代码强制 `pre=MAX/next=MAX`(band 全 attend)→ 显式绝对位 mask 单独驱动 causal,不依赖 qlen(等价:`sparse_mode=0 + pre=MAX/next=MAX`) |
| qlen | saver Const(seq_len)(非累计) | **可不传**(variant B);sparse_mode=1(A')下 qlen 只影响 s1Size(=query.shape=seq,静态)不影响 causal |
| kvlen | saver Const(seq_len/kv_max) | **可不传**(variant B,s2Size=key.shape=kv_max)或修复退化 |

→ 这正是 §6 "variant B + Phase 2 A'(sparse_mode=1)" 目标组合的由来:mask 已多 chunk 正确,缺的是"sparse_mode=1(ALL_MASK)让 mask 独占 causal"+"不传 kvlen/qlen 绕开静态累计值困境"。

## 5. 旋钮空间(均 modeling 层可改,schema/converter 已透传)

`cann_lm/models/fused_ops/fused_infer_attention_score.py`(schema:34-42)、`cann_lm/converter/export/ge_converters/_fused_infer_attention_score_v2_converter.py`(converter:36,48,75,109)对 `inner_precise`/`atten_mask` 均无硬约束(透传),modeling 层改即可。

| 旋钮 | 当前 | 改 | 改动点(file:line) | 假设 |
|---|---|---|---|---|
| **A'** | sparse_mode=3(full)/4(sliding) | 3→**1**(ALL_MASK) | `cann_lm/models/gemma4/modeling_gemma4.py` 的 sparse_mode 设置（full 层;sliding 保持 4） | 代码强制 `pre=MAX/next=MAX`(band 全 attend)→ **显式 mask 单独驱动 causal**。注:`3→0`(NO_MASK)单独是 **no-op**(mode 0 + attr pre=MAX/next=0 + mask 仍走 causal band,与 mode 3 同);要 mode 0 须同时改 `pre=MAX/next=MAX`(记作 A'',与 A' 等价)。**对多 chunk 是必选**(让绝对位 mask 独占 causal)。**概率最高** |
| **B** | inner_precise=0(默认) | 0→**1** | `modeling_gemma4.py` 的 FIA op 调用加 `inner_precise=1` kwarg | 改 softmax 数值路径,若退化是数值污染,高精度档可能正确处理 |
| **C** | mask float(0.0/finfo.min) | float→**bool**(False=attend/True=mask) | `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full`/`_build_mask_sliding` 返回 `~(k<=q)` bool;`_pad_mask_to_kv_square` 跳过 bool 转换 | kernel mask 应用路径分流(加性 score+=min → 乘性/skip) |
| **D** | mask rank 4D `[b,1,seq,kv_max]` | 4D→**2D `[seq,kv_max]`**(MASK_DIM SS) | `export_wrapper.py` 的 `_build_mask_full` 去 `unsqueeze(0).unsqueeze(0).expand(...)`;`modeling_gemma4.py` 的 `_pad_mask_to_kv_square` 跳过 pad | kernel MASK_DIM code path 分流(B1SS→SS) |
| **E** | masked 值 `finfo.min`(gemma4) | →**-1e4**(有限大负) | `export_wrapper.py` 的 `_build_mask_full`/`_build_mask_sliding` 的 `min_val = -1e4` | 避免 -inf 在 online softmax merge 全掩码块传播;**只对 gemma4 有效**(qwen3.5 gen 本就用 -1e4 仍退化,已耗尽) |

### 5.1 self-control mask 构造规范(旋钮 C+D+A' 的组合落地形态)

要让 atten_mask **完全自控**(不依赖 sparse_mode 的 band、不依赖 qlen 绝对定位),把旋钮 C(dtype bool)+ D(rank 矩形)+ A'(sparse_mode=1) 组合成一套 mask 构造规范:

| 维度　　　　| 形态　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 理由　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| -------------| ------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dtype　　　 | `torch.bool`　　　　　　　　　　　　　　　　　　　　　　　　　　 | gemma4 FIA 已用 bool 工作过(经 `_pad_mask_to_kv_square` float→bool);走"乘性/block-skip"路径,避免 float `finfo.min`(-inf)在 online softmax merge 全掩码块传播(退化根因推测) |
| value 约定　| `False=attend, True=mask out`　　　　　　　　　　　　　　　　　　| gemma4 实证(`mask.bool()`:0.0→False=attend, finfo.min→True=mask)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| shape　　　 | `[batch, 1, q_len, kv_max]`(4D B1SS,**矩形**,q_dim=q_len 不 pad) | q_dim 匹配 `actual_seq_qlen`,kv_dim 匹配 `actual_seq_kvlen=kv_max`;跳过 `_pad_mask_to_kv_square` 的方阵 padding(无 query 的 pad 行本就无影响)　　　　　　　　　　　　　　　|
| content　　 | **绝对位 causal**　　　　　　　　　　　　　　　　　　　　　　　　| 多 chunk 正确核心:prefill chunk N `mask[q,k]=(k > start+q)`;decode `mask[0,k]=(k > cached_len)`。query@绝对位 attend `kv[0:绝对位+1]`　　　　　　　　　　　　　　　　　　　|
| sparse_mode | `1`(ALL_MASK)　　　　　　　　　　　　　　　　　　　　　　　　　　| 代码强制 `pre=MAX/next=MAX` → band 全 attend → **显式 mask 单独驱动 causal**(见 §4.2/§5 A' 论证)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

**构造代码**:
```python
# prefill chunk N (start = chunk_idx * seq_len)
q_abs = (start + torch.arange(seq_len)).unsqueeze(1)   # [seq,1] 绝对位
k = torch.arange(kv_max).unsqueeze(0)                    # [1, kv_max]
mask = (k > q_abs)   # bool, True=mask out (k > 绝对位, 含未写位 [start+seq:kv_max])
mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, kv_max)  # [b,1,seq,kv_max]

# decode (position = cached_len)
k = torch.arange(kv_max)
mask = (k > cached_len).view(1,1,1,kv_max).expand(batch,1,1,kv_max)  # [b,1,1,kv_max]
```

**多 chunk 正确性**:chunk N query@`start+q` attend `kv[0:start+q+1]`(mask 屏蔽 `k>start+q`);未写位 `[start+seq:kv_max]` 被 mask(True)屏蔽 → kernel 即便读零位 KV,mask 已标 mask out。配 sparse_mode=1(band 全 attend,不干扰)→ mask 独占因果。decode 同理 attend `kv[0:cached_len]`。

**两个不确定性(Phase 0/2 实证)**:
1. FIA 是否接受**矩形 bool `[b,1,seq,kv_max]`**(q≠kv):`MASK_DIM_B1SS=4` 命名暗示方阵。若强制方阵,保留 `_pad_mask_to_kv_square` padding(pad 行无 query 不影响),只改 dtype 为 bool——Phase 2 D 测。
2. **sparse_mode=1(ALL_MASK)语义**:代码强制 `pre=MAX/next=MAX`(band 全 attend),但公开文档"全mask"措辞歧义。信任代码,Phase 2 A' 实证:若输出全错/全 attend → 语义反了,改用 `sparse_mode=0 + pre=MAX/next=MAX`(A'' 等价)。

**实现改动点**(供 plan 落地):

| 文件:line | 改动 |
|---|---|
| `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full` | 返回 `~(k <= q_abs)` bool(去 float/min_val);保持 `[b,1,seq,kv_max]` 矩形 |
| `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_sliding` | 同上(sliding 层另加窗口条件 `& (diff <= w-1)`) |
| `cann_lm/models/gemma4/modeling_gemma4.py` 的 `_pad_mask_to_kv_square` | 跳过(mask 已 bool 矩形);或保留只做 dtype 校验 |
| `cann_lm/models/gemma4/modeling_gemma4.py` 的 sparse_mode 设置 | `sparse_mode=3` → `1`(ALL_MASK) |
| `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen.py` 的 inline mask | inline mask 改 bool 绝对位;qwen3.5 re-enable FIA 后同步 |

→ 此形态即 §6 Phase 2 **C+D+A'** 组合,也是 §7 决策树"variant B + A"目标组合的 mask 侧面具化。

## 6. 执行计划

### 前置不变量(每轮保持)
- gemma4 FIA 已在 export 激活(`prefill.om` 35 个 FIA),改 modeling 直接影响 export。
- 始终显式 `export SOC_VERSION=...`(`examples/test_converter/export_llm_model/main.cpp` 的 SOC_VERSION 默认值与注释不符)。
- fusion cfg 不动(关 AUB + AddRmsNorm,`scripts/set_env_compile.sh` 的 fusion cfg 段,跨 SoC 复用)。
- 三套 env 互不叠加:`converter`(导出)/`om`(编译)/`npu`(出字)。

### Phase 0 — 双 variant baseline(910 + 950 各测)

| variant | 实现 | actualSeqLenFlag | 测什么 |
|---|---|---|---|
| **A** | `cann_lm/converter/saver.py` 的 prefill kvlen 加 `kvlen_value=kv_cache_max_len`（`kv_cache_max_len` 在 `saver.py` 的 `save_epair_prefill_decode` 参数作用域可见） | true | 复现文档退化(Const 路径) |
| **B** | `cann_lm/models/gemma4/modeling_gemma4.py` 的 FIA op 调用改 `actual_seq_qlen=None; actual_seq_kvlen=None`(不构造 tensor);**saver 不动**(ep_json input[5]/[6] 空 → `cann_lm/converter/export/epair_utils.py` 的 `inline_fused_attn_seq_constants` 遇空 input skip) | **false** | 兜底 tiling 路径退化是否复现/不同(文档未测) |

**执行**(每 variant 每 SoC 一轮):
```bash
# 导出
source scripts/set_env_export.sh
bash examples/test_converter/export_llm_model/cann_lm_cli_export/run_pd6_cases.sh 2   # case_2 = gemma4 prefill-decode

# OM 编译(910)
source scripts/set_env_compile.sh && export SOC_VERSION=Ascend910_9362
# (main.cpp: EpairModelLoader → aclgrphBuildModel → prefill.om)

# NPU 出字(单 chunk,prompt ≤ PREFILL_SEQ=128)
source scripts/set_env_npu.sh
python3 examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_gen.py

# 950 同上,SOC_VERSION=Ascend950DT
```

**判据**:
- 910 variant A:预期退化(全 pad token 0)。variant B:**关键**——若不退化 → 兜底 tiling 路径绕过退化,静态方案有戏。
- 950 任一 variant:**关键分支点**。若不退化 → arch35 kernel 无此问题,950 直接静态 kv_max 可用,**后续所有 Phase 只针对 910**。若退化 → 950 也走 Phase 2。
- 910 任一 variant 不退化(CANN 版本漂移)→ 前提动摇,停。

### Phase 1 — E 旋钮(便宜,基于 baseline OM 不重编)
gemma4 `export_wrapper.py` 的 `_build_mask_full`/`_build_mask_sliding` 的 `min_val` `finfo.min` → `-1e4`,重跑 `run_gemma4_gen.py`(同 OM)。只对 gemma4 有效。
PASS → Phase 5;FAIL → Phase 2。

### Phase 2 — 单旋钮扫(gemma4×910,每轮 re-export+recompile)
基于 Phase 0 更有戏的 variant(variant B 单 chunk 不退化 → 基于 B;否则基于 A)扫 §5 的 A'/B/C/D 4 旋钮,每轮只改一个(其余保持 baseline)。每轮 re-export + recompile + run_gemma4_gen.py 单 chunk。
**判据**:`examples/.../om_npu_verify/run_gemma4_om.py` 的 `maxdiff<1.0 → MATCH` + 首 token argmax 对齐 eager。记录哪个单独命中。

### Phase 3 — 组合(若 Phase 2 无单旋钮命中)
**目标组合 = variant B(modeling None)+ A'(sparse_mode=1)**:真·全静态 + mask 驱动 + 多 chunk 正确(§4.3/§5.1)。优先测此组合的单+多 chunk(即 §5.1 的 self-control mask 形态)。
不命中试 B+A'+C / B+A'+D / A+C(上限 3 组合,gemma4×910)。
仍无命中 → 退化是 kernel 硬限制,静态不可行,回退动态 kvlen(超出本计划目标)。

### Phase 4 — 跨模型×跨 SoC 验证(命中组合)
- **gemma4×950**:命中组合 re-export+recompile on 950,`run_gemma4_gen.py` 验证。
- **qwen3.5**(摩擦大):
  1. `cann_lm/models/qwen3_5/modeling_qwen3_5.py` call site 切回 `fused_attention_forward`(当前全 eager,FIA 是 dead code);
  2. 同步命中旋钮值到 `modeling_qwen3_5.py` 的 sparse_mode（A'）、inner_precise（B）;
  3. mask 旋钮 C/D/E 同步到 `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen.py` 的 inline mask;
  4. re-export+recompile on 910 & 950;
  5. `run_qwen3_5_gen.py` 单 chunk("1+1="→"2")。

### Phase 5 — 多 chunk 终判(硬终判)
命中组合 + 多 chunk prefill:
- gemma4 `run_gemma4_gen.py` prompt>128(多 chunk,跨 chunk 上下文)。
- qwen3.5 `run_qwen3_5_gen.py` 9 chunk 算术递增("11. 11+1=12. 12+1=13.")。
- 910(+950 若 Phase 0 也退化)各跑。
**终判 PASS**:多 chunk 出字正确 + 无 nan + maxdiff<1.0。这是"静态 kv_max 对 prefill 真正可用"的判据。

## 7. 决策树

```
Phase 0 (variant A + B × 910 + 950)
├─ 950 任一 variant 不退化 → 950 静态 kv_max 直接可用,后续只解 910
├─ 910 variant B 不退化(单 chunk chunk 0) → B + A'(sparse_mode=1) 测多 chunk
│   ├─ 多 chunk PASS → 终态 = B + A'(最干净:无 saver fix、无 PartitionedCall、mask 独占 causal) ✓
│   └─ 多 chunk FAIL → 叠 C/D/E
├─ 910 variant B 也退化 → Phase 2 独立扫 A'/B/C/D(基于 A)
│   ├─ 单命中 → Phase 4(跨模型+950)→ Phase 5
│   ├─ 无单命中 → Phase 3 组合(B+A' 优先)→ Phase 4 → Phase 5
│   └─ 仍无 → 退化是 kernel 硬限制,回退动态 kvlen
├─ 910 都不退化(任一) → 直接 Phase 5 多 chunk 终判
└─ 910 baseline 不复现退化 → 前提动摇(CANN 版本漂移),停
```

## 8. 已确认决策

1. **首发模型 = gemma4**(FIA 已在 export 激活,改 saver kv_max 即可复现退化;qwen3.5 需先 re-enable FIA call site,摩擦大,放 Phase 4)。
2. **终判必须多 chunk 过**(多 chunk 是真实 prefill 场景,静态 kv_max 若不能跨 chunk 就没意义)。
3. **950 baseline 不退化 → 950 直接静态 kv_max,只解 910**(arch35 kernel 不同,可能无此退化)。
4. **Phase 2 先 gemma4×910 扫 4 旋钮**(A'/B/C/D),控制 re-export 轮次。
5. **variant B:kvlen/qlen 都不传**(最彻底静态;无 data dependency → 无 PartitionedCall;sparse_mode=1(ALL_MASK)下 qlen 只影响 s1Size=seq 静态,不影响 causal)。

## 9. 关键 file:line 速查

| 主题 | 文件:line |
|---|---|
| saver prefill kvlen=seq_len(当前) | `cann_lm/converter/saver.py` 的 prefill kvlen 设置 |
| saver prefill 改 kv_max(variant A) | `cann_lm/converter/saver.py` 的 prefill kvlen 加 `kvlen_value=kv_cache_max_len`（`kv_cache_max_len` 参数见 `save_epair_prefill_decode` 签名） |
| saver decode kvlen=kv_max(已落地) | `cann_lm/converter/saver.py` 的 decode kvlen 设置 |
| inline_fused_attn_seq_constants 实现 | `cann_lm/converter/export/epair_utils.py` 的 `inline_fused_attn_seq_constants`（kvlen 默认 None→用 qlen;遇空 input skip） |
| modeling FIA call(variant B 改 None) | `cann_lm/models/gemma4/modeling_gemma4.py` 的 FIA op 调用（qlen/kvlen 构造 + op 调用） |
| sparse_mode A' 旋钮(3→1 ALL_MASK) | `cann_lm/models/gemma4/modeling_gemma4.py` 的 sparse_mode 设置（full/sliding） |
| inner_precise B 旋钮 | `cann_lm/models/gemma4/modeling_gemma4.py` 的 FIA op 调用加 kwarg |
| mask 取值 E 旋钮 | `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full`/`_build_mask_sliding` 的 `min_val` |
| mask dtype C 旋钮 | `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full`/`_build_mask_sliding` |
| mask rank D 旋钮 | `cann_lm/models/gemma4/export_wrapper.py` 的 `_build_mask_full` 去 unsqueeze |
| _pad_mask_to_kv_square(float 矩形→bool 方阵) | `cann_lm/models/gemma4/modeling_gemma4.py` 的 `_pad_mask_to_kv_square` |
| qwen3.5 re-enable FIA | `cann_lm/models/qwen3_5/modeling_qwen3_5.py` 的 FIA call site（切回 `fused_attention_forward`，原 dead code 已弃） |
| qwen3.5 gen inline mask | `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen.py` 的 inline mask |
| FIA schema + fake + CPU eager | `cann_lm/models/fused_ops/fused_infer_attention_score.py` 的 schema+fake+CPU impl |
| FIA GE converter | `cann_lm/converter/export/ge_converters/_fused_infer_attention_score_v2_converter.py` 的 GE converter |
| OM 编译入口(SOC_VERSION env) | `examples/test_converter/export_llm_model/main.cpp` 的 SOC_VERSION 设置 |
| OM 编译入口(GE_FUSION_SWITCH_FILE) | `examples/test_converter/export_llm_model/main.cpp` 的 GE_FUSION_SWITCH_FILE 设置 |
| fusion cfg 内容 | `scripts/set_env_compile.sh` 的 fusion cfg 段 |
| CLI 批量导出(case_2=gemma4) | `examples/test_converter/export_llm_model/cann_lm_cli_export/run_pd6_cases.sh` |
| prefill NPU 验证(gemma4) | `examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_gen.py` |
| prefill NPU 验证(qwen3.5) | `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen.py` |
| 数值对照(maxdiff) | `examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_om.py` / `run_qwen3_5_om*.py` |

## 10. 上游 FIA 源码引用(ops-transformer,gitcode `cann/ops-transformer`)

| 结论 | 上游 file:line |
|---|---|
| `actual_seq_kvlen` OPTIONAL + ValueDepend | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp` |
| `TilingInputsDataDependency(ACTUAL_SEQ_KV_INDEX)` | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_register.cpp` |
| BNSD `s2Size = keyShape_->GetShapeS()`(不读 kvlen) | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp` `GetS2SizeForBatchContinuous` |
| TND 才 `GetS2SizeFromActualSeqLens`(读 kvlen) | 同上 |
| `actualSeqLenFlag` 字段 | `attention/fused_infer_attention_score/op_host/fia_tiling_info.h` `class FiaTilingInfo` |
| mask_dim 分流(SS/BSS/B1SS) | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_constants.h` `MASK_DIM_*` |
| sparse_mode 常量(NO_MASK=0/.../BAND=4/TREE=9) | `attention/fused_infer_attention_score/op_host/fia_tiling_info.h` |

## 11. 待验证假设(风险记录)

| 假设 | 验证点 | 风险 |
|---|---|---|
| variant B(actualSeqLenFlag=false)退化行为与 variant A 不同 | Phase 0 variant B 910 | 若退化相同 → B 无优势,仍可作终态实现(更干净)但需叠旋钮 |
| sparse_mode=1(ALL_MASK)让 kernel band 全 attend,显式 mask 独占 causal(不再用自有 causal) | Phase 2 A' | 若 kernel 仍强制自有 causal → 多 chunk 必须动态 qlen,静态死;若 ALL_MASK 语义反了(全屏蔽)→ 改 A''(sparse_mode=0+pre=MAX/next=MAX) |
| q_len>1 退化可被 attr/mask 绕过(非硬限制) | Phase 2-3 | **部分证伪 + 突破**:sparse_mode=3/1 + kvlen=Const/None 均退化;但 **sparse_mode=0 + kvlen=None + bool mask** 成功(§14.9)——移除 band 冲突 + 区段 override 后,kernel honor bool mask(转 -inf)→ 静态可行 |
| 950(arch35)kernel 无此退化 | Phase 0 950 | 若也退化 → 950 也走 Phase 2,工作量翻倍 |
| `_pad_mask_to_kv_square` 跳过(variant D 矩形 mask)被 FIA 接受 | Phase 2 D | 若 FIA 强制方阵 → D 失效,方阵 padding 不影响因果可保留 |

## 12. 终态优先级(若多个组合都 PASS)

1. **✅ sparse_mode=0 + kvlen=None + bool mask(已验证,§14.9)**:真·静态(无 PartitionedCall、无 saver fix)、mask 独占 causal、多 chunk 正确。
2. ~~variant B + sparse_mode=0~~(已并入 1)。
3. 动态 kvlen(违反静态约束,不用)。

## 14.9 突破方案:sparse_mode=0 + kvlen=None + bool mask(2026-08-24 验证)

### 改动(modeling_gemma4.py)
- `fused_attention_forward:261-264`:`sparse_mode` full 层 3→**0**(NO_MASK),sliding 层 4→**0**。
- `fused_attention_forward:255`:`actual_seq_kvlen = None`(不传 kvlen)。
- `actual_seq_qlen` 保持 `torch.tensor([q_len])`(lifted 常量,静态,无 PartitionedCall)。
- mask 构造不变(`_build_mask_full`/`_pad_mask_to_kv_square` bool 方阵 `[B,1,kv_max,kv_max]`,绝对位 causal,逐 chunk 用 `start_pos`)。

### 验证(Ascend910 arch22,静态 graph)

**gemma4(prefill+decode gen,多 chunk)**
| 测试 | prompt | chunks | 输出 | 结果 |
|---|---|---|---|---|
| 单 chunk | "The capital of China is"(14 tok) | 1 | `The capital of China is **Beijing**.` | ✓ |
| 2 chunk | 137 / 168 / 251 tok | 2 | 连贯正确(wedding cake 响应) | ✓ |
| 3 chunk | 275 tok | 3(128+128+19) | `This is an exciting and ambitious project!...` | ✓ |
| graph 静态 | — | — | PartitionedCall=0, tilingDepend=0 | ✓ 静态 |

**qwen3.5(单 OM forward 对比 HF,`run_qwen3_5_om_emb.py`)+ prefill-decode emb 多 chunk(`run_qwen3_5_gen_emb.py`)**
| 测试 | 结果 |
|---|---|
| 单 OM forward: OM top5 == HF top5 == [198, 271, 0, 353, 2921] | ✓ 5/5 MATCH |
| 单 OM forward: OM vs HF logits maxdiff | 0.203(< 5.0 阈值)|
| 单 OM forward: has_nan(主输出 + 48 delta 输出)| False ✓ |
| prefill-decode emb 单 chunk gen(1+1=,4 tok)| ✓ `2+1=3+1=...`(算术递增)|
| **prefill-decode emb 6 chunk gen(48 tok, 每 8 tok)** | ✓ `16+1=17. 17+1=18. 18`(跨 chunk 续数,上下文正确)|

**两模型改动一致**:`fused_attention_forward` `sparse_mode` 3→0、`actual_seq_kvlen=None`;mask 构造不变(bool 方阵绝对位 causal)。qwen3.5 额外:call site `eager_attention_forward`→`fused_attention_forward`。

### 机制(你的"mask -inf 控制有效长度"理论成立)
- **sparse_mode=0(NO_MASK)**:移除强制 causal band → kernel 无自有 band → **被迫用显式 bool mask**。
- **kvlen=None(absent)**:无 kv_max 有效区段 override → mask 的 True(`[actual:kv_max]`)**被 kernel honor(内部转 -inf 等效)→ exp=0 → 排除**。
- **bool mask(方阵,绝对位 causal)**:控制 attend 范围;`[actual:kv_max]`=True → 排除未写区段。

之前失败原因(对比):
- `sparse_mode=3`:强制 causal band 与 mask 冲突 → kernel 用 band 忽略 mask。
- `kvlen=Const(kv_max)`:有效区段=全 buffer → mask 屏蔽区在有效区内不生效。
- **两者都移除(sparse_mode=0 + kvlen=None)→ kernel 纯靠 bool mask → honor → 正确**。

### 为什么 bool 能等效 -inf(你的理论在 arch22 的实现)
arch22 FIA **不接受 float mask**(bf16/fp16 编译失败,`mask_checker` 只认 BOOL/INT8/UINT8)。但 **bool mask 的 True 被 kernel 内部转 -inf**(加性等效:score=QK,True→-inf→exp=0→排除)——**前提是 sparse_mode=0(无 band 冲突)+ kvlen=None(无区段 override)**。在这两个条件下,kernel 纯靠 bool mask,bool 的 True 等效 -inf,你的"mask 控制有效长度"理论成立。

### 限制 / 待验证
- 只在 arch22(910)验证;arch35(950)未测(本机无 950)。arch35 tiling 的 s2Size 也从 key.shape 取,但 sparse_mode=0+kvlen=None 的 kernel honor 行为未在 950 验证(理论应同)。
- mask 仍是 runtime 输入(逐 chunk 重建值,但静态 shape `[B,1,kv_max,kv_max]`)——满足"静态图"(无 PartitionedCall),但 mask 值需逐 chunk 喂(gen 脚本 `build_mask_full` 已支持)。
- sliding 层 sparse_mode=0 + sliding mask(`build_mask_sliding` 窗口 mask):已一起改,3-chunk 测试含 sliding 层,通过。

## 13. 失败回退(已不再需要——§14.9 找到静态解)

**更新(2026-08-24)**:`sparse_mode=0` + `kvlen=None` + bool mask 已验证为静态可行解(§14.9),无需回退动态 kvlen。下列动态路径仅作历史记录(gemma4 旧 baseline 用过):

- 动态 kvlen——算子原生支持(§2.1),modeling `actual_kv_len = new_kv_cache_pos.max()+1`,saver 不内联(留 Data 引用,接受 PartitionedCall)。gemma4 旧 baseline(`view_8`)用此模式。
- decode 仍用 `kvlen=kv_max`(q_len=1 静态可用)——但 §14.9 的静态解对 decode 同样适用(decode 也是 sparse_mode=0 + kvlen=None + mask)。

## 14. 实验结果(2026-08-24, Ascend910 arch22)

> 本机仅 Ascend910 × 2(arch22),**无 950(arch35)**——950 侧未测,留作后续换机。

### 14.1 baseline 验证(kvlen=seq_len=128,单 chunk)— ✓
现有 `prefill.om`+`decode.om`(`/tmp/opencode/gemma4_pd/`),prompt `"The capital of China is"`(14 token < 128 单 chunk)。输出 `The capital of China is **Beijing**.` ✓ 与文档一致。harness + OM 可用。

### 14.2 Phase 0 variant A(kvlen=kv_max=2048,Const,flag=true)— 退化 ✓ 复现
patch prefill ep_json 的 35 个 `InlineSeqKvlen` Const 值 128→2048(qlen 保持 128);重编 `prefill_kvmax.om`。结果:**全 `<pad>`(tok=0)** 退化。复现文档结论。

### 14.3 Phase 2 A'.1(sparse_mode full 3→1 ALL_MASK)— 退化
patch 7 个 full 层 `sparse_mode` 3→1(sliding 28 层保持 4);重编。结果:**全 pad 0** 退化。→ sparse_mode 不是杠杆。

### 14.4 Phase 2 B(inner_precise 0→1)— 退化
patch 35 个 FIA `inner_precise` 0→1;重编。结果:**全 pad 0** 退化(prefill→tok=0)。→ inner_precise 不是杠杆。

### 14.5 C/E 旋钮(N/A,未实测)
- **C(mask dtype float→bool)**:gemma4 FIA **已收 bool mask**(经图内 `_pad_mask_to_kv_square` float→bool 转换),C 已满足,退化仍发生 → mask dtype 非杠杆。
- **E(mask 取值 finfo.min→-1e4)**:bool 转换丢值(`0.0→False`,非零→`True`),finfo.min 与 -1e4 都→`True`(mask)→ E 对 gemma4 FIA 无效。

### 14.6 关键发现:FILL_NONZERO 测试 → kernel 不 honor mask 的 [actual:s2Size] 区段
- 实现:把 chunk 0 past KV buffer `[128:2048]`(原 `torch.zeros`)填非零垃圾 `randn*0.01`(masked 位),用 `prefill_kvmax.om`(kvlen=2048)跑。
- 结果:输出从"全 pad 0"变成 `   \n   \n...`(空白垃圾,非 pad-0,但非正确)。
- **推论**:非零垃圾改变了输出 → kernel **读了 `[128:2048]` 的 kv 值并 attend**。若 mask 被 honor,`[128:2048]` 的值(零或非零)不该影响 q=13 的输出(该区段被 mask 屏蔽);实测影响 → **atten_mask 对 `[actual:s2Size]` 区段未生效**。
- **根因修正**(修订 §3.2 原推断):退化**非"零位 KV 数值 corrupt(online softmax -inf 传播)"**,而是 **mask 应用范围问题**——arch22 kernel 对 `q_len>1 + s2Size>actual` 时,atten_mask 只覆盖 `[0:actual]`,`[actual:s2Size]` 被当成有效 kv attend(无视 mask)。kernel 隐式假设 `s2Size=actual`(kvlen=实际有效长度),对 `s2Size>actual`(kvlen=kv_max)的"额外"区段不应用 mask。

### 14.7 结论
- **Ascend910(arch22)静态 kv_max 对 prefill(q_len>1)不可行**:kernel 不 honor atten_mask 的 `[actual:kv_max]` 区段,attr(sparse_mode/inner_precise)与 mask dtype/rank 均无法修复(mask 已正确提供,被 kernel 忽略)。
- **decode(q_len=1)+kv_max 仍工作**(baseline decode 出字正确):kernel 对 q_len=1 正确 honor mask——退化是 `q_len>1 + s2Size>actual` 专属。
- **mask 已正确**(绝对位 causal,`_build_mask_full` 用 `new_kv_cache_pos`),问题在 kernel 不应用它到额外区段。

### 14.8 路径(按优先级)

**✅ 突破(2026-08-24,已验证):`sparse_mode=0` + `kvlen=None` + bool mask(方阵绝对位 causal)——静态 + 单/多 chunk 全过**

详见 §14.9。这是当前优先级 1 的终态方案。

其余路径(已证伪/次优):
1. 动态 kvlen(modeling `actual_kv_len=new_kv_cache_pos.max()+1`,saver 不内联,走 PartitionedCall)——gemma4 旧 baseline,工作但违反"静态"约束。
2. 950(arch35)测试:arch35 也**不接受 float mask**(`mask_checker::CheckDtypeAndFormat` 拒绝,只 INT8/UINT8/BOOL),与 910 一致;arch35 tiling 的 `s2Size` 仍从 `key.shape` 取(不从 mask shape),矩形 mask 不 bound 读取范围。950 不解此问题。
3. variant B(qlen+kvlen None)/ D(mask rank):s2Size 仍=kv_max,根因不变,已证伪。

