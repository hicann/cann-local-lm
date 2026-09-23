# 权值分离 OM 编译与运行现状

> 日期: 2026-08-28
> 模型: qwen3.5-0.8B
> 环境: CANN 9.2.0, Ascend910 (aarch64), cann_lm_compile 工具

---

## 1. 背景

cann_lm_compile 工具 (`cann_lm/converter/export/csrc/tools/compile_main.cc`) 将 converter 导出的 epair 编译为 CANN OM。模型采用 prefill-decode (PD) 分离模式：prefill 图 (seq=8) 处理 prompt chunk，decode 图 (seq=1) 逐 token 生成。两图共享同一份模型权重（640 个 placeholder 名完全一致），区别仅在 seq 相关的图生成常量（position/mask/arange 等，prefill seq=8 vs decode seq=1 内容不同）。

**目标**：在 PD 分离的前提下，让 prefill + decode 共享一份模型权重完成推理出字，避免权值重复存储（当前两份 .bin 各 1.5 GB = 3 GB）。本文档记录权值分离编译/运行现状、权值差异分析、bundle 共享探索及当前可行方案。

## 2. 权值分离 (GE_EXTERNAL_WEIGHT)

### 2.1 机制

`compile_main.cc` 加 `GE_EXTERNAL_WEIGHT` env knob（`0`=内嵌 / `1`=分文件 / `2`=全合一 `.bin`），映射到 `ge.externalWeight` build option。

| 模式　　　　　　 | .om　　　　　　　　 | 权值　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 运行时加载　　　　　　　　　　　　　　　　|
| ------------------| ---------------------| -----------------------------------------------------------------| -------------------------------------------|
| `0` (内嵌, 默认) | ~1.6 GB (结构+权值) | 内嵌 .om　　　　　　　　　　　　　　　　　　　　　　　　　　　　| `load_from_file`　　　　　　　　　　　　　|
| `2` (全合一)　　 | ~33 MB (仅结构)　　 | `<output 目录>/weight/delegate_graph_weight_combined` (~1.5 GB) | `load_from_file` **自动加载旁路 weight/** |

**关键发现**: `acl.mdl.load_from_file(om_path)` 对分离 OM 自动加载 `.om` 同目录 `weight/` 下的 `.bin`——无需 `aclmdlSetExternalWeightAddress` / config handle，pipeline 无需改动。

### 2.2 约束

- output 路径须与 CWD **同文件系统**：GE 临时权值文件建在 CWD (`tmp_weight_<pid>_0/`)，`aclgrphSaveModel` 时 `MoveFilePath` 用 `rename()` 移到 `<output 目录>/weight/`，跨 fs rename 会 EXDEV 失败。
- `compile_main.cc` 已预建 `<output 目录>/weight/` 目录（GE `MoveFilePath` 不 mkdir）。

### 2.3 出字验证

```
脚本: examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py
模式: prefill+decode 两图分离 OM, embeddings 入图 (非 input_ids)
```

```bash
source scripts/set_env_npu.sh
export PREFILL_OM=$PWD/build/om_output/pd_prefill/prefill.om
export DECODE_OM=$PWD/build/om_output/pd_decode/decode.om
cd examples/test_converter/export_llm_model/om_npu_verify
$DOPT_PYTHON run_qwen3_5_gen_emb.py "Hello, my name is"
```

结果: 3 stages pass, 出字正常 → `' John. I am a 19-year-old male. I am a student at a university. I'`

**注意**: 用 `run_qwen3_5_gen_emb.py`（embeddings 入图变体），不是 `run_qwen3_5_gen.py`（期望 input_ids，报 `unknown OM input embeddings`）。

## 3. 两个 OM 权值文件差异分析

### 3.1 基本对比

| | prefill | decode |
|---|---|---|
| size | 1,589,371,392 B | 1,538,470,4 B (差 13,312 B) |
| md5 | a89cd25d... | 0d62295e... (不同) |
| FileConstant 数 | 339 | 337 (prefill 多 2) |

### 3.2 差异归因

| 类别 | prefill | decode | 说明 |
|---|---|---|---|
| `Const_` (小模型常量/标量) | 2 | 2 | 全共享, 内容一致 |
| `dynamic_const_` 共享 (同 hash) | 325 | 325 | 模型权值 + 公共图常量, **内容一致** |
| `dynamic_const_` prefill 独有 | 7 | — | seq=8 相关 (position/mask/arange) |
| `dynamic_const_` decode 独有 | — | 5 | seq=1 相关 |

**结论**: 模型权值完全一致 (325+2=327 个共享, 同内容 hash)；差异 13,312 B 全来自 12 个 seq 相关的图生成常量 (7+5)，因 prefill(seq=8) 与 decode(seq=1) 的 position/mask 不同。

### 3.3 能否共享 .bin

**不能**。原因:
1. 打包偏移不同 (339 vs 337 FileConstant, offset 全错位)。
2. decode 引用的 5 个 decode 专属图常量在 prefill .bin 中不存在。
3. `load_from_file` 按 OM 硬编码的 offset 从 .bin 取数据，换 .bin = offset 全错。

## 4. 动态算子变体分析

### 4.1 seq 维度: 已固定

- converter `--seq-len` ("Shape-specialized seq_len for torch.export", prefill=8 / decode=1, 具体值)。
- OM 输入 shape 全具体 (如 `attention_mask=[1,1,8,2048]`, 8=seq, 2048=KV_MAX), **0 个符号/None shape**。
- → "seq 可变"是误解；seq 在 export 时已 shape-specialized。

### 4.2 动态变体概念

| | 静态变体 (static variant) | 动态变体 (dynamic variant) |
|---|---|---|
| 绑定 shape | 是, 针对具体 shape 编译 | 否, 处理任意 shape |
| 性能 | 高 (无运行时判断) | 稍低 (运行时分支) |
| 典型 | 标准大小预编译 | LLM 的非标 shape |

`_is_op_dynamic_impl=true` (OM 算子 attr) = 该 op 用了动态变体 kernel。

### 4.3 基线动态算子数

| OM | 总算子 | `_is_op_dynamic_impl=true` | 占比 | PartitionedCall |
|---|---|---|---|---|
| prefill | 1551 | 1158 | 75% | 0 |
| decode | 1423 | 1032 | 72% | 0 |

### 4.4 `GE_JIT_COMPILE=2` 效果

`GE_JIT_COMPILE` (→ `ge.jit_compile`) 作为 **init option** (GlobalOptionsWithSoc / aclgrphBuildInitialize)，**非 build_options** (放 build_options 报 "options illegal" 50331649)。

| | 基线 | `GE_JIT_COMPILE=2` | 变化 |
|---|---|---|---|
| prefill 动态算子 | 1158 (75%) | 838 (54%) | -320 (28%↓) |

剩余 838 (该 SOC 无静态变体): 208 Cast / 108 Transpose / 84 Mul / 79 RmsNorm / 66 Add / 61 ConcatV2D / 60 Swish / 36 TransData / 24 Sigmoid / 24 SplitVD...

### 4.5 TransData (36 个, Conv2D 激活转换)

18× `ConcatV2D→TransData→Conv2D` (ND→NCHW) + 18× `Conv2D→TransData→StridedSliceD` (NCHW→ND)。是 Conv2D 的 NCHW 格式桥接，**非权重 ND→NZ** (权重 FileConstant→MatMul 直连, 0 weight TransData, GE 编译期已处理权重格式)。

## 5. Bundle 权值共享探索

### 5.1 `aclgrphBundleBuildModel` 原型验证

| 路径　　　　　　　　　　　　　　　　　　　　　 | 结果　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　| 权值　　　　　　　　　　　 |
| ------------------------------------------------| ---------------------------------------------------------------------------------| ----------------------------|
| BundleBuild 直传两图 (内嵌)　　　　　　　　　　| ✅ bundle.om 产出 (3.1 GB)　　　　　　　　　　　　　　　　　　　　　　　　　　　 | ❌ 两份 (≈2×1.6 GB, 未去重) |
| ConvertToWeightRefreshableGraphs + BundleBuild | ❌ `CheckVarDesc` 失败 (refreshable 三图只传 infer_graph, variable setup 不完整) | —　　　　　　　　　　　　　|
| Bundle + externalWeight　　　　　　　　　　　　| ❌ save 失败 ("Om path is empty", bundle save 不推 weight 路径)　　　　　　　　　| —　　　　　　　　　　　　　|

### 5.2 结论

- `aclgrphBundleBuildModel` **直传两图 → 权值两份** (3.1 GB, MergeWeights 未去重, Const 没转 variable)。
- `aclgrphConvertToWeightRefreshableGraphs` 产 infer + var_init + var_update **三图**, bundle build 的 `CheckVarDesc` 需完整 variable lifecycle (不只 infer_graph), 简单传 infer_graph 不够 → 断言失败。
- bundle + externalWeight **不兼容** (bundle save 不用 output_file 推 weight 路径)。
- **"一份权值"需更深 CANN bundle + variable 集成** (超出当前原型的简单 BundleBuild 用法)。

### 5.3 权值名一致性 (前提已满足)

prefill + decode 的权值 placeholder 名: 640 个 **全共享** (0 差异), 如 `p_model_lm_head_weight`、`p_model_model_layers_0_input_layernorm_weight` → 两图同名, MergeWeights 去重前提满足。

## 6. 当前可行方案

| 方案 | 权值 | 说明 |
|---|---|---|
| 两次 aclgrphBuildModel + externalWeight=2 (现状) | 两份 .bin (各 1.5 GB) | 模型权值一致但图常量不同, 不能共享; pipeline 无需改 (load_from_file 自动加载) |
| converter 不 bake 图常量 (未实现) | 一份 .bin (1.5 GB) | 图常量运行时计算 → .bin 只含模型权值 → 共享; 需 converter 改 |
| CANN bundle + variable (未跑通) | 一份 (内嵌 ~1.6 GB) | ConvertToWeightRefreshableGraphs → MergeWeights 去重; CheckVarDesc 失败, 需深挖 |

> **§9 深挖结论**："converter 不 bake 图常量"（B）对 qwen3.5 不可行——actual_seq tiling 常量不能改运行时（→PartitionedCall）；GE externalWeight 也是 all-or-nothing（无选择性外置）。
>
> **§11 落地方案（已验证 ✓）**：PD 权值共享走 **externalWeight=1 + 合并 weight 目录**——按内容 sha 命名文件，327 共享同名去重存一份（~1.5GB→共享只存一份，省 ~1.5GB），12 差异项各独立文件（不参与共享，绕开 tiling 难题）。两 OM 同目录共享 weight/，load + 出字验证通过。**这是当前可行的 PD 权值共享路径**（§9 的 actual_seq 阻塞只卡"两 .bin 字节一致"那条路，本方案按文件名去重绕开）。

## 6.1 兜底方案：分档模型 OM 出字

### 思路

在不改动 converter / 不依赖 CANN bundle variable 的前提下，用 **single 模式**（不分 PD）编译一个 **分档 OM**：
- converter 用 `--mode single --seq-len <N>` 导出一个单图 epair（seq=N，不分 prefill/decode）。
- cann_lm_compile 编译该 epair → 一个 OM（内嵌或分离）。
- 只有一个 OM → 自然只有一份权值（无需共享）。

**分档**：针对不同 seq_len 编译多个 OM（如 seq=8 的 prefill 档 + seq=1 的 decode 档），运行时按场景选择对应 OM。

### 与 PD 分离的区别

| | PD 分离 (prefill + decode 两图) | 分档单图 (single) |
|---|---|---|
| 图数 | 2（prefill seq=8 + decode seq=1） | 1（single seq=N） |
| 权值 | 两份 .bin（3 GB） | 一份（1.5 GB） |
| 运行时 | 两个 OMRunner (prefill + decode) | 一个 OMRunner |
| KV cache 跨图传递 | prefill → decode KV 传递 | 同一 OM 内自循环（或外部传） |
| 灵活性 | prefill chunked + decode loop | 固定 seq=N（或分多个档） |

### 验证计划

1. converter 导出 single 模式 epair：
   ```bash
   bash examples/test_converter/export_llm_model/cann_lm_cli_export/run.sh \
     "qwen3_5"  # 默认 single 模式
   ```
2. cann_lm_compile 编译 single epair → OM（内嵌或分离）。
3. 运行时用 `run_qwen3_5_om_emb.py`（单 OM + embeddings 入图验证脚本）跑出字。

### 预期结果

- single OM 只有一份权值（内嵌 ~1.6 GB 或分离 33 MB .om + 1.5 GB .bin）。
- 出字正常（已有 single 模式 OM 验证基础，`run_qwen3_5_om_emb.py` 是 single OM 脚本）。
- 权值无重复 → 省一半存储。

### 局限

- single 模式 seq 固定（不如 PD 分离灵活——PD 的 prefill 可 chunked 处理任意长度 prompt）。
- 若需支持不同 seq 长度，需分档编译多个 OM（每个 seq 一份），存储总量可能反超 PD 分离。
- KV cache 在单图内的处理与 PD 两图传递不同，需确认 `run_qwen3_5_om_emb.py` 的 KV 管理逻辑。

### 适用场景

- 快速验证 / 对比基线（一份权值，确认 OM 出字正确性）。
- seq 长度固定的推理场景（如固定 batch 的 batch 推理）。
- 作为"共享一份权值"的兜底——在 converter 不 bake 图常量 + bundle variable 两条路径落地前的过渡方案。

## 7. 相关 env knob

| knob | → GE option | 位置 | 作用 |
|---|---|---|---|
| `GE_EXTERNAL_WEIGHT` | `ge.externalWeight` | build_options | 权值分离 0/1/2 |
| `GE_JIT_COMPILE` | `ge.jit_compile` | init_options (GlobalOptionsWithSoc) | 在线编译 0/1/2, =2 降动态变体 |
| `GE_OP_BANK_PATH` | `ge.op_bank_path` | init_options | AutoTune 调优库, 产静态 kernel |
| `GE_FUSION_SWITCH_FILE` | `ge.fusionSwitchFile` | init_options | 融合开关配置 |
| `GE_BUFFER_OPT` | `ge.bufferOptimize` | build_options | 图优化级别 |
| `GE_OO_LEVEL` | `ge.oo.level` | build_options | O1/O3 |
| `GE_RUN_MODE` | `ge.graphRunMode` | build_options | 0=推理/1=训练 |
| `GE_SKIP_DSP=1` | graph attr | graph.SetAttr | 跳过动态 shape 分区 |

## 8. 涉及文件

| 文件 | 说明 |
|---|---|
| `cann_lm/converter/export/csrc/tools/compile_main.cc` | 工具主程序: EpairModelLoader → GetGEGraph → aclgrphBuildModel → aclgrphSaveModel; env knob 全在此 |
| `cann_lm/converter/export/csrc/tools/README.md` | 工具使用文档 (安装/构建/运行/样例) |
| `cann_lm/converter/export/csrc/tools/fusion_autoub_off.cfg` | GE 融合开关配置 (静态文件) |
| `cann_lm/converter/export/csrc/inc/external/reg_register.h` | 自研算子注册 (仅 ChunkGatedDeltaRule; 标准算子由 CANN opp 运行时注册) |
| `scripts/cann_lm_compile/build_cpp.sh` | C++ 统一构建入口 |
| `examples/test_converter/export_llm_model/run_compile.sh` | 一键 epair→OM 编译 (build+run+存日志) |
| `scripts/cann_lm_compile/package_run.sh` | makeself .run 自解压包打包 |
| `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py` | 出字验证脚本 (embeddings 入图, 分离 OM) |
| `docs/cann_lm_converter/ops_replacement.md` | OM 算子替换 + 动态变体分析 |

## 9. 权值共享深挖与可行性结论（2026-08-28 探针实证）

> 本节用独立 C++ 探针 + OM dump 实证，裁决 §6 各方案的可行性。**结论：对 qwen3.5（混合线性架构）PD 权值共享被 actual_seq tiling 常量根本性阻塞，唯一干净路径是 §6.1 single 分档 OM。**

### 9.1 方法

1. 构建 `GE_EXTERNAL_WEIGHT=2` 的 single OM（`build/om_output/qwen3_5_single_extw.om` 33MB + `weight/delegate_graph_weight_combined` 1.5GB），`atc --mode=1` dump。
2. 3 个独立 C++ 探针（`/tmp/opencode/probe.cc`、`probe2.cc`，仿 `compile_main.cc`，手建 `FileConstant`+`Const` 图）验证 GE FileConstant 在各 externalWeight 模式下的行为。
3. dump 现有 prefill/decode 内嵌 OM（`epair_qwen3_5_no_quant/{prefill,decode}_s0_nokvlen.om`），按 `(dtype, shape)` 多重集对比 Const 节点定位差异。
4. 查 `libge_compiler.so` 字符串（`ConvertConstToFileConst` / `ConvertFileConstToConst` / `ATTR_NO_NEED_CONSTANT_FOLDING` 等）+ GE 头文件确认无 per-node 外置控制。

### 9.2 GE externalWeight 是纯 all-or-nothing（探针实证）

| externalWeight | 行为 | seq 常量 | 权重 |
|---|---|---|---|
| `0` | `ConvertFileConstToConst` **构建期读文件全内联**（FileConstant→Const+MemcpyAddrAsync，location/offset 消失，数据进 .om） | 内嵌 ✓ | 内嵌 ✗(1.6GB .om) |
| `1`/`2` | `ConvertConstToFileConst` 把 **所有 Const + 预存 FileConstant** 全重安置进受管 combined .bin（location 重写为 `tmp_weight_<pid>_0/<combined>`） | 外置进 .bin ✗ | 外置进 .bin |

- 探针 probe2（预存 FileConstant 指向独立文件 + Const，externalWeight=2）：**预存 FileConstant 也被重安置进 combined .bin**（location 从独立文件变 combined，offset 重分配）。即 GE 不放过任何已存 FileConstant。
- **无 per-node 跳过/大小阈值**：externalWeight=2 下 339 个 Const 全变 FileConstant（连 8 字节标量 `Const_9/10/1628` 都外置），`ATTR_NO_NEED_CONSTANT_FOLDING` 只防折叠不防外置。
- torchair 的 `ge.FileConstant`+`_save_weight2file` 能外置是因走 `.air` + torchair 自有 runtime，**非 acl.mdl OM 路径**，不适用本流程。
- §3.3 / 早期"2 个 Const_ 留内嵌"是误读——那是按文件名前缀数的 FileConstant（仍外置），非内嵌 Const。

### 9.3 真实差异常量

`(dtype, shape)` 对比（prefill 339 Const / decode 337 Const）：

| prefill 独有 (seq=8)　　　　　　　　　　　　　　　　　　　　　 | decode 独有 (seq=1)　　 |
| ----------------------------------------------------------------| -------------------------|
| 4× INT64 [1,1,8]　　　　　　　　　　　　　　　　　　　　　　　 | 4× INT64 [1,1,1]　　　　|
| 2× INT64 [1,16]　　　　　　　　　　　　　　　　　　　　　　　　| 2× INT64 [1,2]　　　　　|
| 1× BOOL [1,1,2040,2048]　　　　　　　　　　　　　　　　　　　　| 1× BOOL [1,1,2047,2048] |
| 2× INT32 [4]/[3]　　　　　　　　　　　　　　　　　　　　　　　 | —　　　　　　　　　　　 |
| **+ 隐藏 value-diff**（同 shape 不同值，dtype×shape 对比漏掉） | 　　　　　　　　　　　　|

**隐藏 value-diff**（被对比漏掉，但内容不同、进 .bin 致 .bin 不一致）：
- `actual_seq_qlen`：INT64 [1]，~~值 q_len=8/1~~（⚠️ §11.1 实测修正：实际 lifted 成 **8/8 同值共享**，非差异项；§9 此处基于假设，错）
- `actual_seq_lengths`：INT32 [1]，值 seq_len=8/1（18 linear 层 delta rule，modeling:600）

### 9.4 根因：混合架构 + 两类 actual_seq tiling 常量

**Qwen3.5-0.8B 是 hybrid 架构**：24 层 = **18 `linear_attention` + 6 `full_attention`**（HF `text_config.layer_types`）。

| 层类型　　| 算子　　　　　　　　　　　　　　　　 | per-OM tiling seq 常量　　　　　　　　　　　　　　　　　　　　　　　 | 值　　|
| -----------| --------------------------------------| ----------------------------------------------------------------------| -------|
| 6 full　　| `npu_fused_infer_attention_score_v2` | `actual_seq_qlen = torch.tensor([q_len])` (modeling:212)　　　　　　 | 8 / 1 |
| 18 linear | `npu_chunk_gated_delta_rule`　　　　 | `actual_seq_lengths = torch.full((B,),seq_len,int32)` (modeling:600) | 8 / 1 |

两者都：① per-OM（prefill=8/decode=1，内容不同）② **tiling 依赖**（告诉 kernel 实际序列长度，编译期须具体值才能静态 tiling）③ 不能改运行时输入（值未知→动态 tiling→PartitionedCall）④ 被 GE 外置进 .bin → **.bin 必差，无法共享**。

`inline_fused_attn_seq_constants`（epair_utils:376）的 docstring 记录了这条历史：lifted 常量按名跨图共享会被 GE 标 `tilingDepend → PartitionedCall`，故改成 per-graph 内联 Const 保静态——代价就是 prefill/decode 值不同、.bin 不一致。

### 9.5 各方案可行性裁决

| 方案　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 裁决　| 原因　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------------------------------------------------------------| -------| -----------------------------------------------------------------------------------------------------------------------------------------------------------|
| A 权重→FileConstant + seq 内嵌 Const (externalWeight=0)　　　　　　　| ❌　　 | GE `ConvertFileConstToConst` 把 FileConstant 也读文件内联，权重进 .om（1.6GB）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| A' 预存 FileConstant(独立文件) + 权重 Const (externalWeight=2)　　　 | ❌　　 | GE 把预存 FileConstant 也重安置进 combined .bin（探针 probe2 实证）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| B seq 常量改运行时 Data 输入　　　　　　　　　　　　　　　　　　　　 | ❌　　 | actual_seq tiling 常量不能改运行时（→PartitionedCall）；position/mask 已是运行时输入且 0 PartitionedCall，但 actual_seq 不行　　　　　　　　　　　　　　　|
| B' eager attention（去 FIA）　　　　　　　　　　　　　　　　　　　　 | ❌(半) | 6 full 层解决（去 actual_seq_qlen + 方形 pad mask）；18 linear 层 `actual_seq_lengths` 仍在；且 eager delta rule → "19418 节点 → GE segfault"（必须融合） |
| OM 后处理手术（按 `_value_sha256` 导共享 .bin + patch FileConstant） | ❌　　 | 无 GE load/modify/save API，裸 hack 二进制脆弱不支持　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **C single 分档 OM (§6.1)**　　　　　　　　　　　　　　　　　　　　　| ✅　　 | 单图一 seq 一份 actual_seq 一份 .bin，无需共享　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

### 9.6 两个追问的回答

- **Q1：去掉 FIA 的 lifted 张量能否静态转出？** 不能。lifted 张量即 `actual_seq_qlen`，是 `npu_fused_infer_attention_score_v2` 的**必填输入**（modeling:225 传入）。只有 `actual_seq_kvlen` 有 None-scheme（kernel 读全 buffer、mask 兜底，:213）；`actual_seq_qlen` **没有** None-scheme，去掉 → FIA 无法确定 q_len → 构建失败/结果错。
- **Q2：换回 eager 能否同时规避两问题？** 只解一半。6 full 层换 eager 去掉 `actual_seq_qlen` + `_pad_mask_to_kv_square`(:171) 方形 pad mask → 这 6 层无动态分支、无差异 ✓；但 18 linear 层仍用 `npu_chunk_gated_delta_rule`，其 `actual_seq_lengths` 是另一个 tiling 依赖常量，仍 per-OM、仍不能改运行时、仍进 .bin → **.bin 仍不一致，共享仍被卡** ✗。

### 9.7 结论

PD 权值共享对 qwen3.5（混合线性架构）被 **actual_seq tiling 常量**根本性阻塞——FIA（6 full 层）和 delta rule（18 linear 层）各有 per-OM actual_seq 常量，都 tiling 依赖、不能改运行时、被 GE 外置进 .bin。**这不是 converter 能绕的，是 op 设计层约束。**

- 兜底 = **§6.1 single 分档 OM**：针对每个 seq_len 编译一个 single OM，运行时按场景选。一图一 seq，自然一份 actual_seq、一份 .bin，无共享需求。
- **实际落地方案 = §11.2 externalWeight=1 + 合并目录**（已验证）：按内容 sha 文件名去重，327 共享权重存一份、12 差异项独立文件，**绕开 actual_seq tiling 阻塞**（差异项不参与共享），PD 权值共享达成（~1.5GB 省 ~1.5GB，load + 出字验证通过）。
- 真正实现 PD 共享需 **op 层改造**：让 FIA / delta rule 从静态 query shape 推导实际序列长度、不收 `actual_seq` 输入（或提供"值编译期已知"的 Data 输入语义），超出 converter 范围。

### 9.8 探针与产物

| 产物 | 说明 |
|---|---|
| `/tmp/opencode/probe.cc` | externalWeight=0 手建 FileConstant→output，验证 GE 内联（Const+MemcpyAddrAsync） |
| `/tmp/opencode/probe2.cc` | 预存 FileConstant + Const，externalWeight=1/2，验证 GE 重安置预存 FileConstant 进 combined .bin |
| `build/om_output/qwen3_5_single_extw.om` + `weight/delegate_graph_weight_combined` | externalWeight=2 single OM（33MB + 1.5GB），dump 确认 339 全 FileConstant |
| `/tmp/opencode/single_om` / `single_extw` / `prefill_om` / `decode_om` | atc --mode=1 dump JSON |
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py:150/186/212/600/621` | eager/fused/delta-rule attention 实现与 actual_seq 常量来源 |
| `cann_lm/converter/export/epair_utils.py:376` | `inline_fused_attn_seq_constants`（lifted→PartitionedCall 历史） |

## 10. 动态 seq 导出 + CANN 分档探索（2026-08-28 实证，路径不可行）

> §9 结论"PD 共享需 op 层改造让 FIA/delta rule 从 query shape 推导 actual_seq"的工程化尝试：把 seq 导出为**动态 Dim**（一个 epair 覆盖 seq 范围），OM 编译用 CANN **分档**（每档静态 kernel），目标一个 OM/一份 .bin。**全程实证：导出链路突破，但分档/运行时被 CANN 预构图路径双 gap 卡死，路径不可行。**

### 10.1 导出链路突破（方案A：actual_seq 从 ge.Shape 派生）

改 `saver.py` 加 `--dynamic-seq`（`dynamic_shapes` 标 seq 维为 `Dim(min=1,max=kv_max-2)`），逐障碍修复后动态 epair 导出成功（`s1` range 1..2046）：

| 障碍 | 修法 |
|---|---|
| mask pad 守卫（`_pad_mask_to_kv_square` 的 `q_len>=kv_len` + `(kv-seq)!=1`）| `Dim max=kv_max-2` |
| conv1d converter 动态 spec（`list(mo.size)` 断言）| 建 `_spec_utils.spec_size_dims`（用 `_symsize`，符号维填 -1）|
| delta-rule converter 动态 spec | 同 helper |
| **actual_seq 物化**（`torch.tensor([q_len])`→`scalar_tensor`，torchair `dtype_promote` 不解析符号 SymInt→ge.Tensor backing）| **方案A**：modeling 传 None + FIA/delta converter 从 `ge.Cast(ge.Gather(ge.Shape(query),[seq_axis]),int64)` 派生 |

actual_seq 不再 bake 成 per-OM 常量（动态下为运行时 `ge.Shape` 派生张量），从根上消除了 §9 的 actual_seq per-OM 差异源。

### 10.2 分档编译——CANN 预构图路径双 gap（不可行）

`compile_main.cc` 加 `ge.inputShape`+`ge.dynamicDims`（init option，离散 seq bins 如 "1;8"），`ge_graph_builder.cc` 分档路径把 Data 动态维设 -1。实证：

| 配置 | 结果 |
|---|---|
| `ge.inputShape`/`ge.dynamicDims` 作 build_options | `50331649 options illegal`（须 init option）|
| 作 init_options + `-1` dims | 编译"成功"但 **slog 0 解析**（GE 完全忽略）|
| runtime gearCount | 全 0（runtime 不识别分档模型）|
| `aclmdlSetInputDynamicDims` | `145003`（非 dynamic_dims 模型）|

**根因（CANN 机制 gap）**：
- `ge.inputShape`/`ge.dynamicDims` **只对 atc/framework 路径生效**（GE 从框架模型构图时应用），对 **epair 预构图（aclgrphBuildModel）路径完全无效**——Data 节点已在图里，GE 不再经 inputShape 处理。
- OM 的 2 个 PartitionedCall 实为 **DSP（shape_range 连续动态分区，`MultiBatchClonePass`+`CondRemovePass` Case→PartitionedCall）**，非离散分档分派。
- 离散分档需 `ge.dynamicDims`（对 epair 忽略）；手动构造 Case/mbatch 节点 = 复刻 GE 内部 `MultiBatchClonePass`，极复杂脆弱。

### 10.3 DSP 兜底——runtime 不可配置 → 结果全 0

shape_range 路径的 OM 跑 unknown-shape runtime（设输入 tensor desc + generous malloc + `acl.mdl.execute`）：**execute 成功但 logits 全 0**（MISMATCH HF）。根因：DSP 模型 runtime 不可配置（gearCount=0、SetInputDynamicDims=145003）→ 2 PartitionedCall 的 Case 分派无 runtime shape 设置 → 走错分支 / `ge.Shape` 派生的 actual_seq 取错值 → FIA tiling 错 → 全 0。

### 10.4 结论

**动态 seq + 分档路径对 epair（aclgrphBuildModel 预构图）不可行**——两个 CANN 机制 gap：
1. `ge.dynamicDims` 离散分档对预构图忽略（0 解析）；
2. DSP（shape_range）模型 runtime 不可配置（gearCount=0/SetInputDynamicDims=145003）→ 跑通但结果错。

**真正成果（动态导出链路，可复用）**：
- `saver.py` `--dynamic-seq` + dynamic_shapes
- `cann_lm/models/qwen3_5/modeling_qwen3_5.py`：`actual_seq_qlen`/`actual_seq_lengths` 传 None + `_is_dynamic_dim` helper（不发守卫检测符号）
- `_fused_infer_attention_score_v2_converter.py`/`_chunk_gated_delta_rule_converter.py`：actual_seq 从 `ge.Shape`+gather 派生
- `_spec_utils.py` + `_conv1d_converter.py`/`_chunk_gated_delta_rule_converter.py`：动态 spec（`_symsize`+符号维 -1）
- `compile_main.cc`：`ge.inputShape`/`ge.dynamicDims` init knob（对 epair 无效，但对 atc 路径有用，保留）

这些**真正破解了 actual_seq-under-dynamic 这个 op 级硬核**（actual_seq 从 per-OM baked 常量变为运行时 `ge.Shape` 派生），是后续若 CANN 支持预构图分档、或换 atc 编译路径时的现成基础。

**落地仍回 §6.1 single 分档 OM 兜底**（已验证可行）；动态 seq+分档待 CANN 预构图分档支持或换编译路径。

### 10.5 动态探索产物

| 产物 | 说明 |
|---|---|
| `examples/.../cann_lm_cli_export/epair_qwen3_5_dynseq/` | 动态 seq epair（s1 range 1..2046）|
| `build/om_output/qwen3_5_dynseq_linux_aarch64.om` + `weight/` | 分档尝试 OM（207MB + 1.5GB，实际是 DSP 非分档，结果全 0）|
| `/tmp/opencode/probe_gears.py` / `probe_runtime.py` | runtime gear/输出 size/SetInputDynamicDims 探针 |
| `examples/.../om_npu_verify/run_qwen3_5_om_emb.py` run_om | 已适配 unknown-shape（设输入 tensor desc + 动态输出 malloc）|

## 11. PD 权值差异精确分析 + 权值共享最终方案（2026-08-28 实证）

> §9 曾基于 `(dtype,shape)` 对比 + 假设判断差异源；本节用 **`_value_sha256` 内容级对比 + 实际值读取**重做精确分析，并**修正 §9 的关键错误**，给出**落地的权值共享方案**（externalWeight=1 + 合并目录）。

### 11.1 externalWeight=2 权值差异精确分析（内容级）

**方法**：构建 prefill + decode `externalWeight=2` OM，dump（atc --mode=1），对 FileConstant 按 `_value_sha256`（内容哈希）对比（非 dtype/shape）。

**结果**：prefill 334 FC / decode 332 FC，**327 共享**（同内容 sha）+ prefill 独有 7 + decode 独有 5 = **12 个真差异项**。

#### 327 共享项——含 §9 关键修正

327 个常量 prefill/decode 内容完全一致（模型权重 + 共享图常量）。**其中含 FIA 的 `actual_seq_qlen`（`lifted_tensor_2`）= 8，prefill 和 decode 都是 8（同 sha，共享）**。

> ⚠️ **修正 §9.4/9.6 的错误**：§9 曾假设 FIA `actual_seq_qlen` = 8(prefill)/1(decode) 是差异源 + tiling 阻塞——**实证错**。它被 lifted 成同值 8（两图共享），**不是差异源、不是阻塞**。（decode 的 FIA q_len=8 而非 1，是新 scheme 下 `actual_seq_kvlen=None`、纯靠 bool mask 控制有效区段，`actual_seq_qlen` 不关键的体现。）

#### 12 个真差异（值 + 来源）

| # | prefill | decode | 来源（modeling） | 性质 |
|---|---|---|---|---|
| 1 | `INT32 [1]=[8]` | `INT32 [1]=[1]` | `actual_seq_lengths=torch.full((B,),seq_len,int32)` (:600, delta rule, 18 linear 层折叠→1) | **TILING**（值=seq）|
| 2 | `BOOL [1,1,2040,2048]` | `[1,1,2047,2048]` | `_pad_mask_to_kv_square` pad 区 (:171-183, FIA) | FIA 专用，内容=pad 区 |
| 3 | `INT64 [1,16]=[0×16]` | `[1,2]=[0,0]` | `col_b=col_n*0` (:418, scatter_kv_to_cache, batch 索引) | 长度=n·seq |
| 4 | `INT64 [1,16]=[0×8,1×8]` | `[1,2]=[0,1]` | `col_n=n_grid[:,:seq]` (:413-416, head 索引, n=2) | 长度=n·seq |
| 5 | `INT64 [1,1,8]=[2048..2055]` | `[1,1,1]=[2048]` | kv_max+arange(seq) 位置偏移张量（常量折叠生成，原 op 丢失） | 值=kv_max+i |
| 6 | `INT32 [4]=[0,2,1,3]` | — | transpose perm（prefill 独有） | 静态 perm |
| 7 | `INT32 [3]=[0,2,1]` | — | transpose perm（prefill 独有） | 静态 perm |

#### 差异分类

1. **TILING 依赖（唯一真阻塞，不能改运行时）**：仅 **delta rule 的 `actual_seq_lengths`**（:600，值 8/1，18 层折叠→1）。FIA 的 `actual_seq_qlen` 已被证伪（共享=8）。
2. **FIA/KV 派生（非 tiling，seq 相关）**：FIA 方形 pad mask (#2)、KV scatter 索引 col_b/col_n (#3-4, 长度 n·seq)、kv 位置偏移 (#5)。
3. **静态 perm（prefill 独有，#6-7）**：`[0,2,1,3]`/`[0,2,1]`，prefill 路径特有 transpose，非 seq 值依赖。

#### 结论（精确）

导致 externalWeight=2 两 .bin 不一致的是这 12 个 seq 依赖常量（内容不同 + 打包 offset 错位）。真正卡"两 .bin 字节一致可共享"的 tiling 常量**只有一个：delta rule 的 `actual_seq_lengths`**（§9 说的"两类 actual_seq"实为一类）。但 §3.3 的结论不变：externalWeight=2 下 .bin offset 错位 + 差异项混入 → 两 .bin 不可直接共享。

### 11.2 权值共享最终方案：externalWeight=1 + 合并 weight 目录（已验证 ✓）

**思路**：不要求两图 .bin 字节一致（被 actual_seq tiling 卡死），改用 **externalWeight=1（每权重一文件，按内容 sha 命名）+ 合并两 OM 的 weight 目录**。同内容权重 → 同 sha → 同名文件 → 合并去重存一份；差异项各存独立文件（本就 per-OM 不同，不参与共享）→ **绕开 tiling 难题**。

**步骤**：
1. `GE_EXTERNAL_WEIGHT=1` 构建 prefill + decode OM（各产 `weight_<内容sha>` 文件）。
2. 合并两 OM 的 `weight/` 目录（同名去重）。
3. 两 OM 放同目录共享 `weight/`，`acl.mdl.load_from_file` 按 FileConstant 的 `location=weight_<sha>` 自动找文件。

**结果**：

| 　　　　　　　　 | externalWeight=2（原现状）　　　　　　　 | **externalWeight=1 + 合并（本方案）**　　　　　　　　　　|
| ------------------| ------------------------------------------| ----------------------------------------------------------|
| prefill 权值文件 | 334 FC 进 1 个 combined .bin　　　　　　 | 334 个 `weight_<sha>` 文件　　　　　　　　　　　　　　　 |
| decode 权值文件　| 332 FC 进 1 个 combined .bin　　　　　　 | 332 个 `weight_<sha>` 文件　　　　　　　　　　　　　　　 |
| 合并后　　　　　 | 两 .bin 各 1.5GB = **3GB**（共享重复存） | **339 文件 = 1.6GB**（327 共享去重存一份 + 12 差异独立） |
| 节省　　　　　　 | —　　　　　　　　　　　　　　　　　　　　| **~1.5GB**（327 个共享权重只存一份）　　　　　　　　　　 |

**验证**：
- 两 OM `load_from_file` 从共享 `weight/` 目录均成功（53 输入/49 输出）。
- PD 出字（`run_qwen3_5_gen_emb.py "Hello, my name is"`）→ `' John. I am a 19-year-old male. I am a student at a university. I'`（与 §2.3 externalWeight=2 出字一致，正确）。

**为何这条路通**（对比 §5 bundle / §9-10 动态+分档 均失败）：按**文件名（内容 sha）去重**而非要求 .bin 字节一致——相同内容自然同名、合并去重，差异项（含 tiling 的 `actual_seq_lengths`）作为独立文件**不参与共享**，tiling 难题被绕开。无 GE 机制 gap（externalWeight=1 是 GE 原生支持，runtime `load_from_file` 自动按 location 找文件）。

### 11.3 落地建议

- **PD 权值共享走 externalWeight=1 + 合并目录**（§11.2，已验证）。
- `compile_main.cc` 已有 `GE_EXTERNAL_WEIGHT` env knob（0/1/2），用 `=1`。
- 部署：两 OM 同目录 + 合并后的 `weight/`（339 文件，1.6GB）；`load_from_file` 自动加载，pipeline 无需改（同 §2.1 externalWeight 的运行时机制，只是分文件而非合一）。
- 局限：339 个小文件（vs 1 个 .bin），文件系统 inode/打开句柄开销略增；若需单文件部署，可打包（但失去按内容去重）。

## 12. gemma4 PD 权值对比（externalWeight=2，2026-08-28 实证）

> §11 以 qwen3.5（混合线性架构，18 linear + 6 full）为对象。本节对比 **gemma4（标准 transformer，全 FIA、无 delta rule）** 的 PD externalWeight=2 权值一致性，验证不同架构下的差异面。

### 12.1 方法

构建 gemma4 prefill + decode `externalWeight=2` OM（`epair_gemma4_no_quant`），dump 后按 `_value_sha256` 内容级对比。

### 12.2 结果

| | prefill | decode |
|---|---|---|
| FileConstant 数 | 555 | 552 |
| **共享（同内容 sha）** | **552（99.5%）** | |
| prefill 独有 | **3** | — |
| decode 独有 | — | **0**（decode ⊂ prefill）|

### 12.3 3 个 prefill 独有差异项（全非 tiling）

| # | dtype/shape | 值 | 说明 |
|---|---|---|---|
| 1 | `INT64 [4]` | `[0,2,1,3]` | transpose perm（int64）|
| 2 | `INT32 [4]` | `[0,2,1,3]` | transpose perm（int32）|
| 3 | `INT64 [1,128]` | 全 0 | 零张量（prefill 独有）|

### 12.4 actual_seq_qlen（共享，非差异）

`c_model_model_layers_0_self_attn_lifted_tensor` = **128，shared=True**（prefill+decode 同值共享，同 qwen3.5 的 lifted 模式）。gemma4 **无 delta rule**（标准 transformer）→ 无 `actual_seq_lengths`；`actual_seq_kvlen` 从 `new_kv_cache_pos.max()+1` 运行时算（modeling:243，非 baked）。**故 gemma4 无任何 actual_seq tiling 差异项**。

### 12.5 与 qwen3.5 对比

| | qwen3.5（混合线性）| gemma4（标准 transformer）|
|---|---|---|
| 差异项总数 | 12（7 prefill + 5 decode）| 3（全 prefill，decode 0）|
| tiling 差异项 | 1（delta `actual_seq_lengths` 8/1）| **0**（无 delta rule；FIA actual_seq_qlen lifted 共享=128）|
| decode ⊂ prefill? | 否（decode 独有 5）| **是**（decode 独有 0，全在 prefill 内）|
| externalWeight=1+merge 文件数 | 339（334+332 去重）| 555（555+552，decode 全⊂prefill，不增文件）|
| 权值一致性 | 327/334（97.9%）| 552/555（99.5%）|

### 12.6 结论

**gemma4 PD 权值高度一致**（552/555 共享，99.5%），3 个 prefill 独有差异项全非 tiling（2 transpose perm + 1 零张量），decode 是 prefill 严格子集。**无 actual_seq tiling 阻塞**——gemma4 无 delta rule，FIA actual_seq_qlen lifted 共享=128，actual_seq_kvlen 运行时算。

对比 qwen3.5（12 差异含 1 tiling、decode 有独有项），**gemma4 远更易共享**：差异面更小（3 vs 12）、无 tiling、decode ⊂ prefill（externalWeight=1+merge 仅 prefill 的 555 文件即覆盖 decode）。佐证 §11 的结论——PD 权值共享走 externalWeight=1 + 合并目录对两种架构都适用，且 gemma4 这类标准 transformer 共享率更高。
