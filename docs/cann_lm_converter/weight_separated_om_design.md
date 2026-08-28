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

| 模式 | .om | 权值 | 运行时加载 |
|---|---|---|---|
| `0` (内嵌, 默认) | ~1.6 GB (结构+权值) | 内嵌 .om | `load_from_file` |
| `2` (全合一) | ~33 MB (仅结构) | `<output 目录>/weight/delegate_graph_weight_combined` (~1.5 GB) | `load_from_file` **自动加载旁路 weight/** |

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
/opt/buildtools/Python-3.11.4/bin/python3 run_qwen3_5_gen_emb.py "Hello, my name is"
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

| 路径 | 结果 | 权值 |
|---|---|---|
| BundleBuild 直传两图 (内嵌) | ✅ bundle.om 产出 (3.1 GB) | ❌ 两份 (≈2×1.6 GB, 未去重) |
| ConvertToWeightRefreshableGraphs + BundleBuild | ❌ `CheckVarDesc` 失败 (refreshable 三图只传 infer_graph, variable setup 不完整) | — |
| Bundle + externalWeight | ❌ save 失败 ("Om path is empty", bundle save 不推 weight 路径) | — |

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
| `scripts/cann_lm_compile/run_compile.sh` | 一键 epair→OM 编译 (build+run+存日志) |
| `scripts/cann_lm_compile/package_run.sh` | makeself .run 自解压包打包 |
| `examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py` | 出字验证脚本 (embeddings 入图, 分离 OM) |
| `docs/cann_lm_converter/ops_replacement.md` | OM 算子替换 + 动态变体分析 |
