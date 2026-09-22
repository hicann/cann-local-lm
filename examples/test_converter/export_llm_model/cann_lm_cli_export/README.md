# cann_lm_cli_export — cann_lm_converter.py CLI 测试

用 `cann_lm/converter/cann_lm_converter.py` CLI 把 gemma4 / qwen3.5 两个 HuggingFace 模型导出为 epair 格式,并验证产物完整性。

## 文件

| 文件 | 作用 |
|------|------|
| `run.sh` | source 环境后循环调 `cann_lm_converter.py` 导出两模型,再调 `verify.py` |
| `verify.py` | 检查每个 `epair_<model>/` 目录产物:ep_json / model_config / weight_info / weights_*.bin / tokenizer |

## 用法

```bash
# 默认: 全部两模型, single 模式, bfloat16, seq_len=8 (快测)
bash run.sh

# 指定模型
bash run.sh "gemma4 qwen3_5"

# prefill-decode 模式 (prefill+decode 两图 + 共享权重, 推理用)
MODE=prefill-decode bash run.sh gemma4

# 指定 dtype / seq_len
DTYPE=float16 SEQ_LEN=128 bash run.sh

# 只导出前 N 层 (slice/debug, 截断 model.model.layers, 不改 config)
NUM_LAYERS=5 bash run.sh

# 组合: 前 5 层 + prefill-decode 模式
NUM_LAYERS=5 MODE=prefill-decode bash run.sh gemma4
```

环境变量(均有默认):

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODE` | `single` | `single`(一张形状特化图) 或 `prefill-decode`(prefill+decode 两图) |
| `DTYPE` | `bfloat16` | 两模型原生 bfloat16 |
| `SEQ_LEN` | `8` | single 模式 `--seq-len`;prefill-decode 模式 `--seq-len` |
| `NUM_LAYERS` | (空=全部) | 只导出前 N 层(`--num-layers`),截断 `model.model.layers`,不改 `config.num_hidden_layers` |
| `GEMMA4_HF_PATH` | (必填) | gemma4 HF 模型目录;`run.sh` 内映射到 `[gemma4]` |
| `QWEN3_5_HF_PATH` | (必填) | qwen3.5 HF 模型目录 |

> `run.sh` 已 `source $PROJECT_ROOT/scripts/set_env_export.sh`(① converter 基底),自动设好 `CONVERTER_PYTHON`(=.venv/bin/python)、`PYTHONPATH`、stubs `LD_LIBRARY_PATH` 等;HF 路径由调用方必填 export。

> `--use-embeddings` 默认 **True**（纯 embeddings，图外查表；CLI 自带，`run.sh` 未暴露开关）。要图内查表直接调 `cann_lm_converter.py ... --no-use-embeddings`。`--mode` 默认 `prefill-decode`（CLI）；`run.sh` 的 `MODE` 默认 `single`（快测）。

## 测试结果

### single 模式 (`MODE=single`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9645 MB | PASS | 562 权重, sliding+full attention, KV 共享 |
| qwen3_5 | 0.75B | bfloat16 | 1948 MB | PASS | 71 权重, 6 full + 18 linear, chunk_gated_delta_rule 融合算子 |

### prefill-decode 模式 (`MODE=prefill-decode`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9663 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (562 权重) |
| qwen3_5 | 0.75B | bfloat16 | 1964 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (71 权重) |

prefill-decode 模式产物含 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json` 两图,权重 `weights_0.bin` + `weight_info.json` 共享一份(`save_epair_prefill_decode` 内部 `save_weights=False` 复用 prefill 的打包结果)。

> 注：上表大小为 `--use-embeddings` 默认 True **之前**的快照；现默认 True，`weights_*.bin` 不含查表权重、另产 `host_weights_*`（gemma4 含 PLE 的 host_weights ~5.5GB），单模型总产物拆分不同但 PASS 状态不变。

## 只导出前 N 层 (`--num-layers` / `NUM_LAYERS`)

截断 `model.model.layers = model.model.layers[:N]`,**不改 `config.num_hidden_layers`**:
- PLE(gemma4)/delta_cache(qwen3.5)/layer_types 仍按全量 config 保持(计算正确,冗余但无害)
- KV buffer 由 `ExportWrapper._num_layers()` 按**实际层数** `len(model.model.layers)` 精确分配(截断后 N 个,无冗余零张量)
- 各模型 forward `for layer in self.layers:` 自动只跑 N 层(保留层 0..N-1,layer_idx 与位置一致)

| 模型 | N=5 single | N=5 prefill-decode | 说明 |
|------|-----------|--------------------|------|
| gemma4 | PASS (6434MB, 102 权重) | — | N≤15 无 shared 层,最干净 |
| qwen3_5 | PASS (1185MB, 71 权重) | PASS (1188MB, 71 权重) | chunk_gated_delta_rule 融合算子,单 GE 节点 |

### qwen3.5 segfault 已解决(遗留任务 3)

qwen3.5 的 linear attention 层(GatedDeltaNet)原本用纯 torch `torch_chunk_gated_delta_rule` 的 Python for 循环,`torch.export` 展开后产生约 19418 个图节点,`export_to_epair` 的 GE 转换阶段图爆炸 segfault。

**已解决**:新增 `cann_lm::chunk_gated_delta_rule` 融合算子(单 GE 节点,不展开循环):
- `cann_lm/models/fused_ops/chunk_gated_delta_rule.py`:schema + fake meta + CPU impl(复用 torch_chunk 做 BHSD↔BSHD 转换)
- `ge_converter.py`:手动定义 `ge.ChunkGatedDeltaRule`(torchair auto_generated 未生成此算子)
- `modeling_qwen3_5.py`:调 `torch.ops.cann_lm.chunk_gated_delta_rule`(prefill + decode 共用)

精度验证:vs HF max_diff=0.0(greedy identical)+ wrapper vs original max_diff=0.0(KV cache 全 match)。

> 注:遗留任务 3(qwen3.5 export_to_epair segfault)已解决，两模型现在都能 export_to_epair。但 qwen3.5 的 **C++ OM 编译仍搁置**（ChunkGatedDeltaRule infershape 失败，见下文"C++ OM 编译"节）。

## 产物布局

`--use-embeddings` 默认 **True**（纯 embeddings，embed_tokens 查表外置到 host）：

```
epair_<model>/
├── ep_json_with_ascend_ir.json   # torch.export 序列化 + ascend_ir 标注 (GE 转换后)
├── weight_info.json              # 权重索引 {weights_0.bin: [{name, offset, size}, ...]}
├── weights_0.bin                 # 权重数据 (bfloat16 → view(int16) 打包; 不含 embed_tokens 查表权重)
├── model_config.json             # {model_type, config: {...}, hf_config: {...}, use_embeddings}
├── tokenizer.json                # 从 HF 复制
├── tokenizer_config.json         # 从 HF 复制
├── host_weights_info.json        # (use_embeddings=True) host 查表权重索引 {bin: [{name,offset,size,dtype,shape}]}
└── host_weights_0.bin            # (use_embeddings=True) embed_tokens / gemma4 embed_tokens_per_layer
```

- `--no-use-embeddings`（图内查表）时不产生 `host_weights_*`，`weights_*.bin` 含 `embed_tokens`。
- prefill-decode 模式额外生成 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json`（权重仍共享一份）。

## 验证项

`verify.py` 检查:

1. `ep_json_with_ascend_ir.json` 存在且 JSON 可解析
2. `model_config.json` 存在,`config` 子键含 `model_type/hidden_size/num_hidden_layers/vocab_size`
3. `weight_info.json` 存在,bin 文件名与实际 `weights_*.bin` 一致,权重条目数 > 0
4. 至少一个 `weights_*.bin`
5. 至少一个 tokenizer 文件
6. 目录总大小 > 0

## C++ OM 编译 (`cann_lm_compile`)

用 `cann_lm/converter/export/csrc/tools/compile_main.cc` 加载 epair → 构建 GE Graph → 编译 OM(`aclgrphBuildModel`)。本机 NPU Ascend910_9362,soc_version 用 `SOC_VERSION` 环境变量(默认 `Ascend910_9362`)。

### 编译 + 运行

```bash
source $PROJECT_ROOT/scripts/set_env_cann.sh        # ② OM 编译基底 (真 CANN; 含 SOC_VERSION + 关 AUB + 权值分离)
bash scripts/build.sh cpp                            # 产 build/cann_lm_compile

./build/cann_lm_compile \
  "$(pwd)/examples/test_converter/export_llm_model/cann_lm_cli_export/epair_gemma4/ep_json_with_ascend_ir.json" \
  "$(pwd)/examples/test_converter/export_llm_model/cann_lm_cli_export/epair_gemma4/weight_info.json" \
  gemma4_n5
```

3 stage:① Load epair(EpairModelLoader)→ ② GetGEGraph → ③ Compile(aclgrphBuildModel → `.om`)。

### 测试结果(NUM_LAYERS=5 截断)

| 模型 | stage 1 Load | stage 2 GetGEGraph | stage 3 BuildModel | OM 大小 | 说明 |
|------|-------------|-------------------|--------------------|---------|------|
| gemma4 | PASS | PASS | **PASS** | 6.3 GB | 补 AddRmsNorm + RotaryMul 算子注册 |
| qwen3_5 | PASS | PASS | **FAIL** | — | ChunkGatedDeltaRule infershape 失败(inputDesc 空) |

### qwen3.5 OM 编译进展与现状(已解决 2026-08-16)

> **2026-08-16 更新**：qwen3.5 全层 prefill-decode epair（`--use-embeddings` + softplain whitelist 分解后）OM 编译**全 3 stage PASS** + NPU 出字连贯（`Hello, my name is` → ` John. I am a 20-year-old male. I am a student at a university.`）。实际阻塞点是 softplus 分解引入的 3 个未注册 GE 算子 `Log1p`/`Greater`/`SelectV2`（见下"已解决(softplus 分解算子未注册)"），注册后全层 epair 不再触发下方记的 ChunkGatedDeltaRule infershape 问题。下方 Conv2D/ChunkGatedDeltaRule 分析为 NUM_LAYERS=5 截断 epair 的历史记录，保留备查。

qwen3.5 的 OM 编译历经多轮修复,Conv2D infershape **已解决**;ChunkGatedDeltaRule infershape 在 NUM_LAYERS=5 截断 epair 上曾失败(历史记录)。

**已解决(softplus 分解算子未注册)**:
- `save_epair_prefill_decode` 补 whitelist 后,`aten.softplus` 分解为 `log1p(exp)` + `Greater`/`SelectV2`(threshold 分支),图里出现 `Log1p`/`Greater`/`SelectV2` GE 节点
- 这 3 个算子未在 `reg_register.h` 注册 → stage2 `GetGEGraph failed`(GE: `Op type "Log1p" is not registered in OperatorFactory`)
- 修复:按 CANN 官方 IR 在 `reg_register.h` 补 `Greater`(仿 GreaterEqual)/`Log1p`(CANN IR)/`SelectV2`(CANN IR 原文) → 全层 epair 编译全过 + 出字 PASS

**已解决(Conv2D infershape)**:
1. **方案 1(C++ inputDesc fallback)**:`ge_graph_builder.cc:1366` PlanConverterNode 对 input 调 `FindTensorValue(op.inputs[i])` 当 fallback_meta,边界输入(FX 名)命中 `tensor_values` 拿到 shape → Unsqueeze input 有 shape
2. **方案 B(Python 填 outputDesc)**:`ge_converter.py` 的 `_convert_conv1d` 手动填 Unsqueeze/Conv2D 的 `output_desc.shape.dim` + `dtype`(从 `meta_outputs.size` 反推 input L_in = (L_out-1)*s + K - 2p),让下游 Conv2D inputDesc(CopyFrom)非空
3. **format NCHW 修复**:C++ `aten_tensor_desc_mapper.cc` 的 `MapGeLayoutString` 只认 "ND",补认 "NCHW"/"NHWC"(Conv2D 要求 input format NCHW,原 desc.layout="ND" 被拒)

**仍失败(ChunkGatedDeltaRule infershape)**:
- ChunkGatedDeltaRule 的 input(query/key/value/beta)来自 `transpose` 的 output,**无 shape.dim**(空)
- transpose 是 torchair 内置中间 op,outputDesc 只设 name(空),下游 ChunkGatedDeltaRule inputDesc 空
- 方案 1 的 C++ fallback 对内部 op→op 边(transpose_12:0)无效(FindTensorValue 找不到内部 GE ref)
- GE infershape 不推空 desc → ChunkGatedDeltaRule InferShapeV2Func 失败

**根本问题**:torchair 中间 op(transpose/unsqueeze)的 outputDesc 不填 shape/dtype(只 name),C++ 不从上游传播(PlanConverterNode outputDesc 传 nullptr),GE infershape 不覆盖已存在的空 TensorDesc → 中间 op→op 边 shape 断链。

**后续解决方向**(搁置,未实施):
1. **ep_json 后处理**:在 `export_to_epair` 后遍历图,对 outputDesc 无 shape 的 op,从 `_meta` base64 或 `tensor_values` 解析 + 沿 Data 边传播(producer output shape → consumer input shape),一次性填全所有中间 op
2. **C++ 全传播**:`PlanConverterNode` 创建下游 op 时,从上游 producer 的 outputDesc 取 shape 填下游 inputDesc(需解析 `OpName:idx` ref + 回溯 producer)
3. **改 torchair**:让 auto_generated 的中间 op 填 outputDesc shape/dtype(上游修复)

### main.cpp / reg_register.h / ge_converter.py / C++ backend 改动

- **`main.cpp`**:soc_version 改用 `std::getenv("SOC_VERSION")`(默认 `Ascend910_9362`,本机芯片;`Ascend910A` 导致 TensorMove/TransposeD 找不到 engine)。`main.cpp` + `CMakeLists.txt` 已上移到 `examples/test_converter/export_llm_model/`（doc 对齐）。
- **`reg_register.h`** 已 **IR 校准**：以 `ops_IR(Ascend 950)/ops/*.md` 的 `REG_OP` 为标准，73 个有 IR 的算子全对齐官方 IR（dtype/inputs/outputs/attrs），8 个无 IR 的自研算子（`QuantMatmulS8S4`/`RotaryMul`/`ChunkGatedDeltaRule`/`ConcatD`/`SoftmaxV3`/`SliceD`/`TransposeD`/`TensorMove`）保留原定义。master 在 `examples/test_converter/export_llm_model/reg_register.h`（84 ops，2026-08-16 补 `Greater`/`Log1p`/`SelectV2`），`om_npu_verify/reg_register.h` 与 `export_epair_and_compile/reg_register.h` 为其软链（单一源）。本机 `DT_BFLOAT16`/`DT_FLOAT32` 经别名可编译，按 IR 原文保留。
  - 历史补算子（现已含在校准版）：gemma4 `AddRmsNorm`+`RotaryMul`；qwen3.5 `ChunkGatedDeltaRule`+`Shape`/`Expand`/`Gather`/`Minimum`/`Pack`/`ScatterElements`/`SplitV`+`Conv2D`/`Squeeze`；`Unsqueeze` 用 `TensorType::ALL()`；2026-08-16 补 qwen3.5 softplus 分解路径的 `Greater`/`Log1p`/`SelectV2`（CANN 官方 IR）。
- **`ge_converter.py`** 补 GE 映射:
  - `cann_lm::chunk_gated_delta_rule` → 手动定义 `ge.ChunkGatedDeltaRule`(torchair auto_generated 未生成此算子),填 output desc shape/dtype(从 meta_outputs)
  - `aten.conv1d` → `ge.Unsqueeze + _npu_convolution + ge.Squeeze`(1D→2D),手动填 Unsqueeze/Conv2D 的 output_desc.shape.dim + dtype(从 meta_outputs.size 反推 L_in)+ layout="NCHW"(避 inputDesc 空传播 + format ND 被拒)
  - `aten.type_as` 的 `meta_outputs` 移到 kwargs(避免 `Converter.__call__` 参数数错)
  - `aten.index_copy` → `ge.InplaceUpdate + ge.Transpose`（KV cache scatter-write；曾注释掉，已取消注释以打通 gemma4 prefill-decode 导出，备份 `ge_converter.py.bak`）
- **C++ backend(`cann_lm/converter/export/csrc/`)**:
  - `ge_graph_builder.cc:1366`:PlanConverterNode 对 input 调 `FindTensorValue(op.inputs[i])` 当 fallback_meta(方案 1,边界输入命中 tensor_values 拿 shape,内部 GE ref 返回 nullptr 无害)
  - `aten_tensor_desc_mapper.cc:91`:`MapGeLayoutString` 补认 "NCHW"/"NHWC"(原只认 "ND",Conv2D 要求 input format NCHW)
  - `saver.py`:`save_epair`/`save_epair_prefill_decode` 加 `whitelist_decompositions`(默认 EpairConfig whitelist,linear/matmul 等分解)

---

## 测试方案（覆盖全部 CLI 场景）

`cann_lm_converter.py` 当前全参数：`--hf-path`/`--output-dir`(必填) + `--dtype`(float32/float16/bfloat16) + `--device`(cpu) + `--max-file-size` + `--kv-cache-max-len` + `--seq-len` + `--use-embeddings`(`BooleanOptionalAction`，自动派生 `--no-use-embeddings`) + `--mode`(single/prefill-decode) + `--num-layers` + 量化互斥组 `--quant-path`/`--quant-method gptq` + `--quant-bits`(w4a8)。

下面两个方案均覆盖**全部参数取值**，按需选用（`--hf-path`/`--output-dir` 省略，两模型路径见 `run.sh`）。

### 方案一：全量覆盖矩阵（隔离，~28 case）

每参数取值至少 1 case，尽量单变量隔离。

**A. 基础（model × mode × use-emb × dtype，pairwise）**

| ID | model | 参数 | 覆盖 |
|---|---|---|---|
| A1 | gemma4 | `--mode single --dtype bfloat16` | single+bf16 baseline |
| A2 | gemma4 | `--mode prefill-decode --dtype bfloat16` | pd+bf16 |
| A3 | qwen3.5 | `--mode single --dtype bfloat16` | 混合架构 single |
| A4 | qwen3.5 | `--mode prefill-decode --dtype bfloat16` | 混合架构 pd |
| A5 | gemma4 | `--mode single --dtype float16` | dtype float16 |
| A6 | qwen3.5 | `--mode single --dtype float32` | dtype float32 |
| A7 | qwen3.5 | `--mode single --dtype bfloat16 --no-use-embeddings` | use-emb False |
| A8 | gemma4 | `--mode single --dtype bfloat16 --no-use-embeddings` | use-emb False（gemma4 PLE 图内） |

**B. 截断/快速（`--num-layers`）**

| ID | model | 参数 | 覆盖 |
|---|---|---|---|
| B1 | gemma4 | `--num-layers 5 --mode single` | 截断（N≤15 无 shared 层） |
| B2 | qwen3.5 | `--num-layers 5 --mode single` | 截断 + ChunkGatedDeltaRule |

**C. 参数边界**

| ID | model | 参数 | 覆盖 | 期望 |
|---|---|---|---|---|
| C1 | qwen3.5 | `--seq-len 256 --mode single` | 长序列 | 节点数随 seq 增 |
| C2 | qwen3.5 | `--max-file-size 1048576 --mode single` | 小 bin 切分 | `weight_info.json` 含多个 `weights_*.bin` |
| C3 | qwen3.5 | `--kv-cache-max-len 512 --mode single` | 小 KV buffer | KV 输入 shape[2]=512 |
| C4 | qwen3.5 | `--kv-cache-max-len 4096 --mode single` | 大 KV buffer | KV shape[2]=4096 |
| C5 | qwen3.5 | `--seq-len 8 --mode single` | 短序列快测 | 最小图 |

**D. 量化**

| ID | model | 参数 | 覆盖 | 前置 |
|---|---|---|---|---|
| D1 | gemma4 | `--quant-path <dopt产物>` | 场景1 消费 dopt（W4A8+自动 embed） | **需 dopt GPTQ 产物** |
| D2 | gemma4 | `--quant-method gptq` | 场景2 spawn dopt | **需 NPU，待实现** |
| D4 | — | `--quant-path X --quant-method gptq` | 互斥组报错 | argparse 拒绝（期望失败） |

**E. 错误/边界（期望失败，fail-fast）**

| ID | model | 参数 | 期望 |
|---|---|---|---|
| E1 | — | `--dtype int8` | argparse choices 拒绝 |
| E2 | — | `--mode fast` | argparse choices 拒绝 |
| E3 | — | （缺 `--hf-path`） | required 拒绝 |

**覆盖核对**：dtype(3)✓ mode(2)✓ use-emb(2)✓ num-layers(full/5)✓ seq-len(8/256)✓ max-file(默认/1M)✓ kv-len(2048/512/4096)✓ quant-path/method/bits/互斥✓ device(cpu)✓

### 方案二：精简版（baseline + 7 新增，最小覆盖）

复用 `run.sh` 作 baseline（2 模型 × {single, pd} × bf16 × seq8 × use-emb=True + `--num-layers 5`，6 case）；用最小模型 **qwen3.5** 当工作马（最快）；堆叠冒烟型边界参数；4 错误折 1 套件。

**baseline（`run.sh` 已覆盖）**：model × mode × bf16 × use-emb-True × num-layers(full + 5)

**新增（7 case）**：

| ID | model | 参数 | 补缺口 |
|---|---|---|---|
| N1 | qwen3.5 | `--mode single --dtype float16` | dtype float16 |
| N2 | qwen3.5 | `--mode single --dtype float32` | dtype float32 |
| N3 | gemma4 | `--mode single --no-use-embeddings` | use-emb False（PLE 图内查表，最难路径） |
| N4 | qwen3.5 | `--mode single --kv-cache-max-len 512 --seq-len 256 --max-file-size 1048576` | **堆叠**：kv-len + seq-len + max-file 三边界 |
| N5 | gemma4 | `--quant-path <dopt产物>` | 场景1 量化（需 dopt） |
| N7 | — | error suite：`--num-layers 999` / `--num-layers 0` / `--quant-path X --quant-method gptq` / `--dtype int8` | 4 个 fail-fast（含互斥） |

**可跑**：N1-N5 + N7 = 6（N6 待 NPU/实现）。

**覆盖核对**：dtype(3)✓ mode(2, baseline)✓ use-emb(2)✓ num-layers(full/5/999/0)✓ seq-len(8/256)✓ max-file(默认/1M)✓ kv-len(2048/512)✓ quant-path/method/bits/互斥✓ device(cpu)✓

**取舍（牺牲了什么）**：
- **N4 堆叠**三边界：覆盖到了，但失败需二分定位（这三个是"参数透传"型，出问题概率低，可接受）。
- **N3 只测 gemma4 `--no-use-embeddings`**：gemma4 PLE 图内查表是最复杂路径；能过则 qwen3.5 图内大概率也过（未显式测）。
- **N6 暂不可跑**：`run_auto_quant` 待实现。

> 再砍可去 N4（边界堆叠，最低价值），但不再覆盖 kv-len/seq-len/max-file 非默认值。
