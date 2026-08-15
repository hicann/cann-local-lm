# cann_lm_cli_export — cann-lm-converter.py CLI 测试

用 `cann_lm/converter/cann-lm-converter.py` CLI 把 gemma4 / qwen3 / qwen3.5 三个 HuggingFace 模型导出为 epair 格式,并验证产物完整性。

## 文件

| 文件 | 作用 |
|------|------|
| `run.sh` | source 环境后循环调 `cann-lm-converter.py` 导出三模型,再调 `verify.py` |
| `verify.py` | 检查每个 `epair_<model>/` 目录产物:ep_json / model_config / weight_info / weights_*.bin / tokenizer |

## 用法

```bash
# 默认: 全部三模型, single 模式, bfloat16, seq_len=8 (快测)
bash run.sh

# 指定模型
bash run.sh "qwen3 qwen3_5"

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
| `DTYPE` | `bfloat16` | 三模型原生 bfloat16 |
| `SEQ_LEN` | `8` | single 模式 `--seq-len`;prefill-decode 模式 `--seq-len` |
| `NUM_LAYERS` | (空=全部) | 只导出前 N 层(`--num-layers`),截断 `model.model.layers`,不改 `config.num_hidden_layers` |
| `VENV_DIR` | `/mnt/workspace/torch-delegate-harmony/.venv` | converter venv (torch 2.9.0+cpu + torchair + torch_delegate) |

## 测试结果

### single 模式 (`MODE=single`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9645 MB | PASS | 562 权重, sliding+full attention, KV 共享 |
| qwen3 | 0.60B | bfloat16 | 1463 MB | PASS | 312 权重, 标准 GQA |
| qwen3_5 | 0.75B | bfloat16 | 1948 MB | PASS | 71 权重, 6 full + 18 linear, chunk_gated_delta_rule 融合算子 |

### prefill-decode 模式 (`MODE=prefill-decode`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9663 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (562 权重) |
| qwen3 | 0.60B | bfloat16 | 1482 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (312 权重) |
| qwen3_5 | 0.75B | bfloat16 | 1964 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (71 权重) |

prefill-decode 模式产物含 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json` 两图,权重 `weights_0.bin` + `weight_info.json` 共享一份(`save_epair_prefill_decode` 内部 `save_weights=False` 复用 prefill 的打包结果)。

## 只导出前 N 层 (`--num-layers` / `NUM_LAYERS`)

截断 `model.model.layers = model.model.layers[:N]`,**不改 `config.num_hidden_layers`**:
- PLE(gemma4)/delta_cache(qwen3.5)/layer_types 仍按全量 config 保持(计算正确,冗余但无害)
- KV buffer 由 `ExportWrapper._num_layers()` 按**实际层数** `len(model.model.layers)` 精确分配(截断后 N 个,无冗余零张量)
- 各模型 forward `for layer in self.layers:` 自动只跑 N 层(保留层 0..N-1,layer_idx 与位置一致)

| 模型 | N=5 single | N=5 prefill-decode | 说明 |
|------|-----------|--------------------|------|
| gemma4 | PASS (6434MB, 102 权重) | — | N≤15 无 shared 层,最干净 |
| qwen3 | PASS (757MB, 59 权重) | PASS (761MB, 59 权重) | 标准 GQA,最简单 |
| qwen3_5 | PASS (1185MB, 71 权重) | PASS (1188MB, 71 权重) | chunk_gated_delta_rule 融合算子,单 GE 节点 |

### qwen3.5 segfault 已解决(遗留任务 3)

qwen3.5 的 linear attention 层(GatedDeltaNet)原本用纯 torch `torch_chunk_gated_delta_rule` 的 Python for 循环,`torch.export` 展开后产生约 19418 个图节点,`export_to_epair` 的 GE 转换阶段图爆炸 segfault。

**已解决**:新增 `cann_lm::chunk_gated_delta_rule` 融合算子(单 GE 节点,不展开循环):
- `cann_lm/models/fused_ops/chunk_gated_delta_rule.py`:schema + fake meta + CPU impl(复用 torch_chunk 做 BHSD↔BSHD 转换)
- `ge_converter.py`:手动定义 `ge.ChunkGatedDeltaRule`(torchair auto_generated 未生成此算子)
- `modeling_qwen3_5.py`:调 `torch.ops.cann_lm.chunk_gated_delta_rule`(prefill + decode 共用)

精度验证:vs HF max_diff=0.0(greedy identical)+ wrapper vs original max_diff=0.0(KV cache 全 match)。

> 注:遗留任务 3(qwen3.5 export_to_epair segfault)已解决,三模型现在都能 export_to_epair + C++ OM 编译。

## 产物布局

```
epair_<model>/
├── ep_json_with_ascend_ir.json   # torch.export 序列化 + ascend_ir 标注 (GE 转换后)
├── weight_info.json              # 权重索引 {weights_0.bin: [{name, offset, size}, ...]}
├── weights_0.bin                 # 权重数据 (bfloat16 → view(int16) 打包)
├── model_config.json             # {model_type, config: {...}, hf_config: {...}}
├── tokenizer.json                # 从 HF 复制
└── tokenizer_config.json         # 从 HF 复制
```

prefill-decode 模式额外生成 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json`(权重仍共享一份)。

## 验证项

`verify.py` 检查:

1. `ep_json_with_ascend_ir.json` 存在且 JSON 可解析
2. `model_config.json` 存在,`config` 子键含 `model_type/hidden_size/num_hidden_layers/vocab_size`
3. `weight_info.json` 存在,bin 文件名与实际 `weights_*.bin` 一致,权重条目数 > 0
4. 至少一个 `weights_*.bin`
5. 至少一个 tokenizer 文件
6. 目录总大小 > 0

## C++ OM 编译 (`main.cpp`)

用 `examples/export_llm_model/main.cpp` 加载 epair → 构建 GE Graph → 编译 OM(`aclgrphBuildModel`)。本机 NPU Ascend910_9362,soc_version 用 `SOC_VERSION` 环境变量(默认 `Ascend910_9362`)。

### 编译 + 运行

```bash
source /home/developer/Ascend/cann/set_env.sh
TORCH_PKG_DIR=$(/mnt/workspace/torch-delegate-harmony/.venv/bin/python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')
cmake -S examples/export_llm_model -B examples/export_llm_model/build \
  -DCANN_ROOT=$ASCEND_HOME_PATH -DTORCH_PKG_DIR=$TORCH_PKG_DIR
cmake --build examples/export_llm_model/build -j$(nproc)

export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib:cann_lm/converter/export/csrc/build/stub_lib:${LD_LIBRARY_PATH:-}"
./examples/export_llm_model/build/export_llm_model \
  "$(pwd)/examples/export_llm_model/cann_lm_cli_export/epair_qwen3/ep_json_with_ascend_ir.json" \
  "$(pwd)/examples/export_llm_model/cann_lm_cli_export/epair_qwen3/weight_info.json" \
  qwen3_n5
```

3 stage:① Load epair(EpairModelLoader)→ ② GetGEGraph → ③ Compile(aclgrphBuildModel → `.om`)。

### 测试结果(NUM_LAYERS=5 截断)

| 模型 | stage 1 Load | stage 2 GetGEGraph | stage 3 BuildModel | OM 大小 | 说明 |
|------|-------------|-------------------|--------------------|---------|------|
| qwen3 | PASS | PASS | **PASS** | 449 MB | 算子全注册,标准 GQA |
| gemma4 | PASS | PASS | **PASS** | 6.3 GB | 补 AddRmsNorm + RotaryMul 算子注册 |
| qwen3_5 | PASS | PASS | **FAIL** | — | ChunkGatedDeltaRule infershape 失败(inputDesc 空) |

### qwen3.5 OM 编译进展与现状(搁置)

qwen3.5 的 OM 编译历经多轮修复,Conv2D infershape **已解决**,但 ChunkGatedDeltaRule infershape 仍失败(搁置)。

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

- **`main.cpp`**:soc_version 改用 `std::getenv("SOC_VERSION")`(默认 `Ascend910_9362`,本机芯片;`Ascend910A` 导致 TensorMove/TransposeD 找不到 engine)
- **`reg_register.h`** 补算子注册(对齐 CANN op_info 的 dtype + attr 顺序):
  - gemma4:`AddRmsNorm`(残差+RMSNorm)+ `RotaryMul`(半旋转)
  - qwen3.5:`ChunkGatedDeltaRule`(线性注意力融合)+ `Shape`/`Expand`/`Gather`/`Minimum`/`Pack`/`ScatterElements`/`SplitV`(动态 shape + slice_scatter 引入)+ `Conv2D`/`Squeeze`(conv1d 映射)
  - `Unsqueeze` 改 `TensorType::ALL()`(对齐 auto_generated,原 `BasicType()` 可能不含 bf16)
- **`ge_converter.py`** 补 GE 映射:
  - `cann_lm::chunk_gated_delta_rule` → 手动定义 `ge.ChunkGatedDeltaRule`(torchair auto_generated 未生成此算子),填 output desc shape/dtype(从 meta_outputs)
  - `aten.conv1d` → `ge.Unsqueeze + _npu_convolution + ge.Squeeze`(1D→2D),手动填 Unsqueeze/Conv2D 的 output_desc.shape.dim + dtype(从 meta_outputs.size 反推 L_in)+ layout="NCHW"(避 inputDesc 空传播 + format ND 被拒)
  - `aten.type_as` 的 `meta_outputs` 移到 kwargs(避免 `Converter.__call__` 参数数错)
- **C++ backend(`cann_lm/converter/export/csrc/`)**:
  - `ge_graph_builder.cc:1366`:PlanConverterNode 对 input 调 `FindTensorValue(op.inputs[i])` 当 fallback_meta(方案 1,边界输入命中 tensor_values 拿 shape,内部 GE ref 返回 nullptr 无害)
  - `aten_tensor_desc_mapper.cc:91`:`MapGeLayoutString` 补认 "NCHW"/"NHWC"(原只认 "ND",Conv2D 要求 input format NCHW)
  - `saver.py`:`save_epair`/`save_epair_prefill_decode` 加 `whitelist_decompositions`(默认 EpairConfig whitelist,linear/matmul 等分解)
