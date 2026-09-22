# gemma4 从模型构建到 OM 导出 — 整体流程

> 端到端:模型代码 → 注册 → 加载 → 导出 epair → C++ 编译 OM
>
> 以 gemma4 为例(含 PLE、纯 embeddings 模式),其他模型(qwen3_5)流程一致,差异见 [converter_design.md](./converter_design.md) "模型差异总览"。
>
> 最后更新: 2026-08-07

## 目录

- [总览](#总览)
- [Phase 1: 模型构建](#phase-1-模型构建-cann_lmgemma4)
- [Phase 2: 注册](#phase-2-注册-model_registrypy)
- [Phase 3: CLI 入口](#phase-3-cli-入口-cann-lm-converterpy)
- [Phase 4: 加载](#phase-4-加载-loaderpy)
- [Phase 5: 导出](#phase-5-导出-saverpy)
- [Phase 6: C++ OM 编译](#phase-6-c-om-编译-examplesexport_llm_modelmaincpp)
- [端到端命令](#端到端命令)
- [产物布局](#产物布局)
- [关键设计点](#关键设计点)

---

## 总览

```
cann_lm/models/gemma4/  (configuration / modeling / export_wrapper)
        │
        ▼  注册
cann_lm/models/model_registry.py  (REGISTRY["gemma4"] = ModelEntry(...))
        │
        ▼  CLI
cann_lm/converter/cann_lm_converter.py  convert()
        │
        ├── loader.load_model()          config.json → model + weights + tokenizer
        │
        ▼
cann_lm/converter/saver.py  save_epair / save_epair_prefill_decode
        │
        ├── wrapper.get_sample_inputs()           算 sample (embeddings + per_layer_token_identity)
        ├── _extract_host_side_weights()          抽查表权重 (detach 前)
        ├── wrapper.detach_externalized_lookups() del 查表子模块 → torch.export 不 lift
        ├── torch.export.export()                 → ExportedProgram
        ├── _assert_externalized_not_lifted()     兜底
        ├── torch_delegate.export_to_epair()      → EpairContents (ep_json + state_dict)
        ├── del model + gc                        OOM 防护
        └── consolidated_weights_save()           weights_*.bin + host_weights_*.bin + ep_json
                │
                ▼
examples/test_converter/export_llm_model/main.cpp
        │
        ├── EpairModelLoader(ep_json, weight_info.json) → GetGEGraph
        ├── aclgrphBuildModel(graph)
        └── aclgrphSaveModel → <name>.om
```

---

## Phase 1: 模型构建 `cann_lm/models/gemma4/`

| 文件 | 职责 |
|------|------|
| `configuration_gemma4.py` | `Gemma4TextConfig` dataclass。PLE 字段 `vocab_size_per_layer_input`/`hidden_size_per_layer_input`;`layer_types`(sliding/full 交替);`global_head_dim`(full 层 KV 维度,与 sliding 的 `head_dim` 不同);`num_kv_shared_layers`(后 n 层 KV 共享);`__post_init__` 算 `layer_types`/`rope_parameters` 等派生量 |
| `modeling_gemma4.py` | `Gemma4TextScaledWordEmbedding`(embed_tokens + PLE 的 embed_tokens_per_layer,`embed_scale=sqrt(embedding_dim)`,`persistent=False` buffer);`Gemma4TextModel.forward`(PLE 流程:`_compute_per_layer_inputs` 查表 → `_project_per_layer_inputs` 投影);`Gemma4ForCausalLM`(lm_head + softcap);`load_gemma4_weights`(HF safetensors → key 重映射 `model.language_model.*`→`model.*`) |
| `export_wrapper.py` | `Gemma4ExportWrapper`,详见下"导出 wrapper" |
| `__init__.py` | 导出 config/model/load_weights/wrapper |

### 导出 wrapper `Gemma4ExportWrapper` 关键方法

- `_drive_layers(...)`:纯 embeddings 分支用**实例方法替换**把 `text_model._compute_per_layer_inputs` 临时替换为返回 host 的 `per_layer_token_identity`,跳过 `embed_tokens_per_layer(input_ids)` 查表 + bf16 反查;model.forward 层循环/mask/rope/lm_head 全复用(不改 modeling)
- `_build_external_extras(input_ids)`:host 算 `per_layer_token_identity = embed_tokens_per_layer(input_ids).reshape(b, s, num_hidden_layers, hidden_size_per_layer_input)`(在 get_sample_inputs 里、detach 前调用)
- `detach_externalized_lookups()`:`del text_model.embed_tokens` + `del text_model.embed_tokens_per_layer`(导出前,使 torch.export 不 lift 查表权重)
- `host_side_weight_names()`:返回 `["model.model.embed_tokens.weight", "model.model.embed_tokens_per_layer.weight"]`(供 host 抽取打包)
- `_get_layer_head_dim(i)`:sliding 层用 `head_dim`,full 层用 `global_head_dim`(per-layer KV 维度)

> qwen3_5 无 PLE:wrapper 的 `_drive_layers` 在 embeddings≠None 时只传 `inputs_embeds`、不传 `input_ids`;`_build_external_extras` 返回空;`detach_externalized_lookups` 只 del `embed_tokens`。详见各模型 `export_wrapper.py`。

---

## Phase 2: 注册 `cann_lm/models/model_registry.py`

```python
REGISTRY["gemma4"] = ModelEntry(
    model_type="gemma4_text",
    config_cls=Gemma4TextConfig,
    model_cls=Gemma4ForCausalLM,
    load_weights_fn=load_gemma4_weights,
    config_json_key="text_config",      # config.json 里嵌在 text_config 下
    wrapper_cls=Gemma4ExportWrapper,
)
```

`detect_model_type(hf_config)` 从 `config.json` 的 `model_type` 字段自动探测,无需 `--model-type`。

---

## Phase 3: CLI 入口 `cann_lm/converter/cann_lm_converter.py`

```python
def convert(hf_path, output_dir, dtype, use_embeddings=False,
            mode="single", num_layers=None, ...):
    artifact = load_model(hf_path, dtype=dtype)
    if num_layers is not None:           # 截断 model.model.layers[:N]
        ...
    if mode == "prefill-decode":
        save_epair_prefill_decode(artifact, output_dir, use_embeddings=use_embeddings, ...)
    else:
        save_epair(artifact, output_dir, use_embeddings=use_embeddings, ...)
```

CLI 参数:`--hf-path --output-dir --dtype --use-embeddings --num-layers --mode {single,prefill-decode} --seq-len --seq-len --kv-cache-max-len --quant-path ...`。

---

## Phase 4: 加载 `cann_lm/converter/loader.py:load_model`

```
1. 读 config.json
2. detect_model_type(hf_config)        → ModelEntry
3. build_config(hf_config, entry)      按 config_json_key 取子字典 + dataclass 字段过滤 + __post_init__
4. entry.model_cls(config).to(dtype).eval()
5. entry.load_weights_fn(model, hf_path, ...)   safetensors → key 重映射 → param.copy_()
6. collect_tokenizer_files(hf_path)    tokenizer.json/config + special_tokens_map (+ 旧格式)
→ ModelArtifact(model, config, hf_path, hf_config, tokenizer_files, model_type, wrapper_cls)
```

---

## Phase 5: 导出 `cann_lm/converter/saver.py`

### `save_epair`(single)/`save_epair_prefill_decode`(两图共享权重)

关键顺序(**顺序很重要**):

```python
1. register_ge_converters()                      # cann_lm.* + aten.* → GE 算子映射 (export_to_epair 前必须)
2. wrapper = artifact.wrapper_cls(model, kv_cache_max_len, batch=1).eval()
3. sample = wrapper.get_sample_inputs(seq_len, use_embeddings)
                                                  # 用 embed_tokens/embed_tokens_per_layer 算 sample
                                                  #   embeddings = embed_tokens(input_ids).detach()
                                                  #   per_layer_token_identity = embed_tokens_per_layer(input_ids).reshape(...)  (gemma4)
4. host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)
                                                  # 在 detach 前从 state_dict 抽查表权重 (供 host)
5. if use_embeddings: wrapper.detach_externalized_lookups()
                                                  # del embed_tokens (+ gemma4 embed_tokens_per_layer)
                                                  # → torch.export 不 lift → weights_*.bin 不含冗余查表权重 (≈ OM)
6. ep = torch.export.export(wrapper, sample, strict=False)
7. if use_embeddings: _assert_externalized_not_lifted(ep, wrapper.host_side_weight_names())
                                                  # 兜底: 查表权重不应出现在 inputs_to_parameters
8. epair = torch_delegate.export_to_epair(ep, config)   # torchair IR 转换 + decomposition
9. del ep, wrapper, artifact.model; gc.collect()        # 大模型 OOM 防护 (epair.ep 持独立引用)
10. consolidated_weights_save(epair, output_dir, host_side_weights=host_side_weights, ...)
11. copy_tokenizer_files(...) + save_model_config(artifact, output_dir, use_embeddings=use_embeddings)
```

prefill-decode 模式:步骤 3 算 prefill_sample + decode_sample → 步骤 5 detach 一次 → 步骤 6-8 分别 export prefill/decode 两图(共享权重,pack_weights 只跑一次)。

### `consolidated_weights_save` `cann_lm/converter/export/epair_utils.py`

产物:
- `weights_*.bin` + `weight_info.json` — 图消费的参数(detach 后不含查表权重,≈ OM;按 `max_file_size` 切分,bf16→int16 view)
- `host_weights_*.bin` + `host_weights_info.json` — 查表权重(embed_tokens + gemma4 embed_tokens_per_layer,条目含 `name/offset/size/dtype/shape`)给 host(仅 `use_embeddings=True`)
- `ep_json_with_ascend_ir.json`(single)或 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json`(prefill-decode)

---

## Phase 6: C++ OM 编译 `examples/test_converter/export_llm_model/main.cpp`

```cpp
td::EpairModelLoader loader(model_json_path, weights_index_path);   // 构造
ge::Graph graph("delegate_graph");
loader.GetGEGraph(&graph);                                            // 构建 GE Graph
// ↓ 内部: GeGraphBuilder
//   EpairModelParser::Parse(ep_json) → ParsedModel
//   GeWeightLoader::Create(weight_info.json, model)  懒加载 weights_*.bin
//   PrepareContext → PreparePlaceholders (Data/Const 节点) → PlanNodes (Ascend 算子) → FinalizeGraph

aclgrphBuildInitialize({{"ge.socVersion", soc_version}});            // 编译
aclgrphBuildModel(graph, {{"input_format","ND"}}, model);
aclgrphSaveModel(output_name.c_str(), model);                        // → <name>.om
aclgrphBuildFinalize();
```

- `reg_register.h` 注册 GE 编译期算子(Add/Mul/MatMul/SoftmaxV2/RmsNorm 等);缺算子报 `UNMAPPED_OPERATOR` 退出
- `PlanNodes` 顶部跳过 `_assert_tensor_metadata` 节点(torch.export 插入的无输出校验)
- 权重 bin 路径从 `weight_info.json` 的 key 推导(相对索引文件目录)

C++ 加载两图共享一份权重:
```cpp
td::EpairModelLoader prefill_loader("prefill_ep_json_with_ascend_ir.json", "weight_info.json");
td::EpairModelLoader decode_loader ("decode_ep_json_with_ascend_ir.json",  "weight_info.json");
```

---

## 端到端命令

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

VENV=/mnt/workspace/torch-delegate-harmony/.venv/bin/python
BIN=/mnt/workspace/project/CANN-LM_5883/examples/test_converter/export_llm_model/build/cann_lm_compile
HF=/mnt/workspace/data/models/gemma-4-E2B-it
OUT=/tmp/opencode/gemma4_out

# ── single 模式 (5 层, use_embeddings) ──
$VENV cann_lm/converter/cann_lm_converter.py \
  --hf-path $HF --output-dir $OUT --dtype bfloat16 \
  --seq-len 8 --use-embeddings --num-layers 5
$BIN $OUT/ep_json_with_ascend_ir.json $OUT/weight_info.json gemma4_out

# ── prefill-decode 模式 (5 层, use_embeddings) ──
$VENV cann_lm/converter/cann_lm_converter.py \
  --hf-path $HF --output-dir $OUT --dtype bfloat16 \
  --use-embeddings --num-layers 5 --mode prefill-decode --seq-len 8
$BIN $OUT/prefill_ep_json_with_ascend_ir.json $OUT/weight_info.json prefill
$BIN $OUT/decode_ep_json_with_ascend_ir.json  $OUT/weight_info.json decode
```

> 环境变量含义见 [AGENTS.md](../../../AGENTS.md) "运行环境设置(converter,已验证)":torchair 源码编译(NO_ASCEND_SDK=1)+ stubs(CPU 上跑 IR 转换),`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 阻止 torch autoload npu backend。

---

## 产物布局

```
OUT/
├── ep_json_with_ascend_ir.json         # single 模式
│   (或 prefill_ep_json_with_ascend_ir.json + decode_ep_json_with_ascend_ir.json)
├── weight_info.json + weights_*.bin     # 图消费的参数 (≈ OM, detach 后不含查表权重)
├── host_weights_info.json              # (仅 use_embeddings=True) host 查表权重索引
├── host_weights_0.bin                  # (仅 use_embeddings=True) embed_tokens + gemma4 embed_tokens_per_layer
├── model_config.json                   # config + hf_config + use_embeddings 标记
└── tokenizer.json / tokenizer_config.json / special_tokens_map.json

+ <name>.om / prefill.om / decode.om    # C++ 编译产物 (与 OUT 同目录或指定)
```

---

## 关键设计点

### 纯 embeddings 模式(`--use-embeddings`)

图输入是 `embeddings`(host 算好的 `embed_tokens(input_ids) * embed_scale`)+ gemma4 的 `per_layer_token_identity`(host 算好的 `embed_tokens_per_layer(input_ids)`),`input_ids` 不入图。host 推理时查表喂这两个张量。

- **gemma4 PLE 外置**:实例方法替换 `_compute_per_layer_inputs` 返回 host 的 `per_layer_token_identity`(不改 modeling);`embed_tokens_per_layer` 查表外置
- **detach 消冗余**:torch.export 会 lift 全部注册 Parameter(含未用查表权重)→ `weights_*.bin` 携带 OM 编译时本就剪掉的冗余。`detach_externalized_lookups()` 在导出前 del 查表子模块 → 不 lift → `weights_*.bin` ≈ OM
- **tied 模型**:del `embed_tokens` 后 `lm_head.weight`(同数据)仍在 → 被 OM bake;host 从 `host_weights` 拿 `embed_tokens.weight` 查表(lm_head/embed_tokens 双角色,非冗余)
- **embed_scale**:`sqrt(hidden_size)` / `hidden_size_per_layer_input**0.5`,config 可推导;`persistent=False` + 纯 embeddings 下不进图;host 从 `model_config.json` 复现

### 几何/数值验证(已通)

- `ep.module()` vs eager:max_diff=0(PLE 实例方法替换经 tracing 存活)
- 纯 embeddings ep == in-graph wrapper == HF(3-prompt)
- weights_0.bin ≈ OM(detach 后对上,无 5.5GB 冗余)
- C++ OM 编译全 PASS(无 `MISSING_PLACEHOLDER_RECORD`)

详见 [converter_design.md](./converter_design.md) "全链路已通" + [export_design.md](./export/export_design.md)。
