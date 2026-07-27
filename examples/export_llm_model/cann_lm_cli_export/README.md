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
| `SEQ_LEN` | `8` | single 模式 `--sample-seq-len`;prefill-decode 模式 `--prefill-seq-len` |
| `NUM_LAYERS` | (空=全部) | 只导出前 N 层(`--num-layers`),截断 `model.model.layers`,不改 `config.num_hidden_layers` |
| `VENV_DIR` | `/mnt/workspace/torch-delegate-harmony/.venv` | converter venv (torch 2.9.0+cpu + torchair + torch_delegate) |

## 测试结果

### single 模式 (`MODE=single`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9645 MB | PASS | 562 权重, sliding+full attention, KV 共享 |
| qwen3 | 0.60B | bfloat16 | 1463 MB | PASS | 312 权重, 标准 GQA |
| qwen3_5 | 0.75B | bfloat16 | — | FAIL | segfault (遗留任务 3) |

### prefill-decode 模式 (`MODE=prefill-decode`)

| 模型 | 参数量 | dtype | 产物大小 | 结果 | 说明 |
|------|--------|--------|----------|------|------|
| gemma4 | 5.03B | bfloat16 | 9663 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (562 权重) |
| qwen3 | 0.60B | bfloat16 | 1482 MB | PASS | prefill+decode 两图 + 共享权重 1 份 (312 权重) |
| qwen3_5 | 0.75B | bfloat16 | — | FAIL | segfault (同 single, 遗留任务 3) |

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
| qwen3_5 | — | — | FAIL segfault(任何含 linear 层的 N,torch_chunk_gated_delta_rule 展开节点爆炸,截断无法避开) |

> qwen3.5 的 linear attention 层在 `layer_types` 的 `[0,1,2,4,5,6,8,...]` 位置(每 4 层 1 个 full),无法连续截取到纯 full,故截断无法避开 segfault。需靠 NPU fused 算子解决(遗留任务 3)。

### qwen3.5 segfault 原因

qwen3.5 的 linear attention 层(GatedDeltaNet)用 `torch_chunk_gated_delta_rule` 的 Python for 循环更新 recurrent state,`torch.export` 展开后产生约 19418 个图节点,`export_to_epair` 的 GE 转换阶段图爆炸 segfault。

**解决方向**(参考 `cann-recipes-infer`):用 NPU fused 算子 `npu_chunk_gated_delta_rule`(prefill)+ `npu_recurrent_gated_delta_rule`(decode)替代 Python 循环,或为 GE 自定义 `chunk_gated_delta_rule` 算子。详见 `docs/cann_lm_converter/converter_design.md` "各模型的全量难点"。

> 注:qwen3.5 的 modeling / wrapper / 单元测试(68 passed)/ HF 精度对比(max_diff=0.0)/ chunk 推理 / 多轮对话均 PASS,**仅 export_to_epair(GE 转换)失败**。

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
