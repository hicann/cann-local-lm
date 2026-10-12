# ppl_compare 使用说明（浮点 vs GPTQ 量化产物 PPL 对比）

> `cann_lm/converter/quant/ppl_compare.py` — 对比浮点 HF 模型与 dopt GPTQ 量化产物的困惑度（PPL），评估 W4A8 量化损伤。单文件脚本，无 dopt/converter 运行时依赖。
>
> 状态: 已实测（Qwen3-0.6B, 2026-10-10） | 最后核对: 2026-10-10

## 原理

GPTQ 产物由 `--hf-type-save` 落盘，是**普通 HF 权重格式**（W4 伪量化值已写回 `weight`，
可抽查每 group(128) 仅 ≤16 个离散值验证）。因此直接 `AutoModelForCausalLM.from_pretrained`
即可加载，无需 dopt 运行时。产物 state_dict 中的 dopt quantizer 参数键（scale/offset 等）
会被 HF 忽略并刷 unexpected-keys 告警，脚本已降级该日志，属预期行为。

对比公平性：两侧使用**相同 dtype / 相同数据 / 相同滑窗**（seq 2048，窗口不重叠，
位置 i 的 logits 预测 token i+1），PPL 差值即纯量化损伤。

## 使用方法

前置：任一装有 `torch` + `transformers` 的 python（NPU 机器需含 `torch_npu`）。

```bash
cd cann_lm/converter/quant

# CPU（无需 NPU，dtype 自动 float32）
python ppl_compare.py \
    --float-model /path/to/Qwen3-0.6B \
    --quant-model /path/to/dopt_qwen3_0dot6_w4a8

# NPU（dtype 自动 float16）
DEVICE=npu python ppl_compare.py \
    --float-model ... --quant-model ...

# 换 held-out 数据 / 调 token 预算
python ppl_compare.py ... --data /path/to/any_text.jsonl --max-tokens 65536
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--float-model` | （必填） | 浮点 HF 模型目录（同时提供 tokenizer，产物目录不含 tokenizer） |
| `--quant-model` | （必填） | dopt GPTQ 产物目录（`--hf-type-save` 格式） |
| `--data` | `calibrate_sets/calibrate_coig_cqia.jsonl` | 评测 jsonl，每行需含 `text` 字段 |
| `--max-tokens` | `16384` | 参与 PPL 计算的 token 上限 |
| `--seq-len` | `2048` | 滑窗长度 |
| `--stride` | `= seq-len` | 窗口步长（默认不重叠） |
| `--dtype` | `auto` | `auto`: cpu→float32 / 其他→float16；可显式 float32/float16/bfloat16 |
| 环境变量 `DEVICE` | `cpu` | `cpu` / `npu`（npu 需 torch_npu），与 dopt 惯例一致 |

## 输出示例

```
[ppl] device=cpu dtype=float32 seq=2048 stride=2048
[ppl] data=.../calibrate_coig_cqia.jsonl tokens=65536
[ppl] float /path/to/Qwen3-0.6B
      ppl=21.2377 (n_pred=65504)
[ppl] quant /path/to/dopt_qwen3_0dot6_w4a8
      ppl=25.1932 (n_pred=65504)
[ppl] 量化影响: 21.2377 -> 25.1932  (+18.62%)
```

## 实测参考（Qwen3-0.6B, W4A8 GPTQ, CPU fp32）

| 数据集 | tokens | float PPL | quant PPL | 损伤 |
|--------|--------|-----------|-----------|------|
| coig_cqia（校准集, in-set） | 65536 | 21.24 | 25.19 | **+18.62%** |
| wikitext-2 test（held-out 英文） | 65536 | 20.84 | 27.62 | +32.56% |

## 注意事项

1. **默认数据即 GPTQ 校准集**（`config.yaml` 的 `train_files` 同一份文件），量化模型见过该数据，
   结果偏乐观；正式评估建议 `--data` 指定 held-out jsonl。
2. **分布偏移会放大损伤**：校准集为中文指令数据，英文维基（wikitext）测试叠加了语言+领域双重
   偏移，+32.6% 不能完全归因于量化本身。贴近部署场景应选同分布 held-out 数据。
3. 增大 `--max-tokens` 可降方差（coig 在 16k/65k 两种预算下结果稳定：+18.5% / +18.6%）。
4. 产物中不含 tokenizer，脚本统一用 `--float-model` 目录的 tokenizer（两侧本就同源）。
