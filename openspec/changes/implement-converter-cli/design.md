## Context

CANN-LM 已有三个模型定义（`cann_lm/models/gemma4`, `qwen3`, `qwen3_5`）和一个 epair 导出工具链（`cann_lm/converter/export/epair_utils.py` + C++ `EpairModelLoader`）。但缺少统一的 CLI 工具将 HF 模型自动导出为 epair 格式。已有 `model_registry.py`、`loader.py`、`saver.py`、`cann-lm-converter.py` 的初版代码和详细设计文档 `converter_design.md`，但代码尚未完全对齐设计（saver 缺少 ExportWrapper/register_ge_converters/assert 过滤，epair_utils 缺少 bfloat16/非持久 buffer 兼容）。

## Goals / Non-Goals

**Goals:**
- 代码对齐 `converter_design.md` 设计文档
- `saver.py` 增加 ExportWrapper、register_ge_converters 调用
- `epair_utils.py` 增加 bfloat16 `view(int16)` 兼容和非持久 buffer 补充
- `cann-lm-converter.py` 确认 auto-detect 和 convert() 程序化入口
- 端到端验证：qwen3-0.6B 导出 + C++ 加载

**Non-Goals:**
- 不修改 `cann_lm/models/` 下的模型定义代码
- 不实现统一 cache 签名（轻量版 ExportWrapper，后续目标）
- 不实现 prefill/decode 分离
- 不实现量化和分区

## Decisions

1. **ExportWrapper 只返回 logits** — 不绕过 `Qwen3Model.forward`，仅解决 dict 返回问题。当前 `attention_mask=None` 跳过 mask 构造，功能不完整但可用于验证 pipeline。正确的 mask 处理（预构造 4D mask）留待后续。

2. **bfloat16 用 `view(int16)`** — 不做 dtype 转换，保留原始字节。C++ 侧按原始 dtype 读取。

3. **非持久 buffer 从 `ep.module().named_buffers()` 补充** — 不用 `ep.constants`（可能引入无关常量和命名不一致）。

4. **register_ge_converters 在 saver 内部调用** — 不是 CLI 的责任，saver 控制导出全流程。

5. **model_config.json 的 to_dict/asdict fallback** — `hasattr(config, "to_dict")` 优先，否则 `dataclasses.asdict`。

## Risks / Trade-offs

- **attention_mask=None 导致无 causal mask** — 导出的模型功能不完整，但可用于验证 pipeline。后续需改为预构造 4D mask。
- **torch.export + run_decompositions 重新插入 assert 节点** — 需要 ep_json 层过滤。这是 torch.export 的内部行为，无法从源头消除。
- **不同模型 weight key 重映射差异** — 在各模型的 `load_*_weights` 内部处理，converter 层不感知。新模型接入需确认 key 是否匹配。
