## Why

CANN-LM 已有三个模型定义（gemma4/qwen3/qwen3_5）和一个 epair 导出工具链（`epair_utils.py` + C++ `EpairModelLoader`），但缺少一个统一的 CLI 工具将 HF 模型自动导出为 epair 格式。当前导出流程需要手动适配每个模型的接口差异（config 加载、forward 签名、weight key 重映射等），且已有设计文档（`converter_design.md`）但代码尚未完全对齐设计。需要一个 CLI 工具串联 loader → saver 流程，支持自动探测模型类型，后续可轻松扩展新模型。

## What Changes

- 新增 `cann_lm/converter/model_registry.py` — 模型注册表 + `detect_model_type` 自动探测
- 新增 `cann_lm/converter/loader.py` — 模型加载（config + weights + tokenizer 收集），适配三个模型的 config 加载差异
- 新增 `cann_lm/converter/saver.py` — epair 保存，含 `ExportWrapper`（统一 forward 返回 logits）、`register_ge_converters` 调用、`_remove_assert_nodes` 过滤
- 新增 `cann_lm/converter/cann-lm-converter.py` — CLI 入口 + 程序化 `convert()` 函数，模型类型从 config.json 自动探测
- 修改 `cann_lm/converter/export/epair_utils.py` — `pack_weights` 增加 bfloat16 兼容（`view(int16)`）和非持久 buffer 补充（`ep.module().named_buffers()`）
- 更新 `cann_lm/converter/__init__.py` — 导出公共 API

## Capabilities

### New Capabilities

- `model-registry`: 模型注册表 — 按 model_type 注册 config_cls/model_cls/load_weights_fn/config_json_key，支持别名查找和自动探测
- `model-loader`: 模型加载器 — 从 HF 路径加载 config + 权重 + tokenizer，适配 from_pretrained/from_json_file/手动字段过滤三种加载方式
- `epair-saver`: epair 保存器 — ExportWrapper 统一返回 logits + register_ge_converters + torch.export + export_to_epair + consolidated_weights_save + tokenizer/config 复制 + .pt2 保存
- `converter-cli`: CLI 入口 — argparse + convert() 程序化入口 + 验证产物完整性

### Modified Capabilities

（无 — 全新功能，不修改已有模型代码）

## Impact

- **新增文件**: `model_registry.py`, `loader.py`, `saver.py`, `cann-lm-converter.py`
- **修改文件**: `epair_utils.py`（bfloat16 + 非持久 buffer）, `__init__.py`
- **依赖**: `torch_delegate`（export_to_epair）, `cann_lm.models`（模型定义 + load_*_weights）, `cann_lm.models.fake_ops`（register_ge_converters）
- **不修改**: `cann_lm/models/` 下的模型定义代码
- **环境变量**: `ASCEND_HOME_PATH`, `PYTHONPATH`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
