## 1. model_registry.py — 模型注册表

- [x] 1.1 `ModelEntry` dataclass（model_type, config_cls, model_cls, load_weights_fn, config_json_key）
- [x] 1.2 注册 gemma4 / qwen3 / qwen3_5 到 `REGISTRY`
- [x] 1.3 `get_entry(name)` 别名查找（gemma-4→gemma4, qwen3.5→qwen3_5）
- [x] 1.4 `detect_model_type(hf_config)` 从 config.json 自动探测
- [x] 1.5 `available_models()` 返回已注册列表

## 2. loader.py — 模型加载

- [x] 2.1 `ModelArtifact` dataclass（model, config, hf_path, hf_config, tokenizer_files, model_type）
- [x] 2.2 `load_model(hf_path, *, dtype, device)` — 自动探测 + 加载全流程
- [x] 2.3 `build_config(hf_config, entry)` — 3 级 fallback（from_pretrained → from_json_file → 手动过滤）
- [x] 2.4 `collect_tokenizer_files(hf_path)` — 收集 tokenizer 相关文件

## 3. saver.py — epair 保存

- [ ] 3.1 `ExportWrapper` 类 — 统一 forward 返回 logits（dict → tensor）
- [ ] 3.2 `register_ge_converters()` 调用 — 在 `export_to_epair` 前
- [ ] 3.3 `torch.export.export(wrapper, sample_inputs, strict=False)` 导出
- [ ] 3.4 `export_to_epair(ep)` + ep_json 过滤断言节点
- [ ] 3.5 `consolidated_weights_save` 保存权重 + ep_json
- [ ] 3.6 `copy_tokenizer_files` + `save_model_config`（to_dict/asdict fallback）
- [ ] 3.7 `epair_save` 保存 .pt2 + 保存 input_0.bin

## 4. epair_utils.py — 兼容性修复

- [ ] 4.1 bfloat16 `view(int16)` 兼容（pack_weights 内部）
- [ ] 4.2 非持久 buffer 从 `ep.module().named_buffers()` 补充
- [ ] 4.3 `import torch` 补充（epair_utils.py 顶部）

## 5. cann-lm-converter.py — CLI

- [x] 5.1 `parse_args()` — --hf-path, --output-dir, --dtype, --device, --max-file-size, --sample-seq-len
- [x] 5.2 `convert()` 程序化入口 — load_model → save_epair → verify
- [x] 5.3 `main()` — argparse 调 convert()
- [ ] 5.4 验证 qwen3-0.6B 端到端导出

## 6. 验证

- [ ] 6.1 Python 导出 qwen3-0.6B 成功
- [ ] 6.2 C++ 加载 epair + 构建 GE Graph 成功
- [ ] 6.3 C++ 编译 OM 成功
- [ ] 6.4 更新 `converter_design.md` 对齐实际代码
