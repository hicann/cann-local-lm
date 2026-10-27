# CHANGELOG

简体中文 | [English](CHANGELOG_en.md)

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## 2026.10

- converter 当前能力：HuggingFace 模型 → epair 导出（torch.export + FX→GE 转换 + 权重分离），支持 W4A8 量化
- 引擎当前能力：三层推理引擎架构，支持 Continuous Batching、KV Cache 管理、异步流式推理、多轮续接、请求取消
- omc 输出按实际 dtype 读取修复（FP16/BF16 golden 对比误判）
- CLI 参数精简：dtype 由 `--target-soc` 派生、`--use-embeddings` 固定 True
- 量化算子 dump + 仿真推理 + hook 一致性验证
- converter 文档清理（删 8 篇废弃文档、合并 converter_design 进 README）
