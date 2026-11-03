# CHANGELOG

简体中文 | [English](CHANGELOG_en.md)

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## 2026.11

- embedding 权重与 scale 合并为 `embedding_weights` 单文件（四段布局：主表 → per_layer 表 → 主 scale → per_layer scale；无文件头、无段间 padding），引擎读端改为单文件加载；表/scale 同文件原子替换，消除"只更新其一"的静默错配，部署产物少一个文件
- `embedding_dequant_scale` 配置键废弃：残留非空值仅产生 WARN，不影响解析；存量双文件产物需按新四段布局重导出后方可加载
- identity（per_layer 段）启用时全文件总长须精确等于四段和（加载期校验，防行宽 W 与文件/图失配）

## 2026.10

- converter 当前能力：HuggingFace 模型 → epair 导出（torch.export + FX→GE 转换 + 权重分离），支持 W4A8 量化
- 引擎当前能力：三层推理引擎架构，支持 Continuous Batching、KV Cache 管理、异步流式推理、多轮续接、请求取消
- omc 输出按实际 dtype 读取修复（FP16/BF16 golden 对比误判）
- CLI 参数精简：dtype 由 `--target-soc` 派生、`--use-embeddings` 固定 True
- 量化算子 dump + 仿真推理 + hook 一致性验证
- converter 文档清理（删 8 篇废弃文档、合并 converter_design 进 README）
