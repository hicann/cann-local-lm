"""cann_local_lm.engine — epair 运行时 / 推理引擎 (占位, 待实现)。

独立 distribution ``cann-local-lm-engine`` (与 ``cann-local-lm`` converter+models 分开打包)。
``cann_local_lm`` 为 PEP 420 namespace 包: cann-local-lm 装 ``cann_local_lm.converter``/``cann_local_lm.models``,
本包装装 ``cann_local_lm.engine`` (api/backend/core/infra/pipeline)。

子目录骨架 (待填代码):
- api: 对外推理 API (prefill/decode 入口)
- backend: 后端适配 (acl/GE/NPU runtime)
- core: 核心数据结构 (KV cache / delta states / sample)
- core/sample: 采样策略
- infra: 基础设施 (内存/流/事件)
- pipeline: 多模型/多 stage 编排
"""
