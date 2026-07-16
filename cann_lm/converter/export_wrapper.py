"""Export wrapper base class.

参考 litert-torch 的 ExportableModuleBase 设计：
- 基类统一对外 I/O 契约 (input_ids/embeddings, attention_mask, position_ids, past_key_values, ...)
- 子类在各模型目录下实现 (e.g. cann_lm/models/qwen3/export_wrapper.py)
- 子类实现 _drive_layers: 将统一 I/O 翻译为原生模型 forward kwargs
- 支持 token_ids (图内查表) 和 embeddings (图外查表) 两种模式
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """统一对外 I/O：cache in/out + output_pos 选位。子类实现 _drive_layers。

    轻量版：子类内部调原生 forward（带 if past is None 分支）
    全量版（后续）：各 attention 加 scatter-write 路径，一张静态图覆盖 prefill+decode

    两种 embedding 模式：
    - use_embeddings=False: 传 input_ids, 图内含 embed_tokens (默认)
    - use_embeddings=True:  传 embeddings, 图内不含 embed_tokens (embedding 量化用)
    """

    def __init__(self, model: nn.Module, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: Any = None,
        new_kv_cache_pos: torch.Tensor = None,
        embeddings: torch.Tensor | None = None,
        output_pos: torch.Tensor | None = None,
    ):
        logits, past_kv_flat = self._drive_layers(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            embeddings,
        )
        if output_pos is not None:
            logits = logits[:, output_pos, :]
        return logits, *past_kv_flat

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """子类实现：统一 I/O → 原生 forward kwargs → 提取 logits + 展平 KV cache"""
        raise NotImplementedError

    def get_sample_inputs(self, seq_len: int = 128, use_embeddings: bool = False) -> tuple:
        """构造导出用示例输入

        轻量版：past_key_values=None (模型内部创建 cache)，mask 为 [1,1,seq_len,seq_len] causal
        后续全量版将使用预分配 KV buffer + scatter-write

        Args:
            seq_len: 示例序列长度
            use_embeddings: True 时传 embeddings (图外查表), False 时传 input_ids (图内查表)
        """
        config = self.config
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        dtype = next(self.model.parameters()).dtype

        input_ids = torch.randint(0, vocab_size, (self.batch, seq_len), dtype=torch.int64)

        # 4D causal mask [batch, 1, seq_len, seq_len] — 走 4D 透传路径
        min_val = torch.finfo(dtype).min
        causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
        attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, seq_len)

        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(self.batch, 1)
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int64)
        output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

        # 轻量版：past_key_values=None (模型内部创建 cache)
        past_key_values = None

        if use_embeddings:
            embeddings = self.model.model.embed_tokens(input_ids).detach()
            return (
                None,
                attention_mask,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                embeddings,
                output_pos,
            )
        else:
            return (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                None,
                output_pos,
            )
