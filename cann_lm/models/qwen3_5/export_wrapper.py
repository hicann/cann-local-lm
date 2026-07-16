"""Qwen3_5 export wrapper — translates unified I/O to native forward kwargs.

qwen3_5 的特殊之处：
- past_key_values 是 SimpleKVCache 对象 (不是 list)
- 混合架构: full_attention + linear_attention 交替
- 有 delta_cache (线性注意力的 recurrent state)
- forward 接受 use_cache 参数

轻量版处理策略：
- 将外部传入的 list[(k,v)] 转换为 SimpleKVCache 再传给模型
- delta_cache 传 None（后续再支持线性注意力 cache）
- 展平 SimpleKVCache 返回值为 [k_0, v_0, k_1, v_1, ...]
- KV cache 层索引通过 config.layer_types 映射 (full_attention 层不是连续的)
"""

from __future__ import annotations

import torch

from cann_lm.converter.export_wrapper import ExportWrapper
from cann_lm.models.qwen3_5.modeling_qwen3_5 import SimpleKVCache


class Qwen3_5ExportWrapper(ExportWrapper):
    """qwen3_5 子类：list ↔ SimpleKVCache 互转 + 调原生 forward"""

    def _get_full_attn_layer_indices(self) -> list[int]:
        """从 config.layer_types 获取 full_attention 层的索引"""
        layer_types = getattr(self.config, "layer_types", [])
        return [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cache = SimpleKVCache()
        if past_key_values is not None:
            full_attn_indices = self._get_full_attn_layer_indices()
            for idx, (k, v) in enumerate(past_key_values):
                layer_idx = full_attn_indices[idx] if idx < len(full_attn_indices) else idx
                cache.update(k, v, layer_idx)

        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "delta_cache": None,
            "inputs_embeds": None,
            "labels": None,
            "use_cache": True,
        }
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        updated_cache = outputs["past_key_values"]

        past_kv_flat: list[torch.Tensor] = []
        if updated_cache is not None:
            for layer_idx in range(len(updated_cache.keys)):
                if updated_cache.keys[layer_idx] is not None:
                    past_kv_flat.append(updated_cache.keys[layer_idx])
                    past_kv_flat.append(updated_cache.values[layer_idx])
        return logits, past_kv_flat
