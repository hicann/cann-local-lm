"""Qwen3 export wrapper — translates unified I/O to native forward kwargs."""

from __future__ import annotations

import torch

from cann_lm.models.export_wrapper import ExportWrapper


class Qwen3ExportWrapper(ExportWrapper):
    """qwen3 子类：调原生 forward + 提取 logits + 展平 list[(k,v)] cache"""

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        per_layer_token_identity: torch.Tensor | None = None,
        delta_states: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # None → [None]*N 触发模型内部 cache 创建 (N = 实际层数, 支持 --num-layers 截断)
        num_layers = self._num_layers()
        if past_key_values is None:
            past_key_values = [None] * num_layers

        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "new_kv_cache_pos": new_kv_cache_pos,
        }
        # qwen3 无 PLE: embeddings 提供时走纯 embeddings (inputs_embeds), 不传 input_ids
        # → model forward `if inputs_embeds is None` 跳过 embed_tokens 查表, input_ids 不入图。
        # - use_embeddings=False (embeddings=None): 传 input_ids, model 查 embed_tokens
        # - use_embeddings=True  (embeddings=真实): 传 inputs_embeds, 不传 input_ids
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        updated_kv = outputs["past_key_values"]

        past_kv_flat: list[torch.Tensor] = []
        if updated_kv is not None:
            for layer_kv in updated_kv:
                if layer_kv is not None:
                    past_kv_flat.append(layer_kv[0])
                    past_kv_flat.append(layer_kv[1])
        return logits, past_kv_flat, []
