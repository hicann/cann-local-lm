"""Gemma4 export wrapper — translates unified I/O to native forward kwargs."""

from __future__ import annotations

import torch

from cann_lm.converter.export_wrapper import ExportWrapper


class Gemma4ExportWrapper(ExportWrapper):
    """gemma4 子类：调原生 forward + 提取 logits + 展平 list[(k,v)] cache

    gemma4 的 forward 不接受 position_ids（内部计算），
    额外有 shared_kv_states 参数（共享 KV，轻量版传 None）。
    """

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "shared_kv_states": None,
            "output_hidden_states": False,
        }
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
                past_kv_flat.append(layer_kv[0])
                past_kv_flat.append(layer_kv[1])
        return logits, past_kv_flat
