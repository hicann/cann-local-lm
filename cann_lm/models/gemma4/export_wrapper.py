"""Gemma4 export wrapper — translates unified I/O to native forward kwargs.

gemma4 的特殊之处：
- forward 不接受 position_ids（内部从 past_key_values 推导）
- 有 shared_kv_states 参数（KV 共享层无 k_proj/v_proj）
- 有 layer_types (sliding_attention + full_attention)
- KV cache 是 list[(k,v) or None]
- 35 层中后 20 层为 KV 共享层 (num_kv_shared_layers=20)

轻量版处理策略：
- past_key_values=None → [None]*N 触发模型内部创建
- 展平后重建为 list[(k,v)] 以支持 decode
- shared_kv_states 传空 dict (避免 shared 层崩溃)
"""

from __future__ import annotations

import torch

from cann_lm.converter.export_wrapper import ExportWrapper


class Gemma4ExportWrapper(ExportWrapper):
    """gemma4 子类：调原生 forward + 展平 KV cache + 处理 shared_kv_states"""

    def _get_layer_head_dim(self, layer_idx: int) -> int:
        """gemma4: full_attention 层用 global_head_dim，sliding 层用 head_dim。

        gemma4 有 per-layer head_dim 差异（sliding 层的 KV 维度 = head_dim，
        full 层的 KV 维度 = global_head_dim）。qwen3/qwen3.5 无此差异（用 base 默认）。
        """
        global_head_dim = getattr(self.config, "global_head_dim", None)
        layer_types = getattr(self.config, "layer_types", None)
        if global_head_dim and layer_types:
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            return global_head_dim if not is_sliding else self.config.head_dim
        return self.config.head_dim

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # None → [None]*N 触发模型内部创建 cache
        num_layers = self.config.num_hidden_layers
        if past_key_values is None:
            kv_list = [None] * num_layers
        elif isinstance(past_key_values, list) and len(past_key_values) > 0 and isinstance(past_key_values[0], (tuple, list)):
            # 已经是 list[(k,v), ...] 格式 → clone 防止 in-place 修改 (ONNX tracing 友好)
            kv_list = [(k, v) for k, v in past_key_values]
        else:
            # 展平的 [k0, v0, k1, v1, ...] → 重建为 [(k0,v0), (k1,v1), ...]
            kv_list = [(past_key_values[i], past_key_values[i + 1]) for i in range(0, len(past_key_values), 2)]

        # shared_kv_states: 传空 dict 避免 shared 层崩溃
        # gemma4 有 num_kv_shared_layers 个层没有 k_proj/v_proj,
        # 它们复用前面的层 KV, 需要 shared_kv_states 字典
        # 轻量版传空 dict (第一个 full attention 层会产生 shared states)
        shared_kv_states = {} if num_layers > 0 else None

        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": kv_list,
            "shared_kv_states": shared_kv_states,
            "output_hidden_states": False,
            "new_kv_cache_pos": new_kv_cache_pos,
            "position_ids": position_ids,
        }
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        updated_kv = outputs["past_key_values"]

        # 展平 list[(k,v)] → [k0, v0, k1, v1, ...]
        past_kv_flat: list[torch.Tensor] = []
        if updated_kv is not None:
            for layer_kv in updated_kv:
                if layer_kv is not None:
                    past_kv_flat.append(layer_kv[0])
                    past_kv_flat.append(layer_kv[1])
        return logits, past_kv_flat
