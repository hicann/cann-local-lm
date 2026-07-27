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

    def __init__(self, model, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__(model, kv_cache_max_len=kv_cache_max_len, batch=batch)
        # delta_cache 用 list 容器持久 (避 nn.Module tensor 属性赋值检测, torch.export 不报;
        # linear attention 的 recurrent state 必须跨 prefill/decode 调用持久).
        self._delta_cache_box = [None]

    def _get_full_attn_layer_indices(self) -> list[int]:
        """从 config.layer_types 获取 full_attention 层的索引(截断到实际层数).

        截断后 len(self.model.model.layers) < config.num_hidden_layers,
        需过滤 i < n 避免 prefill 分支 past_key_values[layer_idx] 越界.
        """
        layer_types = getattr(self.config, "layer_types", [])
        n = len(self.model.model.layers)
        return [i for i, lt in enumerate(layer_types) if lt == "full_attention" and i < n]

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
            if len(past_key_values) == len(full_attn_indices):
                # decode (reconstruct_kv 输出, 6 对, 只 full_attn, 顺序 full_attn_indices)
                for idx, (k, v) in enumerate(past_key_values):
                    cache.update(k, v, full_attn_indices[idx])
            else:
                # prefill (get_sample_inputs 输出, 24 对, 全层, 按 layer_idx 取 full_attn 层)
                for layer_idx in full_attn_indices:
                    k, v = past_key_values[layer_idx]
                    cache.update(k, v, layer_idx)

        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "delta_cache": self._delta_cache_box[0],
            "inputs_embeds": None,
            "labels": None,
            "use_cache": True,
            "new_kv_cache_pos": new_kv_cache_pos,
        }
        # 总是传 input_ids (PLE 用, 不反查); embeddings 非 None 时也传 inputs_embeds (图外查表)
        # - use_embeddings=False (embeddings=None): 只传 input_ids, model 查 embed_tokens
        # - use_embeddings=True (embeddings=真实): 传 input_ids + inputs_embeds, model 用
        #   inputs_embeds (不查 embed_tokens), PLE 用 input_ids (不反查, 避开 bfloat16 == 精度问题)
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        updated_cache = outputs["past_key_values"]
        self._delta_cache_box[0] = outputs.get("delta_cache")   # 持久 delta_cache (跨调用)

        past_kv_flat: list[torch.Tensor] = []
        if updated_cache is not None:
            for layer_idx in range(len(updated_cache.keys)):
                if updated_cache.keys[layer_idx] is not None:
                    past_kv_flat.append(updated_cache.keys[layer_idx])
                    past_kv_flat.append(updated_cache.values[layer_idx])
        return logits, past_kv_flat
