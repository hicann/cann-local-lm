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

from cann_lm.models.export_wrapper import ExportWrapper
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

    def _linear_layer_indices(self) -> list[int]:
        """linear_attention 层索引 (截断到实际层数)."""
        layer_types = getattr(self.config, "layer_types", [])
        n = len(self.model.model.layers)
        return [i for i, lt in enumerate(layer_types) if lt == "linear_attention" and i < n]

    def _build_zero_delta_states(self):
        """构造零 delta_states (conv_flat, recurrent_flat) 供 prefill 图输入.

        prefill 首次无历史 state, conv_state=零 等价 nn.Conv1d(padding=K-1) 的 zero padding
        (depthwise conv: conv_state 零替代 padding 零, output[K-1+k]==eager output[k]).
        让 prefill 也走 export 路径 (delta 图输入输出), 统一 prefill+decode delta tensor 流.
        """
        lin = next(l.linear_attn for l in self.model.model.layers
                   if l.layer_type == "linear_attention")
        dtype = next(self.model.parameters()).dtype
        n_linear = len(self._linear_layer_indices())
        conv_flat = [torch.zeros(self.batch, lin.conv_dim, lin.conv_kernel_size, dtype=dtype)
                     for _ in range(n_linear)]
        rec_flat = [torch.zeros(self.batch, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim, dtype=dtype)
                    for _ in range(n_linear)]
        return (conv_flat, rec_flat)

    def get_sample_inputs(self, seq_len: int = 128, use_embeddings: bool = False,
                          quant_embeddings: bool = False) -> tuple:
        """qwen3_5 override: prefill 也传 delta_states (零, 图输入), 统一 prefill+decode delta 流.

        past KV 只构造 full_attention 层 (6 对), linear 层用 delta_states 不用 KV —
        与 decode 一致, 避免基类 24 对全层 KV 的冗余零 buffer.
        """
        base = super().get_sample_inputs(seq_len, use_embeddings, quant_embeddings)
        base_list = list(base)
        # past KV: 基类构造 num_layers 对(全层), 减到 full_attn 层(与 decode 一致)
        full_idx = self._get_full_attn_layer_indices()
        base_list[3] = [base_list[3][i] for i in full_idx]
        delta = self._build_zero_delta_states()
        while len(base_list) < 9:
            base_list.append(None)
        base_list.append(delta)
        return tuple(base_list)

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
        cache = SimpleKVCache()
        if past_key_values is not None:
            # prefill + decode 统一: past KV 只 full_attn 层(6 对), 按 idx 配 full_attn_indices
            full_attn_indices = self._get_full_attn_layer_indices()
            for idx, (k, v) in enumerate(past_key_values):
                cache.update(k, v, full_attn_indices[idx])

        # delta_states: forward 参数 (export tuple) 或 _delta_cache_box (eager/prefill, SimpleDeltaNetCache)
        ds = delta_states if delta_states is not None else self._delta_cache_box[0]
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": cache,
            "delta_states": ds,
            "labels": None,
            "use_cache": True,
            "new_kv_cache_pos": new_kv_cache_pos,
        }
        # qwen3_5 无 PLE: embeddings 提供时走纯 embeddings (inputs_embeds), 不传 input_ids
        # → model forward `if inputs_embeds is None` 跳过 embed_tokens 查表, input_ids 不入图。
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        else:
            kwargs["input_ids"] = input_ids

        outputs = self.model(**kwargs)
        logits = outputs["logits"]
        updated_cache = outputs["past_key_values"]
        updated_delta = outputs.get("delta_cache")
        if delta_states is None:
            self._delta_cache_box[0] = updated_delta   # 持久 delta_cache (eager 跨调用)

        past_kv_flat: list[torch.Tensor] = []
        if updated_cache is not None:
            for layer_idx in range(len(updated_cache.keys)):
                if updated_cache.keys[layer_idx] is not None:
                    past_kv_flat.append(updated_cache.keys[layer_idx])
                    past_kv_flat.append(updated_cache.values[layer_idx])

        # export 路径: updated_delta=(new_conv_list, new_rec_list), 展平供 host 下次 decode
        delta_out_flat: list[torch.Tensor] = []
        if delta_states is not None and isinstance(updated_delta, tuple):
            new_conv_list, new_rec_list = updated_delta
            for c, r in zip(new_conv_list, new_rec_list):
                delta_out_flat.append(c)
                delta_out_flat.append(r)
        return logits, past_kv_flat, delta_out_flat
