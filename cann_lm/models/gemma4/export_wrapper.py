"""Gemma4 export wrapper — translates unified I/O to native forward kwargs.

gemma4 的特殊之处：
- forward 接受 position_ids（外部传入；为 None 时内部从 past_key_values 推导）
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

    def __init__(self, model, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__(model, kv_cache_max_len=kv_cache_max_len, batch=batch)

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
        per_layer_token_identity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # None → [None]*N 触发模型内部创建 cache (N = 实际层数, 支持 --num-layers 截断)
        num_layers = self._num_layers()
        if past_key_values is None:
            kv_list = [None] * num_layers
        elif isinstance(past_key_values, list) and len(past_key_values) > 0 and isinstance(past_key_values[0], (tuple, list)):
            # 已经是 list[(k,v), ...] 格式 → clone 防止 in-place 修改 (ONNX tracing 友好)
            kv_list = [(k, v) for k, v in past_key_values]
        else:
            # 展平的 [k0, v0, k1, v1, ...] → 重建为 [(k0,v0), (k1,v1), ...]
            kv_list = [(past_key_values[i], past_key_values[i + 1]) for i in range(0, len(past_key_values), 2)]

        # shared_kv_states: 传 None, model forward 内部新建 {} (store 层填, 共享层读, 单次 forward 内流转).
        kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": kv_list,
            "shared_kv_states": None,
            "output_hidden_states": False,
            "new_kv_cache_pos": new_kv_cache_pos,
            "position_ids": position_ids,
        }
        if embeddings is not None and per_layer_token_identity is not None:
            # 纯 embeddings 模式: embed_tokens + embed_tokens_per_layer (PLE) 查表都外置到 host。
            # 用实例方法替换让 model.forward 的 _compute_per_layer_inputs 直接返回 host 算好的
            # per_layer_token_identity, 跳过 embed_tokens_per_layer(input_ids) 查表 + bf16 反查;
            # model 仍走完整 forward (层循环/mask/rope/lm_head 全复用), 不改 modeling。
            text_model = self.model.model
            orig = text_model._compute_per_layer_inputs
            text_model._compute_per_layer_inputs = lambda input_ids_, inputs_embeds_: per_layer_token_identity
            try:
                kwargs["inputs_embeds"] = embeddings
                outputs = self.model(**kwargs)
            finally:
                text_model._compute_per_layer_inputs = orig
        else:
            # 非纯 embeddings (图内查表 embeddings=None, 或旧式双传): 保留原行为。
            if input_ids is not None:
                kwargs["input_ids"] = input_ids
            if embeddings is not None:
                kwargs["inputs_embeds"] = embeddings
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

    def _build_external_extras(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """gemma4 PLE: host 算好 per_layer_token_identity (embed_tokens_per_layer 查表结果),
        外置 PLE 对 input_ids 的依赖。shape [batch, seq, num_hidden_layers, hidden_size_per_layer_input]。
        num_hidden_layers 取 config 全量值 (per_layer_model_projection 按 full 维度建), 与
        _compute_per_layer_inputs 一致; --num-layers 截断时多余切片在层循环里被忽略。
        """
        m = self.model.model
        token_identity = m.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, m.config.num_hidden_layers, m.hidden_size_per_layer_input
        ).detach()
        return (token_identity,)

    def host_side_weight_names(self) -> list[str]:
        """gemma4 纯 embeddings: embed_tokens + embed_tokens_per_layer (PLE) 查表都外置到 host。
        FQN 用 wrapper 层路径 (与 ep.graph_signature 一致, 见基类文档)。
        """
        return ["model.model.embed_tokens.weight", "model.model.embed_tokens_per_layer.weight"]

    def detach_externalized_lookups(self) -> None:
        """gemma4: 额外 detach embed_tokens_per_layer (PLE 查表, 被实例方法替换跳过)。"""
        text_model = self.model.model
        if hasattr(text_model, "embed_tokens"):
            del text_model.embed_tokens
        if hasattr(text_model, "embed_tokens_per_layer"):
            del text_model.embed_tokens_per_layer
