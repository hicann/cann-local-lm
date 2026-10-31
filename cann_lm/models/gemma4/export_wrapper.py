"""Gemma4 export wrapper — bypasses model.forward, drives layers directly.

sliding_attention 的 mask 由 host/pipeline 预构造 (mask_sliding), 连同 mask_full 作为
图输入传入; 图内不再构造 mask (无 masked_fill — NPU 不支持 ge.MaskedFill)。wrapper
绕过 Gemma4TextModel.forward, 直接驱动 decoder layers, 复用 model 子模块
(embed / PLE project / rotary / norm / lm_head)。eager generate 仍走 model.forward
(含 _prepare_causal_mask 的 None/2D 分支), 不受影响。

I/O 契约 (相对基类): 单个 attention_mask 拆成 mask_full + mask_sliding 两个图输入
(host 每 pipeline 步重建)。PLE 开启时 per_layer_token_identity 由 host 预算
(embed_tokens_per_layer 查表外置), wrapper 做 project。
"""

from __future__ import annotations

import torch

from cann_lm.models.export_wrapper import ExportWrapper


class Gemma4ExportWrapper(ExportWrapper):
    """gemma4 子类：绕过 forward 直接驱动层 + host 预构 per-layer mask + PLE project。"""

    def __init__(self, model, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__(model, kv_cache_max_len=kv_cache_max_len, batch=batch)
        cfg = self.config
        self._sliding_window = getattr(cfg, "sliding_window", None)
        self._layer_types = list(getattr(cfg, "layer_types", []) or [])
        ple_h = getattr(cfg, "hidden_size_per_layer_input", 0) or 0
        self._ple_enabled = ple_h > 0
        self._embed_scale = float(cfg.hidden_size**0.5)
        self._ple_embed_scale = float(ple_h**0.5) if ple_h > 0 else 1.0

    def _get_layer_head_dim(self, layer_idx: int) -> int:
        """gemma4: full_attention 层用 global_head_dim，sliding 层用 head_dim。

        gemma4 有 per-layer head_dim 差异（sliding 层的 KV 维度 = head_dim，
        full 层的 KV 维度 = global_head_dim）。qwen3.5 无此差异（用 base 默认）。
        """
        global_head_dim = getattr(self.config, "global_head_dim", None)
        layer_types = getattr(self.config, "layer_types", None)
        if global_head_dim and layer_types:
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            return global_head_dim if not is_sliding else self.config.head_dim
        return self.config.head_dim

    # ──────────────────────────────────────────────────────────────
    # forward (override): 收 mask_full + mask_sliding (替代单 attention_mask)
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        mask_full: torch.Tensor | None = None,
        mask_sliding: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        new_kv_cache_pos: torch.Tensor | None = None,
        embeddings: torch.Tensor | None = None,
        output_pos: torch.Tensor | None = None,
        per_layer_token_identity: torch.Tensor | None = None,
        embeddings_scale: torch.Tensor | None = None,
        per_layer_token_identity_scale: torch.Tensor | None = None,
        sliding_kv_pos: torch.Tensor | None = None,
    ):
        # embeddings + embeddings_scale: raw lookup (不含 embed_scale) × per-token scale。
        # [fold 回迁 2026-10-27, 同 3d1a98f 形态] embed_scale 由图内常量乘补偿:
        # 量化对象是 raw_weight (dopt), int8×per_token_scale≈raw, 缺 Gemma4TextScaledWordEmbedding
        # 的 embed_scale; 7cd26b3 曾把它 fold 进 scale 输入取值, 但引擎 host 喂 scale 工位是
        # 纯搬运 (raw 值, 无 fold), plain 形态图在 bbit 上乱码 (详见 同步代码后出现乱码问题.md §3.6)。
        if embeddings is not None and embeddings_scale is not None:
            dt = self._model_dtype
            embeddings = (
                embeddings.to(dt)
                * embeddings_scale.to(dt).unsqueeze(-1)
                * self._embed_scale
            )
        # per_layer_token_identity + scale: 同理 (PLE embed_scale = sqrt(hidden_size_per_layer_input))
        if (
            per_layer_token_identity is not None
            and per_layer_token_identity_scale is not None
        ):
            dt = self._model_dtype
            per_layer_token_identity = (
                per_layer_token_identity.to(dt)
                * per_layer_token_identity_scale.to(dt).unsqueeze(-1).unsqueeze(-1)
                * self._ple_embed_scale
            )
        logits, past_kv_flat = self._drive_layers(
            input_ids,
            mask_full,
            mask_sliding,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            embeddings,
            per_layer_token_identity,
            sliding_kv_pos,
        )
        if output_pos is not None:
            # index_select 替代 advanced indexing (logits[:, output_pos, :] -> ge.IndexByTensor,
            # 未在 support_ops.md 列出); index_select(1, idx) -> ge.GatherV2 (support_ops.md).
            logits = logits.index_select(1, output_pos)
        return logits, *past_kv_flat

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        mask_full: torch.Tensor | None,
        mask_sliding: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        per_layer_token_identity: torch.Tensor | None = None,
        sliding_kv_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """绕过 Gemma4TextModel.forward, 直接驱动 decoder layers。

        mask 由 host 预构 (mask_full = causal-over-buffer, mask_sliding = causal+window),
        按 layer_types 静态分配给每层; 图内无 mask 构造 (无 masked_fill)。
        PLE: per_layer_token_identity 由 host 预算 (use_embeddings), 否则图内查表;
        project 复用 model 子模块 (per_layer_model_projection / per_layer_projection_norm)。
        """
        m = self.model
        tm = m.model  # Gemma4TextModel
        cfg = tm.config

        # 1. embedding
        if embeddings is not None:
            inputs_embeds = embeddings
        else:
            inputs_embeds = tm.embed_tokens(input_ids)

        # 1a. KV cache list 重建 (None→[None]*N / list[(k,v)] clone / flat 展平→重建)
        #     先归一化为 list[(k,v)|None], 供 position_ids/rope/层驱动统一用.
        num_layers = self._num_layers()
        if past_key_values is None:
            kv_list = [None] * num_layers
        elif (
            isinstance(past_key_values, list)
            and len(past_key_values) > 0
            and isinstance(past_key_values[0], (tuple, list))
        ):
            kv_list = [(k, v) for k, v in past_key_values]
        else:
            kv_list = [
                (past_key_values[i], past_key_values[i + 1])
                for i in range(0, len(past_key_values), 2)
            ]

        # 1b. position_ids auto-prepare (镜像 TextModel.forward; None 时按 past_len 算).
        #     export 路径 position_ids 是图输入 (非 None), 不走此分支; 仅 eager 补.
        if position_ids is None:
            position_ids = tm._prepare_position_ids(inputs_embeds, kv_list)

        # 2. PLE project (复刻 _project_per_layer_inputs; token_identity 由 host 给或图内查表)
        per_layer_inputs = None
        if self._ple_enabled:
            if per_layer_token_identity is None:
                # 非纯 embeddings: 图内查表 embed_tokens_per_layer (大 embedding gather)
                per_layer_token_identity = tm.embed_tokens_per_layer(input_ids).reshape(
                    *input_ids.shape,
                    cfg.num_hidden_layers,
                    tm.hidden_size_per_layer_input,
                )
            ctx = (
                tm.per_layer_model_projection(inputs_embeds)
                * tm.per_layer_model_projection_scale
            )
            ctx = ctx.reshape(
                *inputs_embeds.shape[:-1],
                cfg.num_hidden_layers,
                tm.hidden_size_per_layer_input,
            )
            ctx = tm.per_layer_projection_norm(ctx)
            per_layer_inputs = (
                ctx + per_layer_token_identity
            ) * tm.per_layer_input_scale

        # 3. RoPE (per layer_type, 复用 model 子模块)
        rope_embs = tm._build_rope_embeddings_for_layers(inputs_embeds, position_ids)

        # 4. (KV cache 已在 1a 归一化为 kv_list)

        shared_kv_states: dict = {}

        # 5. 层循环: 按 layer_types 静态分配 mask_i, 复用 layer.forward (含 scatter/attn/mlp/PLE-gate)
        hidden_states = inputs_embeds
        for i, layer in enumerate(tm.layers):
            layer_type = (
                self._layer_types[i] if i < len(self._layer_types) else "full_attention"
            )
            mask_i = mask_sliding if layer_type == "sliding_attention" else mask_full
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            hidden_states = layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=rope_embs[layer_type],
                attention_mask=mask_i,
                past_key_values=kv_list,
                new_kv_cache_pos=new_kv_cache_pos,
                sliding_kv_pos=sliding_kv_pos,
            )

        # 6. norm + lm_head
        hidden_states = tm.norm(hidden_states)
        logits = m.lm_head(hidden_states)

        # 7. 展平 list[(k,v)] → [k0, v0, k1, v1, ...]
        past_kv_flat: list[torch.Tensor] = []
        for layer_kv in kv_list:
            if layer_kv is not None:
                past_kv_flat.append(layer_kv[0])
                past_kv_flat.append(layer_kv[1])
        return logits, past_kv_flat

    # ──────────────────────────────────────────────────────────────
    # host 侧 mask 构造 (get_sample_inputs 用; 不进图, 可用任意算子)
    # ──────────────────────────────────────────────────────────────

    def _build_mask_full(
        self,
        seq_len: int,
        kv_max: int,
        dtype: torch.dtype,
        new_kv_cache_pos: torch.Tensor,
    ) -> torch.Tensor:
        """bool 矩阵 [b,1,seq_len,kv_max], 绝对位 causal。False=attend, True=mask out。

        query@绝对位 new_kv_cache_pos[i] attend kv[0:abs+1]; 未写位 [actual:kv_max] 被
        mask(True) 屏蔽。矩形 q_dim=seq_len 去掉方阵 padding 行 (arch31 tiling 要求
        mask 第三维=S1=seq_len, 不接受 kv_max)。
        """
        mask = torch.ones(self.batch, 1, seq_len, kv_max, dtype=torch.bool)
        q_abs = new_kv_cache_pos.unsqueeze(1)  # [seq, 1] 绝对位
        k = torch.arange(kv_max).unsqueeze(0)  # [1, kv_max]
        attend = k <= q_abs  # [seq, kv_max] True=attend
        mask[:] = ~attend  # False=attend, True=mask out
        return mask

    def _build_mask_sliding(
        self,
        seq_len: int,
        kv_max: int,
        dtype: torch.dtype,
        new_kv_cache_pos: torch.Tensor,
    ) -> torch.Tensor:
        """bool 矩阵 [b,1,seq_len,kv_max], ring-buffer sliding window mask。

        sliding 层 KV cache 是 ring buffer (R=kv_max=sliding_window+seq_len, slot=pos%R)。
        query@abs q → ring slot qs=q%R; attend 窗口 [max(0,q-w+1),q] 映射到 slot:
          - 不绕 (q<w 被 max clamp, 或 qs≥w-1): attend [max(0,qs-w+1), qs]
          - 绕回 (q≥R 且 qs<w-1, 即跨过 R 边界): attend [0,qs] ∪ [R-w+1+qs, R-1]
        kv_max ≤ w 时退化为 mask_full (无 ring 必要)。
        """
        w = self._sliding_window
        if not w or w <= 0 or kv_max <= w:
            return self._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
        R = kv_max
        q = new_kv_cache_pos  # [S] abs
        qs = q % R  # [S] ring slot
        j = torch.arange(R)  # [R] cache slot
        qe = q.unsqueeze(1)  # [S,1]
        qse = qs.unsqueeze(1)  # [S,1]
        je = j.unsqueeze(0)  # [1,R]
        # 不绕: attend [max(0,qs-w+1), qs] (q<w 时 qs-w+1<0 → j≥负数恒真 → clamp [0,qs])
        attend_nw = (je <= qse) & (je >= (qse - w + 1))
        # 绕回: q≥R 且 qs<w-1 → attend [0,qs] ∪ [R-w+1+qs, R-1]
        wrap = (qe >= R) & (qse < (w - 1))  # [S,1]
        attend_w = (je <= qse) | (je >= (R - w + 1 + qse))
        attend = torch.where(wrap, attend_w, attend_nw)  # [S,R]
        mask = torch.ones(self.batch, 1, seq_len, R, dtype=torch.bool)
        mask[:] = ~attend
        return mask

    # ──────────────────────────────────────────────────────────────
    # get_sample_inputs (override): 2 mask + per_layer_token_identity
    # ──────────────────────────────────────────────────────────────

    def get_sample_inputs(
        self,
        seq_len: int = 64,
        use_embeddings: bool = False,
        quant_embeddings: bool = False,
        prefill_seq: int | None = None,
    ) -> tuple:
        """构造导出用示例输入 (scatter-write + host 预构 mask)。

        返回顺序与 forward 一致:
        (input_ids, mask_full, mask_sliding, position_ids, past_key_values,
         new_kv_cache_pos, embeddings, output_pos, per_layer_token_identity, embeddings_scale)
        """
        cfg = self.config
        num_layers = self._num_layers()
        num_kv_heads = cfg.num_key_value_heads
        kv_max = self.kv_cache_max_len
        dtype = self._model_dtype
        w = self._sliding_window
        # sliding 层 KV cache 是 ring buffer: R = sliding_window + prefill_seq (固定, 不跟 chunk seq_len
        # 走; prefill+decode 共用同一 R, KV 才能跨 prefill→decode 传). decode(seq=1) 也用 w+prefill_seq.
        # full 层仍用 kv_cache_max_len。slot = pos % R (跨 chunk 稳定, ring 不越界)。
        _rseq = prefill_seq if prefill_seq else seq_len
        R_sliding = (w + _rseq) if (w and w > 0) else kv_max

        input_ids = torch.randint(
            0, cfg.vocab_size, (self.batch, seq_len), dtype=torch.int64
        )
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int32)
        mask_full = self._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
        mask_sliding = self._build_mask_sliding(
            seq_len, R_sliding, dtype, new_kv_cache_pos
        )
        position_ids = (
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).repeat(self.batch, 1)
        )
        output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)
        # ring 槽位 (sliding 层 scatter 用, host 预算避图内 % ): pos % R_sliding
        sliding_kv_pos = (new_kv_cache_pos % R_sliding).to(torch.int32)

        past_key_values = []
        for i in range(num_layers):
            hd = self._get_layer_head_dim(i)
            is_sliding = (
                i < len(self._layer_types)
                and self._layer_types[i] == "sliding_attention"
            )
            kv_len_i = R_sliding if is_sliding else kv_max
            k = torch.zeros(self.batch, num_kv_heads, kv_len_i, hd, dtype=dtype)
            v = torch.zeros(self.batch, num_kv_heads, kv_len_i, hd, dtype=dtype)
            past_key_values.append((k, v))

        if use_embeddings:
            if quant_embeddings:
                q_w, q_s = self._quant_embeddings["model.model.embed_tokens.weight"]
                emb_q = q_w[input_ids]  # int8 [b, s, hidden]
                emb_scale = q_s[
                    input_ids
                ]  # [fold 回迁 2026-10-27] raw per-token scale (embed_scale 已在 forward 图内乘)
                # PLE quant: gather int8 + per-token scale from _quant_embeddings (dopt Quant_Embed_MinMax)。
                ple_q, ple_scale = None, None
                ple_key = "model.model.embed_tokens_per_layer.weight"
                if self._ple_enabled and ple_key in self._quant_embeddings:
                    m = self.model.model
                    pq_w, pq_s = self._quant_embeddings[ple_key]
                    ple_q = pq_w[input_ids].reshape(
                        *input_ids.shape,
                        cfg.num_hidden_layers,
                        m.hidden_size_per_layer_input,
                    )
                    ple_scale = pq_s[
                        input_ids
                    ]  # [fold 回迁 2026-10-27] raw per-token scale (ple scale 已在 forward 图内乘)
                return (
                    None,
                    mask_full,
                    mask_sliding,
                    position_ids,
                    past_key_values,
                    new_kv_cache_pos,
                    emb_q,
                    output_pos,
                    ple_q,
                    emb_scale,
                    ple_scale,
                    sliding_kv_pos,
                )
            # 非 quant: raw lookup (bypass Gemma4TextScaledWordEmbedding, 不乘 embed_scale),
            # embed_scale 折进 embeddings_scale 在图内乘 (与量化路径统一)。
            tm = self.model.model
            embeddings = tm.embed_tokens.weight[input_ids].detach()
            embeddings_scale = torch.full(
                (self.batch, seq_len), self._embed_scale, dtype=dtype
            )
            extras = self._build_external_extras(input_ids)
            ple = extras[0] if extras else None
            ple_scale = (
                torch.full((self.batch, seq_len), self._ple_embed_scale, dtype=dtype)
                if ple is not None
                else None
            )
            return (
                None,
                mask_full,
                mask_sliding,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                embeddings,
                output_pos,
                ple,
                embeddings_scale,
                ple_scale,
                sliding_kv_pos,
            )
        else:
            # 图内查表: input_ids 入图, embeddings/per_layer_token_identity=None (wrapper 内查)
            return (
                input_ids,
                mask_full,
                mask_sliding,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                None,
                output_pos,
                None,
                None,
                None,
                sliding_kv_pos,
            )

    # ──────────────────────────────────────────────────────────────
    # PLE host-side (保留基类契约)
    # ──────────────────────────────────────────────────────────────

    def _build_external_extras(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """gemma4 PLE: host 筹好 per_layer_token_identity (raw lookup, 不含 embed_scale)。
        embed_scale 由 get_sample_inputs 折进 ple_scale, forward 图内乘。
        shape [batch, seq, num_hidden_layers, hidden_size_per_layer_input]。
        num_hidden_layers 取 config 全量值 (per_layer_model_projection 按 full 维度建), 与
        _compute_per_layer_inputs 一致; --num-layers 截断时多余切片在层循环里被忽略。
        """
        m = self.model.model
        token_identity = (
            m.embed_tokens_per_layer.weight[input_ids]
            .reshape(
                *input_ids.shape,
                m.config.num_hidden_layers,
                m.hidden_size_per_layer_input,
            )
            .detach()
        )
        return (token_identity,)

    def host_side_weight_names(self) -> list[str]:
        """gemma4 纯 embeddings: embed_tokens + embed_tokens_per_layer (PLE) 查表都外置到 host。"""
        return [
            "model.model.embed_tokens.weight",
            "model.model.embed_tokens_per_layer.weight",
        ]

    def detach_externalized_lookups(self) -> None:
        """gemma4: 额外 detach embed_tokens_per_layer (PLE 查表, host 预算跳过)。"""
        text_model = self.model.model
        if hasattr(text_model, "embed_tokens"):
            del text_model.embed_tokens
        if hasattr(text_model, "embed_tokens_per_layer"):
            del text_model.embed_tokens_per_layer

    # ──────────────────────────────────────────────────────────────
    # ONNX 辅助 (override 以匹配新 I/O; saver 不用, 仅供 ONNX 路径)
    # ──────────────────────────────────────────────────────────────

    def make_sample_inputs(
        self, seq_len: int = 8, dtype: torch.dtype | None = None
    ) -> tuple:
        cfg = self.config
        num_kv_heads = cfg.num_key_value_heads
        num_layers = self._num_layers()
        kv_max = self.kv_cache_max_len
        if dtype is None:
            dtype = self._model_dtype

        input_ids = torch.randint(
            0, cfg.vocab_size, (self.batch, seq_len), dtype=torch.int64
        )
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int32)
        mask_full = self._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
        mask_sliding = self._build_mask_sliding(
            seq_len, kv_max, dtype, new_kv_cache_pos
        )
        position_ids = (
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).repeat(self.batch, 1)
        )

        past_key_values = []
        for i in range(num_layers):
            hd = self._get_layer_head_dim(i)
            k = torch.zeros(self.batch, num_kv_heads, kv_max, hd, dtype=dtype)
            v = torch.zeros(self.batch, num_kv_heads, kv_max, hd, dtype=dtype)
            past_key_values.append((k, v))

        # embeddings=None (图内查表), output_pos=None (返回全部 logits), ple=None, scale=None
        return (
            input_ids,
            mask_full,
            mask_sliding,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            None,
            None,
            None,
            None,
        )

    def make_input_output_names(self) -> tuple[list[str], list[str]]:
        n = self._num_layers()
        input_names = [
            "input_ids",
            "mask_full",
            "mask_sliding",
            "position_ids",
            "new_kv_cache_pos",
        ]
        for i in range(n):
            input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names = ["logits"]
        for i in range(n):
            output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
        return input_names, output_names
