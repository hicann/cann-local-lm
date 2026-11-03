"""Export wrapper base class.

Mirrors litert-torch's ExportableModuleBase design:
- A base class unifies the external I/O contract (input_ids/embeddings, attention_mask,
  position_ids, past_key_values, ...).
- Subclasses live under each model dir (e.g. cann_local_lm/models/qwen3_5/export_wrapper.py).
- Subclasses implement _drive_layers: translate the unified I/O into native model forward kwargs.
- Supports both token_ids (in-graph lookup) and embeddings (out-of-graph lookup) modes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Unified external I/O: cache in/out + output_pos. Subclasses implement _drive_layers.

    use_embeddings=False: input_ids in-graph (default). True: embeddings out-of-graph (input_ids is a
    None placeholder); gemma4 also passes per_layer_token_identity for PLE.
    """

    def __init__(
        self, model: nn.Module, *, kv_cache_max_len: int = 2048, batch: int = 1
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch
        self._model_dtype = next(model.parameters()).dtype
        # quant-embeddings mode: {fqn: (int8_weight, per_channel_scale)} set by set_quant_embeddings
        self._quant_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_quant_embeddings(
        self, quant: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Inject host-lookup int8 + per-channel scale (quant-embeddings mode); get_sample_inputs gathers, forward dequants."""
        self._quant_embeddings = dict(quant)

    def _num_layers(self) -> int:
        """Exported layer count (model.model.layers; supports --num-layers truncation)."""
        return len(self.model.model.layers)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: Any = None,
        new_kv_cache_pos: torch.Tensor = None,
        embeddings: torch.Tensor | None = None,
        output_pos: torch.Tensor | None = None,
        per_layer_token_identity: torch.Tensor | None = None,
        embeddings_scale: torch.Tensor | None = None,
        delta_states: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ):
        # quant-embeddings: dequant int8 embeddings + per-token scale -> float (signed, offset=0).
        if embeddings is not None and embeddings_scale is not None:
            dt = self._model_dtype
            embeddings = embeddings.to(dt) * embeddings_scale.to(dt).unsqueeze(-1)
        logits, past_kv_flat, delta_out_flat = self._drive_layers(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            embeddings,
            per_layer_token_identity,
            delta_states,
        )
        if output_pos is not None:
            logits = logits.index_select(1, output_pos)
        return logits, *past_kv_flat, *delta_out_flat

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        per_layer_token_identity: torch.Tensor | None = None,
        delta_states: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Subclass impl: unified I/O -> native forward -> logits + flattened KV/delta.

        per_layer_token_identity: gemma4 PLE only. delta_states (conv_flat, recurrent_flat): qwen3_5
        linear-attn recurrent state, flattened to avoid SimpleDeltaNetCache being mutated as a constant.
        """
        raise NotImplementedError

    def get_sample_inputs(
        self,
        seq_len: int = 128,
        use_embeddings: bool = False,
        quant_embeddings: bool = False,
    ) -> tuple:
        """Build sample inputs for export (scatter-write: pre-allocated zero KV buffer + causal-over-buffer mask).

        Requires attention to use index_copy scatter-write (K dim = kv_cache_max_len); non-scatter
        (torch.cat) models fail export — see converter_design.md TODO 5.
        """
        config = self.config
        num_layers = self._num_layers()
        num_kv_heads = config.num_key_value_heads
        vocab_size = config.vocab_size
        dtype = next(self.model.parameters()).dtype
        kv_max = self.kv_cache_max_len

        input_ids = torch.randint(
            0, vocab_size, (self.batch, seq_len), dtype=torch.int64
        )

        # 4D causal mask [batch, 1, seq_len, kv_max] — causal over buffer (scatter-write).
        # BOOL mask (False=attend, True=mask out): kirinx90 无 FP16→BOOL Cast kernel, 须直传 BOOL.
        q_pos = torch.arange(seq_len, dtype=torch.int32).unsqueeze(1)  # [seq_len, 1]
        k_pos = torch.arange(kv_max, dtype=torch.int32).unsqueeze(0)  # [1, kv_max]
        mask = k_pos <= q_pos  # [seq_len, kv_max] bool (True=attend)
        attention_mask = (
            (~mask).unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)
        )

        position_ids = (
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).repeat(self.batch, 1)
        )
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int32)
        output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

        past_key_values = []
        for i in range(num_layers):
            layer_head_dim = self._get_layer_head_dim(i)
            k = torch.zeros(
                self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype
            )
            v = torch.zeros(
                self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype
            )
            past_key_values.append((k, v))

        if use_embeddings:
            extras = self._build_external_extras(input_ids)
            ple = (
                extras[0] if extras else None
            )  # per_layer_token_identity (gemma4 PLE; qwen3_5=None)
            if quant_embeddings:
                # int8 per-channel: host gathers int8_weight[input_ids] + scale[input_ids]; graph entry dequants (base forward).
                q_w, q_s = self._quant_embeddings["model.model.embed_tokens.weight"]
                emb_q = q_w[input_ids]
                emb_scale = q_s[input_ids]
                return (
                    None,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    new_kv_cache_pos,
                    emb_q,
                    output_pos,
                    ple,
                    emb_scale,
                )
            # Pure embeddings (float): input_ids is a None placeholder; host computes embeddings (+ gemma4 PLE).
            embeddings = self.model.model.embed_tokens(input_ids).detach()
            return (
                None,
                attention_mask,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                embeddings,
                output_pos,
                *extras,
            )
        else:
            # use_embeddings=False: in-graph lookup; torchair export_to_epair accepts the None placeholder.
            return (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                None,
                output_pos,
            )

    # ──────────────────────────────────────────────────────────────
    # ONNX export helpers (generic defaults; subclasses may override)
    # ──────────────────────────────────────────────────────────────

    def _get_layer_head_dim(self, layer_idx: int) -> int:
        """KV head dim per layer (default config.head_dim; override for per-layer head_dim e.g. gemma4)."""
        return self.config.head_dim

    def _build_external_extras(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Pure-embeddings extra host-computed graph inputs beyond embeddings (default empty; override for PLE etc.)."""
        return ()

    def host_side_weight_names(self) -> list[str]:
        """Lookup-weight FQNs to pack as host-side extras in pure-embeddings mode (default embed_tokens.weight; override to add PLE)."""
        return ["model.model.embed_tokens.weight"]

    def detach_externalized_lookups(self) -> None:
        """Detach externalized lookup submodules so torch.export doesn't lift them (weights_*.bin ≈ OM).

        Call after get_sample_inputs + host weight extraction, before torch.export. Relies on forward
        not touching them (embed_tokens skipped via inputs_embeds; gemma4 PLE via instance-method replace).
        """
        text_model = self.model.model
        if hasattr(text_model, "embed_tokens"):
            del text_model.embed_tokens

    def make_sample_inputs(
        self, seq_len: int = 8, dtype: torch.dtype | None = None
    ) -> tuple:
        """Build sample inputs for ONNX export with pre-allocated KV buffer (unlike get_sample_inputs' past=None)."""
        config = self.config
        num_kv_heads = config.num_key_value_heads
        vocab_size = config.vocab_size
        num_layers = self._num_layers()
        kv_max = self.kv_cache_max_len
        if dtype is None:
            dtype = next(self.model.parameters()).dtype

        input_ids = torch.randint(
            0, vocab_size, (self.batch, seq_len), dtype=torch.int64
        )

        # attention_mask [batch, 1, seq_len, kv_max] — causal over buffer (BOOL).
        q_pos = torch.arange(seq_len, dtype=torch.int32).unsqueeze(1)
        k_pos = torch.arange(kv_max, dtype=torch.int32).unsqueeze(0)
        mask = k_pos <= q_pos
        attention_mask = (
            (~mask).unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)
        )

        position_ids = (
            torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).repeat(self.batch, 1)
        )

        past_key_values = []
        for i in range(num_layers):
            layer_head_dim = self._get_layer_head_dim(i)
            k = torch.zeros(
                self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype
            )
            v = torch.zeros(
                self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype
            )
            past_key_values.append((k, v))

        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int32)

        # embeddings=None (in-graph lookup); output_pos=None (return all logits).
        return (
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            None,
            None,
        )

    def make_input_output_names(self) -> tuple[list[str], list[str]]:
        """Generate ONNX input/output names (generic; mixed-layer models like qwen3.5 override)."""
        n = self._num_layers()
        input_names = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "new_kv_cache_pos",
        ]
        for i in range(n):
            input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names = ["logits"]
        for i in range(n):
            output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
        return input_names, output_names
