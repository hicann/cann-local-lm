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
      纯 embeddings 模式 — input_ids 不作为图输入 (None placeholder)。
      gemma4 因 PLE (Per-Layer Embedding) 的 embed_tokens_per_layer 查表也外置,
      额外传 per_layer_token_identity (host 算好的 token identity); qwen3/qwen3_5
      无 PLE, 仅传 embeddings。
    """

    def __init__(self, model: nn.Module, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch
        self._model_dtype = next(model.parameters()).dtype
        # quant-embeddings 模式: {fqn: (int8_weight, per_channel_scale)} by set_quant_embeddings
        self._quant_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_quant_embeddings(self, quant: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """注入 host 查表的 int8 + per-channel scale (quant-embeddings 模式)。

        key = state_dict FQN (如 'model.model.embed_tokens.weight')；value =
        (int8_weight [vocab, hidden], scale [vocab])，来自 dopt Quant_Embed_MinMax
        产物经 materialize_int8_per_channel_symmetric。get_sample_inputs(quant_embeddings=True)
        用它 gather int8+scale 喂图；图入口 dequant 回 float (见 forward)。
        """
        self._quant_embeddings = dict(quant)

    def _num_layers(self) -> int:
        """实际导出层数(支持 --num-layers 截断).

        各模型统一路径 model.model.layers (ForCausalLM → TextModel → layers).
        截断后 len(self.model.model.layers) == N, 小于 config.num_hidden_layers.
        """
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
        # quant-embeddings: int8 embeddings + per-token (gathered per-vocab) scale
        # -> float (model dtype)。signed, 无 zero-point (dopt offset=0)。
        # dequant 在 base forward 做, 子类 _drive_layers 收 float embeddings, 无感。
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
            logits = logits.index_select(1, output_pos.to(torch.int64))
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
        """子类实现：统一 I/O → 原生 forward kwargs → 提取 logits + 展平 KV cache + 展平 delta

        per_layer_token_identity 仅 gemma4 (PLE) 使用; qwen3/qwen3_5 忽略。
        delta_states 仅 qwen3_5 (linear attention recurrent state) 使用; 其他返回 []。
        delta_states = (conv_flat, recurrent_flat): 每层 conv_state / recurrent_state
        展平图输入 (decode 导出用, 避 SimpleDeltaNetCache 对象被 export 当常量 mutate)。
        """
        raise NotImplementedError

    def get_sample_inputs(self, seq_len: int = 128, use_embeddings: bool = False,
                          quant_embeddings: bool = False) -> tuple:
        """构造导出用示例输入 (scatter-write 模式)

        预分配零 KV buffer [batch, heads, kv_cache_max_len, head_dim] +
        causal-over-buffer mask [batch, 1, seq_len, kv_cache_max_len] —
        k_pos <= q_pos 才可见, 未写入位置自动被屏蔽.

        要求模型的 attention 用 ``index_copy`` scatter-write (新 KV 写入 buffer,
        attn 读取的 K = buffer 本身, K 维度 = kv_cache_max_len).
        非 scatter 模型 (仍用 ``torch.cat`` 拼接 past+new, K 维度 =
        kv_cache_max_len + seq_len) 导出会因维度不匹配报 broadcast 错误,
        需先完成 scatter-write 改造 (见 converter_design.md 遗留任务 5).

        Args:
            seq_len: 示例序列长度
            use_embeddings: True 时传 embeddings (图外查表), False 时传 input_ids (图内查表)
        """
        config = self.config
        num_layers = self._num_layers()
        num_kv_heads = config.num_key_value_heads
        vocab_size = config.vocab_size
        dtype = next(self.model.parameters()).dtype
        kv_max = self.kv_cache_max_len

        input_ids = torch.randint(0, vocab_size, (self.batch, seq_len), dtype=torch.int64)

        # 4D causal mask [batch, 1, seq_len, kv_max] — causal over buffer (scatter-write)
        # 与 make_sample_inputs / make_causal_mask_over_buffer 一致.
        min_val = torch.finfo(dtype).min
        q_pos = torch.arange(seq_len, dtype=dtype).unsqueeze(1)        # [seq_len, 1]
        k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)        # [1, kv_max]
        mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
        attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)

        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(self.batch, 1)
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int64)
        output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

        # 预分配零 KV buffer: list[(k, v), ...]
        past_key_values = []
        for i in range(num_layers):
            layer_head_dim = self._get_layer_head_dim(i)
            k = torch.zeros(self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype)
            v = torch.zeros(self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype)
            past_key_values.append((k, v))

        if use_embeddings:
            extras = self._build_external_extras(input_ids)
            ple = extras[0] if extras else None  # per_layer_token_identity (gemma4 PLE; qwen3=None)
            if quant_embeddings:
                # int8 per-channel: host gather int8_weight[input_ids] + scale[input_ids]
                # 图输入 int8 [b,s,hidden] + scale [b,s]；图入口 dequant (base forward)
                q_w, q_s = self._quant_embeddings["model.model.embed_tokens.weight"]
                emb_q = q_w[input_ids]            # int8 [b, s, hidden]
                emb_scale = q_s[input_ids]         # [b, s]
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
            # 纯 embeddings (float): input_ids 不作为图输入 (None placeholder)。
            # host 用 embed_tokens 查表算出 embeddings (+ gemma4 的 per_layer_token_identity);
            # model forward 收到 inputs_embeds → 跳过 embed_tokens(input_ids), input_ids 不入图。
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
            # use_embeddings=False: 图内查表, embeddings=None.
            # _drive_layers 判断 embeddings is None → 只传 input_ids, model 走 embed_tokens 查表。
            # torchair export_to_epair 已接受 None placeholder (不再 AssertionError).
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
    # ONNX export 辅助方法（通用默认实现，子类可 override）
    # ──────────────────────────────────────────────────────────────

    def _get_layer_head_dim(self, layer_idx: int) -> int:
        """每层的 KV head 维度。默认返回 config.head_dim。

        gemma4 等有 per-layer head_dim 的模型 override 此方法
        （sliding 层用 head_dim，full 层用 global_head_dim）。
        """
        return self.config.head_dim

    def _build_external_extras(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """纯 embeddings 模式下, 除 embeddings 外额外的 host 算好图输入 (默认空)。

        子类若有除 embed_tokens 外的、依赖 input_ids 的查表 (如 gemma4 的 PLE
        embed_tokens_per_layer), override 此方法返回 host 算好的张量, 使该查表
        外置、input_ids 不入图。
        """
        return ()

    def host_side_weight_names(self) -> list[str]:
        """纯 embeddings 模式下, 需作为 host-side extra 打包的查表权重 state_dict FQN。

        FQN 用 wrapper 层路径 (与 ep.graph_signature 一致): wrapper.model = ForCausalLM,
        ForCausalLM.model = TextModel, TextModel.embed_tokens → 'model.model.embed_tokens.weight'。
        这些权重因查表外置而不在图中 (pack_weights 不打包); 打包成独立的
        host_weights_info.json + host_weights_*.bin 供 host 推理时查表。
        默认 embed_tokens.weight (tie 时按 storage identity 自动去重 — 仍经 lm_head 留在图内)。
        gemma4 等 override 追加模型特有的查表权重 (embed_tokens_per_layer)。
        """
        return ["model.model.embed_tokens.weight"]

    def detach_externalized_lookups(self) -> None:
        """Detach 外置的查表子模块, 使 torch.export 不 lift 它们 (纯 embeddings 模式)。

        torch.export 会 lift 全部注册 Parameter (含未用的 embed_tokens), 导致 weights_*.bin
        携带 OM 编译时本就剪掉的冗余查表权重。detach 子模块后, 它不进 state_dict → 不 lift
        → weights_*.bin ≈ OM, 无冗余, C++ 也不用为它建 Data 节点。

        必须在 get_sample_inputs (用它算 sample) + host 权重抽取 **之后**、torch.export **之前** 调用。
        依赖: 纯 embeddings 下 forward 不访问这些子模块 (embed_tokens 被 inputs_embeds 跳过)。
        gemma4 override 追加 embed_tokens_per_layer (PLE, 被实例方法替换跳过)。
        """
        text_model = self.model.model
        if hasattr(text_model, "embed_tokens"):
            del text_model.embed_tokens

    def make_sample_inputs(self, seq_len: int = 8, dtype: torch.dtype | None = None) -> tuple:
        """构造 ONNX 导出用示例输入（带预分配 KV buffer 的全量版）。

        与 get_sample_inputs（轻量版，past=None）不同，这里传入非 None 的
        past_key_values（零 tensor）+ cache-extended mask，使 ONNX 图含 KV cache I/O。

        forward 签名: (input_ids, attention_mask, position_ids, past_key_values,
                       new_kv_cache_pos, embeddings, output_pos)
        其中 embeddings=None / output_pos=None 在 tracing 时被 Python 分支解析，
        不会成为 ONNX 输入。past_key_values 是 list[(k,v),...] 会被 ONNX 展平。
        """
        config = self.config
        num_kv_heads = config.num_key_value_heads
        vocab_size = config.vocab_size
        num_layers = self._num_layers()
        kv_max = self.kv_cache_max_len
        if dtype is None:
            dtype = next(self.model.parameters()).dtype

        # 1. input_ids
        input_ids = torch.randint(0, vocab_size, (self.batch, seq_len), dtype=torch.int64)

        # 2. attention_mask [batch, 1, seq_len, kv_max] — causal over buffer
        #    k_pos <= q_pos → attend, k_pos > q_pos → mask (未写入位置自动屏蔽)
        min_val = torch.finfo(dtype).min
        q_pos = torch.arange(seq_len, dtype=dtype).unsqueeze(1)
        k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
        mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
        attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(self.batch, 1, seq_len, kv_max)

        # 3. position_ids
        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(self.batch, 1)

        # 4. past_key_values: list[(k, v), ...] — 零 tensor, per-layer head_dim
        past_key_values = []
        for i in range(num_layers):
            layer_head_dim = self._get_layer_head_dim(i)
            k = torch.zeros(self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype)
            v = torch.zeros(self.batch, num_kv_heads, kv_max, layer_head_dim, dtype=dtype)
            past_key_values.append((k, v))

        # 5. new_kv_cache_pos
        new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int64)

        # 6. embeddings=None (图内查表)  7. output_pos=None (返回全部 logits)
        return (input_ids, attention_mask, position_ids, past_key_values,
                new_kv_cache_pos, None, None)

    def make_input_output_names(self) -> tuple[list[str], list[str]]:
        """生成 ONNX input/output 名称（全 KV cache 模型通用）。

        ONNX 会把 past_key_values (list[(k,v),...]) 展平为独立 tensor 输入。
        qwen3.5 等混合层模型需 override（linear 层用 delta_state，命名不同）。
        """
        n = self._num_layers()
        input_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]
        for i in range(n):
            input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names = ["logits"]
        for i in range(n):
            output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
        return input_names, output_names
