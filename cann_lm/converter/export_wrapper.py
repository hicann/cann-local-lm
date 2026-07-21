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
    """

    def __init__(self, model: nn.Module, *, kv_cache_max_len: int = 2048, batch: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.kv_cache_max_len = kv_cache_max_len
        self.batch = batch

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: Any = None,
        new_kv_cache_pos: torch.Tensor = None,
        embeddings: torch.Tensor | None = None,
        output_pos: torch.Tensor | None = None,
    ):
        logits, past_kv_flat = self._drive_layers(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos,
            embeddings,
        )
        if output_pos is not None:
            logits = logits[:, output_pos, :]
        return logits, *past_kv_flat

    def _drive_layers(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        new_kv_cache_pos: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """子类实现：统一 I/O → 原生 forward kwargs → 提取 logits + 展平 KV cache"""
        raise NotImplementedError

    def get_sample_inputs(self, seq_len: int = 128, use_embeddings: bool = False) -> tuple:
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
        num_layers = config.num_hidden_layers
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
            # use_embeddings=True: 图外查表 (embed_tokens 不在图内), 但 PLE 仍需 input_ids
            # → 同时返回 input_ids + embeddings. _drive_layers 同时传两者给 model:
            # model 用 inputs_embeds (不查 embed_tokens), PLE 用 input_ids (不反查),
            # 避开 bfloat16 下 inputs_embeds == weight 反查的精度问题.
            embeddings = self.model.model.embed_tokens(input_ids).detach()
            return (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                new_kv_cache_pos,
                embeddings,
                output_pos,
            )
        else:
            # use_embeddings=False: 图内查表, embeddings=None.
            # _drive_layers 判断 embeddings is None → 不传 inputs_embeds, 只传 input_ids,
            # model 走 input_ids 路径 (embed_tokens 查表), PLE 有 input_ids 不反查.
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
        num_layers = config.num_hidden_layers
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
        n = self.config.num_hidden_layers
        input_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]
        for i in range(n):
            input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names = ["logits"]
        for i in range(n):
            output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
        return input_names, output_names
