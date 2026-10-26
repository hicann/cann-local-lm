"""量化仿真推理 demo (参考 CANN 文档 qwen2 pattern)。

加载 dopt fakequant 模型 (W4A8: 权重 int4 + 激活 int8 量化) → chat_template → 逐 token argmax 生成 → 看量化后生成质量。

支持单卡 / 两卡两种模式:
  --mode single  (默认): 全模型在一张 NPU, 跳过 embedding 量化 (28GB << 61GB)
  --mode two_card: embedding→NPU0, 42层Linear→NPU1 (Gemma4Model.forward patch, 不用 accelerate)

gemma4 的 model.generate 撞 shared_kv_states 问题 → 用逐 token argmax 替代 (单次 forward → next token → append)。
"""

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

MODEL_PATH = os.environ.get(
    "GEMMA4_MODEL_PATH", "/mnt/workspace/CANN-LM_5883/models/unsloth/gemma-4-E4B-it"
)
QUANT_DIR = os.environ.get("DOPT_QUANT_DIR", "/tmp/opencode/dopt_gemma4_full")

from transformers import AutoTokenizer
from dopt_lm.opt_funcs.eco_stage.train import model_define
from dopt_lm.do_opt import optimize_model, set_quant_state, set_calibrate_state
import dopt_lm.quantizer.qmin_max
import dopt_lm.quantizer.qeval  # noqa: F401  # 注册量化器

QUANT_CFG = {
    "weight": {
        "blocksize": 128,
        "group_size": 128,
        "bit": 4,
        "per_channel": True,
        "weight_algo": "omni_group_min_max",
    },
    "input": {
        "bit": 8,
        "per_channel": False,
        "input_algo": "dyn_s8",
        "unsigned_quant": False,
    },
}


def _skip_embed_quant(model):
    """跳过 embedding 量化 (embed_tokens_per_layer [262144,10752] fakequant 中间张量 11.3GB 单卡放不下)。"""
    for name, m in model.named_modules():
        if hasattr(m, "quant_op") and m.quant_op is not None:
            if "embed" in name or "per_layer_model_projection" in name:
                m.quant_op.set_quant_state(weight=False, input=False)


def _move_to(t, dev):
    """递归移 tensor/dict/tuple 到 dev。"""
    if isinstance(t, torch.Tensor):
        return t.to(dev)
    if isinstance(t, dict):
        return {k: _move_to(v, dev) for k, v in t.items()}
    if isinstance(t, (tuple, list)):
        return type(t)(_move_to(v, dev) for v in t)
    return t


# ═══════════════════════════════════════════════════
# 模型加载 (共用)
# ═══════════════════════════════════════════════════


def load_quant_model(dopt_config, quanted_ckpt):
    """CPU 上加载 + dopt 量化装配 (共用部分)。"""
    model = model_define(MODEL_PATH, torch_dtype=torch.float16)
    model = optimize_model(model, dopt_config)
    model.load_state_dict(
        torch.load(quanted_ckpt, map_location="cpu", weights_only=True), strict=False
    )
    set_quant_state(model, weight_state=True, input_state=True)
    set_calibrate_state(model, False)
    _skip_embed_quant(model)
    model.eval()
    return model


# ═══════════════════════════════════════════════════
# 单卡模式
# ═══════════════════════════════════════════════════


def setup_single_card(model, device_id="1"):
    """单卡: 全模型 → NPU (跳过 embedding 量化后 ~28GB << 61GB)。"""
    import torch_npu  # noqa: F401

    dev = f"npu:{device_id}"
    model = model.to(dev)
    print(f"[单卡] model → {dev}", flush=True)
    return model, dev


# ═══════════════════════════════════════════════════
# 两卡模式 (Gemma4Model.forward patch)
# ═══════════════════════════════════════════════════


def setup_two_card(model, dev0="npu:0", dev1="npu:1"):
    """两卡: embedding→dev0, 42层+norm→dev1, lm_head→dev0 (Gemma4Model.forward patch)。

    不用 accelerate dispatch_model (hooks 干扰 shared_kv_states)。
    只 patch Gemma4Model.forward 一处: embed(dev0) → 全部移 dev1 → layers+norm(dev1) → 移回 dev0。
    """
    inner = model.model

    # 分卡
    inner.embed_tokens.to(dev0)
    inner.embed_tokens_per_layer.to(dev0)
    inner.per_layer_model_projection.to(dev0)
    inner.per_layer_projection_norm.to(dev0)
    for layer in inner.layers:
        layer.to(dev1)
    inner.norm.to(dev1)
    model.lm_head.to(dev0)

    # patch Gemma4Model.forward
    def patched_fwd(
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        # === embed 阶段 (dev0) ===
        if input_ids is not None:
            input_ids = input_ids.to(dev0)
            inputs_embeds = inner.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds.to(dev0)

        if inner.hidden_size_per_layer_input:
            per_layer_inputs = inner.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = inner.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )
        else:
            per_layer_inputs = None

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=inner.config)

        if position_ids is None:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = torch.arange(inputs_embeds.shape[1], device=dev0) + past_seen
            position_ids = position_ids.unsqueeze(0)

        unique_layer_types = getattr(inner, "unique_layer_types", None) or sorted(
            set(inner.config.layer_types)
        )
        position_embeddings = {}
        for lt in unique_layer_types:
            position_embeddings[lt] = inner.rotary_emb(inputs_embeds, position_ids, lt)

        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mask_kwargs = {
            "config": inner.config,
            "inputs_embeds": inputs_embeds[:1],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {}
        for lt in unique_layer_types:
            fn = (
                create_sliding_window_causal_mask
                if lt == "sliding_attention"
                else create_causal_mask
            )
            causal_mask_mapping[lt] = fn(**mask_kwargs)

        # === 移到 dev1 ===
        shared_kv_states = kwargs.pop("shared_kv_states", None) or {}
        hidden_states = inputs_embeds.to(dev1)
        per_layer_inputs = (
            _move_to(per_layer_inputs, dev1) if per_layer_inputs is not None else None
        )
        position_embeddings = _move_to(position_embeddings, dev1)
        causal_mask_mapping = _move_to(causal_mask_mapping, dev1)
        position_ids = position_ids.to(dev1)

        # === layers + norm (dev1, shared_kv_states 局部变量不受干扰) ===
        for i, decoder_layer in enumerate(
            inner.layers[: inner.config.num_hidden_layers]
        ):
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[inner.config.layer_types[i]],
                attention_mask=causal_mask_mapping[inner.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

        hidden_states = inner.norm(hidden_states)

        # === 移回 dev0 ===
        hidden_states = hidden_states.to(dev0)

        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextModelOutputWithPast,
        )

        return Gemma4TextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=None,
        )

    inner.forward = patched_fwd
    print(
        f"[两卡] embed+rotary→{dev0}, 42层+norm→{dev1}, lm_head→{dev0} (Gemma4Model.forward patch)",
        flush=True,
    )
    return model, dev0


# ═══════════════════════════════════════════════════
# 生成 (共用)
# ═══════════════════════════════════════════════════


def generate(model, tokenizer, prompt, first_dev, max_new=128):
    """逐 token argmax 生成 (绕过 gemma4 shared_kv_states generate 问题)。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(first_dev)
    input_ids = model_inputs["input_ids"]
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=generated)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    new_tokens = generated[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ═══════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=["single", "two_card"],
        default="single",
        help="single=单卡 (默认), two_card=两卡 (embedding→NPU0, Linear→NPU1)",
    )
    ap.add_argument(
        "--device-id", default="1", help="单卡模式的 NPU device id (默认 1)"
    )
    ap.add_argument("--max-new", type=int, default=128, help="最大生成 token 数")
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Give me a short introduction to large language model.",
            "who are you?",
        ],
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dopt_config = os.path.join(QUANT_DIR, "dopt_config.json")
    quanted_ckpt = os.path.join(QUANT_DIR, "trained_quant_weight.pth")

    print(f"[加载] fakequant 模型 (W4A8, 256×1024) → {args.mode} ...", flush=True)
    model = load_quant_model(dopt_config, quanted_ckpt)

    if args.mode == "single":
        model, first_dev = setup_single_card(model, args.device_id)
    else:
        model, first_dev = setup_two_card(model)
    print("[就绪]\n")

    for p in args.prompts:
        resp = generate(model, tokenizer, p, first_dev, max_new=args.max_new)
        print(f"[Q] {p}")
        print(f"[A] {resp}\n")


if __name__ == "__main__":
    raise SystemExit(main())
