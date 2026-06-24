"""
Qwen3.5 Pure PyTorch — model assembly.
No transformers dependency — only torch.

Assembles TextModel + ForCausalLM from modular building blocks.
Provides weight loading (safetensors) and CLI inference.

Usage:
  python modeling_qwen3_5_pure.py --model-path /path/to/Qwen3.5-397B-A17B --prompt "Hello"
"""

import json
import os
import struct
import time
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_qwen3_5_pure import (
    Qwen3_5TextConfig,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5DecoderLayer,
    DynamicCache,
    create_causal_mask,
    CausalLMOutput,
)


# ============================================================
# Text Model
# ============================================================

class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)

        if position_ids.ndim == 2:
            text_position_ids = position_ids
            mrope_position_ids = position_ids[None, ...].expand(3, -1, -1)
        else:
            text_position_ids = position_ids[0] if position_ids.shape[0] > 1 else position_ids
            mrope_position_ids = position_ids

        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._make_causal_mask(seq_len, past_key_values, inputs_embeds.device, inputs_embeds.dtype)
            if attention_mask is not None and attention_mask.dim() == 2:
                causal_mask = causal_mask.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        else:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            causal_mask = create_causal_mask(seq_len, past_length, inputs_embeds.device, inputs_embeds.dtype)

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=layer_mask, position_ids=text_position_ids,
                past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state(-1)) or \
           (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    def _make_causal_mask(self, seq_len, past_key_values, device, dtype):
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        return create_causal_mask(seq_len, past_length, device, dtype)


# ============================================================
# ForCausalLM
# ============================================================

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False, labels=None,
                logits_to_keep=0, **kwargs):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss, logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7,
                 top_k=20, top_p=0.95, eos_token_id=248044):
        self.eval()
        device = input_ids.device
        batch, prompt_len = input_ids.shape

        past_key_values = DynamicCache(config=self.config)
        hidden_states, past_key_values = self.model(
            input_ids, use_cache=True, past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states[:, -1:, :])

        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            position_ids = torch.tensor([[prompt_len + len(generated) - 1]], device=device)
            hidden_states, past_key_values = self.model(
                next_token, position_ids=position_ids,
                use_cache=True, past_key_values=past_key_values,
            )
            logits = self.lm_head(hidden_states[:, -1:, :])
            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(next_token.item())
            if next_token.item() == eos_token_id:
                break

        return torch.cat([input_ids, torch.tensor([generated], device=device)], dim=-1)

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, -1:]] = float('-inf')
        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ============================================================
# Safetensors Loader
# ============================================================

def load_safetensors_file(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        result = {}
        for key, metadata in header.items():
            if key == '__metadata__':
                continue
            offset = metadata['data_offsets'][0]
            size = metadata['data_offsets'][1] - offset
            f.seek(8 + header_len + offset)
            tensor_bytes = f.read(size)
            dtype = _safetensors_dtype_to_torch(metadata['dtype'])
            shape = metadata['shape']
            tensor = torch.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
            result[key] = tensor
    return result


def _safetensors_dtype_to_torch(dtype_str):
    mapping = {
        'F16': torch.float16, 'F32': torch.float32, 'BF16': torch.bfloat16,
        'I8': torch.int8, 'I16': torch.int16, 'I32': torch.int32,
        'I64': torch.int64, 'U8': torch.uint8,
    }
    return mapping.get(dtype_str, torch.float32)


def find_safetensors_files(model_path):
    files = []
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            files.append(os.path.join(model_path, f))
    return files


def resolve_model_path(model_path_or_id):
    if os.path.exists(model_path_or_id):
        return os.path.abspath(model_path_or_id)
    print(f"Model '{model_path_or_id}' not found locally.")
    raise FileNotFoundError(f"Model not found: {model_path_or_id}")


# ============================================================
# Weight Loading
# ============================================================

def _build_weight_map(model, state_dict):
    hf_prefix = 'model.layers.'
    mapping = {}
    for name, _ in model.named_parameters():
        if not name.startswith('model.layers.'):
            continue
        rest = name[len('model.layers.'):]
        dot = rest.index('.') if '.' in rest else -1
        if dot < 0:
            continue
        layer_idx = rest[:dot]
        layer_rest = rest[dot+1:]

        if layer_rest.startswith('linear_attn'):
            sub = layer_rest[len('linear_attn.'):]
            if sub.startswith('in_proj_qkv'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_qkv.weight'
            elif sub.startswith('in_proj_z'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_z.weight'
            elif sub.startswith('in_proj_b'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_b.weight'
            elif sub.startswith('in_proj_a'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_a.weight'
            elif sub.startswith('conv1d'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.conv1d.weight'
            else:
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.{sub}'
        elif layer_rest.startswith('self_attn'):
            mapping[name] = f'{hf_prefix}{layer_idx}.self_attn.{layer_rest[len("self_attn."):]}'
        elif layer_rest.startswith('input_layernorm') or layer_rest.startswith('post_attention_layernorm'):
            mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'
        elif layer_rest.startswith('mlp'):
            mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'
        else:
            mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'

    non_layer_keys = {
        'model.embed_tokens.weight': 'model.embed_tokens.weight',
        'model.norm.weight': 'model.norm.weight',
        'lm_head.weight': 'lm_head.weight',
    }
    mapping.update(non_layer_keys)
    return mapping


def load_qwen3_5_weights(model, model_path, dtype=torch.bfloat16):
    safetensors_files = find_safetensors_files(model_path)
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    print(f"Loading {len(safetensors_files)} safetensors file(s)...")
    state_dict = {}
    for sf in safetensors_files:
        print(f"  Loading {os.path.basename(sf)}...")
        state_dict.update(load_safetensors_file(sf))

    mapping = _build_weight_map(model, state_dict)

    loaded = 0
    missing = []
    for name, param in model.named_parameters():
        hf_name = mapping.get(name, name)
        if hf_name in state_dict:
            w = state_dict[hf_name].to(dtype)
            if w.shape == param.shape:
                param.data.copy_(w)
                loaded += 1
                continue
        if name in state_dict and state_dict[name].shape == param.shape:
            param.data.copy_(state_dict[name].to(dtype))
            loaded += 1
            continue
        if name not in ('A_log', 'dt_bias') and \
           not any(x in name for x in ['A_log', 'dt_bias']):
            missing.append(name)

    total = sum(1 for _ in model.named_parameters())
    print(f"Loaded {loaded} / {total} weights")
    if missing:
        print(f"  Missing {len(missing)} weights (first 10):")
        for n in missing[:10]:
            print(f"    - {n}")
    return model


# ============================================================
# Main / CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Qwen3.5 Pure PyTorch Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model directory')
    parser.add_argument('--prompt', type=str, default='Hello, how are you?', help='Input prompt')
    parser.add_argument('--max-new-tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=20, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda, cpu, auto)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map[args.dtype]

    model_path = resolve_model_path(args.model_path)

    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            hf_config = json.load(f)
        text_config_dict = hf_config.get("text_config", hf_config)
        config = Qwen3_5TextConfig.from_dict(text_config_dict)
    else:
        print("Warning: config.json not found, using defaults")
        config = Qwen3_5TextConfig()

    print(f"\nQwen3.5 Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")

    print("\nBuilding model...")
    t0 = time.time()
    model = Qwen3_5ForCausalLM(config)
    model = model.to(dtype).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nLoading weights...")
    load_qwen3_5_weights(model, model_path, dtype)
    model = model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {input_ids.shape[-1]}")

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        generated = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, eos_token_id=tokenizer.eos_token_id,
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

    output_text = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated ({generated.shape[-1] - input_ids.shape[-1]} tokens in {t2-t1:.2f}s):")
    print(output_text)


if __name__ == '__main__':
    main()
