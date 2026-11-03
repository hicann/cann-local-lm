# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#
# Adapted from NVlabs/EoRA (Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES,
#   NVIDIA Source Code License)
#   https://github.com/NVlabs/EoRA
import time

import gc
import torch
import torch.nn as nn
from dopt_lm.common_tools.log import Logger
from dopt_lm.opt_funcs.eco_stage.train import model_define
from dopt_lm.do_opt import paser_quant_config
from .ptq_algo import WeightQuantHessian
from .modelutils import DEV, empty_device_cache, find_layers
from .quant import Quantizer
from .datautils import get_loaders


def get_llama(model, seq_len, torch_dtype=torch.float32):
    model = model_define(model, torch_dtype=torch_dtype)
    model = model.to(DEV)
    model.model.layers = model.model.layers.to("cpu")
    model.seqlen = seq_len
    return model


def sort_sequent(one_layers):
    sequential = [
        [],
        [],
        [],
        [],
    ]
    for layer in one_layers:
        if "self_attn.q" in layer or "self_attn.k" in layer or "self_attn.v" in layer:
            sequential[0].append(layer)
        if "self_attn.o" in layer:
            sequential[1].append(layer)
        if "mlp.gate" in layer or "mlp.up" in layer:
            sequential[2].append(layer)
        if "mlp.down" in layer:
            sequential[3].append(layer)
    return sequential


def _get_weight_config(quant_config, layer_name):
    layer_config = quant_config.get(layer_name)
    if not isinstance(layer_config, dict):
        return None
    weight_config = layer_config.get("weight")
    return weight_config if isinstance(weight_config, dict) else None


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    Logger.info("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
            if hasattr(module, "layer_type"):
                self.layer_type = module.layer_type

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    input_ids_list = None
    if isinstance(dataloader, dict):
        block_size = model.seqlen
        input_embed = dataloader.get("input_embed")
        combined_embed = torch.cat(input_embed, dim=1)  # [:,:args.nsamples,:]
        combined_embed = combined_embed[
            :, : combined_embed.shape[1] // block_size * block_size, :
        ]
        seq_len = combined_embed.shape[1]
        total_steps = len(range(0, seq_len, block_size))
        Logger.info(f"total sample: {total_steps}")
        args.nsamples = total_steps
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device=dev,
        )
        for start_idx in range(0, seq_len, block_size):
            end_idx = start_idx + block_size
            embed = combined_embed[:, start_idx:end_idx, :]
            try:
                embed = embed.to(dev)
                model(inputs_embeds=embed)
            except ValueError:
                pass
        del combined_embed
        gc.collect()
        empty_device_cache()
    else:
        input_ids_list = []
        for batch in dataloader:
            input_ids_list.append(batch[0].clone())
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    empty_device_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    unique_layer_types = getattr(model.model, "unique_layer_types", None)
    causal_mask_map = {}
    position_embeddings_map = {}
    if unique_layer_types is not None:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mask_kwargs = dict(
            config=model.config,
            inputs_embeds=inps[:1],
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        for lt in unique_layer_types:
            mask_fn = (
                create_sliding_window_causal_mask
                if lt == "sliding_attention"
                else create_causal_mask
            )
            causal_mask_map[lt] = mask_fn(**mask_kwargs)
            position_embeddings_map[lt] = model.model.rotary_emb(
                inps[:1], position_ids, lt
            )

    ple_enabled = bool(getattr(model.model, "hidden_size_per_layer_input", None))
    if ple_enabled and input_ids_list is None:
        raise NotImplementedError(
            "Per-Layer Embeddings are enabled but the calibration dataloader provides no "
            "input_ids (embedding-only path). PLE needs input_ids for the token-identity "
            "component; use an input_ids-based dataset (e.g. json)."
        )
    stored_kv = {lt: None for lt in (unique_layer_types or [])}

    def call_layer(j, i, layer, capture_kv=False):
        if ple_enabled:
            ids_cpu = input_ids_list[j]
            orig_embed = model.model.embed_tokens(ids_cpu).to(dev)
            pli = model.model.get_per_layer_inputs(ids_cpu.to(dev), None)
            pli = model.model.project_per_layer_inputs(orig_embed, pli)
            per_layer_input = pli[:, :, i, :]
        else:
            per_layer_input = None

        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        if getattr(attn, "is_kv_shared_layer", False):
            lt = attn.layer_type
            Kt, Vt = stored_kv[lt]
            shared_kv_states = {lt: (Kt[j : j + 1], Vt[j : j + 1])}
        else:
            shared_kv_states = {}

        if unique_layer_types is not None:
            lt = model.model.config.layer_types[i]
            out = layer(
                inps[j].unsqueeze(0),
                attention_mask=causal_mask_map[lt],
                position_ids=position_ids,
                position_embeddings=position_embeddings_map[lt],
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
            )[0]
        elif hasattr(model.model, "rotary_emb"):
            pe = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
            out = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=pe,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
            )[0]
        else:
            out = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
            )[0]

        if capture_kv and getattr(attn, "store_full_length_kv", False):
            lt = attn.layer_type
            Kj, Vj = shared_kv_states[lt]  # (1, num_kv_heads, seq, head_dim) on dev
            if stored_kv[lt] is None:
                stored_kv[lt] = (
                    torch.empty(
                        args.nsamples, *Kj.shape[1:], dtype=Kj.dtype, device=Kj.device
                    ),
                    torch.empty(
                        args.nsamples, *Vj.shape[1:], dtype=Vj.dtype, device=Vj.device
                    ),
                )
            stored_kv[lt][0][j].copy_(Kj[0])
            stored_kv[lt][1][j].copy_(Vj[0])
        return out

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = sort_sequent(full.keys())
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            weight_algo = {}
            for name in subset:
                layer_name_full = "model.layers.%d.%s" % (i, name)
                weight_config = _get_weight_config(args.quant_config, layer_name_full)
                if weight_config is None:
                    continue
                weight_algo[name] = WeightQuantHessian(subset[name])
                weight_algo[name].quantizer = Quantizer()
                wbits = weight_config["bit"]
                weight_algo[name].quantizer.configure(
                    wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    weight_algo[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                if name not in weight_algo:
                    continue
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = call_layer(j, i, layer)
            for h in handles:
                h.remove()

            for name in subset:
                if name not in weight_algo:
                    continue
                groupsize = args.quant_config.get("model.layers.%d.%s" % (i, name))[
                    "weight"
                ]["group_size"]
                blocksize = args.quant_config.get("model.layers.%d.%s" % (i, name))[
                    "weight"
                ].get("blocksize", 128)
                Logger.info(
                    f"layer {i}: {name}, Quantizing ... gs:{groupsize}, blocksize:{blocksize}"
                )
                weight_algo[name].fasterquant(
                    blocksize=blocksize,
                    percdamp=args.percdamp,
                    groupsize=groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = weight_algo[
                    name
                ].quantizer
                weight_algo[name].free()

        for j in range(args.nsamples):
            outs[j] = call_layer(j, i, layer, capture_kv=True)
        layers[i] = layer.cpu()
        del layer
        empty_device_cache()
        inps, outs = outs, inps
    stored_kv = {}
    causal_mask_map = {}
    position_embeddings_map = {}
    empty_device_cache()

    ple_proj_key = "model.per_layer_model_projection"
    ple_weight_config = _get_weight_config(args.quant_config, ple_proj_key)
    if ple_weight_config is not None:
        ple_proj_layer = model.model.per_layer_model_projection.to(dev)
        ple_proj_quant = WeightQuantHessian(ple_proj_layer)
        ple_proj_quant.quantizer = Quantizer()
        wbits = ple_weight_config["bit"]
        groupsize = ple_weight_config["group_size"]
        blocksize = ple_weight_config.get("blocksize", 128)
        ple_proj_quant.quantizer.configure(
            wbits, perchannel=True, sym=args.sym, mse=False
        )
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if input_ids_list is not None:
            for ids_cpu in input_ids_list:
                inputs_embeds = model.model.embed_tokens(ids_cpu.to(dev))
                out = ple_proj_layer(inputs_embeds)
                ple_proj_quant.add_batch(inputs_embeds.data, out.data)
                del inputs_embeds, out
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        Logger.info(
            f"model.per_layer_model_projection, Quantizing ... gs:{groupsize}, blocksize:{blocksize}"
        )
        ple_proj_quant.fasterquant(
            blocksize=blocksize,
            percdamp=args.percdamp,
            groupsize=groupsize,
            actorder=args.act_order,
            static_groups=args.static_groups,
        )
        quantizers[ple_proj_key] = ple_proj_quant.quantizer
        ple_proj_quant.free()
        model.model.per_layer_model_projection = ple_proj_layer.cpu()
        del ple_proj_layer
        empty_device_cache()

    lm_head_weight_config = _get_weight_config(args.quant_config, "lm_head")
    if lm_head_weight_config is not None:
        layer = model.lm_head.to(dev)
        lm_head_q_layer = WeightQuantHessian(layer)
        lm_head_q_layer.quantizer = Quantizer()
        wbits = lm_head_weight_config["bit"]
        groupsize = lm_head_weight_config["group_size"]
        blocksize = lm_head_weight_config.get("blocksize", 128)
        lm_head_q_layer.quantizer.configure(
            wbits, perchannel=True, sym=args.sym, mse=False
        )
        for j in range(args.nsamples):
            lm_head_q_layer.add_batch(inps[j].data, outs[j].data)
        del inps, outs
        empty_device_cache()
        Logger.info(f"lm_head, Quantizing ... gs:{groupsize}, blocksize:{blocksize}")
        lm_head_q_layer.fasterquant(
            blocksize=blocksize,
            percdamp=args.percdamp,
            groupsize=groupsize,
            actorder=args.act_order,
            static_groups=args.static_groups,
        )
        quantizers["lm_head"] = lm_head_q_layer.quantizer
        lm_head_q_layer.free()
        model.lm_head = layer.cpu()
        del layer
        empty_device_cache()
    model.config.use_cache = use_cache

    return quantizers


class QuantArgs:
    def __init__(self, nsamples=256):
        self.seed = 1234
        self.nsamples = nsamples
        self.percdamp = 0.01
        self.nearest = False
        self.new_eval = False
        self.act_order = False
        self.static_groups = False
        self.true_sequential = False
        self.sym = True
        self.model = None
        self.quant_config = None


args = QuantArgs()


def weight_quant_loop(
    model_name_or_path,
    data_set,
    nsamples=256,
    seed=1234,
    seq_len=2048,
    quant_config=None,
    torch_dtype=torch.float32,
):
    args.nsamples = nsamples
    args.model = model_name_or_path
    args.seed = seed
    model = get_llama(model_name_or_path, seq_len, torch_dtype)
    args.quant_config = paser_quant_config(model, quant_config)
    model.eval()

    dataloader, _ = get_loaders(
        data_set, nsamples=nsamples, seed=seed, model=model_name_or_path, seqlen=seq_len
    )
    if not args.nearest:
        quantizers = llama_sequential(model, dataloader, DEV)
    return quantizers, model.state_dict()
