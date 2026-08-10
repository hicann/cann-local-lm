# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2026. All rights reserved.
# 
# Stage1 minimal: QEmbedding keeps only the torch fakequant forward path.
# The NPU embedding simulators (int / fakequant) are removed; npu_simulator
# stays None so set_quant_state's guard still works.
import torch
import torch.nn.functional as F

from dopt_lm.quantizer.qmanager import QClass
from dopt_lm.common_tools.log import Logger as logger
from .utils import get_quantizer_cfg


class QEmbedding(torch.nn.Module): 
    def __init__(
        self, m: torch.nn.Embedding, 
        optimize_cfg, 
        layer_name=None, 
        **argv
    ): 
        super().__init__()
        self.op_type = "Embedding"
        weight_quantizer_cfg, input_quantizer_cfg, output_quantizer_cfg = get_quantizer_cfg(optimize_cfg)
        input_quantizer_cfg = dict()

        if weight_quantizer_cfg: 
            if not weight_quantizer_cfg.get("per_channel", False): 
                weight_quantizer_cfg["per_channel"] = True
                logger.warning("Warning: embedding do per-channel quantization, overwrite per_channel to True")

            self.quant_op = QClass(
                m, 
                weight_quantizer_cfg, 
                input_quantizer_cfg, 
            )
        self.tie_lmhead = weight_quantizer_cfg.get('tie_lmhead', False)
        if self.tie_lmhead: 
            self.weight = m.weight
        else: 
            self.register_buffer("weight", m.weight.clone())
        
        self.layer_name = layer_name
        self.mode = None
        self.padding_idx = m.padding_idx
        self.max_norm = m.max_norm
        self.norm_type = m.norm_type
        self.scale_grad_by_freq = m.scale_grad_by_freq
        self.sparse = m.sparse

        self.npu_simulator = None

    def forward(self, x): 
        weight = self.weight
        if getattr(self, "quant_op", None): 
            weight, x = self.quant_op(weight, x)

        out = F.embedding(
            x, weight, 
            self.padding_idx, 
            self.max_norm, 
            self.norm_type, 
            self.scale_grad_by_freq, 
            self.sparse, 
        )
        return out


class QGemma4ScaledEmbedding(torch.nn.Module):
    def __init__(
        self, m,
        optimize_cfg,
        layer_name=None,
        **argv
    ):
        super().__init__()
        self.op_type = "Embedding"
        weight_quantizer_cfg, input_quantizer_cfg, output_quantizer_cfg = get_quantizer_cfg(optimize_cfg)
        input_quantizer_cfg = dict()

        if weight_quantizer_cfg:
            if not weight_quantizer_cfg.get("per_channel", False):
                weight_quantizer_cfg["per_channel"] = True
                logger.warning("Warning: embedding do per-channel quantization, overwrite per_channel to True")
            self.quant_op = QClass(
                m,
                weight_quantizer_cfg,
                input_quantizer_cfg,
            )
        self.tie_lmhead = weight_quantizer_cfg.get('tie_lmhead', False)
        if self.tie_lmhead:
            self.weight = m.weight
        else:
            self.register_buffer("weight", m.weight.clone())

        self.scalar_embed_scale = getattr(m, "scalar_embed_scale", 1.0)
        if hasattr(m, "embed_scale"):
            self.register_buffer("embed_scale", m.embed_scale.detach().clone(), persistent=False)
        else:
            self.register_buffer("embed_scale", torch.tensor(float(self.scalar_embed_scale)), persistent=False)

        self.layer_name = layer_name
        self.mode = None
        self.padding_idx = m.padding_idx
        self.max_norm = m.max_norm
        self.norm_type = m.norm_type
        self.scale_grad_by_freq = m.scale_grad_by_freq
        self.sparse = m.sparse

        self.npu_simulator = None

    def set_quant_state(self, weight, input):
        if self.quant_op is not None:
            self.quant_op.set_quant_state(weight, input)

    def forward(self, x):
        weight = self.weight
        if getattr(self, "quant_op", None):
            weight, x = self.quant_op(weight, x)

        out = F.embedding(
            x, weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        out = out * self.embed_scale.to(weight.dtype)
        return out
    