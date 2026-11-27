# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2024. All rights reserved.
#
# Stage1 minimal: QLinear keeps only the fakequant torch forward path used to
# load the GPTQ-quantized weights and save the state_dict. LoRA, the NPU
# simulator (fakequant/hardware), profiling and CustomAdd are removed.
import torch
import torch.nn.functional as F

from dopt_lm.quantizer.qmanager import QClass
from dopt_lm.common_tools.log import Logger as logger
from .utils import get_quantizer_cfg


class QLinear(torch.nn.Module):
    def __init__(self, m, optimize_cfg, layer_name=None, **argv):
        super().__init__()
        weight_quantizer_cfg, input_quantizer_cfg, output_quantizer_cfg = (
            get_quantizer_cfg(optimize_cfg)
        )

        self.in_features = m.in_features
        self.out_features = m.out_features
        self.weight = m.weight
        self.bias = m.bias
        self.layer_name = layer_name

        if output_quantizer_cfg:
            if "quant_id" not in output_quantizer_cfg:
                output_quantizer_cfg["quant_id"] = -1
            if "channel_id" not in output_quantizer_cfg:
                output_quantizer_cfg["channel_id"] = 0
        self.quant_op = QClass(
            m,
            weight_quantizer_cfg,
            input_quantizer_cfg,
            output_quantizer_cfg,
        )
        self.quant_op = self.quant_op.to(self.weight.device)

    def set_quant_state(self, weight, input):
        if self.quant_op is not None:
            self.quant_op.set_quant_state(weight, input)

    def forward(self, x):
        weight = self.weight
        dyn_flag = (
            getattr(self.quant_op.input_quantizer, "quant_dtype", "").startswith("dyn")
            and self.quant_op.do_input_quant
        )
        if self.quant_op is not None:
            weight, x = self.quant_op(weight, x)
            if dyn_flag:
                x, s_denormalized = x

        out = F.linear(x, weight.to(x), None)
        if self.quant_op.do_input_quant and self.quant_op.output_quantizer is not None:
            out = self.quant_op.output_quantizer(out)
        if dyn_flag:  # use after lora add
            out = out * s_denormalized
        if self.bias is not None:
            out = out + self.bias
        return out
