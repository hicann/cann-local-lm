#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2024. All rights reserved
import copy
import torch
from dopt_lm.builder import Quantizers


class QClass(torch.nn.Module):
    def __init__(
        self,
        m,
        weight_quantizer_cfg=None,
        input_quantizer_cfg=None,
        output_quantizer_cfg=None,
    ):
        super().__init__()

        self.weight_quantizer = None
        self.do_weight_quant = False
        if weight_quantizer_cfg:
            quantizer_cfg = copy.deepcopy(weight_quantizer_cfg)
            weight_algo = quantizer_cfg.pop("weight_algo")
            self.weight_quantizer = Quantizers.get(weight_algo)(
                m, **quantizer_cfg,
            )
            self.do_weight_quant = self.weight_quantizer is not None

        self.input_quantizer = None
        self.do_input_quant = False
        if input_quantizer_cfg:
            quantizer_cfg = copy.deepcopy(input_quantizer_cfg)
            input_algo = quantizer_cfg.pop("input_algo")
            self.input_quantizer = Quantizers.get(input_algo)(
                m, **quantizer_cfg,
            )
            self.do_input_quant = (
                self.input_quantizer is not None) and input_quantizer_cfg.get("enable", True)

        self.output_quantizer = None
        if output_quantizer_cfg:
            quantizer_cfg = copy.deepcopy(output_quantizer_cfg)
            input_algo = quantizer_cfg.pop("input_algo", "eval_mode")
            if "quant_id" not in quantizer_cfg:
                quantizer_cfg["quant_id"] = -1

            self.output_quantizer = Quantizers.get(input_algo)(
                m, **quantizer_cfg,
            )

        self = self.to(m.weight.device)

    def set_quant_state(self, weight, input):
        self.do_weight_quant = weight
        self.do_input_quant = input

    def set_calibrate_state(self, state):
        if self.do_weight_quant and self.weight_quantizer is not None and hasattr(self.weight_quantizer, "set_calibrate_state"):
            self.weight_quantizer.set_calibrate_state(state)

        if self.do_input_quant and self.input_quantizer is not None and hasattr(self.input_quantizer, "set_calibrate_state"):
            self.input_quantizer.set_calibrate_state(state)

        if self.do_input_quant and self.output_quantizer is not None and hasattr(self.output_quantizer, "set_calibrate_state"):
            self.output_quantizer.set_calibrate_state(state)

    def forward(self, weight, x):
        if self.do_weight_quant and self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)

        if self.do_input_quant and self.input_quantizer is not None:
            x = self.input_quantizer(x)

        return weight, x


    def is_empty(self):
        if self.weight_quantizer is not None:
            return False
        if self.input_quantizer is not None:
            return False
        if self.output_quantizer is not None:
            return False
        return True
