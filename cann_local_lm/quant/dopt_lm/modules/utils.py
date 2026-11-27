# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#
# Stage1 minimal: only get_quantizer_cfg is used by Qlinear / QEmbedding.
import torch


def get_quantizer_cfg(cfg):
    if cfg is None:
        cfg = dict()
    weight_quantizer_cfg = cfg.get("weight")
    input_quantizer_cfg = cfg.get("input")
    output_quantizer_cfg = cfg.get("output")

    return weight_quantizer_cfg, input_quantizer_cfg, output_quantizer_cfg
