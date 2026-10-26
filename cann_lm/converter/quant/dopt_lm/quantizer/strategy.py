#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2026. All rights reserved
import copy


strategy_dict = {
    "float": {},
    "Quant_aigc_ptq": {
        "weight": {
            "bit": 8,
            "per_channel": True,
            "weight_algo": "min_max",
        },
        "input": {
            "bit": 8,
            "per_channel": False,
            "input_algo": "ema_min_max",
            "unsigned_quant": True,
        },
    },
    "Quant_aigc_qat": {
        "weight": {
            "bit": 8,
            "per_channel": True,
            "weight_algo": "omni_min_max",
        },
        "input": {
            "bit": 8,
            "per_channel": False,
            "input_algo": "omni_min_max",
            "unsigned_quant": True,
        },
    },
    "Quant_INT8-4_ECO": {
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
    },
    "Quant_Embed_MinMax": {
        "weight": {
            "bit": 8,
            "per_channel": True,
            "weight_algo": "min_max",
            "fp16": True,
            "tie_lmhead": False,
        }
    },
    "Quant_Dummy": {"quant_info": "default version"},
}


def get_optimize_cfg(quant_strategy):
    group_size = None
    if ";" in quant_strategy:
        quant_strategy, group_size = quant_strategy.split(";")

    if quant_strategy not in strategy_dict:
        raise Exception(f"not support {quant_strategy}")
    cfg = copy.deepcopy(strategy_dict.get(quant_strategy, dict()))

    if group_size:
        cfg["weight"]["group_size"] = int(group_size)
    return cfg
