#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2026. All rights reserved
import os
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
    "Quant_act_weight_eco": {
        "weight": {
            "blocksize": 128,
            "group_size": 256,
            "bit": 4,
            "per_channel": True,
            "weight_algo": "omni_group_min_max",
        },
        "input": {
            "bit": 8,
            "per_channel": False,
            "input_algo": "dyn_s8",
            "unsigned_quant": False
        },
    },
    "Quant_Embed_MinMax" : {
        "target_op" : "embedding_min_max",
        "weight" : {
            "bit"           : 8,
            "per_channel"   : True,
            "weight_algo"   : "min_max",
            "fp16"          : True,
        }
    },
    "Quant_Gemma4_Embed" : {
        "weight" : {
            "bit"           : 8,
            "per_channel"   : True,
            "weight_algo"   : "min_max",
            "fp16"          : True,
            "tie_lmhead"    : False,
        }
    },
    "Quant_Dummy": {"quant_info": "default version"},
    "Quant_lm_head" : {
            "weight" : {
                "bit"           : 4,
                "group_size"    : 256,
                "blocksize"     : 128,
                "per_channel"   : True,
                "weight_algo"   : "omni_group_min_max",
            },
            "input" : {
                "bit"           : 16,
                "per_channel"   : False,
                "input_algo"    : "min_max",
            }
        },
}


def get_optimize_cfg(quant_strategy):
    group_size = None
    if ";" in quant_strategy:
        quant_strategy, group_size = quant_strategy.split(";")

    if quant_strategy not in strategy_dict:
        raise Exception(f"not support {quant_strategy}")
    cfg = strategy_dict.get(quant_strategy, dict())

    if group_size:
        cfg["weight"]["group_size"] = int(group_size)
    return cfg
