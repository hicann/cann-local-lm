#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2027. All rights reserved.
import os
import json
import copy

import torch

from .common_tools.log import Logger as logger
from .modules.qlinear import QLinear
from .modules.qembed import QEmbedding
from .quantizer.strategy import get_optimize_cfg

from .builder import init_builder
from .utils import replace_module_by_names


init_builder()


visible_type_dict = {
    torch.nn.Embedding: QEmbedding,
    "embedding_min_max": QEmbedding,
    torch.nn.Linear: QLinear,
    "Linear": QLinear,
}

invisible_type_dict = {}
if len(set(visible_type_dict.keys()) & set(invisible_type_dict.keys())) != 0:
    duplicate_keys = set(visible_type_dict.keys()) & set(invisible_type_dict.keys())
    raise RuntimeError(
        f"Conflict detected: The following types are both visible and invisible: {duplicate_keys}"
    )
type_dict = {**visible_type_dict, **invisible_type_dict}


def get_target_op_cls(m, parsed_config=None, all_visible=True):
    if parsed_config and parsed_config.get("target_op", None):
        return type_dict[parsed_config["target_op"]]

    if type(m) in visible_type_dict:
        return visible_type_dict[type(m)]
    if m.__class__.__name__ in visible_type_dict:
        return visible_type_dict[m.__class__.__name__]

    if type(m) in type_dict:
        return type_dict[type(m)]
    if m.__class__.__name__ in type_dict:
        return type_dict[m.__class__.__name__]
    return None


def generate_config_file(model, path):
    if os.path.exists(path):
        raise Exception(f"{path} exist")
    layer_dict = dict()
    for name, m in model.named_modules():
        if get_target_op_cls(m, all_visible=False) is None:
            logger.debug(f"{name} type {type(m)} not suitable")
            continue
        layer_dict[name] = {
            "type": str(type(m)),
            "quant_strategy": "float",
        }
    config_data = dict()
    config_data["layer_strategy"] = layer_dict
    with open(path, "w") as json_file:
        json.dump(config_data, json_file, indent=4)


def trim_dict(adict):
    if not isinstance(adict, dict):
        return adict

    keys = copy.deepcopy(adict)
    for k in keys:
        trimed_dict = trim_dict(adict[k])
        if trimed_dict is None:
            adict.pop(k)
    if len(adict) == 0:
        return None
    return adict


def parse_layer_config(layer_config):
    quant_strategy = layer_config["quant_strategy"]
    parsed_config = copy.deepcopy(get_optimize_cfg(quant_strategy))
    skipped_keys = {"lora", "quant_strategy"}
    for k in layer_config:
        if k in skipped_keys:
            continue
        if isinstance(layer_config[k], dict):
            if k not in parsed_config:
                parsed_config[k] = dict()
            parsed_config[k].update(layer_config.get(k, dict()))
            continue
        if k not in parsed_config:
            parsed_config[k] = layer_config[k]
    if layer_config.get("lora", dict()):
        parsed_config["lora"] = copy.deepcopy(layer_config["lora"])
        if "optimize_config" not in parsed_config["lora"]:
            parsed_config["lora"]["optimize_config"] = dict()
        parsed_lora_config = copy.deepcopy(layer_config["lora"])
        if "quant_strategy" in parsed_lora_config:
            parsed_lora_config = parse_layer_config(parsed_lora_config)
        parsed_config["lora"]["optimize_config"].update(parsed_lora_config)
    return parsed_config


def get_optimized_op(m, layer_config, *args, layer_name=None, **argv):
    parsed_config = parse_layer_config(layer_config)
    if "target_op" in layer_config:
        parsed_config["target_op"] = layer_config["target_op"]
    target_op = get_target_op_cls(m, parsed_config)
    if target_op is None:
        raise Exception(f"bad input type {m.__class__.__name__}, please register first")

    parsed_config = trim_dict(parsed_config)
    if parsed_config is None:
        return m
    qm = target_op(m, parsed_config, layer_name=layer_config.pop("layer_name", ""))
    return qm


def paser_quant_config(model, config_file):
    """Parse quant config for GPTQ weight quantization.

    Loads layer_strategy from config file and returns it as-is.
    The layer config dict will be accessed by ptq_loop.py using keys like
    'model.layers.{i}.{name}'.
    """
    if config_file is None:
        logger.info("WARNING!!!!!!!!!!!! NO QUANT CONFIG")
        return model
    with open(config_file, "r") as f:
        config = json.load(f)
    layer_config = config.get("layer_strategy", dict())

    default_support_types = type_dict
    for name, m in model.named_modules():
        if name not in layer_config:
            continue
        logger.info(f"{name} using strategy: {layer_config[name]['quant_strategy']}")
        layer_config[name]["layer_name"] = name
        layer_config[name] = parse_layer_config(layer_config[name])

    return layer_config


def optimize_model(model, config_file):
    """optimize model interface

    Args:
        model (torch.nn.Module): torch model
        config_file (file, needs to load by config load)
    """
    if config_file is None:
        logger.warning("WARNING!!!!!!!!!!!! NO QUANT CONFIG")
        return model

    replace_modules = dict()
    with open(config_file, "r") as f:
        config = json.load(f)
    layer_config = config.get("layer_strategy", dict())

    for name, m in model.named_modules():
        if name in layer_config:
            logger.debug(
                f"{name} using strategy: ", layer_config[name]["quant_strategy"]
            )
            layer_config[name]["layer_name"] = name
            layer_config["m_type"] = m.__class__.__name__
            replace_modules[name] = get_optimized_op(
                m, layer_config[name], layer_name=name
            )

    if replace_modules:
        model = replace_module_by_names(model, replace_modules)

    # not support PP
    return model


def set_calibrate_state(model, calibrate_state):
    for name, m in model.named_modules():
        if hasattr(m, "quant_op") and m.quant_op is not None:
            m.quant_op.set_calibrate_state(calibrate_state)


def set_quant_state(model, weight_state, input_state):
    for name, m in model.named_modules():
        if hasattr(m, "quant_op") and m.quant_op is not None:
            m.quant_op.set_quant_state(weight_state, input_state)
            if (
                hasattr(m, "npu_simulator")
                and m.npu_simulator is not None
                and hasattr(m.npu_simulator, "set_quant_state")
            ):
                m.npu_simulator.set_quant_state(weight_state, input_state)
    return model
