# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
import os
import torch
import torch.nn as nn


SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "npu"}
DEVICE_TYPE = os.getenv("DEVICE", "cuda").strip().lower()
if DEVICE_TYPE not in SUPPORTED_DEVICE_TYPES:
    supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
    raise ValueError(f"Unsupported DEVICE={DEVICE_TYPE!r}; expected one of: {supported}")
if DEVICE_TYPE == "npu":
    import torch_npu  # noqa: F401 - registers the torch.npu backend

DEV = torch.device(DEVICE_TYPE)


def empty_device_cache():
    if DEVICE_TYPE == "npu":
        torch.npu.empty_cache()
    elif DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()


def synchronize_device():
    if DEVICE_TYPE == "npu":
        torch.npu.synchronize()
    elif DEVICE_TYPE == "cuda":
        torch.cuda.synchronize()


def find_layers(module, layers=None, name=''): 
    if layers is None: 
        layers = [nn.Conv2d, nn.Linear]
    if type(module) in layers: 
        return {name: module}
    res = {}
    for name1, child in module.named_children(): 
        res.update(find_layers(
                child, layers=layers,
                name=name + '.' + name1 if name != '' else name1
            )
        )
    return res
