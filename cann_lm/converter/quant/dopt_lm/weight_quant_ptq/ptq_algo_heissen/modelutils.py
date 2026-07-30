# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
import os
import torch
import torch.nn as nn


device_type = os.getenv("DEVICE", "cuda")
DEV = torch.device(f'{device_type}')


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
