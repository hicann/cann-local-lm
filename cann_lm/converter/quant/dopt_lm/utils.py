# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2026. All rights reserved.
# 
# Stage1 minimal: only replace_module_by_names is needed by optimize_model.
# The input-quantizer sharing helpers (lora / llm) are removed.


def replace_module_by_names(model, modules_to_replace): 
    def helper(child, pre_name=""): 
        for n, c in child.named_children(): 
            is_replaced = False
            full_name = n
            if pre_name: 
                full_name = f"{pre_name}.{n}"
            if full_name in modules_to_replace: 
                child.add_module(n, modules_to_replace.pop(full_name))
                is_replaced = True
            if not is_replaced: 
                helper(c, full_name)

    helper(model)
    return model
