# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2027. All rights reserved.
import os
from .common_tools.register import Register

Quantizers = Register("Quantizers")
OPT_OPS = Register("OptimizedOps")


def init_builder(quantizer_path=None):
    """init all essential funcs"""
    root = os.path.dirname(__file__)
    register_libs = [
        "modules",
        "quantizer",
    ]
    for lib in register_libs:
        Register.file_import(os.path.join(root, lib), recursive=True)
    if quantizer_path is not None:
        if not os.path.isabs(quantizer_path):
            raise Exception("Please input an absolute path.")
        for quantizer_file in os.listdir(quantizer_path):
            quantizer_file_local = os.path.join(quantizer_path, quantizer_file)
            if quantizer_file_local[-3:] in {".py", ".so"}:
                Register.file_import(quantizer_file_local, recursive=True)
