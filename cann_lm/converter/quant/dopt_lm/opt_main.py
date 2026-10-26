# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025-2035. All rights reserved.
#
# Stage1 minimal entry. Only the stage1 (GPTQ weight quant) branch is wired;
# stage2 / stage3 are intentionally not imported.
import os
import argparse
from dopt_lm.common_tools.log import Logger
from dopt_lm.opt_funcs.eco_stage.train import generate_config
from dopt_lm.opt_funcs.eco_stage.train import weight_quant_pipline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="model name")
    parser.add_argument("--dopt-config", type=str, default=None)
    parser.add_argument("--optimize-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="output_dir")
    parser.add_argument("--hf_type_save", action="store_true", default=False)
    args = parser.parse_args()
    return args


def generate_quant_config(model_name_or_path, config_path):
    generate_config(model_name_or_path, config_path)


def llm_quant_pipline(**argv):
    weight_quant_pipline(**argv)


if __name__ == "__main__":
    args = parse_args()
    ### step1 generate dopt_quant_config
    if not os.path.exists(args.dopt_config):
        generate_quant_config(args.model_path, args.dopt_config)
        Logger.info("generate plugin quang config please set quant strategy firstly")
        exit()
    optimize_info = {
        "model_path": args.model_path,
        "config_yaml": args.optimize_config,
        "dopt_config": args.dopt_config,
        "output_dir": args.output_dir,
        "hf_type_save": args.hf_type_save,
    }
    ### step2 run llm model Quantification optimization process
    llm_quant_pipline(**optimize_info)
