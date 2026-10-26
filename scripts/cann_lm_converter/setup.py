#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py — cann-lm whl 兼容层（与 pyproject.toml 的 PEP 621 metadata 同步）。
#
# 背景：pyproject.toml 用 [project] 表（PEP 621），需要 setuptools>=61 才能识别。
# 在"装不到 build-system requires 的干净离线环境"（如 build isolation 拉不到
# setuptools>=68 时回退旧版），setuptools<61 读不到 [project] 元数据 → 打出
# UNKNOWN-0.0.0 空包（961B）。本 setup.py 用 setup() 硬编码同一份 metadata，
# 使新旧 setuptools 都能打出正确的 cann_lm-0.1.0.whl。
#
# 同步注意：改 pyproject.toml 的 packages / package-dir / scripts 时，必须同步本文件。

from setuptools import setup

setup(
    name="cann-lm",
    version="0.1.0",
    description="CANN-LM converter: HF model -> epair (gemma4/qwen3.5) + models",
    python_requires=">=3.10",
    # 注: torchair + torch-delegate 不在 PyPI, 需源码/whl 预装 (见 AGENTS.md converter env)
    install_requires=[
        "torch>=2.9",
        "transformers",
        "safetensors",
        "pyyaml",
        "packaging",
        "datasets",
    ],
    entry_points={
        "console_scripts": [
            "cann-lm-converter=cann_lm.converter.cann_lm_converter:main",
        ],
    },
    # 与 pyproject [tool.setuptools] 一致：cann_lm 是 PEP 420 namespace 包
    # （cann-lm 装 converter+models，cann-lm-engine 单独装 engine），本 distribution
    # 不含 cann_lm.engine。显式 packages（find 在 ../../cann_lm 会带上 engine，故显式排除）。
    packages=[
        "cann_lm.converter",
        "cann_lm.converter.export",
        "cann_lm.converter.export.ge_converters",
        "cann_lm.converter.export.quant_swap",
        "cann_lm.converter.export.weight_format",
        "cann_lm.converter.quant",
        "cann_lm.models",
        "cann_lm.models.gemma4",
        "cann_lm.models.qwen3_5",
        "cann_lm.models.fused_ops",
        # dopt_lm (vendored GPTQ 框架, 顶层包, 源码在 quant/dopt_lm)
        "dopt_lm",
        "dopt_lm.common_tools",
        "dopt_lm.modules",
        "dopt_lm.opt_funcs",
        "dopt_lm.opt_funcs.eco_stage",
        "dopt_lm.opt_funcs.weight_quant_ptq",
        "dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian",
        "dopt_lm.quantizer",
    ],
    package_dir={
        "cann_lm": "../../cann_lm",
        "dopt_lm": "../../cann_lm/converter/quant/dopt_lm",
    },
    package_data={
        "cann_lm.converter.quant": ["*.yaml", "*.json", "calibrate_sets/*.jsonl"],
        "cann_lm.converter.export": ["*.json"],
    },
    zip_safe=False,
)
