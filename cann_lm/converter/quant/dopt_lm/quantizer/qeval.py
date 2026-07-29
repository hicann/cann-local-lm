#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2022. All rights reserved
import os
import torch
from dopt_lm.builder import Quantizers
from dopt_lm.common_tools.log import Logger as logger
from .qutils import MIN_SCALE_VALUE, get_clip_range


class QuantizerFuncIntOffset(torch.autograd.Function):
    """Round With Grad"""

    @staticmethod
    def forward(ctx, input_tensor, s_tensor, offset_tensor, negative_clip, positive_clip, min_scale_value=1.e-5, convert_to_fp32=True):
        dtype = input_tensor.dtype
        input_tensor = input_tensor.to(torch.float32)
        s_tensor = s_tensor.clamp(min_scale_value)
        rounded = torch.round(input_tensor / s_tensor + offset_tensor)
        clamped = torch.clamp(rounded, negative_clip, positive_clip)
        out = (clamped - offset_tensor) * s_tensor
        return out.to(dtype)

    @staticmethod
    def backward(ctx, grad):
        # simple STE, because we don't use it for QAT
        return grad.clone(), None, None, None, None, None


class QuantizerFuncFpOffset(torch.autograd.Function):
    """Round With Grad"""

    @staticmethod
    def forward(ctx, input_tensor, s_tensor, offset_tensor, negative_clip, positive_clip, convert_to_fp32=True):
        dtype = input_tensor.dtype
        input_tensor = input_tensor.to(torch.float32)
        s_tensor = s_tensor.clamp(1.e-5)
        rounded = torch.round((input_tensor + offset_tensor) / s_tensor)
        clamped = torch.clamp(rounded, negative_clip, positive_clip)
        out = clamped * s_tensor - offset_tensor
        return out.to(dtype)

    @staticmethod
    def backward(ctx, grad):
        # simple STE, because we don't use it for QAT
        return grad.clone(), None, None, None, None


class DynIntSymQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        positive_clip = torch.tensor(127.).to(x.device).to(torch.float16)

        x_fp16 = x.to(torch.float16)
        s_fp16 = torch.amax(x.abs(), dim=-1, keepdim=True)  # b, token
        s_fp16 = s_fp16.clamp(min=2.e-5)
        s_inv_fp16 = 1. / s_fp16
        qx_normalized = x_fp16 * s_inv_fp16  # [-1, 1]
        qx_fp16 = qx_normalized * positive_clip
        rounded = torch.round(qx_fp16)
        clamped = torch.clamp(rounded, - positive_clip - 1, positive_clip)
        normalized = clamped / positive_clip
        return normalized.to(x), 1. / s_inv_fp16

    @staticmethod
    def backward(ctx, grad, grad_s):
        return grad.clone(), None, None, None, None


@Quantizers.register_func("dyn_s8")
class DynEvalGridS8(torch.nn.Module):
    def __init__(self, *args, **argv):
        super().__init__()
        self.quant_dtype = "dyn_s8"
        self.do_calibrate = True

        self.register_buffer('s', torch.tensor(1. / 127.).to(torch.float16))
        self.register_buffer('offset', torch.tensor(0.).to(torch.float16))
        self.register_buffer('bit', torch.tensor(8.))
        self.register_buffer("unsigned_quant", torch.tensor(0.))

    @classmethod
    def init_from_quantizer(cls, quantizer):
        obj = cls()
        obj = obj.to(quantizer.s.device)
        return obj

    def forward(self, x):
        out, sa = DynIntSymQuantizer.apply(x)
        return out, sa
