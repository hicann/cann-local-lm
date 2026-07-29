#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2027. All rights reserved
import logging

import numpy as np

import torch
from torch.nn.modules.conv import _ConvNd
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance, cosine_similarity

MIN_SCALE_VALUE = 1.0e-5
NF2 = [-1, -0.39491748809814453, 0.1, 1.0]
NF3 = [
    -1,
    -0.5350227355957031,
    -0.2469314038753510,
    0,
    0.1833375245332718,
    0.3819939494132996,
    0.6229856610298157,
    1,
]
NF4 = [
    -1,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1,
]
nf_dict = {
    2: torch.round(torch.tensor(NF2, dtype=torch.float32) * 127),
    3: torch.round(torch.tensor(NF3, dtype=torch.float32) * 127),
    4: torch.round(torch.tensor(NF4, dtype=torch.float32) * 127),
}


def get_clip_range(bit, unsigned_quant, keep_neg_max=True):
    if bit == 2.5:
        if unsigned_quant:
            raise ValueError("2.5bit only support sym mode")
        neg, pos = torch.tensor(-2.0).to(bit), torch.tensor(2.0).to(bit)
        return neg, pos

    if unsigned_quant:
        negative_clip = 0 * bit
        positive_clip = 2**bit - 1
    else:
        negative_clip = -(2 ** (bit - 1))
        positive_clip = 2 ** (bit - 1) - 1
        if not keep_neg_max:
            negative_clip = -positive_clip
    return negative_clip, positive_clip


class RoundWithGrad(torch.autograd.Function):
    """Round With Grad"""

    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor.round()

    @staticmethod
    def backward(ctx, grad):
        return grad


def get_min_max(x_orig, mean_topk=1):
    x = x_orig.detach()
    if mean_topk == 1:
        min_val = torch.min(x).to(x.device)
        max_val = torch.max(x).to(x.device)
    elif mean_topk > 1:
        max_val = x.flatten().topk(mean_topk)[0].mean()
        min_val = -((-x).flatten().topk(mean_topk)[0].mean())
    else:
        raise NotImplementedError
    return min_val, max_val


def get_abs_max_per_channel(x, ch_axis):
    x = torch.transpose(torch.abs(x.detach()), 0, ch_axis)
    x = x.reshape(x.shape[0], -1)
    max_x = torch.max(x.detach(), dim=1).values

    return max_x


def get_min_max_per_channel(x, ch_axis):
    x = torch.transpose(x.detach(), 0, ch_axis)
    x = x.reshape(x.shape[0], -1)
    max_x = torch.max(x.detach(), dim=1).values.to(x.device)
    min_x = torch.min(x.detach(), dim=1).values.to(x.device)
    return min_x, max_x


def do_quantize(x, scale, offset, negative_clip, positive_clip):
    qx = RoundWithGrad.apply(x / scale)
    qoffset = offset

    qx = torch.clamp(qx + qoffset, negative_clip, positive_clip)
    return qx


def do_antiquantize(qx, scale, offset):
    x = (qx - offset) * scale
    return x


def fakequant_forward(x, scale, offset, negative_clip, positive_clip):
    if scale.dim() != x.dim():
        scale = scale.reshape(-1, *([1] * (x.dim() - 1)))
        offset = offset.reshape(-1, *([1] * (x.dim() - 1)))
    x = RoundWithGrad.apply(x / scale)
    qoffset = RoundWithGrad.apply(offset)
    x = torch.clamp(x + qoffset, negative_clip, positive_clip)
    x = (x - qoffset) * scale
    return x


def get_grad_max(x, scale):
    return torch.clamp(
        (x - x * scale).detach() + x * scale, min=torch.tensor(1.0e-8).to(x.device)
    )


def get_grad_min(x, scale):
    return torch.clamp(
        (x - x * scale).detach() + x * scale, max=torch.tensor(0.0).to(x.device)
    )


def fake_nf(x, scale, labels, *args, **argv):
    qx = RoundWithGrad.apply(x / scale)
    init_shape = list(qx.shape)
    with torch.no_grad():
        rqx = qx.reshape([1] + init_shape).detach()
        rlabels = labels.reshape([-1] + [1] * len(init_shape)).detach()
        diff = torch.abs(rlabels - rqx)
        ids = torch.argmin(diff, dim=0)
        qx_selected = labels[ids]
    qx = (qx_selected - qx).detach() + qx
    x = qx * scale
    return x


if __name__ == "__main__":
    x = torch.randint(-128, 127, size=(2, 6))
    scale = torch.tensor([0.04, 0.05]).reshape(x.shape[0], -1)
    fx = x * scale
    qx = fake_nf(x * scale, scale, 3)
    logging.info(fx)
    logging.info(qx)
