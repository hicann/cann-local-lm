# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
import os
import math
import time

import torch
import torch.nn as nn
import transformers

from dopt_lm.common_tools.log import Logger
from .quant import quantize


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class WeightQuantHessian:
    """
        GPTQ
    """
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        weight_matrix = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight_matrix = weight_matrix.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight_matrix = weight_matrix.t()
        self.rows = weight_matrix.shape[0]
        self.columns = weight_matrix.shape[1]
        self.hessian_matrix = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.hessian_matrix *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hessian_matrix += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        weight_matrix = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight_matrix = weight_matrix.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight_matrix = weight_matrix.t()
        weight_matrix = weight_matrix.float()

        tick = time.time()
        group_scale = []
        group_zero = []

        if not self.quantizer.ready():
            self.quantizer.find_params(weight_matrix, weight=True)

        hessian_mat = self.hessian_matrix
        del self.hessian_matrix
        dead = torch.diag(hessian_mat) == 0
        hessian_mat[dead, dead] = 1
        weight_matrix[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(weight_matrix[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(hessian_mat), descending=True)
            weight_matrix = weight_matrix[:, perm]
            hessian_mat = hessian_mat[perm][:, perm]
            invperm = torch.argsort(perm)

        total_losses = torch.zeros_like(weight_matrix)
        quantized_w_matrix = torch.zeros_like(weight_matrix)

        damp = percdamp * torch.mean(torch.diag(hessian_mat))
        diag = torch.arange(self.columns, device=self.dev)
        hessian_mat[diag, diag] += damp
        if os.getenv("DEVICE", "cuda") == 'npu':
            hessian_cpu = hessian_mat.cpu()
            del hessian_mat
            hessian_cpu = torch.linalg.cholesky(hessian_cpu)
            hessian_cpu = torch.cholesky_inverse(hessian_cpu)
            hessian_cpu = torch.linalg.cholesky(hessian_cpu, upper=True)
            hinv = hessian_cpu.to(self.dev)
            del hessian_cpu
        else:
            hessian_mat = torch.linalg.cholesky(hessian_mat)
            hessian_mat = torch.cholesky_inverse(hessian_mat)
            hessian_mat = torch.linalg.cholesky(hessian_mat, upper=True)
            hinv = hessian_mat

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            weight_matrix1 = weight_matrix[:, i1:i2].clone()
            quantized_w_matrix_block = torch.zeros_like(weight_matrix1)
            loss_error = torch.zeros_like(weight_matrix1)
            losses1 = torch.zeros_like(weight_matrix1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = weight_matrix1[:, i]
                d = hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(weight_matrix[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            group_scale.append(self.quantizer.scale)
                            group_zero.append(self.quantizer.zero)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                quantized_w_matrix_block[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                weight_matrix1[:, i:] -= err1.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                loss_error[:, i] = err1

            quantized_w_matrix[:, i1:i2] = quantized_w_matrix_block
            total_losses[:, i1:i2] = losses1 / 2

            weight_matrix[:, i2:] -= loss_error.matmul(hinv[i1:i2, i2:])

        if os.getenv("DEVICE", "cuda") == 'npu':
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
        Logger.info('time %.2f' % (time.time() - tick))
        Logger.info('quant loss: %.6f' % (torch.sum(total_losses).item()))

        self.quantizer.group_scale = torch.cat(group_scale, dim=1)
        self.quantizer.group_zero = torch.cat(group_zero, dim=1)
        if actorder:
            quantized_w_matrix = quantized_w_matrix[:, invperm]
            self.quantizer.group_scale = self.quantizer.group_scale.repeat(1, groupsize)
            self.quantizer.group_zero = self.quantizer.group_zero.repeat(1, groupsize)
            self.quantizer.group_scale = self.quantizer.group_scale[:, invperm].view(self.rows, -1, groupsize).mean(dim=2)
            self.quantizer.group_zero = self.quantizer.group_zero[:, invperm].view(self.rows, -1, groupsize).mean(dim=2)

        self.quantizer.group_scale = self.quantizer.group_scale.flatten().unsqueeze(1).cpu()
        self.quantizer.group_zero = self.quantizer.group_zero.flatten().unsqueeze(1).cpu()

        if isinstance(self.layer, transformers.Conv1D):
            quantized_w_matrix = quantized_w_matrix.t()
        self.layer.weight.data = quantized_w_matrix.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype).to(self.dev)

    def free(self):
        self.hessian_matrix = None
        if os.getenv("DEVICE", "cuda") == 'npu':
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
