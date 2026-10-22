#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import torch
from dopt_lm.builder import Quantizers
from .qutils import get_min_max_per_channel, get_min_max
from .qutils import MIN_SCALE_VALUE
from .qutils import get_clip_range
from .qeval import QuantizerFuncFpOffset, QuantizerFuncIntOffset


class LSQAutograd(torch.autograd.Function):
    @staticmethod
    def forward(x, step, negative_clip, positive_clip):
        not_perform_clamp = negative_clip is None and positive_clip is None
        if not (x.requires_grad or step.requires_grad):
            if not_perform_clamp:
                return (x.div(step).round_().mul_(step),)
            return (
                x.div(step).clamp_(negative_clip, positive_clip).round_().mul_(step),
            )

        # assume that there will be a backward pass
        x_scaled = x / step
        if not_perform_clamp:
            x_clamped = x_scaled.clone()
        else:
            x_clamped = torch.clamp(x_scaled, negative_clip, positive_clip)
        mask_pass_grad = x_clamped == x_scaled

        x_rounded = x_clamped.round_()

        step_grad_mult = None
        if step.requires_grad:
            step_grad_mult = x_rounded - x_scaled * mask_pass_grad

        return x_rounded * step, mask_pass_grad, step_grad_mult

    @staticmethod
    def setup_context(ctx, inputs, output):
        step = inputs[1]
        ctx.set_materialize_grads(False)
        ctx.step_shape = step.shape
        ctx.save_for_backward(*output[1:])

    @staticmethod
    def backward(ctx, grad_output, *ignore_args):
        if grad_output is None:
            return None, None, None, None

        x_grad = step_grad = None
        mask_pass_grad, step_grad_mult = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            x_grad = mask_pass_grad * grad_output
        if ctx.needs_input_grad[1]:
            reduction_dims = tuple(
                i for i, dim_size in enumerate(ctx.step_shape) if dim_size == 1
            )
            if len(reduction_dims) == 0:
                reduction_dims = None
            step_grad = torch.sum(
                step_grad_mult * grad_output, keepdim=True, dim=reduction_dims
            )

        return x_grad, step_grad, None, None


@Quantizers.register_func("min_max")
@Quantizers.register_func("minmax")
class MinMaxGrid(torch.nn.Module):
    def __init__(
        self,
        father,
        bit,
        per_channel=True,
        unsigned_quant=0,
        quant_id=0,
        channel_id=0,
        reserve_bit=0,
        fp16=False,
        **argv,
    ):
        super().__init__()
        self.reserve_bit = reserve_bit
        self.quant_id = quant_id
        self.channel_id = channel_id
        self.per_channel = per_channel
        self.min_scale_value = MIN_SCALE_VALUE
        self.device = father.weight.device

        self.s_shape = self.get_scale_shape(per_channel, father)
        self.init_param(bit)
        self.unsigned_quant = torch.tensor(unsigned_quant * 1.0)

        self.scale_factor = 1.0

        self.do_calibrate = False
        self.fp16 = fp16

    def set_calibrate_state(self, state):
        self.do_calibrate = state

    def get_scale_shape(self, per_channel, father):
        ### should only support weight quantizer ###
        if per_channel:
            channels = father.weight.shape[self.channel_id]
            s_shape = [channels]
        else:
            s_shape = []
        return s_shape

    def get_clip_range(self):
        if self.bit == 2.5:
            if self.unsigned_quant:
                raise ValueError("2.5bit only support sym mode")
            neg, pos = torch.tensor(-2.0).to(self.bit), torch.tensor(2.0).to(self.bit)
            return neg, pos

        if self.unsigned_quant:
            negative_clip = 0 * self.bit
            positive_clip = 2**self.bit - 1
        else:
            negative_clip = -(2 ** (self.bit - 1))
            positive_clip = 2 ** (self.bit - 1) - 1
        return negative_clip, positive_clip

    def init_param(self, bit):
        self.register_buffer("bit", torch.tensor(bit))
        self.register_buffer("s", torch.ones(self.s_shape) * self.min_scale_value)
        self.register_buffer("offset", torch.zeros(self.s_shape))
        self.register_buffer("unsigned_quant", torch.tensor(0.0))

        self.register_buffer("max", torch.ones(self.s_shape) * self.min_scale_value)
        self.register_buffer("min", torch.zeros(self.s_shape))

    def get_min_max_value(self, x):
        if self.per_channel:
            min_tmp, max_tmp = get_min_max_per_channel(x, self.quant_id)
        else:
            min_tmp, max_tmp = get_min_max(x)
        return min_tmp, max_tmp

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        device = x.device

        if self.do_calibrate:
            negative_clip, positive_clip = get_clip_range(
                self.bit - self.reserve_bit, self.unsigned_quant
            )
            mix, max = self.get_min_max_value(x.detach())
            if not self.unsigned_quant:
                if self.bit == 1:
                    # TODO: fix when bit is 1
                    scale = mix / negative_clip
                else:
                    sa = max / positive_clip
                    si = mix / negative_clip
                    scale = torch.max(sa, si)
            else:
                scale = (max - mix) / (positive_clip - negative_clip)

            scale = scale.to(device).clamp(min=self.min_scale_value)

            if self.unsigned_quant == 0:
                offset = torch.zeros_like(scale).to(device)
            elif self.unsigned_quant == 1:
                offset = torch.round(-mix / scale)
            elif self.unsigned_quant == 2:
                offset = -mix
            elif self.unsigned_quant == 3:
                offset = -(max - mix) / 2.0
            else:
                raise Exception(
                    f"not support unsigned_quant {self.unsigned_quant} model"
                )

            if self.fp16:
                scale = scale.to(torch.float16)
                offset = offset.to(torch.float16)
            scale = scale.clamp(min=self.min_scale_value)

            self.max.data.copy_(max.reshape(self.s_shape))
            self.min.data.copy_(mix.reshape(self.s_shape))
            self.s.data.copy_(scale.reshape(self.s_shape))
            self.offset.data.copy_(offset.reshape(self.s_shape))

        else:
            negative_clip, positive_clip = get_clip_range(self.bit, self.unsigned_quant)
            scale = self.s
            offset = self.offset

        scale = scale.to(torch.float32).reshape(self.s_shape)
        offset = offset.to(torch.float32).reshape(self.s_shape)

        # only cuda support fp16 calculate
        if self.fp16 and x.device != torch.device("cpu"):
            x = x.to(torch.float16)
            scale = scale.to(torch.float16)
            offset = offset.to(torch.float16)
        else:
            scale = scale.to(torch.float32)
            offset = offset.to(torch.float32)

        dim_shape = [1] * len(x.shape)
        dim_shape[self.quant_id] = -1
        scale = scale.reshape(dim_shape)
        offset = offset.reshape(dim_shape)
        if self.unsigned_quant.cpu().data.item() in {0.0, 1.0}:
            x = QuantizerFuncIntOffset.apply(
                x, scale, offset, negative_clip, positive_clip
            )
        elif self.unsigned_quant.cpu().data.item() in {2.0, 3.0}:
            x = QuantizerFuncFpOffset.apply(
                x, scale, offset, negative_clip, positive_clip
            )
        else:
            raise Exception(f"not support unsigned_quant {self.unsigned_quant}")
        x = x.to(dtype)
        return x


@Quantizers.register_func("group_min_max")
@Quantizers.register_func("group_minmax")
class GroupMinMaxGridWeight(MinMaxGrid):
    def __init__(
        self,
        father,
        bit,
        per_channel=True,
        unsigned_quant=False,
        quant_id=0,
        group_size=128,
        **argv,
    ):
        # weight input is: cout X cin
        # change group to: cout X group X group_size
        # if cin % group_size != 0, raise Exception
        _, cin = father.weight.shape
        self.group_size = group_size if group_size > 0 else cin

        super().__init__(father, bit, per_channel, unsigned_quant, quant_id, **argv)
        if unsigned_quant:
            self.unsigned_quant = torch.tensor(2.0)
        else:
            self.unsigned_quant = torch.tensor(0.0)

    def get_scale_shape(self, per_channel, father):
        if not per_channel:
            raise ValueError("only support per_channel True")
        cout, cin = father.weight.shape
        if self.group_size < 1:
            self.group_size = cin
        if cin % self.group_size != 0:
            raise ValueError(
                f"cin ({cin}) must be divisible by group_size ({self.group_size})"
            )
        self.groups = cin // self.group_size
        s_shape = [cout * self.groups]
        return s_shape

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)

        cout, cin = x.shape
        target_shape = [-1, self.group_size]
        x = x.reshape(target_shape)

        x = super().forward(x)

        x = x.reshape([cout, cin])
        x = x.to(dtype)

        return x


@Quantizers.register_func("omni_min_max")
@Quantizers.register_func("omni_minmax")
class OmniMinMaxGrid(MinMaxGrid):
    def __init__(
        self,
        father,
        bit,
        per_channel=True,
        unsigned_quant=False,
        quant_id=0,
        init_value=4.0,
        warmup_iters=-1,
        **argv,
    ):
        self.init_value = init_value
        super().__init__(father, bit, per_channel, unsigned_quant, quant_id, **argv)
        self.w_conv = False
        self.w_deconv = False

    def init_param(self, bit):
        self.register_buffer("bit", torch.tensor(bit))
        self.register_buffer("s", torch.ones(self.s_shape, device=self.device))
        self.register_buffer("offset", torch.zeros(self.s_shape, device=self.device))
        self.register_buffer("unsigned_quant", torch.tensor(0.0))

        self.quant_alpha = torch.nn.Parameter(
            torch.ones(self.s_shape, device=self.device) * self.init_value,
            requires_grad=True,
        )
        self.min_iner = None
        self.max_iner = None

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        device = x.device

        negative_clip, positive_clip = self.get_clip_range()
        if self.do_calibrate:
            if self.w_conv:
                max_value = torch.amax(x.detach(), dim=[1, 2, 3])
                min_value = torch.amin(x.detach(), dim=[1, 2, 3])
            elif self.w_deconv:
                max_value = torch.amax(x.detach(), dim=[0, 2, 3])
                min_value = torch.amin(x.detach(), dim=[0, 2, 3])
            else:
                max_value = torch.amax(x.detach(), dim=1)
                min_value = torch.amin(x.detach(), dim=1)
            if self.max_iner is None:
                self.max_iner = max_value
            if self.min_iner is None:
                self.min_iner = min_value
            max_value = torch.max(max_value, self.max_iner)
            min_value = torch.min(min_value, self.min_iner)
            self.max_iner = max_value
            self.min_iner = min_value
            if self.unsigned_quant == 0.0:
                positive_step = max_value / positive_clip
                negative_step = min_value / negative_clip

                scale = (
                    torch.max(positive_step, negative_step)
                    * torch.sigmoid(self.quant_alpha)
                    + 1e-7
                )
                scale = scale.clamp(min=1e-5, max=1e4)
                offset = torch.zeros_like(scale).to(device)
            else:
                scale = (max_value - min_value) / (
                    positive_clip - negative_clip
                ) * torch.sigmoid(self.quant_alpha) + 1e-7
                offset = torch.round(
                    min_value * torch.sigmoid(self.quant_alpha) / scale
                )
            self.s.data.copy_(scale)
            self.offset.data.copy_(offset)

        else:
            scale = self.s
            offset = self.offset

        scale = scale.to(torch.float32).reshape(-1, 1)
        x = LSQAutograd.apply(
            x,
            scale,
            negative_clip,
            positive_clip,
        )[0]

        x = x.to(dtype)
        return x


@Quantizers.register_func("omni_group_min_max")
class OmniGroupMinMaxGridWeight(OmniMinMaxGrid):
    def __init__(
        self,
        father,
        bit,
        per_channel=True,
        unsigned_quant=False,
        quant_id=0,
        group_size=128,
        init_value=4.0,
        **argv,
    ):
        # weight input is: cout X cin
        # change group to: cout X group X group_size
        # if cin % group_size != 0, raise Exception
        self.group_size = group_size

        super().__init__(father, bit, per_channel, unsigned_quant, quant_id, **argv)

    def get_scale_shape(self, per_channel, father):
        if not per_channel:
            raise ValueError("only support per_channel True")
        dims = father.weight.dim()
        if dims != 2:
            raise ValueError(f"Expected 2 dimensions, but got {dims}")
        cin = father.weight.shape[1]
        cout = father.weight.shape[0]
        if cin % self.group_size != 0:
            raise ValueError(
                f"cin ({cin}) must be divisible by group_size ({self.group_size})."
            )
        self.groups = cin // self.group_size
        s_shape = [cout * self.groups]
        return s_shape

    def forward(self, x):
        cout, cin = x.shape
        target_shape = [-1, self.group_size]
        x = x.reshape(target_shape)

        x = super().forward(x)

        x = x.reshape([cout, cin])
        return x



