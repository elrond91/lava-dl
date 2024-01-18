# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Gradient attenuation mechanism."""

import torch

def _GradAmplifier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gain):
        ctx.save_for_backward(torch.autograd.Variable(gain,
                                                      requires_grad=False))
        return input
    
    @staticmethod
    def backward(ctx, grad_ouput):
        gain = ctx.saved_tensors
        return grad_ouput * gain


def grad_amplifier(input, gain=1):
    return _GradAmplifier.apply(input, gain)
