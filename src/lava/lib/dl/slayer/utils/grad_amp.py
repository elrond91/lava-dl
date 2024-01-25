# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Gradient attenuation mechanism."""

import torch

class _GradAmplifier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gain):
        
        # ctx.save_for_backward(torch.autograd.Variable(torch.tensor(gain, 
        #                                                            device=input.device, 
        #                                                            dtype=input.dtype),
        #                                               requires_grad=False))
        ctx.gain = gain
        return input
    
    @staticmethod
    def backward(ctx, grad_ouput):
        #gain = ctx.saved_tensors
        gain = ctx.gain
        #print('grad_amp.py ', [grad_ouput[idx].shape for idx in range(len(grad_ouput))])
        if isinstance(grad_ouput, tuple):
            grad_ouput = torch.stack(grad_ouput, dim=0)
        # print('grad_amp.py ', grad_ouput.shape )
        # print('grad_amp.py ', gain )
        # print('grad_amp.py ', grad_ouput * gain )
        return grad_ouput * gain, None
        #return (g * gain if g is not None else None for g in grad_ouput), None


def grad_amplifier(input, gain=1):
    return _GradAmplifier.apply(input, gain)


class GradAmplifier(torch.nn.Module):
    def __init__(self, gain: float = 1) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, input: torch.tensor) -> torch.tensor:
        return _GradAmplifier.apply(input, self.gain)