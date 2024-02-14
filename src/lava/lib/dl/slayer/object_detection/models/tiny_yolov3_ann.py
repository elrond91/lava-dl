# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..yolo_base import YOLOBase
from .yolov3_ann import CNNBlock

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Network(YOLOBase):
    def __init__(self,
                 yolo_type: str = 'complete', # complete, str, single-head
                 num_classes: int = 80,
                 quantize: bool = False,
                 anchors: List[List[Tuple[float, float]]] = [ 
                        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)] 
                    ]) -> None:
        super().__init__(num_classes=num_classes, anchors=anchors)
        
        self.yolo_type = yolo_type
        self._quantize = quantize
        
        cnn_block = CNNBlock
        conv = nn.Conv2d
        
        if self._quantize:
            from pytorch_quantization import nn as quant_nn
            from .yolov3_ann import CNNBlockQ
            
            conv = quant_nn.Conv2d
            cnn_block = CNNBlockQ
            
        
        yolo_stride = 1 if self.yolo_type == 'complete' else 2
        
        layers = [('0_convbatch',     cnn_block(3, 16, kernel_size = 3, stride = yolo_stride, padding = 1))]
        if self.yolo_type == 'complete':
            layers.append(('1_max',           nn.MaxPool2d(2, 2)))
        layers.append(('2_convbatch',     cnn_block(16, 32, kernel_size = 3, stride = yolo_stride, padding = 1)))
        if self.yolo_type == 'complete':
             layers.append( ('3_max',           nn.MaxPool2d(2, 2)))
        layers.append(('4_convbatch',     cnn_block(32, 64, kernel_size = 3, stride = yolo_stride, padding = 1)))
        if self.yolo_type == 'complete':
            layers.append(('5_max',           nn.MaxPool2d(2, 2)))
        layers.append(('6_convbatch',     cnn_block(64, 128, kernel_size = 3, stride = yolo_stride, padding = 1)))
        if self.yolo_type == 'complete':
            layers.append(('7_max',           nn.MaxPool2d(2, 2)))
        layers.append(('8_convbatch',     cnn_block(128, 256, kernel_size = 3, stride = 1, padding = 1)))
        if self.yolo_type == 'complete':
            layers.append(('9_max',           nn.MaxPool2d(2, 2)))
        layers.append(('10_convbatch',    cnn_block(256, 512, kernel_size = 3, stride = 1, padding = 1)))
        if self.yolo_type == 'complete':
            layers.append(('11_max',          MaxPoolStride1()))
        layers.append(('12_convbatch',    cnn_block(512, 1024, kernel_size =  3, stride = 1, padding = 1)))
        layers.append(('13_convbatch',    cnn_block(1024, 256, kernel_size =  1, stride = 1, padding = 0)))
        self.layers_back_bone = nn.Sequential( OrderedDict(layers))
        
        self.yolo_0_pre = nn.Sequential(OrderedDict([
            ('14_convbatch',    cnn_block(256, 512, kernel_size = 3, stride = 1, padding = 1)),
            ('15_conv',         conv(512, self.num_output, 1, 1, 0)),
        ]))

        if self.yolo_type != 'single-head':
            self.up_1 = nn.Sequential(OrderedDict([
                ('17_convbatch',    cnn_block(256, 128, kernel_size = 1, stride = yolo_stride, padding = 0)),
                ('18_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
            ]))

            self.yolo_1_pre = nn.Sequential(OrderedDict([
                ('19_convbatch',    cnn_block(128 + 256, 256, kernel_size = 3, stride = 1, padding = 1)),
                ('20_conv',         conv(256, self.num_output, 1, 1, 0)),
            ]))
        
        
        if self._quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
        
        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

    def forward(
        self,
        input: torch.tensor,
    ) -> Tuple[Union[torch.tensor, List[torch.tensor]], torch.tensor]:
        
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
            
        # we remove time dependency 
        input = torch.reshape(input, (input.shape[0], input.shape[1], input.shape[2], input.shape[3]))
        input = (input - self.normalize_mean) / self.normalize_std
        
        outputs = [] 
        num_layers = 9 if self.yolo_type == 'complete' else 5
        if self.yolo_type != 'single-head':
            x_b_0 = self.layers_back_bone[:num_layers](input)
            x_b_full = self.layers_back_bone[num_layers:](x_b_0)
            y0 = self.yolo_0_pre(x_b_full)
            
            x_up = self.up_1(x_b_full)
            x_up = torch.cat((x_up, x_b_0), 1)
            y1 = self.yolo_1_pre(x_up)
            
            if self._quantize:
                y0 = self.residual_quantizer(y0)
                y1 = self.residual_quantizer(y1)
            
            y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
            y0 = y0.permute(0, 1, 3, 4, 2)
            
            y1 = y1.view(y1.size(0), self.num_anchors, self.num_classes + 5, y1.size(2), y1.size(3)) 
            y1 = y1.permute(0, 1, 3, 4, 2) 
            
            outputs = [y0, y1]
        else:
            x_b_full = self.layers_back_bone(input)
            y0 = self.yolo_0_pre(x_b_full)
            if self._quantize:
                y0 = self.residual_quantizer(y0)
                
            y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
            y0 = y0.permute(0, 1, 3, 4, 2)
            outputs = [y0]
            
        if self._quantize:
            return outputs
        return outputs, []
    
    
    
    
    """
    1.- as it is tiny_yolo_v3               [float][quant][tensorRT]  [GPU][jetson]
    2.- remove maxpool tiny_yolo_v3_str     [float][quant]
    3.- single head remove maxpool tiny_yolo_v3_str [float][quant]
    """