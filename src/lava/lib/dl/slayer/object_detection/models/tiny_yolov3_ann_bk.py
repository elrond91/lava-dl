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
        
        yolo_stride = 1 if self.yolo_type == 'complete' else 2
        if self._quantize:
            from pytorch_quantization import nn as quant_nn
            self.blocks = torch.nn.ModuleList([
                quant_nn.Conv2d(3, 16, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(16, 32, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(32, 64, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(64, 128, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(128, 256, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(256, 512, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(512, 1024, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(1024, 256, bias=False, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(256, 512, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
                quant_nn.Conv2d(512, self.num_output, 1, 1, 0),
            ])
        else:
            self.blocks = torch.nn.ModuleList([
                nn.Conv2d(3, 16, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
                nn.Conv2d(16, 32, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
                nn.Conv2d(32, 64, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 128, bias=False, kernel_size = 3, stride = yolo_stride, padding = 1),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
                nn.Conv2d(128, 256, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
                nn.Conv2d(512, 1024, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
                nn.Conv2d(1024, 256, bias=False, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, bias=False, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
                nn.Conv2d(512, self.num_output, 1, 1, 0),
            ])

            
        
        #layers = [('convbatch_0',     cnn_block(3, 16, kernel_size = 3, stride = yolo_stride, padding = 1))]
        #if self.yolo_type == 'complete':
        #    layers.append(('max_1',           nn.MaxPool2d(2, 2)))
        #layers.append(('convbatch_2',     cnn_block(16, 32, kernel_size = 3, stride = yolo_stride, padding = 1)))
        #if self.yolo_type == 'complete':
        #     layers.append( ('max_3',           nn.MaxPool2d(2, 2)))
        #layers.append(('convbatch_4',     cnn_block(32, 64, kernel_size = 3, stride = yolo_stride, padding = 1)))
        #if self.yolo_type == 'complete':
        #    layers.append(('max_5',           nn.MaxPool2d(2, 2)))
        #layers.append(('convbatch_6',     cnn_block(64, 128, kernel_size = 3, stride = yolo_stride, padding = 1)))
        #if self.yolo_type == 'complete':
        #    layers.append(('max_7',           nn.MaxPool2d(2, 2)))
        # layers.append(('convbatch_8',     cnn_block(128, 256, kernel_size = 3, stride = 1, padding = 1)))
        # if self.yolo_type == 'complete':
        #     layers.append(('max_9',           nn.MaxPool2d(2, 2)))
        # layers.append(('convbatch_10',    cnn_block(256, 512, kernel_size = 3, stride = 1, padding = 1)))
        # if self.yolo_type == 'complete':
        #     layers.append(('max_11',          MaxPoolStride1()))
        # layers.append(('convbatch_12',    cnn_block(512, 1024, kernel_size =  3, stride = 1, padding = 1)))
        # layers.append(('convbatch_13',    cnn_block(1024, 256, kernel_size =  1, stride = 1, padding = 0)))
        
        # self.layers_back_bone = nn.Sequential(*[layer[1] for layer in layers])
        
        # self.yolo_0_pre = nn.Sequential(OrderedDict([
        #     ('convbatch_14',    cnn_block(256, 512, kernel_size = 3, stride = 1, padding = 1)),
        #     ('conv_15',         conv(512, self.num_output, 1, 1, 0)),
        # ]))
        
        # self.yolo_0_pre = nn.Sequential(
        #     cnn_block(256, 512, kernel_size = 3, stride = 1, padding = 1),
        #     conv(512, self.num_output, 1, 1, 0),
        #  )

        # if self.yolo_type != 'single-head':
        #     self.up_1 = nn.Sequential(OrderedDict([
        #         ('convbatch_17',    cnn_block(256, 128, kernel_size = 1, stride = yolo_stride, padding = 0)),
        #         ('upsample_18',     nn.Upsample(scale_factor=2, mode='nearest')),
        #     ]))

        #     self.yolo_1_pre = nn.Sequential(OrderedDict([
        #         ('convbatch_19',    cnn_block(128 + 256, 256, kernel_size = 3, stride = 1, padding = 1)),
        #         ('conv_20',         conv(256, self.num_output, 1, 1, 0)),
        #     ]))
        
        
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
        #num_layers = 9 if self.yolo_type == 'complete' else 5
        # if self.yolo_type != 'single-head':
        #     x_b_0 = self.layers_back_bone[:num_layers](input)
        #     x_b_full = self.layers_back_bone[num_layers:](x_b_0)
        #     y0 = self.yolo_0_pre(x_b_full)
            
        #     x_up = self.up_1(x_b_full)
        #     x_up = torch.cat((x_up, x_b_0), 1)
        #     y1 = self.yolo_1_pre(x_up)
            
        #     if self._quantize:
        #         y0 = self.residual_quantizer(y0)
        #         y1 = self.residual_quantizer(y1)
            
        #     y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
        #     y0 = y0.permute(0, 1, 3, 4, 2)
            
        #     y1 = y1.view(y1.size(0), self.num_anchors, self.num_classes + 5, y1.size(2), y1.size(3)) 
        #     y1 = y1.permute(0, 1, 3, 4, 2) 
            
        #     outputs = [y0, y1]
        # else:
        #     x_b_full = self.layers_back_bone(input)
        #     y0 = self.yolo_0_pre(x_b_full)
        #     if self._quantize:
        #         y0 = self.residual_quantizer(y0)
                
        #     y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
        #     y0 = y0.permute(0, 1, 3, 4, 2)
        #     outputs = [y0]
        
        for block in self.blocks: 
            input = block(input)
        y0 = input
        if self._quantize:
            y0 = self.residual_quantizer(y0)
            
        y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
        y0 = y0.permute(0, 1, 3, 4, 2)
        outputs = [y0]
     
        # if not self.training:
        #     outputs = [item[..., None] for item in outputs]
        #     outputs_yolo = []
        #     if self.yolo_type != 'single-head':
        #         outputs_yolo.append(self.yolo(outputs[0], self.anchors[0], self._quantize))
        #         outputs_yolo.append(self.yolo(outputs[1], self.anchors[1], self._quantize))
        #     else:
        #         outputs_yolo.append(self.yolo(outputs[0], self.anchors[0], self._quantize))
        #     outputs = torch.concat(outputs_yolo, dim=1)
            
        if self._quantize:
            return outputs
        return outputs, []
    
    
    
    
    """
    1.- as it is tiny_yolo_v3               [float][quant][tensorRT]  [GPU][jetson]
    2.- remove maxpool tiny_yolo_v3_str     [float][quant]
    3.- single head remove maxpool tiny_yolo_v3_str [float][quant]
    """