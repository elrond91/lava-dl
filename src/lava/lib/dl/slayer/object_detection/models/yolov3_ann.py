# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union

import torch
import torch.nn as nn 

from ..yolo_base import YOLOBase


# Defining CNN Block 
class CNNBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.activation = nn.LeakyReLU(0.1) 
        self.use_batch_norm = use_batch_norm 
  
    def forward(self, x): 
        # Applying convolution 
        x = self.conv(x) 
        # Applying BatchNorm and activation if needed 
        if self.use_batch_norm: 
            x = self.bn(x) 
            return self.activation(x) 
        else: 
            return x

# Defining residual block 
class ResidualBlock(nn.Module): 
    def __init__(self, channels, use_residual=True, num_repeats=1): 
        super().__init__() 
          
        # Defining all the layers in a list and adding them based on number of  
        # repeats mentioned in the design 
        res_layers = [] 
        for _ in range(num_repeats): 
            res_layers += [ 
                nn.Sequential( 
                    nn.Conv2d(channels, channels // 2, kernel_size=1), 
                    nn.BatchNorm2d(channels // 2), 
                    nn.LeakyReLU(0.1), 
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(channels), 
                    nn.LeakyReLU(0.1) 
                ) 
            ] 
        self.layers = nn.ModuleList(res_layers) 
        self.use_residual = use_residual 
        self.num_repeats = num_repeats 
      
    # Defining forward pass 
    def forward(self, x): 
        for layer in self.layers: 
            residual = x 
            x = layer(x) 
            if self.use_residual: 
                x = x + residual 
        return x

# Defining scale prediction class 
class ScalePrediction(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super().__init__() 
        # Defining the layers in the network 
        self.pred = nn.Sequential( 
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(2*in_channels), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
        ) 
        self.num_classes = num_classes 
      
    # Defining the forward pass and reshaping the output to the desired output  
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x): 
        output = self.pred(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output


class CNNBlockQ(nn.Module): 
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
        super().__init__() 
        from pytorch_quantization import nn as quant_nn
        self.conv = quant_nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.activation = nn.LeakyReLU(0.1) 
        self.use_batch_norm = use_batch_norm 
  
    def forward(self, x): 
        # Applying convolution 
        x = self.conv(x) 
        # Applying BatchNorm and activation if needed 
        if self.use_batch_norm: 
            x = self.bn(x) 
            return self.activation(x) 
        else: 
            return x

# Defining residual block 
class ResidualBlockQ(nn.Module): 
    def __init__(self, channels, use_residual=True, num_repeats=1): 
        super().__init__() 
        from pytorch_quantization import nn as quant_nn
        # Defining all the layers in a list and adding them based on number of  
        # repeats mentioned in the design 
        res_layers = [] 
        for _ in range(num_repeats): 
            res_layers += [ 
                nn.Sequential( 
                    quant_nn.Conv2d(channels, channels // 2, kernel_size=1), 
                    nn.BatchNorm2d(channels // 2), 
                    nn.LeakyReLU(0.1), 
                    quant_nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(channels), 
                    nn.LeakyReLU(0.1) 
                ) 
            ] 
        self.layers = nn.ModuleList(res_layers) 
        self.use_residual = use_residual 
        self.num_repeats = num_repeats 
      
    # Defining forward pass 
    def forward(self, x): 
        for layer in self.layers: 
            residual = x 
            x = layer(x) 
            if self.use_residual: 
                x = x + residual 
        return x

# Defining scale prediction class 
class ScalePredictionQ(nn.Module): 
    def __init__(self, in_channels, num_classes): 
        super().__init__()
        from pytorch_quantization import nn as quant_nn
        # Defining the layers in the network 
        self.pred = nn.Sequential( 
            quant_nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(2*in_channels), 
            nn.LeakyReLU(0.1), 
            quant_nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1), 
        ) 
        self.num_classes = num_classes 
      
    # Defining the forward pass and reshaping the output to the desired output  
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
    def forward(self, x): 
        output = self.pred(x) 
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
        output = output.permute(0, 1, 3, 4, 2) 
        return output

class Network(YOLOBase):
    def __init__(self,
                 yolo_type: str = '',
                 quantize: bool = False,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [ 
                        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
                        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
                    ]) -> None:
        super().__init__(num_classes=num_classes, anchors=anchors)
        self.num_classes = num_classes 
        self._quantize = quantize
        
        self.cnn_block = CNNBlock
        self.residual_block = ResidualBlock
        self.scale_prediction = ScalePrediction
        
        if self._quantize:
            from pytorch_quantization import nn as quant_nn
            self.cnn_block = CNNBlockQ
            self.residual_block = ResidualBlockQ
            self.scale_prediction = ScalePredictionQ
  
        # Layers list for YOLOv3 
        self.layers = nn.ModuleList([ 
            self.cnn_block(3, 32, kernel_size=3, stride=1, padding=1), 
            self.cnn_block(32, 64, kernel_size=3, stride=2, padding=1), 
            self.residual_block(64, num_repeats=1), 
            self.cnn_block(64, 128, kernel_size=3, stride=2, padding=1), 
            self.residual_block(128, num_repeats=2), 
            self.cnn_block(128, 256, kernel_size=3, stride=2, padding=1), 
            self.residual_block(256, num_repeats=8), 
            self.cnn_block(256, 512, kernel_size=3, stride=2, padding=1), 
            self.residual_block(512, num_repeats=8), 
            self.cnn_block(512, 1024, kernel_size=3, stride=2, padding=1), 
            self.residual_block(1024, num_repeats=4), 
            self.cnn_block(1024, 512, kernel_size=1, stride=1, padding=0), 
            self.cnn_block(512, 1024, kernel_size=3, stride=1, padding=1), 
            self.residual_block(1024, use_residual=False, num_repeats=1), 
            self.cnn_block(1024, 512, kernel_size=1, stride=1, padding=0), 
            self.scale_prediction(512, num_classes=num_classes), 
            
            self.cnn_block(512, 256, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            self.cnn_block(768, 256, kernel_size=1, stride=1, padding=0), 
            self.cnn_block(256, 512, kernel_size=3, stride=1, padding=1), 
            self.residual_block(512, use_residual=False, num_repeats=1), 
            self.cnn_block(512, 256, kernel_size=1, stride=1, padding=0), 
            self.scale_prediction(256, num_classes=num_classes), 
            
            self.cnn_block(256, 128, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=2), 
            self.cnn_block(384, 128, kernel_size=1, stride=1, padding=0), 
            self.cnn_block(128, 256, kernel_size=3, stride=1, padding=1), 
            self.residual_block(256, use_residual=False, num_repeats=1), 
            self.cnn_block(256, 128, kernel_size=1, stride=1, padding=0), 
            self.scale_prediction(128, num_classes=num_classes) 
        ]) 
      
        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        
        if self._quantize:
            self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

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
        route_connections = [] 
  
        for layer in self.layers: 
            if isinstance(layer, self.scale_prediction): 
                outputs.append(layer(input)) 
                continue
            input = layer(input) 
  
            if isinstance(layer, self.residual_block) and layer.num_repeats == 8: 
                route_connections.append(input) 
              
            elif isinstance(layer, nn.Upsample): 
                input = torch.cat([input, route_connections[-1]], dim=1) 
                route_connections.pop()
        
        if self._quantize:
            outputs = self.residual_quantizer(outputs)
                
        if not self.training:
            outputs = [item[..., None] for item in outputs]
            outputs = torch.concat([self.yolo(p, a) for (p, a)
                                            in zip(outputs,  self.anchors)],
                                           dim=1)
        return outputs, []