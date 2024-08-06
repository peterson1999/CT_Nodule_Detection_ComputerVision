#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:25:15 2024

@author: petersonco
"""

import torch.nn as nn
import math

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, conv_channels, padding=1,bias=True,kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
    
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, padding=1,bias=True,kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool3d(2,2)
    
    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        
        return self.maxpooling(block_out)

class LunaModel(nn.Module):
    def __init__(self,in_channels=1,conv_channels=8):
        super().__init__()
        
        self.tail = nn.BatchNorm3d(1) #tail
        
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)
        
        self.linear = nn.Linear(1152, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_batch):
        tail_out = self.tail(input_batch)
        
        block_out = self.block1(tail_out)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        
        conv_flat = block_out.view(block_out.size(0),-1)
        
        linear_out = self.linear(conv_flat)
        head_out = self.softmax(linear_out)
        
        return linear_out, head_out
    
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
            }:
                nn.init.kaiming_normal_(
                m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                    nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

                
        