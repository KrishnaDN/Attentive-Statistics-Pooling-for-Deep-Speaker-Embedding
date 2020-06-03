#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch.nn as nn
from modules.tdnn import TDNN
from modules.Attention_Pooling import Classic_Attention
import torch


class X_vector_Attn_Pooling(nn.Module):
    def __init__(self, input_dim = 40, num_classes=462):
        super(X_vector_Attn_Pooling, self).__init__()
        ## Frame level feature processing
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.2)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=5, dilation=2,dropout_p=0.2)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=7, dilation=3,dropout_p=0.2)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.2)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.2)
        ### Statistics attentive pooling
        self.attention = Classic_Attention(512,512)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512,512 )
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    
    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        attn_weights = self.attention(tdnn5_out)
        stat_pool_out = self.stat_attn_pool(tdnn5_out,attn_weights)
        segment6_out = self.segment6(stat_pool_out)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions,x_vec