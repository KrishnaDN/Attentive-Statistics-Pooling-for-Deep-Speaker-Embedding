#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:05:55 2020

@author: krishna
"""


import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = lin_out.bmm(v_view).squeeze()
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized


class ScaledDotProduct_attention(nn.Module):
    """
    Scaled dot product attention
    """

    def __init__(self, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.scaling = self.embed_dim ** -0.5
    
    
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        
        self.reset_parameters()
        
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(2 * embed_dim))
        
    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
        
    def in_proj_qk(self, query):
        return self._in_proj(query).chunk(2, dim=-1)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        
    def forward(self,query,key):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        q, k = self.in_proj_qk(query)
        q *= self.scaling
        return q,k
        attn_weights = torch.bmm(q, k.transpose(1, 2))    
        return attn_weights
