# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy, math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

class Transformer_Encoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_pos, key_padding_mask=None, attn_mask=None, verbose=False):
        src=src.permute(1,0,2)
        src_pos=src_pos.permute(1,0,2)

        memory,list_attn_maps = self.encoder(src, src_attn_mask=attn_mask, src_key_padding_mask=key_padding_mask, src_pos=src_pos, verbose=verbose)

        memory=memory.permute(1,0,2)
        return memory, list_attn_maps
        

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None, verbose=False):
        output = src
        list_attn_maps=[]

        for layer in self.layers:
            output,attn_map = layer(output, src_attn_mask=src_attn_mask,
                           src_key_padding_mask=src_key_padding_mask, src_pos=src_pos,verbose=verbose)
            list_attn_maps.append(attn_map)
        if self.norm is not None:
            output = self.norm(output)

        return output,list_attn_maps

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]


        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_attn_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None, verbose=False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, src_pos)
        
        
        src2,attn_map = self.self_attn(q, k, value=src2, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src,attn_map

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None, verbose=False):
        if self.normalize_before:
            return self.forward_pre(src, src_attn_mask, src_key_padding_mask, src_pos, verbose=verbose)
        return self.forward_post(src, src_attn_mask, src_key_padding_mask, src_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


#Below is the position encoding
#Attention is all you need
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):#dropout, 
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x_pe = Variable(self.pe[:, :x.size(1)],requires_grad=False)
        x_pe = x_pe.repeat(x.size(0),1,1)
        return x_pe

        #x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        #return self.dropout(x)
