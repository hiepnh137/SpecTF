import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F




class FullFreqAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullFreqAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        '''
            queries, keys, values are complex numbers
        '''
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        origin = scores.clone()
        # print('queries: ', queries)
        # print('keys: ', keys)
        # print('scores: ',scores)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        # print(scores)
        # print('scores: ', scores.shape)
        scores = scores.abs()
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # A = self.dropout(torch.sigmoid(scale * scores))
        A = A.unsqueeze(-1)
        zeros = torch.zeros_like(A).to(queries.device) 
        A = torch.cat([A, zeros], dim=-1)

        A = torch.view_as_complex(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), origin
        else:
            return V.contiguous(), None
        

class FreqAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(FreqAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection_real = nn.Linear(d_model, d_keys * n_heads)
        self.query_projection_imag = nn.Linear(d_model, d_keys * n_heads)

        self.key_projection_real = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection_imag = nn.Linear(d_model, d_keys * n_heads)

        self.value_projection_real = nn.Linear(d_model, d_values * n_heads)
        self.value_projection_imag = nn.Linear(d_model, d_values * n_heads)

        self.out_projection_real = nn.Linear(d_values * n_heads, d_model)
        self.out_projection_imag = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.sparsity_threshold = 0.01
    
    def projection_layer(self, enc, real_layer, imag_layer):
        real = real_layer(enc.real) - imag_layer(enc.imag)
        imag = imag_layer(enc.real) + real_layer(enc.imag)

        output = torch.stack([real, imag], dim=-1)
        output = F.softshrink(output, lambd=self.sparsity_threshold)
        output = torch.view_as_complex(output)

        return output
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.projection_layer(queries, self.query_projection_real, self.query_projection_imag).view(B, L, H, -1)
        keys = self.projection_layer(keys, self.key_projection_real, self.key_projection_imag).view(B, S, H, -1)
        values = self.projection_layer(values, self.value_projection_real, self.key_projection_imag).view(B, S, H, -1)
        # queries = queries.view(B, S, H, -1)
        # keys = keys.view(B, S, H, -1)
        # values = values.view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        out = out.view(B, L, -1)
        # out = self.projection_layer(out, self.out_projection_real, self.out_projection_imag)
        return out, attn
        # return self.out_projection(out), attn
