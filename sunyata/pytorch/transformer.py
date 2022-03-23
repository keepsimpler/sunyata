from sympy import expand
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, expanded_dim, num_heads, dropout):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(hidden_dim, num_heads, expanded_dim, dropout)


    def forward(self, x):
        _, seq_len, _ = x.shape
        attn_mask = torch.full((seq_len, seq_len), -float('Inf'), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        return self.transformer_layer(x)