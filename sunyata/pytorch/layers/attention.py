import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, scale: float=None, dropout=0.,
                 is_to_qkv_bias=False, is_to_out=True, is_to_out_bias=False,
                 is_mask=True, is_softmax=True):
        super().__init__()
        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads

        self.scale = head_dim ** -0.5 if scale is None else scale

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=is_to_qkv_bias)
        self.attend = nn.Softmax(dim=-1) if is_softmax else nn.Identity()
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=is_to_out_bias),
            nn.Dropout(dropout),
         ) if is_to_out else nn.Identity()

        self.is_mask, self.is_softmax = is_mask, is_softmax

    def forward(self, x):
        _, seq_len, _ = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.is_mask and self.is_softmax:
            attn_mask = torch.full((seq_len, seq_len), -float('Inf'), device=x.device, dtype=x.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            dots = dots + attn_mask
        elif self.is_mask and not self.is_softmax:
            attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype)
            attn_mask = torch.tril(attn_mask)
            dots = dots * attn_mask

        attn = self.attend(dots)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
