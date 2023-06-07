# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# %%
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, scale: float=None, dropout=0.,
                 is_mask=True, is_softmax=True, fore_mask=True):
        super().__init__()
        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads

        self.scale = head_dim ** -0.5 if scale is None else scale

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        
        self.attend = nn.Softmax(dim=-1) if is_softmax else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
         )

        self.is_mask, self.is_softmax, self.fore_mask = is_mask, is_softmax, fore_mask

    def forward(self, x):
        _, seq_len, _ = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.is_mask and self.is_softmax:
            attn_mask = torch.full((seq_len, seq_len), -float('Inf'), device=x.device, dtype=x.dtype)
            if self.fore_mask:
                attn_mask = torch.triu(attn_mask, diagonal=1)
            else:
                attn_mask = torch.tril(attn_mask, diagonal=-1)
            dots = dots + attn_mask
        elif self.is_mask and not self.is_softmax:
            attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype)
            attn_mask = torch.tril(attn_mask)
            dots = dots * attn_mask
            # dots = F.normalize(dots, dim=-1, p=2)

        attn = self.attend(dots)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

# %%
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()

    def forward(self, x, context = None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out
# %%
input = torch.randn(2, 3, 224, 224)
input = input.permute(0, 2, 3, 1)
input = rearrange(input, 'b ... d -> b (...) d')
b, _, input_dim = input.shape
# %%
query_dim = 256
latents = nn.Parameter(torch.randn(1, query_dim))
latents = repeat(latents, 'n d -> b n d', b = b)
# %%
cross_attn = Attention(
    query_dim=query_dim,
    context_dim=input_dim,
    heads=1,
    dim_head=64,
)
# %%
output = cross_attn(latents, input)
# %%
