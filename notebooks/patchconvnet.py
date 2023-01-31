
# %%
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class Learned_Aggregation_Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        q = q * self.scale
        attn = q @ k.transpose(-2,-1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        

# %%
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output
# %%
drop_path(torch.randn(2,4,2,2), drop_prob=0.5, training=True)
# %%
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
# %%
droppath = DropPath(drop_prob=0.5)
droppath(torch.randn(4,2,2,2))
# %%
class NodeOP(nn.Module):
    def __init__(self, Unit:nn.Module, *args, **kwargs):
        super(NodeOP, self).__init__()
        self.unit = Unit(*args, **kwargs)

    def forward(self, *xs):
        sum_xs = sum(xs)
        output = self.unit(sum_xs)
        return output
# %%
from sunyata.pytorch.arch.foldnet import ConvMixerLayer
# %%
x = torch.randn(2,8,8,8)
ConvMixerLayer(hidden_dim=8, kernel_size=3)(x).shape
# %%
NodeOP(ConvMixerLayer, hidden_dim=8, kernel_size=3)(*[x]).shape
# %%
class Network(nn.Module):
    def __init__(self, num_layers:int, Unit:nn.Module, *args, **kwargs):
        super(Network, self).__init__()
        self.layers = nn.ModuleList([
            NodeOP(Unit, *args, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, *xs):
        for layer in self.layers:
            x = layer(*xs)
            xs = xs + (x,)
        return xs
# %%
output = Network(num_layers=2, Unit=ConvMixerLayer, hidden_dim=8, kernel_size=3)(x)
len(output)
# %%
