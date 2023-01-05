"""
Based on deepattn7, for text rather than images
"""
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, LayerScaler, Residual
from sunyata.pytorch.arch.textconv import Conv1dWithLeftPad

class Squeeze(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool1d((1,)),
            nn.Flatten(),
            LayerScaler(hidden_dim, init_scale),
        )

    def forward(self, x):
        # x shape (batch_size, hidden_dim, height, weight)
        squeezed = self.squeeze(x)
        return squeezed


class Attn(nn.Module):
    def __init__(
        self,
        num_heads: int,
        temperature: float = 1.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature

    def forward(self, query, keys):
        # query: (batch_size, hidden_dim) -> (batch_size heads head_dim)
        query = Rearrange('b (heads head_dim) -> b heads head_dim', heads=self.num_heads)(query)
        # keys: (current_depth, batch_size, hidden_dim) -> (current_depth, batch_size, heads, head_dim)
        keys = Rearrange('d b (heads head_dim) -> d b heads head_dim', heads=self.num_heads)(keys)

        attn = torch.einsum('b e h, d b e h -> d b e', query, keys)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)
        return attn


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.squeeze = Squeeze(hidden_dim, init_scale)
        self.attn = Attn(num_heads, temperature)
        self.num_heads = num_heads
        self.query_idx = query_idx

    def forward(self, xs, all_squeezed):
        # xs shape (current_depth, batch_size, hidden_dim, seq_len)
        squeezed = self.squeeze(xs[-1]).unsqueeze(0)
        all_squeezed = torch.cat([all_squeezed, squeezed])
        # all_squeezed shape (current_depth, batch_size, hidden_dim)
        query = all_squeezed[self.query_idx,:,:]
        keys = all_squeezed
        attended = self.attn(query, keys)
        # attended shape (current_depth, batch_size, num_heads)
        xs = Rearrange('d b (heads head_dim) s -> d b heads head_dim s', heads=self.num_heads)(xs)
        x_new = torch.einsum('d b e h s, d b e -> b e h s', xs, attended)
        x_new = Rearrange('b heads head_dim s -> b (heads head_dim) s')(x_new)
        # x_new shape (batch_size, hidden_dim, seq_len)
        return x_new, all_squeezed


class Block(nn.Sequential):
    def __init__(self, hidden_dim, kernel_size, groups, expansion, is_ff, norm_layer):
        super().__init__(
            Residual(nn.Sequential(
                Conv1dWithLeftPad(hidden_dim, kernel_size, groups),
                nn.GELU(),
                norm_layer(hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            )),
            Residual(nn.Sequential(
                nn.Conv1d(hidden_dim, expansion * hidden_dim if is_ff else hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(expansion * hidden_dim, hidden_dim, kernel_size=1) if is_ff else nn.Identity(),
                norm_layer(hidden_dim)
            ))
        )

class Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
        expansion: int = 1,
        is_ff: bool = True,
        groups: int = 1,
        norm_layer = nn.BatchNorm1d
    ):
        super().__init__()
        self.attn_layer = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)
        self.block = Block(hidden_dim, kernel_size, groups, expansion, is_ff, norm_layer)

    def forward(self, xs, all_squeezed):
        x_new, all_squeezed = self.attn_layer(xs, all_squeezed)
        # x_new shape (batch_size, hidden_dim, seq_len)
        x_next = self.block(x_new)
        x_next = x_next.unsqueeze(0)
        return torch.cat((xs, x_next), dim=0), all_squeezed


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 64
    vocab_size: int = 1000
    seq_len: int = 128

    kernel_size: int = 3

    embed_init_func: Callable = nn.init.xavier_normal_  # nn.init.zeros_

    groups: int = 64

    is_ff: bool = False
    expansion: int = 2

    drop_rate: float = 0.
    num_heads: int = 1
    query_idx: int = -1   
    temperature: float = 1.
    init_scale: float = 1.

    # LayerNorm1d nn.GroupNorm(1, cfg.hidden_dim) nn.InstanceNorm1d(cfg.hidden_dim, affine=True) nn.BatchNorm1d
    norm_layer: nn.Module =  nn.BatchNorm1d



class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            Layer(cfg.hidden_dim, cfg.num_heads, cfg.kernel_size, cfg.query_idx, cfg.temperature, cfg.init_scale, 
                    cfg.expansion, cfg.is_ff, cfg.groups, cfg.norm_layer)
            for _ in range(cfg.num_layers)
        ])

        self.final_attn = AttnLayer(cfg.hidden_dim, cfg.num_heads, cfg.query_idx, cfg.temperature, cfg.init_scale)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        cfg.embed_init_func(self.embed.weight.data)
        
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.digup.weight = self.embed.weight

        self.cfg = cfg
        
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        xs = x.unsqueeze(0)
        all_squeezed = torch.zeros(0, device=x.device)
        for layer in self.layers:
            xs, all_squeezed = layer(xs, all_squeezed)
        x, all_squeezed = self.final_attn(xs, all_squeezed)
        x = x.permute(0, 2, 1)
        x = self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss    
