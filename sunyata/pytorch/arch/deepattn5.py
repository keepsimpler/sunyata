"""
Squeeze of all the previous xs as keys and queries.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, ConvMixerLayer, LayerScaler


class Squeeze(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LayerScaler(hidden_dim, init_scale),
        )

    def forward(self, x):
        current_depth, batch_size, hidden_dim, _, _ = x.shape
        x2 = Rearrange('d b h w1 w2 -> (d b) h w1 w2', d = current_depth, b = batch_size)(x)
        squeezed = self.squeeze(x2)
        squeezed = Rearrange('(d b) h -> d b h', d = current_depth, b = batch_size)(squeezed)
        return squeezed


class Attn(nn.Module):
    def __init__(
        self,
        temperature: float = 1.,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, keys):
        # query: (batch_size, hidden_dim)
        # keys: (current_depth, batch_size, hidden_dim)
        attn = torch.einsum('b h, d b h -> d b', query, keys)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)
        return attn


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.squeeze = Squeeze(hidden_dim, init_scale)
        self.attn = Attn(temperature)
        self.query_idx = query_idx

    def forward(self, xs):
        # x shape (current_depth, batch_size, hidden_dim, height, width)
        squeezed = self.squeeze(xs)
        # squeezed shape (current_depth, batch_size, hidden_dim)
        query = squeezed[self.query_idx,:,:]
        keys = squeezed
        attended = self.attn(query, keys)
        # attended shape (current_depth, batch_size)
        x_new = torch.einsum('d b h v w, d b -> b h v w', xs, attended)
        # x_new shape (batch_size, hidden_dim, height, width)
        return x_new


class Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.attn_layer = AttnLayer(hidden_dim, query_idx, temperature, init_scale)
        self.block = ConvMixerLayer(hidden_dim, kernel_size)

    def forward(self, xs):
        x_new = self.attn_layer(xs)
        # x_new shape (batch_size, hidden_dim, height, width)
        x_next = self.block(x_new)
        x_next = x_next.unsqueeze(0)
        return torch.cat((xs, x_next), dim=0)        


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0. 
    query_idx: int = -1   
    temperature: float = 1.
    init_scale: float = 1.


class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            Layer(cfg.hidden_dim, cfg.kernel_size, cfg.query_idx, cfg.temperature, cfg.init_scale)
            for _ in range(cfg.num_layers)
        ])

        self.final_attn = AttnLayer(cfg.hidden_dim, cfg.query_idx, cfg.temperature, cfg.init_scale)

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),  # eps>6.1e-5 to avoid nan in half precision
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg
        
    def forward(self, x):
        x = self.embed(x)
        xs = x.unsqueeze(0)
        for layer in self.layers:
            xs = layer(xs)
        x = self.final_attn(xs)
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
