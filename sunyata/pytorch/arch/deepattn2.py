"""
xs as list to save memory.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, ConvMixerLayer, LayerScaler


class Attn(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.,
        init_scale: float = 1.,
        query_idx: int = -1,
    ):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            LayerScaler(hidden_dim, init_scale),
        )
        self.temperature = temperature
        self.query_idx = query_idx
        
    def forward(self, *xs):
        squeezed = [self.squeeze(x) for x in xs]
        squeezed = torch.stack(squeezed)  
        attn = torch.einsum('d b h, b h -> b d', squeezed, squeezed[self.query_idx,:,:])
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=-1)
        
        next_x = xs[0] * attn[:, 0, None, None, None]
        for i, x in enumerate(xs[1:]):
            next_x = next_x + x * attn[:, i, None, None, None]
        return next_x


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        drop_rate: float = 0.,
        temperature: float = 1.,
        init_scale: float = 1.,
        query_idx: int = -1,
    ):
        super().__init__()
        self.attn = Attn(hidden_dim, temperature, init_scale, query_idx)
        self.unit = ConvMixerLayer(hidden_dim, kernel_size, drop_rate)

    def forward(self, *xs):
        x = self.attn(*xs)
        x= self.unit(x)
        return xs + (x,)


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    drop_rate: float=0.
    
    query_idx_exp: float = 1.
    query_idx_denominator: int = 1
    query_idx_shift: int = 0
    temperature: float = 1.
    init_scale: float = 1.


class DeepAttn(BaseModule):
    def __init__(self, cfg:DeepAttnCfg):
        super().__init__(cfg)
        
        query_idxs = [
            max(0, int(current_depth ** cfg.query_idx_exp // cfg.query_idx_denominator) - cfg.query_idx_shift)
            for current_depth in range(cfg.num_layers + 1)
        ]
        self.layers = nn.ModuleList([
            AttnLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate,
                      cfg.temperature, cfg.init_scale, query_idx)
            for query_idx in query_idxs[:-1]
        ])
        
        self.final_attn = Attn(cfg.hidden_dim, temperature=cfg.temperature, init_scale=cfg.init_scale, query_idx=query_idxs[-1])
        
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),  # 
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        xs = (x,)
        for layer in self.layers:
            xs = layer(*xs)
        x = self.final_attn(*xs)
        x= self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
