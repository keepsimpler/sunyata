
"""
xs as list to save memory. with keys of keys of keys of x.
attn_depth
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Block, LayerScaler


class Attn(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.,
        init_scale: float = 1.,
        attn_depth: int = 1,
    ):
        super().__init__()
        assert attn_depth >= 1
        self.squeezes = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                LayerScaler(hidden_dim, init_scale),
            ) for _ in range(attn_depth)
            ]
        self.temperature = temperature
        self.attn_depth = attn_depth
        
    def forward(self, *xs):
        for i in range(self.attn_depth):
            squeezed = [self.squeezes[i](x) for x in xs]
            squeezed = torch.stack(squeezed)  
            attn = torch.einsum('d b h, e b h -> b d e', squeezed, squeezed)
            attn = attn / self.temperature
            attn = F.softmax(attn, dim=-1)
            if i == 0:
                final_attn = attn.mean(dim=-2)
            else:
                final_attn = torch.einsum('b d e, b d -> b d', attn, final_attn)
        
        next_x = xs[0] * final_attn[:, 0, None, None, None]
        for i, x in enumerate(xs[1:]):
            next_x = next_x + x * final_attn[:, i, None, None, None]
        return next_x


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        drop_rate: float = 0.,
        temperature: float = 1.,
        init_scale: float = 1.,
        attn_depth: int = 1,
    ):
        super().__init__()
        self.attn = Attn(hidden_dim, temperature, init_scale, attn_depth)
        self.unit = Block(hidden_dim, kernel_size, drop_rate)

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

    drop_rate: float = 0.    
    temperature: float = 1.
    init_scale: float = 1.    
    attn_depth: int = 2


class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.attn_layers = nn.ModuleList([
            AttnLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate,
                      cfg.temperature, cfg.init_scale, cfg.attn_depth)
            for _ in range(cfg.num_layers)
        ])

        self.final_attn = Attn(cfg.hidden_dim, cfg.temperature, cfg.init_scale, cfg.attn_depth)

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
        xs = (x,)
        for attn_layer in self.attn_layers:
            xs = attn_layer(*xs)
        x = self.final_attn(*xs)
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
