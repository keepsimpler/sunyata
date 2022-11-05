from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


from sunyata.pytorch.arch.base import BaseCfg, BaseModule
from sunyata.pytorch.arch.deepattn import AttnLayer
from sunyata.pytorch.arch.convnext2 import Block, LayerNorm


class Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: int,
        query_idx: int = -1,
        temperature: float = 1.,
        init_scale: float = 1.,
        drop_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.attn_layer = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)
        self.block = Block(hidden_dim, drop_rate, layer_scale_init_value)

    def forward(self, xs, all_squeezed):
        x_new, all_squeezed = self.attn_layer(xs, all_squeezed)
        # x_new shape (batch_size, hidden_dim, height, width)
        x_next = self.block(x_new)
        x_next = x_next.unsqueeze(0)
        return torch.cat((xs, x_next), dim=0), all_squeezed


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    num_heads: int = 1
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0. 
    query_idx: int = -1   
    temperature: float = 1.
    init_scale: float = 1.
    layer_scale_init_value: float = 1e-6


class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            Layer(cfg.hidden_dim, cfg.num_heads, cfg.kernel_size, cfg.query_idx, cfg.temperature, cfg.init_scale, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.final_attn = AttnLayer(cfg.hidden_dim, cfg.num_heads, cfg.query_idx, cfg.temperature, cfg.init_scale)

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            LayerNorm(cfg.hidden_dim, eps=1e-6, data_format="channels_first"),
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            LayerNorm(cfg.hidden_dim, eps=1e-6, data_format="channels_last")
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg
        
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        xs = x.unsqueeze(0)
        all_squeezed = torch.zeros(0, device=x.device)
        for layer in self.layers:
            xs, all_squeezed = layer(xs, all_squeezed)
        x, all_squeezed = self.final_attn(xs, all_squeezed)
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
