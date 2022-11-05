from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Residual
from sunyata.pytorch.arch.deepattn import AttnLayer
from sunyata.pytorch.arch.convnext2 import Block, LayerNorm


@dataclass
class IsotropicCfg(BaseCfg):
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
    head_init_scale: float = 1.

    expansion: int = 4


class Isotropic(BaseModule):
    def __init__(self, cfg: IsotropicCfg):
        super().__init__(cfg)

        drop_rates = [x.item() for x in torch.linspace(0, cfg.drop_rate, cfg.num_layers)]

        self.layers = nn.ModuleList([
            Residual(Block(cfg.hidden_dim, drop_rates[i], cfg.layer_scale_init_value, cfg.kernel_size, cfg.expansion))
            for i in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            LayerNorm(cfg.hidden_dim, eps=1e-6, data_format="channels_first"),
        )
        
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            LayerNorm(cfg.hidden_dim, eps=1e-6, data_format="channels_last"),
            self.head
        )

        self.apply(self._init_weights)
        self.head.weight.data.mul_(cfg.head_init_scale)
        self.head.bias.data.mul_(cfg.head_init_scale)

        self.cfg = cfg
        
    def _init_weights(self,m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
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
