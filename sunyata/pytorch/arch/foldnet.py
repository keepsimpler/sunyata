from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Residual


@dataclass
class FoldNetCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    fold_num: int = 1


class ResConvMixer(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            Residual(nn.Sequential(
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            )
            ) for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
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


class Block(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )


class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, Unit:nn.Module, hidden_dim: int, kernel_size: int, fold_num:int):
        super(FoldBlock, self).__init__()
        self.fold_num = fold_num
        units = []
        for i in range(max(1, fold_num - 1)):
            units += [Unit(hidden_dim, kernel_size)]
        self.units = nn.ModuleList(units)
        
    def forward(self, *xs):
        xs = list(xs)
        if self.fold_num == 1:
            xs[0] = xs[0] + self.units[0](xs[0])
            return xs
        for i in range(self.fold_num - 1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs


class FoldNet(BaseModule):
    def __init__(self, cfg:FoldNetCfg, fold_num: int):
        super().__init__(cfg)
        
        assert cfg.hidden_dim % fold_num == 0
        
        self.layers = nn.ModuleList([
            FoldBlock(Block, cfg.hidden_dim // fold_num, cfg.kernel_size, fold_num)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg
        self.fold_num = fold_num

    def forward(self, x):
        x = self.embed(x)
        x = torch.chunk(x, self.fold_num, dim = 1)
        for layer in self.layers:
            x= layer(*x)
        x = torch.cat(x, dim = 1)
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