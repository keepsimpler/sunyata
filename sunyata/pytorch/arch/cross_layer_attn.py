# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from collections import OrderedDict
from typing import List, Optional
from dataclasses import dataclass

from sunyata.pytorch.arch.base import BaseCfg, ConvMixerLayer
from sunyata.pytorch_lightning.base import BaseModule

#  %%
class Attn(nn.Module):
    """
    Attention among layers.
    """
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.,
    ):
        super().__init__()
        self.scaler = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, groups=hidden_dim)
        self.temperature = temperature

    def forward(self, *prev_features):
        depth = len(prev_features)
        batch_size, hidden_dim, height, width = prev_features[0].shape
        prev_features = torch.cat(prev_features, dim=0)
        squeezed = self.scaler(prev_features)
        squeezed = squeezed.view(-1, batch_size, hidden_dim, height, width)
        attn = torch.einsum('b h v w, d b h v w -> d b v w', squeezed[-1], squeezed)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)
        prev_features = prev_features.view(-1, batch_size, hidden_dim, height, width)
        x_new = torch.einsum('d b h v w, d b v w -> b h v w', prev_features, attn)
        x_new = x_new * depth
        return x_new


# %%
class CLA2d(nn.Module):
    """
    Cross Layer Attention.
    """
    def __init__(
        self,
        hidden_dim: int,
        groups: Optional[int] = None,
        temperature: float = 1.,
        multiply_depth: bool = False,
        memory_efficient: bool = True,
    ):
        super().__init__()
        groups = hidden_dim if groups is None else groups
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.keys = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, groups=groups)
        self.query = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, groups=groups)
        self.temperature = temperature
        self.multiply_depth = multiply_depth
        self.memory_efficient = memory_efficient

    def attn_function(self, inputs: List[Tensor]) -> Tensor:
        stacked_features = torch.stack(inputs, dim=0)
        depth, batch_size, hidden_dim, height, width = stacked_features.shape
        concated_features = stacked_features.view(-1, *stacked_features.shape[2:])
        concated_features = self.pool(concated_features)
        keys = self.keys(concated_features)
        keys = keys.view(depth, batch_size, hidden_dim, 1, 1)
        query = self.query(concated_features[-batch_size:])
        attn = torch.einsum('b h v w, d b h v w -> d b v w', query, keys)
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)
        output = torch.einsum('d b h v w, d b v w -> b h v w', stacked_features, attn)
        if self.multiply_depth:
            output = output * depth
        return output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def call_checkpoint_attn(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.attn_function(inputs)

        return cp.checkpoint(closure, *input)

    def forward(self, prev_features: List[Tensor]) -> Tensor:
        if self.memory_efficient and self.any_requires_grad(prev_features):
            attn_output = self.call_checkpoint_attn(prev_features)
        else:
            attn_output = self.attn_function(prev_features)
        return attn_output

# %%
@dataclass
class CLAConvMixerCfg(BaseCfg):
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    temperature: float = 1.
    groups: Optional[int] = None
    multiply_depth: bool = False
    memory_efficient: bool = True

# %%
class CLAConvMixer(BaseModule):
    def __init__(self, cfg:CLAConvMixerCfg):
        super().__init__(cfg)
        
        self.cla_layers = nn.ModuleList([
            CLA2d(cfg.hidden_dim, cfg.groups, cfg.temperature, cfg.multiply_depth, cfg.memory_efficient)
            for _ in range(cfg.num_layers)
        ])

        self.convmixer_layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
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

    def forward(self, x):
        x0 = self.embed(x)
        x1 = self.convmixer_layers[0](x0)
        features = [x0, x1]
        for cla_layer, convmixer_layer in zip(self.cla_layers[:-1], self.convmixer_layers[1:]):
            new_features = cla_layer(features)
            new_features = convmixer_layer(new_features)
            features.append(new_features)
        final_features = self.cla_layers[-1](features)
        x = self.digup(final_features)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


# %%
if __name__ == "__main__":
    # %%
    batch_size, hidden_dim, height, width = 3, 8, 6, 6
    input1 = torch.randn(batch_size, hidden_dim, height, width) # (batch_size, hidden_dim, height, width)
    input2 = torch.randn(batch_size, hidden_dim, height, width) # (batch_size, hidden_dim, height, width)
    input = [input1, input2]
    # %%
    attn = Attn(hidden_dim, temperature=1.)
    # %%
    output = attn(*input)
    output.shape
    # %%
    cross_layer_attn = CLA2d(hidden_dim, temperature=1.)
    # %%
    output = cross_layer_attn(input)
    output.shape
    # %%
    cla_convmixer_cfg = CLAConvMixerCfg()
    # %%
    cla_convmixer = CLAConvMixer(cla_convmixer_cfg)
    # %%
    batch = torch.randn(2, 3, 16, 16)
    # %%
    logits = cla_convmixer(batch)
