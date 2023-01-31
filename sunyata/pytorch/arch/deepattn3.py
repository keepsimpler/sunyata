"""
Squeeze of the current x as keys and queries. Initialize empty tensor to save all xs.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        squeezed = self.squeeze(x)
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


class AttnNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        kernel_size: int,
        drop_rate: float = 0.,
        temperature: float = 1.,
        init_scale: float = 1.,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvMixerLayer(hidden_dim, kernel_size, drop_rate)
            for _ in range(num_layers)
        ])
        self.squeezes = nn.ModuleList([
            Squeeze(hidden_dim, init_scale)
            for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            Attn(temperature)
            for _ in range(num_layers)
        ])
        self.first_squeeze = Squeeze(hidden_dim, init_scale)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor):
        batch_size, hidden_dim, height, width = x.shape
        # all_output = torch.empty(
        #     self.num_layers + 1, batch_size, hidden_dim, height, width,
        #     dtype=x.dtype,
        #     device=x.device
        # )
        all_output = x.unsqueeze(0)
        all_squeezed = self.first_squeeze(x).unsqueeze(0)
        next_x = x
        for i, (block, squeeze, attn) in enumerate(zip(self.blocks, self.squeezes, self.attentions)):
            next_output = block(next_x)
            all_output = torch.cat([all_output, next_output.unsqueeze(0)])
            squeezed = squeeze(next_output)
            all_squeezed = torch.cat([all_squeezed, squeezed.unsqueeze(0)])
            attended = attn(query = squeezed, keys = all_squeezed)
            next_x = torch.einsum('d b h v w, d b -> b h v w', all_output, attended)

        return next_x


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.    
    temperature: float = 1.
    init_scale: float = 1.


class DeepAttn(BaseModule):
    def __init__(self, cfg: DeepAttnCfg):
        super().__init__(cfg)

        self.attn_net = AttnNet(cfg.num_layers, cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate, cfg.temperature, cfg.init_scale)

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
        x = self.attn_net(x)
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
