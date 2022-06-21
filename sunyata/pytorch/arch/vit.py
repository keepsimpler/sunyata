from dataclasses import dataclass
from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
from einops import repeat
from einops.layers.torch import Rearrange
import warmup_scheduler
from sunyata.pytorch.arch.base import BaseCfg, BaseModule

from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration, DeepBayesInferCfg
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

@dataclass
class ViTCfg(BaseCfg):
    hidden_dim: int = 128
    image_size: int = 64  # 224
    patch_size: int = 8  # 16
    num_classes: int = 200

    transformer: TransformerCfg = None    

    pool: str = 'cls' # or 'mean'
    channels: int = 3

    emb_dropout: float = 0. 

    warmup_epoch: int = 5


class ViT(BaseModule):
    def __init__(self, cfg: ViTCfg):
        super().__init__(cfg)

        self.save_hyperparameters("cfg")

        self.layers = nn.Sequential(*[
            TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
        ])

        image_height, image_width = pair(cfg.image_size)
        patch_height, patch_width = pair(cfg.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width
        assert cfg.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, cfg.hidden_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)

        self.pool = cfg.pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        self.cfg = cfg

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.dropout(x)

        x = self.layers(x)

        x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        logits = self.mlp_head(x_chosen)

        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

