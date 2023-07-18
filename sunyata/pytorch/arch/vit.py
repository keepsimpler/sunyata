from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch_lightning.base import BaseModule


from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration
from sunyata.pytorch.layer.transformer import (
    TransformerCfg,
    TransformerLayer, 
    TransformerLayerPreNorm,
    TransformerLayerPostNorm,
    )

@dataclass
class ViTCfg(BaseCfg):
    transformer: TransformerCfg = TransformerCfg(hidden_dim=128)

    hidden_dim: int = 128
    image_size: int = 224
    patch_size: int = 7
    num_classes: int = 100

    pool: str = 'mean' # or 'cls'

    emb_dropout: float = 0. 


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
        patch_dim = 3 * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        assert cfg.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = cfg.pool

        if self.pool == 'cls':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.hidden_dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        elif self.pool == 'mean':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, cfg.hidden_dim))

        self.emb_dropout = nn.Dropout(cfg.emb_dropout)
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        self.cfg = cfg

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding

        x = self.emb_dropout(x)

        x = self.layers(x)

        x = self.final_ln(x)
        x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        logits = self.mlp_head(x_chosen)

        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


class ViTPreNorm(ViT):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            TransformerLayerPreNorm(cfg.transformer) for _ in range(cfg.num_layers)
        ])


class ViTPostNorm(ViT):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            TransformerLayerPostNorm(cfg.transformer) for _ in range(cfg.num_layers)
        ])


class IterViTPreNorm(ViTPreNorm):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding

        x = self.emb_dropout(x)

        logits = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        for layer in self.layers:
            x = layer(x)
            x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            logits = logits + x_chosen
            logits = self.final_ln(logits)

        logits = self.mlp_head(logits)

        return logits

    

class BayesViT(ViT):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        log_prior = torch.zeros(1, cfg.num_classes)
        self.register_buffer('log_prior', log_prior) 

    def forward(self, img):
        batch_size, _, _, _ = img.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x)

            x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

            logits = self.mlp_head(x_chosen)

            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

