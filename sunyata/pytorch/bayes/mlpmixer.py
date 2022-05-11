from dataclasses import dataclass
from functools import partial
from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from sunyata.pytorch.bayes.core import log_bayesian_iteration

pair = lambda x: x if isinstance(x, tuple) else (x, x)

@dataclass
class DeepBayesInferMlpMixerCfg:
    image_size: int = 32  # 224
    patch_size: int = 4  # 16
    hidden_dim: int = 128
    expansion_factor: int = 4
    expansion_factor_token: float = 0.5

    num_layers: int = 8
    num_classes: int = 10
    channels: int = 3
    dropout: float = 0. 

    is_bayes: bool = True
    is_prior_as_params: bool =False



class DeepBayesInferMlpMixer(pl.LightningModule):
    def __init__(self, cfg:DeepBayesInferMlpMixerCfg):
        super().__init__()

        image_h, image_w = pair(cfg.image_size)
        assert (image_h % cfg.patch_size) == 0 and (image_w % cfg.patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // cfg.patch_size) * (image_w // cfg.patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.layers = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(cfg.hidden_dim, FeedForward(num_patches, cfg.expansion_factor, cfg.dropout, chan_first)),
                PreNormResidual(cfg.hidden_dim, FeedForward(cfg.hidden_dim, cfg.expansion_factor_token, cfg.dropout, chan_last))
            ) for _ in range(cfg.num_layers)]
        )
        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=cfg.patch_size, p2=cfg.patch_size),
            nn.Linear((cfg.patch_size ** 2) * cfg.channels, cfg.hidden_dim)
        )
        self.digup = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.is_bayes = cfg.is_bayes

        log_prior = torch.zeros(1, cfg.num_classes)
        if cfg.is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior)        
    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        if self.is_bayes:
            for layer in self.layers:
                x = layer(x)
                logits = self.digup(x)
                log_posterior = log_bayesian_iteration(log_prior, logits)
                log_prior = log_posterior
        else:
            x = self.layers(x)
            logits = self.digup(x)
            log_posterior = log_bayesian_iteration(log_prior, logits)

        return log_posterior

    def training_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("train_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("val_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )