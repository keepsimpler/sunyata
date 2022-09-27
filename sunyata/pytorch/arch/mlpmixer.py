from dataclasses import dataclass
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

@dataclass
class MlpMixerCfg:
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

    num_epochs: int = 10
    learning_rate: float = 1e-3
    optimizer_method: str = "Adam"  # or "AdamW"
    learning_rate_scheduler: str = "CosineAnnealing"
    weight_decay: float = None  # of "AdamW"



class DeepBayesInferMlpMixer(pl.LightningModule):
    def __init__(self, cfg:MlpMixerCfg):
        super().__init__()

        image_h, image_w = pair(cfg.image_size)
        assert (image_h % cfg.patch_size) == 0 and (image_w % cfg.patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // cfg.patch_size) * (image_w // cfg.patch_size)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.layers = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(cfg.hidden_dim, FeedForward(num_patches, cfg.expansion_factor, cfg.dropout, chan_first)),
                PreNormResidual(cfg.hidden_dim, FeedForward(cfg.hidden_dim, cfg.expansion_factor_token, cfg.dropout, chan_last))
            ) for _ in range(cfg.num_layers)
        ])
        if not cfg.is_bayes:
            self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to one layer

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

        self.cfg = cfg  
    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x)
            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only support Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
        


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