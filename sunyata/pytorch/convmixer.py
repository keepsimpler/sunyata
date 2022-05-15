from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl

from sunyata.pytorch.bayes.core import log_bayesian_iteration


@dataclass
class DeepBayesInferConvMixerCfg:
    hidden_dim: int = 256
    num_layers: int = 8
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    is_bayes: bool = True
    is_prior_as_params: bool = False

    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    optimizer_method: str = "Adam"  # or "AdamW"
    learning_rate_scheduler: str = "CosineAnnealing"
    weight_decay: float = None  # of "AdamW"


class DeepBayesInferConvMixer(pl.LightningModule):
    def __init__(self, cfg:DeepBayesInferConvMixerCfg):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(cfg.hidden_dim)
                )),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])
        if not cfg.is_bayes:
            self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to one layer

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

        log_prior = torch.zeros(1, cfg.num_classes)
        if cfg.is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior) 

        self.cfg = cfg

    def forward(self, x)        :
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)

        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            logits = self.digup(x)   # / 5 / (i+1)
            log_posterior = log_bayesian_iteration(log_prior, logits)
            # self.log("log_posterior", log_posterior)
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
        

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


