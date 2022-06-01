from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Residual
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


@dataclass
class TextConvCfg(BaseCfg):                                            
    hidden_dim: int = 64
    vocab_size: int = 1000
    seq_len: int = 128

    kernel_size: int = 3

    # LayerNorm1d nn.GroupNorm(1, cfg.hidden_dim) nn.InstanceNorm1d(cfg.hidden_dim, affine=True)
    norm_layer: nn.Module = nn.BatchNorm1d 


class ResConvCLM(BaseModule):
    """
    Residual Convolution Neural Network for Causal Language Modeling
    """
    def __init__(self, cfg: TextConvCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        # nn.init.zeros_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    Conv1dWithLeftPad(cfg.hidden_dim, cfg.kernel_size),
                    nn.GELU(),
                    nn.BatchNorm1d(cfg.hidden_dim)
                )),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)
        x = self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss


class SumConvCLM(ResConvCLM):
    """
    Summed Convolution Neural Network for Causal Language Modeling
    """
    def __init__(self, cfg:TextConvCfg):
        super().__init__(cfg)
        # nn.init.zeros_(self.embed.weight.data)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                Conv1dWithLeftPad(cfg.hidden_dim, cfg.kernel_size),
                nn.GELU(),
                nn.BatchNorm1d(cfg.hidden_dim),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x1 = x
        for layer in self.layers:
            x1 = layer(x1)
            x = x + x1
        x = x.permute(0, 2, 1)
        x = self.digup(x)
        return x


class BayesConvCLM(ResConvCLM):
    """
    Bayesian Convolution Neural Network for Causal Language Modeling
    """
    def __init__(self, cfg:TextConvCfg):
        super().__init__(cfg)

        nn.init.zeros_(self.embed.weight.data)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Sequential(
                    Conv1dWithLeftPad(cfg.hidden_dim, cfg.kernel_size),
                    nn.GELU(),
                    # nn.BatchNorm1d(cfg.hidden_dim),
                    nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                    nn.GELU(),
                    # nn.BatchNorm1d(cfg.hidden_dim),
                )
            ) for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        log_prior = torch.zeros_like(x).unsqueeze(-1).repeat((1, 1, self.cfg.vocab_size))

        x = self.embed(x)
        # chosen = self.digup(x)
        # log_prior = log_bayesian_iteration(log_prior, chosen)

        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)

            chosen = x.permute(0, 2, 1)
            logits = self.digup(chosen)
            log_prior = log_bayesian_iteration(log_prior, logits)

        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        logits = logits.permute(0, 2, 1)
        loss = F.nll_loss(logits, target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss


class Conv1dWithLeftPad(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim)
        self.kernel_size = kernel_size

    def forward(self, x):
        return self.conv1d(F.pad(x, (self.kernel_size - 1, 0)))


class LayerNorm1d(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x
