from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.bayes import log_bayesian_iteration


@dataclass
class TextConvBayesCfg:                                            
    hidden_dim: int = 64
    vocab_size: int = 1000
    seq_len: int = 128
    batch_size: int = 16

    kernel_size: int = 3

    num_layers: int = 8

    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = None
    optimizer_method: str = "Adam"
    learning_rate_scheduler: str = "CosineAnnealing"


class TextConvBayes(pl.LightningModule):
    def __init__(self, cfg: TextConvBayesCfg):
        super().__init__()
        self.save_hyperparameters("cfg")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        nn.init.zeros_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        # nn.init.zeros_(self.digup.weight.data)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Sequential(
                    Conv1dWithLeftPad(cfg.hidden_dim, cfg.kernel_size),
                    nn.GELU(),
                    nn.BatchNorm1d(cfg.hidden_dim),  # LayerNorm1d nn.GroupNorm(1, cfg.hidden_dim) nn.InstanceNorm1d(cfg.hidden_dim, affine=True)
                    nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm1d(cfg.hidden_dim)
                )
            ) for _ in range(cfg.num_layers)
        ])

        self.cfg = cfg

    def forward(self, x):
        log_prior = torch.zeros_like(x).unsqueeze(-1).repeat((1, 1, self.cfg.vocab_size))

        x = self.embed(x)
        x = x.permute(0, 2, 1)
        logits = x
        for layer in self.layers:
            logits = layer(logits)
            # x = x + logits

            chosen = logits.permute(0, 2, 1)
            chosen = self.digup(chosen)
            log_posterior = log_bayesian_iteration(log_prior, chosen)
            log_prior = log_posterior

        return log_posterior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        logits = logits.permute(0, 2, 1)
        loss = F.nll_loss(logits, target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
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



class Conv1dWithLeftPad(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim)
        self.kernel_size = kernel_size

    def forward(self, x):
        return self.conv1d(F.pad(x, (self.kernel_size - 1, 0)))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) # + x

