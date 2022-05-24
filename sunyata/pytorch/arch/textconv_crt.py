from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextConvCrtCfg:                                            
    r1: int = 122
    r2: int = 121
    expansion: int = 1
    hidden_dim: int = (r1 + r2) * expansion

    vocab_size: int = 1000
    seq_len: int = 128
    batch_size: int = 16

    kernel_size: int = 3

    num_layers: int = 6

    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = None
    optimizer_method: str = "Adam"
    learning_rate_scheduler: str = "CosineAnnealing"


class TextConvCrt(pl.LightningModule):
    def __init__(self, cfg: TextConvCrtCfg):
        super().__init__()
        self.save_hyperparameters("cfg")

        cfg.hidden_dim = (cfg.r1 + cfg.r2) * cfg.expansion
        self.embed_r1 = nn.Embedding(cfg.r1, cfg.r1 * cfg.expansion)
        self.embed_r2 = nn.Embedding(cfg.r2, cfg.r2 * cfg.expansion)
        self.digup_r1 = nn.Linear(cfg.hidden_dim, cfg.r1)
        self.digup_r2 = nn.Linear(cfg.hidden_dim, cfg.r2)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    Conv1dThenLeftPad(cfg.hidden_dim, cfg.kernel_size),
                    nn.GELU(),
                    nn.BatchNorm1d(cfg.hidden_dim)  # LayerNorm1d nn.GroupNorm(1, cfg.hidden_dim) nn.InstanceNorm1d(cfg.hidden_dim, affine=True)
                )),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])

        self.cfg = cfg

    def forward(self, x_r1, x_r2):
        x_r1 = self.embed_r1(x_r1)
        x_r2 = self.embed_r2(x_r2)
        x = torch.cat((x_r1, x_r2), dim=-1)
        x = x.permute(0, 2, 1)
        x1 = x
        for layer in self.layers:
            x1 = layer(x1)
            x = x + x1
        # x = self.layers(x)
        x = x.permute(0, 2, 1)
        x_r1 = self.digup_r1(x)  #  x[:, :, :self.cfg.r1]
        x_r2 = self.digup_r2(x)  # x[:, :, self.cfg.r1:]
        return x_r1, x_r2

    def _step(self, batch, mode="train"):  # or "val"
        (input_row, input_col), target = batch
        logits_row, logits_col = self.forward(input_row, input_col)
        probs_row = torch.softmax(logits_row, dim=-1)
        probs_col = torch.softmax(logits_col, dim=-1)
        probs = torch.einsum('b s p, b s q -> b s p q', probs_row, probs_col)
        probs = probs.reshape(*probs.shape[:2], -1)
        log_probs = torch.log(probs)
        log_probs = log_probs.permute(0, 2, 1)
        loss = F.nll_loss(log_probs, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_probs.argmax(dim=1) == target).float().mean()
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


class LayerNorm1d(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x


class Conv1dThenLeftPad(nn.Module):
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

