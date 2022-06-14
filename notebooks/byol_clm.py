import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


def loss_fn(x: torch.Tensor, y: torch.Tensor):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


@dataclass
class BYOL_CLM_Cfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None

    ema_tau: float = 0.999

    transformer: TransformerCfg = None


class BYOL_CLM(BaseModule):
    def __init__(self, cfg:BYOL_CLM_Cfg):
        super().__init__(cfg)

        self.online_encoder = nn.Sequential(
            nn.Embedding(cfg.vocab_size, cfg.hidden_dim),
            nn.Sequential(*[
                TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
            ])
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.train(False)

        self.online_predictor = TransformerLayer(cfg.transformer)

        self.loss_fn = loss_fn
        self.cfg = cfg

    def forward(self, input, target):
        online_proj = self.online_encoder(input)
        online_pred = self.online_predictor(online_proj)

        with torch.no_grad():
            target_proj = self.target_encoder(target)
            target_proj.detach_()

        loss = self.loss_fn(online_pred, target_proj)

        return loss.mean()

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        loss = self.forward(input, target)
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val") 

