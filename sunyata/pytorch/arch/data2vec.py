import math
import copy
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer

from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch_lightning.base import BaseModule
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


@dataclass
class Data2VecCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None

    k : int = 4
    ema_tau: float = 0.999
    smooth_l1_loss_beta = 1.
    normalize_targets: bool = True
   
    transformer: TransformerCfg = None    


class Encoder(nn.Module):
    def __init__(self, cfg: Data2VecCfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        torch.nn.init.xavier_normal(self.embed.weight.data)
        self.transformers = nn.ModuleList([
            TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
        ])
    def forward(self, x):
        x = self.embed(x)
        outputs = [transformer(x) for transformer in self.transformers]
        return outputs


class Data2VecCLM(BaseModule):
    def __init__(self, cfg:Data2VecCfg):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')

        self.online_encoder = Encoder(cfg)

        self.online_predictor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        # self.target_encoder.eval()
        set_requires_grad(self.target_encoder, False)

        for layer in self.target_encoder.transformers:
            layer.attention.is_mask = False

        # self.online_predictor = TransformerLayer(cfg.transformer)

        self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=cfg.smooth_l1_loss_beta)
        self.cfg = cfg

    def forward(self, input, target):
        online_proj = self.online_encoder(input)[-1]
        online_pred = self.online_predictor(online_proj)

        with torch.no_grad():
            target_proj = self.target_encoder(target)[:-1]
            target_proj = target_proj[-self.cfg.k:]  # take the last k transformer layers
            target_proj = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_proj]
            target_proj = sum(target_proj) / len(target_proj)
            if self.cfg.normalize_targets:
                target_proj = F.layer_norm(target_proj.float(), target_proj.shape[-1:])
            target_proj.detach_()

        loss = self.loss_fn(online_pred.float(), target_proj.float()).sum(dim=-1).sum().div(online_pred.size(0))
        return loss

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        loss = self.forward(input, target)
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")
   

class BYOL_EMA(Callback):
    def __init__(self, initial_tau: float=0.999, do_tau_update: bool=True):
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau
        self.do_tau_update = do_tau_update

    def on_train_batch_end(
        self, 
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        online_encoder = pl_module.online_encoder
        target_encoder = pl_module.target_encoder

        self.update_weights(online_encoder, target_encoder)

        if self.do_tau_update:
            self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_encoder: nn.Module, target_encoder: nn.Module):
        for online_params, target_params in zip(online_encoder.parameters(), target_encoder.parameters()):
            target_params.data = self.current_tau * target_params.data + (1 - self.current_tau) * online_params.data