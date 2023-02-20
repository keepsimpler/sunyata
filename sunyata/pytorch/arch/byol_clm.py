import math
import copy
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer

from sunyata.pytorch.arch.base import BaseCfg, set_requires_grad
from sunyata.pytorch_lightning.base import BaseModule

from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


# loss function

def loss_fn(x, y):
    # x = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def across_loss_fn(x: torch.Tensor, y: torch.Tensor):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    across_cosine_loss = torch.einsum('b s h, b t h -> b s t', x, y)
    max_loss = torch.diagonal(across_cosine_loss, 0, dim1=1, dim2=2)
    min_loss = across_cosine_loss.clone()
    min_loss.diagonal(dim1=-1, dim2=-2).zero_()
    # loss = (min_loss.sum() - max_loss.sum()) / (torch.numel(cosine_loss))
    loss = 2 + min_loss.mean() - max_loss.mean()    


@dataclass
class BYOL_CLM_Cfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None

    ema_tau: float = 0.999
    smooth_l1_loss_beta = 1.
   
    transformer: TransformerCfg = None    

    
class BYOL_CLM(BaseModule):
    def __init__(self, cfg:BYOL_CLM_Cfg):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')

        # self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        self.online_encoder = nn.Sequential(
            nn.Embedding(cfg.vocab_size, cfg.hidden_dim),
            nn.Sequential(*[
                TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
            ]),
            # BatchNorm(cfg.hidden_dim)
        )
        torch.nn.init.xavier_normal_(self.online_encoder[0].weight.data)  # online_encoder[0]

        self.target_encoder = copy.deepcopy(self.online_encoder)
        # self.target_encoder.eval()
        set_requires_grad(self.target_encoder, False)

        for layer in self.target_encoder[1]:
            layer.attention.is_mask = False

        self.online_predictor = TransformerLayer(cfg.transformer)
        # self.online_predictor = BatchNorm(cfg.hidden_dim)

        self.loss_fn = loss_fn  # infoNCE
        self.cfg = cfg

    def forward(self, input, target):
        # input = self.embed(input)
        online_proj = self.online_encoder(input)
        online_pred = self.online_predictor(online_proj)

        with torch.no_grad():
            # target = self.embed(target)
            target_proj = self.target_encoder(target)
            target_proj.detach_()

        return online_pred, target_proj

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        online_pred, target_proj = self.forward(input, target)
        loss = self.loss_fn(online_pred, target_proj).mean()  # sum()
        self.log(mode + "_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        online_pred, target_proj = self.forward(input, target)
        logits = online_pred @ self.online_encoder[0].weight.T
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)



class BYOL_EMA(Callback):
    def __init__(self, initial_ema_tau: float=0.99, initial_center_tau: float=0.99, do_ema_tau_update: bool=True):
        super().__init__()
        self.initial_ema_tau = initial_ema_tau
        self.current_ema_tau = initial_ema_tau
        self.do_ema_tau_update = do_ema_tau_update
        self.initial_center_tau = initial_center_tau

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

        if self.do_ema_tau_update:
            self.current_ema_tau = self.update_tau(pl_module, trainer)

        old_teacher_centers = pl_module.teacher_centers
        last_teacher_centers = pl_module.last_teacher_centers
        new_teacher_centers = self.initial_center_tau * old_teacher_centers + (1 - self.initial_center_tau) * last_teacher_centers
        pl_module.teacher_centers.copy_(new_teacher_centers)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_ema_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_encoder: nn.Module, target_encoder: nn.Module):
        for online_params, target_params in zip(online_encoder.parameters(), target_encoder.parameters()):
            target_params.data = self.current_ema_tau * target_params.data + (1 - self.current_ema_tau) * online_params.data