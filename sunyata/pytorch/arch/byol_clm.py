import math
import copy
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer

from sunyata.pytorch.arch.base import BaseCfg, BaseModule
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss function

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
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

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        self.online_encoder = nn.Sequential(
            # nn.Embedding(cfg.vocab_size, cfg.hidden_dim),
            nn.Sequential(*[
                TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
            ]),
            BatchNorm(cfg.hidden_dim)
        )
        torch.nn.init.xavier_normal_(self.embed.weight.data)  # online_encoder[0]

        self.target_encoder = copy.deepcopy(self.online_encoder)
        # self.target_encoder.eval()
        set_requires_grad(self.target_encoder, False)

        for layer in self.target_encoder[0]:
            layer.attention.is_mask = False

        # self.online_predictor = TransformerLayer(cfg.transformer)
        self.online_predictor = BatchNorm(cfg.hidden_dim)

        self.loss_fn = loss_fn
        self.cfg = cfg

    def forward(self, input, target):
        input = self.embed(input)
        online_proj = self.online_encoder(input)
        online_pred = self.online_predictor(online_proj)

        with torch.no_grad():
            target = self.embed(target)
            target_proj = self.target_encoder(target)
            target_proj.detach_()

        return online_pred, target_proj

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        online_pred, target_proj = self.forward(input, target)
        loss = self.loss_fn(online_pred, target_proj).mean()
        self.log(mode + "_loss", loss)
        return loss


class BatchNorm(nn.Module):
    def __init__(self, hidden_dim, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(1, hidden_dim))
        self.adds = nn.Parameter(torch.zeros(1, hidden_dim))
        self.register_buffer('vars', torch.ones(1, 1, hidden_dim))
        self.register_buffer('means', torch.zeros(1, 1, hidden_dim))

    def update_stats(self, x):
        # x has dims (batch_size, seq_len, hidden_dim)
        m = x.mean((0,1), keepdim=True)
        v = x.var((0,1), keepdim=True)
        self.means.lerp_(m, self.mom)
        self.vars.lerp_(v, self.mom)
        return m, v

    def forward(self, x):
        if self.training:
            with torch.no_grad(): m, v = self.update_stats(x)
        else: m,v = self.means, self.vars
        x = (x-m) / (v+self.eps).sqrt()
        return x * self.mults + self.adds


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