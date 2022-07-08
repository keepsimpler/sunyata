"""
Transformer for Causal Language Modeling
"""

import copy
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from sunyata.pytorch.arch.base import BaseCfg, BaseModule
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration
from sunyata.pytorch.layer.transformer import TransformerLayer, TransformerLayerNoShortcut, TransformerLayerPostNorm, TransformerLayerPreNorm

from sunyata.pytorch.layer.transformer import TransformerCfg
from pytorch_lightning.callbacks import Callback


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


@dataclass
class TransformerCLMCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None
    
    transformer: TransformerCfg = None

    alpha: float = 1.
    student_temp: float = 0.9
    teacher_temp: float = 0.04
    ema_tau: float = 0.99
    center_tau: float = 0.99
    is_sharing_weight: bool = False
    is_last_norm: bool = False
    is_train_inner: bool = True
    

def loss_fn(
    teacher_logits,
    student_logits,
    target,
    teacher_temp,
    student_temp,
    teacher_centers,
    alpha,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim=-1)
    teacher_probs = ((teacher_logits - teacher_centers) / teacher_temp).softmax(dim=-1)
    consistency_loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim=-1).mean()

    student_logits = student_logits.log_softmax(dim=-1).permute(0, 2, 1)
    class_loss = F.nll_loss(student_logits, target)
    loss = alpha * class_loss + (1 - alpha) * consistency_loss
    return loss, class_loss, consistency_loss


class SelfDistillationCLM(BaseModule):
    def __init__(self, cfg:TransformerCLMCfg, model: type=TransformerLayer):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

        self.online_encoder = nn.Sequential(
            nn.Embedding(cfg.vocab_size, cfg.hidden_dim),
            nn.Sequential(*[model(cfg.transformer) for _ in range(cfg.num_layers)]),
            nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        )
        self.init_weights()

        if cfg.is_sharing_weight:
            self.online_encoder[2].weight = self.online_encoder[0].weight
        
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        for layer in self.target_encoder[1]:
            layer.attention.is_mask = False

        # self.class_loss = nn.CrossEntropyLoss()
        # self.consistence_loss = nn.KLDivLoss(log_target=False)

        self.register_buffer('teacher_centers', torch.zeros(1, 1, cfg.vocab_size))
        self.register_buffer('last_teacher_centers', torch.zeros(1, 1, cfg.vocab_size))

        self.cfg = cfg
        
    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.online_encoder[0].weight.data)
        torch.nn.init.xavier_normal_(self.online_encoder[2].weight.data)

    def forward(self, input):
        return self.online_encoder(input)
    
    def forward_target(self, target):
        with torch.no_grad():
            target_proj = self.target_encoder(target)
            teacher_centers = target_proj.mean(dim=(0,1), keepdim=True)
            self.last_teacher_centers.copy_(teacher_centers)
            # target_proj = target_proj - teacher_centers
            # target_proj = target_proj.exp()
            # target_proj.detach_()
        return target_proj

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        student_logits = self.forward(input)
        teacher_logits = self.forward_target(target)
        loss, class_loss, consistency_loss = loss_fn(teacher_logits, student_logits, target, self.cfg.teacher_temp, self.cfg.student_temp, self.teacher_centers, self.cfg.alpha)
        # consistence_loss = self.consistence_loss(student_logits / self.cfg.student_temp, teacher_logits / self.cfg.student_temp)
        self.log(mode + "_consistence_loss", consistency_loss)
        # student_logits = student_logits.permute(0, 2, 1)
        # class_loss = F.cross_entropy(student_logits, target)
        self.log(mode + "_class_loss", class_loss)
        # loss = self.cfg.alpha * class_loss + (1 - self.cfg.alpha) * consistence_loss
        self.log(mode + "_loss", loss)
        accuracy = (student_logits.permute(0, 2, 1).argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss


class TransformerCLMBayes(BaseModule):
    """
    Transformer for Causal Language Modeling.    
    """
    def __init__(self, cfg: TransformerCLMCfg, model: type=TransformerLayer):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.layers = nn.Sequential(*[model(cfg.transformer) for _ in range(cfg.num_layers)])

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.digup.weight = self.embed.weight

        log_prior = torch.zeros(1, 1, cfg.vocab_size)
        self.register_buffer('log_prior', log_prior)
        
        self.cfg = cfg
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        log_prior = repeat(self.log_prior, '1 1 n -> b s n', b=batch_size, s=seq_len)

        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)
            logits = self.digup(x)
            log_prior = log_bayesian_iteration(log_prior, logits)

        return log_prior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        log_posterior = log_posterior.permute(0, 2, 1)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (log_posterior.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


class TransformerCLM(BaseModule):
    """
    Transformer for Causal Language Modeling.    
    """
    def __init__(self, cfg: TransformerCLMCfg, model: type=TransformerLayer):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.layers = nn.Sequential(*[model(cfg.transformer) for _ in range(cfg.num_layers)])

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.init_weights()

        if cfg.is_sharing_weight:
            self.digup.weight = self.embed.weight
        
        self.cfg = cfg
        
    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.digup(x)
        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


class TransformerCLMSplit(TransformerCLM):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.layers.parameters(), 'lr': 0.},
            {'params': self.digup.parameters(), 'lr': 1e-3},
            # {'params': self.embed.parameters(), 'lr': 0.}
        ], lr=self.cfg.learning_rate)
        return optimizer


class AdjustLR(Callback):
    # def on_train_epoch_end(self, trainer, pl_module):
    #     print(trainer.current_epoch)
    #     optimizer = pl_module.optimizers()
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         if i == 0:
    #             param_group['lr'] = 1e-3 if trainer.current_epoch % 2 == 0 else 0.
    #         if i == 1:
    #             param_group['lr'] = 0. if trainer.current_epoch % 2 == 0 else 1e-3

    def on_train_batch_end(
        self, 
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        optimizer = pl_module.optimizers()
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                param_group['lr'] = 1e-3 if batch_idx % 2 == 0 else 0.
            if i == 1:
                param_group['lr'] = 0. if batch_idx % 2 == 0 else 1e-3
    

class TransformerCLMNoShortcut(TransformerCLM):
    def __init__(self, cfg:TransformerCLMCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[TransformerLayerNoShortcut(cfg.transformer) for _ in range(cfg.num_layers)])

    def forward(self, x):
        h = self.embed(x)
        h2 = h
        for layer in self.layers:
            h1, h2 = layer(h2)
            h = h + h1 + h2
        logits = self.digup(h)
        return logits

