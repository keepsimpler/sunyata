from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration


@dataclass
class TransformerCLMBayesCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None

    ff_act_nn: nn.Module = nn.GELU()

    attn_scale: float = None


class BayesTransformerLayer(nn.Module):
    def __init__(self, cfg:TransformerCLMBayesCfg):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            cfg.ff_act_nn
        )

        self.to_k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.to_q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        # self.to_v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.scale = cfg.hidden_dim ** -0.5 if cfg.attn_scale is None else cfg.attn_scale


    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        x = self.ff(x)
        k = self.to_k(x)
        q = self.to_q(x)
        # v = self.to_v(x)
        dots = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype)
        # attn_mask = torch.tril(attn_mask)
        # dots = dots * attn_mask

        # x = torch.einsum('b i j, b j v -> b i v', dots, v)

        return dots, x


class TransformerCLMBayes(pl.LightningModule):
    def __init__(self, cfg:TransformerCLMBayesCfg):
        super().__init__()
        self.save_hyperparameters("cfg")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        # nn.init.zeros_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        # nn.init.zeros_(self.digup.weight.data)

        self.layers = nn.ModuleList([BayesTransformerLayer(cfg) for _ in range(cfg.num_layers)])

        # self.prior = nn.Parameter(torch.ones(1, 1, cfg.vocab_size) / cfg.vocab_size)  # cfg.seq_len - 1
        # self.prior = nn.Parameter(F.gumbel_softmax(torch.zeros(1, 1, cfg.vocab_size), tau=10, dim=-1))

        self.cfg = cfg

    def forward(self, x):
        batch_size, seq_len = x.shape
        # prior = F.softmax(self.prior, dim=-1)
        # prior = repeat(prior, '1 1 v -> b s v', b=batch_size, s=seq_len)        
        prior = torch.ones_like(x).unsqueeze(-1).repeat((1, 1, self.cfg.vocab_size)) / self.cfg.vocab_size
        # log_prior = torch.zeros_like(x, dtype=torch.float).unsqueeze(-1).repeat((1, 1, self.cfg.vocab_size))
        # prior = F.gumbel_softmax(logits, tau=2, dim=-1)

        x = self.embed(x)

        for layer in self.layers:
            dots, x = layer(x)
            attn_mask = torch.full((seq_len, seq_len), -float('Inf'), device=x.device, dtype=x.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            dots = dots + attn_mask
            logits = self.digup(x)
            log_posterior = log_bayesian_iteration(torch.log(prior), logits)
            posterior = torch.exp(log_posterior)
            trans_matrix = F.softmax(dots, dim=-1)
            posterior = torch.einsum('b i j, b j v -> b i v', trans_matrix, posterior)
            prior = posterior

        return posterior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        posterior = self.forward(input)
        posterior = posterior.permute(0, 2, 1)
        log_posterior = torch.log(posterior)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=1) == target).float().mean()
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


    