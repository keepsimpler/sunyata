from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Residual
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration
from sunyata.pytorch.arch.textconv import Conv1dWithLeftPad
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


@dataclass
class ConvTransCfg(BaseCfg):                                            
    hidden_dim: int = 64
    vocab_size: int = 1000
    seq_len: int = 128

    kernel_size: int = 3

    # LayerNorm1d nn.GroupNorm(1, cfg.hidden_dim) nn.InstanceNorm1d(cfg.hidden_dim, affine=True) nn.BatchNorm1d
    norm_layer: nn.Module =  nn.BatchNorm1d

    transformer: TransformerCfg = None


class ConvTransCLM(BaseModule):
    """
    Residual Convolution Neural Network for Causal Language Modeling
    """
    def __init__(self, cfg: ConvTransCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        # nn.init.zeros_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        self.conv_layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    Conv1dWithLeftPad(cfg.hidden_dim, cfg.kernel_size),
                    nn.GELU(),
                    cfg.norm_layer(cfg.hidden_dim)
                )),
                Residual(nn.Sequential(
                    nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim*4, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(cfg.hidden_dim*4, cfg.hidden_dim, kernel_size=1),
                    cfg.norm_layer(cfg.hidden_dim)
                ))
            ) for _ in range(cfg.num_layers//2)
        ])

        self.trans_layers = nn.Sequential(*[TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers//2)])

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = self.trans_layers(x)
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


