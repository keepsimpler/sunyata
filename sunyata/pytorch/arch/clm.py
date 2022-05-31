"""
Transformer for Causal Language Modeling
"""

from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.layer.transformer import TransformerLayer, TransformerLayerNoShortcut

from sunyata.pytorch.layer.transformer import TransformerCfg


@dataclass
class TransformerCLMCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None
    
    transformer: TransformerCfg = None

    is_sharing_weight: bool = False
    

class TransformerCLM(pl.LightningModule):
    """
    Transformer for Causal Language Modeling.    
    """
    def __init__(self, cfg: TransformerCLMCfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.layers = nn.Sequential(*[TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)])

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.init_weights()

        if cfg.is_sharing_weight:
            self.digup.weight = self.embed.weight
        
        self.vocab_size, self.hidden_dim, self.learning_rate = cfg.vocab_size, cfg.hidden_dim, cfg.learning_rate
        
    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h)

        logits = self.digup(h)
        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


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

