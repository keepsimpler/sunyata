from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, BaseModule, set_requires_grad
from sunyata.pytorch.arch.loss import InfoNCE, ECELoss
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer


@dataclass
class ContrastiveCLMCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None
    temperature: float = None
    alpha: float = None
    transformer: TransformerCfg = None


class ContrastiveCLM(BaseModule):
    def __init__(self, cfg:ContrastiveCLMCfg):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        torch.nn.init.xavier_normal_(self.embed.weight.data)

        self.layers = nn.Sequential(
            nn.Sequential(*[
                TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
            ]),
        )
        if cfg.temperature is None:
            self.temperature = cfg.hidden_dim ** 0.5
        else:
            self.temperature = cfg.temperature
        self.loss_fn = InfoNCE(temperature=self.temperature)
        self.ece_loss = ECELoss()

        self.cfg = cfg

    def forward(self, input, target):
        input_embedded = self.embed(input)
        output_embedded = self.layers(input_embedded)

        target_embedded = self.embed(target)

        return output_embedded, target_embedded

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)
        loss = self.loss_fn(output_embedded, target_embedded)
        self.log(mode + "_loss", loss)
        cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        self.log(mode + "cosine_loss", cosine_loss, prog_bar=True)
        logits = output_embedded @ self.embed.weight.T
        class_loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("train_class_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)
        logits = output_embedded @ self.embed.weight.T
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        ece_loss, avg_confidence, avg_accuracy = self.ece_loss(logits, target)
        self.log("ece_loss", ece_loss, on_step=True, on_epoch=True)
        self.log("avg_confidence", avg_confidence, on_step=True, on_epoch=True)
        self.log("avg_accuracy", avg_accuracy, on_step=True, on_epoch=True)



class ContrastiveCLMInBatch(ContrastiveCLM):
    """With Random negative samples in Batch"""
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)

        batch_size, seq_len, hidden_dim = output_embedded.shape
        shuffled_batch_idx = torch.randperm(batch_size, device=output_embedded.device)
        target_embedded_shuffled = target_embedded[shuffled_batch_idx, :, :]  # .index_select(0, shuffled_batch_idx)
        
        logits = torch.einsum('b s n, b t n -> b s t', output_embedded, target_embedded_shuffled)
        logits_diagonal = torch.einsum('b s n, b s n -> b s', output_embedded, target_embedded)
        logits = logits.diagonal_scatter(logits_diagonal, dim1=1, dim2=2)
        logits_diagonal /= self.temperature
        labels = torch.arange(0, seq_len, dtype=torch.long, device=output_embedded.device).repeat(batch_size).reshape(batch_size, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log(mode + "_loss", loss)
        cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        self.log(mode + "cosine_loss", cosine_loss)
        return loss


class ContrastiveCLMRandom(ContrastiveCLM):
    """With Random negative samples in the whole vocabulary"""
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)

        batch_size, seq_len, hidden_dim = output_embedded.shape
        random_target = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len), device=output_embedded.device)
        random_target_embedded = self.embed(random_target)

        logits = torch.einsum('b s n, b t n -> b s t', output_embedded, random_target_embedded)
        logits_diagonal = torch.einsum('b s n, b s n -> b s', output_embedded, target_embedded)
        logits = logits.diagonal_scatter(logits_diagonal, dim1=1, dim2=2)
        logits_diagonal /= self.temperature
        labels = torch.arange(0, seq_len, dtype=torch.long, device=output_embedded.device).repeat(batch_size).reshape(batch_size, seq_len)
        loss = F.cross_entropy(logits, labels)
        self.log(mode + "_loss", loss)
        cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        self.log(mode + "cosine_loss", cosine_loss)
        return loss
