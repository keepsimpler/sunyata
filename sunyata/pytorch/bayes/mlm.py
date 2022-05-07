import math
from functools import reduce
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sunyata.pytorch.bayes.core import DeepBayesInferCfg, log_bayesian_iteration

@dataclass
class DeepBayesInferMLMCfg(DeepBayesInferCfg):
    vocab_size: int = None
    seq_len: int = None
    is_mask: bool = False

    mask_prob: float = 0.15
    replace_prob: float = 0.9
    mask_token_id: int = 1  # id of special token <mask> in the vocabulary of tokenizer
    pad_token_id: int = 0  # id of special token <pad>

    mask_ignore_token_ids = set([2, 3])  # ids of special tokens <unk>, '\n'


class DeepBayesInferMLM(pl.LightningModule):
    def __init__(self, layers:nn.ModuleList, cfg:DeepBayesInferMLMCfg):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        nn.init.xavier_normal_(self.embed.weight.data)

        self.digup = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        )

        self.vocab_size = cfg.vocab_size
        self.mask_ignore_token_ids = cfg.mask_ignore_token_ids
        self.mask_prob = cfg.mask_prob
        self.replace_prob = cfg.replace_prob
        self.mask_token_id = cfg.mask_token_id
        self.pad_token_id = cfg.pad_token_id
        self.learning_rate = cfg.learning_rate

    def forward(self, input: torch.Tensor):
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        target = input[mask]  # input.masked_fill(~mask, self.pad_token_id)
        log_prior = torch.zeros_like(target).unsqueeze(-1).repeat((1, self.vocab_size))

        masked_input = input.clone().detach()
        replace_prob = prob_mask_like(input, self.replace_prob)
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        hidden_features = self.embed(masked_input)

        for layer in self.layers:
            hidden_features = layer(hidden_features)
            chosen_hidden_features = hidden_features[mask]
            logits = self.digup(chosen_hidden_features)

            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        loss = F.nll_loss(log_posterior, target)
        return loss

    def training_step(self, batch, batch_idx):
        input = batch
        loss = self.forward(input)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch
        loss = self.forward(input)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer    



def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()
