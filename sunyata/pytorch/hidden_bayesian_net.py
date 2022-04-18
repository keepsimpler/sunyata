"""Deep Bayesian Inference Architecture

A deep Bayesian inference architecture includes four parts:
- Deep Bayesian Inference composed with a chain of Bayesian iterations.
- Neural Network composed with a chain of layers.
- Embed Block mapping from the observable space to the hidden space.
- Dig-up Block mapping from the hidden space to the probability space.

Classes:

Functions:

"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBayesInferLM(pl.LightningModule):
    """
    Deep Bayesian Inference Architecture of Language Models.

    Parameters
    ----------
    layers: ModuleList
        a chain of neural network layers

    Attributes
    ----------
    layers: ModuleList
        a chain of neural network layers, that transform hidden feature vectors
    embed: Module
        map word sequence to hidden feature vectors
    digup: Module
        map hidden feature vectors to logits, i.e. unscaled log probabilities
    
    """
    def __init__(self, layers: nn.ModuleList, vocab_size: int, hidden_dim: int, learning_rate: float, 
                    has_layernorm: bool=False, sharing_weight: bool=False, prior_as_params: bool=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.digup = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.init_weights()

        if sharing_weight:
            self.digup.weight = self.embed.weight
        
        if has_layernorm:
            self.digup = nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-12),
                self.digup
            )

        if prior_as_params:
            raise RuntimeError("Prior as parameters still have not been implemented.")
            # self.prior = nn.Parameter(torch.zeros(1, ))

        self.vocab_size, self.hidden_dim, self.learning_rate = vocab_size, hidden_dim, learning_rate
        
    def init_weights(self) -> None:
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_features = self.embed(input)

        prior = torch.ones_like(input).unsqueeze(-1).repeat((1, 1, self.vocab_size))
        log_prior = torch.log(prior)

        for layer in self.layers:
            hidden_features = layer(hidden_features)
            logits = self.digup(hidden_features)

            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def training_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        log_posterior = log_posterior.permute((0, 2, 1))
        loss = F.nll_loss(log_posterior, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        log_posterior = log_posterior.permute((0, 2, 1))
        loss = F.nll_loss(log_posterior, target)
        self.log("val_loss", loss)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def log_bayesian_iteration(log_prior: torch.Tensor, evidence_candidated: torch.Tensor) -> torch.Tensor:
    log_total_evidence = torch.logsumexp(log_prior + evidence_candidated, dim=-1, keepdim=True)
    log_posterior = log_prior + evidence_candidated - log_total_evidence
    return log_posterior



if __name__ == "__main__":
    batch_size, seq_len, vocab_size, hidden_dim = 2, 8, 16, 8
    batch = torch.randint(0, vocab_size-1, (batch_size, seq_len+1))
    input = batch[:, :-1]

    layers = [nn.Identity()]
    hidden_bayesian_net = DeepBayesInferLM(layers, vocab_size, hidden_dim, learning_rate=1e-3)

    log_posterior = hidden_bayesian_net(input)

    print(log_posterior)