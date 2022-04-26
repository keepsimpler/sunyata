import torch
import torch.nn as nn

from sunyata.pytorch.bayes import DeepBayesInferLM, DeepBayesInferLMCfg


def test_hidden_bayesian_net():

    cfg = DeepBayesInferLMCfg(vocab_size=16, hidden_dim=8, seq_len=8, batch_size=2, learning_rate=1e-3)
    batch = torch.randint(0, cfg.vocab_size-1, (cfg.batch_size, cfg.seq_len+1))
    input = batch[:, :-1]

    layers = [nn.Identity()]
    deep_bayesian_net = DeepBayesInferLM(layers, cfg)

    log_posterior = deep_bayesian_net(input)
    posterior = torch.exp(log_posterior)

    assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 
