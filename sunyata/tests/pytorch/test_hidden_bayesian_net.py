import torch
import torch.nn as nn

from sunyata.pytorch.bayes import DeepBayesInferLM


def test_hidden_bayesian_net():
    batch_size, seq_len, vocab_size, hidden_dim = 2, 8, 16, 8
    batch = torch.randint(0, vocab_size-1, (batch_size, seq_len+1))
    input = batch[:, :-1]

    layers = [nn.Identity()]
    hidden_bayesian_net = DeepBayesInferLM(layers, vocab_size, hidden_dim, learning_rate=1e-3)

    posterior = hidden_bayesian_net(input)
    # assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.))

    # posterior = posterior.permute((0, 2, 1))
