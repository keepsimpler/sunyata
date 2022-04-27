import torch
import torch.nn as nn

from sunyata.pytorch.bayes import DeepBayesInferLM, DeepBayesInferLMCfg
from sunyata.pytorch.bayes.vision import DeepBayesInferVision, DeepBayesInferVisionCfg
from sunyata.pytorch.layers.transformer import TransformerLayer


def test_deep_bayes_net_for_lm():

    cfg = DeepBayesInferLMCfg(vocab_size=16, hidden_dim=8, num_heads=2, expanded_dim=16, seq_len=8, batch_size=2, learning_rate=1e-3)
    batch = torch.randint(0, cfg.vocab_size-1, (cfg.batch_size, cfg.seq_len+1))
    input = batch[:, :-1]

    layers = [TransformerLayer(cfg) for _ in range(4)]
    deep_bayesian_net = DeepBayesInferLM(layers, cfg)

    log_posterior = deep_bayesian_net(input)
    posterior = torch.exp(log_posterior)

    assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 


def test_deep_bayes_net_for_vision():

    cfg = DeepBayesInferVisionCfg(
        image_size = 64,
        patch_size = 16,
        num_classes = 200,
        hidden_dim= 1024,
        num_layers= 6,
        num_heads = 16,
        expanded_dim = 2048,
    #     is_mask = False,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.1,
        emb_dropout = 0.1,
        batch_size=32, 
        learning_rate=1e-3)

    batch = torch.randint(0, cfg.vocab_size-1, (cfg.batch_size, cfg.seq_len+1))
    input = batch[:, :-1]

    layers = [TransformerLayer(cfg) for _ in range(4)]
    deep_bayesian_net = DeepBayesInferLM(layers, cfg)

    log_posterior = deep_bayesian_net(input)
    posterior = torch.exp(log_posterior)

    assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 
