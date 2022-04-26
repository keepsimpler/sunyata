import torch
import torch.nn as nn
from sunyata.pytorch.layers.attention import Attention
from sunyata.pytorch.layers.transformer import TransformerLayer
from sunyata.pytorch.bayes import DeepBayesInferCfg

def test_attention():

    batch_size, seq_len, hidden_dim = 2, 6, 8
    input = torch.randn((batch_size, seq_len, hidden_dim))
    num_heads = 2

    attention = Attention(hidden_dim, num_heads)

    output = attention(input)
    assert output.shape == (batch_size, seq_len, hidden_dim)

def test_transformer():
    cfg = DeepBayesInferCfg(hidden_dim=8, num_heads=2, expanded_dim=16)
    transformer = TransformerLayer(cfg)

    input = torch.randn(2, 6, cfg.hidden_dim)
    output = transformer(input)
    print(output.shape)


# if __name__ == "__main__":
#     test_attention()
#     test_transformer()