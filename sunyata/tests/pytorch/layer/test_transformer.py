import torch
import torch.nn as nn
from sunyata.pytorch.layer.attention import SelfAttention
from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

def test_attention():

    batch_size, seq_len, hidden_dim = 2, 6, 8
    input = torch.randn((batch_size, seq_len, hidden_dim))
    num_heads = 2

    attention = SelfAttention(hidden_dim, num_heads)

    output = attention(input)
    assert output.shape == (batch_size, seq_len, hidden_dim)

def test_transformer():
    cfg = TransformerCfg(
        hidden_dim = 64,
        num_heads = 2,
        expanded_dim= 2*64,
        is_softmax=True,
        is_ff=True,
        is_ff_layernorm=True,
    )
    
    transformer = TransformerLayer(cfg)

    input = torch.randn(2, 6, cfg.hidden_dim)
    output = transformer(input)
    print(output.shape)


# if __name__ == "__main__":
#     test_attention()
#     test_transformer()