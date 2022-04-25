import torch
import torch.nn as nn
from sunyata.pytorch.layers import attention
from sunyata.pytorch.layers.attention import Attention

def test_attention():

    batch_size, seq_len, hidden_dim = 2, 6, 8
    input = torch.randn((batch_size, seq_len, hidden_dim))
    num_heads = 2

    attention = Attention(hidden_dim, num_heads)

    output = attention(input)
    assert output.shape == (batch_size, seq_len, hidden_dim)
