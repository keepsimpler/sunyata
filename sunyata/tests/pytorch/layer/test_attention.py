# %%
import torch
import torch.nn as nn
from einops import rearrange, repeat

from sunyata.pytorch.layer.attention import EfficientChannelAttention, Attention
# %%
def test_efficient_channel_attention():
    input = torch.randn(2, 3, 224, 224)
    eca = EfficientChannelAttention(kernel_size=3)
    output = eca(input)
    assert output.shape == (2, 3)


# %%
def test_attention():
    input = torch.randn(2, 3, 224, 224)
    input = input.permute(0, 2, 3, 1)
    input = rearrange(input, 'b ... d -> b (...) d')
    b, input_dim, context_dim = input.shape

    latent_dim = 1
    query_dim = 256
    latents = nn.Parameter(torch.randn(latent_dim, query_dim))
    latents = repeat(latents, 'n d -> b n d', b = b)

    cross_attn = Attention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=1,
        dim_head=64,
    )

    output = cross_attn(latents, input)

    assert output.shape == (b, latent_dim, query_dim)

