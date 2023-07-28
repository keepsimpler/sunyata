# %%
from einops import rearrange, repeat
import torch
import torch.nn as nn
from sunyata.pytorch.arch.convmixer import ConvMixerCfg, IterAttnConvMixer
# %%
def test_IterAttnConvMixer():
    cfg = ConvMixerCfg(
        patch_size = 7,
    )

    input = torch.randn(cfg.batch_size, 3, 224, 224)

    model = IterAttnConvMixer(cfg)

    output = model(input)
    assert output.shape == (cfg.batch_size, cfg.num_classes)

    # x = model.embed(input)

    # x = x.permute(0, 2, 3, 1)
    # x = rearrange(x, 'b ... d -> b (...) d')

    # latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))
    # latent = repeat(latent, 'n d -> b n d', b = cfg.batch_size)
