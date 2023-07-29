# %%
import torch

from sunyata.pytorch.layer.transformer import TransformerCfg
from sunyata.pytorch.arch.vit import ViT, ViTCfg, IterViTPreNorm, IterAttnViTPreNorm

# %%
def test_vit():
    cfg = ViTCfg()
    cfg.pool = 'mean'
    input = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    vit_model = ViT(cfg)
    output = vit_model(input)
    assert output.shape == (2, cfg.num_classes)

# %%
def test_iter_vit_prenorm():
    cfg = ViTCfg()
    cfg.pool = 'mean'
    input = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    vit_model = IterViTPreNorm(cfg)
    output = vit_model(input)
    assert output.shape == (2, cfg.num_classes)

# %%
def test_iter_attn_vit_prenorm():
    cfg = ViTCfg()
    cfg.pool = 'mean'
    input = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    vit_model = IterAttnViTPreNorm(cfg)
    output = vit_model(input)
    assert output.shape == (2, cfg.num_classes)

# %%
