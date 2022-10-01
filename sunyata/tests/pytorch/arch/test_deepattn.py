# %%
import torch

from sunyata.pytorch.arch.deepattn import DeepAttn, DeepAttnCfg, Attn, AttnLayer

# %%
def test_DeepAttn():
    cfg = DeepAttnCfg(
        hidden_dim = 256,
        kernel_size = 5,
        patch_size = 2,
        num_classes = 200,
        
        drop_rate = 0.,
        temperature = 1.,
        init_scale = 1.,
        attn_depth = 2,
        
        num_layers = 8,
        batch_size = 128,
        num_epochs = 50,
        learning_rate = 1e-2,
        optimizer_method = "AdamW",
        weight_decay = 0.1,
        learning_rate_scheduler= "LinearWarmupCosineAnnealingLR",
        warmup_epochs = 10,  # 2//5 * num_epoches
        warmup_start_lr = 1e-5,
    )
    model = DeepAttn(cfg)

    height, width = 32, 32
    x = torch.randn(cfg.batch_size, 3, height, width)

    output = model(x)
    assert output.shape == (cfg.batch_size, cfg.num_classes)
# %%
def test_AttnLayer():
    batch_size = 2
    hidden_dim = 64
    kernel_size = 3
    drop_rate = 0.
    temperature = 1.
    init_scale = 1.
    attn_depth = 2
    attn_layer = AttnLayer(hidden_dim, kernel_size, drop_rate, temperature, init_scale, attn_depth)
    
    x = torch.randn(batch_size, hidden_dim, 32, 32)

    xs = attn_layer(x,x)
    assert len(xs) == 3

# %%
def test_Attn():
    batch_size = 2
    hidden_dim = 3
    height, width = 4, 4
    x = torch.randn(batch_size, hidden_dim, height, width)
    attn = Attn(hidden_dim=hidden_dim,temperature=1., init_scale=1., attn_depth=2)

    assert attn(x,x,x).shape == x.shape

# %%
