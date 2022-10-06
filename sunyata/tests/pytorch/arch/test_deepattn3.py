# %%
import torch

from sunyata.pytorch.arch.deepattn3 import DeepAttn, DeepAttnCfg

# %%
def test_DeepAttn():
    cfg = DeepAttnCfg(
        hidden_dim = 64,
        kernel_size = 5,
        patch_size = 2,
        num_classes = 200,
        
        drop_rate = 0.,
        temperature = 1.,
        init_scale = 1.,
        
        num_layers = 2,
        batch_size = 16,
        num_epochs = 50,
        learning_rate = 1e-2,
        optimizer_method = "AdamW",
        weight_decay = 0.1,
        learning_rate_scheduler= "LinearWarmupCosineAnnealingLR",
        warmup_epochs = 10,  # 2//5 * num_epoches
        warmup_start_lr = 1e-5,
    )
    model = DeepAttn(cfg).cuda()

    height, width = 32, 32
    x = torch.randn(cfg.batch_size, 3, height, width).cuda()

    output = model(x)
    assert output.shape == (cfg.batch_size, cfg.num_classes)

# %%
