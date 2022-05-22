# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sunyata.pytorch.data.tiny_imagenet import TinyImageNet, TinyImageNetDataModule
from sunyata.pytorch.arch.vision import DeepBayesInferVision, DeepBayesInferVisionCfg
from sunyata.pytorch.layer.transformer import TransformerLayer
# %%
cfg = DeepBayesInferVisionCfg(
    is_pre_layernorm=False,

    is_attn=True,
    is_to_qkv=False,
    # attn_scale = 1.,
    is_mask = False,
    is_softmax=False,
    is_to_out=False,
    is_attn_shortcut=False,

    is_inner_layernorm=False,

    is_ff=False,
    is_ff_shortcut=False,

    is_post_layernorm=True,

    is_layernorm_before_digup=False,

    batch_size = 16, 
    learning_rate = 1e-3)

# %%
train_transforms = transforms.Compose(
    [
#         transforms.Resize(224),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

tiny_image_net_datamodule = TinyImageNetDataModule(
    batch_size=cfg.batch_size, root='.data/',
    train_transforms=train_transforms, 
    val_transforms=train_transforms)
# %%
csv_logger = pl_loggers.CSVLogger(save_dir="lightning_logs/", 
    name="tiny-imagenet", version=1)
trainer = pl.Trainer(gpus=1, 
                     max_epochs=1, 
                     limit_train_batches=100,  # 1.0 
                     limit_val_batches=10,  # 1.0 
                     log_every_n_steps=10,
                     logger=csv_logger)

# %%
layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
deep_bayes_net = DeepBayesInferVision(layers, cfg)
trainer.fit(deep_bayes_net, tiny_image_net_datamodule)

# %%
csv_logger = pl_loggers.CSVLogger(save_dir="lightning_logs/", 
    name="tiny-imagenet", version=2)
trainer = pl.Trainer(gpus=1, 
                     max_epochs=1, 
                     limit_train_batches=100,  # 1.0 
                     limit_val_batches=10,  # 1.0 
                     log_every_n_steps=10,
                     logger=csv_logger)

# %%
layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
layers = [nn.Sequential(*layers)]
deep_bayes_net = DeepBayesInferVision(layers, cfg)
trainer.fit(deep_bayes_net, tiny_image_net_datamodule)
# %%
# layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
# deep_bayes_net = DeepBayesInferVision(layers, cfg)
# lr_finder = trainer.tuner.lr_find(deep_bayes_net, tiny_image_net_datamodule)
# lr_finder.results
# lr_finder.suggestion()

