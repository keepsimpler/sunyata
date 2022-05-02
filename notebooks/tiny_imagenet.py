# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sunyata.pytorch.tiny_imagenet import TinyImageNet, TinyImageNetDataModule
from sunyata.pytorch.bayes.vision import DeepBayesInferVision, DeepBayesInferVisionCfg
from sunyata.pytorch.layers.transformer import TransformerLayer
# %%
cfg = DeepBayesInferVisionCfg(
    # image_size = 64,
    # patch_size = 8,
    # num_classes = 200,
    # hidden_dim= 128,  #  1024
    # num_heads = 2,  # 16
    # expanded_dim = 256,  # 2048
    # is_mask = False,
    # pool = 'cls',
    # channels = 3,
    # emb_dropout = 0.,
    # num_layers = 8,
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
