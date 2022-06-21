# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sunyata.pytorch.data.tiny_imagenet import TinyImageNet, TinyImageNetDataModule
from sunyata.pytorch.arch.vit import ViT, ViTCfg
from sunyata.pytorch.layer.transformer import TransformerLayer, TransformerCfg
# %%
hidden_dim = 128
cfg = ViTCfg(
    hidden_dim = hidden_dim,
    image_size = 64,
    patch_size = 8,
    num_classes = 200,

    transformer = TransformerCfg(
        hidden_dim = hidden_dim,
        num_heads = 2,
        expanded_dim= 2*hidden_dim,
        is_softmax=True,
    ),
    batch_size = 64,
    num_layers = 4,
    num_epochs = 1,
    learning_rate = 1e-3 # 1e-3  3e-4

)

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
vit = ViT(cfg)
vit.summarize(max_depth=2)

# %%
input, target = next(iter(tiny_image_net_datamodule.train_dataloader()))
output = vit(input)
# %%
csv_logger = pl_loggers.CSVLogger(save_dir="lightning_logs/", 
    name="tiny-imagenet", version=1)
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     logger=csv_logger)

# %%
trainer.fit(vit, tiny_image_net_datamodule)
