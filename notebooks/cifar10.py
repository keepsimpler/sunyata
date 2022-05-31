# %%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
# %%

pl.seed_everything(7)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".data")
BATCH_SIZE = 32
NUM_WORKERS = 0

# %%
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)
# %%
from sunyata.pytorch.arch.convmixer import ConvMixer, ConvMixerCfg

cfg = ConvMixerCfg(
    hidden_dim = 256,
    num_layers = 8,
    kernel_size = 5,
    patch_size = 2,
    num_classes = 10,

    is_bayes = True,
    is_prior_as_params = False,

    num_epochs = 1,
    learning_rate = 1e-3,
    optimizer_method = "Adam",  # or "AdamW"
    learning_rate_scheduler = "CosineAnnealing",
    weight_decay = None,  # of "AdamW"
)
# %%
model = ConvMixer(cfg)

trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=cfg.num_epochs,
    gpus=1,
    logger=CSVLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, cifar10_dm)

# %%
trainer.test(model, datamodule=cifar10_dm)
# %%
