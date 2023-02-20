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
from sunyata.pytorch.arch.base import Residual
from sunyata.pytorch_lightning.base import BaseModule

# %%

pl.seed_everything(7)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".data")
BATCH_SIZE = 32
NUM_WORKERS = 0

# %%
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandAugment(num_ops=2, magnitude=12),
    torchvision.transforms.ColorJitter(0, 0, 0),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
    torchvision.transforms.RandomErasing(p=0)
])

# train_transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         cifar10_normalization(),
#     ]
# )

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

    num_epochs = 25,
    learning_rate = 5e-2,
    optimizer_method = "AdamW",
    weight_decay = 0.01,
    learning_rate_scheduler= "LinearWarmupCosineAnnealingLR",
    warmup_epochs = 10,
)

# %%
class ConvMixer(BaseModule):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(cfg.hidden_dim)
                )),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            ) for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.target_embed = nn.Embedding(cfg.num_classes, cfg.hidden_dim)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            # nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output_embedded = self.forward(input)
        # with torch.no_grad():
        target_embedded = self.target_embed(target)
        # cosine_loss = nn.SmoothL1Loss()(output_embedded, target_embedded)
        cosine_loss = nn.MSELoss()(output_embedded, target_embedded)
        # cosine_loss = 2 - 2 * (output_embedded * target_embedded).sum(dim=(-1,)).mean()
        # cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        self.log(mode + "_loss", cosine_loss, prog_bar=True)
        logits = output_embedded @ self.target_embed.weight.T

        class_loss = F.cross_entropy(logits, target)
        self.log(mode + "_class_loss", class_loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return cosine_loss

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
# trainer.test(model, datamodule=cifar10_dm)
# %%
