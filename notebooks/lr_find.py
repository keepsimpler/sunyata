# %%
from sunyata.pytorch_lightning.tiny_imagenet import TinyImageNetDataModule
# %%
from torchvision import transforms

val_transforms = transforms.Compose([
    transforms.Resize(64 + 30),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
tiny_imagenet = TinyImageNetDataModule(
    root=".data/", 
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    )
# %%
from sunyata.pytorch_lightning.convmixer import PlConvMixer
from sunyata.pytorch.arch.convmixer import ConvMixerCfg
# %%
cfg = ConvMixerCfg(
    batch_size = 128,
    num_layers = 8,
    num_epochs = 50,
    optimizer_method = "AdamW",  # Lamb
    learning_rate = 1e-2,
    weight_decay = 1e-2,
    learning_rate_scheduler = "LinearWarmupCosineAnnealingLR",
    warmup_epochs = 10,
    warmup_start_lr = 1e-5,
    hidden_dim = 256,
    patch_size = 4,
    num_classes = 10,
    scale = 0.0625 / 24,
)
cfg

# %%
model = PlConvMixer(cfg)
model.lr = model.cfg.learning_rate
# %%
import lightning.pytorch as pl

trainer = pl.Trainer()
# %%
from lightning.pytorch.tuner.tuning import Tuner
# %%
tuner = Tuner(trainer)
# %%
tuner.lr_find(model, datamodule=tiny_imagenet)
# %%
