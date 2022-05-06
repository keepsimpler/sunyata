# %%
import torch
import torch.nn as nn
from torchvision import transforms

from sunyata.pytorch.tiny_imagenet import TinyImageNet, TinyImageNetDataModule

from sunyata.pytorch.convnext import BottleNeckBlock, ConvNextStage, ConvNextStem, ConvNextEncoder
from sunyata.pytorch.convnext import LayerScaler, ConvNextForImageClassification, ClassificationHead


# %%
train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

tiny_image_net_datamodule = TinyImageNetDataModule(
    batch_size=32, root='.data/',
    train_transforms=train_transforms, 
    val_transforms=train_transforms)

# %%
batch = next(iter(tiny_image_net_datamodule.train_dataloader()))
input, target = batch
# %%
classifier = ConvNextForImageClassification(in_channels=3, stem_features=64, depths=[3,3,9,3], widths=[96, 192, 384, 768])
classifier.training_step(batch, batch_idx=None)

# %%
block = BottleNeckBlock(3, 3)
block(input).shape
# %%
stage = ConvNextStage(3, 64, depth=1)
stage(input).shape
# %%
stem = ConvNextStem(3, 64)
stem(input).shape
# %%
encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3,3,9,3], widths=[96, 192, 384, 768])
encoder_output = encoder(input)
encoder_output.shape
# %%
layer_scaler = LayerScaler(init_value=1e-6, dimensions=3)
# %%
layer_scaler(input).shape
# %%
classification_head = ClassificationHead(num_channels=2048, num_classes=1000)
# %%
nn.AdaptiveAvgPool2d((1, 1))(encoder_output).shape
# %%
nn.Flatten(1)(nn.AdaptiveAvgPool2d((1, 1))(encoder_output)).shape
# %%
# tiny: [96, 192, 384, 768]