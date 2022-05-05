# %%
import torch

from sunyata.pytorch.convnext import BottleNeckBlock, ConvNextStage, ConvNextStem, ConvNextEncoder

image = torch.rand(1, 3, 224, 224)
block = BottleNeckBlock(3, 64)
block(image).shape
# %%
stage = ConvNextStage(3, 64, depth=2)
stage(image).shape
# %%
stem = ConvNextStem(3, 64)
stem(image).shape
# %%
encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3,3,9,3], widths=[256, 512, 1024, 2048])
encoder(image).shape
# %%
