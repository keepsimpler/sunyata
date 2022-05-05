# %%
import torch

from sunyata.pytorch.convnext import BottleNeckBlock, ConvNextStage, ConvNextStem, ConvNextEncoder

x = torch.rand(1, 32, 7, 7)
block = BottleNeckBlock(32, 64)
block(x).shape
# %%
stage = ConvNextStage(32, 64, depth=2)
stage(x).shape
# %%
stem = ConvNextStem(3, 64)
stem(x).shape
# %%
image = torch.rand(1, 3, 224, 224)
encoder = ConvNextEncoder(in_channels=3, stem_features=64, depths=[3,3,9,3], widths=[256, 512, 1024, 2048])
encoder(image).shape
# %%
