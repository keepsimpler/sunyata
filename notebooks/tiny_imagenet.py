# %%
from sunyata.pytorch.tiny_imagenet import TinyImageNet
from torch.utils.data import DataLoader
from torchvision import transforms

# %%
train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# %%
train_data = TinyImageNet(split='train', transform=train_transforms)

image, target = train_data[0]
image.shape

# %%
batch_size = 2
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

len(train_data), len(train_loader)

# %%
import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from sunyata.pytorch.bayes.vision import DeepBayesInferVision, DeepBayesInferVisionCfg


# %%
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn =fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# %%
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# %%
cfg = DeepBayesInferVisionCfg(
    image_size = 224,
    patch_size = 16,
    num_classes = 200,
    hidden_dim= 1024,
    num_layers= 6,
    num_heads = 16,
    mlp_dim = 2048,
    pool = 'cls',
    channels = 3,
    dim_head = 64,
    dropout = 0.1,
    emb_dropout = 0.1,
    batch_size=2, 
    learning_rate=1e-3)

# %%
image, target = next(iter(train_loader))
image.shape, target.shape

# %%
layers = [nn.Identity()]
deep_bayes_net = DeepBayesInferVision(layers, cfg)

log_posterior = deep_bayes_net(image)
posterior = torch.exp(log_posterior)

assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 

# %%
