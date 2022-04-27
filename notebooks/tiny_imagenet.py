# %%
from sunyata.pytorch.tiny_imagenet import TinyImageNet, TinyImageNetDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

# %%
train_transforms = transforms.Compose(
    [
        # transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# %%
train_data = TinyImageNet(root='.data', split='train', transform=train_transforms)

image, target = train_data[0]
image.shape

# %%
batch_size = 2
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

len(train_data), len(train_loader)
# %%
image, target = next(iter(train_loader))
image.shape, target.shape


# %%
import torch
from torch import nn

from sunyata.pytorch.bayes.vision import DeepBayesInferVision, DeepBayesInferVisionCfg
from sunyata.pytorch.layers.transformer import TransformerLayer

# %%
cfg = DeepBayesInferVisionCfg(
    image_size = 224,
    patch_size = 16,
    num_classes = 200,
    hidden_dim= 1024,
    num_layers= 6,
    num_heads = 16,
    expanded_dim = 2048,
    pool = 'cls',
    channels = 3,
    dim_head = 64,
    dropout = 0.1,
    emb_dropout = 0.1,
    batch_size=2, 
    learning_rate=1e-3)

# %%
transformer = TransformerLayer(cfg)
layers = [TransformerLayer(cfg) for _ in range(2)]

# %%
deep_bayes_net = DeepBayesInferVision(layers, cfg)

log_posterior = deep_bayes_net(image)
posterior = torch.exp(log_posterior)

assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 

