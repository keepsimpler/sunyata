# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration

from sunyata.pytorch.arch.convmixer import BayesConvMixer2, ConvMixer, ConvMixerCfg
from sunyata.pytorch.data.tiny_imagenet import TinyImageNetDataModule
# %%
checkpoint_path = ".data/results/epoch=39-step=31280.ckpt"
# %%
cfg = ConvMixerCfg(
    hidden_dim = 256,
    num_layers = 16,
    kernel_size = 5,
    patch_size = 2,
    num_classes = 200,
    
    batch_size = 128,
    num_epochs = 40,
    learning_rate = 1e-2,
    optimizer_method = "AdamW",
    weight_decay = 0.01,
    learning_rate_scheduler= "OneCycleLR",  # LinearWarmupCosineAnnealingLR
#     warmup_epochs = 10,  # 2//5 * num_epoches
)
# %%
model = BayesConvMixer2.load_from_checkpoint(checkpoint_path, cfg=cfg)
model.summarize(max_depth=2)
# %%
import torchvision
normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)

tiny_image_net_datamodule = TinyImageNetDataModule(
    batch_size=cfg.batch_size, root='.data/',
    # train_transforms=train_transforms, 
    val_transforms=val_transforms)
# %%
input, target = next(iter(tiny_image_net_datamodule.val_dataloader()))
# %%
with torch.no_grad():
    log_posterior = model(input)
log_posterior.shape
# %%
loss = F.nll_loss(log_posterior, target)
accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
loss, accuracy
# %%
(log_posterior.argmax(dim=-1) == target)
# %%
feature_maps = []
log_priors = []
log_prior = torch.zeros(cfg.batch_size, cfg.num_classes)
with torch.no_grad():
    input_embedded = model.embed(input)
    for layer in model.layers:
        input_embedded = layer(input_embedded)
        feature_maps.append(input_embedded)
        logits = model.digup(input_embedded) 
        log_prior = log_bayesian_iteration(log_prior, logits)
        log_priors.append(log_prior)

# %%
len(feature_maps), len(log_priors), feature_maps[0].shape, log_priors[0].shape
# %%
torch.exp(log_priors[0]).sum(-1)
# %%
feature_maps[0][0].std()
# %%
processed_feature_maps = []
for feature_map in feature_maps:
    processed = feature_map[0].sum(dim=0) / feature_map[0].shape[0]
    processed_feature_maps.append(processed.numpy())
# %%
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')