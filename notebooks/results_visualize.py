# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sunyata.pytorch.arch.bayes import log_bayesian_iteration

from sunyata.pytorch.data.tiny_imagenet import TinyImageNetDataModule, TinyImageNet
from sunyata.pytorch.convmixer import DeepBayesInferConvMixer, DeepBayesInferConvMixerCfg
# %%
cfg = DeepBayesInferConvMixerCfg(
    hidden_dim = 256,
    num_layers = 16,
    kernel_size = 5,
    patch_size = 2,
    num_classes = 200,

    is_bayes = True,
    is_prior_as_params = False,

    num_epochs = 40,
    learning_rate = 1e-3,
    optimizer_method = "Adam",  # or "AdamW"
    learning_rate_scheduler = "CosineAnnealing",
    weight_decay = None,  # of "AdamW"
)

# model = DeepBayesInferConvMixer(cfg)
model = DeepBayesInferConvMixer.load_from_checkpoint(
    ".data/results/convmixer-tiny-imagenet/version_0/checkpoints/epoch=39-step=31280.ckpt",
    cfg=cfg)

# %%
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
val_transforms = transforms.Compose(
    [
#         transforms.Resize(224),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
)

# dataset_path = "/home/fengwf/Code/"
dataset_path = ".data/"
dataset = TinyImageNet(dataset_path, split='val', transform=val_transforms)
val_dataloader = DataLoader(dataset, batch_size=32)
# %%
# csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/",
#                  name="convmixer-tiny-imagenet", version=0)
# trainer = pl.Trainer(max_epochs=1, 
#                      limit_train_batches=100,  # 1.0 
#                      limit_val_batches=10,  # 1.0 
#                      logger=csv_logger)
# trainer.validate(model, dataloaders=val_dataloader)
# %%
input, target = next(iter(val_dataloader))
input.shape, target.shape
# %%
log_posterior = model(input)
log_posterior.shape
# %%
predicted = log_posterior.argmax(dim=-1)
predicted, target


# %%
batch_size, _, _, _ = input.shape
log_prior = torch.zeros(batch_size, cfg.num_classes)
# %%
x = model.embed(input)
x.shape
# %%
activations = []
for i, layer in enumerate(model.layers):
    x = layer(x)
    logits = model.digup(x)
    print(torch.min(logits.data), torch.max(logits.data), torch.mean(logits.data))
    log_posterior = log_bayesian_iteration(log_prior, logits)
    activations.append(log_posterior.data.detach())
    log_prior = log_posterior

# %%
len(activations), activations[0].shape
# %%
all_log_posteriores = torch.stack(activations).permute(1,0,2)
# %%
torch.max(torch.exp(all_log_posteriores)), torch.min(all_log_posteriores), torch.mean(all_log_posteriores)
# %%
torch.max(model.digup[2].bias), torch.min(model.digup[2].bias), torch.mean(model.digup[2].bias)
torch.max(model.digup[2].weight), torch.min(model.digup[2].weight), torch.mean(model.digup[2].weight)
# %%
all_log_posteriores.shape
# %%
import seaborn as sns; sns.set_theme()
# %%
ax = sns.heatmap(all_log_posteriores[0])

# %% none bayesian
cfg.is_bayes = False
compared_model = DeepBayesInferConvMixer.load_from_checkpoint(
    ".data/results/convmixer-tiny-imagenet/version_1/checkpoints/epoch=39-step=31280.ckpt",
    cfg=cfg)


# %%
compared_log_posterior = compared_model(input)
compared_log_posterior

# %%
torch.max(compared_model.digup[2].bias), torch.min(compared_model.digup[2].bias), torch.mean(compared_model.digup[2].bias)
torch.max(compared_model.digup[2].weight), torch.min(compared_model.digup[2].weight), torch.mean(compared_model.digup[2].weight)


# %%
log_posterior_list = []
target_list = []
with torch.no_grad():
    for input, target in val_dataloader:
        log_posterior = compared_model(input)
        log_posterior_list.append(log_posterior)
        target_list.append(target)
        print(len(target_list))
# %%
log_posteriors = torch.cat(log_posterior_list)
targets = torch.cat(target_list)
# %%
F.nll_loss(log_posteriors, targets)  # 2.0313
# %%
n_bins = 15
bin_boundaries = torch.linspace(0, 1, n_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]
bin_boundaries
# %%
softmaxs = torch.exp(log_posteriors)
softmaxs.sum(-1)
# %%
confidences, predictions = torch.max(softmaxs, 1)
accuracies = predictions.eq(targets)
sum(accuracies)
# %%
ece = torch.zeros(1)
# %%
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    # Calculated |confidence - accuracy| in each bin
    in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
    prop_in_bin = in_bin.float().mean()
    if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
# %%
ece  # 0.0280
# %%
