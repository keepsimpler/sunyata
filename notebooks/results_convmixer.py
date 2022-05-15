# %%
import pandas as pd
from sunyata.pytorch.convmixer import DeepBayesInferConvMixerCfg
# %%
models = ["convmixer", "mlpmixer", "vit"]
datasets = ["cifar10", "tiny-imagenet"]
# %%
cfg = DeepBayesInferConvMixerCfg(
    hidden_dim = 256,
    num_layers = 20,
    kernel_size = 5,
    patch_size = 2,
    num_classes = 10,

    is_bayes = False,
    is_prior_as_params = False,

    batch_size=256,
    num_epochs = 40,
    learning_rate = 1e-3,
    optimizer_method = "Adam",  # or "AdamW"
    learning_rate_scheduler = "CosineAnnealing",
    weight_decay = None,  # of "AdamW"
)

df_cfg = pd.Series(cfg.__dict__)
df_cfg['model'] = 'convmixer'
df_cfg['dataset'] = 'tiny-imagenet'

# %%
df = pd.read_csv('.data/results/metrics.csv')
# %%
df.head()
# %%
train_metrics = ['train_loss', 'train_accuracy', 'epoch', 'step']
val_metrics = ['epoch', 'step', 'val_loss', 'val_accuracy']
# %%
df_train_metrics = df.loc[:, train_metrics]
df_train_metrics = df_train_metrics.dropna()
df_train_metrics.head()
# %%
df_val_metrics = df.loc[:, val_metrics]
df_val_metrics = df_val_metrics.dropna()
df_val_metrics.head()
# %%
df_train_metrics[df_cfg.index.tolist()] = df_cfg.values
# %%
df_val_metrics[df_cfg.index.tolist()] = df_cfg.values

# %%
df_val_metrics.head()
# %%
df_train_metrics.head()
# %%
df_train_all = df_train_all.append(df_train_metrics)
df_val_all = df_val_all.append(df_val_metrics)
df_train_all.to_csv('.data/convmixer_train_metrics.csv')
df_val_all.to_csv(".data/convmixer_val_metrics.csv")
# %%
df_train_all = pd.read_csv(".data/convmixer_train_metrics.csv")
df_train_all.info()
# %%
df_val_all = pd.read_csv(".data/convmixer_val_metrics.csv")
df_val_all.info()


# %%
