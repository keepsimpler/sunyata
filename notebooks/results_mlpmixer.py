# %%
import pandas as pd
from sunyata.pytorch.arch.mlpmixer import MlpMixerCfg, DeepBayesInferMlpMixer
# %%
models = ["convmixer", "mlpmixer", "vit"]
datasets = ["cifar10", "tiny-imagenet"]
# %%
first = False

cfg = MlpMixerCfg(
    image_size = 64,
    patch_size = 8,
    hidden_dim = 256,
    expansion_factor = 4,
    expansion_factor_token = 0.5,

    num_layers = 20,
    num_classes = 200,
    channels = 3,
    dropout = 0. ,

    is_bayes = True,
    is_prior_as_params =False,

    num_epochs = 100,
    learning_rate = 1e-3,
    optimizer_method = "Adam",  # or "AdamW"
    learning_rate_scheduler = "CosineAnnealing"
#     weight_decay: float = None  # of "AdamW"

)

df_cfg = pd.Series(cfg.__dict__)
df_cfg['model'] = 'mlpmixer'
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
df_train_metrics.info()
# %%
df_val_metrics = df.loc[:, val_metrics]
df_val_metrics = df_val_metrics.dropna()
df_val_metrics.head()
df_val_metrics.info()
# %%
df_train_metrics[df_cfg.index.tolist()] = df_cfg.values
df_train_metrics.info()
# %%
df_val_metrics[df_cfg.index.tolist()] = df_cfg.values
df_val_metrics.info()

# %%
if first:
    df_train_all = df_train_metrics
    df_val_all = df_val_metrics
else:
    df_train_all = df_train_all.append(df_train_metrics)
    df_val_all = df_val_all.append(df_val_metrics)

df_train_all.to_csv('.data/mlpmixer_train_metrics.csv')
df_val_all.to_csv(".data/mlpmixer_val_metrics.csv")
# %%

df_train_all = pd.read_csv(".data/mlpmixer_train_metrics.csv")
df_train_all.info()
# %%
df_val_all = pd.read_csv(".data/mlpmixer_val_metrics.csv")
df_val_all.info()

# %%
