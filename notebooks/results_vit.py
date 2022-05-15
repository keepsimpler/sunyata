# %%
import pandas as pd
from sunyata.pytorch.bayes.vision import DeepBayesInferVisionCfg
# %%
models = ["convmixer", "mlpmixer", "vit"]
datasets = ["cifar10", "tiny-imagenet"]
# %%
first = False

cfg = DeepBayesInferVisionCfg(
    image_size=64,
    patch_size=8,
    hidden_dim=128,
    
    # attention
    num_heads=8,
    is_mask = False,
    attn_dropout = 0.,
    
    # feed forward
    expanded_dim=256,
    ff_dropout = 0.,
    
    is_bayes = False,
    
    # training
    num_layers = 20,
    batch_size = 256, 
    num_epochs = 10,
    optimizer_method = "Adam",  # AdamW
    learning_rate = 3e-4,
    learning_rate_scheduler = "CosineAnnealing"  # "OneCycle", "Step"
)

df_cfg = pd.Series(cfg.__dict__)
df_cfg['model'] = 'vit'
df_cfg['dataset'] = 'tiny-imagenet'

df_cfg = df_cfg.drop(['is_attn', 'attn_scale', 'is_to_qkv', 'is_to_qkv_bias', 'is_to_out', 
            'is_to_out_bias', 'is_softmax', 'is_ff', 'is_ff_bias', 'is_nonlinear',
            'is_attn_shortcut', 'is_ff_shortcut', 'gamma', 'factor', 'patience', 
            'warmup_epoch'], axis=0)
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

df_train_all.to_csv('.data/vit_train_metrics.csv')
df_val_all.to_csv(".data/vit_val_metrics.csv")
# %%

df_train_all = pd.read_csv(".data/vit_train_metrics.csv")
df_train_all.info()
# %%
df_val_all = pd.read_csv(".data/vit_val_metrics.csv")
df_val_all.info()

# %% Analysis
df_train_all.head()
# %%
df_train_all.hidden_dim.unique()
# %%
for col in df_train_all:
    print(col)
# %%
df_train_all["hidden_dim"]
# %%
