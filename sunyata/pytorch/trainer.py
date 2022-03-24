# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sunyata.pytorch.hidden_bayesian_net import HiddenBayesianNet
from sunyata.pytorch.transformer import TransformerLayer
from sunyata.pytorch.wikitext import WikiText2DataModule

import pytorch_lightning as pl
# %%
batch_size = 32
seq_len = 32
wikitext2_data_module = WikiText2DataModule(batch_size, seq_len + 1)
# %%
hidden_dim = 200
expanded_dim = 1 * hidden_dim
num_heads = 2
dropout = 0.2
transformer = TransformerLayer(hidden_dim, expanded_dim, num_heads, dropout)
# %%
num_layers = 4
# layers = [TransformerLayer(hidden_dim, expanded_dim, num_heads, dropout).cuda() for _ in range(num_layers)]
layers = [nn.Linear(hidden_dim, hidden_dim).cuda() for _ in range(num_layers)]
# %%
# layers = nn.Sequential(*layers)
# layers = [layers]
# %%
# layers = [nn.Identity()]
# %%
vocab_size = wikitext2_data_module.vocab_size
learning_rate = 1e-3
hidden_bayesian_net = HiddenBayesianNet(layers, vocab_size, hidden_dim, learning_rate, sharing_weight=False)
# %%
trainer = pl.Trainer(gpus=1)
# %%
trainer.fit(hidden_bayesian_net, wikitext2_data_module)
# %%
