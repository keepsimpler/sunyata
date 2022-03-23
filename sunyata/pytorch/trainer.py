# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sunyata.pytorch.hidden_bayesian_net import HiddenBayesianNet
from sunyata.pytorch.transformer import TransformerLayer
from sunyata.pytorch.fake_data import FakeLMDataset

import pytorch_lightning as pl
# %%
trainer = pl.Trainer(gpus=1)
# %%
vocab_size=2**8
fake_lm_dataset = FakeLMDataset(sequences_num=2**12, vocab_size=vocab_size, seq_len=32)
# %%
train_dataloader = DataLoader(fake_lm_dataset, batch_size=8)
# %%
hidden_dim = 16
expanded_dim = 2 * hidden_dim
num_heads = 8
dropout = 0.1
transformer = TransformerLayer(embed_dim=hidden_dim, hidden_dim=expanded_dim, num_heads=num_heads, dropout=dropout)
# %%
num_layers = 2
layers = [TransformerLayer(embed_dim=hidden_dim, hidden_dim=expanded_dim, num_heads=num_heads, dropout=dropout).cuda() for _ in range(num_layers)]

layers = nn.Sequential(*layers)
# %%
layers = [nn.Identity()]
# %%
learning_rate = 1e-3
hidden_bayesian_net = HiddenBayesianNet(layers, vocab_size, hidden_dim, learning_rate)
# %%
trainer.fit(hidden_bayesian_net, train_dataloader)
# %%
