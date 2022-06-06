# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token)

from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

# %%
batch_size, vocab_size, seq_len = 2, 1000, 16
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=batch_size,
                   vocab_size=vocab_size,
                   seq_len=seq_len,
                   collate_fn=shift_one_token)  # shift_one_token  None
# %%
wikitext2.tokenizer.decode(wikitext2.train_data[0].tolist(), skip_special_tokens=False)
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
# https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

# %%
input, target = next(iter(wikitext2.train_dataloader()))
input.shape, target.shape

# %%
num_layers = 2
cfg = TransformerCfg(
    hidden_dim = 16,
    num_heads = 2,
    expanded_dim= 2*16,
)
# %%
embed = nn.Embedding(vocab_size, cfg.hidden_dim)
torch.nn.init.xavier_normal_(embed.weight.data)
embed
# %%
layers = nn.Sequential(*[TransformerLayer(cfg) for _ in range(num_layers)])
layers
# %%
embedded = embed(input)
embedded.shape
# %%
output = layers(embedded)
output.shape
# %%
input, target
# %%
embed.weight[738]
# %%
embedded_target = embed(target)
embedded_target.shape
# %%
loss_beta = 1.
loss_fn = nn.SmoothL1Loss(reduction='mean', beta = loss_beta)
# %%
loss = loss_fn(output, embedded_target) # .sum(dim=-1).sum().div(output.shape[0])
loss
# %%
loss.backward()
# %%
embed.weight.grad.data
# %%
layers[0].attention.to_qkv.weight.grad.data, layers[1].attention.to_qkv.weight.grad.data
# %%
model = nn.Sequential(
    nn.Embedding(vocab_size, cfg.hidden_dim),
    nn.Sequential(*[TransformerLayer(cfg) for _ in range(num_layers)])
)

# %%
output = model(input)
output.shape
# %%
embedded_target = model[0](target)
embedded_target.shape
# %%
embedded_target = torch.randn(batch_size, seq_len-1, cfg.hidden_dim)
embedded_target.shape
# %%
model[0].weight.grad[930]
# %%
model[1][0].attention.to_qkv.weight.grad.data

# %%
embedded.mean().backward()
# %%
embed.weight.grad[930]
# %%
