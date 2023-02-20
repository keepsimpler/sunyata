# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch_lightning.base import BaseModule


# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token)

from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

from sunyata.pytorch.arch.data2vec import Data2VecCfg, Data2VecCLM, BYOL_EMA

# %%
hidden_dim = 64
cfg = Data2VecCfg(
    vocab_size = 1000,
    seq_len = 256,
    hidden_dim = hidden_dim,
    ema_tau = 0.999,  # 0.9999
    normalize_targets = False,
    transformer = TransformerCfg(
        hidden_dim = hidden_dim,
        num_heads = 2,
        expanded_dim= 2*hidden_dim,
    ),

    batch_size = 64,
    num_layers = 4,
    num_epochs = 1,
    learning_rate = 1e-3 # 1e-3  3e-4
)

# %%
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=cfg.batch_size,
                   vocab_size=cfg.vocab_size,
                   seq_len=cfg.seq_len,
                   collate_fn=shift_one_token)  # shift_one_token  None
# %%
wikitext2.tokenizer.decode(wikitext2.train_data[0].tolist(), skip_special_tokens=False)
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
# https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

# %%
input, target = next(iter(wikitext2.train_dataloader()))
input.shape, target.shape

# %%
byol_clm = Data2VecCLM(cfg)
byol_clm.summarize(max_depth=2)
# %%
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_2") # , version=2
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     enable_checkpointing=False,
                     callbacks=[BYOL_EMA(cfg.ema_tau)],
                    #  limit_train_batches=100,  # 1.0 
                    #  limit_val_batches=10,  # 1.0 
                     log_every_n_steps=50,
                     logger=csv_logger)

# %%
trainer.fit(byol_clm, wikitext2)

# %%
# for i, (input, target) in enumerate(wikitext2.train_dataloader()):
loss = byol_clm(input, target)
loss
# %%
