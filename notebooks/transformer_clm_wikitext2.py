# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token,
                                            )
from sunyata.pytorch.layer.transformer import TransformerCfg
from sunyata.pytorch.arch.clm import TransformerCLM, TransformerCLMCfg, TransformerCLMNoShortcut

# %%
cfg = TransformerCLMCfg(
    vocab_size = 20000,
    seq_len = 128,
    hidden_dim = 84,
    transformer = TransformerCfg(
        hidden_dim = 84,
        num_heads = 4,
        expansion= 2*84,
        is_softmax=True,
        is_ff=True,
        is_ff_layernorm=True,
    ),

    batch_size = 16,
    num_layers = 10,
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

# %%|
transformer_clm = TransformerCLM(cfg)
# transformer_clm = TransformerCLMNoShortcut(cfg)
transformer_clm.summarize(max_depth=2)

# %%
loss = transformer_clm._step((input, target))
loss
# %%
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_2") # , version=2
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     enable_checkpointing=False,
                    #  limit_train_batches=100,  # 1.0 
                    #  limit_val_batches=10,  # 1.0 
                     log_every_n_steps=50,
                     logger=csv_logger)

# %%
trainer.fit(transformer_clm, wikitext2)


# %%
