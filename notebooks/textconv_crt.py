# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token,
                                            split_to_two_parts)
from sunyata.pytorch.arch.textconv_crt import TextConvCrt, TextConvCrtCfg

# %%
cfg = TextConvCrtCfg(
    r1 = 100,
    r2 = 100,
    expansion = 2,
    vocab_size = 10000,
    seq_len = 128,
    batch_size = 16,
    kernel_size = 3,
    num_layers = 12,

    num_epochs = 10,
    learning_rate = 1e-3
)

# %%
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=cfg.batch_size,
                   vocab_size=cfg.vocab_size,
                   seq_len=cfg.seq_len,
                   collate_fn=split_to_two_parts,
                   is_shuffle=True)  # shift_one_token  None
# %%
wikitext2.tokenizer.decode(wikitext2.train_data[0].tolist(), skip_special_tokens=False)
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
# https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

# %%
(input_row, input_col), target = next(iter(wikitext2.train_dataloader()))
input_row.shape, target.shape

# %%
textconvcrt = TextConvCrt(cfg)
textconvcrt.summarize(max_depth=2)

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
trainer.fit(textconvcrt, wikitext2)


# %%
