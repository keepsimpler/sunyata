# %%
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sunyata.pytorch.data.wikitext import WikiTextDataModule

from sunyata.pytorch.arch.mlm import DeepBayesInferMLM, DeepBayesInferMLMCfg
from sunyata.pytorch.layer.transformer import TransformerLayer

# %%
cfg = DeepBayesInferMLMCfg(
    vocab_size = 20000,
    seq_len = 128,
    hidden_dim = 64,
    num_heads = 2,
    expanded_dim = 128,
    batch_size = 8,
    learning_rate = 1e-3,
)
# %%
data_dir = ".data/wikitext/"
batch_size = cfg.batch_size
vocab_size = cfg.vocab_size
seq_len = cfg.seq_len

wikitext103_datamodule = WikiTextDataModule("103", data_dir, batch_size, vocab_size, seq_len, is_collate=False)
# %%
csv_logger = pl_loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_103", version=1)
trainer = pl.Trainer(gpus=1, 
                     max_epochs=1, 
                     limit_train_batches=100,  # 1.0 
                     limit_val_batches=10,  # 1.0 
                     log_every_n_steps=10,
                     logger=csv_logger)

# %%
layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
deep_bayes_net = DeepBayesInferMLM(layers, cfg)
# %%
input = next(iter(wikitext103_datamodule.train_dataloader()))

# %%
trainer.fit(deep_bayes_net, wikitext103_datamodule)


# %%
csv_logger = pl_loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_103", version=1)
trainer = pl.Trainer(gpus=1, 
                     max_epochs=1, 
                     limit_train_batches=100,  # 1.0 
                     limit_val_batches=10,  # 1.0 
                     log_every_n_steps=10,
                     logger=csv_logger)

# %%
layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]
layers = [nn.Sequential(*layers)]
deep_bayes_net = DeepBayesInferMLM(layers, cfg)
trainer.fit(deep_bayes_net, wikitext103_datamodule)
# %%
output = deep_bayes_net(input)
# %%
output.shape
# %%
