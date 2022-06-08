# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, BaseModule

# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token)

from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

# %%
@dataclass
class LatentLearnCLMCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None

    loss_beta = 1.

    
    transformer: TransformerCfg = None    

# %%
hidden_dim = 64
cfg = LatentLearnCLMCfg(
    vocab_size = 1000,
    seq_len = 64,
    hidden_dim = hidden_dim,
    transformer = TransformerCfg(
        hidden_dim = hidden_dim,
        num_heads = 2,
        expanded_dim= 2*hidden_dim,
        is_attn=False,
    ),

    batch_size = 16,
    num_layers = 2,
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
class LatentLearnCLM(BaseModule):
    def __init__(self, cfg:LatentLearnCLMCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        self.layers = nn.Sequential(*[TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)])

        # self.loss_fn = nn.SmoothL1Loss(reduction='none', beta = cfg.loss_beta)
        self.cosine_loss_fn = nn.CosineSimilarity(dim=-1)
        self.cfg = cfg

    def forward(self, input):
        embedded = self.embed(input)
        output = self.layers(embedded)
        return output

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output = self.forward(input)
        embedded_target = self.embed(target)  # forward

        output_norm = torch.norm(output, p=2, dim=-1)
        target_norm = torch.norm(embedded_target, p=2, dim=-1)
        denominator = torch.einsum('b s, b t -> b s t', output_norm, target_norm)
        denominator = torch.max(denominator, torch.tensor(1e-8))
        cosine_loss = torch.einsum('b s h, b t h -> b s t', output, embedded_target)
        cosine_loss = cosine_loss / denominator
        max_loss = torch.diagonal(cosine_loss, 0, dim1=1, dim2=2)
        min_loss = cosine_loss.clone()
        min_loss.diagonal(dim1=-1, dim2=-2).zero_()
        # loss = (min_loss.sum() - max_loss.sum()) / (torch.numel(cosine_loss))
        loss = 2 + min_loss.mean() - max_loss.mean()
        # loss = self.loss_fn(output, embedded_target).sum(dim=-1).sum().div(output.shape[0])
        # loss = - self.cosine_loss_fn(output, embedded_target).mean()
        self.log(mode + "_loss", loss)
        return loss
# %%
latent_learn = LatentLearnCLM(cfg)
latent_learn.summarize(max_depth=2)
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
trainer.fit(latent_learn, wikitext2)

# %%
for i, (input, target) in enumerate(wikitext2.train_dataloader()):
    output = latent_learn(input)
    embedded_target = latent_learn.embed(target)
    loss = latent_learn.cosine_loss_fn(output, embedded_target).mean()
    print(loss)
    if i>100:break


# %%
output_norm = torch.norm(output, p=2, dim=-1)
target_norm = torch.norm(embedded_target, p=2, dim=-1)
denominator = torch.einsum('b s, b t -> b s t', output_norm, target_norm)
denominator = torch.max(denominator, torch.tensor(1e-8))
cosine_loss = torch.einsum('b s h, b t h -> b s t', output, embedded_target)
cosine_loss = cosine_loss / denominator

# %%
max_loss = torch.diagonal(cosine_loss, 0, dim1=1, dim2=2)
min_loss = cosine_loss.clone()
min_loss.diagonal(dim1=-1, dim2=-2).zero_()
loss = 2 + min_loss.mean() - max_loss.mean()
loss
# %%
cosine_loss[0,1]
# %%
