# %%
# from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, BaseModule, set_requires_grad
from sunyata.pytorch.arch.loss import InfoNCE, BarlowTwins, ECELoss

# %%
from sunyata.pytorch.data.wikitext import (WikiTextDataModule,
                                            shift_one_token)

from sunyata.pytorch.layer.transformer import TransformerCfg, TransformerLayer

from sunyata.pytorch.arch.contrastive_clm import ContrastiveCLMCfg, ContrastiveCLM, ContrastiveCLMCov
# %%
hidden_dim = 64
cfg = ContrastiveCLMCfg(
    vocab_size = 1000,
    seq_len = 256,
    hidden_dim = hidden_dim,
    transformer = TransformerCfg(
        hidden_dim = hidden_dim,
        num_heads = 2,
        expanded_dim= 2*hidden_dim,
        is_softmax=True,
    ),
    temperature = 1.,
    alpha = 0.,
    batch_size = 64,
    num_layers = 1,
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
class LatentAndCLM(BaseModule):
    def __init__(self, cfg:ContrastiveCLMCfg):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        torch.nn.init.xavier_normal_(self.embed.weight.data)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.digup.weight = self.embed.weight  # do not add .data

        self.layers = nn.Sequential(
            nn.Sequential(*[
                TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
            ]),
        )
        if cfg.temperature is None:
            temperature = cfg.hidden_dim ** 0.5
        else:
            temperature = cfg.temperature
        self.loss_fn = InfoNCE(temperature=temperature)

    def forward(self, input, target):
        # with torch.no_grad():
        input_embedded = self.embed(input)
        target_embedded = self.embed(target)

        output_embedded = self.layers(input_embedded)
        return output_embedded, target_embedded

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)
        # target_embedded.detach_()
        cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        self.log(mode + "_cosine_loss", cosine_loss)
        infonce_loss = self.loss_fn(output_embedded, target_embedded)
        logits = self.digup(output_embedded)  # output_embedded @ self.embed.weight.T  #
        class_loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        loss = cfg.alpha * class_loss + (1 - cfg.alpha) * infonce_loss
        self.log(mode + "_class_loss", class_loss)
        self.log(mode + "_infonce_loss", infonce_loss)
        self.log(mode + "_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output_embedded, target_embedded = self.forward(input, target)
        # logits = output_embedded @ self.embed.weight.T
        logits = self.digup(output_embedded)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("val_loss", loss)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy)

# %%
contrastive_clm = ContrastiveCLMCov(cfg)
contrastive_clm.summarize(max_depth=2)
# %%
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_2") # , version=2
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     enable_checkpointing=True,
                    #  callbacks=[BYOL_EMA(cfg.ema_tau)],
                    #  limit_train_batches=100,  # 1.0 
                    #  limit_val_batches=10,  # 1.0 
                     log_every_n_steps=50,
                     logger=csv_logger)

# %%
trainer.fit(contrastive_clm, wikitext2)
checkpoint_version = contrastive_clm.logger.version

# %%
class ContrastiveCLMTune(BaseModule):
    def __init__(self, cfg:ContrastiveCLMCfg, checkpoint_path, is_fine_tune:bool=False):
        super().__init__(cfg)
        self.save_hyperparameters('cfg')
        self.contrastive_clm = ContrastiveCLM.load_from_checkpoint(checkpoint_path)
        if is_fine_tune:
            set_requires_grad(self.contrastive_clm, False)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        torch.nn.init.xavier_normal_(self.digup.weight.data)

        # self.digup.weight.data = self.contrastive_clm.embed.weight.clone().detach()

    def forward(self, input):
        input_embedded = self.contrastive_clm.embed(input)

        output_embedded = self.contrastive_clm.layers(input_embedded)
        logits = self.digup(output_embedded)
        return logits

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log(mode + "_loss", loss)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

# %%
import os
checkpoint_path = os.path.join(f"./lightning_logs/wikitext_2/version_{checkpoint_version}/checkpoints/epoch=0-step=164.ckpt")
latent_clm2 = ContrastiveCLMTune(cfg, checkpoint_path, is_fine_tune=False)
# %%
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_2") # , version=2
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     enable_checkpointing=True,
                    #  callbacks=[BYOL_EMA(cfg.ema_tau)],
                    #  limit_train_batches=100,  # 1.0 
                    #  limit_val_batches=10,  # 1.0 
                     log_every_n_steps=50,
                     logger=csv_logger)

# %%
trainer.fit(latent_clm2, wikitext2)
# checkpoint_version = contrastive_clm.logger.version


# %%
# for i, (input, target) in enumerate(wikitext2.train_dataloader()):
output_embedded, target_embedded = contrastive_clm(input, target)
output_embedded.shape, target_embedded.shape
# %%
torch.std(target_embedded, dim=1), torch.mean(target_embedded, dim=1)
# %%
torch.std(output_embedded, dim=1), torch.mean(output_embedded, dim=1)
# %%
-nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
# %%
target_embedded[0,0]
# %%
target_embedded[0,1]

# %%
output_embedded[0,0]
# %%
output_embedded[0,1]
# %%
