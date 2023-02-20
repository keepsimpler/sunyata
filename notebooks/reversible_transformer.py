# %%
from dataclasses import dataclass
from typing import Iterable, Callable
from sunyata.pytorch.data.wikitext import WikiTextDataModule, shift_one_token
from sunyata.pytorch.layer.transformer import RevTransformerLayer, TransformerCfg

import pytorch_lightning as pl
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
# batch_size, seq_len, hidden_dim = 2, 3, 4
# # %%
# cfg = TransformerCfg(
#     hidden_dim=hidden_dim,
#     num_heads=1,
#     expanded_dim=2,
# )
# # %%
# rev_transformer = RevTransformerLayer(cfg)
# # %%
# rev_transformer
# # %%
# x1 = torch.randn(batch_size, seq_len, hidden_dim)
# x2 = torch.randn(batch_size, seq_len, hidden_dim)
# # %%
# y1, y2 = rev_transformer.forward1(x1, x2)
# # %%
# recovered_x1, recovered_x2 = rev_transformer.forward2(y1, y2)
# # %%
# assert torch.allclose(x1, recovered_x1) and torch.allclose(x2, recovered_x2)
# # %%
# rev_transformers = nn.Sequential(
#     *[RevTransformerLayer(cfg) for _ in range(3)]
# )
# # %%
# for rev_transformer in rev_transformers:
#     x1, x2 = rev_transformer.forward1(x1, x2)
# # %%
# for rev_transformer in rev_transformers[::-1]:
#     x1, x2 = rev_transformer.forward2(x1, x2)
# %%
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch_lightning.base import BaseModule

# %%
@dataclass
class RevTransformerCLMCfg(BaseCfg):
    vocab_size: int = None
    seq_len: int = None
    hidden_dim: int = None
    
    transformer: TransformerCfg = None


# %%
class RevTransformerCLM(BaseModule):
    """
    Transformer for Causal Language Modeling.    
    """
    def __init__(self, cfg: RevTransformerCLMCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.layers = nn.Sequential(*[RevTransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)])

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        # torch.nn.init.normal_(self.embed.weight, std=0.1)
        # torch.nn.init.xavier_normal_(self.embed.weight)
        self.digup = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.digup.weight = self.embed.weight
        
        self.cfg = cfg
        
    def forward1(self, input_embedded:torch.Tensor):
        # with torch.no_grad():
        #     input_embedded = self.embed(input)
        batch_size, seq_len, hidden_dim = input_embedded.shape
        assert hidden_dim % 2 == 0
        x1 = input_embedded[:,:,:hidden_dim//2]
        x2 = input_embedded[:,:,-hidden_dim//2:]

        for layer in self.layers:
            x1, x2 = layer.forward1(x1, x2)

        output_embedded = torch.cat((x1, x2), dim=-1)
        return output_embedded

    def forward2(self, input_embedded:torch.Tensor):
        # with torch.no_grad():
        #     input_embedded = self.embed(input)
        batch_size, seq_len, hidden_dim = input_embedded.shape
        assert hidden_dim % 2 == 0
        x1 = input_embedded[:,:,:hidden_dim//2]
        x2 = input_embedded[:,:,-hidden_dim//2:]

        for layer in self.layers[::-1]:
            x1, x2 = layer.forward2(x1, x2)

        output_embedded = torch.cat((x1, x2), dim=-1)
        return output_embedded

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        # for layer in self.layers:
        #     layer.attention.fore_mask = False

        with torch.no_grad():
            input_embedded = self.embed(input)
        output_embedded = self.forward1(input_embedded)
        with torch.no_grad():
            target_embedded = self.embed(target)
            # target_embedded.detach_()
        output_embedded_2 = self.forward2(target_embedded)

        # cosine_loss = nn.SmoothL1Loss()(output_embedded, target_embedded)
        cosine_loss = nn.MSELoss()(output_embedded, target_embedded)
        # cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
        # cosine_loss = 2 - 2 * (output_embedded * target_embedded).sum(dim=(-1,)).mean()
        self.log(mode + "_cosine_loss", cosine_loss, prog_bar=True)
        cosine_loss_2 = nn.MSELoss()(output_embedded_2, input_embedded)
        # cosine_loss_2 = - nn.CosineSimilarity(dim=-1)(output_embedded_2, input_embedded).mean()
        self.log(mode + "_cosine_loss_2", cosine_loss_2, prog_bar=True)
        logits = self.digup(output_embedded)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target)
        self.log("val_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy, prog_bar=True)
        return cosine_loss

    # def validation_step(self, batch, batch_idx):
    #     input, target = batch
    #     for layer in self.layers:
    #         layer.attention.is_mask = False
    #     output_embedded = self.forward1(input)
    #     target_embedded = self.embed(target)
    #     # logits = output_embedded @ self.embed.weight.T
    #     logits = self.digup(output_embedded)
    #     loss = F.cross_entropy(logits.permute(0, 2, 1), target)
    #     self.log("val_loss", loss, prog_bar=True)
    #     accuracy = (logits.argmax(dim=-1) == target).float().mean()
    #     self.log("val_accuracy", accuracy, prog_bar=True)
    #     cosine_loss = - nn.CosineSimilarity(dim=-1)(output_embedded, target_embedded).mean()
    #     self.log("val_cosine_loss", cosine_loss, prog_bar=True)


# %%
hidden_dim = 64
cfg = RevTransformerCLMCfg(
    vocab_size = 1000,
    seq_len = 256,
    hidden_dim = hidden_dim,
    transformer = TransformerCfg(
        hidden_dim = hidden_dim // 2,
        num_heads = 1,
        expanded_dim= 2*hidden_dim,
        is_softmax=True,
        fore_mask=True,
    ),

    batch_size = 64,
    num_layers = 1,
    num_epochs = 2,
    learning_rate = 1e-1, # 1e-3  3e-4
    optimizer_method = "RevSGD",
)
# %%
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=cfg.batch_size,
                   vocab_size=cfg.vocab_size,
                   seq_len=cfg.seq_len,
                   collate_fn=shift_one_token,
                   is_shuffle=False)  # shift_one_token  None
# %%
wikitext2.tokenizer.decode(wikitext2.train_data[0].tolist(), skip_special_tokens=False)
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
# https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

# %%
input, target = next(iter(wikitext2.train_dataloader()))
input.shape, target.shape

# %%
rev_transformer_clm = RevTransformerCLM(cfg)
rev_transformer_clm.summarize(max_depth=2)

# %%
csv_logger = pl.loggers.CSVLogger(save_dir="lightning_logs/", 
    name="wikitext_2") # , version=2
trainer = pl.Trainer(gpus=1, 
                     max_epochs=cfg.num_epochs, 
                     enable_checkpointing=False,
                    #  callbacks=[BYOL_EMA(cfg.ema_tau, cfg.center_tau)],
                    #  callbacks=[AdjustLR()],
                    #  limit_train_batches=100,  # 1.0 
                    #  limit_val_batches=10,  # 1.0 
                     log_every_n_steps=200,
                     logger=csv_logger)

# %%
trainer.fit(rev_transformer_clm, wikitext2)

# %%
