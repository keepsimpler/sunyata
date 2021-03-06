from dataclasses import dataclass
from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
from einops import repeat
from einops.layers.torch import Rearrange
import warmup_scheduler

from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration, DeepBayesInferCfg

@dataclass
class DeepBayesInferVisionCfg(DeepBayesInferCfg):
    image_size: int = 64  # 224
    patch_size: int = 8  # 16
    num_classes: int = 200

    is_mask: bool = False
    is_bayes: bool = True

    pool: str = 'cls' # or 'mean'
    channels: int = 3

    emb_dropout: float = 0. 

    warmup_epoch: int = 5


class DeepBayesInferVision(pl.LightningModule):
    def __init__(self, layers: List, cfg: DeepBayesInferVisionCfg, steps_per_epoch: int=None):
        super().__init__()

        self.save_hyperparameters("cfg")
        self.layers = nn.ModuleList(layers)
        if not cfg.is_bayes:
            self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to one layer

        image_height, image_width = pair(cfg.image_size)
        patch_height, patch_width = pair(cfg.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width
        assert cfg.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, cfg.hidden_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)

        self.pool = cfg.pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim) if cfg.is_layernorm_before_digup else nn.Identity(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        log_prior = torch.zeros(1, cfg.num_classes)
        if cfg.is_prior_as_params:
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.register_buffer('log_prior', log_prior)

        self.num_epochs, self.learning_rate, self.learning_rate_scheduler = cfg.num_epochs, cfg.learning_rate, cfg.learning_rate_scheduler
        self.optimizer_method = cfg.optimizer_method
        self.steps_per_epoch = steps_per_epoch
        self.cfg = cfg

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        log_prior = repeat(self.log_prior, '1 n -> b n', b=batch_size)        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

            x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

            logits = self.mlp_head(x_chosen)

            log_posterior = log_bayesian_iteration(log_prior, logits)
            log_prior = log_posterior

        return log_posterior

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log(mode + "_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")
    
    def configure_optimizers(self):
        if self.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            raise Exception("Only support Adam and AdamW optimizer now.")

        if self.learning_rate_scheduler == "Step":
            lr_scheduler = StepLR(optimizer, step_size=1, gamma=self.cfg.gamma)
        elif self.learning_rate_scheduler == "OneCycle":
            lr_scheduler = OneCycleLR(optimizer, max_lr=2*self.learning_rate,
                steps_per_epoch=self.steps_per_epoch, epochs=self.num_epochs)
        elif self.learning_rate_scheduler == "ReduceLROnPlateau":
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, 'min', factor=self.cfg.factor, patience=self.cfg.patience),
                "monitor": "val_loss",
                "frequency": 1
            }
        elif self.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        elif self.learning_rate_scheduler == "WarmupThenCosineAnnealing":
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
            lr_scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=self.cfg.warmup_epoch,
                                                                    after_scheduler=base_scheduler)
        else:
            lr_scheduler = None
            # raise Exception("Only support StepLR and OneCycleLR learning rate schedulers now.")
            
        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]


def pair(t):
    return t if isinstance(t, tuple) else (t, t)



if __name__ == "__main__":
    cfg = DeepBayesInferVisionCfg(batch_size=2, learning_rate=1e-3)

    image = torch.randn(cfg.batch_size, cfg.channels, cfg.image_size, cfg.image_size)
    target = torch.randint(0, cfg.num_classes, (cfg.batch_size,))

    layers = [nn.Identity()]
    deep_bayes_net = DeepBayesInferVision(layers, cfg)

    log_posterior = deep_bayes_net(image)
    posterior = torch.exp(log_posterior)

    assert torch.allclose(posterior.sum(dim=-1), torch.tensor(1.)) 
