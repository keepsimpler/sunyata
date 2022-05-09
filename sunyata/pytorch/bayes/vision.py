from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from einops import repeat
from einops.layers.torch import Rearrange

from sunyata.pytorch.bayes.core import log_bayesian_iteration, DeepBayesInferCfg

@dataclass
class DeepBayesInferVisionCfg(DeepBayesInferCfg):
    image_size: int = 64  # 224
    patch_size: int = 8  # 16
    num_classes: int = 200

    is_mask=False

    pool: str = 'cls' # or 'mean'
    channels: int = 3

    emb_dropout: float = 0. 


class DeepBayesInferVision(pl.LightningModule):
    def __init__(self, layers: nn.ModuleList, cfg: DeepBayesInferVisionCfg, steps_per_epoch: int=None):
        super().__init__()

        self.save_hyperparameters("cfg")
        self.layers = nn.ModuleList(layers)

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
        self.steps_per_epoch = steps_per_epoch

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

    def training_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("train_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        log_posterior = self.forward(input)
        loss = F.nll_loss(log_posterior, target)
        self.log("val_loss", loss)
        accuracy = (log_posterior.argmax(dim=-1) == target).float().mean()
        self.log("val_accuracy", accuracy)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.learning_rate_scheduler == "Step":
            lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        elif self.learning_rate_scheduler == "OneCycle":
            lr_scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch, epochs=self.num_epochs)
        else:
            raise Exception("Only support StepLR and OneCycleLR learning rate schedulers now.")

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
