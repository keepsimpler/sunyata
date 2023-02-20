import torch
import pytorch_lightning as pl

from sunyata.pytorch.arch.base import BaseCfg, RevSGD


class BaseModule(pl.LightningModule):
    def __init__(self, cfg:BaseCfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "RevSGD":
            optimizer = RevSGD(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr, last_epoch=self.cfg.last_epoch)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]   

