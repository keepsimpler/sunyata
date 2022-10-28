
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Block


def create_model(num_classes: int, first_conv: bool=True):
    model = torchvision.models.resnext50_32x4d(pretrained=False, num_classes=num_classes)
    if not first_conv:
        model.conv1 = nn.Identity()
    model.maxpool = nn.Identity()
    return model


@dataclass
class ResNext50Cfg(BaseCfg):
    num_classes: int = 200
    first_conv: bool = True


class ResNext50(BaseModule):
    def __init__(self, cfg: ResNext50Cfg):
        super().__init__(cfg)

        self.model = create_model(cfg.num_classes, cfg.first_conv)

        self.cfg = cfg
        
    def forward(self, x):
        x = self.model(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss    
