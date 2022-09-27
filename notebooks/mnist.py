# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
# %%
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(28 * 28)
        self.embed = nn.Embedding(10, 28 * 28)
        self.digup = torch.nn.Linear(28 * 28, 10, bias=False)
        self.embed.weight = self.digup.weight
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.bn(x)
        x = self.digup(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.view(x.size(0), -1)
        target_embedded = self.embed(y)
        mse_loss = nn.MSELoss()(x, target_embedded)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return mse_loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=2e-0)
# %%
mnist_model = MNISTModel()

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_ds = MNIST('.data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=256)
# %%
trainer = Trainer(
    gpus=1,
    max_epochs=2,
    logger=CSVLogger("lightning_logs/", name="mnist"),
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
# %%
trainer.fit(mnist_model, train_loader)
# %%
