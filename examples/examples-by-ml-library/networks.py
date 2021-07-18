import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

# pylint: disable=missing-class-docstring


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class ExampleLightningNet(pl.LightningModule):
    def __init__(self):
        super(ExampleLightningNet, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        train_loss = F.mse_loss(y_hat, y)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        val_loss = F.mse_loss(y_hat, y)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.05
        )
        return [optimizer], [scheduler]
