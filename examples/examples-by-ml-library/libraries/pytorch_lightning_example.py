import tempfile

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


# pylint: disable=missing-class-docstring
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


def _train_example_model() -> ExampleLightningNet:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset(as_numpy=True)
    data_set = TensorDataset(X_test, y_test)
    val_dataloader = DataLoader(data_set)

    data_set = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(data_set)

    # Train the model
    model = ExampleLightningNet()
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = pl.Trainer(max_epochs=5, default_root_dir=tmp_dir)
        trainer.fit(model, train_dataloader, val_dataloader)

    results = mean_squared_error(y_test, model(X_test).detach().numpy())
    print(f"🔍  Fit model MSE={results}.")
    return model, trainer


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a PyTorch model
    model, trainer = _train_example_model()

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the pytorch lightning model to the "{DIABETES_DOMAIN}" domain.'
    )
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model, trainer=trainer)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the pytorch lightning "{DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(DIABETES_DOMAIN, model_id)
    model.eval()

    _, X_test, _, y_test = load_diabetes_dataset(as_numpy=True)
    results = mean_squared_error(y_test, model(X_test).detach().numpy())
    print(f"🔍  Loaded model MSE={results}.")
