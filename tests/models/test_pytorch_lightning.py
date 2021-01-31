#    Copyright 2020 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from modelstore.models.pytorch_lightning import (
    PyTorchLightningManager,
    _save_lightning_model,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
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
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        val_loss = F.mse_loss(y_hat, y)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return [optimizer], []


@pytest.fixture
def pytorchlightning_model():
    return ExampleLightningNet()


@pytest.fixture
def train_loader():
    x = torch.rand(20, 10)
    y = torch.rand(20).view(-1, 1)
    data_set = TensorDataset(x, y)
    return DataLoader(data_set, num_workers=0)


@pytest.fixture
def val_loader():
    x = torch.rand(2, 10)
    y = torch.rand(2).view(-1, 1)
    data_set = TensorDataset(x, y)
    return DataLoader(data_set, num_workers=0)


@pytest.fixture
def pytorchlightning_trainer(
    tmp_path, pytorchlightning_model, train_loader, val_loader
):
    trainer = pl.Trainer(max_epochs=5, default_root_dir=tmp_path)
    trainer.fit(pytorchlightning_model, train_loader, val_loader)
    return trainer


@pytest.fixture
def pytorchlightning_manager():
    return PyTorchLightningManager()


def test_model_info(pytorchlightning_manager, pytorchlightning_model):
    exp = {"library": "pytorch_lightning", "type": "ExampleLightningNet"}
    res = pytorchlightning_manager._model_info(model=pytorchlightning_model)
    assert exp == res


def test_model_data(pytorchlightning_manager, pytorchlightning_model):
    exp = {}
    res = pytorchlightning_manager._model_data(model=pytorchlightning_model)
    assert exp == res


def test_required_kwargs(pytorchlightning_manager):
    assert pytorchlightning_manager._required_kwargs() == ["trainer", "model"]


def test_get_functions(pytorchlightning_manager):
    assert (
        len(
            pytorchlightning_manager._get_functions(
                trainer="trainer", model="model"
            )
        )
        == 1
    )


def test_get_params(pytorchlightning_manager):
    exp = {}
    res = pytorchlightning_manager._get_params()
    assert exp == res


def models_equal(model_a: nn.Module, module_b: nn.Module):
    for a_params, lb_params in zip(model_a.parameters(), module_b.parameters()):
        assert a_params.data.ne(lb_params.data).sum() == 0


def test_save_model(pytorchlightning_model, pytorchlightning_trainer, tmp_path):
    exp = os.path.join(tmp_path, "checkpoint.pt")
    file_path = _save_lightning_model(
        tmp_path, pytorchlightning_trainer, pytorchlightning_model
    )
    assert exp == file_path

    model = ExampleLightningNet.load_from_checkpoint(file_path)
    models_equal(pytorchlightning_model, model)
