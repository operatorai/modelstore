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
    MODEL_CHECKPOINT,
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
def lightning_model():
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
def lightning_trainer(tmp_path, lightning_model, train_loader, val_loader):
    trainer = pl.Trainer(max_epochs=5, default_root_dir=tmp_path)
    trainer.fit(lightning_model, train_loader, val_loader)
    return trainer


@pytest.fixture
def lightning_manager():
    return PyTorchLightningManager()


def assert_models_equal(
    model_a: pl.LightningModule, module_b: pl.LightningModule
):
    for a_params, lb_params in zip(model_a.parameters(), module_b.parameters()):
        assert a_params.data.ne(lb_params.data).sum() == 0


def test_model_info(lightning_manager, lightning_model):
    exp = {"library": "pytorch_lightning", "type": "ExampleLightningNet"}
    res = lightning_manager._model_info(model=lightning_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("pytorch_lightning", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(lightning_manager, ml_library, should_match):
    assert (
        lightning_manager._is_same_library({"library": ml_library})
        == should_match
    )


def test_model_data(lightning_manager, lightning_model):
    exp = {}
    res = lightning_manager._model_data(model=lightning_model)
    assert exp == res


def test_required_kwargs(lightning_manager):
    assert lightning_manager._required_kwargs() == ["trainer", "model"]


def test_matches_with(lightning_manager, lightning_trainer, lightning_model):
    assert lightning_manager.matches_with(
        trainer=lightning_trainer, model=lightning_model
    )
    assert not lightning_manager.matches_with(model="a-string-value")
    assert not lightning_manager.matches_with(classifier=lightning_trainer)


def test_get_functions(lightning_manager, lightning_model, lightning_trainer):
    assert (
        len(
            lightning_manager._get_functions(
                trainer=lightning_trainer, model=lightning_model
            )
        )
        == 1
    )


def test_get_params(lightning_manager):
    exp = {}
    res = lightning_manager._get_params()
    assert exp == res


def test_save_model(tmp_path, lightning_model, lightning_trainer):
    exp = os.path.join(tmp_path, "checkpoint.pt")
    file_path = _save_lightning_model(tmp_path, lightning_trainer)
    assert exp == file_path

    loaded_model = ExampleLightningNet.load_from_checkpoint(file_path)
    assert_models_equal(lightning_model, loaded_model)


def test_load_model(tmp_path, lightning_manager, lightning_trainer):
    # Save the model to a tmp directory
    file_path = os.path.join(tmp_path, MODEL_CHECKPOINT)
    lightning_trainer.save_checkpoint(file_path)

    #  Load the model
    loaded_model = lightning_manager.load(
        tmp_path,
        {
            "model": {
                "model_type": {
                    "type": "ExampleLightningNet",
                }
            }
        },
    )

    # Expect the two to be the same
    assert_models_equal(lightning_trainer.model, loaded_model)
