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
import torch
from modelstore.models.pytorch import (
    MODEL_PT,
    PyTorchManager,
    _save_model,
    _save_state_dict,
)
from torch import nn, optim
from torch.nn import functional as F

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-class-docstring


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def assert_models_equal(model_a: nn.Module, model_b: nn.Module):
    assert type(model_a) == type(model_b)
    for a_params, lb_params in zip(model_a.parameters(), model_b.parameters()):
        assert a_params.data.ne(lb_params.data).sum() == 0


@pytest.fixture
def pytorch_model():
    return ExampleNet()


@pytest.fixture
def pytorch_optim(pytorch_model):
    return optim.SGD(pytorch_model.parameters(), lr=0.001, momentum=0.9)


@pytest.fixture
def pytorch_manager():
    return PyTorchManager()


def test_model_info(pytorch_manager, pytorch_model):
    exp = {"library": "pytorch", "type": "ExampleNet"}
    res = pytorch_manager._model_info(model=pytorch_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("pytorch", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(pytorch_manager, ml_library, should_match):
    assert (
        pytorch_manager._is_same_library({"library": ml_library})
        == should_match
    )


def test_model_data(pytorch_manager, pytorch_model):
    exp = {}
    res = pytorch_manager._model_data(model=pytorch_model)
    assert exp == res


def test_required_kwargs(pytorch_manager):
    assert pytorch_manager._required_kwargs() == ["model"]


def test_matches_with(pytorch_manager, pytorch_model, pytorch_optim):
    assert pytorch_manager.matches_with(
        model=pytorch_model, optimizer=pytorch_optim
    )
    assert not pytorch_manager.matches_with(model="a-string-value")
    assert not pytorch_manager.matches_with(classifier=pytorch_model)


def test_get_functions(pytorch_manager, pytorch_model, pytorch_optim):
    assert (
        len(
            pytorch_manager._get_functions(
                model=pytorch_model, optimizer=pytorch_optim
            )
        )
        == 2
    )


def test_get_params(pytorch_manager, pytorch_model, pytorch_optim):
    exp = pytorch_optim.state_dict()
    res = pytorch_manager._get_params(
        model=pytorch_model, optimizer=pytorch_optim
    )
    assert exp == res


def test_save_model(pytorch_model, tmp_path):
    exp = os.path.join(tmp_path, "model.pt")
    file_path = _save_model(tmp_path, pytorch_model)
    assert exp == file_path

    model = torch.load(file_path)
    assert_models_equal(pytorch_model, model)


def test_save_state_dict(pytorch_model, pytorch_optim, tmp_path):
    exp = os.path.join(tmp_path, "checkpoint.pt")
    file_path = _save_state_dict(tmp_path, pytorch_model, pytorch_optim)
    assert file_path == exp

    state_dict = torch.load(file_path)
    model = ExampleNet()

    model.load_state_dict(state_dict["model_state_dict"])
    assert_models_equal(pytorch_model, model)


def test_save_state_dict_without_optimizer(pytorch_model, tmp_path):
    exp = os.path.join(tmp_path, "checkpoint.pt")
    file_path = _save_state_dict(tmp_path, pytorch_model)
    assert file_path == exp

    state_dict = torch.load(file_path)
    model = ExampleNet()

    model.load_state_dict(state_dict["model_state_dict"])
    assert_models_equal(pytorch_model, model)


def test_load_model(tmp_path, pytorch_manager, pytorch_model):
    # Save the model to a tmp directory
    torch.save(pytorch_model, os.path.join(tmp_path, MODEL_PT))

    #  Load the model
    loaded_model = pytorch_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert_models_equal(pytorch_model, loaded_model)
