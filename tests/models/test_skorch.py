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
import json
import os

import numpy as np
import pytest
from skorch import NeuralNetClassifier
from torch import nn

from modelstore.metadata import metadata
from modelstore.models.common import save_joblib
from modelstore.models.skorch import MODEL_JOBLIB, SkorchManager

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


class ExampleModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super(ExampleModule, self).__init__()

        self.dense0 = nn.Linear(5, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


def assert_models_equal(model_a: nn.Module, model_b: nn.Module):
    assert isinstance(model_a, type(model_b))
    for a_params, lb_params in zip(
        model_a.module_.parameters(), model_b.module_.parameters()
    ):
        assert a_params.data.ne(lb_params.data).sum() == 0


@pytest.fixture
def skorch_model(classification_data):
    X, y = classification_data
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    net = NeuralNetClassifier(
        ExampleModule,
        max_epochs=1,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    net.fit(X, y)
    return net


@pytest.fixture
def skorch_manager():
    return SkorchManager()


def test_model_info(skorch_manager, skorch_model):
    exp = metadata.ModelType("skorch", "NeuralNetClassifier", None)
    res = skorch_manager.model_info(model=skorch_model)
    assert exp == res


def test_model_data(skorch_manager, skorch_model):
    res = skorch_manager.model_data(model=skorch_model)
    assert res is None


def test_required_kwargs(skorch_manager):
    assert skorch_manager._required_kwargs() == ["model"]


def test_matches_with(skorch_manager, skorch_model):
    assert skorch_manager.matches_with(model=skorch_model)
    assert not skorch_manager.matches_with(model="a-string-value")
    assert not skorch_manager.matches_with(classifier=skorch_model)


def test_get_functions(skorch_manager, skorch_model):
    assert len(skorch_manager._get_functions(model=skorch_model)) == 2
    with pytest.raises(TypeError):
        skorch_manager._get_functions(model="not-a-skorch-model")


def test_get_params(skorch_manager, skorch_model):
    try:
        result = skorch_manager.get_params(model=skorch_model)
        json.dumps(result)
    except Exception as exc:
        pytest.fail(f"Exception when dumping params: {str(exc)}")


def test_load_model(tmp_path, skorch_manager, skorch_model):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, skorch_model, MODEL_JOBLIB)
    assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

    #  Load the model
    loaded_model = skorch_manager.load(tmp_path, None)

    # Expect the two to be the same
    assert_models_equal(loaded_model, skorch_model)
