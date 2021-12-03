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
from modelstore.models.common import save_joblib
from modelstore.models.skorch import MODEL_JOBLIB, SkorchManager
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetClassifier
from tests.models.utils import classification_data
from torch import nn

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


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
    assert type(model_a) == type(model_b)
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
    exp = {"library": "skorch", "type": "NeuralNetClassifier"}
    res = skorch_manager._model_info(model=skorch_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("skorch", True),
        ("sklearn", False),
        ("xgboost", False),
    ],
)
def test_is_same_library(skorch_manager, ml_library, should_match):
    assert (
        skorch_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(skorch_manager, skorch_model):
    exp = {}
    res = skorch_manager._model_data(model=skorch_model)
    assert exp == res


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
        result = skorch_manager._get_params(model=skorch_model)
        json.dumps(result)
    except Exception as e:
        pytest.fail(f"Exception when dumping params: {str(e)}")


def test_load_model(tmp_path, skorch_manager, skorch_model):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, skorch_model, MODEL_JOBLIB)
    assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

    #  Load the model
    loaded_model = skorch_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert_models_equal(loaded_model, skorch_model)
