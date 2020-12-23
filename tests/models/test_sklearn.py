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
import numpy as np
import pandas as pd
import pytest
from modelstore.models.sklearn import SKLearnManager, _feature_importances
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def sklearn_model():
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "ls",
    }
    return GradientBoostingRegressor(**params)


@pytest.fixture
def sklearn_manager():
    return SKLearnManager()


def test_model_info(sklearn_manager, sklearn_model):
    exp = {"library": "sklearn", "type": "GradientBoostingRegressor"}
    res = sklearn_manager._model_info(model=sklearn_model)
    assert exp == res


def test_model_data(sklearn_manager, sklearn_model):
    labels = np.array([0, 1, 1, 0, 1])
    exp = {"labels": {"shape": [5], "values": {0: 2, 1: 3}}}
    res = sklearn_manager._model_data(model=sklearn_model, y_train=labels)
    assert exp == res


def test_required_kwargs(sklearn_manager):
    assert sklearn_manager._required_kwargs() == ["model"]


def test_get_functions(sklearn_manager, sklearn_model):
    assert len(sklearn_manager._get_functions(model=sklearn_model)) == 1
    with pytest.raises(TypeError):
        sklearn_manager._get_functions(model="not-an-sklearn-model")


def test_get_params(sklearn_manager, sklearn_model):
    exp = sklearn_model.get_params()
    res = sklearn_manager._get_params(model=sklearn_model)
    assert exp == res


def test_feature_importances(sklearn_model):
    X_train, y_train = make_classification(
        n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
    )
    df = pd.DataFrame(
        X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])]
    )
    sklearn_model.fit(df, y_train)
    exp = {f: w for f, w in zip(df, sklearn_model.feature_importances_)}
    res = _feature_importances(sklearn_model, df)
    assert exp == res
