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
from functools import partial

import numpy as np
import pandas as pd
import pytest
from modelstore.models.common import save_joblib
from modelstore.models.sklearn import (
    MODEL_JOBLIB,
    SKLearnManager,
    _feature_importances,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def sklearn_tree():
    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_split": 5,
        "n_jobs": 1,
    }
    return RandomForestRegressor(**params)


@pytest.fixture
def sklearn_logistic():
    return LogisticRegression()


@pytest.fixture
def sklearn_pipeline(sklearn_tree):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", sklearn_tree),
        ]
    )


@pytest.fixture
def sklearn_manager():
    return SKLearnManager()


@pytest.mark.parametrize(
    "model_type,expected",
    [
        (
            RandomForestRegressor,
            {"library": "sklearn", "type": "RandomForestRegressor"},
        ),
        (
            LogisticRegression,
            {"library": "sklearn", "type": "LogisticRegression"},
        ),
        (
            partial(
                Pipeline, steps=[("regressor", RandomForestRegressor(n_jobs=1))]
            ),
            {"library": "sklearn", "type": "Pipeline"},
        ),
    ],
)
def test_model_info(sklearn_manager, model_type, expected):
    res = sklearn_manager._model_info(model=model_type())
    assert expected == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("sklearn", True),
        ("xgboost", False),
    ],
)
def test_is_same_library(sklearn_manager, ml_library, should_match):
    assert (
        sklearn_manager._is_same_library({"library": ml_library})
        == should_match
    )


def test_model_data(sklearn_manager, sklearn_tree):
    labels = np.array([0, 1, 1, 0, 1])
    exp = {"labels": {"shape": [5], "values": {0: 2, 1: 3}}}
    res = sklearn_manager._model_data(model=sklearn_tree, y_train=labels)
    assert exp == res


def test_required_kwargs(sklearn_manager):
    assert sklearn_manager._required_kwargs() == ["model"]


def test_matches_with(sklearn_manager, sklearn_tree):
    assert sklearn_manager.matches_with(model=sklearn_tree)
    assert not sklearn_manager.matches_with(model="a-string-value")
    assert not sklearn_manager.matches_with(classifier=sklearn_tree)


def test_get_functions(sklearn_manager, sklearn_tree):
    assert len(sklearn_manager._get_functions(model=sklearn_tree)) == 1
    with pytest.raises(TypeError):
        sklearn_manager._get_functions(model="not-an-sklearn-model")


@pytest.mark.parametrize(
    "model_type",
    [
        RandomForestRegressor,
        LogisticRegression,
    ],
)
def test_get_params(sklearn_manager, model_type):
    try:
        result = sklearn_manager._get_params(model=model_type())
        json.dumps(result)
    except Exception as e:
        pytest.fail(f"Exception when dumping params: {str(e)}")


def test_get_params_from_pipeline(sklearn_manager):
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(), ["columns"]),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("classifier", RandomForestRegressor()),
        ]
    )
    result = sklearn_manager._get_params(model=pipeline)
    assert result == {}


def test_feature_importances_tree_model(sklearn_tree, classification_data):
    X_train, y_train = classification_data
    df = pd.DataFrame(
        X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])]
    )
    sklearn_tree.fit(df, y_train)
    exp = dict(zip(df, sklearn_tree.feature_importances_))
    res = _feature_importances(sklearn_tree, df)
    assert exp == res


def test_feature_importances_pipeline(sklearn_pipeline, classification_data):
    X_train, y_train = classification_data
    df = pd.DataFrame(
        X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])]
    )
    sklearn_pipeline.fit(df, y_train)
    exp = {"regressor": sklearn_pipeline.steps[1][1].feature_importances_}
    res = _feature_importances(sklearn_pipeline, df)
    assert (exp["regressor"] == res["regressor"]).all()


def test_load_model(tmp_path, sklearn_manager, sklearn_tree):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, sklearn_tree, MODEL_JOBLIB)
    assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

    #  Load the model
    loaded_model = sklearn_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert type(loaded_model) == type(sklearn_tree)
    assert loaded_model.get_params() == sklearn_tree.get_params()
