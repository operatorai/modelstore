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

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelstore.metadata import metadata
from modelstore.models.common import save_joblib
from modelstore.models.sklearn import MODEL_JOBLIB, SKLearnManager, _feature_importances

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


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
            metadata.ModelType("sklearn", "RandomForestRegressor", None),
        ),
        (
            LogisticRegression,
            metadata.ModelType("sklearn", "LogisticRegression", None),
        ),
        (
            partial(Pipeline, steps=[("regressor", RandomForestRegressor(n_jobs=1))]),
            metadata.ModelType("sklearn", "Pipeline", None),
        ),
    ],
)
def test_model_info(sklearn_manager, model_type, expected):
    res = sklearn_manager.model_info(model=model_type())
    assert expected == res


def test_model_data(sklearn_manager, sklearn_tree):
    res = sklearn_manager.model_data(model=sklearn_tree)
    assert res is None


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
        result = sklearn_manager.get_params(model=model_type())
        json.dumps(result)
        # pylint: disable=broad-except
    except Exception as exc:
        pytest.fail(f"Exception when dumping params: {str(exc)}")


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
    result = sklearn_manager.get_params(model=pipeline)
    assert result == {}


def test_feature_importances_tree_model(sklearn_tree, classification_data):
    X_train, y_train = classification_data
    df = pd.DataFrame(X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])])
    sklearn_tree.fit(df, y_train)
    exp = dict(zip(df, sklearn_tree.feature_importances_))
    res = _feature_importances(sklearn_tree, df)
    assert exp == res


def test_feature_importances_pipeline(sklearn_pipeline, classification_data):
    X_train, y_train = classification_data
    df = pd.DataFrame(X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])])
    sklearn_pipeline.fit(df, y_train)
    exp = {"regressor": sklearn_pipeline.steps[1][1].feature_importances_}
    res = _feature_importances(sklearn_pipeline, df)
    assert (exp["regressor"] == res["regressor"]).all()


def test_load_model(tmp_path, sklearn_manager, sklearn_tree):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, sklearn_tree, MODEL_JOBLIB)
    assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

    #  Load the model
    loaded_model = sklearn_manager.load(tmp_path, None)

    # Expect the two to be the same
    assert type(loaded_model) == type(sklearn_tree)
    assert loaded_model.get_params() == sklearn_tree.get_params()
