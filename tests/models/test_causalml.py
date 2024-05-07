#    Copyright 2024 Neal Lathia
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
import pandas as pd

import pytest
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import BaseSRegressor, XGBRRegressor
from causalml.propensity import ElasticNetPropensityModel

from modelstore.metadata import metadata
from modelstore.models.common import save_joblib
from modelstore.models.causalml import MODEL_FILE, CausalMLManager

# pylint: disable=unused-import
from tests.models.utils import classification_data


# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def causalml_meta_learner(classification_data) -> BaseSRegressor:
    X_train, y_train = classification_data

    # Simulate dummy date for a control / treatment group
    X_train = pd.DataFrame(X_train)
    X_train["experiment"] = np.tile([0, 1], X_train.shape[0] // 2)

    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_split": 5,
        "n_jobs": 1,
    }
    model = BaseSRegressor(learner=RandomForestRegressor(**params))
    model.fit(X=X_train, treatment=X_train["experiment"], y=y_train)
    return model


@pytest.fixture
def causalml_propensity_model() -> ElasticNetPropensityModel:
    params = {
        "penalty": "elasticnet",
        "solver": "saga",
        "Cs": 0.01,
    }
    return ElasticNetPropensityModel(**params)


@pytest.fixture
def causalml_regressor() -> XGBRRegressor:
    params = {
        "effect_learner_objective": "reg:squarederror",
        "effect_learner_n_estimators": 10,
    }
    return XGBRRegressor(**params)


@pytest.fixture
def causalml_manager():
    return CausalMLManager()


@pytest.mark.parametrize(
    "model_type,expected",
    [
        (
                BaseSRegressor,
                metadata.ModelType("causalml", "BaseSRegressor", None),
        ),
        (
                ElasticNetPropensityModel,
                metadata.ModelType("causalml", "ElasticNetPropensityModel", None),
        ),
        (
                XGBRRegressor,
                metadata.ModelType("causalml", "XGBRRegressor", None),
        ),
    ],
)
def test_model_info(causalml_manager, model_type, expected):
    res = causalml_manager.model_info(model=model_type())
    assert expected == res


def test_model_data(causalml_manager, causalml_meta_learner):
    res = causalml_manager.model_data(model=causalml_meta_learner)
    assert res is None


def test_required_kwargs(causalml_manager):
    assert causalml_manager._required_kwargs() == ["model"]


def test_matches_with(causalml_manager, causalml_meta_learner, causalml_propensity_model, causalml_regressor):
    assert causalml_manager.matches_with(model=causalml_meta_learner)
    assert causalml_manager.matches_with(model=causalml_propensity_model)
    assert causalml_manager.matches_with(model=causalml_regressor)
    assert not causalml_manager.matches_with(model="a-string-value")
    assert not causalml_manager.matches_with(classifier=causalml_meta_learner)


def test_get_functions(causalml_manager, causalml_meta_learner):
    assert len(causalml_manager._get_functions(model=causalml_meta_learner)) == 1
    with pytest.raises(TypeError):
        causalml_manager._get_functions(model="not-a-causalml-model")


@pytest.mark.parametrize(
    "model_type",
    [
        XGBRRegressor,
        ElasticNetPropensityModel,
    ],
)
def test_get_params(causalml_manager, model_type):
    try:
        result = causalml_manager.get_params(model=model_type())
        json.dumps(result)
        # pylint: disable=broad-except
    except Exception as exc:
        pytest.fail(f"Exception when dumping params: {str(exc)}")


def test_load_model(tmp_path, causalml_manager, causalml_meta_learner):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, causalml_meta_learner, MODEL_FILE)
    assert model_path == os.path.join(tmp_path, MODEL_FILE)

    #  Load the model
    loaded_model = causalml_manager.load(tmp_path, None)

    # Expect the two to be the same
    assert type(loaded_model) == type(causalml_meta_learner)
