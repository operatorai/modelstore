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
import pandas as pd
import pytest
import statsmodels.api as sm

from modelstore.metadata import metadata
from modelstore.models.statsmodels import MODEL_PICKLE, StatsModelsManager

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def statsmodels_model(classification_data):
    X_train, y_train = classification_data
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    return model


@pytest.fixture
def statsmodels_manager():
    return StatsModelsManager()


def test_model_info(statsmodels_manager, statsmodels_model):
    res = statsmodels_manager.model_info(model=statsmodels_model)
    assert res == metadata.ModelType("statsmodels", "RegressionResultsWrapper", None)


def test_model_data(statsmodels_manager, statsmodels_model):
    res = statsmodels_manager.model_data(model=statsmodels_model)
    assert res is None


def test_required_kwargs(statsmodels_manager):
    assert statsmodels_manager._required_kwargs() == ["model"]


def test_matches_with(statsmodels_manager, statsmodels_model):
    assert statsmodels_manager.matches_with(model=statsmodels_model)
    assert not statsmodels_manager.matches_with(model="a-string-value")
    assert not statsmodels_manager.matches_with(classifier=statsmodels_model)


def test_get_functions(statsmodels_manager, statsmodels_model):
    assert len(statsmodels_manager._get_functions(model=statsmodels_model)) == 1
    with pytest.raises(TypeError):
        statsmodels_manager._get_functions(model="not-a-statsmodels-model")


def test_get_params(statsmodels_manager, statsmodels_model):
    result = statsmodels_manager.get_params(model=statsmodels_model)
    assert isinstance(result, dict)
    assert len(result) > 0
    try:
        json.dumps(result)
    except Exception as exc:
        pytest.fail(f"Params are not JSON-serialisable: {str(exc)}")


def test_load_model(tmp_path, statsmodels_manager, statsmodels_model):
    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, MODEL_PICKLE)
    statsmodels_model.save(model_path)
    assert os.path.exists(model_path)

    # Load the model
    loaded_model = statsmodels_manager.load(tmp_path, None)

    # Expect type and params to match
    assert type(loaded_model) == type(statsmodels_model)
    assert np.allclose(loaded_model.params, statsmodels_model.params)
