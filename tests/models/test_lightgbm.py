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

import lightgbm as lgb
import numpy as np
import pytest
from modelstore.models.lightgbm import (
    MODEL_FILE,
    MODEL_JSON,
    LightGbmManager,
    _model_file_path,
    dump_model,
    save_model,
)
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def lgb_model(classification_data):
    X_train, y_train = classification_data
    train_data = lgb.Dataset(X_train, label=y_train)
    param = {"num_leaves": 31, "objective": "binary", "num_threads": 1}
    return lgb.train(param, train_data, num_boost_round=1)


@pytest.fixture
def lgb_manager():
    return LightGbmManager()


def assert_models_equal(
    model_a: lgb.Booster, model_b: lgb.Booster, classification_data
):
    # Same type
    assert type(model_a) == type(model_b)
    assert model_a.model_to_string() == model_b.model_to_string()

    # Same predictions
    X_train, _ = classification_data
    np.testing.assert_allclose(
        model_a.predict(X_train), model_b.predict(X_train)
    )


def test_model_info(lgb_manager, lgb_model):
    exp = {"library": "lightgbm", "type": "Booster"}
    res = lgb_manager._model_info(model=lgb_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("lightgbm", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(lgb_manager, ml_library, should_match):
    assert lgb_manager._is_same_library({"library": ml_library}) == should_match


def test_model_data(lgb_manager, lgb_model):
    exp = {}
    res = lgb_manager._model_data(model=lgb_model)
    assert exp == res


def test_required_kwargs(lgb_manager):
    assert lgb_manager._required_kwargs() == ["model"]


def test_matches_with(lgb_manager, lgb_model):
    assert lgb_manager.matches_with(model=lgb_model)
    assert not lgb_manager.matches_with(model="a-string-value")
    assert not lgb_manager.matches_with(classifier=lgb_model)


def test_get_functions(lgb_manager, lgb_model):
    assert len(lgb_manager._get_functions(model=lgb_model)) == 2


def test_get_params(lgb_manager, lgb_model):
    exp = {
        "num_leaves": 31,
        "objective": "binary",
        "num_iterations": 1,
        "early_stopping_round": None,
        "num_threads": 1,
    }
    res = lgb_manager._get_params(model=lgb_model)
    assert exp == res


def test_save_model(tmp_path, lgb_model, classification_data):
    exp = os.path.join(tmp_path, "model.txt")
    res = save_model(tmp_path, lgb_model)
    assert res == exp

    loaded_model = lgb.Booster(model_file=res)
    assert_models_equal(lgb_model, loaded_model, classification_data)


def test_dump_model(tmp_path, lgb_model, classification_data):
    exp = os.path.join(tmp_path, MODEL_JSON)
    res = dump_model(tmp_path, lgb_model)

    assert os.path.exists(exp)
    assert res == exp

    # Models can't be loaded back from JSON
    # https://stackoverflow.com/questions/52170236/lightgbm-loading-from-json
    try:
        with open(res, "r") as lines:
            json.loads(lines.read())
    except:
        pytest.fail("Cannot load dumped model as JSON")


def test_load_model(tmp_path, lgb_manager, lgb_model, classification_data):
    # Save the model to a tmp directory
    model_path = _model_file_path(tmp_path)
    lgb_model.save_model(model_path)
    assert model_path == os.path.join(tmp_path, MODEL_FILE)
    assert os.path.exists(model_path)

    #  Load the model
    loaded_model = lgb_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert_models_equal(lgb_model, loaded_model, classification_data)
