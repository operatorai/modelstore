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
import xgboost as xgb
from modelstore.models import xgboost
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def xgb_model(classification_data):
    X_train, y_train = classification_data
    model = xgb.XGBClassifier(use_label_encoder=False, n_jobs=1)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def xgb_manager():
    return xgboost.XGBoostManager()


def test_model_info(xgb_manager, xgb_model):
    exp = {"library": "xgboost", "type": "XGBClassifier"}
    res = xgb_manager._model_info(model=xgb_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("xgboost", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(xgb_manager, ml_library, should_match):
    assert xgb_manager._is_same_library({"library": ml_library}) == should_match


def test_model_data(xgb_manager, xgb_model):
    exp = {}
    res = xgb_manager._model_data(model=xgb_model)
    assert exp == res


def test_required_kwargs(xgb_manager):
    assert xgb_manager._required_kwargs() == ["model"]


def test_matches_with(xgb_manager, xgb_model):
    assert xgb_manager.matches_with(model=xgb_model)
    assert not xgb_manager.matches_with(model="a-string-value")
    assert not xgb_manager.matches_with(classifier=xgb_model)


def test_get_functions(xgb_manager):
    assert len(xgb_manager._get_functions(model="model")) == 3


def test_get_params(xgb_manager, xgb_model):
    exp = xgb_model.get_xgb_params()
    res = xgb_manager._get_params(model=xgb_model)
    assert exp == res


def test_save_model(xgb_model, tmp_path):
    res = xgboost.save_model(tmp_path, xgb_model)
    exp = os.path.join(tmp_path, "model.xgboost")
    assert os.path.exists(exp)
    assert res == exp


def test_dump_model(xgb_model, tmp_path):
    res = xgboost.dump_model(tmp_path, xgb_model)
    exp = os.path.join(tmp_path, "model.json")
    assert os.path.exists(exp)
    assert res == exp


def test_model_config(xgb_model, tmp_path):
    res = xgboost.model_config(tmp_path, xgb_model)
    exp = os.path.join(tmp_path, "config.json")
    assert os.path.exists(exp)
    assert res == exp


def test_load_model(tmp_path, xgb_manager, xgb_model, classification_data):
    # Some fields in xgboost get_params change when loading
    # or are nans; we cannot compare them in this test
    ignore_params = ["missing", "tree_method"]

    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, xgboost.MODEL_FILE)
    xgb_model.save_model(model_path)
    xgb_model_params = xgb_model.get_params()
    for param in ignore_params:
        xgb_model_params.pop(param)

    #  Load the model
    loaded_model = xgb_manager.load(
        tmp_path,
        {
            "model": {
                "model_type": {
                    "type": "XGBClassifier",
                },
            }
        },
    )

    # Expect the two to be the same
    assert type(loaded_model) == type(xgb_model)

    # They should also have the same params
    loaded_model_params = loaded_model.get_params()
    for param in ignore_params:
        loaded_model_params.pop(param)
    assert xgb_model_params == loaded_model_params
