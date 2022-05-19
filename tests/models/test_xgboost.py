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
import numpy as np
from modelstore.models import xgboost

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def xgb_model(classification_data):
    X_train, y_train = classification_data
    model = xgb.XGBClassifier(use_label_encoder=False, n_jobs=1)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def xgb_booster(xgb_model):
    return xgb_model.get_booster()


@pytest.fixture
def xgb_manager():
    return xgboost.XGBoostManager()


def test_model_info(xgb_manager, xgb_model):
    model_info = {"library": "xgboost", "type": "XGBClassifier"}
    result = xgb_manager._model_info(model=xgb_model)
    assert model_info == result


def test_booster_model_info(xgb_manager, xgb_booster):
    model_info = {"library": "xgboost", "type": "Booster"}
    result = xgb_manager._model_info(model=xgb_booster)
    assert model_info == result


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
    # Note: currently unimplemented for xgboost
    result = xgb_manager._model_data(model=xgb_model)
    assert result == {}


def test_required_kwargs(xgb_manager):
    assert xgb_manager._required_kwargs() == ["model"]


def test_matches_with(xgb_manager, xgb_model, xgb_booster):
    assert xgb_manager.matches_with(model=xgb_model)
    assert xgb_manager.matches_with(model=xgb_booster)
    assert not xgb_manager.matches_with(model="a-string-value")
    assert not xgb_manager.matches_with(classifier=xgb_model)


def test_get_functions(xgb_manager, xgb_model):
    assert len(xgb_manager._get_functions(model=xgb_model)) == 3


def test_get_params(xgb_manager, xgb_model):
    exp = xgb_model.get_xgb_params()
    result = xgb_manager._get_params(model=xgb_model)
    assert exp == result


def test_get_booster_params(xgb_manager, xgb_booster):
    # Cannot retrieve xgb params for low-level xgboost xgb.Booster object
    result = xgb_manager._get_params(model=xgb_booster)
    assert result == {}


def test_save_model(xgb_model, tmp_path):
    res = xgboost.save_model(tmp_path, xgb_model)
    exp = os.path.join(tmp_path, "model.xgboost")
    assert os.path.exists(exp)
    assert res == exp


def test_dump_booster(xgb_booster, tmp_path):
    res = xgboost.dump_booster(tmp_path, xgb_booster)
    exp = os.path.join(tmp_path, "model.json")
    assert os.path.exists(exp)
    assert res == exp


def test_save_booster_config(xgb_booster, tmp_path):
    res = xgboost.save_booster_config(tmp_path, xgb_booster)
    exp = os.path.join(tmp_path, "config.json")
    assert os.path.exists(exp)
    assert res == exp


def test_load_model(tmp_path, xgb_manager, xgb_model, classification_data):
    # Some fields in xgboost get_params change when loading
    # or are nans; we cannot compare them in this test
    ignore_params = ["missing", "tree_method"]

    # Get the model predictions
    X_train, _ = classification_data
    y_pred = xgb_model.predict(X_train)

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
    assert isinstance(loaded_model, type(xgb_model))

    # They should have the same predictions
    y_loaded_pred = xgb_model.predict(X_train)
    assert np.allclose(y_pred, y_loaded_pred)

    # They should also have the same params
    loaded_model_params = loaded_model.get_params()
    for param in ignore_params:
        loaded_model_params.pop(param)
    assert xgb_model_params == loaded_model_params


def test_load_booster(tmp_path, xgb_manager, xgb_booster, classification_data):
    # Get the model predictions
    X_train, _ = classification_data
    y_pred = xgb_booster.predict(xgb.DMatrix(X_train))

    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, xgboost.MODEL_FILE)
    xgb_booster.save_model(model_path)

    #  Load the model
    loaded_model = xgb_manager.load(
        tmp_path,
        {
            "model": {
                "model_type": {
                    "type": "Booster",
                },
            }
        },
    )

    # Expect the two to be the same
    assert isinstance(loaded_model, type(xgb_booster))

    # They should have the same predictions
    y_loaded_pred = xgb_booster.predict(xgb.DMatrix(X_train))
    assert np.allclose(y_pred, y_loaded_pred)
