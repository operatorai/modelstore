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
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def xgboost_manager():
    return xgboost.XGBoostManager()


def test_model_info(xgboost_manager, xgb_model):
    exp = {"library": "xgboost", "type": "XGBClassifier"}
    res = xgboost_manager._model_info(model=xgb_model)
    assert exp == res


def test_model_data(xgboost_manager, xgb_model):
    exp = {}
    res = xgboost_manager._model_data(model=xgb_model)
    assert exp == res


def test_required_kwargs(xgboost_manager):
    assert xgboost_manager._required_kwargs() == ["model"]


def test_get_functions(xgboost_manager):
    assert len(xgboost_manager._get_functions(model="model")) == 3


def test_get_params(xgboost_manager, xgb_model):
    exp = xgb_model.get_xgb_params()
    res = xgboost_manager._get_params(model=xgb_model)
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
