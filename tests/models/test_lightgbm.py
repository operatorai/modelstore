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
from modelstore.models import lightgbm
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def lgb_model(classification_data):
    X_train, y_train = classification_data
    train_data = lgb.Dataset(X_train, label=y_train)
    param = {"num_leaves": 31, "objective": "binary"}
    return lgb.train(param, train_data, 3)


@pytest.fixture
def lightgbm_manager():
    return lightgbm.LightGbmManager()


def test_model_info(lightgbm_manager, lgb_model):
    exp = {"library": "lightgbm", "type": "Booster"}
    res = lightgbm_manager._model_info(model=lgb_model)
    assert exp == res


def test_model_data(lightgbm_manager, lgb_model):
    exp = {}
    res = lightgbm_manager._model_data(model=lgb_model)
    assert exp == res


def test_required_kwargs(lightgbm_manager):
    assert lightgbm_manager._required_kwargs() == ["model"]


def test_get_functions(lightgbm_manager):
    assert len(lightgbm_manager._get_functions(model="model")) == 2


def test_get_params(lightgbm_manager, lgb_model):
    exp = {
        "num_leaves": 31,
        "objective": "binary",
        "num_iterations": 3,
        "early_stopping_round": None,
    }
    res = lightgbm_manager._get_params(model=lgb_model)
    assert exp == res


def test_save_model(lgb_model, tmp_path):
    exp = os.path.join(tmp_path, "model.txt")
    res = lightgbm.save_model(tmp_path, lgb_model)
    assert res == exp

    model = lgb.Booster(model_file=res)
    assert lgb_model.model_to_string() == model.model_to_string()


def test_dump_model(lgb_model, tmp_path):
    exp = os.path.join(tmp_path, "model.json")
    res = lightgbm.dump_model(tmp_path, lgb_model)

    assert os.path.exists(exp)
    assert res == exp
    try:
        with open(exp, "r") as lines:
            json.loads(lines.read())
    except:
        # Fail if we cannot load
        assert False
