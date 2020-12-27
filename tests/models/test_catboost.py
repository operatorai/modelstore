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

import catboost as ctb
import pytest
from modelstore.models import catboost
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def catb_model(tmpdir, classification_data):
    model = ctb.CatBoostClassifier(
        loss_function="MultiClass", train_dir=str(tmpdir)
    )
    X_train, y_train = classification_data
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def catboost_manager():
    return catboost.CatBoostManager()


def test_model_info(catboost_manager, catb_model):
    exp = {"library": "catboost", "type": "CatBoostClassifier"}
    res = catboost_manager._model_info(model=catb_model)
    assert exp == res


def test_model_data(catboost_manager, catb_model):
    exp = {}
    res = catboost_manager._model_data(model=catb_model)
    assert exp == res


def test_required_kwargs(catboost_manager):
    assert catboost_manager._required_kwargs() == ["model"]


def test_get_functions(catboost_manager, catb_model):
    assert len(catboost_manager._get_functions(model=catb_model)) == 4


def test_get_params(catboost_manager, catb_model):
    exp = catb_model.get_params()
    res = catboost_manager._get_params(model=catb_model)
    assert exp == res


@pytest.mark.parametrize("fmt", ["json", "cbm", "onnx"])
def test_save_model(fmt, catb_model, tmp_path):
    exp = os.path.join(tmp_path, f"model.{fmt}")
    res = catboost.save_model(tmp_path, model=catb_model, fmt=fmt)
    assert os.path.exists(exp)
    assert res == exp


def test_dump_attributes(catb_model, tmp_path):
    res = catboost.dump_attributes(tmp_path, catb_model)
    exp = os.path.join(tmp_path, "model_attributes.json")
    assert os.path.exists(exp)
    assert res == exp

    config_keys = [
        "tree_count",
        "random_seed",
        "learning_rate",
        "feature_names",
        "feature_importances",
        "evals_result",
        "best_score",
    ]
    with open(res, "r") as lines:
        data = json.loads(lines.read())
    assert all(x in data for x in config_keys)
