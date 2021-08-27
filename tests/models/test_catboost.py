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
import numpy as np
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
def catb_manager():
    return catboost.CatBoostManager()


def test_model_info(catb_manager, catb_model):
    exp = {"library": "catboost", "type": "CatBoostClassifier"}
    res = catb_manager._model_info(model=catb_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("catboost", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(catb_manager, ml_library, should_match):
    assert (
        catb_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(catb_manager, catb_model):
    exp = {}
    res = catb_manager._model_data(model=catb_model)
    assert exp == res


def test_required_kwargs(catb_manager):
    assert catb_manager._required_kwargs() == ["model"]


def test_matches_with(catb_manager, catb_model):
    assert catb_manager.matches_with(model=catb_model)
    assert not catb_manager.matches_with(model="a-string-value")
    assert not catb_manager.matches_with(catboost_model=catb_model)


def test_get_functions(catb_manager, catb_model):
    assert len(catb_manager._get_functions(model=catb_model)) == 4


def test_get_params(catb_manager, catb_model):
    exp = catb_model.get_params()
    res = catb_manager._get_params(model=catb_model)
    assert exp == res


@pytest.mark.parametrize("fmt", ["json", "cbm", "onnx"])
def test_save_model(fmt, catb_model, tmp_path, classification_data):
    exp = os.path.join(tmp_path, f"model.{fmt}")
    res = catboost.save_model(tmp_path, model=catb_model, fmt=fmt)
    assert os.path.exists(exp)
    assert res == exp

    model = ctb.CatBoostClassifier()
    model.load_model(res, format=fmt)
    X_train, _ = classification_data
    assert np.allclose(catb_model.predict(X_train), model.predict(X_train))


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


def test_load_model(tmp_path, catb_manager, catb_model):
    # Save the model to a tmp directory
    model_path = catboost.save_model(tmp_path, catb_model, fmt="cbm")
    assert model_path == os.path.join(
        tmp_path, catboost._MODEL_PREFIX.format("cbm")
    )

    #  Load the model
    loaded_model = catb_manager.load(
        tmp_path,
        {
            "model": {
                "model_type": {
                    "type": "CatBoostClassifier",
                },
            }
        },
    )

    # Expect the two to be the same
    assert type(loaded_model) == type(catb_model)
    assert loaded_model.get_params() == catb_model.get_params()
