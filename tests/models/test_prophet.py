#    Copyright 2021 Neal Lathia
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
import random

import pytest
from modelstore.models.prophet import MODEL_FILE, ProphetManager, save_model
from prophet import Prophet

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def prophet_model():
    num_dimensions = 40
    model = AnnoyIndex(num_dimensions, "angular")
    for i in range(1000):
        vector = [random.gauss(0, 1) for z in range(num_dimensions)]
        model.add_item(i, vector)
    model.build(10)
    return model


@pytest.fixture
def prophet_manager():
    return ProphetManager()


def assert_same_model(model_a: Prophet, model_b: Prophet):
    assert type(model_a) == type(model_b)
    assert model_a.params == model_b.params


def test_model_info(prophet_manager, prophet_model):
    exp = {"library": "prophet", "type": "Prophet"}
    res = prophet_manager._model_info(model=prophet_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("prophet", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(prophet_manager, ml_library, should_match):
    assert (
        prophet_manager._is_same_library({"library": ml_library})
        == should_match
    )


def test_model_data(prophet_manager, prophet_model):
    exp = {}
    res = prophet_manager._model_data(model=prophet_model)
    assert exp == res


def test_required_kwargs(prophet_manager):
    assert prophet_manager._required_kwargs() == ["model"]


def test_matches_with(prophet_manager, prophet_model):
    assert prophet_manager.matches_with(model=prophet_model)
    assert not prophet_manager.matches_with(model="a-string-value")
    assert not prophet_manager.matches_with(catboost_model=prophet_model)


def test_get_functions(prophet_manager, prophet_model):
    assert len(prophet_manager._get_functions(model=prophet_model)) == 1


def test_get_params(prophet_manager, prophet_model):
    exp = {
        "num_dimensions": annoy_model.f,
        "num_trees": 10,
        "metric": "angular",
    }
    res = prophet_manager._get_params(model=prophet_model)
    assert exp == res


def test_save_model(tmp_path, prophet_model):
    exp = os.path.join(tmp_path, MODEL_FILE)
    res = save_model(tmp_path, model=prophet_model)
    assert os.path.exists(exp)
    assert res == exp

    loaded_model = AnnoyIndex(annoy_model.f, "angular")
    loaded_model.load(res)
    assert_same_model(prophet_model, loaded_model)


def test_load_model(tmp_path, prophet_manager, prophet_model):
    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, MODEL_FILE)
    annoy_model.save(model_path)

    #  Load the model
    loaded_model = prophet_manager.load(
        tmp_path,
        {
            "model": {
                "parameters": {"num_dimensions": 40, "metric": "angular"},
            }
        },
    )

    # Expect the two to be the same
    assert_same_model(prophet_model, loaded_model)
