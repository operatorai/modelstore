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
import random

import pytest
from annoy import AnnoyIndex
from modelstore.models.annoy import MODEL_FILE, AnnoyManager, save_model

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def annoy_model():
    num_dimensions = 40
    model = AnnoyIndex(num_dimensions, "angular")
    for i in range(1000):
        vector = [random.gauss(0, 1) for z in range(num_dimensions)]
        model.add_item(i, vector)
    model.build(10)
    return model


@pytest.fixture
def annoy_manager():
    return AnnoyManager()


def assert_same_model(model_a: AnnoyIndex, model_b: AnnoyIndex):
    assert type(model_a) == type(model_b)
    assert model_a.get_nns_by_item(0, 10) == model_b.get_nns_by_item(0, 10)


def test_model_info(annoy_manager, annoy_model):
    exp = {"library": "annoy", "type": "Annoy"}
    res = annoy_manager._model_info(model=annoy_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("annoy", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(annoy_manager, ml_library, should_match):
    assert (
        annoy_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(annoy_manager, annoy_model):
    exp = {}
    res = annoy_manager._model_data(model=annoy_model)
    assert exp == res


def test_required_kwargs(annoy_manager):
    assert annoy_manager._required_kwargs() == ["model", "metric", "num_trees"]


def test_matches_with(annoy_manager, annoy_model):
    assert annoy_manager.matches_with(model=annoy_model)
    assert not annoy_manager.matches_with(model="a-string-value")
    assert not annoy_manager.matches_with(catboost_model=annoy_model)


def test_get_functions(annoy_manager, annoy_model):
    assert len(annoy_manager._get_functions(model=annoy_model)) == 1


def test_get_params(annoy_manager, annoy_model):
    exp = {
        "num_dimensions": annoy_model.f,
        "num_trees": 10,
        "metric": "angular",
    }
    res = annoy_manager._get_params(
        model=annoy_model, num_trees=10, metric="angular"
    )
    assert exp == res


def test_save_model(tmp_path, annoy_model):
    exp = os.path.join(tmp_path, MODEL_FILE)
    res = save_model(tmp_path, model=annoy_model)
    assert os.path.exists(exp)
    assert res == exp

    loaded_model = AnnoyIndex(annoy_model.f, "angular")
    loaded_model.load(res)
    assert_same_model(annoy_model, loaded_model)


def test_load_model(tmp_path, annoy_manager, annoy_model):
    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, MODEL_FILE)
    annoy_model.save(model_path)

    #  Load the model
    loaded_model = annoy_manager.load(
        tmp_path,
        {
            "model": {
                "parameters": {"num_dimensions": 40, "metric": "angular"},
            }
        },
    )

    # Expect the two to be the same
    assert_same_model(annoy_model, loaded_model)
