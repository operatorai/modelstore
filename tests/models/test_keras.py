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
import pytest
from modelstore.models.keras import (
    MODEL_DIRECTORY,
    KerasManager,
    _save_model,
    save_json,
)
from tensorflow import keras

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def keras_model():
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


@pytest.fixture
def keras_manager():
    return KerasManager()


def assert_models_equal(
    model_a: keras.Model, model_b: keras.Model, assert_predictions: bool
):
    # Same high-level structure
    assert type(model_a) == type(model_b)
    assert model_a.count_params() == model_b.count_params()
    assert len(model_a.layers) == len(model_b.layers)

    # Same structure
    for i in range(len(model_a.layers)):
        assert (
            model_a.layers[i].__class__.__name__
            == model_b.layers[i].__class__.__name__
        )
        assert model_a.layers[i].name == model_b.layers[i].name

    if assert_predictions:
        # Same predictions
        test_input = np.random.random((128, 10))
        np.testing.assert_allclose(
            model_a.predict(test_input), model_b.predict(test_input)
        )


def test_model_info(keras_manager):
    exp = {"library": "keras"}
    res = keras_manager._model_info()
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("keras", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(keras_manager, ml_library, should_match):
    assert (
        keras_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(keras_manager, keras_model):
    exp = {}
    res = keras_manager._model_data(model=keras_model)
    assert exp == res


def test_required_kwargs(keras_manager):
    assert keras_manager._required_kwargs() == ["model"]


def test_matches_with(keras_manager, keras_model):
    assert keras_manager.matches_with(model=keras_model)
    assert not keras_manager.matches_with(model="a-string-value")
    assert not keras_manager.matches_with(network=keras_model)


def test_get_functions(keras_manager, keras_model):
    assert len(keras_manager._get_functions(model=keras_model)) == 2


def test_get_params(keras_manager, keras_model):
    exp = keras_model.optimizer.get_config()
    res = keras_manager._get_params(model=keras_model)
    assert exp == res


def test_save_model(keras_model, tmp_path):
    exp = os.path.join(tmp_path, "model")
    model_path = _save_model(tmp_path, keras_model)
    assert exp == model_path
    assert os.path.isdir(model_path)

    model = keras.models.load_model(model_path)
    assert_models_equal(model, keras_model, assert_predictions=True)


def test_model_json(keras_model, tmp_path):
    exp = os.path.join(tmp_path, "model_config.json")
    file_path = save_json(tmp_path, "model_config.json", keras_model.to_json())
    assert file_path == exp
    with open(file_path, "r") as lines:
        model_json = json.loads(lines.read())
    model = keras.models.model_from_json(model_json)
    assert_models_equal(model, keras_model, assert_predictions=False)


def test_load_model(tmp_path, keras_manager, keras_model):
    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, MODEL_DIRECTORY)
    keras_model.save(model_path)

    #  Load the model
    loaded_model = keras_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert_models_equal(loaded_model, keras_model, assert_predictions=True)
