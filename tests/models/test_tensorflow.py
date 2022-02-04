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
import json

import numpy as np
import pytest
import tensorflow as tf
from modelstore.models.tensorflow import (
    MODEL_DIRECTORY,
    TensorflowManager,
    _save_model,
    _save_weights,
    save_json,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
tf.config.threading.set_inter_op_parallelism_threads(1)


@pytest.fixture()
def tf_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


@pytest.fixture
def tf_manager():
    return TensorflowManager()


def assert_models_equal(
    model_a: tf.keras.Model, model_b: tf.keras.Model, assert_predictions: bool = True
):
    # Same high-level structure
    assert type(model_a) == type(model_b)
    assert model_a.count_params() == model_b.count_params()
    assert len(model_a.layers) == len(model_b.layers)

    # Same structure
    for i in range(len(model_a.layers)):
        assert (
            model_a.layers[i].__class__.__name__ == model_b.layers[i].__class__.__name__
        )

    # Same predictions
    if assert_predictions:
        test_input = np.random.random((128, 10))
        np.testing.assert_allclose(
            model_a.predict(test_input), model_b.predict(test_input)
        )


def test_model_info(tf_manager):
    exp = {"library": "tensorflow"}
    res = tf_manager._model_info()
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("tensorflow", True),
        ("keras", True),
        ("xgboost", False),
    ],
)
def test_is_same_library(tf_manager, ml_library, should_match):
    assert tf_manager._is_same_library({"library": ml_library}) == should_match


def test_model_data(tf_manager, tf_model):
    exp = {}
    res = tf_manager._model_data(model=tf_model)
    assert exp == res


def test_required_kwargs(tf_manager):
    assert tf_manager._required_kwargs() == ["model"]


def test_matches_with(tf_manager, tf_model):
    assert tf_manager.matches_with(model=tf_model)
    assert not tf_manager.matches_with(model="a-string-value")
    assert not tf_manager.matches_with(classifier=tf_model)


def test_get_functions(tf_manager, tf_model):
    assert len(tf_manager._get_functions(model=tf_model)) == 3


def test_get_params(tf_manager, tf_model):
    exp = tf_model.optimizer.get_config()
    res = tf_manager._get_params(model=tf_model)
    assert exp == res


def test_save_model(tmp_path, tf_model):
    exp = os.path.join(tmp_path, "model")
    model_path = _save_model(tmp_path, tf_model)
    assert exp == model_path
    assert os.path.isdir(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    assert_models_equal(tf_model, loaded_model)


def test_save_weights(tf_model, tmp_path):
    exp = os.path.join(tmp_path, "checkpoint")
    file_path = _save_weights(tmp_path, model=tf_model)
    assert file_path == exp
    assert os.path.isfile(file_path)

    loaded_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    loaded_model.load_weights(file_path).expect_partial()
    assert_models_equal(tf_model, loaded_model)


def test_model_json(tf_model, tmp_path):
    exp = os.path.join(tmp_path, "model_config.json")
    file_path = save_json(tmp_path, "model_config.json", tf_model.to_json())
    assert file_path == exp
    with open(file_path, "r") as lines:
        model_json = json.loads(lines.read())
    model = tf.keras.models.model_from_json(model_json)
    assert_models_equal(model, tf_model, assert_predictions=False)


def test_load_model(tmp_path, tf_manager, tf_model):
    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, MODEL_DIRECTORY)
    tf_model.save(model_path)

    #  Load the model
    loaded_model = tf_manager.load(tmp_path, {})

    # Expect the two to be the same
    assert_models_equal(tf_model, loaded_model)
