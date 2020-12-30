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

import keras
import numpy as np
import pytest
import tensorflow as tf
from modelstore.models.tensorflow import (
    TensorflowManager,
    _save_model,
    _save_weights,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture()
def tf_model():
    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


@pytest.fixture
def tf_manager():
    return TensorflowManager()


def test_model_info(tf_manager, tf_model):
    exp = {"library": "tensorflow"}
    res = tf_manager._model_info()
    assert exp == res


def test_model_data(tf_manager, tf_model):
    exp = {}
    res = tf_manager._model_data(model=tf_model)
    assert exp == res


def test_required_kwargs(tf_manager):
    assert tf_manager._required_kwargs() == ["model"]


def test_get_functions(tf_manager, tf_model):
    assert len(tf_manager._get_functions(model=tf_model)) == 2


def test_get_params(tf_manager, tf_model):
    exp = tf_model.optimizer.get_config()
    res = tf_manager._get_params(model=tf_model)
    assert exp == res


def test_save_model(tf_model, tmp_path):
    exp = os.path.join(tmp_path, "model")
    model_path = _save_model(tmp_path, tf_model)
    assert exp == model_path
    assert os.path.isdir(model_path)

    model = tf.keras.models.load_model(model_path)
    test_input = np.random.random((128, 10))
    np.testing.assert_allclose(
        tf_model.predict(test_input), model.predict(test_input)
    )


def test_save_weights(tf_model, tmp_path):
    exp = os.path.join(tmp_path, "checkpoint")
    file_path = _save_weights(tmp_path, model=tf_model)
    assert file_path == exp
    assert os.path.isfile(file_path)

    test_input = np.random.random((128, 10))
    pre_preds = tf_model.predict(test_input)
    tf_model.load_weights(file_path)
    post_preds = tf_model.predict(test_input)
    np.testing.assert_allclose(pre_preds, post_preds)
