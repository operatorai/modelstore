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
from typing import Tuple

import numpy as np
import onnxruntime as rt
import pytest
from modelstore.models import onnx
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def onnx_model(classification_data):
    X_train, y_train = classification_data
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    initial_type = [("float_input", FloatTensorType([None, 5]))]
    model = convert_sklearn(clr, initial_types=initial_type)
    return model


@pytest.fixture
def onnx_inference(onnx_model):
    return rt.InferenceSession(onnx_model.SerializeToString())


@pytest.fixture
def onnx_manager():
    return onnx.OnnxManager()


def get_predictions(sess: InferenceSession, classification_data: Tuple):
    X_train, _ = classification_data

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    return sess.run([label_name], {input_name: X_train.astype(np.float32)})[0]


def test_model_info(onnx_manager, onnx_model):
    exp = {"library": "onnx", "type": "ModelProto"}
    res = onnx_manager._model_info(model=onnx_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("onnx", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(onnx_manager, ml_library, should_match):
    assert (
        onnx_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(onnx_manager, onnx_model):
    exp = {}
    res = onnx_manager._model_data(model=onnx_model)
    assert exp == res


def test_required_kwargs(onnx_manager):
    assert onnx_manager._required_kwargs() == ["model"]


def test_matches_with(onnx_manager, onnx_model):
    assert onnx_manager.matches_with(model=onnx_model)
    assert not onnx_manager.matches_with(model="a-string-value")
    assert not onnx_manager.matches_with(wrong_kwarg_keyword=onnx_model)


def test_get_functions(onnx_manager, onnx_model):
    assert len(onnx_manager._get_functions(model=onnx_model)) == 1


def test_get_params(onnx_manager, onnx_model):
    res = onnx_manager._get_params(model=onnx_model)
    assert {} == res


def test_save_model(tmp_path, onnx_model, onnx_inference, classification_data):
    # Get the current predictions
    model_pred = get_predictions(onnx_inference, classification_data)

    # Save the model to file
    exp = os.path.join(tmp_path, f"model.onnx")
    res = onnx.save_model(tmp_path, model=onnx_model)
    assert os.path.exists(exp)
    assert res == exp

    # Load the saved model and get its predictions
    sess = rt.InferenceSession(res)
    loaded_pred = get_predictions(sess, classification_data)
    assert np.allclose(model_pred, loaded_pred)


def test_load_model(
    tmp_path, onnx_manager, onnx_model, onnx_inference, classification_data
):
    # Get the current predictions
    model_pred = get_predictions(onnx_inference, classification_data)

    # Save the model to a tmp directory
    model_path = onnx.save_model(tmp_path, onnx_model)
    assert model_path == os.path.join(tmp_path, onnx.MODEL_FILE)

    #  Load the model
    loaded_model = onnx_manager.load(tmp_path, {})
    loaded_pred = get_predictions(loaded_model, classification_data)

    # Expect the two to be the same
    assert type(loaded_model) == type(onnx_inference)
    assert np.allclose(model_pred, loaded_pred)
