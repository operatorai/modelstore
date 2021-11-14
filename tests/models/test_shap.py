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

import pytest
import shap as shp
from modelstore.models import shap
from sklearn.ensemble import RandomForestClassifier

# pylint: disable=unused-import
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def shap_explainer(classification_data):
    X_train, y_train = classification_data
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    return shp.TreeExplainer(clr)


@pytest.fixture
def shap_manager():
    return shap.ShapManager()


def test_model_info(shap_manager, shap_explainer):
    exp = {"library": "shap", "type": "Tree"}
    res = shap_manager._model_info(explainer=shap_explainer)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("shap", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(shap_manager, ml_library, should_match):
    assert (
        shap_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(shap_manager, shap_explainer):
    exp = {}
    res = shap_manager._model_data(explainer=shap_explainer)
    assert exp == res


def test_required_kwargs(shap_manager):
    assert shap_manager._required_kwargs() == ["explainer"]


def test_matches_with(shap_manager, shap_explainer):
    assert shap_manager.matches_with(explainer=shap_explainer)
    assert not shap_manager.matches_with(explainer="a-string-value")
    assert not shap_manager.matches_with(wrong_kwarg_keyword=shap_explainer)
    assert not shap_manager.matches_with(
        explainer=shap_explainer, model="a-model"
    )


def test_get_functions(shap_manager, shap_explainer):
    assert len(shap_manager._get_functions(explainer=shap_explainer)) == 1


def test_get_params(shap_manager, shap_explainer):
    res = shap_manager._get_params(explainer=shap_explainer)
    assert {} == res


def test_save_explainer(tmp_path, shap_explainer, classification_data):
    # Get the shap values
    X_train, _ = classification_data
    shap_values = shap_explainer.shap_values(X_train)[0]

    # Save the explainer to file
    exp = os.path.join(tmp_path, shap.EXPLAINER_FILE)
    res = shap.save_explainer(tmp_path, explainer=shap_explainer)
    assert os.path.exists(exp)
    assert res == exp

    # Load the saved explainer and get its predictions
    with open(res, "rb") as f:
        loaded_expl = shp.Explainer.load(f)
    import pdb

    pdb.set_trace()
    # sess = rt.InferenceSession(res)
    # loaded_pred = get_predictions(sess, classification_data)
    # assert np.allclose(model_pred, loaded_pred)


# def test_load_model(
#     tmp_path, onnx_manager, onnx_model, onnx_inference, classification_data
# ):
#     # Get the current predictions
#     model_pred = get_predictions(onnx_inference, classification_data)

#     # Save the model to a tmp directory
#     model_path = onnx.save_model(tmp_path, onnx_model)
#     assert model_path == os.path.join(tmp_path, onnx.MODEL_FILE)

#     #  Load the model
#     loaded_model = onnx_manager.load(tmp_path, {})
#     loaded_pred = get_predictions(loaded_model, classification_data)

#     # Expect the two to be the same
#     assert type(loaded_model) == type(onnx_inference)
#     assert np.allclose(model_pred, loaded_pred)
