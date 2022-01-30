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

import joblib
import numpy as np
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
    assert shap_manager._is_same_library({"library": ml_library}) == should_match


def test_model_data(shap_manager, shap_explainer):
    exp = {}
    res = shap_manager._model_data(explainer=shap_explainer)
    assert exp == res


def test_required_kwargs(shap_manager):
    assert shap_manager._required_kwargs() == ["explainer"]


def test_matches_with(shap_manager, shap_explainer):
    assert shap_manager.matches_with(explainer=shap_explainer)
    assert shap_manager.matches_with(explainer=shap_explainer, model="a-model")
    assert not shap_manager.matches_with(explainer="a-string-value")
    assert not shap_manager.matches_with(wrong_kwarg_keyword=shap_explainer)


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
    res = shap.save_joblib(tmp_path, shap_explainer, file_name=shap.EXPLAINER_FILE)
    assert os.path.exists(exp)
    assert res == exp

    # Load the saved explainer and get its predictions
    with open(res, "rb") as f:
        loaded_expl = joblib.load(f)
    loaded_shap_values = loaded_expl.shap_values(X_train)[0]
    assert np.allclose(shap_values, loaded_shap_values)


def test_load_model(tmp_path, shap_manager, shap_explainer, classification_data):
    # Get the shap values
    X_train, _ = classification_data
    shap_values = shap_explainer.shap_values(X_train)[0]

    # Save the explainer to file
    exp = os.path.join(tmp_path, shap.EXPLAINER_FILE)
    res = shap.save_joblib(tmp_path, shap_explainer, file_name=shap.EXPLAINER_FILE)
    assert exp == res

    #  Load the model
    loaded_expl = shap_manager.load(tmp_path, {})
    loaded_shap_values = loaded_expl.shap_values(X_train)[0]

    # Expect the two to be the same
    assert type(shap_explainer) == type(loaded_expl)
    assert np.allclose(shap_values, loaded_shap_values)
