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

import pytest
import shap as shp
import numpy as np
from modelstore.models.common import save_joblib
from modelstore.models.multiple_models import MultipleModelsManager
from modelstore.models.shap import ShapManager, EXPLAINER_FILE
from modelstore.models.sklearn import SKLearnManager, MODEL_JOBLIB
from sklearn.ensemble import RandomForestRegressor
from tests.models.utils import classification_data

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def sklearn_tree(classification_data):
    X_train, y_train = classification_data
    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_split": 5,
        "n_jobs": 1,
    }
    clf = RandomForestRegressor(**params)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture
def shap_explainer(sklearn_tree):
    return shp.TreeExplainer(sklearn_tree)


@pytest.fixture
def multiple_model_manager():
    managers = [SKLearnManager(), ShapManager()]
    return MultipleModelsManager(managers)


def test_model_info_with_explainer(
    multiple_model_manager, sklearn_tree, shap_explainer
):
    exp = {
        "library": "multiple-models",
        "models": [
            {"library": SKLearnManager.NAME, "type": "RandomForestRegressor"},
            {"library": ShapManager.NAME, "type": "Tree"},
        ],
    }
    res = multiple_model_manager._model_info(
        model=sklearn_tree,
        explainer=shap_explainer,
    )
    assert res == exp


def test_matches_with(multiple_model_manager, sklearn_tree, shap_explainer):
    assert multiple_model_manager.matches_with(
        model=sklearn_tree, explainer=shap_explainer
    )


def test_get_functions(multiple_model_manager, sklearn_tree, shap_explainer):
    functions = multiple_model_manager._get_functions(
        model=sklearn_tree,
        explainer=shap_explainer,
    )
    # Two functions: save the model and explainer
    assert len(functions) == 2


def test_get_functions_incorrect_types(multiple_model_manager, sklearn_tree):
    with pytest.raises(TypeError):
        multiple_model_manager._get_functions(
            model=sklearn_tree,
            explainer="not-a-shap-explainer",
        )


def test_get_params(multiple_model_manager, sklearn_tree, shap_explainer):
    try:
        result = multiple_model_manager._get_params(
            model=sklearn_tree,
            explainer=shap_explainer,
        )
        assert "sklearn" in result
        assert "shap" in result
        json.dumps(result)
    except Exception as e:
        pytest.fail(f"Exception when dumping params: {str(e)}")


def test_load_model(tmp_path, multiple_model_manager, sklearn_tree, shap_explainer):
    # Save the model to a tmp directory
    model_path = save_joblib(tmp_path, sklearn_tree, MODEL_JOBLIB)
    assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

    # Save the explainer to file
    exp = os.path.join(tmp_path, EXPLAINER_FILE)
    res = save_joblib(tmp_path, shap_explainer, file_name=EXPLAINER_FILE)
    assert exp == res

    #  Load the model
    loaded_models = multiple_model_manager.load(
        tmp_path,
        {
            "model": {
                "model_type": {
                    "models": [
                        {"library": ShapManager.NAME},
                        {"library": SKLearnManager.NAME},
                    ]
                }
            }
        },
    )

    # Expect the two models to have been loaded
    assert len(loaded_models) == 2

    sklearn_model = loaded_models[SKLearnManager.NAME]
    assert type(sklearn_model) == type(sklearn_tree)
    assert sklearn_model.get_params() == sklearn_tree.get_params()

    explainer_model = loaded_models[ShapManager.NAME]
    assert type(explainer_model) == type(shap_explainer)
