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

import pytest
import shap as shp
from modelstore.models.common import save_joblib
from modelstore.models.shap import ShapManager
from modelstore.models.sklearn import MODEL_JOBLIB, SKLearnManager
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
def sklearn_manager():
    return SKLearnManager()


@pytest.fixture
def shap_manager():
    return ShapManager()


def test_model_info_with_explainer(
    sklearn_manager, sklearn_tree, shap_explainer, shap_manager
):
    exp = {
        "library": "sklearn",
        "type": "RandomForestRegressor",
        "explainer": {
            "library": "shap",
            "type": "Tree",
        },
    }
    res = sklearn_manager._model_info(
        model=sklearn_tree,
        explainer=shap_explainer,
        explainer_manager=shap_manager,
    )
    assert res == exp


def test_matches_with(sklearn_manager, sklearn_tree, shap_explainer):
    assert sklearn_manager.matches_with(
        model=sklearn_tree, explainer=shap_explainer
    )


def test_get_functions(
    sklearn_manager, sklearn_tree, shap_explainer, shap_manager
):
    functions = sklearn_manager._get_functions(
        model=sklearn_tree,
        explainer=shap_explainer,
        explainer_manager=shap_manager,
    )
    # Two functions: save the model and explainer
    assert len(functions) == 2
    with pytest.raises(TypeError):
        sklearn_manager._get_functions(
            model=sklearn_tree,
            explainer="not-a-shap-explainer",
            explainer_manager=shap_manager,
        )


def test_get_params(
    sklearn_manager, sklearn_tree, shap_explainer, shap_manager
):
    try:
        result = sklearn_manager._get_params(
            model=sklearn_tree,
            explainer=shap_explainer,
            explainer_manager=shap_manager,
        )
        json.dumps(result)
    except Exception as e:
        pytest.fail(f"Exception when dumping params: {str(e)}")


# def test_load_model(tmp_path, sklearn_manager, sklearn_tree):
#     # Save the model to a tmp directory
#     model_path = save_joblib(tmp_path, sklearn_tree, MODEL_JOBLIB)
#     assert model_path == os.path.join(tmp_path, MODEL_JOBLIB)

#     #  Load the model
#     loaded_model = sklearn_manager.load(tmp_path, {})

#     # Expect the two to be the same
#     assert type(loaded_model) == type(sklearn_tree)
#     assert loaded_model.get_params() == sklearn_tree.get_params()
