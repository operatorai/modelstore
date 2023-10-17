#    Copyright 2023 Neal Lathia
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

import shap
from modelstore.model_store import ModelStore
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> Pipeline:
    X_train, X_test, y_train, y_test = load_regression_dataset()

    # Train a model using an sklearn pipeline
    params = {
        "n_estimators": 250,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")

    explainer = shap.TreeExplainer(model)

    # Example only
    shap_values = explainer.shap_values(X_test)[0]
    print(f"🔍  Shap values={shap_values[:10]}.")

    return model, explainer


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model and an explainer
    model, explainer = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the sklearn model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model, explainer=explainer)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading sklearn/shap modelsL domain="{model_domain}" model={model_id}')
    models = modelstore.load(model_domain, model_id)
    clf = models["sklearn"]
    shp = models["shap"]

    # Run some example predictions
    _, X_test, _, y_test = load_regression_dataset()
    results = mean_squared_error(y_test, clf.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")

    # Run some example explanations
    shap_values = shp.shap_values(X_test)[0]
    print(f"🔍  Shap values={shap_values[:10]}.")
