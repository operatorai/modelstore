#    Copyright 2024 Neal Lathia
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
import numpy as np
import pandas as pd

from libraries.util.datasets import load_causal_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN
from causalml.inference.meta import XGBTRegressor, BaseSRegressor
from causalml.metrics import qini_score
from lightgbm.sklearn import LGBMRegressor

from modelstore.model_store import ModelStore


def _train_example_model() -> XGBTRegressor:
    X_train, X_test, y_train, y_test, treatment_vector_train, treatment_vector_test = load_causal_regression_dataset()

    params = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.01,
        "n_jobs": 1,
        "verbose": -1
    }

    # Train causal regressor
    lgbm = LGBMRegressor(**params)
    model = BaseSRegressor(learner=lgbm)
    model.fit(X_train, treatment_vector_train, y_train)

    X_test = pd.DataFrame(X_test)
    X_test["causal_scores"] = model.predict(X_test)
    X_test["outcomes"] = y_test
    X_test["treatment"] = treatment_vector_test

    result = qini_score(
        X_test[["causal_scores", "outcomes", "treatment"]],
        outcome_col="outcomes",
        treatment_col="treatment",
    )
    print(f"🔍  Trained model Qini score={result}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a causalml regressor
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the causalml model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the causalml "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Run some example predictions
    _, X_test, _, y_test, _, treatment_vector_test = load_causal_regression_dataset()

    X_test = pd.DataFrame(X_test)
    X_test["causal_scores"] = model.predict(X_test)
    X_test["outcomes"] = y_test
    X_test["treatment"] = treatment_vector_test

    result = qini_score(
        X_test[["causal_scores", "outcomes", "treatment"]],
        outcome_col="outcomes",
        treatment_col="treatment",
    )
    print(f"🔍  Loaded model Qini score={result}.")
