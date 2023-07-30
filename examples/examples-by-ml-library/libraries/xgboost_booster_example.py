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

import xgboost as xgb
from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN
from sklearn.metrics import mean_squared_error

from modelstore.model_store import ModelStore


def _train_example_model() -> xgb.Booster:
    # Load the data
    X_train, X_test, y_train, y_test = load_regression_dataset()

    # Train a model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    booster = model.get_booster()

    results = mean_squared_error(y_test, booster.predict(xgb.DMatrix(X_test)))
    print(f"🔍  Trained model MSE={results}.")
    return booster


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the xgboost booster to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the xgboost booster "{model_domain}" domain model={model_id}')
    booster = modelstore.load(model_domain, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_regression_dataset()
    results = mean_squared_error(y_test, booster.predict(xgb.DMatrix(X_test)))
    print(f"🔍  Loaded model MSE={results}.")
