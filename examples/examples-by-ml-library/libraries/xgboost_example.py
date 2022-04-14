from typing import Tuple

import xgboost as xgb
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> xgb.XGBRegressor:
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

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> Tuple[str, str]:
    # Train a model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the xgboost model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return DIABETES_DOMAIN, meta_data["model"]["model_id"]


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the xgboost "{DIABETES_DOMAIN}" domain model={model_id}')
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_regression_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
