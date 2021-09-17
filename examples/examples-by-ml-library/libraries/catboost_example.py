import catboost as ctb
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> ctb.CatBoostRegressor:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train the model
    print("🤖  Training a CatBoostRegressor")
    model = ctb.CatBoostRegressor(allow_writing_files=False)
    model.fit(X_train, y_train)

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Fit model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a Catboost model
    model = _train_example_model()

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the catboost model to the "{DIABETES_DOMAIN}" domain.'
    )
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the catboost "{DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
