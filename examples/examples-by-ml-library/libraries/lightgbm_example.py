import lightgbm as lgb
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> lgb.Booster:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train the model
    print("🤖  Training a light GBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, y_test)
    num_round = 5
    param = {"num_leaves": 31, "objective": "binary"}
    model = lgb.train(
        param, train_data, num_round, valid_sets=[validation_data]
    )

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a Light GBM model
    model = _train_example_model()

    # Upload the model to the model store
    print(
        f'⤴️  Uploading the light gbm model to the "{DIABETES_DOMAIN}" domain.'
    )
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading the light gbm "{DIABETES_DOMAIN}" domain model={model_id}'
    )
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
