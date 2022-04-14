import tensorflow as tf
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> tf.keras.models.Sequential:
    # Load the data
    X_train, X_test, y_train, y_test = load_regression_dataset()

    # Train a model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10)

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the tensorflow model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the tensorflow "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Run some test predictions
    _, X_test, _, y_test = load_regression_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
