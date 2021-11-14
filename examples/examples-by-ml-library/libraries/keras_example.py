from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> keras.Model:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train a model
    print(f"🤖  Training a keras model...")
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10)

    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a word2vec model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the keras model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the keras "{DIABETES_DOMAIN}" domain model={model_id}')
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some test predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
