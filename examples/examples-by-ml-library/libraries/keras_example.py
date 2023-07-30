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

from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from modelstore.model_store import ModelStore


def _train_example_model() -> keras.Model:
    # Load the data
    X_train, X_test, y_train, y_test = load_regression_dataset()

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


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the keras "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Run some test predictions
    _, X_test, _, y_test = load_regression_dataset()
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
