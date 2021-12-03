import numpy as np
from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error
from skorch.regressor import NeuralNetRegressor
from torch import nn

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


class ExampleModule(nn.Module):
    def __init__(self, num_units=1):
        super(ExampleModule, self).__init__()
        self.linear = nn.Linear(10, num_units)

    def forward(self, X, **kwargs):
        return self.linear(X)


def _train_example_model() -> NeuralNetRegressor:
    # Load the data
    X_train, X_test, y_train, y_test = load_diabetes_dataset(as_numpy=True)

    # Train a model
    net = NeuralNetRegressor(
        ExampleModule,
        max_epochs=1,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    net.fit(X_train, y_train)

    results = mean_squared_error(y_test, net.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")
    return net


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a skorch model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the skorch model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the skorch "{DIABETES_DOMAIN}" domain model={model_id}')
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset(as_numpy=True)
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")
