import mxnet as mx
import numpy as np
from modelstore.model_store import ModelStore
from mxnet.gluon import nn
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> nn.HybridSequential:
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(1, activation="relu"))
    net.initialize(ctx=mx.cpu(0))
    net.hybridize()

    X_train, _, _, _ = load_diabetes_dataset()
    net(mx.ndarray.array(X_train))
    return net


def run_example_predictions(model: nn.HybridSequential):
    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    y_pred = model(mx.ndarray.array(X_test))
    y_pred = np.squeeze(y_pred.asnumpy())
    print(f"📊  Model has mse={mean_squared_error(y_pred, y_test)}.")


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model
    model = _train_example_model()
    run_example_predictions(model)

    # Upload the model to the model store
    print(f'⤴️  Uploading the mxnet model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model, epoch=0)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the mxnet "{DIABETES_DOMAIN}" domain model={model_id}')
    model = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    run_example_predictions(model)
