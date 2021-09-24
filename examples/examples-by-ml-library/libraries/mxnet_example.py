import mxnet as mx
import numpy as np
from modelstore.model_store import ModelStore
from mxnet.gluon import nn
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def random_y():
    y = np.random.rand(10, 10)
    return mx.ndarray.array(y)


def _train_example_model() -> nn.HybridSequential:
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(20, activation="relu"))
    net.initialize(ctx=mx.cpu(0))
    net.hybridize()
    net(random_y())
    return net


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the mxnet model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the onnx "{DIABETES_DOMAIN}" domain model={model_id}')
    sess = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    y_pred = sess.run(None, {"X": X_test.astype(np.float32)})[0]
    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Loaded model MSE={results}.")
