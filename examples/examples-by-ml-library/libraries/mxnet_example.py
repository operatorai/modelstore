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

import mxnet as mx
import numpy as np
from modelstore.model_store import ModelStore
from mxnet.gluon import nn
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_regression_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> nn.HybridSequential:
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(10, activation="relu"))
        net.add(nn.Dense(1, activation="relu"))
    net.initialize(ctx=mx.cpu(0))
    net.hybridize()

    X_train, X_test, _, y_test = load_regression_dataset()
    net(mx.ndarray.array(X_train))
    # Etc.

    y_pred = net(mx.ndarray.array(X_test))
    y_pred = np.squeeze(y_pred.asnumpy())
    print(f"📊  Model has mse={mean_squared_error(y_pred, y_test)}.")
    return net


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the mxnet model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model, epoch=0)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the mxnet "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_regression_dataset()
    y_pred = model(mx.ndarray.array(X_test))
    y_pred = np.squeeze(y_pred.asnumpy())
    print(f"📊  Model has mse={mean_squared_error(y_pred, y_test)}.")
