#    Copyright 2021 Neal Lathia
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
import os
import warnings

import mxnet as mx
import numpy as np
import pytest
from modelstore.models import mxnet
from mxnet.gluon import nn


# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
def random_x():
    y = np.random.rand(10, 10)
    return mx.ndarray.array(y)


@pytest.fixture
def mxnet_model():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(20, activation="relu"))
    net.initialize(ctx=mx.cpu(0))
    net.hybridize()
    net(random_x())
    return net


@pytest.fixture
def mxnet_manager():
    return mxnet.MxnetManager()


def test_model_info(mxnet_manager, mxnet_model):
    exp = {"library": "mxnet", "type": "HybridSequential"}
    res = mxnet_manager._model_info(model=mxnet_model)
    assert exp == res


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("mxnet", True),
        ("sklearn", False),
    ],
)
def test_is_same_library(mxnet_manager, ml_library, should_match):
    assert (
        mxnet_manager._is_same_library({"library": ml_library}) == should_match
    )


def test_model_data(mxnet_manager, mxnet_model):
    exp = {}
    res = mxnet_manager._model_data(model=mxnet_model)
    assert exp == res


def test_required_kwargs(mxnet_manager):
    assert mxnet_manager._required_kwargs() == ["model", "epoch"]


def test_matches_with(mxnet_manager, mxnet_model):
    assert mxnet_manager.matches_with(model=mxnet_model)
    assert not mxnet_manager.matches_with(model="a-string-value")
    assert not mxnet_manager.matches_with(wrong_kwarg_keyword=mxnet_model)


def test_get_functions(mxnet_manager, mxnet_model):
    assert len(mxnet_manager._get_functions(model=mxnet_model, epoch=3)) == 1


def test_get_params(mxnet_manager, mxnet_model):
    res = mxnet_manager._get_params(model=mxnet_model, epoch=3)
    assert {"epoch": 3} == res


def test_save_model(tmp_path, mxnet_model):
    x = random_x()
    y_pred = mxnet_model(x).asnumpy()

    # Save the model to file
    results = mxnet.save_model(tmp_path, model=mxnet_model, epoch=0)
    assert len(results) == 2
    assert all(os.path.exists(x) for x in results)

    # Load the saved model and get its predictions

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = nn.SymbolBlock.imports(results[0], ["data"], results[1])
    y_loaded_pred = loaded(x).asnumpy()

    assert np.allclose(y_pred, y_loaded_pred)


def test_load_model(tmp_path, mxnet_manager, mxnet_model):
    # Get the current predictions
    x = random_x()
    y_pred = mxnet_model(x).asnumpy()

    # Save the model to a tmp directory
    mxnet.save_model(tmp_path, mxnet_model, epoch=0)

    #  Load the model
    loaded_model = mxnet_manager.load(
        tmp_path,
        {
            "model": {
                "parameters": {"epoch": 0},
            }
        },
    )
    y_loaded_pred = loaded_model(x).asnumpy()

    # Expect the two to be the same
    assert isinstance(loaded_model, nn.SymbolBlock)
    assert np.allclose(y_pred, y_loaded_pred)
