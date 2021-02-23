#    Copyright 2020 Neal Lathia
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

import numpy as np
import pandas as pd
import pytest
from fastai.callback.schedule import fit_one_cycle
from fastai.learner import load_learner
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import tabular_learner
from modelstore.models.fastai import FastAIManager, _export_model, _save_model

# Not using the * import because it triggers fastcore tests (missing fixture errors)
# from fastai.tabular.all import *

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


def get_row():
    row = {"y": np.random.randint(2)}
    row.update({f"x{i}": np.random.uniform() for i in range(10)})
    return row


@pytest.fixture
def fastai_dl(tmp_path):
    df = pd.DataFrame([get_row() for _ in range(10)])
    return TabularDataLoaders.from_df(df, path=tmp_path, y_names=["y"])


@pytest.fixture
def fastai_learner(fastai_dl):
    learner = tabular_learner(fastai_dl)
    # The optimizer is not initialized until learn is called
    learner.fit_one_cycle(n_epoch=1)
    return learner


@pytest.fixture
def fastai_manager():
    return FastAIManager()


def test_model_info(fastai_manager):
    exp = {"library": "fastai"}
    res = fastai_manager._model_info()
    assert exp == res


def test_model_data(fastai_manager, fastai_learner):
    exp = {}
    res = fastai_manager._model_data(learner=fastai_learner)
    assert exp == res


def test_required_kwargs(fastai_manager):
    assert fastai_manager._required_kwargs() == ["learner"]


def test_get_functions(fastai_manager, fastai_learner):
    assert len(fastai_manager._get_functions(learner=fastai_learner)) == 2


def test_get_params(fastai_manager, fastai_learner):
    exp = fastai_learner.opt.state_dict()
    res = fastai_manager._get_params(learner=fastai_learner)
    assert exp == res


def test_save_model(fastai_learner, fastai_dl, tmp_path):
    exp = os.path.join(tmp_path, "models", "learner.pth")
    model_path = _save_model(tmp_path, fastai_learner)

    assert exp == model_path
    assert os.path.exists(model_path)

    test_input = pd.DataFrame([get_row()]).iloc[0]
    learner = tabular_learner(fastai_dl, path=tmp_path)
    learner.load("learner")

    _, _, saved_probs = fastai_learner.predict(test_input)
    _, _, loaded_probs = learner.predict(test_input)

    np.testing.assert_allclose(saved_probs.numpy(), loaded_probs.numpy())


def test_export_model(fastai_learner, tmp_path):
    exp = os.path.join(tmp_path, "learner.pkl")
    model_path = _export_model(tmp_path, fastai_learner)

    assert exp == model_path
    assert os.path.exists(model_path)

    learner = load_learner(model_path)

    test_input = pd.DataFrame([get_row()]).iloc[0]
    _, _, saved_probs = fastai_learner.predict(test_input)
    _, _, loaded_probs = learner.predict(test_input)

    np.testing.assert_allclose(saved_probs.numpy(), loaded_probs.numpy())
