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
from pathlib import Path

import numpy as np
import pytest

# pylint: disable=unused-import
from fastai.learner import load_learner
from fastai.callback.schedule import fit_one_cycle
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner, tabular_learner

from modelstore.metadata import metadata
from modelstore.models.fastai import (
    LEARNER_FILE,
    FastAIManager,
    _export_model,
    _save_model,
)

from tests.models.utils import (
    classification_data,
    classification_df,
    classification_row,
    is_macos,
)

# Not using the * import because it triggers fastcore tests (missing fixture errors)
# from fastai.tabular.all import *

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def fai_dl(classification_df, tmp_path):
    return TabularDataLoaders.from_df(classification_df, path=tmp_path, y_names=["y"])


@pytest.fixture
def fai_learner(fai_dl) -> TabularLearner:
    learner = tabular_learner(fai_dl)
    # The optimizer is not initialized until learn is called
    learner.fit_one_cycle(n_epoch=1)
    return learner


@pytest.fixture
def fai_manager():
    return FastAIManager()


def assert_models_equal(
    model_a: TabularLearner, model_b: TabularLearner, classification_row
):
    assert type(model_a) == type(model_b)
    _, _, saved_probs = model_a.predict(classification_row)
    _, _, loaded_probs = model_b.predict(classification_row)
    np.testing.assert_allclose(saved_probs.numpy(), loaded_probs.numpy())


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_model_info(fai_manager):
    expected = metadata.ModelType("fastai", None, None)
    res = fai_manager.model_info()
    assert expected == res


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_model_data(fai_manager, fai_learner):
    res = fai_manager.model_data(learner=fai_learner)
    assert res is None


def test_required_kwargs(fai_manager):
    assert fai_manager._required_kwargs() == ["learner"]


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_matches_with(fai_manager, fai_learner):
    assert fai_manager.matches_with(learner=fai_learner)
    assert not fai_manager.matches_with(learner="a-string-value")
    assert not fai_manager.matches_with(model=fai_learner)


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_get_functions(fai_manager, fai_learner):
    assert len(fai_manager._get_functions(learner=fai_learner)) == 2


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_get_params(fai_manager, fai_learner):
    exp = {}
    res = fai_manager.get_params(learner=fai_learner)
    assert exp == res


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_save_model(tmp_path, fai_learner, fai_dl, classification_row):
    exp = os.path.join(tmp_path, "models", "learner.pth")
    model_path = _save_model(tmp_path, fai_learner)

    assert exp == model_path
    assert os.path.exists(model_path)

    learner = tabular_learner(fai_dl, path=tmp_path)
    learner.load("learner")


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_export_model(tmp_path, fai_learner, classification_row):
    exp = os.path.join(tmp_path, "learner.pkl")
    model_path = _export_model(tmp_path, fai_learner)

    assert exp == model_path
    assert os.path.exists(model_path)

    loaded_learner = load_learner(model_path)
    assert_models_equal(fai_learner, loaded_learner, classification_row)


@pytest.mark.skipif(is_macos(), reason="fastai tries to force MPS hardware")
def test_load_model(tmp_path, fai_manager, fai_learner, classification_row):
    # Save the model to a tmp directory
    fai_learner.path = Path(tmp_path)
    fai_learner.export(LEARNER_FILE)

    #  Load the model
    loaded_learner = fai_manager.load(
        tmp_path,
        metadata.Summary(
            model=None,
            code=metadata.Code(
                runtime=None,
                user=None,
                created=None,
                dependencies={"fastai": "2.2.7"},
                git=None,
            ),
            storage=None,
            modelstore=None,
        ),
    )

    # Expect the two to be the same
    assert_models_equal(fai_learner, loaded_learner, classification_row)
