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
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import pytest
from modelstore.models.model_file import ModelFileManager
from tests.models.test_xgboost import xgb_model

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def model_file(tmp_dir):
    file_path = os.path.join(tmp_dir, "model.txt")
    with open(file_path, "w") as out:
        out.write("example-model-content")
    return file_path


@pytest.fixture
def model_file_manager():
    return ModelFileManager()


def test_model_info(model_file_manager):
    exp = {"library": "model_file"}
    assert model_file_manager._model_info() == exp


@pytest.mark.parametrize(
    "ml_library,should_match",
    [
        ("sklearn", False),
        ("xgboost", False),
        ("model_file", True),
    ],
)
def test_is_model_type(model_file_manager, ml_library, should_match):
    assert (
        model_file_manager._is_model_type({"library": ml_library})
        == should_match
    )


def test_required_kwargs(model_file_manager):
    assert model_file_manager._required_kwargs() == ["model"]


def test_matches_with(model_file_manager, model_file):
    assert model_file_manager.matches_with(model=model_file)
    assert not model_file_manager.matches_with(model="a-string-value")
    assert not model_file_manager.matches_with(classifier=model_file)
    assert not model_file_manager.matches_with(model=xgb_model)


def test_get_functions(model_file_manager, model_file):
    assert len(model_file_manager._get_functions(model=model_file)) == 1
    with pytest.raises(TypeError):
        model_file_manager._get_functions(model="not-a-persisted-model-file")


def test_get_params(model_file_manager, model_file):
    assert model_file_manager._get_params(model=model_file) == {}
