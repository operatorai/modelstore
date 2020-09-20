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
import tarfile
from pathlib import Path

import pytest

from modelstore.models.modelmanager import ModelManager

# pylint: disable=protected-access
# pylint: disable=missing-class-docstring


class MockModelManager(ModelManager):
    @classmethod
    def required_dependencies(cls) -> list:
        return []

    def _get_functions(self, **kwargs) -> list:
        return [
            mock_save_model,
            mock_save_config,
        ]

    def _required_kwargs(self) -> list:
        return ["model", "config"]


def mock_save_model(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "model.joblib")
    Path(path).touch()
    return path


def mock_save_config(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "config.json")
    Path(path).touch()
    return path


def test_collect_files(tmp_path):
    mngr = MockModelManager()
    exp = [
        os.path.join(tmp_path, "python-info.json"),
        os.path.join(tmp_path, "model.joblib"),
        os.path.join(tmp_path, "config.json"),
    ]
    res = mngr._collect_files(tmp_path)
    assert res == exp


def test_validate_kwargs():
    mngr = MockModelManager()
    with pytest.raises(TypeError):
        # Missing model= kwarg
        mngr._validate_kwargs(config="config")
    mngr._validate_kwargs(model="model", config="config")


def test_create_archive():
    target = os.path.join(os.getcwd(), "artifacts.tar.gz")
    if os.path.exists(target):
        os.remove(target)

    mngr = MockModelManager()
    mngr.create_archive(model="model", config="config")
    assert os.path.exists(target)

    exp = [
        "python-info.json",
        "model.joblib",
        "config.json",
    ]
    with tarfile.open(target) as tar:
        files = [f.name for f in tar.getmembers()]
        assert files == exp
