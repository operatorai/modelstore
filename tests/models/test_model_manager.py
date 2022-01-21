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
from typing import Any, Optional, Union

import pytest
from modelstore.models.model_manager import ModelManager
from modelstore.storage.local import FileSystemStorage

# pylint: disable=protected-access
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name


class MockCloudStorage(FileSystemStorage):
    def __init__(self, tmpdir):
        super().__init__(root_dir=str(tmpdir))
        self.called = False

    def upload(
        self,
        domain: str,
        local_path: str,
        extras: Optional[Union[str, list]] = None,
    ):
        self.called = True


class MockModelManager(ModelManager):
    def __init__(self, tmpdir):
        super().__init__("mock", storage=MockCloudStorage(tmpdir))

    @classmethod
    def name(cls) -> str:
        return "mock"

    def _model_info(self, **kwargs) -> dict:
        return {}

    def _model_data(self, **kwargs) -> dict:
        return {}

    @classmethod
    def required_dependencies(cls) -> list:
        return []

    def _get_functions(self, **kwargs) -> list:
        return [
            mock_save_model,
            mock_save_config,
        ]

    def _get_params(self, **kwargs) -> dict:
        return {}

    def _required_kwargs(self) -> list:
        return ["model", "config"]

    def matches_with(self, **kwargs) -> bool:
        return True

    def load(self, model_path: str, meta_data: dict) -> Any:
        raise NotImplementedError()


def mock_save_model(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "model.joblib")
    Path(path).touch()
    return path


def mock_save_config(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "config.json")
    Path(path).touch()
    return path


@pytest.fixture
def mock_manager(tmpdir):
    return MockModelManager(tmpdir)


def test_collect_files(mock_manager):
    tmp_path = mock_manager.storage.root_prefix
    exp = sorted(
        [
            os.path.join(tmp_path, "model-info.json"),
            os.path.join(tmp_path, "python-info.json"),
            os.path.join(tmp_path, "model.joblib"),
            os.path.join(tmp_path, "config.json"),
        ]
    )
    res = sorted(mock_manager._collect_files(tmp_path))
    assert res == exp


def test_validate_kwargs(mock_manager):
    with pytest.raises(TypeError):
        # Missing model= kwarg
        mock_manager._validate_kwargs(config="config")
    mock_manager._validate_kwargs(model="model", config="config")


def test_upload(mock_manager):
    mock_manager.upload(domain="model", model="model", config="config")
    assert mock_manager.storage.called


def test_create_archive(mock_manager):
    target = os.path.join(os.getcwd(), "artifacts.tar.gz")
    if os.path.exists(target):
        os.remove(target)

    mock_manager._create_archive(model="model", config="config")
    exp = sorted(
        [
            "model-info.json",
            "python-info.json",
            "model.joblib",
            "config.json",
        ]
    )
    with tarfile.open(target) as tar:
        files = sorted([f.name for f in tar.getmembers()])
    assert len(files) == len(exp)
    assert files == exp
    os.remove(target)
