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
from mock import patch

from modelstore.metadata import metadata
from modelstore.models.model_manager import ModelManager
from modelstore.storage.local import FileSystemStorage

# pylint: disable=protected-access
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


class MockCloudStorage(FileSystemStorage):
    def __init__(self, tmpdir):
        super().__init__(root_dir=str(tmpdir))
        self.called = False

    # pylint: disable=unused-argument
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

    def model_info(self, **kwargs) -> metadata.ModelType:
        return metadata.ModelType("mock", None, None)

    def model_data(self, **kwargs) -> metadata.Dataset:
        return None

    def required_dependencies(self) -> list:
        return []

    def _get_functions(self, **kwargs) -> list:
        return [
            mock_save_model,
            mock_save_config,
        ]

    def get_params(self, **kwargs) -> dict:
        return {}

    def _required_kwargs(self) -> list:
        return ["model", "config"]

    def matches_with(self, **kwargs) -> bool:
        return True

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)
        return True


def mock_save_model(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "model.joblib")
    Path(path).touch()
    return path


def mock_save_config(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "config.json")
    Path(path).touch()
    return path


@pytest.fixture
def mock_file(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "extra-file.csv")
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
            os.path.join(tmp_path, "model.joblib"),
            os.path.join(tmp_path, "config.json"),
        ]
    )
    res = sorted(mock_manager._collect_files(tmp_path))
    assert res == exp


def test_collect_extras_single_file(mock_manager, mock_file):
    res = mock_manager._collect_extras(extra_files=mock_file)
    assert isinstance(res, set)
    assert len(res) == 1

    # Deprecated approach
    res = mock_manager._collect_extras(extras=mock_file)
    assert isinstance(res, set)
    assert len(res) == 1


def test_collect_extras_removes_duplicates(mock_manager, mock_file):
    res = mock_manager._collect_extras(extra_files=[mock_file, mock_file])
    assert isinstance(res, set)
    assert len(res) == 1

    # Deprecated approach
    res = mock_manager._collect_extras(extras=[mock_file, mock_file])
    assert isinstance(res, set)
    assert len(res) == 1


def test_collect_extras_removes_missing_files(mock_manager, mock_file):
    res = mock_manager._collect_extras(extra_files=[mock_file, "missing-file.txt"])
    assert isinstance(res, set)
    assert len(res) == 1

    # Deprecated approach
    res = mock_manager._collect_extras(extras=[mock_file, "missing-file.txt"])
    assert isinstance(res, set)
    assert len(res) == 1


def test_collect_extras_removes_missing_directories(mock_manager, mock_file, tmpdir):
    res = mock_manager._collect_extras(extra_files=[mock_file, tmpdir])
    assert isinstance(res, set)
    assert len(res) == 1

    # Deprecated approach
    res = mock_manager._collect_extras(extras=[mock_file, tmpdir])
    assert isinstance(res, set)
    assert len(res) == 1


def test_validate_kwargs(mock_manager):
    with pytest.raises(TypeError):
        # Missing model= kwarg
        mock_manager._validate_kwargs(config="config")
    mock_manager._validate_kwargs(model="model", config="config")


def test_upload(mock_manager):
    mock_manager.upload(
        domain="test-domain",
        model_id="test-model",
        model="model",
        config="config",
    )
    assert mock_manager.storage.called


@patch("modelstore.models.model_manager.get_python_version")
def test_load_from_different_python(mock_python_version, mock_manager):
    mock_python_version.return_value = f"python:2.7.13"
    meta_data = metadata.Summary.generate(
        code_meta_data=metadata.Code.generate(deps_list=[]),
        model_meta_data=None,
        storage_meta_data=None,
    )
    with pytest.warns(RuntimeWarning):
        mock_manager.load("/path/to/file", meta_data)


def test_create_archive(mock_manager, mock_file):
    target = os.path.join(os.getcwd(), "artifacts.tar.gz")
    if os.path.exists(target):
        os.remove(target)

    mock_manager._create_archive(
        model="model",
        config="config",
        extra_files=mock_file,
    )
    exp = sorted(
        [
            "model-info.json",
            "model.joblib",
            "config.json",
            os.path.join("extras", "extra-file.csv"),
        ]
    )
    with tarfile.open(target) as tar:
        files = sorted([f.name for f in tar.getmembers()])
    assert len(files) == len(exp)
    assert files == exp
    os.remove(target)
