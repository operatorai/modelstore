#    Copyright 2022 Neal Lathia
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
from pathlib import PosixPath
import shutil
import os

from mock import patch
import pytest

from modelstore.model_store import ModelStore
from modelstore.models.managers import _LIBRARIES
from modelstore.storage.local import FileSystemStorage
from modelstore.utils.exceptions import DomainNotFoundException, ModelNotFoundException

# pylint: disable=unused-import
from tests.test_utils import (
    libraries_without_sklearn,
    iter_only_sklearn,
    validate_library_attributes,
    model_file,
)

# pylint: disable=missing-function-docstring


@pytest.mark.parametrize(
    "should_create",
    [
        True,
        False,
    ],
)
def test_from_file_system_existing_root(tmp_path: PosixPath, should_create: bool):
    store = ModelStore.from_file_system(
        root_directory=str(tmp_path), create_directory=should_create
    )
    assert isinstance(store.storage, FileSystemStorage)
    validate_library_attributes(store, allowed=_LIBRARIES, not_allowed=[])


@pytest.mark.parametrize(
    "should_create",
    [
        True,
        False,
    ],
)
def test_from_file_system_missing_root(tmp_path: PosixPath, should_create: bool):
    root_directory = os.path.join(tmp_path, "unit-test")
    assert not os.path.exists(root_directory)
    if should_create:
        store = ModelStore.from_file_system(
            root_directory=root_directory, create_directory=should_create
        )
        assert os.path.exists(root_directory)
        assert os.path.isdir(root_directory)
        assert isinstance(store.storage, FileSystemStorage)
        validate_library_attributes(store, allowed=_LIBRARIES, not_allowed=[])
        # Clean up
        shutil.rmtree(root_directory)
    else:
        with pytest.raises(Exception):
            _ = ModelStore.from_file_system(
                root_directory=root_directory, create_directory=should_create
            )


@patch("modelstore.model_store.iter_libraries", side_effect=iter_only_sklearn)
def test_from_file_system_only_sklearn(_mock_iter_libraries, tmp_path):
    store = ModelStore.from_file_system(root_directory=str(tmp_path))
    assert isinstance(store.storage, FileSystemStorage)
    validate_library_attributes(
        store, allowed=["sklearn"], not_allowed=libraries_without_sklearn()
    )


def test_model_not_found(tmp_path: PosixPath, model_file: str):
    store = ModelStore.from_file_system(root_directory=str(tmp_path))
    with pytest.raises(DomainNotFoundException):
        store.get_model_info("missing-domain", "missing-model")

    store.upload(domain="existing-domain", model_id="test-model-id-1", model=model_file)
    with pytest.raises(ModelNotFoundException):
        store.get_model_info("existing-domain", "missing-model")
