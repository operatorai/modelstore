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
from tempfile import TemporaryDirectory
import os
import shutil

import pytest

from modelstore.metadata import metadata
from modelstore.storage.local import FileSystemStorage
from modelstore.utils.exceptions import DomainNotFoundException

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_NAME,
    file_contains_expected_contents,
    remote_file_path,
    remote_path,
    push_temp_file,
    push_temp_files,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


@pytest.fixture
def file_system_storage(tmp_path):
    return FileSystemStorage(root_dir=str(tmp_path))


def test_create_model_store_with_location(tmp_path):
    root_dir = os.path.join(str(tmp_path), "test-true")
    assert not os.path.exists(root_dir)
    file_system_storage = FileSystemStorage(
        root_dir=root_dir,
        create_directory=True,
    )
    assert file_system_storage.validate()
    assert os.path.exists(file_system_storage.root_prefix)
    shutil.rmtree(file_system_storage.root_prefix)


def test_create_model_store_with_missing_location(tmp_path):
    root_dir = os.path.join(str(tmp_path), "test-false")
    assert not os.path.exists(root_dir)
    file_system_storage = FileSystemStorage(
        root_dir=root_dir,
        create_directory=False,
    )
    assert not os.path.exists(root_dir)
    with pytest.raises(FileNotFoundError):
        _ = file_system_storage.validate()


def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    monkeypatch.setenv("MODEL_STORE_ROOT_PREFIX", "~")
    try:
        _ = FileSystemStorage()
        # pylint: disable=bare-except
    except:
        pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in FileSystemStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(Exception):
        _ = FileSystemStorage()


def test_validate(file_system_storage):
    assert file_system_storage.validate()
    assert os.path.exists(file_system_storage.root_prefix)


def test_push(file_system_storage):
    result = push_temp_file(file_system_storage)

    # The correct remote prefix is returned
    assert result == remote_file_path()


def test_pull(file_system_storage):
    # Push a file to storage
    _ = push_temp_file(file_system_storage)

    # Pull the file back from storage
    with TemporaryDirectory() as tmp_dir:
        result = file_system_storage._pull(
            remote_file_path(),
            tmp_dir,
        )
        exp_file = os.path.join(tmp_dir, TEST_FILE_NAME)
        assert result == exp_file
        assert os.path.exists(result)
        assert file_contains_expected_contents(exp_file)


@pytest.mark.parametrize(
    "file_exists,should_call_delete",
    [
        (
            False,
            False,
        ),
        (
            True,
            True,
        ),
    ],
)
def test_remove(file_exists, should_call_delete, file_system_storage):
    if file_exists:
        # Push a file to storage
        _ = push_temp_file(file_system_storage)
    prefix = remote_file_path()
    assert file_system_storage._remove(prefix) == should_call_delete
    assert not os.path.exists(prefix)


def test_read_json_objects_ignores_non_json(file_system_storage):
    # Create files with different suffixes
    prefix = remote_path()
    push_temp_files(file_system_storage, prefix)

    # Read the json files at the prefix
    items = file_system_storage._read_json_objects(prefix)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(file_system_storage):
    # Push a file that doesn't contain JSON to storage
    remote_path = push_temp_file(file_system_storage, contents="not json")

    # Read the json files at the prefix
    item = file_system_storage._read_json_object(remote_path)
    assert item is None


def test_list_models_missing_domain(file_system_storage):
    with pytest.raises(DomainNotFoundException):
        file_system_storage.list_models("domain-that-doesnt-exist")


def test_storage_location(file_system_storage):
    # Asserts that the location meta data is correctly formatted
    prefix = remote_file_path()
    expected = metadata.Storage.from_path(
        storage_type="file_system",
        root=file_system_storage.root_prefix,
        path=prefix,
    )
    result = file_system_storage._storage_location(prefix)
    assert result == expected


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            metadata.Storage(
                type=None,
                path="/path/to/file",
                bucket=None,
                container=None,
                prefix=None,
            ),
            False,
            "/path/to/file",
        ),
    ],
)
def test_get_location(file_system_storage, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            file_system_storage._get_storage_location(meta_data)
    else:
        assert file_system_storage._get_storage_location(meta_data) == result


@pytest.mark.parametrize(
    "state_name,should_create,expect_exists",
    [
        ("state-1", False, False),
        ("state-2", True, True),
    ],
)
def test_state_exists(file_system_storage, state_name, should_create, expect_exists):
    if should_create:
        file_system_storage.create_model_state(state_name)
    assert file_system_storage.state_exists(state_name) == expect_exists
