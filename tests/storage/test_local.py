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

import mock
import pytest
from modelstore.storage.local import FileSystemStorage

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_NAME,
    file_contains_expected_contents,
    remote_file_path,
    remote_path,
    temp_file,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def file_system_storage(tmp_path):
    return FileSystemStorage(root_path=str(tmp_path))


def test_create_from_environment_variables():
    # Does not fail when environment variables exist
    with mock.patch.dict(
        os.environ,
        {
            "MODEL_STORE_ROOT": "~",
        },
    ):
        # pylint: disable=bare-except
        try:
            _ = FileSystemStorage()
        except:
            pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in FileSystemStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = FileSystemStorage()


def test_validate(file_system_storage):
    assert file_system_storage.validate()
    assert os.path.exists(file_system_storage.root_dir)


def test_push(temp_file, remote_file_path, file_system_storage):
    result = file_system_storage._push(temp_file, remote_file_path)
    assert result == os.path.join(
        file_system_storage.root_dir, remote_file_path
    )


def test_pull(temp_file, tmp_path, remote_file_path, file_system_storage):
    # Push the file to storage
    remote_destination = file_system_storage._push(temp_file, remote_file_path)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = file_system_storage._pull(remote_destination, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


def test_remove(temp_file, remote_file_path, file_system_storage):
    # Push the file to storage
    remote_destination = file_system_storage._push(temp_file, remote_file_path)

    # Remove the file
    file_system_storage._remove(remote_destination)

    # The file no longer exists
    assert not os.path.exists(
        os.path.join(file_system_storage.root_dir, remote_file_path)
    )


def test_read_json_objects_ignores_non_json(
    tmp_path, remote_path, file_system_storage
):
    # Create files with different suffixes
    for file_type in ["txt", "json"]:
        source = os.path.join(tmp_path, f"test-file-source.{file_type}")
        with open(source, "w") as out:
            out.write(json.dumps({"key": "value"}))

        # Push the file to storage
        remote_destination = os.path.join(
            remote_path, f"test-file-destination.{file_type}"
        )
        file_system_storage._push(source, remote_destination)

    # Read the json files at the prefix
    items = file_system_storage._read_json_objects(remote_path)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(
    temp_file, remote_file_path, file_system_storage
):
    # Push a file that doesn't contain JSON to storage
    remote_path = file_system_storage._push(temp_file, remote_file_path)

    # Read the json files at the prefix
    item = file_system_storage._read_json_object(remote_path)
    assert item is None


def test_list_versions_missing_domain(file_system_storage):
    versions = file_system_storage.list_versions("domain-that-doesnt-exist")
    assert len(versions) == 0


def test_storage_location(file_system_storage):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "file_system",
        "path": prefix,
    }
    assert file_system_storage._storage_location(prefix) == exp


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            {
                "path": "/path/to/file",
            },
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
def test_state_exists(
    file_system_storage, state_name, should_create, expect_exists
):
    if should_create:
        file_system_storage.create_model_state(state_name)
    assert file_system_storage.state_exists(state_name) == expect_exists
