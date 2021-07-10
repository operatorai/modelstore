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
def fs_model_store(tmp_path):
    return FileSystemStorage(root_path=str(tmp_path))


def test_validate(fs_model_store):
    assert fs_model_store.validate()
    assert os.path.exists(fs_model_store.root_dir)


def test_push(temp_file, remote_file_path, fs_model_store):
    result = fs_model_store._push(temp_file, remote_file_path)
    assert result == os.path.join(fs_model_store.root_dir, remote_file_path)


def test_pull(temp_file, tmp_path, remote_file_path, fs_model_store):
    # Push the file to storage
    remote_destination = fs_model_store._push(temp_file, remote_file_path)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = fs_model_store._pull(remote_destination, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


def test_read_json_objects_ignores_non_json(
    tmp_path, remote_path, fs_model_store
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
        fs_model_store._push(source, remote_destination)

    # Read the json files at the prefix
    items = fs_model_store._read_json_objects(remote_path)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(
    temp_file, remote_file_path, fs_model_store
):
    # Push a file that doesn't contain JSON to storage
    remote_path = fs_model_store._push(temp_file, remote_file_path)

    # Read the json files at the prefix
    item = fs_model_store._read_json_object(remote_path)
    assert item is None


def test_list_versions_missing_domain(fs_model_store):
    versions = fs_model_store.list_versions("domain-that-doesnt-exist")
    assert len(versions) == 0


def test_storage_location(fs_model_store):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "file_system",
        "path": prefix,
    }
    assert fs_model_store._storage_location(prefix) == exp


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
def test_get_location(fs_model_store, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            fs_model_store._get_storage_location(meta_data)
    else:
        assert fs_model_store._get_storage_location(meta_data) == result


@pytest.mark.parametrize(
    "state_name,should_create,expect_exists",
    [
        ("state-1", False, False),
        ("state-2", True, True),
    ],
)
def test_state_exists(fs_model_store, state_name, should_create, expect_exists):
    if should_create:
        fs_model_store.create_model_state(state_name)
    assert fs_model_store.state_exists(state_name) == expect_exists
