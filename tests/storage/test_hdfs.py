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
import platform
import pytest

from modelstore.metadata import metadata
from modelstore.storage.hdfs import HdfsStorage

# pylint: disable=unused-import
from tests.storage.test_utils import (
    remote_file_path,
    remote_path,
    push_temp_file,
    push_temp_files,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


def is_not_mac() -> bool:
    return platform.system() != 'Darwin'


@pytest.fixture
def storage(tmp_path):
    return HdfsStorage(root_prefix=str(tmp_path), create_directory=True)


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    for key in HdfsStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.setenv(key, "a-value")
    try:
        _ = HdfsStorage()
    except KeyError:
        pytest.fail("Failed to initialise storage from env variables")


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
def test_validate(storage):
    assert storage.validate()


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
def test_push_and_pull(storage, tmp_path):
    # pylint: disable=import-outside-toplevel
    import pydoop.hdfs as hdfs
    prefix = push_temp_file(storage)
    files = hdfs.ls(prefix)
    assert len(files) == 1
    result = storage._pull(
        prefix,
        str(tmp_path),
    )
    assert os.path.exists(result)
    hdfs.rm(files[0])


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
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
def test_remove(storage, file_exists, should_call_delete):
    if file_exists:
        # Push a file to storage
        _ = push_temp_file(storage)
    prefix = remote_file_path()
    assert storage._remove(prefix) == should_call_delete
    assert not os.path.exists(prefix)


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
def test_read_json_objects_ignores_non_json(storage):
    # pylint: disable=import-outside-toplevel
    import pydoop.hdfs as hdfs
    # Create files with different suffixes
    prefix = remote_path()
    _ = [hdfs.rm(f) for f in hdfs.ls(prefix)]
    push_temp_files(storage, prefix)

    # Read the json files at the prefix
    items = storage._read_json_objects(prefix)
    assert len(items) == 1
    _ = [hdfs.rm(f) for f in hdfs.ls(prefix)]


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
def test_storage_location(storage):
    prefix = remote_path()
    # Asserts that the location meta data is correctly formatted
    expected = metadata.Storage.from_path(
        storage_type="hdfs",
        root=storage.root_prefix,
        path=prefix,
    )
    assert storage._storage_location(prefix) == expected


@pytest.mark.skipif(is_not_mac(), reason="no hadoop in ci")
@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            metadata.Storage(
                type="hdfs",
                root="",
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
def test_get_location(storage, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
