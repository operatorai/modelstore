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
from unittest.mock import ANY
import mock
import pytest

from urllib3.response import HTTPResponse
from minio import Minio
from minio.datatypes import Object

from modelstore.metadata import metadata
from modelstore.storage.minio import MinIOStorage

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_NAME,
    file_contains_expected_contents,
    remote_file_path,
    remote_path,
    push_temp_file,
    push_temp_files,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring
_MOCK_BUCKET_NAME = "minio-bucket"


@pytest.fixture
def mock_minio():
    client = mock.create_autospec(Minio)
    client.bucket_exists.return_value = True
    return client


@pytest.fixture
def storage(mock_minio):
    return MinIOStorage(bucket_name=_MOCK_BUCKET_NAME, client=mock_minio)


def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    for key in MinIOStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.setenv(key, "a-value")
    try:
        _ = MinIOStorage()
    except KeyError:
        pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in MinIOStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = MinIOStorage()


def test_validate(storage):
    assert storage.validate()


def test_push(storage):
    push_temp_file(storage)
    storage.client.put_object.assert_called_with(
        _MOCK_BUCKET_NAME, remote_file_path(), ANY, ANY
    )


def test_pull(storage):
    prefix = remote_file_path()
    storage._pull(
        prefix,
        "tmp_dir",
    )
    storage.client.fget_object.assert_called_with(
        _MOCK_BUCKET_NAME, prefix, "tmp_dir/test-file.txt"
    )


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
    prefix = remote_file_path()
    objects = []
    if file_exists:
        objects = [1]
    storage.client.list_objects.return_value = objects
    assert storage._remove(prefix) == should_call_delete
    storage.client.list_objects.assert_called_with(
        _MOCK_BUCKET_NAME, prefix, recursive=False
    )
    if file_exists:
        storage.client.remove_object.assert_called_with(_MOCK_BUCKET_NAME, prefix)
    else:
        storage.client.remove_object.assert_not_called()


def test_read_json_objects_ignores_non_json(storage):
    storage.client.list_objects.return_value = [
        Object(bucket_name=_MOCK_BUCKET_NAME, object_name="text-file.txt"),
    ]
    items = storage._read_json_objects("")
    assert len(items) == 0


def test_read_json_object_fails_gracefully(storage):
    obj = mock.create_autospec(HTTPResponse)
    obj.readlines.return_value = "not json"
    storage.client.get_object.return_value = obj
    item = storage._read_json_object(remote_path())
    assert item is None


def test_storage_location(storage):
    prefix = remote_path()
    # Asserts that the location meta data is correctly formatted
    expected = metadata.Storage.from_bucket(
        storage_type="minio:s3.amazonaws.com",
        bucket=_MOCK_BUCKET_NAME,
        prefix=prefix,
    )
    assert storage._storage_location(prefix) == expected


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            metadata.Storage(
                type="minio:s3.amazonaws.com",
                path=None,
                bucket=_MOCK_BUCKET_NAME,
                container=None,
                prefix="/path/to/file",
            ),
            False,
            "/path/to/file",
        ),
        (
            metadata.Storage(
                type="minio:s3.amazonaws.com",
                path=None,
                bucket="a-different-bucket-name",
                container=None,
                prefix="/path/to/file",
            ),
            True,
            None,
        ),
        (
            metadata.Storage(
                type="minio:http://0.0.0.0",
                path=None,
                bucket=_MOCK_BUCKET_NAME,
                container=None,
                prefix="/path/to/file",
            ),
            True,
            None,
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
