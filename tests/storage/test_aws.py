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

import boto3
import pytest
from moto import mock_s3

from modelstore.metadata import metadata
from modelstore.storage.aws import AWSStorage

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

_MOCK_BUCKET_NAME = "existing-bucket"


@pytest.fixture(autouse=True)
def moto_boto():
    with mock_s3():
        conn = boto3.resource("s3")
        conn.create_bucket(Bucket=_MOCK_BUCKET_NAME)
        yield conn


def get_file_contents(moto_boto, prefix):
    return (
        moto_boto.Object(_MOCK_BUCKET_NAME, prefix).get()["Body"].read().decode("utf-8")
    )


def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    monkeypatch.setenv("MODEL_STORE_AWS_BUCKET", _MOCK_BUCKET_NAME)
    # pylint: disable=bare-except
    try:
        _ = AWSStorage()
    except:
        pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in AWSStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = AWSStorage()


@pytest.mark.parametrize(
    "bucket_name,validate_should_pass",
    [
        (
            "missing-bucket",
            False,
        ),
        (
            _MOCK_BUCKET_NAME,
            True,
        ),
    ],
)
def test_validate(bucket_name, validate_should_pass):
    storage = AWSStorage(bucket_name=bucket_name)
    assert storage.validate() == validate_should_pass


def test_push(moto_boto):
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    result = push_temp_file(storage)

    # The correct remote prefix is returned
    assert result == remote_file_path()

    # The remote file has the right contents
    assert get_file_contents(moto_boto, result) == TEST_FILE_CONTENTS


def test_pull():
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    _ = push_temp_file(storage)

    # Pull the file back from storage
    with TemporaryDirectory() as tmp_dir:
        result = storage._pull(
            remote_file_path(),
            tmp_dir,
        )

        # The correct local path is returned
        assert result == os.path.join(tmp_dir, TEST_FILE_NAME)

        # The local file exists, with the right content
        assert os.path.exists(result)
        assert file_contains_expected_contents(result)


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
def test_remove(file_exists, should_call_delete):
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    if file_exists:
        _ = push_temp_file(storage)

    # pylint: disable=bare-except
    prefix = remote_file_path()
    assert storage._remove(prefix) == should_call_delete


def test_read_json_objects_ignores_non_json(tmp_path):
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_path()
    push_temp_files(storage, prefix)

    # Read the json files at the prefix
    items = storage._read_json_objects(prefix)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(tmp_path):
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    # Push a file that doesn't contain JSON to storage
    remote_path = push_temp_file(storage, contents="not json")

    # Read the json files at the prefix
    item = storage._read_json_object(remote_path)
    assert item is None


def test_storage_location():
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_path()
    # Asserts that the location meta data is correctly formatted
    expected = metadata.Storage.from_bucket(
        storage_type="aws:s3",
        bucket=_MOCK_BUCKET_NAME,
        prefix=prefix,
    )
    assert storage._storage_location(prefix) == expected


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            metadata.Storage(
                type=None,
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
                type=None,
                path=None,
                bucket="a-different-bucket",
                container=None,
                prefix="/path/to/file",
            ),
            True,
            None,
        ),
    ],
)
def test_get_location(meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
