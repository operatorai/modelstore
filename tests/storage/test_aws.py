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

import boto3
import pytest
from modelstore.storage.aws import AWSStorage
from moto import mock_s3

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_NAME,
    file_contains_expected_contents,
    remote_file_path,
    remote_path,
    temp_file,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
_MOCK_BUCKET_NAME = "existing-bucket"


@pytest.fixture(autouse=True)
def moto_boto():
    with mock_s3():
        conn = boto3.resource("s3")
        conn.create_bucket(Bucket=_MOCK_BUCKET_NAME)
        yield conn


def get_file_contents(moto_boto, prefix):
    return (
        moto_boto.Object(_MOCK_BUCKET_NAME, prefix)
        .get()["Body"]
        .read()
        .decode("utf-8")
    )


def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    monkeypatch.setenv("MODEL_STORE_AWS_BUCKET", _MOCK_BUCKET_NAME)
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


def test_push(tmp_path, moto_boto):
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_file_path()
    result = storage._push(temp_file(tmp_path), prefix)

    # The correct remote prefix is returned
    assert result == prefix

    # The remote file has the right contents
    assert get_file_contents(moto_boto, result) == TEST_FILE_CONTENTS


def test_pull(tmp_path):
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_file_path()
    remote_destination = storage._push(temp_file(tmp_path), prefix)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = storage._pull(remote_destination, tmp_path)

    # The correct local path is returned
    assert result == local_destination

    # The local file exists, with the right content
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


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
def test_remove(tmp_path, file_exists, should_call_delete):
    # Push a file to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    remote_destination = remote_file_path()
    if file_exists:
        storage._push(temp_file(tmp_path), remote_destination)

    try:
        assert storage._remove(remote_destination) == should_call_delete
    except:
        # Should fail gracefully here
        pytest.fail("Remove raised an exception")


def test_read_json_objects_ignores_non_json(tmp_path):
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_path()
    # Create files with different suffixes
    for file_type in ["txt", "json"]:
        source = os.path.join(tmp_path, f"test-file-source.{file_type}")
        with open(source, "w") as out:
            # content
            out.write(json.dumps({"key": "value"}))

        # Push the file to storage
        remote_destination = os.path.join(
            prefix, f"test-file-destination.{file_type}"
        )
        storage._push(source, remote_destination)

    # Read the json files at the prefix
    items = storage._read_json_objects(prefix)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(tmp_path):
    # Push a file that doesn't contain JSON to storage
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_file_path()
    text_file = os.path.join(tmp_path, "test.txt")
    with open(text_file, "w") as out:
        out.write("some text in a file")
    remote_path = storage._push(text_file, prefix)

    # Read the json files at the prefix
    item = storage._read_json_object(remote_path)

    # Return None if we can't decode the JSON
    assert item is None


def test_storage_location():
    storage = AWSStorage(bucket_name=_MOCK_BUCKET_NAME)
    prefix = remote_path()
    # Asserts that the location meta data is correctly formatted
    exp = {
        "type": "aws:s3",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert storage._storage_location(prefix) == exp


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            {
                "bucket": _MOCK_BUCKET_NAME,
                "prefix": "/path/to/file",
            },
            False,
            "/path/to/file",
        ),
        (
            {
                "bucket": "a-different-bucket",
                "prefix": "/path/to/file",
            },
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
