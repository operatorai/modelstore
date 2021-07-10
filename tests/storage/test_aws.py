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


@pytest.fixture
def aws_model_store():
    return AWSStorage(bucket_name=_MOCK_BUCKET_NAME)


def test_validate_existing_bucket(aws_model_store):
    assert aws_model_store.validate()


def test_validate_missing_bucket():
    aws_model_store = AWSStorage(bucket_name="missing-bucket")
    assert not aws_model_store.validate()


def test_push(temp_file, remote_file_path, moto_boto, aws_model_store):
    def get_file_contents(prefix):
        return (
            moto_boto.Object(_MOCK_BUCKET_NAME, prefix)
            .get()["Body"]
            .read()
            .decode("utf-8")
        )

    result = aws_model_store._push(temp_file, remote_file_path)
    assert result == remote_file_path
    assert get_file_contents(result) == TEST_FILE_CONTENTS


def test_pull(temp_file, tmp_path, remote_file_path, aws_model_store):
    # Push the file to storage
    remote_destination = aws_model_store._push(temp_file, remote_file_path)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = aws_model_store._pull(remote_destination, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


def test_read_json_objects_ignores_non_json(
    tmp_path, remote_path, aws_model_store
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
        aws_model_store._push(source, remote_destination)

    # Read the json files at the prefix
    items = aws_model_store._read_json_objects(remote_path)
    assert len(items) == 1


def test_read_json_object_fails_gracefully(
    temp_file, remote_file_path, aws_model_store
):
    # Push a file that doesn't contain JSON to storage
    remote_path = aws_model_store._push(temp_file, remote_file_path)

    # Read the json files at the prefix
    item = aws_model_store._read_json_object(remote_path)
    assert item is None


def test_storage_location(aws_model_store, remote_path):
    # Asserts that the location meta data is correctly formatted
    exp = {
        "type": "aws:s3",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": remote_path,
    }
    assert aws_model_store._storage_location(remote_path) == exp


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
def test_get_location(aws_model_store, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            aws_model_store._get_storage_location(meta_data)
    else:
        assert aws_model_store._get_storage_location(meta_data) == result
