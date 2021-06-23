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

import boto3
import pytest
from modelstore.storage.aws import AWSStorage, _get_location
from moto import mock_s3

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
_MOCK_BUCKET_NAME = "existing-bucket"


def get_file_contents(conn, prefix):
    return (
        conn.Object(_MOCK_BUCKET_NAME, prefix)
        .get()["Body"]
        .read()
        .decode("utf-8")
    )


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


def test_push_and_pull(tmp_path, moto_boto, aws_model_store):
    # Create a file
    source = os.path.join(tmp_path, "test-file-source.txt")
    with open(source, "w") as out:
        out.write("expected-result")

    # Push the file to storage
    remote_destination = "prefix/to/file/test-file-destination.txt"
    result = aws_model_store._push(source, remote_destination)
    assert result == remote_destination
    assert get_file_contents(moto_boto, remote_destination) == "expected-result"

    # Pull the file back from storage
    meta_data = {
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": remote_destination,
    }
    local_destination = os.path.join(tmp_path, "test-file-destination.txt")
    result = aws_model_store._pull(meta_data, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    with open(result, "r") as lines:
        contents = lines.read()
        assert contents == "expected-result"


# @TODO missing tests!
# def test_read_json_objects(self, path: str) -> list:
# pass

# def test_read_json_object(self, path: str) -> dict:
# pass


def test_storage_location(aws_model_store):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "aws:s3",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert aws_model_store._storage_location(prefix) == exp


def test_get_location() -> str:
    # Asserts that pulling the location out of meta data
    # is correct
    exp = "/path/to/file"
    meta = {
        "type": "aws:s3",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": exp,
    }
    assert _get_location(_MOCK_BUCKET_NAME, meta) == exp
