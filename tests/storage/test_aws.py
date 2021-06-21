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
from modelstore.storage.aws import AWSStorage
from modelstore.storage.util.paths import get_archive_path
from moto import mock_s3

# pylint: disable=redefined-outer-name


def get_file_contents(conn, prefix):
    return (
        conn.Object("existing-bucket", prefix)
        .get()["Body"]
        .read()
        .decode("utf-8")
    )


@pytest.fixture(autouse=True)
def moto_boto():
    with mock_s3():
        conn = boto3.resource("s3")
        conn.create_bucket(Bucket="existing-bucket")
        yield conn


def test_validate():
    aws_model_store = AWSStorage(bucket_name="existing-bucket")
    assert aws_model_store.validate()
    aws_model_store = AWSStorage(bucket_name="missing-bucket")
    assert not aws_model_store.validate()


def test_upload(tmp_path, moto_boto):
    # Create a test file
    source = os.path.join(tmp_path, "test-file.txt")
    with open(source, "w") as out:
        out.write("file-contents")

    # Upload it to the model store
    aws_model_store = AWSStorage(bucket_name="existing-bucket")
    model_path = get_archive_path("test-domain", source)
    rsp = aws_model_store.upload("test-domain", "test-model-id", source)

    # Assert meta-data is correct
    assert rsp["type"] == "aws:s3"
    assert rsp["bucket"] == "existing-bucket"
    assert rsp["prefix"] == model_path

    # Assert that the uploaded file was created
    assert get_file_contents(moto_boto, model_path) == "file-contents"
