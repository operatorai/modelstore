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
from unittest import mock

import pytest
from google.cloud import storage
from google.cloud.storage.blob import Blob
from modelstore.storage.gcloud import GoogleCloudStorage

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
_MOCK_BUCKET_NAME = "gcloud-bucket"


@pytest.fixture
def gcloud_bucket():
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_bucket.name = _MOCK_BUCKET_NAME
    return mock_bucket


@pytest.fixture
def gcloud_client(gcloud_bucket):
    mock_client = mock.create_autospec(storage.Client)

    # Buckets
    gcloud_bucket.client = mock_client
    mock_client.get_bucket.return_value = gcloud_bucket
    mock_client.list_buckets.return_value = [gcloud_bucket]

    # Blobs
    mock_blob = mock.create_autospec(storage.Blob)
    mock_blob.download_as_string.return_value = "file-contents"
    gcloud_bucket.blob.return_value = mock_blob

    return mock_client


@pytest.fixture
def gcloud_model_store(gcloud_client):
    return GoogleCloudStorage(
        project_name="", bucket_name=_MOCK_BUCKET_NAME, client=gcloud_client
    )


def test_validate_existing_bucket(gcloud_model_store):
    assert gcloud_model_store.validate()


def test_validate_missing_bucket(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="missing-bucket", client=gcloud_client
    )
    assert not gcloud_model_store.validate()


def test_push(temp_file, remote_file_path, gcloud_model_store):
    result = gcloud_model_store._push(temp_file, remote_file_path)
    assert result == remote_file_path


def test_pull(temp_file, tmp_path, remote_file_path, gcloud_model_store):
    # Push the file to storage
    remote_destination = gcloud_model_store._push(temp_file, remote_file_path)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = gcloud_model_store._pull(remote_destination, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


def test_read_json_objects_ignores_non_json(gcloud_bucket, gcloud_model_store):
    gcloud_model_store.client.list_blobs.return_value = [
        Blob(name="test-file-source-1.txt", bucket=gcloud_bucket),
        Blob(name="test-file-source-2.txt", bucket=gcloud_bucket),
    ]
    # Argument (remote prefix) is ignored here because of mock above
    items = gcloud_model_store._read_json_objects("")
    assert len(items) == 0


def test_read_json_object_fails_gracefully(gcloud_model_store):
    # Read a file that does not contain any JSON
    # Argument (remote prefix) is ignored here because of mock above
    item = gcloud_model_store._read_json_object("")
    assert item is None


def test_storage_location(gcloud_model_store):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "google:cloud-storage",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert gcloud_model_store._storage_location(prefix) == exp


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
def test_get_location(gcloud_model_store, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            gcloud_model_store._get_storage_location(meta_data)
    else:
        assert gcloud_model_store._get_storage_location(meta_data) == result
