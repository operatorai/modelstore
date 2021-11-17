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
def gcloud_storage(gcloud_client):
    return GoogleCloudStorage(
        project_name="", bucket_name=_MOCK_BUCKET_NAME, client=gcloud_client
    )


def test_create_from_environment_variables():
    # Does not fail when environment variables exist
    with mock.patch.dict(
        os.environ,
        {
            "MODEL_STORE_GCP_PROJECT": "project",
            "MODEL_STORE_GCP_BUCKET": _MOCK_BUCKET_NAME,
        },
    ):
        # pylint: disable=bare-except
        try:
            _ = GoogleCloudStorage()
        except:
            pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in GoogleCloudStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = GoogleCloudStorage()


def test_validate_existing_bucket(gcloud_storage):
    assert gcloud_storage.validate()


def test_validate_missing_bucket(gcloud_client):
    gcloud_storage = GoogleCloudStorage(
        project_name="", bucket_name="missing-bucket", client=gcloud_client
    )
    assert not gcloud_storage.validate()


def test_push(temp_file, remote_file_path, gcloud_storage):
    result = gcloud_storage._push(temp_file, remote_file_path)
    assert result == remote_file_path


def test_pull(temp_file, tmp_path, remote_file_path, gcloud_storage):
    # Push the file to storage
    remote_destination = gcloud_storage._push(temp_file, remote_file_path)

    # Pull the file back from storage
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    result = gcloud_storage._pull(remote_destination, tmp_path)
    assert result == local_destination
    assert os.path.exists(local_destination)
    assert file_contains_expected_contents(local_destination)


def test_remove(temp_file, remote_file_path, gcloud_storage):
    # Push the file to storage
    remote_destination = gcloud_storage._push(temp_file, remote_file_path)

    # Remove the file
    gcloud_storage._remove(remote_destination)

    # Trying to read the file errors
    # @TODO assert the file has been removed


def test_read_json_objects_ignores_non_json(gcloud_bucket, gcloud_storage):
    gcloud_storage.client.list_blobs.return_value = [
        Blob(name="test-file-source-1.txt", bucket=gcloud_bucket),
        Blob(name="test-file-source-2.txt", bucket=gcloud_bucket),
    ]
    # Argument (remote prefix) is ignored here because of mock above
    items = gcloud_storage._read_json_objects("")
    assert len(items) == 0


def test_read_json_object_fails_gracefully(gcloud_storage):
    # Read a file that does not contain any JSON
    # Argument (remote prefix) is ignored here because of mock above
    item = gcloud_storage._read_json_object("")
    assert item is None


def test_storage_location(gcloud_storage):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "google:cloud-storage",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert gcloud_storage._storage_location(prefix) == exp


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
def test_get_location(gcloud_storage, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            gcloud_storage._get_storage_location(meta_data)
    else:
        assert gcloud_storage._get_storage_location(meta_data) == result
