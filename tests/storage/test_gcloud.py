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


def gcloud_bucket():
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_bucket.name = _MOCK_BUCKET_NAME
    return mock_bucket


def gcloud_client(
    bucket_exists: bool,
    files_exist: bool,
    file_contents: str = TEST_FILE_CONTENTS,
):
    mock_client = mock.create_autospec(storage.Client)
    mock_buckets = []
    if bucket_exists:
        mock_bucket = gcloud_bucket()
        mock_bucket.client = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_buckets.append(mock_bucket)

        mock_blob = mock.create_autospec(storage.Blob)
        mock_blob.exists.return_value = files_exist
        if files_exist:
            mock_blob.download_as_string.return_value = file_contents
        mock_bucket.blob.return_value = mock_blob
    mock_client.list_buckets.return_value = mock_buckets
    return mock_client


def gcloud_storage(
    mock_client: storage.Client, bucket_name: str = _MOCK_BUCKET_NAME
):
    return GoogleCloudStorage(
        project_name="project-name",
        bucket_name=bucket_name,
        client=mock_client,
    )


def test_create_from_environment_variables(monkeypatch):
    monkeypatch.setenv("MODEL_STORE_GCP_PROJECT", "project")
    monkeypatch.setenv("MODEL_STORE_GCP_BUCKET", _MOCK_BUCKET_NAME)
    # Does not fail when environment variables exist
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


@pytest.mark.parametrize(
    "bucket_exists,validate_should_pass",
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
def test_validate(bucket_exists, validate_should_pass):
    mock_client = gcloud_client(bucket_exists=bucket_exists, files_exist=False)
    storage = gcloud_storage(mock_client)
    assert storage.validate() == validate_should_pass


def test_push(tmp_path):
    # Create a client
    mock_client = gcloud_client(bucket_exists=True, files_exist=False)
    storage = gcloud_storage(mock_client)

    # Push a file
    prefix = remote_file_path()
    result = storage._push(temp_file(tmp_path), prefix)

    # Assert that the correct prefix is returned
    # and that an upload happened
    assert result == prefix

    # Assert that an upload happened
    mock_bucket = storage.client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)
    mock_blob.upload_from_file.assert_called()


def test_pull(tmp_path):
    # Create a client
    mock_client = gcloud_client(bucket_exists=True, files_exist=True)
    storage = gcloud_storage(mock_client)

    # Pull the file back from storage
    prefix = remote_file_path()
    result = storage._pull(prefix, tmp_path)

    # Assert returned path
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    assert result == local_destination

    #  Assert download happened
    mock_bucket = storage.client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)
    mock_blob.download_to_filename.assert_called_with(local_destination)


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
    # Create a client
    mock_client = gcloud_client(bucket_exists=True, files_exist=file_exists)
    storage = gcloud_storage(mock_client)
    prefix = remote_file_path()
    mock_bucket = mock_client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)
    try:
        # Remove the file
        file_removed = storage._remove(prefix)
        if should_call_delete:
            assert file_removed
            # Asserts that removing the file results in a removal
            mock_blob.delete.assert_called()
        else:
            assert not file_removed
            # Asserts that we don't call delete on a file that doesn't exist
            mock_blob.delete.assert_not_called()
    except:
        # Should fail gracefully here
        pytest.fail("Remove raised an exception")


def test_read_json_objects_ignores_non_json():
    mock_client = gcloud_client(bucket_exists=True, files_exist=False)
    mock_client.list_blobs.return_value = [
        Blob(name="test-file-source-1.txt", bucket=_MOCK_BUCKET_NAME),
        Blob(name="test-file-source-2.txt", bucket=_MOCK_BUCKET_NAME),
    ]
    # Argument (remote prefix) is ignored here because of mock above
    storage = gcloud_storage(mock_client)
    items = storage._read_json_objects("")
    assert len(items) == 0


def test_read_json_object_fails_gracefully():
    prefix = remote_file_path()
    mock_client = gcloud_client(
        bucket_exists=True, files_exist=True, file_contents="not json"
    )
    storage = gcloud_storage(mock_client)

    # Read a file that does not contain any JSON
    # Argument (remote prefix) is ignored here because of mock above
    item = storage._read_json_object(prefix)
    assert item is None


def test_storage_location():
    mock_client = gcloud_client(bucket_exists=False, files_exist=False)
    storage = gcloud_storage(mock_client)

    # Asserts that the location meta data is correctly formatted
    prefix = remote_file_path()
    exp = {
        "type": "google:cloud-storage",
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
    mock_client = gcloud_client(bucket_exists=False, files_exist=False)
    storage = gcloud_storage(mock_client)

    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
