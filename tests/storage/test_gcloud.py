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
import mock

import pytest
from google.cloud import storage
from google.cloud.storage.blob import Blob

from modelstore.metadata import metadata
from modelstore.storage.gcloud import GoogleCloudStorage

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_NAME,
    remote_file_path,
    push_temp_file,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=missing-function-docstring
_MOCK_BUCKET_NAME = "gcloud-bucket"
_MOCK_PROJECT_NAME = "project-name"


def gcloud_bucket(bucket_exists: bool):
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_bucket.name = _MOCK_BUCKET_NAME
    mock_bucket.exists.return_value = bucket_exists
    return mock_bucket


def gcloud_blob(blob_exists: bool, file_contents: str):
    mock_blob = mock.create_autospec(storage.Blob)
    mock_blob.exists.return_value = blob_exists
    if blob_exists:
        mock_blob.download_as_string.return_value = file_contents
    return mock_blob


def gcloud_anon_client(bucket_exists: bool, blob_exists: bool, file_contents: str):
    # Create a storage client
    mock_client = mock.create_autospec(storage.Client)

    # Add a bucket to the client; anonymous clients use .bucket
    # instead of .get_bucket
    mock_bucket = gcloud_bucket(bucket_exists)
    mock_client.bucket.return_value = mock_bucket

    # If the bucket exists, add a file to it
    if bucket_exists:
        mock_blob = gcloud_blob(blob_exists, file_contents)
        mock_bucket.blob.return_value = mock_blob
        if blob_exists:
            # The anonymous client can list the blobs
            mock_client.list_blobs.return_value = [mock_blob]
    return mock_client


def gcloud_authed_client(bucket_exists: bool, blob_exists: bool, file_contents: str):
    # Create a storage client
    mock_client = mock.create_autospec(storage.Client)

    # Add a bucket to the client
    mock_bucket = gcloud_bucket(bucket_exists)
    mock_client.get_bucket.return_value = mock_bucket

    # If the bucket exists, add a file to it
    if bucket_exists:
        mock_blob = gcloud_blob(blob_exists, file_contents)
        mock_bucket.blob.return_value = mock_blob
    return mock_client


def gcloud_storage(mock_client: storage.Client, is_anon_client: bool):
    return GoogleCloudStorage(
        project_name=_MOCK_PROJECT_NAME,
        bucket_name=_MOCK_BUCKET_NAME,
        client=mock_client,
        is_anon_client=is_anon_client,
    )


def gcloud_client(
    bucket_exists: bool,
    blob_exists: bool,
    is_anon_client: bool,
    file_contents: str = TEST_FILE_CONTENTS,
):
    if is_anon_client:
        client = gcloud_anon_client(bucket_exists, blob_exists, file_contents)
        return client, gcloud_storage(client, is_anon_client)
    else:
        client = gcloud_authed_client(bucket_exists, blob_exists, file_contents)
        return client, gcloud_storage(client, is_anon_client)


def test_create_from_environment_variables(monkeypatch):
    monkeypatch.setenv("MODEL_STORE_GCP_PROJECT", _MOCK_PROJECT_NAME)
    monkeypatch.setenv("MODEL_STORE_GCP_BUCKET", _MOCK_BUCKET_NAME)
    # Does not fail when environment variables exist
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


@pytest.mark.parametrize(
    "bucket_exists,is_anon_client,validate_should_pass",
    [
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, True),
    ],
)
def test_validate(bucket_exists, is_anon_client, validate_should_pass):
    _, storage = gcloud_client(
        bucket_exists=bucket_exists, blob_exists=False, is_anon_client=is_anon_client
    )
    assert storage.validate() == validate_should_pass


def test_push():
    _, storage = gcloud_client(
        bucket_exists=True, blob_exists=False, is_anon_client=False
    )

    # Push a file
    result = push_temp_file(storage)

    # Assert that the correct prefix is returned
    # and that an upload happened
    assert result == remote_file_path()

    # Assert that an upload happened
    mock_bucket = storage.client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(result)
    mock_blob.upload_from_file.assert_called()


def test_anonymous_push():
    _, storage = gcloud_client(
        bucket_exists=True, blob_exists=False, is_anon_client=True
    )

    with pytest.raises(NotImplementedError):
        _ = push_temp_file(storage)


def test_pull(tmp_path):
    _, storage = gcloud_client(
        bucket_exists=True, blob_exists=True, is_anon_client=False
    )

    # Pull the file back from storage
    prefix = remote_file_path()
    result = storage._pull(prefix, tmp_path)

    # Assert returned path
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    assert result == local_destination

    # Assert download happened
    mock_bucket = storage.client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)
    mock_blob.download_to_filename.assert_called_with(local_destination)


def test_anonymous_pull(tmp_path):
    _, storage = gcloud_client(
        bucket_exists=True, blob_exists=True, is_anon_client=True
    )

    # Pull the file back from storage
    prefix = remote_file_path()
    result = storage._pull(prefix, tmp_path)

    # Assert returned path
    local_destination = os.path.join(tmp_path, TEST_FILE_NAME)
    assert result == local_destination

    # Assert download happened
    mock_bucket = storage.client.bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)
    mock_blob.download_to_filename.assert_called_with(local_destination)


@pytest.mark.parametrize(
    "blob_exists,should_call_delete",
    [
        (False, False),
        (True, True),
    ],
)
def test_remove(blob_exists, should_call_delete):
    mock_client, storage = gcloud_client(
        bucket_exists=True, blob_exists=blob_exists, is_anon_client=False
    )

    prefix = remote_file_path()
    mock_bucket = mock_client.get_bucket(storage.bucket_name)
    mock_blob = mock_bucket.blob(prefix)

    file_removed = storage._remove(prefix)
    assert file_removed == should_call_delete
    if should_call_delete:
        # Asserts that removing the file results in a removal
        mock_blob.delete.assert_called()
    else:
        # Asserts that we don't call delete on a file that doesn't exist
        mock_blob.delete.assert_not_called()


def test_anonymous_remove():
    _, storage = gcloud_client(
        bucket_exists=True, blob_exists=True, is_anon_client=True
    )

    prefix = remote_file_path()
    with pytest.raises(NotImplementedError):
        _ = storage._remove(prefix)


def test_read_json_objects_ignores_non_json():
    mock_client, storage = gcloud_client(
        bucket_exists=True, blob_exists=False, is_anon_client=False
    )
    mock_client.list_blobs.return_value = [
        Blob(name="test-file-source-1.txt", bucket=_MOCK_BUCKET_NAME),
        Blob(name="test-file-source-2.txt", bucket=_MOCK_BUCKET_NAME),
    ]
    items = storage._read_json_objects("")
    assert len(items) == 0


def test_read_json_object_fails_gracefully():
    _, storage = gcloud_client(
        bucket_exists=True,
        blob_exists=True,
        is_anon_client=False,
        file_contents="not json",
    )
    prefix = remote_file_path()

    # Read a file that does not contain any JSON
    # Argument (remote prefix) is ignored here because of mock above
    item = storage._read_json_object(prefix)
    assert item is None


def test_storage_location():
    _, storage = gcloud_client(
        bucket_exists=False, blob_exists=False, is_anon_client=False
    )

    # Asserts that the location meta data is correctly formatted
    prefix = remote_file_path()
    expected = metadata.Storage.from_bucket(
        storage_type="google:cloud-storage",
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
    _, storage = gcloud_client(
        bucket_exists=False, blob_exists=False, is_anon_client=False
    )

    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
