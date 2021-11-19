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
from unittest import mock

import pytest
from azure.storage.blob import (
    BlobClient,
    BlobProperties,
    BlobServiceClient,
    ContainerClient,
    StorageStreamDownloader,
)
from modelstore.storage.azure import AzureBlobStorage

# pylint: disable=unused-import
from tests.storage.test_utils import (
    TEST_FILE_CONTENTS,
    TEST_FILE_LIST,
    TEST_FILE_NAME,
    file_contains_expected_contents,
    remote_file_path,
    remote_path,
    temp_file,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=no-member
_MOCK_CONTAINER_NAME = "existing-container"


@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    # The AzureStorage client assumes that this environ variable
    #  has been set
    with mock.patch.dict(
        os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-value"}
    ):
        yield


def mock_blob_client(file_exists: bool):
    # Mocks the Azure Blob Client; reading a file using
    # this mock will return a static value "{"k": "v"}"
    blob_client = mock.create_autospec(BlobClient)
    blob_client.exists.return_value = file_exists
    if file_exists:
        blob_stream = mock.create_autospec(StorageStreamDownloader)
        blob_stream.readall.return_value = str.encode(TEST_FILE_CONTENTS)
        blob_client.download_blob.return_value = blob_stream
    return blob_client


def mock_container_client(container_exists: bool, files_exist: bool):
    # Mocks the Azure Container Client; listing the files
    # in this container will return a static list ["a.json", "b.json", "c.json"]
    mock_container_client = mock.create_autospec(ContainerClient)
    mock_container_client.exists.return_value = container_exists
    mock_container_client.get_blob_client.return_value = mock_blob_client(
        file_exists=files_exist
    )
    if files_exist:
        mock_container_client.list_blobs.return_value = [
            BlobProperties(name=x) for x in TEST_FILE_LIST
        ]
    return mock_container_client


def mock_blob_service_client(container_exists: bool, files_exist: bool):
    # Mocks the Azure Service Client
    container_client = mock_container_client(container_exists, files_exist)
    mock_client = mock.create_autospec(BlobServiceClient)
    mock_client.get_container_client.return_value = container_client
    return mock_client


def azure_storage(blob_service_client):
    return AzureBlobStorage(
        container_name=_MOCK_CONTAINER_NAME, client=blob_service_client
    )


def test_create_from_environment_variables(monkeypatch):
    # Does not fail when environment variables exist
    monkeypatch.setenv("MODEL_STORE_AZURE_CONTAINER", _MOCK_CONTAINER_NAME)
    # pylint: disable=bare-except
    try:
        _ = AzureBlobStorage()
    except:
        pytest.fail("Failed to initialise storage from env variables")


def test_create_fails_with_missing_environment_variables(monkeypatch):
    # Fails when environment variables are missing
    for key in AzureBlobStorage.BUILD_FROM_ENVIRONMENT.get("required", []):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(KeyError):
        _ = AzureBlobStorage()


@pytest.mark.parametrize(
    "container_exists,validate_should_pass",
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
def test_validate(container_exists, validate_should_pass):
    blob_service_client = mock_blob_service_client(
        container_exists=container_exists,
        files_exist=False,
    )
    storage = azure_storage(blob_service_client)
    assert storage.validate() == validate_should_pass


def test_push(tmp_path):
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=False,
    )
    storage = azure_storage(blob_service_client)

    # Push a file to storage
    prefix = remote_file_path()
    storage._push(temp_file(tmp_path), prefix)

    # Asserts that pushing a file results in an upload
    blob_client = storage._blob_client(prefix)
    blob_client.upload_blob.assert_called()


def test_pull(tmp_path):
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=True,
    )
    storage = azure_storage(blob_service_client)

    # Pull a file from storage
    prefix = remote_file_path()
    result = storage._pull(prefix, tmp_path)

    # Asserts that pulling a file results in a download
    blob_client = storage._blob_client(prefix)
    blob_client.download_blob.assert_called()
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
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=file_exists,
    )
    storage = azure_storage(blob_service_client)
    prefix = remote_file_path()
    try:
        file_removed = storage._remove(prefix)
        blob_client = storage._blob_client(prefix)
        if should_call_delete:
            assert file_removed
            # Asserts that removing the file results in a removal
            blob_client.delete_blob.assert_called()
        else:
            assert not file_removed
            # Asserts that we don't call delete on a file that doesn't exist
            blob_client.delete_blob.assert_not_called()
    except:
        # Should fail gracefully here
        pytest.fail("Remove raised an exception")


def test_read_json_objects():
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=True,
    )
    storage = azure_storage(blob_service_client)
    # Assert that listing the files at a prefix results in 3
    # files (returned statically in the mock)
    result = storage._read_json_objects("path/to/files")
    storage._container_client().list_blobs.assert_called_with("path/to/files/")
    assert len(result) == 3


def test_read_json_object():
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=True,
    )
    storage = azure_storage(blob_service_client)
    # Assert that reading a JSON object triggers a download
    # and returns the mocked content
    result = storage._read_json_object("path/to/files")
    storage._blob_client("path/to/files").download_blob.assert_called()
    assert json.dumps(result) == TEST_FILE_CONTENTS


def test_storage_location():
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=True,
    )
    storage = azure_storage(blob_service_client)
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "azure:blob-storage",
        "container": _MOCK_CONTAINER_NAME,
        "prefix": prefix,
    }
    assert storage._storage_location(prefix) == exp


@pytest.mark.parametrize(
    "meta_data,should_raise,result",
    [
        (
            {
                "container": _MOCK_CONTAINER_NAME,
                "prefix": "/path/to/file",
            },
            False,
            "/path/to/file",
        ),
        (
            {
                "container": "a-different-bucket",
                "prefix": "/path/to/file",
            },
            True,
            None,
        ),
    ],
)
def test_get_location(meta_data, should_raise, result):
    # Create a mock storage instance
    blob_service_client = mock_blob_service_client(
        container_exists=True,
        files_exist=False,
    )
    storage = azure_storage(blob_service_client)
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            storage._get_storage_location(meta_data)
    else:
        assert storage._get_storage_location(meta_data) == result
