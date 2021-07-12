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
from tests.storage.test_utils import temp_file

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


@pytest.fixture
def mock_blob_client():
    # Mocks the Azure Blob Client; reading a file using
    # this mock will return a static value "{"k": "v"}"
    blob_client = mock.create_autospec(BlobClient)
    blob_stream = mock.create_autospec(StorageStreamDownloader)
    blob_stream.readall.return_value = str.encode(json.dumps({"k": "v"}))
    blob_client.download_blob.return_value = blob_stream
    return blob_client


@pytest.fixture
def azure_client(mock_blob_client):
    # Mocks the Azure Container Client; listing the files
    # in this container will return a static list ["a.json", "b.json", "c.json"]
    mock_container_client = mock.create_autospec(ContainerClient)
    mock_container_client.exists.return_value = True
    mock_container_client.list_blobs.return_value = [
        BlobProperties(name=x + ".json") for x in ["a", "b", "c"]
    ]
    mock_container_client.get_blob_client.return_value = mock_blob_client
    mock_client = mock.create_autospec(BlobServiceClient)
    mock_client.get_container_client.return_value = mock_container_client
    return mock_client


@pytest.fixture
def azure_storage(azure_client):
    return AzureBlobStorage(
        container_name=_MOCK_CONTAINER_NAME, client=azure_client
    )


def test_validate_existing_container(azure_storage):
    assert azure_storage.validate()


def test_validate_missing_container(mock_blob_client):
    mock_container_client = mock.create_autospec(ContainerClient)
    mock_container_client.exists.return_value = False
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_client = mock.create_autospec(BlobServiceClient)
    mock_client.get_container_client.return_value = mock_container_client

    azure_storage = AzureBlobStorage(
        container_name="missing-container", client=mock_client
    )
    assert not azure_storage.validate()


def test_push(azure_storage, temp_file):
    # Asserts that pushing a file results in an upload
    azure_storage._push(temp_file, "destination")
    blob_client = azure_storage._blob_client("destination")
    blob_client.upload_blob.assert_called()


def test_pull(azure_storage, tmp_path):
    # Asserts that pulling a file results in a download
    azure_storage._pull("source", tmp_path)
    blob_client = azure_storage._blob_client("destination")
    blob_client.download_blob.assert_called()
    with open(os.path.join(tmp_path, "source"), "r") as lines:
        contents = lines.read()
        assert contents == '{"k": "v"}'


def test_read_json_objects(azure_storage):
    # Assert that listing the files at a prefix results in 3
    # files (returned statically in the mock)
    result = azure_storage._read_json_objects("path/to/files")
    azure_storage._container_client().list_blobs.assert_called_with(
        "path/to/files/"
    )
    assert len(result) == 3


def test_read_json_object(azure_storage):
    # Assert that reading a JSON object triggers a download
    #  and returns the mocked content
    result = azure_storage._read_json_object("path/to/files")
    azure_storage._blob_client("path/to/files").download_blob.assert_called()
    assert result == {"k": "v"}


def test_storage_location(azure_storage):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "azure:blob-storage",
        "container": _MOCK_CONTAINER_NAME,
        "prefix": prefix,
    }
    assert azure_storage._storage_location(prefix) == exp


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
def test_get_location(azure_storage, meta_data, should_raise, result):
    # Asserts that pulling the location out of meta data is correct
    if should_raise:
        with pytest.raises(ValueError):
            azure_storage._get_storage_location(meta_data)
    else:
        assert azure_storage._get_storage_location(meta_data) == result
