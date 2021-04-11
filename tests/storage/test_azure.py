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
from pathlib import Path
from unittest import mock

import pytest
from azure.storage.blob import (
    BlobClient,
    BlobProperties,
    BlobServiceClient,
    ContainerClient,
    StorageStreamDownloader,
)
from modelstore.storage.azure import (
    AzureBlobStorage,
    _format_location,
    _get_location,
)
from modelstore.storage.util.paths import get_archive_path

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=no-member


@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    with mock.patch.dict(
        os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-value"}
    ):
        yield


@pytest.fixture
def mock_blob_client():
    blob_client = mock.create_autospec(BlobClient)
    blob_stream = mock.create_autospec(StorageStreamDownloader)
    blob_stream.readall.return_value = str.encode(json.dumps({"k": "v"}))
    blob_client.download_blob.return_value = blob_stream
    return blob_client


@pytest.fixture
def azure_client(mock_blob_client):
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
def azure_client_no_container(mock_blob_client):
    mock_container_client = mock.create_autospec(ContainerClient)
    mock_container_client.exists.return_value = False
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_client = mock.create_autospec(BlobServiceClient)
    mock_client.get_container_client.return_value = mock_container_client
    return mock_client


@pytest.fixture
def temp_file(tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()
    return source


@pytest.fixture
def azure_storage(azure_client):
    return AzureBlobStorage(
        container_name="existing-container", client=azure_client
    )


def test_validate(azure_storage, azure_client_no_container):
    assert azure_storage.validate()
    azure_storage = AzureBlobStorage(
        container_name="missing-container", client=azure_client_no_container
    )
    assert not azure_storage.validate()


def test_push(azure_storage, temp_file):
    azure_storage._push(temp_file, "destination")
    blob_client = azure_storage._blob_client("destination")
    blob_client.upload_blob.assert_called()


def test_pull(azure_storage, temp_file):
    source = {
        "container": "existing-container",
        "prefix": "source",
    }
    azure_storage._pull(source, temp_file)
    blob_client = azure_storage._blob_client("destination")
    blob_client.download_blob.assert_called()
    with open(temp_file, "r") as lines:
        contents = lines.read()
        assert contents == '{"k": "v"}'


def test_upload(azure_storage, temp_file):
    model_path = get_archive_path("test-domain", temp_file)
    rsp = azure_storage.upload("test-domain", "test-model-id", temp_file)

    blob_client = azure_storage._blob_client(model_path)
    blob_client.upload_blob.assert_called()

    assert rsp["type"] == "azure:blob-storage"
    assert rsp["prefix"] == model_path
    assert rsp["container"] == azure_storage.container_name


def test_read_json_objects(azure_storage):
    result = azure_storage._read_json_objects("path/to/files")
    azure_storage._container_client().list_blobs.assert_called_with(
        "path/to/files/"
    )
    assert len(result) == 3


def test_read_json_object(azure_storage):
    result = azure_storage._read_json_object("path/to/files")
    azure_storage._blob_client("path/to/files").download_blob.assert_called()
    assert result == {"k": "v"}


def test_format_location():
    container_name = "my-container"
    prefix = "/path/to/file"
    exp = {
        "type": "azure:blob-storage",
        "container": container_name,
        "prefix": prefix,
    }
    assert _format_location(container_name, prefix) == exp


def test_get_location() -> str:
    exp = "/path/to/file"
    container_name = "my-container"
    meta = {
        "type": "azure:blob-storage",
        "container": container_name,
        "prefix": "/path/to/file",
    }
    assert _get_location(container_name, meta) == exp
