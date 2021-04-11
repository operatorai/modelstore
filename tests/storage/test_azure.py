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
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import modelstore
import pytest
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient
from modelstore.storage.azure import AzureBlobStorage
from modelstore.storage.util.paths import (
    get_archive_path,
    get_domain_path,
    get_domains_path,
    get_versions_path,
)

# pylint: disable=redefined-outer-name


@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    with mock.patch.dict(
        os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test-value"}
    ):
        yield


@pytest.fixture
def azure_client():
    # Blob client
    mock_blob_client = mock.create_autospec(BlobClient)

    # Container clinet
    mock_container_client = mock.create_autospec(ContainerClient)
    mock_container_client.exists.return_value = True
    mock_container_client.get_blob_client.return_value = mock_blob_client

    # Client
    mock_client = mock.create_autospec(BlobServiceClient)
    mock_client.get_container_client.return_value = mock_container_client
    return mock_client


def test_validate(azure_client):
    azure_storage = AzureBlobStorage(
        container_name="existing-container", client=azure_client
    )
    assert azure_storage.validate()
    # azure_storage = AzureBlobStorage(
    #     container_name="missing-container", client=azure_client
    # )
    # import pdb

    # pdb.set_trace()
    # assert not azure_storage.validate()


def test_upload(azure_client, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    azure_storage = AzureBlobStorage(
        container_name="existing-container", client=azure_client
    )
    model_path = get_archive_path("test-domain", source)
    rsp = azure_storage.upload("test-domain", "test-model-id", source)

    # pylint disable=protected-access
    blob_client = azure_storage._blob_client(model_path)
    blob_client.upload_blob.assert_called()

    assert rsp["type"] == "azure:blob-storage"
    assert rsp["prefix"] == model_path
    assert rsp["container"] == azure_storage.container_name
