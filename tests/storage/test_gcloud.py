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
from pathlib import Path
from unittest import mock

import pytest
from google.cloud import storage
from modelstore.storage.gcloud import (
    GoogleCloudStorage,
    _format_location,
    _get_location,
)
from modelstore.storage.util.paths import get_archive_path

# pylint: disable=redefined-outer-name
_MOCK_BUCKET_NAME = "gcloud-bucket"


@pytest.fixture
def gcloud_client():
    mock_client = mock.create_autospec(storage.Client)

    # Buckets
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_bucket.name = _MOCK_BUCKET_NAME
    mock_bucket.client = mock_client

    # Blobs
    mock_blob = mock.create_autospec(storage.Blob)
    mock_bucket.blob.return_value = mock_blob

    mock_client.list_buckets.return_value = [mock_bucket]
    return mock_client


def test_validate_existing_bucket(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name=_MOCK_BUCKET_NAME, client=gcloud_client
    )
    assert gcloud_model_store.validate()


def test_validate_missing_bucket(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="missing-bucket", client=gcloud_client
    )
    assert not gcloud_model_store.validate()


def test_upload(gcloud_client, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    gcloud_model_store = GoogleCloudStorage(
        project_name="project-name",
        bucket_name=_MOCK_BUCKET_NAME,
        client=gcloud_client,
    )
    model_path = get_archive_path("test-domain", source)
    rsp = gcloud_model_store.upload("test-domain", "test-model-id", source)

    bucket = gcloud_client.get_bucket
    bucket.assert_called_with(gcloud_model_store.bucket_name)

    blob = bucket(gcloud_model_store.bucket_name).blob
    blob.assert_called_with(model_path)
    blob(model_path).upload_from_file.assert_called()

    assert rsp["type"] == "google:cloud-storage"
    assert rsp["prefix"] == model_path
    assert rsp["bucket"] == gcloud_model_store.bucket_name


def test_format_location():
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "google:cloud-storage",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert _format_location(_MOCK_BUCKET_NAME, prefix) == exp


def test_get_location() -> str:
    # Asserts that pulling the location out of meta data
    # is correct
    exp = "/path/to/file"
    meta = {
        "type": "google:cloud-storage",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": exp,
    }
    assert _get_location(_MOCK_BUCKET_NAME, meta) == exp
