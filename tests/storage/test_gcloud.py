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
from unittest import mock

import pytest
from google.cloud import storage
from modelstore.storage.gcloud import GoogleCloudStorage, _get_location

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


def test_storage_location(gcloud_model_store):
    # Asserts that the location meta data is correctly formatted
    prefix = "/path/to/file"
    exp = {
        "type": "google:cloud-storage",
        "bucket": _MOCK_BUCKET_NAME,
        "prefix": prefix,
    }
    assert gcloud_model_store._storage_location(prefix) == exp


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
