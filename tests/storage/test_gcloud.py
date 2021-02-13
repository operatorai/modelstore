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
from google.cloud import storage
from modelstore.storage.gcloud import GoogleCloudStorage
from modelstore.storage.util.paths import (
    get_archive_path,
    get_domain_path,
    get_domains_path,
    get_versions_path,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def gcloud_client():
    mock_client = mock.create_autospec(storage.Client)

    # Buckets
    mock_bucket = mock.create_autospec(storage.Bucket)
    mock_bucket.name = "existing-bucket"
    mock_bucket.client = mock_client

    # Blobs
    mock_blob = mock.create_autospec(storage.Blob)
    mock_bucket.blob.return_value = mock_blob

    mock_client.list_buckets.return_value = [mock_bucket]
    return mock_client


def test_validate(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="existing-bucket", client=gcloud_client
    )
    assert gcloud_model_store.validate()
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="missing-bucket", client=gcloud_client
    )
    assert not gcloud_model_store.validate()


def test_upload(gcloud_client, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    gcloud_model_store = GoogleCloudStorage(
        project_name="project-name",
        bucket_name="existing-bucket",
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


def test_set_meta_data(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="existing-bucket", client=gcloud_client
    )
    gcloud_model_store.set_meta_data(
        "test-domain", "model-123", {"key": "value"}
    )

    # Expected two uploads
    meta_data = get_domain_path("test-domain")
    bucket = gcloud_client.get_bucket
    bucket.assert_called_with(gcloud_model_store.bucket_name)
    blob = bucket(gcloud_model_store.bucket_name).blob
    blob.assert_called_with(meta_data)
    assert blob(meta_data).upload_from_file.call_count == 2


def test_list_versions(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="existing-bucket", client=gcloud_client
    )

    domain = "test-domain"
    for model in ["model-1", "model-2"]:
        created = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        meta_data = {
            "model": {
                "domain": domain,
                "model_id": model,
            },
            "code": {
                "created": created,
            },
            "modelstore": modelstore.__version__,
        }
        gcloud_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)

    gcloud_model_store.list_versions(domain)
    versions_for_domain = get_versions_path(domain) + "/"
    gcloud_client.list_blobs.assert_called_with(
        "existing-bucket",
        prefix=versions_for_domain,
        delimiter="/",
    )


def test_list_domains(gcloud_client):
    gcloud_model_store = GoogleCloudStorage(
        project_name="", bucket_name="existing-bucket", client=gcloud_client
    )

    model = "test-model"
    for domain in ["domain-1", "domain-2"]:
        created = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        meta_data = {
            "model": {
                "domain": domain,
                "model_id": model,
            },
            "code": {
                "created": created,
            },
            "modelstore": modelstore.__version__,
        }
        gcloud_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)

    gcloud_model_store.list_domains()
    domains = get_domains_path() + "/"
    gcloud_client.list_blobs.assert_called_with(
        "existing-bucket",
        prefix=domains,
        delimiter="/",
    )
