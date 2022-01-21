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
from datetime import datetime, timedelta
from pathlib import Path

import modelstore
import pytest
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
    get_archive_path,
    get_domain_path,
    get_model_state_path,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def mock_blob_storage(tmp_path):
    mock_storage = FileSystemStorage(str(tmp_path))
    # Adds two domains, each with two models
    # Note: this only adds meta-data, doesn't add any artifacts
    upload_time = datetime.now()
    for domain in ["domain-1", "domain-2"]:
        for model in ["model-1", "model-2"]:
            created = upload_time.strftime("%Y/%m/%d/%H:%M:%S")
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
            mock_storage.set_meta_data(domain, model, meta_data)
            upload_time += timedelta(hours=1)
    return mock_storage


def get_file_contents(file_path: str):
    with open(file_path, "r") as lines:
        return lines.read()


def test_state_exists(mock_blob_storage):
    assert not mock_blob_storage.state_exists("foo")


def test_get_metadata_path(mock_blob_storage):
    exp = os.path.join(
        mock_blob_storage.root_prefix,
        MODELSTORE_ROOT_PREFIX,
        "domain",
        "versions",
        "model-id.json",
    )
    res = mock_blob_storage._get_metadata_path("domain", "model-id")
    assert exp == res


def test_set_meta_data(mock_blob_storage):
    # Create a meta data string
    meta_dict = {"key": "value"}
    meta_str = json.dumps(meta_dict)

    # Set the meta data against a fake model
    mock_blob_storage.set_meta_data("test-domain", "model-123", meta_dict)

    # Expected two uploads
    # (1) The meta data for the 'latest' model
    meta_data = os.path.join(
        mock_blob_storage.root_prefix, get_domain_path("", "test-domain")
    )
    assert get_file_contents(meta_data) == meta_str

    # (2) The meta data for a specific model
    meta_data_path = mock_blob_storage._get_metadata_path("test-domain", "model-123")
    assert get_file_contents(meta_data_path) == meta_str


def test_get_meta_data(mock_blob_storage):
    exp = {"domain": "domain-1", "model_id": "model-2"}
    res = mock_blob_storage.get_meta_data("domain-1", "model-2")
    assert res["model"] == exp


@pytest.mark.parametrize(
    "domain,model_id",
    [(None, "model-2"), ("", "model-2"), ("domain-1", None), ("domain-1", "")],
)
def test_get_meta_data_undefined_input(mock_blob_storage, domain, model_id):
    with pytest.raises(ValueError):
        mock_blob_storage.get_meta_data(domain, model_id)


def test_upload(mock_blob_storage, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    model_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_archive_path(mock_blob_storage.root_prefix, "test-domain", source),
    )
    rsp = mock_blob_storage.upload("test-domain", source)
    assert rsp["type"] == "file_system"
    assert rsp["path"] == model_path
    assert os.path.exists(model_path)


def test_upload_extras(mock_blob_storage, tmp_path):
    # A 'model' file to be uploaded
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    # An additional file to upload alongside the model
    extra_path = os.path.join(tmp_path, "extra-file.txt")
    Path(extra_path).touch()

    # Upload the model
    rsp = mock_blob_storage.upload("test-domain", source, extras=extra_path)
    assert rsp["type"] == "file_system"

    # The model will be uploaded to the right place
    model_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_archive_path(mock_blob_storage.root_prefix, "test-domain", source),
    )
    assert rsp["path"] == model_path
    assert os.path.exists(model_path)

    # The extras are also uploaded alongside the model
    uploaded_extra_path = os.path.join(
        os.path.split(model_path)[0],
        "extra-file.txt",
    )
    assert os.path.exists(uploaded_extra_path)


def test_download_latest(mock_blob_storage):
    pass


def test_download(mock_blob_storage):
    pass


#     def download(self, local_path: str, domain: str, model_id: str = None):
#         """Downloads an artifacts archive for a given (domain, model_id) pair.
#         If no model_id is given, it defaults to the latest model in that
#         domain"""
#         model_meta = None
#         if model_id is None:
#             model_domain = get_domain_path(domain)
#             model_meta = self._read_json_object(model_domain)
#             logger.info("Latest model is: %f", model_meta["model"]["model_id"])
#         else:
#             model_meta_path = get_metadata_path(domain, model_id)
#             model_meta = self._read_json_object(model_meta_path)
#         return self._pull(model_meta["storage"], local_path)


def test_list_domains(mock_blob_storage):
    domains = mock_blob_storage.list_domains()
    assert len(domains) == 2
    # The results should be reverse time sorted
    assert domains[0] == "domain-2"
    assert domains[1] == "domain-1"


def test_list_versions(mock_blob_storage):
    # List the models in domain-1; we expect two
    versions = mock_blob_storage.list_versions("domain-1")
    assert len(versions) == 2
    # The results should be reverse time sorted
    assert versions[0] == "model-2"
    assert versions[1] == "model-1"


def test_create_model_state(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")

    # Assert that a file at this location was created
    state_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_model_state_path(mock_blob_storage.root_prefix, "production"),
    )
    state_meta = json.loads(get_file_contents(state_path))
    assert state_meta["state_name"] == "production"


def test_create_model_state_exists(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")
    assert mock_blob_storage.state_exists("production")
    assert not mock_blob_storage.state_exists("a-new-state")


def test_set_model_state_unknown_state(mock_blob_storage):
    with pytest.raises(Exception):
        # Try to set a state without creating it first
        mock_blob_storage.set_model_state("domain", "model-id", "a-new-state")


def test_set_model_state(mock_blob_storage):
    mock_blob_storage.create_model_state("production")
    mock_blob_storage.set_model_state("domain-1", "model-1", "production")

    # Listing versions
    items = mock_blob_storage.list_versions("domain-1", "production")
    assert len(items) == 1
    assert items[0] == "model-1"


def test_unset_model_state(mock_blob_storage):
    mock_blob_storage.create_model_state("production")
    mock_blob_storage.set_model_state("domain-1", "model-1", "production")

    # State now exists
    items = mock_blob_storage.list_versions("domain-1", "production")
    assert len(items) == 1
    assert items[0] == "model-1"

    # Unset the state
    mock_blob_storage.unset_model_state("domain-1", "model-1", "production")

    # State has been removed
    items = mock_blob_storage.list_versions("domain-1", "production")
    assert len(items) == 0
