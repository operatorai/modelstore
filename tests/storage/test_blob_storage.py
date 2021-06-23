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
    get_archive_path,
    get_domain_path,
    get_metadata_path,
    get_model_state_path,
)

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


@pytest.fixture
def mock_blob_storage(tmp_path):
    return FileSystemStorage(str(tmp_path))


def get_file_contents(file_path: str):
    with open(file_path, "r") as lines:
        return lines.read()


def test_state_exists(mock_blob_storage):
    assert not mock_blob_storage.state_exists("foo")


def test_set_meta_data(mock_blob_storage):
    # Create a meta data string
    meta_dict = {"key": "value"}
    meta_str = json.dumps(meta_dict)

    # Set the meta data against a fake model
    mock_blob_storage.set_meta_data("test-domain", "model-123", meta_dict)

    # Expected two uploads
    # The meta data for the 'latest' model
    meta_data = os.path.join(
        mock_blob_storage.root_dir, get_domain_path("test-domain")
    )
    assert get_file_contents(meta_data) == meta_str

    # The meta data for a specific model
    meta_data = os.path.join(
        mock_blob_storage.root_dir,
        get_metadata_path("test-domain", "model-123"),
    )
    assert get_file_contents(meta_data) == meta_str


def test_upload(mock_blob_storage, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    model_path = os.path.join(
        mock_blob_storage.root_dir,
        get_archive_path("test-domain", source),
    )
    rsp = mock_blob_storage.upload("test-domain", "test-model-id", source)
    assert rsp["type"] == "file_system"
    assert rsp["path"] == model_path
    assert os.path.exists(model_path)


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
    # Create and set meta data for two domains
    upload_time = datetime.now()
    model = "test-model"
    for domain in ["domain-1", "domain-2"]:
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
        mock_blob_storage.set_meta_data(domain, model, meta_data)
        upload_time += timedelta(hours=1)

    # List back the domains; we expect two
    domains = mock_blob_storage.list_domains()
    assert len(domains) == 2
    # The results should be reverse time sorted
    assert domains[0] == "domain-2"
    assert domains[1] == "domain-1"


def test_list_versions(mock_blob_storage):
    # Create and set meta data for two models in the same domain
    upload_time = datetime.now()
    domain = "test-domain"
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
        mock_blob_storage.set_meta_data(domain, model, meta_data)
        upload_time += timedelta(hours=1)

    # List back the models; we expect two
    versions = mock_blob_storage.list_versions(domain)
    assert len(versions) == 2
    # The results should be reverse time sorted
    assert versions[0] == "model-2"
    assert versions[1] == "model-1"


def test_create_model_state(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")

    # Assert that a file at this location was created
    state_path = os.path.join(
        mock_blob_storage.root_dir, get_model_state_path("production")
    )
    state_meta = json.loads(get_file_contents(state_path))
    assert state_meta["state_name"] == "production"


def test_set_model_state(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")


#     def set_model_state(self, domain: str, model_id: str, state_name: str):
#         """ Adds the given model ID the set that are in the state_name path """
#         if not self._state_exists(state_name):
#             logger.debug("Model state '%s' does not exist", state_name)
#             raise Exception(f"State '{state_name}' does not exist")
#         model_meta_data_path = get_metadata_path(domain, model_id)
#         versions_path = get_versions_path(domain, state_name)
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             meta_data = self._pull(model_meta_data_path, tmp_dir)
#             self._push(meta_data, versions_path)
