#    Copyright 2022 Neal Lathia
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
import uuid

import modelstore
import pytest
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
    get_domain_path,
    get_models_path,
)


def mock_meta_data(domain: str, model_id: str, inc_time: int):
    upload_time = datetime.now() + timedelta(hours=inc_time)
    return {
        "model": {
            "domain": domain,
            "model_id": model_id,
        },
        "code": {"created": upload_time.strftime("%Y/%m/%d/%H:%M:%S")},
        "modelstore": modelstore.__version__,
    }


@pytest.fixture
def mock_model_file(tmp_path):
    model_file = os.path.join(tmp_path, "test-file.txt")
    Path(model_file).touch()
    return model_file


@pytest.fixture
def mock_blob_storage(tmp_path):
    return FileSystemStorage(str(tmp_path))


def assert_file_contents_equals(file_path: str, expected: dict):
    with open(file_path, "r") as lines:
        actual = lines.read()
    assert json.dumps(expected) == actual


def test_list_domains(mock_blob_storage):
    # Create two models in two domains
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    meta_data = mock_meta_data("domain-2", "model-1", inc_time=1)
    mock_blob_storage.set_meta_data("domain-2", "model-1", meta_data)

    # The results should be reverse time sorted
    domains = mock_blob_storage.list_domains()
    assert len(domains) == 2
    assert domains[0] == "domain-2"
    assert domains[1] == "domain-1"


def test_list_models(mock_blob_storage):
    # Create two models in one domain
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    meta_data = mock_meta_data("domain-1", "model-2", inc_time=1)
    mock_blob_storage.set_meta_data("domain-1", "model-2", meta_data)

    # List the models in domain-1; we expect two
    models = mock_blob_storage.list_models("domain-1")
    assert len(models) == 2

    # The results should be reverse time sorted
    assert models[0] == "model-2"
    assert models[1] == "model-1"


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
    # Set the meta data of a fake model
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Expected two files to be created
    # (1) The meta data for the 'latest' model
    domain_meta_data_path = get_domain_path(mock_blob_storage.root_prefix, "domain-1")
    assert_file_contents_equals(domain_meta_data_path, meta_data)

    # (2) The meta data for a specific model
    model_meta_data_path = os.path.join(
        get_models_path(mock_blob_storage.root_prefix, "domain-1"),
        f"model-1.json",
    )
    assert_file_contents_equals(model_meta_data_path, meta_data)


def test_get_meta_data(mock_blob_storage):
    # Set the meta data of a fake model
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Retrieve it back
    assert mock_blob_storage.get_meta_data("domain-1", "model-1") == meta_data


@pytest.mark.parametrize(
    "domain,model_id",
    [(None, "model-2"), ("", "model-2"), ("domain-1", None), ("domain-1", "")],
)
def test_get_meta_data_undefined_input(mock_blob_storage, domain, model_id):
    with pytest.raises(ValueError):
        mock_blob_storage.get_meta_data(domain, model_id)
