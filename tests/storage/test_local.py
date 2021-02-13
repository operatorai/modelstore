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

import modelstore
import pytest
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    get_archive_path,
    get_domain_path,
    get_metadata_path,
)

# pylint: disable=protected-access
# pylint: disable=redefined-outer-name


@pytest.fixture
def fs_model_store(tmp_path):
    return FileSystemStorage(root_path=str(tmp_path))


def test_validate(fs_model_store):
    assert fs_model_store.validate()
    assert os.path.exists(fs_model_store.root_dir)


def test_upload(fs_model_store, tmp_path):
    source = os.path.join(tmp_path, "test-file.txt")
    Path(source).touch()

    model_path = os.path.join(
        fs_model_store.root_dir,
        get_archive_path("test-domain", source),
    )
    rsp = fs_model_store.upload("test-domain", "test-model-id", source)
    assert rsp["type"] == "file_system"
    assert rsp["path"] == model_path
    assert os.path.exists(model_path)


def test_set_meta_data(fs_model_store):
    for model in ["model-1", "model-2"]:
        created = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
        meta_data = {"created": created, "model_id": model}
        fs_model_store.set_meta_data("test-domain", model, meta_data)

        # Expected files
        meta_data_path = os.path.join(
            fs_model_store.root_dir,
            get_metadata_path("test-domain", model),
        )
        assert os.path.exists(meta_data_path)
        latest_version_path = os.path.join(
            fs_model_store.root_dir, get_domain_path("test-domain")
        )
        assert os.path.exists(latest_version_path)


def test_list_versions(fs_model_store):
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
        fs_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)
    versions = fs_model_store.list_versions("test-domain")
    assert len(versions) == 2
    # Reverse time sorted
    assert versions[0] == "model-2"
    assert versions[1] == "model-1"


def test_list_domains(fs_model_store):
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
        fs_model_store.set_meta_data(domain, model, meta_data)
        time.sleep(1)
    domains = fs_model_store.list_domains()
    assert len(domains) == 2
    # Reverse time sorted
    assert domains[0] == "domain-2"
    assert domains[1] == "domain-1"
