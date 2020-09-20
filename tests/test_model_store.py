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
import uuid
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from modelstore.clouds.file_system import FileSystemStorage
from modelstore.model_store import ModelStore
from modelstore.models.managers import ML_LIBRARIES
from modelstore.models.missingmanager import MissingDepManager
from modelstore.models.modelmanager import ModelManager
from modelstore.models.sklearn import SKLearnManager

# pylint: disable=protected-access


def validate_library_attributes(
    store: ModelStore, allowed: list, not_allowed: list
):
    # During dev mode, all libraries will be installed
    for library in allowed:
        assert hasattr(store, library)
        mgr = store.__getattribute__(library)
        assert issubclass(type(mgr), ModelManager)
        assert not isinstance(mgr, MissingDepManager)

    for library in not_allowed:
        assert hasattr(store, library)
        mgr = store.__getattribute__(library)
        assert issubclass(type(mgr), ModelManager)
        assert isinstance(mgr, MissingDepManager)
        with pytest.raises(ModuleNotFoundError):
            mgr.create_archive(model="test")

    # Libraries that have not been implemented (yet!)
    for library in ["tensorflow"]:
        assert not hasattr(store, library)


@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud(mock_gcloud):
    mock_gcloud("project-name", "gcs-bucket-name").validate.return_value = True

    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    assert store.storage._extract_mock_name() == "GoogleCloudStorage()"
    validate_library_attributes(store, allowed=ML_LIBRARIES, not_allowed=[])


def test_from_file_system(tmp_path):
    store = ModelStore.from_file_system(root_directory=str(tmp_path))
    assert isinstance(store.storage, FileSystemStorage)
    validate_library_attributes(store, allowed=ML_LIBRARIES, not_allowed=[])


def only_sklearn(library):
    if library == "sklearn":
        return SKLearnManager
    return partial(MissingDepManager, library=library)


@patch("modelstore.model_store.get_manager", side_effect=only_sklearn)
@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud_only_sklearn(mock_gcloud, _):
    mock_gcloud("project-name", "gcs-bucket-name").validate.return_value = True

    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    assert store.storage._extract_mock_name() == "GoogleCloudStorage()"

    libraries = ML_LIBRARIES.copy()
    libraries.pop("sklearn")
    validate_library_attributes(
        store, allowed=["sklearn"], not_allowed=libraries
    )


@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_upload(mock_gcloud, tmp_path):
    mock_gcloud("project-name", "gcs-bucket-name").validate.return_value = True
    mock_gcloud(
        "project-name", "gcs-bucket-name"
    ).get_name.return_value = "name"
    mock_gcloud("project-name", "gcs-bucket-name").upload.return_value = {
        "bucket": "gcs-bucket-name"
    }
    tmp_file = os.path.join(tmp_path, "test.txt")
    Path(tmp_file).touch()

    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")

    meta_data = store.upload("test-domain", tmp_file)
    # A call to upload() will upload the:
    # (1) The model archive itself
    # (2) The meta-data
    assert store.storage.upload.call_count == 1
    assert store.storage.set_meta_data.call_count == 1

    # Asserting that keys exist; values are tested separately
    keys = ["model", "storage", "meta"]
    assert all(k in meta_data for k in keys)

    keys = ["runtime", "user"]
    assert all(k in meta_data["meta"] for k in keys)

    assert meta_data["storage"]["name"] == "name"
    assert meta_data["storage"]["location"]["bucket"] == "gcs-bucket-name"
    assert meta_data["model"]["domain"] == "test-domain"
    try:
        # Test that the model_id is a valid UUID
        uuid.UUID(meta_data["model"]["model_id"])
    except ValueError:
        pytest.fail("Invalid uuid")
