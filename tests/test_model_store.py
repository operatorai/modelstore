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
from functools import partial
from unittest.mock import patch

import pytest
from modelstore.model_store import ModelStore
from modelstore.models.managers import ML_LIBRARIES
from modelstore.models.missingmanager import MissingDepManager
from modelstore.models.modelmanager import ModelManager
from modelstore.storage.local import FileSystemStorage

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
            mgr.upload(domain="test", model="test")


@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud(mock_gcloud):
    mocked_gcloud = mock_gcloud("project-name", "gcs-bucket-name")
    mocked_gcloud.validate.return_value = True

    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    validate_library_attributes(store, allowed=ML_LIBRARIES, not_allowed=[])


def test_from_file_system(tmp_path):
    store = ModelStore.from_file_system(root_directory=str(tmp_path))
    assert isinstance(store.storage, FileSystemStorage)
    validate_library_attributes(store, allowed=ML_LIBRARIES, not_allowed=[])


def only_sklearn(_):
    for k, v in ML_LIBRARIES.items():
        if k == "sklearn":
            yield k, v()
        else:
            yield k, partial(MissingDepManager, library=k)()


@patch("modelstore.model_store.iter_libraries", side_effect=only_sklearn)
@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud_only_sklearn(mock_gcloud, _):
    mocked_gcloud = mock_gcloud("project-name", "gcs-bucket-name")
    mocked_gcloud.validate.return_value = True
    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    libraries = ML_LIBRARIES.copy()
    libraries.pop("sklearn")
    validate_library_attributes(
        store, allowed=["sklearn"], not_allowed=libraries
    )
