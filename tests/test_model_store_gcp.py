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
from mock import patch

from modelstore.model_store import ModelStore
from modelstore.models.managers import _LIBRARIES

# pylint: disable=unused-import
from tests.test_utils import (
    iter_only_sklearn,
    validate_library_attributes,
)

# pylint: disable=missing-function-docstring


@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud(mock_gcloud):
    mocked_gcloud = mock_gcloud("project-name", "gcs-bucket-name")
    mocked_gcloud.validate.return_value = True

    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    validate_library_attributes(store, allowed=_LIBRARIES, not_allowed=[])


@patch("modelstore.model_store.iter_libraries", side_effect=iter_only_sklearn)
@patch("modelstore.model_store.GoogleCloudStorage", autospec=True)
def test_from_gcloud_only_sklearn(mock_gcloud, libraries_without_sklearn):
    mocked_gcloud = mock_gcloud("project-name", "gcs-bucket-name")
    mocked_gcloud.validate.return_value = True
    store = ModelStore.from_gcloud("project-name", "gcs-bucket-name")
    validate_library_attributes(
        store, allowed=["sklearn"], not_allowed=libraries_without_sklearn
    )
