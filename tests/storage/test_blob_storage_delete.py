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
from datetime import datetime
import uuid

import modelstore
import pytest
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
    get_archive_path,
    get_domain_path,
    get_model_state_path,
    get_models_path,
)


def test_delete_model(mock_blob_storage):
    # Set the meta data against a fake model
    mock_blob_storage.set_meta_data("test-domain", "model-123", {"key": "value"})

    # Assert the meta data file has been created
    meta_data_path = mock_blob_storage._get_metadata_path("test-domain", "model-123")
    assert os.path.exists(meta_data_path)

    # Assert the meta data file is deleted
    mock_blob_storage.delete_model("test-domain", "model-123")
    assert not os.path.exists(meta_data_path)
