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
import os
import json
from datetime import datetime, timedelta

import pytest
import modelstore
from modelstore.storage.local import FileSystemStorage
from modelstore.storage.util.paths import (
    MODELSTORE_ROOT_PREFIX,
    get_model_state_path,
)


@pytest.fixture
def mock_blob_storage(tmp_path):
    return FileSystemStorage(str(tmp_path))


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


def assert_file_contents_equals(file_path: str, expected: dict):
    with open(file_path, "r") as lines:
        actual = lines.read()
    assert json.dumps(expected) == actual


def test_state_exists(mock_blob_storage):
    assert not mock_blob_storage.state_exists("foo")


def test_create_model_state(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")

    # Assert that a file at this location was created
    state_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_model_state_path(mock_blob_storage.root_prefix, "production"),
    )
    assert_file_contents_equals(
        state_path,
        {
            "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
            "state_name": "production",
        },
    )


def test_create_model_state_exists(mock_blob_storage):
    # Create a model state
    mock_blob_storage.create_model_state("production")
    assert mock_blob_storage.state_exists("production")
    assert not mock_blob_storage.state_exists("a-new-state")


def test_list_model_states(mock_blob_storage):
    # Create a model states
    mock_blob_storage.create_model_state("staging")
    mock_blob_storage.create_model_state("production")

    # List them back
    model_states = mock_blob_storage.list_model_states()
    assert len(model_states) == 2
    assert "production" in model_states
    assert "staging" in model_states


def test_set_model_state_unknown_state(mock_blob_storage):
    with pytest.raises(Exception):
        # Try to set a state without creating it first
        mock_blob_storage.set_model_state("domain", "model-id", "a-new-state")


def test_set_model_state(mock_blob_storage):
    # Create a models in a domain
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Create a state and set the model to that state
    mock_blob_storage.create_model_state("production")
    mock_blob_storage.set_model_state("domain-1", "model-1", "production")

    # Listing versions
    items = mock_blob_storage.list_models("domain-1", "production")
    assert len(items) == 1
    assert items[0] == "model-1"


def test_unset_model_state(mock_blob_storage):
    # Create a models in a domain
    meta_data = mock_meta_data("domain-1", "model-1", inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", "model-1", meta_data)

    # Create a state and set the model to that state
    mock_blob_storage.create_model_state("production")
    mock_blob_storage.set_model_state("domain-1", "model-1", "production")

    # State now exists
    items = mock_blob_storage.list_models("domain-1", "production")
    assert len(items) == 1
    assert items[0] == "model-1"

    # Unset the state
    mock_blob_storage.unset_model_state("domain-1", "model-1", "production")

    # State has been removed
    items = mock_blob_storage.list_models("domain-1", "production")
    assert len(items) == 0
