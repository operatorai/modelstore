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
from datetime import datetime

import pytest
from modelstore.storage.states.model_states import ReservedModelStates
from modelstore.storage.util.paths import (
    get_model_state_path,
)

# pylint: disable=unused-import
from tests.storage.test_blob_storage import (
    mock_blob_storage,
    mock_meta_data,
)

# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring


def assert_file_contents_equals(file_path: str, expected: dict):
    # pylint: disable=unspecified-encoding
    with open(file_path, "r") as lines:
        actual = lines.read()
    assert json.dumps(expected) == actual


def test_state_exists(mock_blob_storage):
    assert not mock_blob_storage.state_exists("foo")


def test_create_model_state(mock_blob_storage):
    state_name = "production"
    # Create a model state
    mock_blob_storage.create_model_state(state_name)

    # Assert that a file at this location was created
    state_path = os.path.join(
        mock_blob_storage.root_prefix,
        get_model_state_path(mock_blob_storage.root_prefix, state_name),
    )
    assert_file_contents_equals(
        state_path,
        {
            "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
            "state_name": state_name,
        },
    )


def test_create_model_state_exists(mock_blob_storage):
    state_name = "production"

    # Create a model state
    mock_blob_storage.create_model_state(state_name)

    # Assert it exists
    assert mock_blob_storage.state_exists(state_name)
    assert not mock_blob_storage.state_exists("a-new-state")


def test_list_model_states(mock_blob_storage):
    model_states = ["staging", "production"]
    # Create model states
    for model_state in model_states:
        mock_blob_storage.create_model_state(model_state)

    # List them back
    results = mock_blob_storage.list_model_states()
    assert len(results) == 2
    for model_state in model_states:
        assert model_state in model_states


def test_set_model_state_unknown_state(mock_blob_storage):
    with pytest.raises(Exception):
        # Try to set a state without creating it first
        mock_blob_storage.set_model_state("domain", "model-id", "a-new-state")


def test_set_and_unset_model_state(mock_blob_storage):
    state_name = "production"
    model_id = "model-1"
    domain_id = "domain-1"

    # Create a models in a domain
    meta_data = mock_meta_data("domain-1", model_id, inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", model_id, meta_data)

    # Create a state and set the model to that state
    mock_blob_storage.create_model_state(state_name)
    assert mock_blob_storage.state_exists(state_name)

    # Set the model to that state
    mock_blob_storage.set_model_state(domain_id, model_id, state_name)

    # Listing versions
    items = mock_blob_storage.list_models(domain_id, state_name)
    assert len(items) == 1
    assert items[0] == model_id

    # Unset the state
    mock_blob_storage.unset_model_state(domain_id, model_id, state_name)

    # Model has been removed from the state
    items = mock_blob_storage.list_models(domain_id, state_name)
    assert len(items) == 0


def test_set_and_unset_reserved_model_state(mock_blob_storage):
    state_name = ReservedModelStates.DELETED.value
    model_id = "model-1"
    domain_id = "domain-1"

    # Create a models in a domain
    meta_data = mock_meta_data("domain-1", model_id, inc_time=0)
    mock_blob_storage.set_meta_data("domain-1", model_id, meta_data)

    # Set the model to that state
    mock_blob_storage.set_model_state(domain_id, model_id, state_name)

    # The state exists, without needing to be explicitly created
    assert mock_blob_storage.state_exists(state_name)

    # The model is in the new state
    items = mock_blob_storage.list_models(domain_id, state_name)
    assert len(items) == 1
    assert items[0] == model_id

    # Unset the state
    mock_blob_storage.unset_model_state(domain_id, model_id, state_name)

    # The action was not allowed, so the model still exists in this state
    items = mock_blob_storage.list_models(domain_id, state_name)
    assert len(items) == 1
    assert items[0] == model_id

    # Unset the state, explicitly flagging that we are modifying a
    # reserved state
    mock_blob_storage.unset_model_state(
        domain_id,
        model_id,
        state_name,
        modifying_reserved=True,
    )

    # The model has been removed from the state
    items = mock_blob_storage.list_models(domain_id, state_name)
    assert len(items) == 0
