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
from pathlib import PosixPath
import os

import pytest

from modelstore.model_store import ModelStore
from modelstore.storage.states.model_states import ReservedModelStates
from modelstore.utils.exceptions import (
    ModelExistsException,
    DomainNotFoundException,
    ModelNotFoundException,
)

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-import
from tests.test_utils import (
    model_file,
)


@pytest.fixture
def model_store(tmp_path: PosixPath):
    return ModelStore.from_file_system(root_directory=str(tmp_path))


def test_model_not_found(model_store: ModelStore, model_file: str):
    with pytest.raises(DomainNotFoundException):
        model_store.get_model_info("missing-domain", "missing-model")

    model_store.upload(
        domain="existing-domain", model_id="test-model-id-1", model=model_file
    )
    with pytest.raises(ModelNotFoundException):
        model_store.get_model_info("existing-domain", "missing-model")


def test_model_exists(model_store: ModelStore, model_file: str):
    domain = "test-domain"
    model_id = "test-model-id-1"

    # No models => domain not found => model doesn't exist
    assert not model_store.model_exists(domain, model_id)

    # Upload model 2 => domain exists => model 1 still doesn't exist
    model_store.upload(domain=domain, model_id="test-model-id-2", model=model_file)
    assert not model_store.model_exists(domain, model_id)

    # Upload model 1 => model exists
    model_store.upload(domain=domain, model_id=model_id, model=model_file)
    assert model_store.model_exists(domain, model_id)


def test_extra_metadata(model_store: ModelStore, model_file: str):
    extra_metadata = {"required_columns": ["col1", "col2"]}
    # Domain exists, but the model does not
    meta_data = model_store.upload(
        domain="test-domain",
        model_id="test-model-id-1",
        model=model_file,
        extra_metadata=extra_metadata,
    )

    # Extras are appended to the returned meta data
    assert meta_data["extra"] == extra_metadata

    # Extras are returned when querying for a model
    meta_data = model_store.get_model_info(
        domain="test-domain",
        model_id="test-model-id-1",
    )
    assert meta_data["extra"] == extra_metadata


def test_model_upload_doesnt_overwrite_existing_model(
    model_store: ModelStore, model_file: str
):
    domain = "test-domain"
    model_id = "test-model-id-1"
    model_store.upload(domain, model_id, model=model_file)

    with pytest.raises(ModelExistsException):
        model_store.upload(domain, model_id, model=model_file)


def test_model_upload_overwrites_deleted_model(
    model_store: ModelStore, model_file: str
):
    domain = "test-domain"
    model_id = "test-model-id-1"
    model_store.upload(domain, model_id, model=model_file)
    model_store.delete_model(domain, model_id, skip_prompt=True)
    model_ids = model_store.list_models(domain, ReservedModelStates.DELETED.value)
    assert model_id in model_ids

    model_store.upload(domain, model_id, model=model_file)
    model_ids = model_store.list_models(domain, ReservedModelStates.DELETED.value)
    assert model_id not in model_ids
