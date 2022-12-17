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
from typing import List, Callable
import tempfile
import os

from modelstore import ModelStore
from modelstore.utils import exceptions


def assert_get_missing_model_raises(model_store: ModelStore, domain: str, _: dict):
    """Calling get_model_info() for a missing model raise an exception"""
    try:
        _ = model_store.get_model_info(domain, "missing-model")
    except exceptions.ModelNotFoundException:
        print("✅  Modelstore raises a ModelNotFoundException if it can't find a model")
        return
    raise AssertionError("failed to raise ModelNotFoundException")


def assert_list_domains(model_store: ModelStore, domain: str, _: dict):
    """The result of listing all domains contains the `domain` value"""
    domains = model_store.list_domains()
    assert domain in domains
    print(f"✅  list_domains() contains domain={domain}")


def assert_get_domain(model_store: ModelStore, domain: str, _: dict):
    """The result of listing all domains contains the `domain` value"""
    meta_data = model_store.get_domain(domain)
    assert isinstance(meta_data, dict)
    # @TODO better assertions
    print("✅  get_domain() returns a domain's information")


def assert_list_models(model_store: ModelStore, model_domain: str, meta_data: dict):
    """The result of listing models contains the recently uploaded model"""
    model_id = meta_data["model"]["model_id"]
    model_ids = model_store.list_models(domain=model_domain)
    assert model_id in model_ids
    print(f"✅  list_models() contains model={model_id}")


def assert_get_model(model_store: ModelStore, model_domain: str, meta_data: dict):
    """Can retrieve an uploaded model's meta data"""
    model_id = meta_data["model"]["model_id"]
    stored_meta_data = model_store.get_model_info(model_domain, model_id)
    assert meta_data == stored_meta_data
    print("✅  Retrieved the model's meta data")


def assert_load_model(model_store: ModelStore, model_domain: str, meta_data: dict):
    """Model store can load a model back into memory"""
    model_id = meta_data["model"]["model_id"]
    model = model_store.load(model_domain, model_id)
    print(f"✅  Loaded model={model_id} into memory with type={type(model)}")


def assert_download_model(model_store: ModelStore, model_domain: str, meta_data: dict):
    """A model archive can be downloaded"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_id = meta_data["model"]["model_id"]
        model_path = model_store.download(tmp_dir, model_domain, model_id)
        assert os.path.exists(model_path)
        # @TODO assert on extra_files
    print(f"✅  Downloaded model={model_id}")


def assert_update_model_state(
    model_store: ModelStore, model_domain: str, meta_data: dict
):
    """A model's state can be set and unset"""
    model_id = meta_data["model"]["model_id"]
    state_name = "production"
    model_store.set_model_state(model_domain, model_id, state_name)
    model_ids = model_store.list_models(model_domain, state_name=state_name)
    assert model_id in model_ids

    model_store.remove_model_state(model_domain, model_id, state_name)
    model_ids = model_store.list_models(model_domain, state_name=state_name)
    assert model_id not in model_ids


def assert_delete_model(model_store: ModelStore, model_domain: str, meta_data: dict):
    """Deleting a model removes it from the store"""
    model_id = meta_data["model"]["model_id"]
    state_name = "production"
    model_store.set_model_state(model_domain, model_id, state_name)
    model_store.delete_model(model_domain, model_id, skip_prompt=True)
    try:
        _ = model_store.get_model_info(model_domain, model_id)
    except exceptions.ModelDeletedException:
        print("✅  Raises a ModelDeletedException after deleting a model")
        model_ids = model_store.list_models(model_domain, state_name=state_name)
        assert model_id not in model_ids
        print("✅  Deleted models are removed from states")
        return
    raise AssertionError("ModelDeletedException not raised after delete_model()")


def get_actions() -> List[Callable]:
    """Returns the set of actions that can be run on a model_store
    after a model has been uploaded"""
    return [
        assert_get_missing_model_raises,
        assert_list_domains,
        assert_get_domain,
        assert_list_models,
        assert_get_model,
        assert_load_model,
        assert_download_model,
        assert_update_model_state,
        assert_delete_model,  # Note: this action must be last
    ]
