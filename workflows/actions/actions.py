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
from typing import Any, List
from modelstore import ModelStore

# pylint: disable=import-error
from workflows.actions import (
    storage,
    models
)

MODEL_DOMAIN = "diabetes-boosting-demo"


def run_on_storage(model_store: ModelStore):
    """ Runs a series of actions on `model_store` that don't require a model """
    for func in storage.get_actions():
        func(model_store, MODEL_DOMAIN)


def run_with_model(model_store: ModelStore, model: Any,
    extra_metadata: dict, extra_files: List[str]):
    """ Runs a series of actions on `model_store` using `model` """
    meta_data = model_store.upload(
        domain=MODEL_DOMAIN,
        model=model,
        extra_metadata=extra_metadata,
        extras=extra_files,
    )
    model_id = meta_data["model"]["model_id"]
    print(f"✅  Finished uploading the model={model_id}")

    for func in models.get_actions():
        func(model_store, MODEL_DOMAIN, meta_data)

       # This demo downloads models; we'll store them into a temporary
    # directory
    #tmp_dir = tempfile.mkdtemp()

    # # List all of the domains
    # demos.list_domains(modelstore)

    # # Get the information about a specific domain
    # meta_data = modelstore.get_domain(model_domain)
    # print(f"✅  Meta data about domain={model_domain}: {meta_data}")

    # # List the models in the diabest-boosting-demo domain
    # demos.list_models_in_domain(modelstore, model_domain)

    # # Download the models back
    # demos.download_model(modelstore, model_domain, model_id)

    # # Download the latest model
    # demos.download_latest_model(modelstore, model_domain)

    # # Load models back into memory
    # demos.load_model(modelstore, model_domain, model_id)

    # # Create a couple of new model states
    # state_names = ["staging", "production"]
    # for state_name in state_names:
    #     print(f"✅  Creating model state={state_name}:")
    #     modelstore.create_model_state(state_name)

    # # List them back
    # demos.list_model_states(modelstore, state_names)

    # # Set the model to the production state
    # demos.set_model_state(modelstore, model_domain, model_id, "production")

    # # List the models that are in production - the new model is there
    # demos.list_models_in_domain_with_state(
    #     modelstore, model_domain, "production", model_id
    # )

    # # Remove a state from a model
    # demos.remove_model_state(modelstore, model_domain, model_id, "production")

    # # Set the model to a different state
    # demos.set_model_state(modelstore, model_domain, model_id, "staging")
    # demos.list_models_in_domain_with_state(
    #     modelstore, model_domain, "staging", model_id
    # )

    # # Delete the model!
    # demos.delete_model(modelstore, model_domain, model_id)
    # for state_name in state_names:
    #     # The model no longer appears in the listing by state
    #     model_ids = demos.list_models_in_domain_with_state(
    #         modelstore, model_domain, state_name
    #     )
    #     assert model_id not in model_ids

    # # The model no longer appears when listing all of them
    # model_ids = demos.list_models_in_domain(modelstore, model_domain)
    # assert model_id not in model_ids

    # # You get an informative exception message
    # try:
    #     meta_data = modelstore.get_model_info(model_domain, model_id)
    # except ModelDeletedException:
    #     print(
    #         "✅  Modelstore raises a ModelDeletedException if a model has been deleted"
    #     )

    # print("✅  Demo finished!")
