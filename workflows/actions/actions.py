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
    cli,
    models,
    storage,
)

MODEL_DOMAIN = "diabetes-boosting-demo"


def run_on_storage(model_store: ModelStore):
    """Runs a series of actions on `model_store` that don't require a model"""
    for func in storage.get_actions():
        print(f"🔍  Running {str(func)}")
        func(model_store, MODEL_DOMAIN)
    print("✅  Storage assertions passed")


def run_with_model(
    model_store: ModelStore, model: Any, extra_metadata: dict, extra_files: List[str]
):
    """Runs a series of actions on `model_store` using `model`"""
    meta_data = model_store.upload(
        domain=MODEL_DOMAIN,
        model=model,
        extra_metadata=extra_metadata,
        extras=extra_files,
    )
    model_id = meta_data["model"]["model_id"]
    print(f"✅  Finished uploading the model={model_id}")

    for func in models.get_actions():
        print(f"🔍  Running {str(func)}")
        func(model_store, MODEL_DOMAIN, meta_data)
    print("✅  Model assertions passed")


def run_cli_commands(model_path: str):
    """Runs a series of CLI commands"""
    model_id = cli.assert_upload_runs(MODEL_DOMAIN, model_path)
    cli.assert_download_runs(MODEL_DOMAIN, model_id)
