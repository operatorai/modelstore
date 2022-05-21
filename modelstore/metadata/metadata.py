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
from datetime import datetime

import modelstore
from modelstore.metadata import dependencies, revision, runtime


def generate_for_model(
    domain: str,
    model_id: str,
    model_info: dict,
    model_params: dict = None,
    model_data: dict = None,
) -> dict:
    """ Generates the meta-data dict for a model """
    metadata = {
        "domain": domain,
        "model_id": model_id,
        "model_type": _remove_nones(model_info),
    }
    if model_params is not None:
        metadata["parameters"] = _remove_nones(model_params)
    if model_data is not None:
        metadata["data"] = _remove_nones(model_data)
    return metadata


def generate_for_code(deps_list: dict) -> dict:
    """ Generates the meta data for the code being run to create the model """
    versioned_deps = dependencies.get_dependency_versions(deps_list)
    metadata = {
        "runtime": f"python:{runtime.get_python_version()}",
        "user": runtime.get_user(),
        "created": datetime.now().strftime("%Y/%m/%d/%H:%M:%S"),
        "dependencies": _remove_nones(versioned_deps),
    }
    git_meta = revision.git_meta()
    if git_meta is not None:
        metadata["git"] = git_meta
    return metadata


def generate(
    model_meta: dict,
    storage_meta: dict,
    code_meta: dict,
) -> dict:
    """ Combines all of the meta data into a single dictionary """
    return {
        "model": model_meta,
        "storage": storage_meta,
        "code": code_meta,
        "modelstore": modelstore.__version__,
    }


def _remove_nones(values) -> dict:
    return {k: v for k, v in values.items() if v is not None}
