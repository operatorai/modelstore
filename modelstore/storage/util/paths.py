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
import os
from datetime import datetime
from typing import Optional

MODELSTORE_ROOT_PREFIX = "operatorai-model-store"

# @TODO move into blob_storage / override in local


def get_archive_path(root_dir: str, domain: str, local_path: str) -> str:
    """Creates a bucket prefix where a model archive will be stored.
    I.e.: :code:`operatorai-model-store/<domain>/<prefix>/<file-name>`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.

        local_path (str): The path to the local file that is
        to be updated.
    """
    file_name = os.path.split(local_path)[1]
    # Future: enable different types of prefixes
    # Warning! Mac OS translates ":" in paths to "/"
    prefix = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    return os.path.join(root_dir, MODELSTORE_ROOT_PREFIX, domain, prefix, file_name)


def get_models_path(
    root_dir: str, domain: str, state_name: Optional[str] = None
) -> str:
    """Creates a path where a meta-data file about a model is stored.
    I.e.: :code:`operatorai-model-store/<domain>/versions/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
        state_name (str): A model's state tag (e.g. "prod" or "archived")
    """
    if state_name is not None:
        return os.path.join(
            root_dir, MODELSTORE_ROOT_PREFIX, domain, "versions", state_name
        )
    return os.path.join(root_dir, MODELSTORE_ROOT_PREFIX, domain, "versions")


def get_domains_path(root_dir: str) -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`operatorai-model-store/domains/`
    """
    return os.path.join(root_dir, MODELSTORE_ROOT_PREFIX, "domains")


def get_domain_path(root_dir: str, domain: str) -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`operatorai-model-store/domains/<domain>.json`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    domains_path = get_domains_path(root_dir)
    return os.path.join(domains_path, f"{domain}.json")


def get_model_states_path(root_dir: str) -> str:
    """Creates a path where meta-data about the model states are
    stored, i.e.: :code:`operatorai-model-store/model_states/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    return os.path.join(root_dir, MODELSTORE_ROOT_PREFIX, "model_states")


def get_model_state_path(root_dir: str, state_name: str) -> str:
    """Creates a path where meta-data about a model states is
    stored, i.e.: :code:`operatorai-model-store/model_states/<state_name>.json`

    Args:
        state_name (str): The name of the model state (e.g., "prod").
    """
    model_states = get_model_states_path(root_dir)
    return os.path.join(model_states, f"{state_name}.json")
