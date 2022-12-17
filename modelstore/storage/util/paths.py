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


def get_root_path(root_dir: str) -> str:
    """Returns the root location of the model registry"""
    return os.path.join(
        root_dir,
        MODELSTORE_ROOT_PREFIX,
    )


def get_archive_path(root_dir: str, domain: str, model_id: str, local_path: str) -> str:
    """Creates a path where a model archive will be stored.
    I.e.: :code:`<root>/<domain>/<date-based prefix>/<model-id>/<file-name>`

    Args:
        root_dir (str): The root directory/prefix for this type
        of storage

        domain (str): A group of models that are trained for the
        same end-use are given the same domain.

        model_id (str): The specific identifier for this model

        local_path (str): The path to the local file that is
        to be updated.
    """
    return os.path.join(
        get_root_path(root_dir),
        domain,
        # Warning! Mac OS translates ":" in paths to "/"
        datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
        model_id,
        os.path.split(local_path)[1],
    )


def get_model_versions_path(
    root_dir: str, domain: str, state_name: Optional[str] = None
) -> str:
    """Creates a path where meta-data files about models are stored.
    I.e.: :code:`<root>/<domain>/versions/[state]/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
        state_name (str): A model's state tag (e.g. "prod" or "archived")
    """
    versions = os.path.join(
        get_root_path(root_dir),
        domain,
        "versions",
    )
    if state_name is not None:
        return os.path.join(versions, state_name)
    return versions


def get_model_version_path(
    root_dir: str, domain: str, model_id: str, state_name: Optional[str] = None
) -> str:
    """Creates a path where a meta-data file about a model is stored.
    I.e.: :code:`<root>/<domain>/versions/[state]/<model-id>.json`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.

        model_id (str): A UUID4 string that identifies this specific
        model.
    """
    return os.path.join(
        get_model_versions_path(root_dir, domain, state_name),
        f"{model_id}.json",
    )


def get_domains_path(root_dir: str) -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`<root>/domains/`
    """
    return os.path.join(
        get_root_path(root_dir),
        "domains",
    )


def get_domain_path(root_dir: str, domain: str) -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`operatorai-model-store/domains/<domain>.json`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    return os.path.join(
        get_domains_path(root_dir),
        f"{domain}.json",
    )


def get_model_states_path(root_dir: str) -> str:
    """Creates a path where meta-data about the model states are
    stored, i.e.: :code:`operatorai-model-store/model_states/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    return os.path.join(
        get_root_path(root_dir),
        "model_states",
    )


def get_model_state_path(root_dir: str, state_name: str) -> str:
    """Creates a path where meta-data about a model states is
    stored, i.e.: :code:`operatorai-model-store/model_states/<state_name>.json`

    Args:
        state_name (str): The name of the model state (e.g., "prod").
    """
    return os.path.join(
        get_model_states_path(root_dir),
        f"{state_name}.json",
    )
