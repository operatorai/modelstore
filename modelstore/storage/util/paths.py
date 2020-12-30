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

_ROOT = "operatorai-model-store"


def get_archive_path(domain: str, local_path: str) -> str:
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
    prefix = datetime.now().strftime("%Y/%m/%d/%H:%M:%S")
    return os.path.join(_ROOT, domain, prefix, file_name)


def get_versions_path(domain: str) -> str:
    """Creates a path where a meta-data file about a model is stored.
    I.e.: :code:`operatorai-model-store/<domain>/versions/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    return os.path.join(_ROOT, domain, "versions")


def get_metadata_path(domain: str, model_id: str) -> str:
    """Creates a path where a meta-data file about a model is stored.
    I.e.: :code:`operatorai-model-store/<domain>/versions/<model-id>.json`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
        
        model_id (str): A UUID4 string that identifies this specific
        model.
    """
    versions_path = get_versions_path(domain)
    return os.path.join(versions_path, f"{model_id}.json")


def get_domains_path() -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`operatorai-model-store/domains/`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    return os.path.join(_ROOT, "domains")


def get_domain_path(domain: str) -> str:
    """Creates a path where meta-data about the latest trained model
    is stored, i.e.: :code:`operatorai-model-store/domains/<domain>.json`

    Args:
        domain (str): A group of models that are trained for the
        same end-use are given the same domain.
    """
    domains_path = get_domains_path()
    return os.path.join(domains_path, f"{domain}.json")
