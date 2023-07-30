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
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional

from modelstore.metadata import metadata
from modelstore.metadata.code.dependencies import module_exists


class CloudStorage(ABC):

    """
    Abstract class capturing a type of cloud storage
    (e.g., Google Cloud, AWS, other)
    """

    __metaclass__ = ABCMeta

    def __init__(self, required_deps: list):
        for dep in required_deps:
            if not module_exists(dep):
                raise ModuleNotFoundError(f"{dep} not installed.")

    @abstractmethod
    def validate(self) -> bool:
        """Runs any required validation steps - e.g.,
        checking that a cloud bucket exists"""
        raise NotImplementedError()

    @abstractmethod
    def upload(self, domain: str, model_id: str, local_path: str) -> metadata.Storage:
        """Uploads an archive to this type of storage
        :param extras can be a path to a file or list of files
        if these are specified, those files are upload
        to the same storage prefix too."""
        raise NotImplementedError()

    @abstractmethod
    def set_meta_data(self, domain: str, model_id: str, meta_data: metadata.Summary):
        """Stores a model's meta data"""
        raise NotImplementedError()

    @abstractmethod
    def get_meta_data(self, domain: str, model_id: str) -> metadata.Summary:
        """Returns a model's meta data"""
        raise NotImplementedError()

    @abstractmethod
    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        raise NotImplementedError()

    @abstractmethod
    def delete_model(
        self,
        domain: str,
        model_id: str,
        meta_data: metadata.Summary,
        skip_prompt: bool = False,
    ):
        """Deletes a model artifact from storage."""
        raise NotImplementedError()

    @abstractmethod
    def get_domain(self, domain: str) -> dict:
        """Returns information about the domain"""
        raise NotImplementedError()

    @abstractmethod
    def list_domains(self) -> list:
        """Returns a list of all the existing model domains"""
        raise NotImplementedError()

    @abstractmethod
    def list_models(self, domain: str, state_name: Optional[str] = None) -> list:
        """Returns a list of a model's versions"""
        raise NotImplementedError()

    @abstractmethod
    def list_model_states(self) -> list:
        """Lists the model states that have been created"""
        raise NotImplementedError()

    @abstractmethod
    def create_model_state(self, state_name: str):
        """Creates a state label that can be used to tag models"""
        raise NotImplementedError()

    @abstractmethod
    def delete_model_state(self, state_name: str, skip_prompt: bool):
        """Deletes a model state"""
        raise NotImplementedError()

    @abstractmethod
    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """Sets the given model ID to the given state"""
        raise NotImplementedError()

    @abstractmethod
    def unset_model_state(
        self,
        domain: str,
        model_id: str,
        state_name: str,
        modifying_reserved: bool = False,
    ):
        """Removes the given model ID from the set that are in the state_name path"""
        raise NotImplementedError()

    @abstractmethod
    def get_model_states(self, domain: str, model_id: str) -> List[str]:
        """Retrieves the states that have been set for a given model"""
        raise NotImplementedError()
