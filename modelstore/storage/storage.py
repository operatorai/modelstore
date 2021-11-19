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
from typing import Optional, Union

from modelstore.meta.dependencies import module_exists


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
    def upload(
        self,
        domain: str,
        local_path: str,
        extras: Optional[Union[str, list]] = None,
    ) -> dict:
        """Uploads an archive to this type of storage
        :param extras can be a path to a file or list of files
        if these are specified, those files are upload
        to the same storage prefix too."""
        raise NotImplementedError()

    @abstractmethod
    def set_meta_data(self, domain: str, model_id: str, meta_data: dict):
        """ Annotates a model with some given meta data """
        raise NotImplementedError()

    @abstractmethod
    def get_meta_data(self, domain: str, model_id: str) -> dict:
        """ Returns a model's meta data """
        raise NotImplementedError()

    @abstractmethod
    def download(self, local_path: str, domain: str, model_id: str = None):
        """Downloads an artifacts archive for a given (domain, model_id) pair.
        If no model_id is given, it defaults to the latest model in that
        domain"""
        raise NotImplementedError()

    @abstractmethod
    def list_domains(self) -> list:
        """ Returns a list of all the existing model domains """
        raise NotImplementedError()

    @abstractmethod
    def list_versions(
        self, domain: str, state_name: Optional[str] = None
    ) -> list:
        """ Returns a list of a model's versions """
        raise NotImplementedError()

    @abstractmethod
    def create_model_state(self, state_name: str):
        """ Creates a state label that can be used to tag models """
        raise NotImplementedError()

    @abstractmethod
    def set_model_state(self, domain: str, model_id: str, state_name: str):
        """ Sets the given model ID to the given state """
        raise NotImplementedError()

    @abstractmethod
    def unset_model_state(self, domain: str, model_id: str, state_name: str):
        """ Removes the given model ID from the set that are in the state_name path """
        raise NotImplementedError()
