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
import shutil
import tarfile
import tempfile
import uuid
from abc import ABC, ABCMeta, abstractmethod

from modelstore.clouds.storage import CloudStorage
from modelstore.meta import metadata
from modelstore.meta.dependencies import save_dependencies, save_model_info


class ModelManager(ABC):

    """
    ModelManager is an abstract class that we use to create and upload archives
    that contains all of the model-related artifacts.
    """

    __metaclass__ = ABCMeta

    def __init__(self, storage: CloudStorage = None):
        super().__init__()
        self.storage = storage

    @classmethod
    def required_dependencies(cls) -> list:
        """ Returns a list of dependencies that
        must be pip installed for this ModelManager to work"""
        raise NotImplementedError()

    @classmethod
    def optional_dependencies(cls) -> list:
        """ Returns a list of dependencies that, if installed
        are useful to log info about """
        return ["pip", "setuptools", "numpy", "scipy", "pandas"]

    @classmethod
    def _get_dependencies(cls) -> list:
        return cls.required_dependencies() + cls.optional_dependencies()

    @abstractmethod
    def _get_functions(self, **kwargs) -> list:
        """
        Returns a list of functions to call to save the model
        and any other required data
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """ Returns the name of this model type """
        raise NotImplementedError()

    @abstractmethod
    def _required_kwargs(self) -> list:
        """
        The kwargs that must be set when calling
        create_archive()
        """
        raise NotImplementedError()

    def _validate_kwargs(self, **kwargs):
        """ Ensures that the required kwargs are set
        """
        for arg in self._required_kwargs():
            if arg not in kwargs:
                raise TypeError(f"Please specify {arg}=<value>")

    @abstractmethod
    def model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        raise NotImplementedError()

    def _collect_files(self, tmp_dir: str, **kwargs) -> list:
        """ Returns a list of files created in tmp_dir that will form
        part of the artifacts archive.
        """
        file_paths = [
            save_dependencies(tmp_dir, self._get_dependencies()),
            save_model_info(tmp_dir, self.name(), self.model_info(**kwargs)),
        ]
        for func in self._get_functions(**kwargs):
            rsp = func(tmp_dir)
            if isinstance(rsp, list):
                # Multiple files saved by this function call
                file_paths += rsp
            else:
                # Single file saved by this function call
                file_paths.append(rsp)
        return file_paths

    def _create_archive(self, **kwargs) -> str:
        """
        Creates the `artifacts.tar.gz` archive which contains
        all of the files of the model

        Uploads the archive to storage. This function returns
        a dictionary of meta-data that is associated with this model,
        including an id.
        """
        self._validate_kwargs(**kwargs)
        archive_name = "artifacts.tar.gz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_paths = self._collect_files(tmp_dir, **kwargs)
            result = os.path.join(tmp_dir, archive_name)

            # Creates a tarfile and adds all of the files to it
            with tarfile.open(result, "w:gz") as tar:
                for file_path in file_paths:
                    file_name = os.path.split(file_path)[1]
                    tar.add(name=file_path, arcname=file_name)

            # Move the archive to the current working directory
            archive_path = os.path.join(os.getcwd(), archive_name)
            shutil.move(result, archive_path)
        return archive_path

    def upload(self, domain: str, **kwargs) -> dict:
        """
        Creates the `artifacts.tar.gz` archive which contains
        all of the files of the model

        Uploads the archive to storage. This function returns
        a dictionary of meta-data that is associated with this model,
        including an id.
        """
        _validate_domain(domain)
        self._validate_kwargs(**kwargs)

        model_id = str(uuid.uuid4())
        archive_path = self._create_archive(**kwargs)
        location = self.storage.upload(domain, archive_path)
        meta_data = metadata.generate(
            self.name(), model_id, domain, location, self._get_dependencies(),
        )
        self.storage.set_meta_data(domain, model_id, meta_data)
        os.remove(archive_path)
        return meta_data


def _validate_domain(domain: str):
    if len(domain) == 0:
        raise ValueError("Please provide a non-empty domain name.")
    if domain in [
        "versions",
        "domains",
        "modelstore",
        "operatorai-model-store",
    ]:
        raise ValueError("Please use a different domain name.")
