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

import numpy as np
from modelstore.meta import metadata
from modelstore.meta.dependencies import save_dependencies, save_model_info
from modelstore.storage.storage import CloudStorage


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
        """Returns a list of dependencies that
        must be pip installed for this ModelManager to work"""
        raise NotImplementedError()

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
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

    @abstractmethod
    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        """
        raise NotImplementedError()

    @abstractmethod
    def _required_kwargs(self) -> list:
        """
        The kwargs that must be set when calling
        create_archive()
        """
        raise NotImplementedError()

    def _validate_kwargs(self, **kwargs):
        """Ensures that the required kwargs are set"""
        for arg in self._required_kwargs():
            if arg not in kwargs:
                raise TypeError(f"Please specify {arg}=<value>")

    @abstractmethod
    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        raise NotImplementedError()

    @abstractmethod
    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        raise NotImplementedError()

    def _collect_files(self, tmp_dir: str, **kwargs) -> list:
        """Returns a list of files created in tmp_dir that will form
        part of the artifacts archive.
        """
        file_paths = [
            save_dependencies(tmp_dir, self._get_dependencies()),
            save_model_info(tmp_dir, self._model_info(**kwargs)),
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

        # Meta-data about the model
        model_id = str(uuid.uuid4())
        model_meta = metadata.generate_for_model(
            domain=domain,
            model_id=model_id,
            model_info=self._model_info(**kwargs),
            model_params=_format_numpy(self._get_params(**kwargs)),
            model_data=self._model_data(**kwargs),
        )

        # Meta-data about the code
        code_meta = metadata.generate_for_code(self._get_dependencies())

        # Meta-data about the model location
        archive_path = self._create_archive(**kwargs)
        storage_meta = self.storage.upload(domain, model_id, archive_path)

        # Generate the combined meta-data and add it to the store
        meta_data = metadata.generate(model_meta, storage_meta, code_meta)
        self.storage.set_meta_data(domain, model_id, meta_data)
        os.remove(archive_path)
        return meta_data


def _format_numpy(model_params: dict) -> dict:
    for k, v in model_params.items():
        if isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
            model_params[k] = float(v)
        if isinstance(v, np.ndarray):
            model_params[k] = v.tolist()
        if isinstance(v, dict):
            model_params[k] = _format_numpy(v)
    return model_params


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
