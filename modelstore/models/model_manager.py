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
from typing import Any, Optional
import os
import shutil
import tarfile
import tempfile
import warnings
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from modelstore.metadata import metadata
from modelstore.metadata.code.runtime import get_python_version
from modelstore.storage.storage import CloudStorage


class ModelManager(ABC):

    """
    ModelManager is an abstract class that we use to create and upload archives
    that contains all of the model-related artifacts.
    """

    __metaclass__ = ABCMeta

    def __init__(self, ml_library: str, storage: CloudStorage = None):
        super().__init__()
        self.ml_library = ml_library
        self.storage = storage

    @abstractmethod
    def matches_with(self, **kwargs) -> bool:
        """Returns whether the kwargs being uploaded
        are an instance of the current manager"""
        raise NotImplementedError()

    def required_dependencies(self) -> list:
        """Returns a list of dependencies that
        must be pip installed for this ModelManager to work"""
        raise NotImplementedError()

    # pylint: disable=no-self-use
    def optional_dependencies(self) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        return ["pip", "setuptools", "numpy", "scipy", "pandas"]

    def get_dependencies(self) -> list:
        """Returns the full list of dependencies"""
        return self.required_dependencies() + self.optional_dependencies()

    @abstractmethod
    def _get_functions(self, **kwargs) -> list:
        """
        Returns a list of functions to call to save the model
        and any other required data
        """
        if not self.matches_with(**kwargs):
            raise TypeError(f"This model is not an {self.ml_library} model!")
        return []

    # pylint: disable=unused-argument
    def get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        """
        # Note: explainer params are currently omitted
        return {}

    @abstractmethod
    def _required_kwargs(self) -> list:
        """
        The kwargs that must be set when calling
        create_archive()
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        """Loads a model, stored in model_path, back into memory"""
        version = get_python_version()
        if meta_data is not None and meta_data.code is not None:
            if version != meta_data.code.runtime:
                train = f"model was trained with {meta_data.code.runtime}"
                load = f"but is being loaded with {version}"
                warnings.warn(
                    f"{train}, {load}",
                    category=RuntimeWarning,
                )

    def _validate_kwargs(self, **kwargs):
        """Ensures that the required kwargs are set"""
        for arg in self._required_kwargs():
            if arg not in kwargs:
                raise TypeError(f"Please specify {arg}=<value>")

    def model_info(self, **kwargs) -> metadata.ModelType:
        """Returns meta-data about the model's type"""
        class_name = type(kwargs["model"]).__name__ if "model" in kwargs else None
        return metadata.ModelType.generate(
            library=self.ml_library,
            class_name=class_name,
        )

    def model_data(self, **kwargs) -> Optional[metadata.Dataset]:
        """Returns meta data about the training data"""
        return metadata.Dataset.generate(
            kwargs.get("X_train"),
            kwargs.get("y_train"),
        )

    def _collect_files(self, tmp_dir: str, **kwargs) -> list:
        """Returns a list of files created in tmp_dir that will form
        part of the artifacts archive.
        """
        file_paths = [
            self.model_info(**kwargs).dumps(tmp_dir),
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

    def _collect_extras(self, **kwargs) -> set:
        if "extras" in kwargs:
            warnings.warn(
                "extras= is deprecated; use extra_files=",
                category=DeprecationWarning,
            )
        extras = kwargs.get("extra_files", kwargs.get("extras"))
        if extras is None:
            return []
        extra_paths = extras if isinstance(extras, list) else [extras]
        return set(f for f in extra_paths if os.path.isfile(f))

    def _create_archive(self, **kwargs) -> str:
        """
        Creates the `artifacts.tar.gz` archive which contains
        all of the files of the model
        """
        self._validate_kwargs(**kwargs)
        archive_name = "artifacts.tar.gz"
        archive_path = os.path.join(os.getcwd(), archive_name)
        if os.path.exists(archive_path):
            raise FileExistsError(f"modelstore cannot create an: {archive_name} file.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = os.path.join(tmp_dir, archive_name)
            with tarfile.open(result, "w:gz") as tar:
                # Add all of the model files to the top-level
                # of the archive
                for file_path in self._collect_files(tmp_dir, **kwargs):
                    file_name = os.path.split(file_path)[1]
                    tar.add(name=file_path, arcname=file_name)

                # Add any extra files to a sub-directory of
                # the archive
                for file_path in self._collect_extras(**kwargs):
                    file_name = os.path.split(file_path)[1]
                    tar.add(
                        name=file_path,
                        arcname=os.path.join("extras", file_name),
                    )

            # Move the archive to the current working directory
            shutil.move(result, archive_path)
        return archive_path

    def upload(
        self,
        domain: str,
        model_id: str,
        **kwargs,
    ) -> metadata.Summary:
        """
        Creates the `artifacts.tar.gz` archive which contains
        all of the files of the model and uploads the archive to storage.

        This function returns a dictionary of meta-data that is associated
        with this model, including an id.
        """
        _validate_domain(domain)
        self._validate_kwargs(**kwargs)

        # Create meta data about the model & code
        model_meta_data = metadata.Model.generate(
            domain=domain,
            model_id=model_id,
            model_type=self.model_info(**kwargs),
            parameters=_format_numpy(self.get_params(**kwargs)),
            data=self.model_data(**kwargs),
        )

        # Create the model archive and return
        # meta-data about its location
        archive_path = self._create_archive(**kwargs)

        # Upload the model archive and any additional extras
        storage_meta_data = self.storage.upload(domain, model_id, archive_path)
        meta_data = metadata.Summary.generate(
            code_meta_data=metadata.Code.generate(deps_list=self.get_dependencies()),
            model_meta_data=model_meta_data,
            storage_meta_data=storage_meta_data,
            extra_metadata=kwargs.get("extra_metadata"),
        )

        # Save the combined meta-data to storage
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
