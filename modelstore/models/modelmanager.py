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
from abc import ABC, ABCMeta, abstractmethod

from modelstore.meta.dependencies import save_dependencies


class ModelManager(ABC):

    """
    ModelManager is an abstract class that we use to create an archive
    that contains all of the model-related artifacts that need to be stored
    and uploaded to the model store.
    """

    __metaclass__ = ABCMeta

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

    @abstractmethod
    def _required_kwargs(self) -> list:
        """
        The kwargs that must be set when calling
        create_archive()
        """

    def _validate_kwargs(self, **kwargs):
        """ Ensures that the required kwargs are set
        """
        for arg in self._required_kwargs():
            if arg not in kwargs:
                raise TypeError(f"Please specify {arg}=<value>")

    def _collect_files(self, tmp_dir: str, **kwargs) -> list:
        """ Returns a list of files created in tmp_dir that will form
        part of the artifacts archive.
        """
        file_paths = [save_dependencies(tmp_dir, self._get_dependencies())]
        for func in self._get_functions(**kwargs):
            rsp = func(tmp_dir)
            if isinstance(rsp, list):
                # Multiple files saved by this function call
                file_paths += rsp
            else:
                # Single file saved by this function call
                file_paths.append(rsp)
        return file_paths

    def create_archive(self, **kwargs) -> str:
        """
        Creates the `artifacts.tar.gz` archive which contains
        all of the files of the model
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

            # Moves the file into the current working directory
            target = os.path.join(os.getcwd(), archive_name)
            shutil.move(result, target)
        return target
