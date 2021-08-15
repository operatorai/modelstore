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
from functools import partial

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage


class ModelFileManager(ModelManager):

    """
    Upload model files that have already been persisted to disk
    to the model store
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("persisted_files", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        # The model manager does not depend on anything
        return []

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        return [
            "pip",
            "setuptools",
            "pickle",
            "joblib",
        ]

    def matches_with(self, **kwargs) -> bool:
        if "model" in kwargs:
            # model is a path to a file
            return os.path.exists(kwargs["model"])
        return False

    def _get_functions(self, **kwargs) -> list:
        """
        Return a function that when called copies
        the model file to the tmp_dir
        """
        return [
            partial(copy_file, source=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        return {}

    def _required_kwargs(self) -> list:
        return ["model"]

    def _model_info(self, **kwargs) -> dict:
        return {"library": self.ml_library}


def copy_file(source, tmp_dir) -> str:
    destination = os.path.join(
        tmp_dir,
        os.path.split(source)[1],
    )
    shutil.copy2(source, destination)
    return destination
