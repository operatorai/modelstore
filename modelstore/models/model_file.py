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
from pathlib import PosixPath
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage


class ModelFileManager(ModelManager):

    """
    Upload model files that have already been persisted to disk
    to the model store. This approach is intended for usage with
    any ML framework that is not (yet) supported by modelstore
    """

    NAME = "model_file"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        # The model manager does not depend on anything
        return []

    def optional_dependencies(self) -> list:
        return [
            "pip",
            "setuptools",
            "pickle",
            "joblib",
        ]

    def matches_with(self, **kwargs) -> bool:
        if "model" in kwargs:
            # model is a path to a file
            model_path = kwargs["model"]
            if isinstance(model_path, PosixPath) or isinstance(model_path, str):
                if os.path.isdir(model_path):
                    # @Future - support adding directories
                    return False
                return os.path.exists(kwargs["model"])
        return False

    def _get_functions(self, **kwargs) -> list:
        """
        Return a function that when called copies
        the model file to the tmp_dir
        """
        if not self.matches_with(**kwargs):
            raise TypeError("model is not a path to a file!")
        return [
            partial(copy_file, source=kwargs["model"]),
        ]

    def _required_kwargs(self) -> list:
        return ["model"]

    def _model_info(self, **kwargs) -> dict:
        return {"library": self.ml_library}

    def load(self, model_path: str, meta_data: dict) -> Any:
        """
        If a model was saved to disk and uploaded with this manager,
        then we can't load it back into memory because we don't know
        how to!
        """
        raise ValueError("cannot load model_file models into memory")


def copy_file(tmp_dir, source) -> str:
    destination = os.path.join(
        tmp_dir,
        os.path.split(source)[1],
    )
    shutil.copy2(str(source), str(destination))
    return destination
