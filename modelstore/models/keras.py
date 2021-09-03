#    Copyright 2021 Neal Lathia
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
from functools import partial
from typing import Any

from modelstore.models.common import save_json
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

MODEL_CONFIG = "model_config.json"
MODEL_DIRECTORY = "model"


class KerasManager(ModelManager):

    """
    Model persistence for Keras models:
    https://keras.io/api/models/model_saving_apis/
    https://keras.io/guides/serialization_and_saving/
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("keras", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return [
            "h5py",
            "numpy",
            "scipy",
            "tensorflow",
        ]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from tensorflow import keras

        return isinstance(kwargs.get("model"), keras.Model)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("model is not a keras.Model!")
        return [
            partial(_save_model, model=kwargs["model"]),
            partial(
                save_json,
                file_name=MODEL_CONFIG,
                data=kwargs["model"].to_json(),
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        return kwargs["model"].optimizer.get_config()

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from tensorflow import keras

        model_dir = _get_model_path(model_path)
        return keras.models.load_model(model_dir)


def _get_model_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_model(tmp_dir: str, model: "keras.Model") -> str:
    file_path = _get_model_path(tmp_dir)
    model.save(file_path)
    return file_path
