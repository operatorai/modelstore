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
MODEL_CHECKPOINT = "checkpoint"
MODEL_DIRECTORY = "model"


class TensorflowManager(ModelManager):

    """
    Model persistence for tensoflow models:
    https://www.tensorflow.org/tutorials/keras/save_and_load
    """

    NAME = "tensorflow"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
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
        model = kwargs["model"]
        return [
            partial(_save_weights, model=model),
            partial(_save_model, model=model),
            partial(
                save_json,
                file_name=MODEL_CONFIG,
                data=kwargs["model"].to_json(),
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        return kwargs["model"].optimizer.get_config()

    def _is_same_library(self, meta_data: dict) -> bool:
        is_tf = super()._is_same_library(meta_data)
        is_keras = meta_data.get("library") == "keras"
        return is_tf or is_keras

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from tensorflow import keras

        model_path = _model_file_path(model_path)
        return keras.models.load_model(model_path)


def _model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_weights(tmp_dir: str, model: "keras.Model") -> str:
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
    weights_path = os.path.join(tmp_dir, MODEL_CHECKPOINT)
    model.save_weights(weights_path)
    return weights_path


def _save_model(tmp_dir: str, model: "keras.Model") -> str:
    model_path = _model_file_path(tmp_dir)
    os.makedirs(model_path)
    model.save(model_path)
    return model_path
