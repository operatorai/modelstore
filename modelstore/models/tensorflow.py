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
from packaging import version

from modelstore.metadata import metadata
from modelstore.models.common import save_json
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

MODEL_CONFIG = "model_config.json"
MODEL_CHECKPOINT = "checkpoint"
MODEL_DIRECTORY = "model"

# Tensorflow >= 2.16.0 introduces Keras 3.0 by default
TF_VERSION_CHECK = "2.16.0"
KERAS_3_WEIGHTS_FILE_EXTENSION = "weights.h5"
KERAS_3_MODEL_FILE_EXTENSION = "keras"


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
            "tensorflow",
        ]
    
    def optional_dependencies(self) -> list:
        return super().optional_dependencies() + ["scipy"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        try:
            from transformers import TFPreTrainedModel

            # transformer tensorflow models are an instance of keras.Model
            # but we want to upload them using the transformers manager
            # we therefore check specifically for this case
            if isinstance(kwargs.get("model"), TFPreTrainedModel):
                return False
        except ImportError:
            # transformers does not exist
            pass
        except RuntimeError:
            # transformers cannot import tensorflow stuff
            pass
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

    def get_params(self, **kwargs) -> dict:
        return kwargs["model"].optimizer.get_config()

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)

        # pylint: disable=import-outside-toplevel
        from tensorflow import keras

        model_path = _model_file_path(model_path)
        # Alternative model storage file format for Keras 3.0
        if _is_tensorflow_using_keras3_api():
            model_path = f"{model_path}.{KERAS_3_MODEL_FILE_EXTENSION}"

        return keras.models.load_model(model_path)


def _is_tensorflow_using_keras3_api():
    # pylint: disable=import-outside-toplevel
    import tensorflow as tf

    # Tensorflow >= 2.16.0 requires keras >= 3.0.0
    # This requires us to save models differently
    # https://keras.io/guides/migrating_to_keras_3/
    # Here we check the version of tensorflow, so we can conditionally support both old and new behaviours
    tf_version = tf.__version__
    return version.parse(tf_version) >= version.parse(TF_VERSION_CHECK)


def _model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_DIRECTORY)


def _save_weights(tmp_dir: str, model: "keras.Model") -> str:
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
    weights_path = os.path.join(tmp_dir, MODEL_CHECKPOINT)
    if _is_tensorflow_using_keras3_api():
        weights_path = f"{weights_path}.{KERAS_3_WEIGHTS_FILE_EXTENSION}"
    model.save_weights(weights_path)
    return weights_path


def _save_model(tmp_dir: str, model: "keras.Model") -> str:
    model_path = _model_file_path(tmp_dir)
    os.makedirs(model_path)
    if _is_tensorflow_using_keras3_api():
        model_path = f"{model_path}.{KERAS_3_MODEL_FILE_EXTENSION}"
    model.save(model_path)
    return model_path
