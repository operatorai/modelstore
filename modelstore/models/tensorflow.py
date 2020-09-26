#    Copyright 2020 Neal Lathia

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from functools import partial

from modelstore.models.modelmanager import ModelManager


class TensorflowManager(ModelManager):

    """
    Model persistence for tensoflow models:
    https://www.tensorflow.org/tutorials/keras/save_and_load
    """

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

    def _get_functions(self, **kwargs) -> list:
        model = kwargs["model"]
        return [
            partial(_save_weights, model=model),
            partial(_save_model, model=model),
        ]


def _save_weights(tmp_dir: str, model: "keras.Model") -> str:
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
    weights_path = os.path.join(tmp_dir, "checkpoint")
    model.save_weights(weights_path)
    return weights_path


def _save_model(tmp_dir: str, model) -> str:
    model_path = os.path.join(tmp_dir, "model")
    os.makedirs(model_path)
    model.save(model_path)
    return model_path