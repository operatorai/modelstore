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

from modelstore.models.common import save_json
from modelstore.models.modelmanager import ModelManager

MODEL_CONFIG = "model_config.json"
MODEL_DIRECTORY = "model"


class KerasManager(ModelManager):

    """
    Model persistence for Keras models:
    https://keras.io/api/models/model_saving_apis/
    https://keras.io/guides/serialization_and_saving/
    """

    @classmethod
    def required_dependencies(cls) -> list:
        return [
            "keras",
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

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"library": "keras"}

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

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
        """
        Returns a dictionary containing the optimizer
        parameters
        """
        return kwargs["model"].optimizer.get_config()


def _save_model(tmp_dir: str, model: "keras.Model") -> str:
    file_path = os.path.join(tmp_dir, MODEL_DIRECTORY)
    model.save(file_path)
    return file_path
