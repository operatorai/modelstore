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

from modelstore.models.common import load_joblib, save_joblib
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

MODEL_JOBLIB = "model.joblib"
MODEL_PARAMS_FILE = "model.pt"


class SkorchManager(ModelManager):

    """
    Model persistence for skorch models:
    https://skorch.readthedocs.io/en/stable/user/save_load.html
    """

    NAME = "skorch"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["skorch", "torch"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["Cython", "joblib", "threadpoolctl"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from skorch import NeuralNet

        return isinstance(kwargs.get("model"), NeuralNet)

    def _model_data(self, **kwargs) -> dict:
        return {}

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not a skorch NeuralNet!")

        return [
            partial(save_joblib, model=kwargs["model"], file_name=MODEL_JOBLIB),
            partial(save_params, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters that are available
        """
        # @TODO future
        return {}

    def load(self, model_path: str, meta_data: dict) -> Any:
        # @Future: check if loading into same version of joblib
        # as was used for saving
        file_name = os.path.join(model_path, MODEL_JOBLIB)
        return load_joblib(file_name)


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_PARAMS_FILE)


def save_params(tmp_dir: str, model: "skorch.NeuralNet") -> str:
    """
    From the skorch docs
    save_params() does not store learned attributes on the net.
    E.g., skorch.classifier.NeuralNetClassifier remembers the classes it encountered
    during training in the classes_ attribute. This attribute will be missing after
    load_params(). Therefore, if you need it, you should pickle.dump() the whole net.
    """
    # pylint: disable=import-outside-toplevel
    file_path = _model_file_path(tmp_dir)
    logger.debug("Saving skorch model to %s", file_path)
    model.save_params(f_params=file_path)
    return file_path
