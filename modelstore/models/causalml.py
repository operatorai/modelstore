#    Copyright 2024 Neal Lathia
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

from modelstore.metadata import metadata
from modelstore.models.common import load_joblib, save_joblib
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

MODEL_FILE = "model.joblib"


class CausalMLManager(ModelManager):

    """
    Model persistence for Causal ML models:
    https://causalml.readthedocs.io/en/latest/index.html
    """

    NAME = "causalml"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["causalml"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["Cython", "joblib"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import causalml

        return any(
            [
                isinstance(kwargs.get("model"), causalml.inference.meta.base.BaseLearner),
                isinstance(kwargs.get("model"), causalml.propensity.PropensityModel),
            ]
        )

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not a Causal ML model!")

        return [partial(save_joblib, model=kwargs["model"], file_name=MODEL_FILE)]

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)

        # @Future: check if loading into same version of joblib
        # as was used for saving
        file_name = os.path.join(model_path, MODEL_FILE)
        return load_joblib(file_name)
