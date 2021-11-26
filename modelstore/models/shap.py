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

EXPLAINER_FILE = "explainer.joblib"


class ShapManager(ModelManager):

    """
    Model persistence for shap expainers
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("shap", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return ["shap", "joblib"]

    def _required_kwargs(self):
        return ["explainer"]

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the explainer type """
        return {
            "library": self.ml_library,
            "type": type(kwargs["explainer"]).__name__,
        }

    def matches_with(self, **kwargs) -> bool:
        # Exclude cases where the user wants to upload a model AND explainer
        # this use case will be dealt with in other model managers
        if len(kwargs) > 1:
            return False

        # pylint: disable=import-outside-toplevel
        from shap import Explainer

        return isinstance(kwargs.get("explainer"), Explainer)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Explainer is not a shap.Explainer!")

        return [
            partial(
                save_joblib, model=kwargs["explainer"], file_name=EXPLAINER_FILE
            ),
        ]

    def load(self, model_path: str, meta_data: dict) -> "shap.Explainer":
        explainer_path = _explainer_file_path(model_path)
        return load_joblib(explainer_path)


def _explainer_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, EXPLAINER_FILE)
