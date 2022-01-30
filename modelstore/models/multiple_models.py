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
from typing import Any, List

from modelstore.models.model_manager import ModelManager
from modelstore.models.managers import get_manager
from modelstore.storage.storage import CloudStorage


class MultipleModelsManager(ModelManager):

    """
    Persistence for multiple models
    E.g. pairs of (model, explainer) that need to be saved together
    """

    NAME = "multiple-models"

    def __init__(
        self,
        managers: List[ModelManager],
        storage: CloudStorage = None,
    ):
        super().__init__(self.NAME, storage)
        self.managers = managers

    def _required_kwargs(self) -> list:
        requirements = []
        for manager in self.managers:
            requirements += manager._required_kwargs()
        return list(set(requirements))

    def required_dependencies(self) -> list:
        dependencies = []
        for manager in self.managers:
            dependencies += manager.required_dependencies()
        return list(set(dependencies))

    def matches_with(self, **kwargs) -> bool:
        for manager in self.managers:
            if not manager.matches_with(**kwargs):
                return False
        return True

    def _get_functions(self, **kwargs) -> list:
        functions = []
        for manager in self.managers:
            functions += manager._get_functions(**kwargs)
        return functions

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {
            "library": self.ml_library,
            "models": [manager._model_info(**kwargs) for manager in self.managers],
        }

    def _get_params(self, **kwargs) -> dict:
        return {
            manager.ml_library: manager._get_params(**kwargs)
            for manager in self.managers
        }

    def load(self, model_path: str, meta_data: dict) -> Any:
        models = {}
        for model_type in meta_data["model"]["model_type"]["models"]:
            ml_library = model_type["library"]
            manager = get_manager(ml_library, self.storage)
            models[ml_library] = manager.load(model_path, meta_data)
        return models
