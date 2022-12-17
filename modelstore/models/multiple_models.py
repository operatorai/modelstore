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

from modelstore.metadata import metadata
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
            # pylint: disable=protected-access
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
            # pylint: disable=protected-access
            functions += manager._get_functions(**kwargs)
        return functions

    def model_info(self, **kwargs) -> metadata.ModelType:
        """Returns meta-data about the model's type"""
        return metadata.ModelType.generate(
            library=self.ml_library,
            models=[m.model_info(**kwargs) for m in self.managers],
        )

    def get_params(self, **kwargs) -> dict:
        return {
            # pylint: disable=protected-access
            manager.ml_library: manager.get_params(**kwargs)
            for manager in self.managers
        }

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        models = {}
        for model in meta_data.model_type().models:
            manager = get_manager(model.library, self.storage)
            models[model.library] = manager.load(model_path, meta_data)
        return models
