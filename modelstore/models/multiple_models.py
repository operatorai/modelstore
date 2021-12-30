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
from modelstore.storage.storage import CloudStorage


class MultipleModelsManager(ModelManager):

    """
    Persistence for multiple models, e.g. pairs of (model, explainer)
    that need to be saved together
    """

    def __init__(
        self,
        managers: List[ModelManager],
        storage: CloudStorage = None,
    ):
        super().__init__("model-and-explainer", storage)
        self.managers = managers

    def _required_kwargs(self):
        return self.model._required_kwargs() + self.explainer._required_kwargs()

    def matches_with(self, **kwargs) -> bool:
        return self.model.matches_with(
            **kwargs
        ) and self.explainer.matches_with(**kwargs)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not an onnx.ModelProto!")

        return self.model._get_functions(
            **kwargs
        ) + self.explainer._get_functions(**kwargs)

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        raise NotImplementedError()
