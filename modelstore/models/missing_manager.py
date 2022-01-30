#    Copyright 2020 Neal Lathia
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
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger


class MissingDepManager(ModelManager):

    """
    MissingDepManager is used when a dependency is not
    installed; it overrides the ModelManager functionality
    and gives the user informative error messages
    """

    NAME = "missing"

    def __init__(self, library: str, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)
        self.library = library

    def required_dependencies(cls) -> list:
        return []

    def matches_with(self, **kwargs) -> bool:
        return False

    def _get_functions(self, **kwargs) -> list:
        return []

    def _get_params(self, **kwargs) -> dict:
        return None

    def _required_kwargs(self) -> list:
        return []

    def _model_info(self, **kwargs) -> dict:
        return None

    def _model_data(self, **kwargs) -> dict:
        return None

    def upload(self, domain: str, **kwargs) -> str:
        logger.error("Error: %s is not installed", self.library)
        logger.error("Please install it and try again")
        raise ModuleNotFoundError(f"{self.library} is not installed")

    def load(self, model_path: str, meta_data: dict) -> Any:
        logger.error("Error: %s is not installed", self.library)
        logger.error("Please install it and try again")
        raise ModuleNotFoundError(f"{self.library} is not installed")
