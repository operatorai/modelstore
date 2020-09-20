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

from modelstore.models.modelmanager import ModelManager
from modelstore.utils.log import logger


class MissingDepManager(ModelManager):

    """
        MissingDepManager is used when a dependency is not
        installed; it overrides the ModelManager functionality
        and gives the user informative error messages
        """

    def __init__(self, library: str):
        super().__init__()
        self.library = library

    @classmethod
    def required_dependencies(cls) -> list:
        return []

    def _get_functions(self, **kwargs) -> list:
        return []

    def _required_kwargs(self) -> list:
        return []

    def create_archive(self, **kwargs) -> str:
        logger.error("Error: %s is not installed", self.library)
        logger.error("Please install it and try again")
        raise ModuleNotFoundError("Library is not installed")
