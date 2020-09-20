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
from functools import partial

from modelstore.meta.dependencies import module_exists
from modelstore.models.catboost import CatBoostManager
from modelstore.models.keras import KerasManager
from modelstore.models.missingmanager import MissingDepManager
from modelstore.models.modelmanager import ModelManager
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.xgboost import XGBoostManager

ML_LIBRARIES = {
    "catboost": CatBoostManager,
    "pytorch": PyTorchManager,  # Adding twice as this is a common typo
    "torch": PyTorchManager,
    "sklearn": SKLearnManager,
    "xgboost": XGBoostManager,
    "pytorch": PyTorchManager,
    "keras": KerasManager,
}


def get_manager(library: str) -> ModelManager:
    mngr = ML_LIBRARIES.get(library, MissingDepManager)
    if all(module_exists(x) for x in mngr.required_dependencies()):
        return mngr
    return partial(MissingDepManager, library=library)
