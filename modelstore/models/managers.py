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
from typing import Iterator

from modelstore.clouds.storage import CloudStorage
from modelstore.meta.dependencies import module_exists
from modelstore.models.catboost import CatBoostManager
from modelstore.models.keras import KerasManager
from modelstore.models.missingmanager import MissingDepManager
from modelstore.models.modelmanager import ModelManager
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.tensorflow import TensorflowManager
from modelstore.models.transformers import TransformersManager
from modelstore.models.xgboost import XGBoostManager

ML_LIBRARIES = {
    "catboost": CatBoostManager,
    "keras": KerasManager,
    "pytorch": PyTorchManager,
    "sklearn": SKLearnManager,
    "tensorflow": TensorflowManager,
    "transformers": TransformersManager,
    "xgboost": XGBoostManager,
}


def iter_libraries(storage: CloudStorage = None) -> Iterator[ModelManager]:
    for library, mngr in ML_LIBRARIES.items():
        if all(module_exists(x) for x in mngr.required_dependencies()):
            yield library, mngr(storage)
        else:
            yield library, MissingDepManager(library, storage)
