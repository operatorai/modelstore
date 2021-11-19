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

from modelstore.meta.dependencies import module_exists
from modelstore.models.annoy import AnnoyManager
from modelstore.models.catboost import CatBoostManager
from modelstore.models.fastai import FastAIManager
from modelstore.models.gensim import GensimManager
from modelstore.models.keras import KerasManager
from modelstore.models.lightgbm import LightGbmManager
from modelstore.models.missing_manager import MissingDepManager
from modelstore.models.model_file import ModelFileManager
from modelstore.models.model_manager import ModelManager
from modelstore.models.mxnet import MxnetManager
from modelstore.models.onnx import OnnxManager
from modelstore.models.prophet import ProphetManager
from modelstore.models.pytorch import PyTorchManager
from modelstore.models.pytorch_lightning import PyTorchLightningManager
from modelstore.models.shap import ShapManager
from modelstore.models.sklearn import SKLearnManager
from modelstore.models.skorch import SkorchManager
from modelstore.models.tensorflow import TensorflowManager
from modelstore.models.transformers import TransformersManager
from modelstore.models.xgboost import XGBoostManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

ML_LIBRARIES = {
    "annoy": AnnoyManager,
    "catboost": CatBoostManager,
    "fastai": FastAIManager,
    "file": ModelFileManager,
    "gensim": GensimManager,
    "keras": KerasManager,
    "lightgbm": LightGbmManager,
    "mxnet": MxnetManager,
    "onnx": OnnxManager,
    "prophet": ProphetManager,
    "pytorch": PyTorchManager,
    "pytorch_lightning": PyTorchLightningManager,
    "sklearn": SKLearnManager,
    "skorch": SkorchManager,
    "tensorflow": TensorflowManager,
    "transformers": TransformersManager,
    "xgboost": XGBoostManager,
}

EXPLAINER_LIBRARIES = {
    "shap": ShapManager,
}


def _iter_managers(
    managers: dict, storage: CloudStorage = None
) -> Iterator[ModelManager]:
    """Iterates of a dict of ModelManagers and yields
    the ones that are available in the current environment,
    based on checking for dependencies.
    """
    for library, mngr in managers.items():
        if all(module_exists(x) for x in mngr.required_dependencies()):
            yield library, mngr(storage)
        else:
            logger.debug("Skipping: %s, not installed.", library)
            yield library, MissingDepManager(library, storage)


def iter_libraries(storage: CloudStorage = None) -> Iterator[ModelManager]:
    return _iter_managers(ML_LIBRARIES, storage)


def iter_explainers(storage: CloudStorage = None) -> Iterator[ModelManager]:
    return _iter_managers(EXPLAINER_LIBRARIES, storage)


def matching_manager(managers: list, **kwargs) -> ModelManager:
    for manager in managers:
        if manager.matches_with(**kwargs):
            logger.debug(f"Auto matched with: %s", manager.ml_library)
            return manager
    raise ValueError("could not find matching manager")
