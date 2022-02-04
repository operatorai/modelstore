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
from typing import Iterator, List

from modelstore.meta.dependencies import module_exists
from modelstore.models.annoy import AnnoyManager
from modelstore.models.catboost import CatBoostManager
from modelstore.models.fastai import FastAIManager
from modelstore.models.gensim import GensimManager
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

_LIBRARIES = {
    m.NAME: m
    for m in [
        AnnoyManager,
        CatBoostManager,
        FastAIManager,
        ModelFileManager,
        GensimManager,
        LightGbmManager,
        MxnetManager,
        OnnxManager,
        ProphetManager,
        PyTorchManager,
        PyTorchLightningManager,
        ShapManager,
        SKLearnManager,
        SkorchManager,
        TensorflowManager,
        TransformersManager,
        XGBoostManager,
    ]
}


def iter_libraries(storage: CloudStorage = None) -> Iterator[ModelManager]:
    """Iterates of a dict of ModelManagers and yields
    the ones that are available in the current environment,
    based on checking for dependencies.
    """
    for name, library in _LIBRARIES.items():
        manager = library(storage)
        if all(module_exists(x) for x in manager.required_dependencies()):
            logger.debug("Adding: %s", name)
            yield name, manager
        else:
            logger.debug("Skipping: %s, not installed.", name)
            yield name, MissingDepManager(name, storage)


def matching_managers(managers: list, **kwargs) -> List[ModelManager]:
    managers = [m for m in managers if m.matches_with(**kwargs)]
    if len(managers) == 0:
        raise ValueError("could not find matching manager")
    return managers


def get_manager(name: str, storage: CloudStorage = None) -> ModelManager:
    manager = _LIBRARIES[name](storage)
    if all(module_exists(x) for x in manager.required_dependencies()):
        return manager
    raise ValueError(
        "could not create manager for %s: dependencies not installed", name
    )
