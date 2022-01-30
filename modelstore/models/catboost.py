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
import os
from functools import partial
from typing import Any

from modelstore.models.common import save_json
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

_MODEL_PREFIX = "model.{}"
MODEL_ATTRIBUTES = "model_attributes.json"


class CatBoostManager(ModelManager):

    """
    Model persistence for catboost models:
    https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html#python-reference_catboost_save_model

    As JSON (e.g., for inspection):
    https://catboost.ai/docs/features/export-model-to-json.html#export-model-to-json

    As ONNX (e.g., for inference):
    https://catboost.ai/docs/concepts/apply-onnx-ml.html
    """

    NAME = "catboost"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["catboost", "onnxruntime"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import catboost

        return isinstance(kwargs.get("model"), catboost.CatBoost)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not a CatBoost model!")

        # pool parameter, from the catboost docs:
        # The dataset previously used for training.
        # This parameter is required if the model contains categorical features and the output format is cpp, python, or JSON.
        return [
            partial(
                save_model,
                model=kwargs["model"],
                fmt="json",
                pool=kwargs.get("pool"),
            ),
            partial(
                save_model,
                model=kwargs["model"],
                fmt="cbm",
                pool=kwargs.get("pool"),
            ),
            # onnx: only datasets without categorical features are currently supported
            partial(
                save_model,
                model=kwargs["model"],
                fmt="onnx",
                pool=kwargs.get("pool"),
            ),
            partial(dump_attributes, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        https://catboost.ai/docs/concepts/python-reference_catboost_get_params.html
        """
        return kwargs["model"].get_params()

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import catboost

        model_types = {
            "CatBoostRegressor": catboost.CatBoostRegressor,
            "CatBoostClassifier": catboost.CatBoostClassifier,
        }
        model_type = self._get_model_type(meta_data)
        if model_type not in model_types:
            raise ValueError(f"Cannot load catboost model type: {model_type}")

        logger.debug("Loading catboost model from %s", model_path)
        file_path = _model_file_path(model_path, fmt="cbm")
        model = model_types[model_type]()
        return model.load_model(file_path, format="cbm")


def _model_file_path(tmp_dir: str, fmt: str) -> str:
    return os.path.join(tmp_dir, _MODEL_PREFIX.format(fmt))


def save_model(
    tmp_dir: str, model: "catboost.CatBoost", fmt: str, pool: Any = None
) -> str:
    """CatBoost supports storing models in multiple formats:
    https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html#python-reference_catboost_save_model
    """
    logger.debug("Saving catboost model as %s", fmt)
    file_path = _model_file_path(tmp_dir, fmt)
    model.save_model(file_path, format=fmt, pool=pool)
    return file_path


def dump_attributes(tmp_dir: str, model: "catboost.CatBoost") -> str:
    logger.debug("Dumping model config")
    config = {
        "tree_count": model.tree_count_,
        "random_seed": model.random_seed_,
        "learning_rate": model.learning_rate_,
        "feature_names": model.feature_names_,
        "feature_importances": model.feature_importances_.tolist(),
        "evals_result": model.evals_result_,
        "best_score": model.best_score_,
    }
    if model.best_iteration_ is not None:
        config["best_iteration"] = model.best_iteration_
    if len(model.classes_) != 0:
        # Return the names of classes for classification models.
        # An empty list is returned for all other models.
        config["classes"] = model.classes_.tolist()
    return save_json(tmp_dir, MODEL_ATTRIBUTES, config)
