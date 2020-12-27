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
from modelstore.models.modelmanager import ModelManager
from modelstore.utils.log import logger

_MODEL_PREFIX = "model.{}"
MODEL_JSON = _MODEL_PREFIX.format("json")
MODEL_CBM = _MODEL_PREFIX.format(".cbm")
MODEL_ONNX = _MODEL_PREFIX.format("onnx")
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

    @classmethod
    def required_dependencies(cls) -> list:
        return ["catboost", "onnxruntime"]

    def _required_kwargs(self):
        return ["model"]

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"library": "catboost", "type": type(kwargs["model"]).__name__}

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

    def _get_functions(self, **kwargs) -> list:
        import catboost

        if not isinstance(kwargs["model"], catboost.CatBoost):
            raise TypeError("Model is not a CatBoost model!")

        return [
            partial(save_model, model=kwargs["model"], fmt="json"),
            partial(save_model, model=kwargs["model"], fmt="cbm"),
            partial(save_model, model=kwargs["model"], fmt="onnx"),
            partial(dump_attributes, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        https://catboost.ai/docs/concepts/python-reference_catboost_get_params.html
        """
        return kwargs["model"].get_params()


def save_model(
    tmp_dir: str, model: "catboost.CatBoost", fmt: str, pool: Any = None
) -> str:
    """CatBoost supports storing models in multiple formats:
    https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html#python-reference_catboost_save_model
    """
    logger.debug("Saving catboost model as %s", fmt)
    target = os.path.join(tmp_dir, _MODEL_PREFIX.format(fmt))
    model.save_model(target, format=fmt, pool=pool)
    return target


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
