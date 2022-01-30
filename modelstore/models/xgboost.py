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

MODEL_FILE = "model.xgboost"
MODEL_JSON = "model.json"
MODEL_CONFIG = "config.json"


class XGBoostManager(ModelManager):

    """
    Model persistence for xgboost models:
    https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    """

    NAME = "xgboost"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["xgboost"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["sklearn"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import xgboost as xgb

        return isinstance(kwargs.get("model"), xgb.XGBModel)

    def _get_functions(self, **kwargs) -> list:
        return [
            partial(save_model, model=kwargs["model"]),
            partial(dump_model, model=kwargs["model"]),
            partial(model_config, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        return kwargs["model"].get_xgb_params()

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import xgboost as xgb

        model_types = {
            "XGBRegressor": xgb.XGBRegressor,
            "XGBClassifier": xgb.XGBClassifier,
            "XGBModel": xgb.XGBModel,
            # Future: other types
        }
        model_type = self._get_model_type(meta_data)
        if model_type not in model_types:
            raise ValueError(f"Cannot load xgboost model type: {model_type}")

        logger.debug("Loading xgboost model from %s", model_path)
        target = _model_file_path(model_path)
        model = model_types[model_type]()
        model.load_model(target)
        return model


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_FILE)


def save_model(tmp_dir: str, model: "xgb.XGBModel") -> str:
    """From the docs:
    The model is saved in an XGBoost internal format which is universal
    among the various XGBoost interfaces.
    """
    logger.debug("Saving xgboost model")
    file_path = _model_file_path(tmp_dir)
    model.save_model(file_path)
    return file_path


def dump_model(tmp_dir: str, model: "xgb.XGBModel") -> str:
    """From the docs:
    Dump model into a text or JSON file.  Unlike `save_model`, the
    output format is primarily used for visualization or interpretation
    """
    logger.debug("Dumping xgboost model")
    model_file = os.path.join(tmp_dir, MODEL_JSON)
    model.get_booster().dump_model(model_file, with_stats=True, dump_format="json")
    return model_file


def model_config(tmp_dir: str, model: "xgb.XGBModel") -> str:
    logger.debug("Dumping model config")
    config = model.get_booster().save_config()
    return save_json(tmp_dir, MODEL_CONFIG, config)
