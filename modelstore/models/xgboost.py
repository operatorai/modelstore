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

from modelstore.models.common import save_json
from modelstore.models.modelmanager import ModelManager
from modelstore.utils.log import logger

MODEL_FILE = "model.xgboost"
MODEL_JSON = "model.json"
MODEL_CONFIG = "config.json"


class XGBoostManager(ModelManager):

    """
    Model persistence for xgboost models:
    https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
    """

    @classmethod
    def required_dependencies(cls) -> list:
        return ["xgboost"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        deps = super().optional_dependencies()
        return deps + ["sklearn"]

    def _required_kwargs(self):
        return ["model"]

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"library": "xgboost", "type": type(kwargs["model"]).__name__}

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

    def _get_functions(self, **kwargs) -> list:
        return [
            partial(save_model, model=kwargs["model"]),
            partial(dump_model, model=kwargs["model"]),
            partial(model_config, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        """
        return kwargs["model"].get_xgb_params()


def save_model(tmp_dir: str, model: "xgb.XGBModel") -> str:
    """From the docs:
    The model is saved in an XGBoost internal format which is universal
    among the various XGBoost interfaces.
    """
    logger.debug("Saving xgboost model")
    target = os.path.join(tmp_dir, MODEL_FILE)
    model.save_model(target)
    return target


def dump_model(tmp_dir: str, model: "xgb.XGBModel") -> str:
    """From the docs:
    Dump model into a text or JSON file.  Unlike `save_model`, the
    output format is primarily used for visualization or interpretation
    """
    logger.debug("Dumping xgboost model")
    model_file = os.path.join(tmp_dir, MODEL_JSON)
    model.get_booster().dump_model(
        model_file, with_stats=True, dump_format="json"
    )
    return model_file


def model_config(tmp_dir: str, model: "xgb.XGBModel") -> str:
    logger.debug("Dumping model config")
    config = model.get_booster().save_config()
    return save_json(tmp_dir, MODEL_CONFIG, config)
