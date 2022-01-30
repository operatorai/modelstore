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
import json
import os
from functools import partial
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

MODEL_JSON = "model.json"
MODEL_FILE = "model.txt"


class LightGbmManager(ModelManager):

    """
    Model persistence for light gbm models:
    https://lightgbm.readthedocs.io/en/latest/Python-Intro.html#training
    """

    NAME = "lightgbm"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["lightgbm"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import lightgbm as lgb

        return isinstance(kwargs.get("model"), lgb.Booster)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not a lgb.Booster!")

        return [
            partial(save_model, model=kwargs["model"]),
            partial(dump_model, model=kwargs["model"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        return kwargs["model"].params

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import lightgbm as lgb

        model_file = _model_file_path(model_path)
        return lgb.Booster(model_file=model_file)


def _model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, MODEL_FILE)


def save_model(tmp_dir: str, model: "lgb.Booster") -> str:
    """From the docs: dump model into a txt file"""
    logger.debug("Saving lightgbm model")
    model_file = _model_file_path(tmp_dir)
    model.save_model(model_file)
    return model_file


def dump_model(tmp_dir: str, model: "lgb.Booster") -> str:
    """From the docs: dump model into JSON file"""
    logger.debug("Dumping lightgbm model as JSON")
    model_file = os.path.join(tmp_dir, MODEL_JSON)
    with open(model_file, "w") as out:
        model_json = model.dump_model()
        out.write(json.dumps(model_json))
    return model_file
