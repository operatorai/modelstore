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

MODEL_FILE = "prophet_model.json"


class ProphetManager(ModelManager):

    """
    Model persistence for Prophet models:
    https://facebook.github.io/prophet/docs/additional_topics.html#saving-models
    """

    NAME = "prophet"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["pystan", "prophet"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from prophet import Prophet

        return isinstance(kwargs.get("model"), Prophet)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not a Prophet model!")

        return [
            partial(
                save_model,
                model=kwargs["model"],
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Reference:
        https://facebook.github.io/prophet/docs/additional_topics.html#updating-fitted-models
        """
        model = kwargs["model"]
        params = {}
        for pname in ["k", "m", "sigma_obs"]:
            params[pname] = model.params[pname][0][0]
        for pname in ["delta", "beta", "trend"]:
            params[pname] = model.params[pname][0].tolist()
        return params

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from prophet.serialize import model_from_json

        file_path = _model_file_path(model_path)
        with open(file_path, "r") as fin:
            m_json = json.load(fin)
        return model_from_json(m_json)


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_FILE)


def save_model(tmp_dir: str, model: "prophet.Prophet") -> str:
    # pylint: disable=import-outside-toplevel
    from prophet.serialize import model_to_json

    file_path = _model_file_path(tmp_dir)
    logger.debug("Saving prophet model to %s", file_path)
    with open(file_path, "w") as fout:
        fout.write(json.dumps(model_to_json(model)))
    return file_path
