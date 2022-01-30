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

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

MODEL_FILE = "model.ann"


class AnnoyManager(ModelManager):

    """
    Model persistence for Annoy models:
    https://github.com/spotify/annoy
    """

    NAME = "annoy"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["annoy"]

    def _required_kwargs(self):
        return ["model", "metric", "num_trees"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from annoy import AnnoyIndex

        return isinstance(kwargs.get("model"), AnnoyIndex)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("Model is not an AnnoyIndex!")

        return [
            partial(
                save_model,
                model=kwargs["model"],
            ),
        ]

    def _get_params(self, **kwargs) -> dict:
        return {
            "num_dimensions": kwargs["model"].f,
            "num_trees": kwargs["num_trees"],
            "metric": kwargs["metric"],
        }

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from annoy import AnnoyIndex

        # Extract these from the meta_data
        num_dimensions = int(meta_data["model"]["parameters"]["num_dimensions"])
        metric = meta_data["model"]["parameters"]["metric"]

        model = AnnoyIndex(num_dimensions, metric)
        model.load(_model_file_path(model_path))
        return model


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_FILE)


def save_model(tmp_dir: str, model: "annoy.AnnoyIndex") -> str:
    file_path = _model_file_path(tmp_dir)
    logger.debug("Saving annoy model to %s", file_path)
    model.save(file_path)
    return file_path
