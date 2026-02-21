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

from modelstore.metadata import metadata
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

MODEL_PICKLE = "model.pkl"


def _save_model(tmp_dir, model, file_name):
    path = os.path.join(tmp_dir, file_name)
    model.save(path)
    return path


class StatsModelsManager(ModelManager):

    """
    Model persistence for statsmodels fitted result objects:
    https://www.statsmodels.org/stable/index.html
    """

    NAME = "statsmodels"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["statsmodels"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from statsmodels.base.wrapper import ResultsWrapper

        return isinstance(kwargs.get("model"), ResultsWrapper)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not a statsmodels ResultsWrapper!")

        return [partial(_save_model, model=kwargs["model"], file_name=MODEL_PICKLE)]

    def get_params(self, **kwargs) -> dict:
        try:
            params = kwargs["model"].params
            if hasattr(params, "to_dict"):
                return params.to_dict()
            # numpy array — convert to a plain dict keyed by position
            return {str(i): float(v) for i, v in enumerate(params)}
        except Exception:
            return {}

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)
        # pylint: disable=import-outside-toplevel
        from statsmodels.iolib.smpickle import load_pickle

        return load_pickle(os.path.join(model_path, MODEL_PICKLE))
