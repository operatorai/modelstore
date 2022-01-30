#    Copyright 2021 Neal Lathia
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

GENSIM_MODEL = "gensim.model"
GENSIM_KEYED_VECTORS = "gensim.wordvectors"


class GensimManager(ModelManager):

    """
    Model persistence for scikit-learn models:
    https://scikit-learn.org/stable/modules/model_persistence.html
    """

    NAME = "gensim"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["gensim"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["Levenshtein"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import gensim

        return isinstance(kwargs.get("model"), gensim.utils.SaveLoad)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not a gensim SaveLoad model")

        funcs = [partial(_save_model, model=kwargs["model"])]
        if hasattr(kwargs["model"], "wv"):
            funcs.append(partial(_save_vectors, model=kwargs["model"]))
        return funcs

    def _get_params(self, **kwargs) -> dict:
        params = kwargs["model"].__dict__
        # The instance attributes contain a lot of information, including
        # the model's keyed vectors; so we filter this down for now
        params = {k: v for k, v in params.items() if type(v) in [int, str, float]}
        return params

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from gensim.models import Word2Vec

        model_type = self._get_model_type(meta_data)
        if model_type != "Word2Vec":
            raise ValueError(f"modelstore cannot load gensim '{model_type}' models")

        model_file = _model_file_path(model_path)
        return Word2Vec.load(model_file)


def _model_file_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, GENSIM_MODEL)


def _save_model(tmp_dir: str, model: "gensim.utils.SaveLoad") -> str:
    file_path = _model_file_path(tmp_dir)
    model.save(file_path)
    return file_path


def _save_vectors(tmp_dir: str, model: "gensim.utils.SaveLoad") -> str:
    file_path = os.path.join(tmp_dir, GENSIM_KEYED_VECTORS)
    model.wv.save(file_path)
    return file_path
