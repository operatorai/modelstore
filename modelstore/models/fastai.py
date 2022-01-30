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
from pathlib import Path
from typing import Any

from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger

LEARNER_DIRECTORY = "learner"
LEARNER_FILE = "learner.pkl"


class FastAIManager(ModelManager):

    """
    Model persistence for fastai models:
    https://docs.fast.ai/learner.html#Learner.save
    https://docs.fast.ai/learner.html#Learner.export
    """

    NAME = "fastai"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["fastai"]

    def optional_dependencies(self) -> list:
        deps = super(FastAIManager, self).optional_dependencies()
        return deps + [
            "matplotlib",
            "pillow",
            "torchvision",
            "fastcore",
            "sklearn",
            "fastprogress",
            "torch",
            "spacy",
        ]

    def _required_kwargs(self):
        return ["learner"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        import fastai

        if fastai.__version__.startswith("1.0"):
            from fastai.basic_train import Learner
        else:
            from fastai.learner import Learner

        return isinstance(kwargs.get("learner"), Learner)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("learner is not a fastai Learner!")

        return [
            partial(_save_model, learner=kwargs["learner"]),
            partial(_export_model, learner=kwargs["learner"]),
        ]

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        import fastai

        if fastai.__version__.startswith("1.0"):
            from fastai.basic_train import load_learner
        else:
            from fastai.learner import load_learner

        version = meta_data["code"].get("dependencies", {}).get("fastai", "?")
        if version != fastai.__version__:
            logger.warn(
                "Model was saved with fastai==%s, trying to load it with fastai==%s",
                version,
                fastai.__version__,
            )

        model_file = _model_file_path(model_path)
        return load_learner(model_file)


def _model_file_path(parent_dir: str) -> str:
    return os.path.join(parent_dir, LEARNER_FILE)


def _save_model(tmp_dir: str, learner: "fastai.learner.Leader") -> str:
    # learner.save(file) will write to: self.path/self.model_dir/file;
    learner_path = learner.path
    learner.path = Path(tmp_dir)

    file_path = learner.save(LEARNER_DIRECTORY, with_opt=True)

    # Revert value
    learner.path = learner_path
    return str(file_path)


def _export_model(tmp_dir: str, learner: "fastai.learner.Leader") -> str:
    # learner.export(file) will write to: self.path/fname
    learner_path = learner.path
    learner.path = Path(tmp_dir)
    learner.export(LEARNER_FILE)

    # Revert value
    learner.path = learner_path
    return _model_file_path(tmp_dir)
