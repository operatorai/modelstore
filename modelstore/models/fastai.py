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

LEARNER_DIRECTORY = "learner"
LEARNER_FILE = "learner.pkl"


class FastAIManager(ModelManager):

    """
    Model persistence for fastai models:
    https://docs.fast.ai/learner.html#Learner.save
    https://docs.fast.ai/learner.html#Learner.export
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("fastai", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return [
            "fastai",
        ]

    @classmethod
    def optional_dependencies(cls) -> list:
        deps = super(FastAIManager, cls).optional_dependencies()
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
        from fastai.learner import Learner

        return isinstance(kwargs.get("learner"), Learner)

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("learner is not a fastai.learner.Learner!")

        return [
            partial(_save_model, learner=kwargs["learner"]),
            partial(_export_model, learner=kwargs["learner"]),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        @ TODO: extract useful info from
        kwargs["learner"].opt.state_dict()
        """
        return {}

    def load(self, model_path: str, meta_data: dict) -> Any:
        # pylint: disable=import-outside-toplevel
        from fastai.learner import load_learner

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
