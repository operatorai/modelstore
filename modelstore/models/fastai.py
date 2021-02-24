#    Copyright 2020 Neal Lathia

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from functools import partial
from pathlib import Path

from modelstore.models.modelmanager import ModelManager
from modelstore.models.util import convert_tensors

LEARNER = "learner"


class FastAIManager(ModelManager):

    """
    Model persistence for fastai models:
    https://docs.fast.ai/learner.html#Learner.save
    https://docs.fast.ai/learner.html#Learner.export
    """

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

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"library": "fastai"}

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        return {}

    def _get_functions(self, **kwargs) -> list:
        learn = kwargs["learner"]
        return [
            partial(_save_model, learner=learn),
            partial(_export_model, learner=learn),
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        @ TODO: extract useful info from
        kwargs["learner"].opt.state_dict()
        """
        return {}


def _save_model(tmp_dir: str, learner: "fastai.learner.Leader") -> str:
    from fastai.learner import Learner

    if not isinstance(learner, Learner):
        raise TypeError("learner is not a fastai.learner.Learner!")

    # learner.save(file) will write to: self.path/self.model_dir/file;
    learner_path = learner.path
    learner.path = Path(tmp_dir)

    file_path = learner.save(LEARNER)

    # Revert value
    learner.path = learner_path
    return str(file_path)


def _export_model(tmp_dir: str, learner: "fastai.learner.Leader") -> str:
    from fastai.learner import Learner

    if not isinstance(learner, Learner):
        raise TypeError("learner is not a fastai.learner.Learner!")

    file_name = LEARNER + ".pkl"
    # learner.export(file) will write to: self.path/fname
    learner_path = learner.path
    learner.path = Path(tmp_dir)
    learner.export(file_name)

    # Revert value
    learner.path = learner_path
    return os.path.join(tmp_dir, file_name)
