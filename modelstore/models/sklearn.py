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

from functools import partial

from modelstore.models.common import save_joblib
from modelstore.models.modelmanager import ModelManager

MODEL_JOBLIB = "model.joblib"


class SKLearnManager(ModelManager):

    """
    Model persistence for scikit-learn models:
    https://scikit-learn.org/stable/modules/model_persistence.html
    """

    @classmethod
    def name(cls) -> str:
        """ Returns the name of this model type """
        return "sklearn"

    @classmethod
    def required_dependencies(cls) -> list:
        return ["sklearn"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """ Returns a list of dependencies that, if installed
        are useful to log info about """
        deps = super().optional_dependencies()
        return deps + ["Cython", "joblib", "threadpoolctl"]

    def _required_kwargs(self):
        return ["model"]

    def model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {"type": type(kwargs["model"]).__name__}

    def _get_functions(self, **kwargs) -> list:
        import sklearn

        if not isinstance(kwargs["model"], sklearn.base.BaseEstimator):
            raise TypeError("This model is not an sklearn model!")
        return [
            partial(save_joblib, model=kwargs["model"], fn=MODEL_JOBLIB),
        ]
