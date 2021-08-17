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

from modelstore.meta import datasets
from modelstore.models.common import load_joblib, save_joblib
from modelstore.models.model_manager import ModelManager
from modelstore.models.util import convert_numpy
from modelstore.storage.storage import CloudStorage

MODEL_JOBLIB = "model.joblib"


class SKLearnManager(ModelManager):

    """
    Model persistence for scikit-learn models:
    https://scikit-learn.org/stable/modules/model_persistence.html
    """

    def __init__(self, storage: CloudStorage = None):
        super().__init__("sklearn", storage)

    @classmethod
    def required_dependencies(cls) -> list:
        return ["sklearn"]

    @classmethod
    def optional_dependencies(cls) -> list:
        """Returns a list of dependencies that, if installed
        are useful to log info about"""
        deps = super().optional_dependencies()
        return deps + ["Cython", "joblib", "threadpoolctl"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        try:
            import xgboost as xgb

            # xgboost models are an instance of sklearn.base.BaseEstimator
            # but we want to upload them using the xgboost manager
            # we therefore check specifically for this case
            if isinstance(kwargs.get("model"), xgb.XGBModel):
                return False
        except ImportError:
            pass

        import sklearn

        return isinstance(kwargs.get("model"), sklearn.base.BaseEstimator)

    def _model_data(self, **kwargs) -> dict:
        """ Returns meta-data about the data used to train the model """
        data = {}
        if "X_train" in kwargs:
            features = datasets.describe_dataset(kwargs["X_train"])
            features.update(
                _feature_importances(kwargs["model"], kwargs["X_train"])
            )
            data["features"] = features
        if "y_train" in kwargs:
            data["labels"] = datasets.describe_dataset(kwargs["y_train"])
        return data

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not an sklearn model!")

        # @Future idea: export/save in onnx format?
        return [
            partial(save_joblib, model=kwargs["model"], file_name=MODEL_JOBLIB)
        ]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params
        """
        params = kwargs["model"].get_params()
        # Pipelines contain a ton of things that are not JSON serializable
        # the same params exist separately in get_params(), so we just drop
        # the bits that could not be serialized
        if "steps" in params:
            for key, _ in params["steps"]:
                params.pop(key, None)
            params.pop("steps", None)
        return convert_numpy(params)

    def load(self, model_path: str, meta_data: dict) -> Any:
        """
        Loads a joblib model, stored in model_path, back into memory
        """
        # @Future: check if loading into same version of joblib
        # as was used for saving
        file_name = os.path.join(model_path, MODEL_JOBLIB)
        return load_joblib(file_name)


def _feature_importances(
    model: "BaseEstimator", x_train: "pandas.DataFrame"
) -> dict:
    result = {}
    if datasets.is_pandas_dataframe(x_train):
        weights = _get_weights(model)
        if weights is not None:
            return dict(zip(x_train, weights))
        if hasattr(model, "steps"):
            # Scikit pipelines
            for key, step in model.steps:
                weights = _get_weights(step)
                if weights is not None:
                    result[key] = weights
    return result


def _get_weights(model: "BaseEstimator"):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return model.coef_[0]
