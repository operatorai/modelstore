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

from modelstore.metadata import metadata
from modelstore.metadata.dataset.types import is_pandas_dataframe
from modelstore.models.common import load_joblib, save_joblib
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage

MODEL_JOBLIB = "model.joblib"


class SKLearnManager(ModelManager):

    """
    Model persistence for scikit-learn models:
    https://scikit-learn.org/stable/modules/model_persistence.html
    """

    NAME = "sklearn"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["sklearn"]

    def optional_dependencies(self) -> list:
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

        from sklearn.base import BaseEstimator

        return isinstance(kwargs.get("model"), BaseEstimator)

    def model_data(self, **kwargs) -> metadata.Dataset:
        # @TODO add _feature_importances()
        return metadata.Dataset.generate(
            kwargs.get("X_train"),
            kwargs.get("y_train"),
        )

    def _get_functions(self, **kwargs) -> list:
        if not self.matches_with(**kwargs):
            raise TypeError("This model is not an sklearn model!")

        # @Future idea: export/save in onnx format?
        return [partial(save_joblib, model=kwargs["model"], file_name=MODEL_JOBLIB)]

    def get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params
        """
        from sklearn.pipeline import Pipeline

        if isinstance(kwargs.get("model"), Pipeline):
            # Pipelines contain a ton of things that are not JSON serializable
            # the same params exist separately in get_params(), so we just drop
            # the bits that could not be serialized
            return {}
        try:
            params = kwargs["model"].get_params()
            # Check if params is json serializable
            json.dumps(params)
            return params
        except TypeError:
            return {}

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)

        # @Future: check if loading into same version of joblib
        # as was used for saving
        file_name = os.path.join(model_path, MODEL_JOBLIB)
        return load_joblib(file_name)


def _feature_importances(model: "BaseEstimator", x_train: "pandas.DataFrame") -> dict:
    result = {}
    if is_pandas_dataframe(x_train):
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
