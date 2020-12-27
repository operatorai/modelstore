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

from modelstore.meta import datasets
from modelstore.models.common import save_joblib
from modelstore.models.modelmanager import ModelManager

MODEL_JOBLIB = "model.joblib"


class SKLearnManager(ModelManager):

    """
    Model persistence for scikit-learn models:
    https://scikit-learn.org/stable/modules/model_persistence.html
    """

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

    def _model_info(self, **kwargs) -> dict:
        """ Returns meta-data about the model's type """
        return {
            "library": "sklearn",
            "type": type(kwargs["model"]).__name__,
        }

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
        import sklearn

        if not isinstance(kwargs["model"], sklearn.base.BaseEstimator):
            raise TypeError("This model is not an sklearn model!")

        # @TODO: export/save in onnx format
        return [partial(save_joblib, model=kwargs["model"], fn=MODEL_JOBLIB)]

    def _get_params(self, **kwargs) -> dict:
        """
        Returns a dictionary containing any model parameters
        that are available
        https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params
        """
        return kwargs["model"].get_params()


def _feature_importances(
    model: "BaseEstimator", x_train: "pd.DataFrame"
) -> dict:
    if datasets.is_pandas_dataframe(x_train):
        if hasattr(model, "feature_importances_"):
            return dict(zip(x_train, model.feature_importances_))
        if hasattr(model, "coef_"):
            return dict(zip(x_train, model.coef_[0]))
    return {}
