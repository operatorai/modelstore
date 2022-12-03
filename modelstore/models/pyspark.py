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
from modelstore.utils.log import logger

MODEL_DIRECTORY = "pyspark"


class PySparkManager(ModelManager):

    """
    Model persistence for PySpark MLLib models:
    https://spark.apache.org/docs/3.3.1/ml-guide.html
    https://www.sparkitecture.io/machine-learning/model-saving-and-loading
    """

    NAME = "pyspark"

    def __init__(self, storage: CloudStorage = None):
        super().__init__(self.NAME, storage)

    def required_dependencies(self) -> list:
        return ["pyspark"]

    def optional_dependencies(self) -> list:
        deps = super().optional_dependencies()
        return deps + ["py4j"]

    def _required_kwargs(self):
        return ["model"]

    def matches_with(self, **kwargs) -> bool:
        # pylint: disable=import-outside-toplevel
        from pyspark.ml.classification import Model

        return isinstance(kwargs.get("model"), Model)

    def _get_functions(self, **kwargs) -> list:
        return [
            # @TODO consider PMML too: https://github.com/jpmml/pyspark2pmml
            partial(save_model, model=kwargs["model"])
        ]

    def get_params(self, **kwargs) -> dict:
        model = kwargs["model"]
        if hasattr(model, "extractParamMap"):
            return model.extractParamMap()
        return {}

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)

        # pylint: disable=import-outside-toplevel
        from pyspark.ml import classification as psk
        model_types = {
            "DecisionTreeClassificationModel": psk.DecisionTreeClassificationModel,
            "DecisionTreeRegressionModel": psk.DecisionTreeRegressionModel,
            "FMClassificationModel": psk.FMClassificationModel,
            "GBTClassificationModel": psk.GBTClassificationModel,
            "LinearSVCModel": psk.LinearSVCModel,
            "LogisticRegressionModel": psk.LogisticRegressionModel,
            "MultilayerPerceptronClassificationModel": psk.MultilayerPerceptronClassificationModel,
            "NaiveBayesModel": psk.NaiveBayesModel,
            "OneVsRestModel": psk.OneVsRestModel,
            "ProbabilisticClassificationModel": psk.ProbabilisticClassificationModel,
            "RandomForestClassificationModel": psk.RandomForestClassificationModel,
        }
        model_type = meta_data.model_type().type
        if model_type not in model_types:
            raise ValueError(f"Cannot load pyspark model type: {model_type}")

        logger.debug("Loading xgboost model from %s", model_path)
        target = _model_files_path(model_path)
        model = model_types[model_type].load(target)
        return model


def _model_files_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, MODEL_DIRECTORY)


def save_model(tmp_dir: str, model) -> str:
    """From the docs:
    The model is saved in an XGBoost internal format which is universal
    among the various XGBoost interfaces.
    """
    logger.debug("Saving xgboost model")
    file_path = _model_files_path(tmp_dir)
    model.save(file_path)
    return file_path
