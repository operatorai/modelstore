#    Copyright 2022 Neal Lathia
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
from typing import Any, List

from modelstore.metadata import metadata
from modelstore.models.model_manager import ModelManager
from modelstore.storage.storage import CloudStorage
from modelstore.utils.log import logger


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
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import Model
        from pyspark.ml import Model as mlModel
        from pyspark.ml.classification import _JavaProbabilisticClassifier

        # Warning: for Apache Spark prior to 2.0.0, save isn't
        # available yet for the Pipeline API.

        model = kwargs.get("model")
        return any(
            [
                isinstance(model, Pipeline),
                isinstance(model, _JavaProbabilisticClassifier),
                isinstance(model, mlModel),
                isinstance(model, Model),
            ]
        )

    def _get_functions(self, **kwargs) -> list:
        return [
            # @TODO consider PMML too: https://github.com/jpmml/pyspark2pmml
            partial(save_model, model=kwargs["model"])
        ]

    def load(self, model_path: str, meta_data: metadata.Summary) -> Any:
        super().load(model_path, meta_data)

        # pylint: disable=import-outside-toplevel
        from pyspark.ml import classification
        from pyspark.ml import PipelineModel

        model_types = {
            "PipelineModel": PipelineModel,
            "DecisionTreeClassificationModel": classification.DecisionTreeClassificationModel,
            "DecisionTreeRegressionModel": classification.DecisionTreeRegressionModel,
            "FMClassificationModel": classification.FMClassificationModel,
            "GBTClassificationModel": classification.GBTClassificationModel,
            "LinearSVCModel": classification.LinearSVCModel,
            "LogisticRegressionModel": classification.LogisticRegressionModel,
            "MultilayerPerceptronClassificationModel": classification.MultilayerPerceptronClassificationModel,
            "NaiveBayesModel": classification.NaiveBayesModel,
            "OneVsRestModel": classification.OneVsRestModel,
            "ProbabilisticClassifier": classification.ProbabilisticClassificationModel,
            "RandomForestClassificationModel": classification.RandomForestClassificationModel,
        }
        model_type = meta_data.model_type().type
        if model_type not in model_types:
            raise ValueError(f"Cannot load pyspark model type: {model_type}")

        logger.debug("Loading pyspark model from %s", model_path)
        model = model_types[model_type].load(model_path)
        return model


def save_model(tmp_dir: str, model: "pyspark.ml.Model") -> List[str]:
    """Saves the pyspark model"""
    logger.debug("Saving pyspark model")
    target = os.path.join(tmp_dir, "pyspark")
    model.save(target)
    return [
        os.path.join(target, "metadata"),
        os.path.join(target, "stages"),
    ]
