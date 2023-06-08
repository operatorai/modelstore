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
import platform
import pytest
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

from modelstore.metadata import metadata
from modelstore.models import pyspark

# pylint: disable=unused-import
from tests.models.utils import classification_data, classification_df

# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name


@pytest.fixture(autouse=True)
def spark():
    sc = SparkContext.getOrCreate()
    return SQLContext(sc)


@pytest.fixture
def spark_df(spark, classification_df):
    features = [c for c in classification_df.columns if c.startswith("x")]
    df = spark.createDataFrame(classification_df)
    # Convert features into an individual column
    assembler = VectorAssembler(inputCols=features, outputCol="x")
    df = assembler.transform(df).drop(*features)
    return df


@pytest.fixture
def spark_model(spark_df):
    rf = RandomForestClassifier(labelCol="y", featuresCol="x", numTrees=5)
    pipeline = Pipeline(stages=[rf])
    return pipeline.fit(spark_df)


@pytest.fixture
def spark_manager():
    return pyspark.PySparkManager()


def test_model_info(spark_manager, spark_model):
    exp = metadata.ModelType("pyspark", "PipelineModel", None)
    result = spark_manager.model_info(model=spark_model)
    assert exp == result


def test_model_data(spark_manager, spark_model):
    res = spark_manager.model_data(model=spark_model)
    assert res is None


def test_required_kwargs(spark_manager):
    assert spark_manager._required_kwargs() == ["model"]


def test_matches_with(spark_manager, spark_model):
    assert spark_manager.matches_with(model=spark_model)
    assert not spark_manager.matches_with(model="a-string-value")
    assert not spark_manager.matches_with(classifier=spark_model)


def test_get_functions(spark_manager, spark_model):
    assert len(spark_manager._get_functions(model=spark_model)) == 1


def test_get_params(spark_manager, spark_model):
    result = spark_manager.get_params(model=spark_model)
    assert result == {}


def test_save_model(spark_model, tmp_path):
    res = pyspark.save_model(tmp_path, spark_model)
    exp = [
        os.path.join(tmp_path, "pyspark", "metadata"),
        os.path.join(tmp_path, "pyspark", "stages"),
    ]
    assert exp == res
    exists_fn = os.path.exists
    if platform.system() == 'Darwin':
        # Running hadoop locally, so need to check
        # for the files in hdfs
        import pydoop.hdfs as hdfs
        exists_fn = hdfs.path.exists
    assert all(exists_fn(x) for x in exp)


def test_load_model(tmp_path, spark_manager, spark_model, spark_df):
    # Get the model predictions
    y_pred = spark_model.transform(spark_df).toPandas()

    # Save the model to a tmp directory
    model_path = os.path.join(tmp_path, "model")
    spark_model.save(str(model_path))

    #  Load the model
    loaded_model = spark_manager.load(
        model_path,
        metadata.Summary(
            model=metadata.Model(
                domain=None,
                model_id=None,
                model_type=metadata.ModelType(
                    library=None,
                    type="PipelineModel",
                    models=None,
                ),
                parameters=None,
                data=None,
            ),
            code=None,
            storage=None,
            modelstore=None,
        ),
    )

    # Expect the two to be the same
    assert isinstance(loaded_model, type(spark_model))

    # They should have the same predictions
    y_loaded_pred = loaded_model.transform(spark_df).toPandas()
    assert np.allclose(y_pred["prediction"], y_loaded_pred["prediction"])
