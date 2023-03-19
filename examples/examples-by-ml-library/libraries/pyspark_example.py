#    Copyright 2023 Neal Lathia
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

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import RandomForestRegressor

from modelstore.model_store import ModelStore
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_regression_dataframe
from libraries.util.domains import DIABETES_DOMAIN


def _spark_dataset(sqlContext):
    df = load_regression_dataframe()
    features = [c for c in df.columns if c != "y"]
    spark_df = sqlContext.createDataFrame(df)
    assembler = VectorAssembler(inputCols=features, outputCol="x")
    return assembler.transform(spark_df).drop(*features)


def _train_example_model() -> PipelineModel:
    sc = SparkSession.builder.getOrCreate()
    sqlContext = SQLContext(sc)

    # Load the data into Spark
    spark_df = _spark_dataset(sqlContext)

    # Train a pipeline
    rf = RandomForestRegressor(labelCol="y", featuresCol="x", numTrees=5)
    pipeline = Pipeline(stages=[rf])
    model = pipeline.fit(spark_df)

    predictions = model.transform(spark_df).toPandas()
    y_pred = predictions["prediction"]
    y_test = predictions["y"]
    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the pyspark model to the "{DIABETES_DOMAIN}" domain.')
    return modelstore.upload(DIABETES_DOMAIN, model=model)


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Create a context
    sc = SparkSession.builder.getOrCreate()
    sqlContext = SQLContext(sc)

    # Load the model back into memory!
    print(f'⤵️  Loading the pyspark "{model_domain}" domain model={model_id}')
    model = modelstore.load(model_domain, model_id)

    # Load the data into Spark
    spark_df = _spark_dataset(sqlContext)

    # Run some example predictions
    predictions = model.transform(spark_df).toPandas()
    y_pred = predictions["prediction"]
    y_test = predictions["y"]
    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Loaded model MSE={results}.")
