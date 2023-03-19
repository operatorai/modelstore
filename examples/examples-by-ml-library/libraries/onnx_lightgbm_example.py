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

import numpy as np
import onnx
from modelstore.model_store import ModelStore

from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm,
)

from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_classification_dataset
from libraries.util.domains import BREAST_CANCER_DOMAIN


def _train_example_model() -> onnx.ModelProto:
    X_train, X_test, y_train, y_test = load_classification_dataset()

    print(f"🔍  Training a light gbm classifier")
    clf = LGBMClassifier(random_state=12)
    clf.fit(X_train, y_train)

    print(f"🔍  Converting the model to onnx")
    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    model = convert_sklearn(
        clf,
        "lightgbm",
        [("X", FloatTensorType([None, X_train.shape[1]]))],
        target_opset={"": 12, "ai.onnx.ml": 2},
    )

    print(f"🔍  Loading the onnx model as an inference session")
    sess = InferenceSession(model.SerializeToString())
    y_pred = sess.run(None, {"X": X_test.astype(np.float32)})[0]

    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Trained model MSE={results}.")
    return model


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model
    model = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the onnx model to the "{BREAST_CANCER_DOMAIN}" domain.')
    meta_data = modelstore.upload(BREAST_CANCER_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_domain: str, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the onnx "{model_domain}" domain model={model_id}')
    sess = modelstore.load(model_domain, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_classification_dataset()
    y_pred = sess.run(None, {"X": X_test.astype(np.float32)})[0]
    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Loaded model MSE={results}.")
