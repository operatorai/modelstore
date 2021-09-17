import numpy as np
import onnx
from modelstore.model_store import ModelStore
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> onnx.ModelProto:
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    print(f"🔍  Training a random forest regressor")
    clf = RandomForestRegressor(random_state=12)
    clf.fit(X_train, y_train)

    print(f"🔍  Converting the model to onnx")
    model = to_onnx(clf, X_train[:1].astype(np.float32), target_opset=12)

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
    print(f'⤴️  Uploading the onnx model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(f'⤵️  Loading the onnx "{DIABETES_DOMAIN}" domain model={model_id}')
    sess = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    y_pred = sess.run(None, {"X": X_test.astype(np.float32)})[0]
    results = mean_squared_error(y_test, y_pred)
    print(f"🔍  Loaded model MSE={results}.")
