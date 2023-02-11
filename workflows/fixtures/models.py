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
import joblib

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb


def _load_dataset():
    diabetes = load_diabetes()
    X_train, _, y_train, _ = train_test_split(
        diabetes.data, diabetes.target, test_size=0.1, random_state=13
    )
    return X_train, y_train


def _train_sklearn(X_train, y_train):
    model = GradientBoostingRegressor(
        **{
            "n_estimators": 500,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "absolute_error",
        }
    )
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    return model


def iter_models():
    """Generator for test models"""
    X_train, y_train = _load_dataset()
    models = [_train_sklearn, _train_xgboost]
    for model in models:
        yield model(X_train, y_train)


def iter_model_files(tmp_dir: str):
    """Generator for test model files"""
    for model in iter_models():
        model_path = os.path.join(tmp_dir, "model.joblib")
        joblib.dump(model, model_path)
        yield model_path
