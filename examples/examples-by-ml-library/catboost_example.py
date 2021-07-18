import catboost as ctb
from modelstore.model_store import ModelStore

from datasets import load_diabetes_dataset


def train_catboost_regressor():
    X_train, y_train = load_diabetes_dataset()

    print("🤖  Training a CatBoostRegressor")
    model = ctb.CatBoostRegressor()
    model.fit(X_train, y_train)
    return model


def run_catboost_example(modelstore: ModelStore) -> dict:
    model_domain = "diabetes-boosting-demo"
    model = train_catboost_regressor()

    # Alternative: modelstore.catboost.upload(model=model)
    print(f'⤴️  Uploading the catboost model to the "{model_domain}" domain.')
    return modelstore.upload(model_domain, model=model)
