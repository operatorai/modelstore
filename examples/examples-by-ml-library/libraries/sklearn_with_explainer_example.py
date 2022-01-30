import shap
from modelstore.model_store import ModelStore
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libraries.util.datasets import load_diabetes_dataset
from libraries.util.domains import DIABETES_DOMAIN


def _train_example_model() -> Pipeline:
    X_train, X_test, y_train, y_test = load_diabetes_dataset()

    # Train a model using an sklearn pipeline
    params = {
        "n_estimators": 250,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "ls",
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    results = mean_squared_error(y_test, model.predict(X_test))
    print(f"🔍  Trained model MSE={results}.")

    explainer = shap.TreeExplainer(model)

    # Example only
    shap_values = explainer.shap_values(X_test)[0]
    print(f"🔍  Shap values={shap_values[:10]}.")

    return model, explainer


def train_and_upload(modelstore: ModelStore) -> dict:
    # Train a scikit-learn model and an explainer
    model, explainer = _train_example_model()

    # Upload the model to the model store
    print(f'⤴️  Uploading the sklearn model to the "{DIABETES_DOMAIN}" domain.')
    meta_data = modelstore.upload(DIABETES_DOMAIN, model=model, explainer=explainer)
    return meta_data


def load_and_test(modelstore: ModelStore, model_id: str):
    # Load the model back into memory!
    print(
        f'⤵️  Loading sklearn/shap modelsL domain="{DIABETES_DOMAIN}" model={model_id}'
    )
    models = modelstore.load(DIABETES_DOMAIN, model_id)

    # Run some example predictions
    _, X_test, _, y_test = load_diabetes_dataset()
    results = mean_squared_error(y_test, models["sklearn"].predict(X_test))
    print(f"🔍  Loaded model MSE={results}.")

    # Run some example explanations
    _, X_test, _, _ = load_diabetes_dataset()
    shap_values = models["shap"].shap_values(X_test)[0]
    print(f"🔍  Shap values={shap_values[:10]}.")
